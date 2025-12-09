#!/usr/bin/env python3
"""
Large-scale persona generation test with rate limit monitoring, token tracking, and timing.

Tests creating 250 personas while monitoring:
- Rate limits (RPM/TPM)
- Token usage per request and cumulative
- Execution time
- Success/failure rates
- 429 errors and retries

Run: python test_persona_scale.py --count 250
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import AzureOpenAI

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"
)

# Rate limits (adjust based on your Azure deployment)
TPM_LIMIT = int(os.getenv("AZURE_TPM_LIMIT", "150000"))
RPM_LIMIT = int(os.getenv("AZURE_RPM_LIMIT", "1500"))

# System prompt for persona generation
GENERATION_SYSTEM_PROMPT = """You are an expert persona generator creating realistic audience member profiles.

Generate a realistic, believable individual that:
- Embodies the spirit and characteristics of the base persona
- Has internally consistent traits and behaviors
- Feels like a real person, not a stereotype

NAME GENERATION RULES:
- Generate a completely RANDOM and UNIQUE full name for each person
- NEVER repeat or reuse names across profiles—each name must be distinct
- Use diverse first names and surnames—avoid common/overused names
- Match the name to the persona's location, ethnicity, and gender
- Be creative: draw from a wide variety of cultural naming conventions

You MUST respond with valid JSON containing EXACTLY these fields:
- "name": (string) A realistic full name appropriate for the persona's demographic
- "about": (string) Behavioral description focusing on interests, digital habits, creative pursuits, and lifestyle
- "goalsAndMotivations": (array of 3 strings) List of goals and motivations
- "frustrations": (array of 3 strings) List of frustrations
- "needState": (string) Current psychological or motivational state
- "occasions": (string) Contextual situations for content engagement

IMPORTANT: Return ONLY the JSON object with actual values. Do NOT return a schema definition or type descriptions."""


# ============================================================================
# Metrics Tracking
# ============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single API request."""
    request_id: int
    start_time: float
    end_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = False
    error: str = ""
    retry_count: int = 0
    rate_limited: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0


@dataclass
class AggregateMetrics:
    """Aggregate metrics for the entire test run."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_retries: int = 0
    
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    
    start_time: float = 0.0
    end_time: float = 0.0
    
    requests: list = field(default_factory=list)
    
    # Rate tracking (per minute windows)
    tokens_per_minute: list = field(default_factory=list)
    requests_per_minute: list = field(default_factory=list)
    
    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0
    
    @property
    def avg_tokens_per_request(self) -> float:
        return self.total_tokens / self.successful_requests if self.successful_requests else 0.0
    
    @property
    def avg_request_duration(self) -> float:
        if not self.requests:
            return 0.0
        durations = [r.duration for r in self.requests if r.success]
        return sum(durations) / len(durations) if durations else 0.0
    
    @property
    def requests_per_second(self) -> float:
        return self.successful_requests / self.total_duration if self.total_duration else 0.0
    
    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.total_duration if self.total_duration else 0.0


# ============================================================================
# Sample Persona Templates
# ============================================================================

SAMPLE_PERSONAS = [
    {
        "about": "A tech-savvy marketing professional who thrives on innovation and digital trends.",
        "goals_and_motivations": "Career growth, staying ahead of industry trends, work-life balance",
        "frustrations": "Information overload, tight deadlines, limited budgets",
        "need_state": "Seeking efficient tools and strategies to maximize productivity",
        "occasions": "Morning planning sessions, lunch break learning, evening wind-down",
    },
    {
        "about": "A creative entrepreneur building a sustainable lifestyle brand.",
        "goals_and_motivations": "Business growth, environmental impact, authentic connections",
        "frustrations": "Limited resources, market competition, scaling challenges",
        "need_state": "Looking for community and mentorship opportunities",
        "occasions": "Weekend strategy sessions, networking events, creative brainstorming",
    },
    {
        "about": "A healthcare professional balancing clinical work with continuous education.",
        "goals_and_motivations": "Patient care excellence, professional development, research contribution",
        "frustrations": "Administrative burden, time constraints, keeping up with medical advances",
        "need_state": "Seeking streamlined workflows and evidence-based resources",
        "occasions": "Early morning rounds, conference attendance, evening study",
    },
    {
        "about": "A financial analyst with a passion for data-driven decision making.",
        "goals_and_motivations": "Accurate forecasting, career advancement, thought leadership",
        "frustrations": "Data quality issues, stakeholder communication, regulatory changes",
        "need_state": "Looking for advanced analytics tools and industry insights",
        "occasions": "Quarterly reporting, market analysis, client presentations",
    },
    {
        "about": "A remote worker navigating the digital nomad lifestyle.",
        "goals_and_motivations": "Location independence, meaningful work, cultural experiences",
        "frustrations": "Time zone challenges, connectivity issues, work isolation",
        "need_state": "Seeking community and reliable remote work infrastructure",
        "occasions": "Co-working sessions, travel planning, virtual team meetings",
    },
]

SAMPLE_SCREENER_QUESTIONS = [
    {"question": "What is your primary industry?", "answer": "Technology"},
    {"question": "How many years of experience do you have?", "answer": "5-10 years"},
    {"question": "What is your company size?", "answer": "50-200 employees"},
    {"question": "What is your role level?", "answer": "Mid-level manager"},
]


def create_generation_prompt(persona_index: int) -> str:
    """Create a generation prompt for a persona."""
    persona = SAMPLE_PERSONAS[persona_index % len(SAMPLE_PERSONAS)]
    
    screener_lines = [
        f"- **Q**: {q['question']}\n  **A**: {q['answer']}"
        for q in SAMPLE_SCREENER_QUESTIONS
    ]
    screener_section = "\n".join(screener_lines)
    
    return f"""Generate a detailed audience member profile for the following persona:

## Base Persona Template
- **About**: {persona['about']}
- **Goals & Motivations**: {persona['goals_and_motivations']}
- **Frustrations**: {persona['frustrations']}
- **Need State**: {persona['need_state']}
- **Occasions**: {persona['occasions']}

## Screener Responses
{screener_section}

## Important Guidelines
1. Use the screener responses to inform lifestyle, work environment, and behavioral descriptions
2. Ensure the generated profile is consistent with the screener answers
3. The profile should feel like a real person, not a stereotype
4. Generate a RANDOM, UNIQUE full name—be creative and diverse

Generate a complete, realistic audience member profile as JSON."""


# ============================================================================
# Rate Limit Header Checker
# ============================================================================


def check_rate_limit_headers() -> dict:
    """Check current rate limit status via raw API call."""
    raw_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    
    raw_response = raw_client.chat.completions.with_raw_response.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": "Hi"}],
        max_completion_tokens=1,
    )
    
    headers = raw_response.headers
    return {
        "remaining_requests": headers.get("x-ratelimit-remaining-requests"),
        "remaining_tokens": headers.get("x-ratelimit-remaining-tokens"),
        "limit_requests": headers.get("x-ratelimit-limit-requests"),
        "limit_tokens": headers.get("x-ratelimit-limit-tokens"),
        "reset_requests": headers.get("x-ratelimit-reset-requests"),
        "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
    }


# ============================================================================
# Persona Generation with Metrics
# ============================================================================


async def generate_single_persona(
    client: AzureChatOpenAI,
    request_id: int,
    semaphore: asyncio.Semaphore,
    metrics: AggregateMetrics,
    max_retries: int = 3,
) -> RequestMetrics:
    """Generate a single persona with full metrics tracking."""
    
    request_metrics = RequestMetrics(request_id=request_id, start_time=time.time())
    prompt = create_generation_prompt(request_id)
    
    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.ainvoke(messages)
                
                # Extract token usage from response metadata
                token_usage = response.response_metadata.get("token_usage", {})
                request_metrics.prompt_tokens = token_usage.get("prompt_tokens", 0)
                request_metrics.completion_tokens = token_usage.get("completion_tokens", 0)
                request_metrics.total_tokens = token_usage.get("total_tokens", 0)
                
                # Validate JSON response
                content = response.content
                if content:
                    json.loads(content)  # Validate it's valid JSON
                
                request_metrics.success = True
                request_metrics.end_time = time.time()
                request_metrics.retry_count = attempt
                
                # Update aggregate metrics
                metrics.successful_requests += 1
                metrics.total_prompt_tokens += request_metrics.prompt_tokens
                metrics.total_completion_tokens += request_metrics.completion_tokens
                metrics.total_tokens += request_metrics.total_tokens
                
                return request_metrics
                
            except Exception as e:
                error_str = str(e)
                request_metrics.retry_count = attempt + 1
                
                if "429" in error_str or "rate" in error_str.lower():
                    request_metrics.rate_limited = True
                    metrics.rate_limited_requests += 1
                    # Exponential backoff for rate limits
                    wait_time = min(60, 2 ** (attempt + 1))
                    print(f"  ⚠️  Request {request_id}: Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    request_metrics.error = error_str[:100]
                    await asyncio.sleep(1 * (attempt + 1))
                
                if attempt == max_retries - 1:
                    request_metrics.end_time = time.time()
                    metrics.failed_requests += 1
                    metrics.total_retries += request_metrics.retry_count
    
    return request_metrics


async def run_persona_generation_test(
    count: int,
    max_concurrent: int,
    output_file: Path | None = None,
) -> AggregateMetrics:
    """Run the full persona generation test."""
    
    print("\n" + "=" * 70)
    print("LARGE-SCALE PERSONA GENERATION TEST")
    print("=" * 70)
    print(f"  Personas to generate: {count}")
    print(f"  Max concurrent:       {max_concurrent}")
    print(f"  Deployment:           {DEPLOYMENT}")
    print(f"  TPM Limit:            {TPM_LIMIT:,}")
    print(f"  RPM Limit:            {RPM_LIMIT:,}")
    print(f"  Start time:           {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Check initial rate limits
    print("\n📊 Checking initial rate limits...")
    try:
        initial_limits = check_rate_limit_headers()
        print(f"  Remaining requests: {initial_limits.get('remaining_requests', 'N/A')}")
        print(f"  Remaining tokens:   {initial_limits.get('remaining_tokens', 'N/A')}")
        print(f"  Request limit:      {initial_limits.get('limit_requests', 'N/A')}")
        print(f"  Token limit:        {initial_limits.get('limit_tokens', 'N/A')}")
    except Exception as e:
        print(f"  ⚠️  Could not check rate limits: {e}")
    
    # Initialize client
    client = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=DEPLOYMENT,
        temperature=0.8,
        max_tokens=1024,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    
    # Initialize metrics
    metrics = AggregateMetrics(
        total_requests=count,
        start_time=time.time(),
    )
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Progress tracking
    completed = 0
    last_progress_time = time.time()
    progress_interval = 10  # Print progress every 10 completions
    
    print(f"\n🚀 Starting generation of {count} personas...")
    print("-" * 70)
    
    async def generate_with_progress(request_id: int) -> RequestMetrics:
        nonlocal completed, last_progress_time
        result = await generate_single_persona(client, request_id, semaphore, metrics)
        
        completed += 1
        
        # Print progress at intervals
        if completed % progress_interval == 0 or completed == count:
            elapsed = time.time() - metrics.start_time
            rate = completed / elapsed if elapsed > 0 else 0
            tokens_rate = metrics.total_tokens / elapsed if elapsed > 0 else 0
            
            print(
                f"  [{completed:4d}/{count}] "
                f"✅ {metrics.successful_requests} "
                f"❌ {metrics.failed_requests} "
                f"⚠️  {metrics.rate_limited_requests} rate-limited | "
                f"{rate:.1f} req/s | "
                f"{tokens_rate:.0f} tok/s | "
                f"Total: {metrics.total_tokens:,} tokens"
            )
        
        return result
    
    # Run all requests
    tasks = [generate_with_progress(i) for i in range(count)]
    results = await asyncio.gather(*tasks)
    
    metrics.end_time = time.time()
    metrics.requests = list(results)
    
    # Check final rate limits
    print("\n📊 Checking final rate limits...")
    try:
        final_limits = check_rate_limit_headers()
        print(f"  Remaining requests: {final_limits.get('remaining_requests', 'N/A')}")
        print(f"  Remaining tokens:   {final_limits.get('remaining_tokens', 'N/A')}")
    except Exception as e:
        print(f"  ⚠️  Could not check rate limits: {e}")
    
    # Print summary
    print_summary(metrics)
    
    # Save results if output file specified
    if output_file:
        save_results(metrics, output_file)
    
    return metrics


def print_summary(metrics: AggregateMetrics) -> None:
    """Print a comprehensive summary of the test results."""
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print("\n📈 REQUEST STATISTICS")
    print("-" * 40)
    print(f"  Total requests:        {metrics.total_requests}")
    print(f"  Successful:            {metrics.successful_requests}")
    print(f"  Failed:                {metrics.failed_requests}")
    print(f"  Rate limited:          {metrics.rate_limited_requests}")
    print(f"  Total retries:         {metrics.total_retries}")
    success_rate = (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests else 0
    print(f"  Success rate:          {success_rate:.1f}%")
    
    print("\n🎯 TOKEN USAGE")
    print("-" * 40)
    print(f"  Prompt tokens:         {metrics.total_prompt_tokens:,}")
    print(f"  Completion tokens:     {metrics.total_completion_tokens:,}")
    print(f"  Total tokens:          {metrics.total_tokens:,}")
    print(f"  Avg tokens/request:    {metrics.avg_tokens_per_request:.0f}")
    
    print("\n⏱️  TIMING")
    print("-" * 40)
    print(f"  Total duration:        {metrics.total_duration:.2f}s ({metrics.total_duration/60:.1f} min)")
    print(f"  Avg request duration:  {metrics.avg_request_duration:.2f}s")
    print(f"  Requests/second:       {metrics.requests_per_second:.2f}")
    print(f"  Tokens/second:         {metrics.tokens_per_second:.0f}")
    
    print("\n📊 RATE LIMIT ANALYSIS")
    print("-" * 40)
    effective_rpm = metrics.successful_requests / (metrics.total_duration / 60) if metrics.total_duration else 0
    effective_tpm = metrics.total_tokens / (metrics.total_duration / 60) if metrics.total_duration else 0
    print(f"  Effective RPM:         {effective_rpm:.0f} (limit: {RPM_LIMIT})")
    print(f"  Effective TPM:         {effective_tpm:,.0f} (limit: {TPM_LIMIT:,})")
    print(f"  RPM utilization:       {(effective_rpm / RPM_LIMIT * 100):.1f}%")
    print(f"  TPM utilization:       {(effective_tpm / TPM_LIMIT * 100):.1f}%")
    
    # Identify bottleneck
    rpm_headroom = RPM_LIMIT - effective_rpm
    tpm_headroom = TPM_LIMIT - effective_tpm
    if effective_rpm > 0 and effective_tpm > 0:
        rpm_pct = effective_rpm / RPM_LIMIT
        tpm_pct = effective_tpm / TPM_LIMIT
        if rpm_pct > tpm_pct:
            print(f"  Bottleneck:            RPM (request-limited)")
        else:
            print(f"  Bottleneck:            TPM (token-limited)")
    
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    # Calculate optimal concurrency
    avg_duration = metrics.avg_request_duration
    if avg_duration > 0:
        # Theoretical max concurrent based on RPM
        theoretical_concurrent = (RPM_LIMIT / 60) * avg_duration
        safe_concurrent = int(theoretical_concurrent * 0.7)
        safe_concurrent = max(1, min(safe_concurrent, 100))
        print(f"  Recommended max_concurrent: {safe_concurrent}")
    
    # Estimate time for larger runs
    if metrics.requests_per_second > 0:
        for target in [500, 1000, 2000, 5000]:
            est_time = target / metrics.requests_per_second
            print(f"  Est. time for {target:,} personas: {est_time/60:.1f} min")
    
    print("=" * 70)


def save_results(metrics: AggregateMetrics, output_file: Path) -> None:
    """Save detailed results to a JSON file."""
    
    results = {
        "summary": {
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "rate_limited_requests": metrics.rate_limited_requests,
            "total_retries": metrics.total_retries,
            "total_prompt_tokens": metrics.total_prompt_tokens,
            "total_completion_tokens": metrics.total_completion_tokens,
            "total_tokens": metrics.total_tokens,
            "avg_tokens_per_request": metrics.avg_tokens_per_request,
            "total_duration_seconds": metrics.total_duration,
            "avg_request_duration_seconds": metrics.avg_request_duration,
            "requests_per_second": metrics.requests_per_second,
            "tokens_per_second": metrics.tokens_per_second,
        },
        "config": {
            "deployment": DEPLOYMENT,
            "tpm_limit": TPM_LIMIT,
            "rpm_limit": RPM_LIMIT,
        },
        "requests": [
            {
                "request_id": r.request_id,
                "duration": r.duration,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "success": r.success,
                "error": r.error,
                "retry_count": r.retry_count,
                "rate_limited": r.rate_limited,
            }
            for r in metrics.requests
        ],
        "timestamp": datetime.now().isoformat(),
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test large-scale persona generation with rate limit and token monitoring"
    )
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=250,
        help="Number of personas to generate (default: 250)",
    )
    parser.add_argument(
        "-m", "--max-concurrent",
        type=int,
        default=15,
        help="Maximum concurrent API calls (default: 15)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/persona_scale_test_results.json"),
        help="Output file for detailed results (default: data/persona_scale_test_results.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )
    
    args = parser.parse_args()
    
    print("🔍 Large-Scale Persona Generation Test")
    print(f"   Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"   Deployment: {DEPLOYMENT}")
    print(f"   Time: {datetime.now().isoformat()}")
    
    output_file = None if args.no_save else args.output
    
    asyncio.run(
        run_persona_generation_test(
            count=args.count,
            max_concurrent=args.max_concurrent,
            output_file=output_file,
        )
    )


if __name__ == "__main__":
    main()
