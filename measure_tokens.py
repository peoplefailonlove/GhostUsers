"""
Measure actual token usage for survey batch calls to compute optimal max_concurrent.

Run: python measure_tokens.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_5", "gpt-5")

# Your system prompt (same as run_survey_parallel.py)
SYSTEM_PROMPT = """
You are an expert survey respondent simulator.

Your job is to generate answers for the given persona using the rules,
constraints, and question metadata provided.

The user message will contain a JSON object with:
- "persona_json": a single respondent persona
- "questions_json": a list of survey questions

Each question contains:
- id
- text
- type (radio, checkbox, grid, select, number, text, html)
- options (when applicable)
- required (true/false)
- notes
- logic_validations (e.g., EMPLOY, COSIZE/REVENUE, FUNCTION, DM rules)

Mandatory rules:
- Always stay consistent with persona characteristics.
- Honor any required/qualification logic implied by notes and logic_validations.
- If the question type is "text", answer in 1–2 concise sentences.
- If type is "radio", select exactly one valid option value from 'options'.
- If "checkbox", select 1–4 valid option values unless logic restricts.
- If "number", return a realistic numeric value within constraints if implied.
- If "grid/matrix", return an object mapping row IDs/labels to column values.
- If "html", always answer with an empty string "".
- Never invent new options or option IDs.

Output format (STRICT):
Return ONLY a JSON array, where each item has this structure:

{
  "id": "<question_id>",
  "answer": <VALUE>
}

Examples:
- radio:   "answer": "r3"
- checkbox:"answer": ["r1", "r4"]
- number:  "answer": 7
- text:    "answer": "Short explanation..."
- html:    "answer": ""

Do NOT include explanations or reasoning. Only return valid JSON.
""".strip()


def measure_single_batch():
    """Simulate a single batch call and measure tokens."""
    
    # Sample persona (realistic size)
    sample_persona = {
        "member_id": "test_001",
        "name": "Sarah Johnson",
        "about": "Marketing Director at a mid-sized tech company with 10 years of experience in B2B marketing.",
        "goals_and_motivations": [
            "Increase brand awareness",
            "Generate qualified leads",
            "Improve marketing ROI"
        ],
        "frustrations": [
            "Limited budget",
            "Difficulty measuring campaign effectiveness",
            "Keeping up with digital trends"
        ],
        "need_state": "Looking for integrated marketing solutions",
        "occasions": "Quarterly planning, campaign launches"
    }
    
    # Sample questions batch (30 questions - your batch_size)
    sample_questions = []
    for i in range(30):
        sample_questions.append({
            "id": f"Q{i+1}",
            "text": f"Sample question {i+1} about your preferences and experiences with the product category?",
            "type": "radio" if i % 3 == 0 else ("checkbox" if i % 3 == 1 else "text"),
            "options": [
                {"value": "r1", "label": "Strongly disagree"},
                {"value": "r2", "label": "Disagree"},
                {"value": "r3", "label": "Neutral"},
                {"value": "r4", "label": "Agree"},
                {"value": "r5", "label": "Strongly agree"},
            ] if i % 3 != 2 else [],
            "required": True,
            "notes": ["Please answer based on your experience"],
            "logic_validations": {}
        })
    
    user_payload = {
        "persona_json": sample_persona,
        "questions_json": sample_questions,
    }
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    
    print("=" * 60)
    print("MEASURING TOKEN USAGE FOR SURVEY BATCH")
    print("=" * 60)
    print(f"\nBatch size: 30 questions")
    print(f"System prompt length: ~{len(SYSTEM_PROMPT)} chars")
    print(f"User payload length: ~{len(json.dumps(user_payload))} chars")
    
    # Make the call
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
    )
    
    usage = response.usage
    
    print(f"\n📊 TOKEN USAGE (single batch of 30 questions):")
    print(f"   Prompt tokens:     {usage.prompt_tokens}")
    print(f"   Completion tokens: {usage.completion_tokens}")
    print(f"   Total tokens:      {usage.total_tokens}")
    
    return usage


def compute_max_concurrent(usage):
    """Compute optimal max_concurrent based on measured usage."""
    
    TPM_LIMIT = 150_000
    RPM_LIMIT = 1_500
    
    tokens_per_request = usage.total_tokens
    
    # How many requests can we do per minute (token-limited)?
    max_requests_by_tokens = TPM_LIMIT // tokens_per_request
    
    # How many requests can we do per minute (request-limited)?
    max_requests_by_rpm = RPM_LIMIT
    
    # The actual limit is the lower of the two
    effective_max_rpm = min(max_requests_by_tokens, max_requests_by_rpm)
    
    # Average request duration (assume ~2 seconds for GPT response)
    avg_request_duration_sec = 2.0
    
    # Max concurrent = requests_per_minute * (duration / 60)
    # But we want to stay safe, so use 70% of theoretical max
    theoretical_concurrent = (effective_max_rpm / 60) * avg_request_duration_sec
    safe_concurrent = int(theoretical_concurrent * 0.7)
    
    # Ensure at least 1, cap at reasonable max
    safe_concurrent = max(1, min(safe_concurrent, 50))
    
    print(f"\n" + "=" * 60)
    print("COMPUTING OPTIMAL max_concurrent")
    print("=" * 60)
    print(f"\n📋 Your Limits:")
    print(f"   TPM: {TPM_LIMIT:,}")
    print(f"   RPM: {RPM_LIMIT:,}")
    
    print(f"\n📊 Per Request:")
    print(f"   Tokens per request: {tokens_per_request}")
    print(f"   Avg duration: ~{avg_request_duration_sec}s")
    
    print(f"\n🧮 Calculations:")
    print(f"   Max requests/min (by TPM): {max_requests_by_tokens}")
    print(f"   Max requests/min (by RPM): {max_requests_by_rpm}")
    print(f"   Effective limit: {effective_max_rpm} req/min")
    print(f"   Theoretical concurrent: {theoretical_concurrent:.1f}")
    
    print(f"\n✅ RECOMMENDED max_concurrent: {safe_concurrent}")
    print(f"   (70% of theoretical to leave headroom)")
    
    # Also show what this means for runtime
    print(f"\n⏱️  Runtime Estimates (with max_concurrent={safe_concurrent}):")
    for num_personas in [10, 50, 100, 200]:
        batches_per_persona = 3  # assuming ~90 questions / 30 batch_size
        total_batches = num_personas * batches_per_persona
        minutes = (total_batches / effective_max_rpm) + (total_batches / safe_concurrent * avg_request_duration_sec / 60)
        print(f"   {num_personas} personas: ~{minutes:.1f} minutes")
    
    return safe_concurrent


def main():
    usage = measure_single_batch()
    recommended = compute_max_concurrent(usage)
    
    print(f"\n" + "=" * 60)
    print(f"UPDATE run_survey_parallel.py:")
    print(f"=" * 60)
    print(f"\n   Change: max_concurrent = {recommended}")
    print(f"\n   In function: run_survey_for_all_members()")
    print("=" * 60)


if __name__ == "__main__":
    main()
