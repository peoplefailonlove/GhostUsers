#!/usr/bin/env python3
"""Audience Characteristics Generator using Pydantic for structured output."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from llm_judge import check_persona

load_dotenv()

PROVIDER_AZURE = "azure"

# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class GeneratedProfile(BaseModel):
    """LLM output schema for generated audience profile."""

    name: str = Field(
        description="A realistic full name appropriate for the persona's demographic background (location, ethnicity, gender)"
    )
    about: str = Field(
        description="Behavioral description focusing on interests, digital habits, creative pursuits, and lifestyle preferences"
    )
    goalsAndMotivations: list[str] = Field(description="List of 3 goals/motivations")
    frustrations: list[str] = Field(description="List of 3 frustrations")
    needState: str = Field(description="Current psychological or motivational state")
    occasions: str = Field(description="Contextual situations for content engagement")


class GeneratedMember(BaseModel):
    """Complete generated member with ID and profile."""

    member_id: str
    name: str
    about: str
    goals_and_motivations: list[str]
    frustrations: list[str]
    need_state: str
    occasions: str


# ============================================================================
# System Prompt with JSON Schema
# ============================================================================

GENERATION_SYSTEM_PROMPT = """You are an expert persona generator creating realistic audience member profiles.

Generate a realistic, believable individual that:
- Embodies the spirit and characteristics of the base persona
- Has internally consistent traits and behaviors
- Feels like a real person, not a stereotype

NAME GENERATION RULES:
- Generate a completely RANDOM and UNIQUE full name for each person
- NEVER repeat or reuse names across profiles—each name must be distinct
- Use diverse first names and surnames—avoid common/overused names like "Ritvik", "Priya", "Sharma", "Nair"
- Match the name to the persona's location, ethnicity, and gender
- Be creative: draw from a wide variety of cultural naming conventions

You MUST respond with valid JSON containing EXACTLY these fields:
- "name": (string) A realistic full name appropriate for the persona's demographic
- "about": (string) Behavioral description focusing on interests, digital habits, creative pursuits, and lifestyle
- "goalsAndMotivations": (array of 3 strings) List of goals and motivations
- "frustrations": (array of 3 strings) List of frustrations
- "needState": (string) Current psychological or motivational state
- "occasions": (string) Contextual situations for content engagement

Example output:
{
    "name": "Kavitha Menon",
    "about": "A creative professional who thrives on innovation...",
    "goalsAndMotivations": [
        "To scale business operations while maintaining quality",
        "To continuously learn emerging industry trends",
        "To create lasting impact through meaningful work"
    ],
    "frustrations": [
        "Managing workflows with limited team resources",
        "Maintaining quality standards under tight deadlines",
        "Limited access to premium tools and platforms"
    ],
    "needState": "Driven and resourceful, seeking growth opportunities",
    "occasions": "Engages with content during morning planning and evening wind-down"
}

IMPORTANT: Return ONLY the JSON object with actual values. Do NOT return a schema definition or type descriptions."""


def create_generation_prompt(member: dict[str, Any]) -> str:
    """
    Create the generation prompt for a single audience member.

    Args:
        member: Audience member dictionary with attributes, persona_template, and screener_responses

    Returns:
        Formatted prompt string for characteristic generation
    """
    persona = member.get("persona_template", {})
    screener_responses = member.get("screener_responses", [])

    # Format screener Q&A
    screener_section = ""
    if screener_responses:
        screener_lines = []
        for response in screener_responses:
            question = response.get("question", "N/A")
            answer = response.get("answer", "N/A")
            screener_lines.append(f"- **Q**: {question}\n  **A**: {answer}")
        screener_section = "\n".join(screener_lines)
    else:
        screener_section = "No screener responses available."

    prompt = f"""Generate a detailed audience member profile for the following persona:

## Base Persona Template
- **About**: {persona.get('about', 'N/A')}
- **Goals & Motivations**: {persona.get('goals_and_motivations', 'N/A')}
- **Frustrations**: {persona.get('frustrations', 'N/A')}
- **Need State**: {persona.get('need_state', 'N/A')}
- **Occasions**: {persona.get('occasions', 'N/A')}

Above information is enough to understand persona's traits and behavior. Use the screener responses below to create variations and generate a complete, realistic audience member profile as JSON.

## Screener Responses
{screener_section}

## Important Guidelines
1. Use the screener responses to inform lifestyle, work environment, and behavioral descriptions
2. Ensure the generated profile is consistent with the screener answers
3. The profile should feel like a real person, not a stereotype
4. Maintain the spirit of the base persona while adapting to the screener context
5. Generate a RANDOM, UNIQUE full name—avoid common names like "Ritvik", "Priya", "Sharma", "Nair". Be creative and diverse

Generate a complete, realistic audience member profile as JSON."""

    # Add feedback from previous failed validation attempt
    if member.get("previous_validation_feedback"):
        feedback = member["previous_validation_feedback"]
        prompt += f"""

## ⚠️ PREVIOUS ATTEMPT WAS REJECTED
Reason: {feedback}

Fix this issue. Do NOT make the same mistake again."""

    return prompt


def _create_azure_client() -> tuple[AzureChatOpenAI, str]:
    """
    Create Azure OpenAI LangChain client.

    Returns:
        Tuple of (client, deployment_name)

    Raises:
        ValueError: If required environment variables are not set
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"
    )
    api_version = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")

    if not api_key or not endpoint:
        raise ValueError(
            "Azure OpenAI not configured. Set the following environment variables:\n"
            "  - AZURE_OPENAI_API_KEY\n"
            "  - AZURE_OPENAI_ENDPOINT\n"
            "  - AZURE_OPENAI_DEPLOYMENT_NAME (optional, defaults to gpt-4o)\n"
            "  - OPENAI_API_VERSION (optional)"
        )

    client = AzureChatOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        temperature=0.8,
        max_tokens=4096,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    return client, deployment


async def _call_llm(
    client: AzureChatOpenAI,
    deployment: str,
    prompt: str,
) -> str:
    """
    Make an async LLM API call and return the response content.

    Args:
        client: AzureChatOpenAI client
        deployment: Azure deployment name (unused, configured in client)
        prompt: User prompt

    Returns:
        Response content string

    Raises:
        ValueError: If API returns no content
    """
    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = await client.ainvoke(messages)

    content = response.content
    if not content:
        raise ValueError("API returned empty content")
    return str(content).strip()


def _parse_llm_response(
    content: str, member_id: str, audience_index: int
) -> GeneratedMember:
    """
    Parse LLM response using Pydantic validation.

    Args:
        content: JSON string from LLM
        member_id: ID for this member
        audience_index: Index of the audience

    Returns:
        GeneratedMember with validated data

    Raises:
        ValidationError: If response doesn't match schema
        json.JSONDecodeError: If response isn't valid JSON
    """
    # Parse JSON and validate with Pydantic
    data = json.loads(content)
    profile = GeneratedProfile.model_validate(data)

    return GeneratedMember(
        member_id=member_id,
        name=profile.name,
        about=profile.about,
        goals_and_motivations=profile.goalsAndMotivations,
        frustrations=profile.frustrations,
        need_state=profile.needState,
        occasions=profile.occasions,
    )


class ValidatedMember(BaseModel):
    """Generated member with validation scores."""

    member_id: str
    name: str
    about: str
    goals_and_motivations: list[str]
    frustrations: list[str]
    need_state: str
    occasions: str
    screener_score: float
    parent_score: float
    validation_passed: bool
    validation_explanation: str
    validation_attempt: int  # Which attempt succeeded (1-indexed)


# ============================================================================
# Validation Thresholds (never trust the model's pass/fail)
# ============================================================================

SCREENER_THRESHOLD = 0.88
PARENT_THRESHOLD = 0.88
MAX_VALIDATION_RETRIES = 3
MAX_GENERATION_ROUNDS = 5  # Max rounds to try filling the sample size


async def generate_member(
    client: AzureChatOpenAI,
    deployment: str,
    member: dict[str, Any],
) -> tuple[ValidatedMember | None, int]:
    """
    Generate characteristics for a single audience member with smart retry validation.

    Uses LLM judge to validate generated personas. On failure, feeds the rejection
    reason back into the next generation attempt for self-correction.

    Args:
        client: AsyncAzureOpenAI client
        deployment: Azure deployment name
        member: Member data with persona_template and screener_responses

    Returns:
        Tuple of (ValidatedMember or None, number of attempts used)
    """
    member_id = member.get("member_id", "unknown")
    audience_index = member.get("audience_index", -1)
    persona_template = member.get("persona_template", {})
    screener_responses = member.get("screener_responses", [])

    # Create a mutable copy for feedback injection
    member_state = dict(member)

    for attempt in range(1, MAX_VALIDATION_RETRIES + 1):
        try:
            # 1. Generate the persona (prompt includes feedback if available)
            prompt = create_generation_prompt(member_state)
            content = await _call_llm(client, deployment, prompt)
            generated = _parse_llm_response(content, member_id, audience_index)

            # 2. Run LLM judge
            judgment = await check_persona(
                persona_template,
                screener_responses,
                generated.model_dump()
            )

            # 3. WE decide pass/fail (never trust the model)
            screener_score = judgment.get("screener_score", 0.0)
            parent_score = judgment.get("parent_score", 0.0)
            passed = (screener_score >= SCREENER_THRESHOLD and 
                      parent_score >= PARENT_THRESHOLD)

            # 4. Build the result with scores
            result = ValidatedMember(
                member_id=generated.member_id,
                name=generated.name,
                about=generated.about,
                goals_and_motivations=generated.goals_and_motivations,
                frustrations=generated.frustrations,
                need_state=generated.need_state,
                occasions=generated.occasions,
                screener_score=round(screener_score, 3),
                parent_score=round(parent_score, 3),
                validation_passed=passed,
                validation_explanation=judgment.get("explanation", ""),
                validation_attempt=attempt
            )

            # 5. If passed → accept and return
            if passed:
                print(f"  {member_id}: ✓ PASSED on attempt {attempt} → "
                      f"screener={screener_score:.3f}, parent={parent_score:.3f}")
                return result, attempt

            # 6. If failed → log and prepare for retry with feedback
            print(f"  {member_id}: ✗ FAILED attempt {attempt}/{MAX_VALIDATION_RETRIES} → "
                  f"{judgment.get('explanation', 'No reason')}")

            if attempt < MAX_VALIDATION_RETRIES:
                # Inject rejection reason for next generation attempt
                member_state["previous_validation_feedback"] = judgment.get("explanation", "")
                await asyncio.sleep(0.3 * attempt)  # Brief backoff
            else:
                # Final attempt failed - return with failed flag
                print(f"  {member_id}: ✗ REJECTED after {MAX_VALIDATION_RETRIES} attempts")
                return result, attempt

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  {member_id}: Parse error on attempt {attempt}: {e}")
            if attempt < MAX_VALIDATION_RETRIES:
                member_state["previous_validation_feedback"] = f"JSON/validation error: {e}"
                await asyncio.sleep(0.5 * attempt)
            else:
                print(f"  {member_id}: ✗ REJECTED after {MAX_VALIDATION_RETRIES} attempts (parse errors)")
                return None, attempt

        except Exception as e:
            print(f"  {member_id}: API error on attempt {attempt}: {e}")
            if attempt < MAX_VALIDATION_RETRIES:
                await asyncio.sleep(1.0 * attempt)
            else:
                print(f"  {member_id}: ✗ REJECTED after {MAX_VALIDATION_RETRIES} attempts (API errors)")
                return None, attempt

    return None, MAX_VALIDATION_RETRIES


def convert_persona_to_template(persona: dict[str, Any]) -> dict[str, Any]:
    """
    Convert persona object from input format to persona_template format.

    Args:
        persona: Persona object with fields like personaName, about, etc.

    Returns:
        Persona template dictionary with standardized field names
    """
    return {
        "id": persona.get("id"),
        "name": persona.get("personaName", ""),
        "type": persona.get("personaType", ""),
        "gender": persona.get("gender", ""),
        "age": persona.get("age"),
        "location": persona.get("location", ""),
        "ethnicity": persona.get("ethnicity", ""),
        "about": persona.get("about", ""),
        "goals_and_motivations": persona.get("goalsAndMotivations", ""),
        "frustrations": persona.get("frustrations", ""),
        "need_state": persona.get("needState", ""),
        "occasions": persona.get("occasions", ""),
    }


def convert_audience_to_members(
    audience_data: dict[str, Any], audience_index: int
) -> list[dict[str, Any]]:
    """
    Convert audience data to member format expected by generation functions.

    Args:
        audience_data: Audience dictionary with persona, screenerQuestions, and sampleSize
        audience_index: Index of this audience in the input

    Returns:
        List of member dictionaries ready for characteristic generation
    """
    persona = audience_data.get("persona", {})
    persona_template = convert_persona_to_template(persona)
    screener_questions = audience_data.get("screenerQuestions", [])
    sample_size = audience_data.get("sampleSize", 1)

    members = []
    for idx in range(sample_size):
        member = {
            "member_id": f"AUD{audience_index}_{idx + 1:04d}",
            "audience_index": audience_index,
            "persona_template": persona_template,
            "screener_responses": screener_questions,
        }
        members.append(member)

    return members


# ============================================================================
# Progress Tracking
# ============================================================================


class ProgressTracker:
    """Thread-safe progress tracker for parallel generation with validation stats."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.passed = 0
        self.failed = 0
        self.total_attempts = 0
        self._lock = asyncio.Lock()

    async def record(self, member_id: str, passed: bool, attempts: int) -> None:
        """Record a completed member with its validation result."""
        async with self._lock:
            self.completed += 1
            self.total_attempts += attempts
            if passed:
                self.passed += 1
            else:
                self.failed += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current validation statistics."""
        return {
            "completed": self.completed,
            "passed": self.passed,
            "failed": self.failed,
            "total_attempts": self.total_attempts,
            "avg_attempts": round(self.total_attempts / self.completed, 2) if self.completed > 0 else 0,
            "pass_rate": round((self.passed / self.completed) * 100, 1) if self.completed > 0 else 0,
        }


# ============================================================================
# Parallel Generation
# ============================================================================


async def _generate_with_semaphore(
    client: AzureChatOpenAI,
    deployment: str,
    member: dict[str, Any],
    semaphore: asyncio.Semaphore,
    progress: ProgressTracker,
) -> ValidatedMember | None:
    """Generate a single member with rate limiting and progress tracking."""
    async with semaphore:
        result, attempts = await generate_member(client, deployment, member)
        if result:
            await progress.record(result.member_id, result.validation_passed, attempts)
        else:
            # Generation completely failed
            await progress.record(member.get("member_id", "unknown"), False, attempts)
        return result


def _build_audience_result_from_members(
    audience_data: dict[str, Any],
    audience_index: int,
    members: list[ValidatedMember],
    generation_time: float,
    progress: ProgressTracker,
) -> dict[str, Any]:
    """Build result dictionary from a list of validated (passing) members."""
    generated = [m.model_dump() for m in members]
    
    total_screener_score = sum(m.screener_score for m in members)
    total_parent_score = sum(m.parent_score for m in members)
    total_attempts = sum(m.validation_attempt for m in members)
    
    count = len(members)
    avg_screener = total_screener_score / count if count > 0 else 0.0
    avg_parent = total_parent_score / count if count > 0 else 0.0
    avg_attempts = total_attempts / count if count > 0 else 0.0

    stats = progress.get_stats()

    return {
        "generated_audience": generated,
        "metadata": {
            "audience_index": audience_index,
            "sample_size": audience_data.get("sampleSize", 0),
            "actual_count": count,
            "persona": audience_data.get("persona", {}),
            "screener_questions": audience_data.get("screenerQuestions", []),
            "generation_stats": {
                "total_generated": count,
                "all_validation_passed": True,  # Only passing members included
                "total_attempts_made": stats.get("total_attempts", 0),
                "total_passed": stats.get("passed", 0),
                "total_failed": stats.get("failed", 0),
                "avg_screener_score": round(avg_screener, 3),
                "avg_parent_score": round(avg_parent, 3),
                "avg_attempts_per_member": round(avg_attempts, 2),
            },
            "generation_time_seconds": round(generation_time, 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


async def generate_audience_characteristics(
    client: AzureChatOpenAI,
    deployment: str,
    audience_data: dict[str, Any],
    audience_index: int,
    max_concurrent: int = 10,
) -> tuple[dict[str, Any], ProgressTracker]:
    """
    Generate characteristics for all members in a single audience.
    
    Guarantees exactly sampleSize PASSING members by regenerating failed ones.

    Args:
        client: AsyncAzureOpenAI client
        deployment: Azure deployment name
        audience_data: Audience dictionary with persona, screenerQuestions, sampleSize
        audience_index: Index of this audience
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        Tuple of (result dictionary, progress tracker with stats)
    """
    start_time = time.time()
    sample_size = audience_data.get("sampleSize", 1)
    persona_template = convert_persona_to_template(audience_data.get("persona", {}))
    screener_questions = audience_data.get("screenerQuestions", [])

    print(f"\nGenerating {sample_size} members for Audience {audience_index}...")

    semaphore = asyncio.Semaphore(max_concurrent)
    progress = ProgressTracker(sample_size)
    
    # Collect passing members
    passed_members: list[ValidatedMember] = []
    member_counter = 0
    generation_round = 0
    
    while len(passed_members) < sample_size and generation_round < MAX_GENERATION_ROUNDS:
        generation_round += 1
        needed = sample_size - len(passed_members)
        
        if generation_round > 1:
            print(f"  Round {generation_round}: Need {needed} more passing members...")
        
        # Create members for this round
        members_to_generate = []
        for _ in range(needed):
            member_counter += 1
            members_to_generate.append({
                "member_id": f"AUD{audience_index}_{member_counter:04d}",
                "audience_index": audience_index,
                "persona_template": persona_template,
                "screener_responses": screener_questions,
            })
        
        # Generate in parallel
        tasks = [
            _generate_with_semaphore(client, deployment, m, semaphore, progress)
            for m in members_to_generate
        ]
        results = await asyncio.gather(*tasks)
        
        # Collect only passing members
        for result in results:
            if result and result.validation_passed:
                passed_members.append(result)
                if len(passed_members) >= sample_size:
                    break
    
    # Check if we got enough
    if len(passed_members) < sample_size:
        print(f"  ⚠️ Warning: Only got {len(passed_members)}/{sample_size} passing members after {MAX_GENERATION_ROUNDS} rounds")
    
    # Renumber member IDs sequentially for clean output
    final_members: list[ValidatedMember] = []
    for idx, member in enumerate(passed_members[:sample_size]):
        # Create new member with sequential ID
        renumbered = ValidatedMember(
            member_id=f"AUD{audience_index}_{idx + 1:04d}",
            name=member.name,
            about=member.about,
            goals_and_motivations=member.goals_and_motivations,
            frustrations=member.frustrations,
            need_state=member.need_state,
            occasions=member.occasions,
            screener_score=member.screener_score,
            parent_score=member.parent_score,
            validation_passed=member.validation_passed,
            validation_explanation=member.validation_explanation,
            validation_attempt=member.validation_attempt
        )
        final_members.append(renumbered)

    result = _build_audience_result_from_members(
        audience_data, audience_index, final_members, time.time() - start_time, progress
    )
    return result, progress


async def generate_all_parallel(
    client: AzureChatOpenAI,
    deployment: str,
    audiences: list[dict[str, Any]],
    max_concurrent: int = 10,
) -> tuple[list[dict[str, Any]], ProgressTracker]:
    """
    Generate characteristics for ALL audiences.
    
    Each audience is processed to guarantee exactly sampleSize passing members.
    
    Returns:
        Tuple of (list of audience results, combined progress tracker)
    """
    start_time = time.time()
    total_samples = sum(aud.get("sampleSize", 0) for aud in audiences)
    
    print(f"\nGenerating {total_samples} members across {len(audiences)} audiences...")
    print(f"(Will regenerate until each audience has exactly sampleSize passing members)")

    # Process each audience (they handle their own retry logic)
    enriched = []
    combined_progress = ProgressTracker(total_samples)
    
    for idx, aud in enumerate(audiences):
        result, aud_progress = await generate_audience_characteristics(
            client, deployment, aud, idx, max_concurrent
        )
        enriched.append(result)
        # Aggregate stats
        combined_progress.passed += aud_progress.passed
        combined_progress.failed += aud_progress.failed
        combined_progress.total_attempts += aud_progress.total_attempts
        combined_progress.completed += aud_progress.completed

    print(f"\nCompleted in {time.time() - start_time:.2f}s")
    return enriched, combined_progress


async def run_generation_async(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = 10,
    parallel_mode: bool = True,
) -> dict[str, Any]:
    """
    Run characteristic generation on all audiences in the input file using async.
    Uses Azure OpenAI.

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write generated results
        max_concurrent: Maximum concurrent API calls (global limit in parallel mode)
        parallel_mode: If True, generate all audiences/personas in parallel with
                       a single global semaphore. If False, process audiences
                       sequentially with per-audience concurrency.

    Returns:
        Complete results dictionary with generated characteristics
    """
    # Initialize Azure OpenAI async client
    client, deployment = _create_azure_client()
    print(f"Provider: Azure OpenAI (deployment: {deployment})")

    # Load input data (personas_input format with audiences array)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audiences = data.get("audiences", [])
    total_samples = sum(aud.get("sampleSize", 0) for aud in audiences)

    print(f"Loaded {len(audiences)} audiences from {input_path}")
    print(f"Total samples to generate: {total_samples}")
    print(f"Parallel mode: {parallel_mode}")

    global_progress = None
    
    if parallel_mode:
        enriched_audiences, global_progress = await generate_all_parallel(
            client, deployment, audiences, max_concurrent
        )
    else:
        # Sequential audiences with per-audience concurrency
        enriched_audiences = []
        combined_progress = ProgressTracker(total_samples)
        for idx, audience_data in enumerate(audiences):
            result, aud_progress = await generate_audience_characteristics(
                client, deployment, audience_data, idx, max_concurrent
            )
            enriched_audiences.append(result)
            # Aggregate stats
            combined_progress.passed += aud_progress.passed
            combined_progress.failed += aud_progress.failed
            combined_progress.total_attempts += aud_progress.total_attempts
            combined_progress.completed += aud_progress.completed
        global_progress = combined_progress

    # Compile results - all members in output are passing
    total_generated = sum(
        aud["metadata"]["generation_stats"]["total_generated"]
        for aud in enriched_audiences
    )
    total_failed = sum(
        aud["metadata"]["generation_stats"]["total_failed"] for aud in enriched_audiences
    )

    # Get validation stats
    validation_stats = global_progress.get_stats() if global_progress else {}

    results = {
        "project_name": data.get("projectName"),
        "project_description": data.get("projectDescription"),
        "project_id": data.get("projectId"),
        "user_id": data.get("userId"),
        "request_id": data.get("requestId"),
        "generation_model": f"{PROVIDER_AZURE}:{deployment}",
        "provider": PROVIDER_AZURE,
        "total_audiences": len(enriched_audiences),
        "total_members_processed": total_generated + total_failed,
        "total_successfully_generated": total_generated,
        "total_failed": total_failed,
        "validation_summary": {
            "passed": validation_stats.get("passed", 0),
            "failed": validation_stats.get("failed", 0),
            "pass_rate_percent": validation_stats.get("pass_rate", 0),
            "avg_attempts_per_member": validation_stats.get("avg_attempts", 0),
            "total_validation_attempts": validation_stats.get("total_attempts", 0),
        },
        "audiences": enriched_audiences,
    }

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def run_generation(
    input_path: Path,
    output_path: Path,
    max_concurrent: int = 10,
    parallel_mode: bool = True,
) -> dict[str, Any]:
    """
    Synchronous wrapper for run_generation_async.

    Args:
        input_path: Path to audience samples JSON file
        output_path: Path to write generated results
        max_concurrent: Maximum concurrent API calls
        parallel_mode: If True, generate all audiences/personas in parallel

    Returns:
        Complete results dictionary with generated characteristics
    """
    return asyncio.run(
        run_generation_async(input_path, output_path, max_concurrent, parallel_mode)
    )


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary of generation results."""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Project: {results.get('project_name')}")
    print(f"Model: {results.get('generation_model')}")
    print(f"Total Members Processed: {results.get('total_members_processed')}")
    print(f"Successfully Generated: {results.get('total_successfully_generated')}")
    print(f"Failed: {results.get('total_failed')}")
    
    # Validation summary
    val_summary = results.get("validation_summary", {})
    if val_summary:
        print("\n" + "-" * 60)
        print("VALIDATION SUMMARY")
        print("-" * 60)
        passed = val_summary.get("passed", 0)
        total = passed + val_summary.get("failed", 0)
        pass_rate = val_summary.get("pass_rate_percent", 0)
        avg_attempts = val_summary.get("avg_attempts_per_member", 0)
        print(f"Passed: {passed}/{total} ({pass_rate}%)")
        print(f"Average attempts per member: {avg_attempts}")
    
    print("-" * 60)

    for aud in results.get("audiences", []):
        metadata = aud.get("metadata", {})
        stats = metadata.get("generation_stats", {})
        print(f"\nAudience {metadata.get('audience_index')}:")
        print(f"  Persona: {metadata.get('persona', {}).get('personaName', 'N/A')}")
        print(f"  Sample Size: {metadata.get('sample_size', 0)}")
        print(f"  Generated (all passed): {stats.get('total_generated', 0)}")
        print(f"  Avg Screener Score: {stats.get('avg_screener_score', 0)}")
        print(f"  Avg Parent Score: {stats.get('avg_parent_score', 0)}")

        # Show a sample generated member
        generated_audience = aud.get("generated_audience", [])
        for member in generated_audience:
            if "generation_error" not in member:
                print(f"\n  Sample Generated Member ({member.get('member_id')}):")
                about = member.get("about", "")
                if len(about) > 150:
                    about = about[:150] + "..."
                print(f"    About: {about}")
                print(f"    Need State: {member.get('need_state')}")
                goals = member.get("goals_and_motivations", [])
                if goals:
                    print(f"    Goals: {goals[0]}...")
                break


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate detailed audience characteristics using persona templates "
        "and screener questions as input to Azure OpenAI LLM."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/digital_persona_output_10.json"),
        help="Path to personas input JSON file (default: data/digital_persona_output_10.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/audience_characteristics_small_langchain.json"),
        help="Path to output file (default: data/audience_characteristics_small.json)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process audiences sequentially instead of in parallel",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    print("Starting characteristic generation...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max Concurrent Requests: {args.concurrent}")

    start_time = time.time()

    try:
        results = run_generation(
            args.input,
            args.output,
            args.concurrent,
            parallel_mode=not args.sequential,
        )

        elapsed_time = time.time() - start_time

        print_summary(results)
        print(f"\nFull results written to: {args.output}")
        print(f"\n{'=' * 60}")
        print(f"TOTAL TIME: {elapsed_time:.2f} seconds")
        print(f"{'=' * 60}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()
