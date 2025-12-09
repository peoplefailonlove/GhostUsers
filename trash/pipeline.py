#!/usr/bin/env python3
"""
Pipeline orchestrator for the full survey workflow:
    1. /process - Parse document to extract questionnaire JSON
    2. /generate_audience - Generate synthetic audience characteristics
    3. update_persona_api - Called automatically by generate_audience
    4. /simulate-survey - Run LLM-based survey simulation

Usage:
    python pipeline.py --input-blob-url <URL> --project-id <ID> --user-id <ID>

Or call run_pipeline() programmatically.
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# Logging setup
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pipeline")

# API base URL (default to localhost for development)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class PipelineError(Exception):
    """Custom exception for pipeline failures."""

    def __init__(self, step: str, message: str, response: Optional[dict] = None):
        self.step = step
        self.message = message
        self.response = response
        super().__init__(f"[{step}] {message}")


def call_process(
    input_blob_url: str,
    output_container: str = "samplefiles",
    output_blob_prefix: str = "output_file",
) -> dict[str, Any]:
    """
    Step 1: Call /process endpoint to parse document and extract questionnaire JSON.

    Args:
        input_blob_url: Full blob URL to DOCX/PDF document
        output_container: Container for output JSON
        output_blob_prefix: Prefix for output blob name

    Returns:
        Response dict with output_blob (SAS URL) and questions_labels
    """
    logger.info("Step 1: Calling /process endpoint...")

    payload = {
        "input_blob_url": input_blob_url,
        "output_container": output_container,
        "output_blob_prefix": output_blob_prefix,
    }

    response = requests.post(
        f"{API_BASE_URL}/process/",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )

    data = response.json()

    if data.get("status") != "success":
        raise PipelineError("process", f"Failed: {data.get('message', 'Unknown error')}", data)

    logger.info(f"Step 1 complete: Extracted {data.get('questions_extracted')} questions")
    logger.info(f"  Output blob: {data.get('output_blob')}")

    return data


def call_generate_audience(
    input_data: dict[str, Any],
    output_blob_prefix: str = "audience_output",
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """
    Step 2: Call /generate_audience endpoint to generate synthetic audience.

    This endpoint is async - it returns a task_id immediately and processes in background.
    The update_persona_api is called automatically when generation completes.

    Args:
        input_data: Full input JSON with projectId, userId, audiences, etc.
        output_blob_prefix: Prefix for output blob name
        max_concurrent: Max concurrent LLM calls

    Returns:
        Response dict with task_id and status
    """
    logger.info("Step 2: Calling /generate_audience endpoint...")

    payload = {
        "input_data": input_data,
        "output_blob_prefix": output_blob_prefix,
        "max_concurrent": max_concurrent,
    }

    response = requests.post(
        f"{API_BASE_URL}/generate_audience/",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )

    if response.status_code != 200:
        raise PipelineError(
            "generate_audience",
            f"HTTP {response.status_code}: {response.text}",
        )

    data = response.json()
    logger.info(f"Step 2 initiated: task_id={data.get('task_id')}, status={data.get('status')}")
    logger.info("  Note: Generation runs in background. update_persona_api will be called on completion.")

    return data


def call_simulate_survey(
    questionnaire_blob_url: str,
    audience_blob_url: str,
    output_blob_prefix: str = "survey_llm_results",
    task_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Step 4: Call /simulate-survey endpoint to run LLM-based survey simulation.

    Args:
        questionnaire_blob_url: SAS URL to questionnaire JSON
        audience_blob_url: SAS URL to generated audience JSON
        output_blob_prefix: Prefix for output blob name
        task_id: Task ID from generate_audience step (for tracking/correlation)

    Returns:
        Response dict with output_blob, total_members, total_questions, etc.
    """
    logger.info("Step 4: Calling /simulate-survey endpoint...")
    if task_id:
        logger.info(f"  Linked to generate_audience task_id: {task_id}")

    payload = {
        "questionnaire_blob_url": questionnaire_blob_url,
        "audience_blob_url": audience_blob_url,
        "output_blob_prefix": output_blob_prefix,
    }
    
    if task_id:
        payload["task_id"] = task_id

    response = requests.post(
        f"{API_BASE_URL}/simulate-survey/",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=3600,  # Long timeout for survey simulation
    )

    if response.status_code != 200:
        raise PipelineError(
            "simulate-survey",
            f"HTTP {response.status_code}: {response.text}",
        )

    data = response.json()

    if data.get("status") != "success":
        raise PipelineError("simulate-survey", f"Failed: {data}", data)

    logger.info(f"Step 4 complete: Simulated survey for {data.get('total_members')} members")
    logger.info(f"  Total questions: {data.get('total_questions')}")
    logger.info(f"  Processing time: {data.get('processing_time_seconds')}s")
    logger.info(f"  Output blob: {data.get('output_blob')}")
    if task_id:
        logger.info(f"  Correlated with task_id: {task_id}")

    return data


def run_pipeline(
    input_blob_url: str,
    project_id: int,
    user_id: int,
    project_name: str = "Pipeline Project",
    project_description: str = "",
    audiences: Optional[list[dict]] = None,
    output_container: str = "samplefiles",
    max_concurrent: int = 10,
    run_survey: bool = True,
    audience_blob_url: Optional[str] = None,
    task_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the complete pipeline:
        1. /process - Parse document
        2. /generate_audience - Generate synthetic audience (calls update_persona_api internally)
        3. /simulate-survey - Run survey simulation (optional)

    Args:
        input_blob_url: Full blob URL to input DOCX/PDF document
        project_id: Project ID for tracking
        user_id: User ID for tracking
        project_name: Project name
        project_description: Project description
        audiences: List of audience definitions (if None, must be provided separately)
        output_container: Container for intermediate outputs
        max_concurrent: Max concurrent LLM calls
        run_survey: Whether to run the survey simulation step
        audience_blob_url: If provided, skip generate_audience and use this URL directly
        task_id: Task ID from a previous generate_audience call (for correlation)

    Returns:
        Dict with results from each step
    """
    start_time = time.time()
    results = {
        "project_id": project_id,
        "user_id": user_id,
        "steps": {},
    }

    try:
        # Step 1: Process document
        process_result = call_process(
            input_blob_url=input_blob_url,
            output_container=output_container,
            output_blob_prefix="questionnaire",
        )
        results["steps"]["process"] = process_result
        questionnaire_blob_url = process_result["output_blob"]

        # Step 2: Generate audience (if audiences provided)
        if audiences and not audience_blob_url:
            input_data = {
                "projectId": project_id,
                "userId": user_id,
                "projectName": project_name,
                "projectDescription": project_description,
                "json_file_url": questionnaire_blob_url,
                "audiences": audiences,
            }

            generate_result = call_generate_audience(
                input_data=input_data,
                output_blob_prefix="audience",
                max_concurrent=max_concurrent,
            )
            results["steps"]["generate_audience"] = generate_result

            # Note: generate_audience is async, so we can't get the output URL immediately
            # The update_persona_api is called automatically when it completes
            logger.info("Step 2 & 3: Audience generation started (async). update_persona_api will be called on completion.")

            if run_survey:
                logger.warning(
                    "Survey simulation requested but generate_audience is async. "
                    "Please wait for generation to complete and call simulate-survey separately "
                    "with the audience_blob_url from the generation output."
                )
                results["steps"]["simulate_survey"] = {
                    "status": "skipped",
                    "reason": "generate_audience is async - call simulate-survey separately after generation completes",
                }

        elif audience_blob_url and run_survey:
            # Step 4: Simulate survey (if audience URL provided directly)
            survey_result = call_simulate_survey(
                questionnaire_blob_url=questionnaire_blob_url,
                audience_blob_url=audience_blob_url,
                output_blob_prefix="survey_results",
                task_id=task_id,
            )
            results["steps"]["simulate_survey"] = survey_result
            if task_id:
                results["task_id"] = task_id

        results["total_time_seconds"] = round(time.time() - start_time, 2)
        results["status"] = "success"

        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {results['total_time_seconds']}s")

        return results

    except PipelineError as e:
        logger.error(f"Pipeline failed at step '{e.step}': {e.message}")
        results["status"] = "failed"
        results["error"] = {
            "step": e.step,
            "message": e.message,
            "response": e.response,
        }
        results["total_time_seconds"] = round(time.time() - start_time, 2)
        return results

    except Exception as e:
        logger.exception(f"Unexpected error in pipeline: {e}")
        results["status"] = "failed"
        results["error"] = {"message": str(e)}
        results["total_time_seconds"] = round(time.time() - start_time, 2)
        return results


def run_survey_only(
    questionnaire_blob_url: str,
    audience_blob_url: str,
    output_blob_prefix: str = "survey_results",
    task_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run only the survey simulation step.

    Useful when generate_audience has completed asynchronously and you have
    both the questionnaire and audience blob URLs.

    Args:
        questionnaire_blob_url: SAS URL to questionnaire JSON
        audience_blob_url: SAS URL to generated audience JSON
        output_blob_prefix: Prefix for output blob name
        task_id: Task ID from generate_audience (for correlation/tracking)

    Returns:
        Survey simulation result
    """
    return call_simulate_survey(
        questionnaire_blob_url=questionnaire_blob_url,
        audience_blob_url=audience_blob_url,
        output_blob_prefix=output_blob_prefix,
        task_id=task_id,
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the full survey pipeline: process → generate_audience → simulate-survey"
    )
    parser.add_argument(
        "--input-blob-url",
        required=True,
        help="Full blob URL to input DOCX/PDF document",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        required=True,
        help="Project ID",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="User ID",
    )
    parser.add_argument(
        "--project-name",
        default="Pipeline Project",
        help="Project name",
    )
    parser.add_argument(
        "--audiences-json",
        help="Path to JSON file containing audiences array",
    )
    parser.add_argument(
        "--audience-blob-url",
        help="SAS URL to pre-generated audience JSON (skips generate_audience step)",
    )
    parser.add_argument(
        "--task-id",
        help="Task ID from generate_audience (for correlation with simulate-survey)",
    )
    parser.add_argument(
        "--output-container",
        default="samplefiles",
        help="Container for intermediate outputs",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent LLM calls",
    )
    parser.add_argument(
        "--no-survey",
        action="store_true",
        help="Skip survey simulation step",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Override API base URL",
    )

    args = parser.parse_args()

    # Override API base URL if provided
    global API_BASE_URL
    if args.api_base_url:
        API_BASE_URL = args.api_base_url

    # Load audiences from file if provided
    audiences = None
    if args.audiences_json:
        with open(args.audiences_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            audiences = data.get("audiences", data) if isinstance(data, dict) else data

    # Run pipeline
    result = run_pipeline(
        input_blob_url=args.input_blob_url,
        project_id=args.project_id,
        user_id=args.user_id,
        project_name=args.project_name,
        audiences=audiences,
        output_container=args.output_container,
        max_concurrent=args.max_concurrent,
        run_survey=not args.no_survey,
        audience_blob_url=args.audience_blob_url,
        task_id=args.task_id,
    )

    # Print result
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    exit(main())
