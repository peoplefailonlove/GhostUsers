# app.py
import asyncio
import json
import os
import tempfile
import shutil
import logging
import time
from urllib.parse import urlparse, unquote
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
    BlobSasPermissions,
)

# Import your parser (json_ext.py must be next to this app.py or in PYTHONPATH)
try:
    from json_ext import extract_document_to_json
except Exception as e:
    # Fail fast with helpful message if import fails
    raise ImportError(
        f"Failed to import extract_document_to_json from json_ext.py: {e}"
    )

try:
    from questionnaire_parser import extract_questions_labels
except Exception as e:
    # Fail fast with helpful message if import fails
    raise ImportError(
        f"Failed to import extract_questions_labels from questionnaire_parser.py: {e}"
    )

try:
    from generate_audience_characteristics import (
        _create_azure_client,
        generate_audience_characteristics,
    )
except Exception as e:
    raise ImportError(
        f"Failed to import from generate_audience_characteristics.py: {e}"
    )

# Load .env
load_dotenv()

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("blob-doc-parser")

# FastAPI App
app = FastAPI(
    title="Document Parser & Audience Generator API",
    description="API for document parsing and synthetic audience generation",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Trailing slash redirect middleware
class TrailingSlashMiddleware(BaseHTTPMiddleware):
    """Redirect requests without trailing slash to include one."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        # Skip if already has trailing slash, is root, or has file extension
        if not path.endswith("/") and "." not in path.split("/")[-1] and path != "/":
            return RedirectResponse(
                url=str(request.url).replace(path, path + "/", 1),
                status_code=307,  # Preserve HTTP method
            )
        return await call_next(request)


app.add_middleware(TrailingSlashMiddleware)

# Constants for audience generation
AUDIENCE_OUTPUT_CONTAINER = "generated-synthetic-audience"


# ============== Request/Response Models ==============


# Document Parser Request
class ProcessRequest(BaseModel):
    input_blob_url: str  # full blob URL to DOCX/PDF
    output_container: str = "samplefiles"  # where output JSON will be saved
    output_blob_prefix: str = (
        "output_file"  # prefix for output JSON (timestamp appended)
    )


# Audience Generation Request/Response
class GenerateAudienceRequest(BaseModel):
    """Request body for audience generation with inline JSON data."""

    input_data: dict  # Direct JSON data containing audiences
    output_blob_prefix: str = "audience_output"  # Prefix for output JSON
    max_concurrent: int = Field(default=10, ge=1, le=50)


class GenerateAudienceResponse(BaseModel):
    """Response for audience generation (full result)."""

    task_id: str
    status: str
    output_blob: str  # SAS URL for output
    total_audiences: int
    total_members_processed: int
    total_successfully_generated: int
    total_failed: int
    processing_time_seconds: float


class GenerateAudienceAsyncResponse(BaseModel):
    """Immediate response for async audience generation."""

    task_id: str
    status: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


# Helper: parse container + blob name from full blob URL
def parse_blob_url(blob_url: str):
    parsed = urlparse(blob_url)
    path = parsed.path.lstrip("/")  # samplefiles/input_file.docx

    if "/" not in path:
        raise ValueError("Invalid blob URL format. Expected /container/blobname")

    container, blob_name = path.split("/", 1)
    return unquote(container), unquote(blob_name)


# Helper: create timestamped output blob name (now includes short uuid for uniqueness)
def generate_output_blob_name(prefix: str):
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short_uuid = uuid4().hex[:8]
    safe_prefix = prefix.replace(" ", "_")
    return f"{safe_prefix}_{stamp}_{short_uuid}.json"


# Helper: parse AccountName and AccountKey from connection string
def parse_account_from_connection_string(conn_str: str):
    parts = {}
    for segment in conn_str.split(";"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()
    account_name = parts.get("AccountName")
    account_key = parts.get("AccountKey")
    return account_name, account_key


@app.post("/process")
@app.post("/process/", include_in_schema=False)
def process_file(req: ProcessRequest):
    logger.info(f"Received request for blob: {req.input_blob_url}")

    # Get connection string from env
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        return {
            "status": "Error",
            "message": "AZURE_STORAGE_CONNECTION_STRING not set",
        }

    # Parse account info (needed for SAS)
    account_name, account_key = parse_account_from_connection_string(conn_str)
    if not account_name or not account_key:
        logger.error(
            "AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING. "
            "SAS URL generation requires a connection string with AccountKey."
        )
        return {
            "status": "Error",
            "message": "AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING (cannot generate SAS).",
        }

    # Create BlobServiceClient
    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
    except Exception as e:
        logger.exception("Failed to create BlobServiceClient")
        return {
            "status": "Error",
            "message": f"Failed to create BlobServiceClient: {e}",
        }

    # Parse input blob URL
    try:
        container, blob_name = parse_blob_url(req.input_blob_url)
        logger.info(f"Parsed container={container}, blob={blob_name}")
    except Exception as e:
        return {
            "status": "Error",
            "message": f"Invalid input blob URL: {e}",
        }

    tmp_dir = None

    try:
        # Create a temp dir (unique per request)
        tmp_dir = tempfile.mkdtemp(prefix="doc_parse_")

        # ---------- Download input blob ----------
        input_uuid = uuid4().hex[:8]
        original_filename = Path(blob_name).name
        local_input_path = os.path.join(tmp_dir, f"{input_uuid}_{original_filename}")

        try:
            blob_client = blob_service.get_blob_client(
                container=container,
                blob=blob_name,
            )
            logger.info(f"Downloading blob to {local_input_path}")
            data = blob_client.download_blob().readall()
            with open(local_input_path, "wb") as f:
                f.write(data)
            logger.info("Download complete")
        except Exception as e:
            logger.exception("Blob download failed")
            return {
                "status": "Error",
                "message": f"Failed to download input blob: {e}",
            }

        # ---------- Run parser ----------
        output_blob_name = generate_output_blob_name(req.output_blob_prefix)
        local_output_path = os.path.join(tmp_dir, output_blob_name)

        try:
            logger.info("Running parser...")
            # New extractor returns dict grouped by category
            result = extract_document_to_json(
                file_path=local_input_path,
                output_path=local_output_path,
            )
            logger.info("Parsing completed")
            if not result:
                raise RuntimeError("Parser returned no result")
        except Exception as e:
            logger.exception("Parser failed")
            return {
                "status": "Error",
                "message": f"Parser failed: {e}",
            }

        # Ensure output file exists
        if not os.path.exists(local_output_path):
            return {
                "status": "Error",
                "message": "Parser did not create output file",
            }

        # ---------- Upload output JSON ----------
        try:
            output_blob_client = blob_service.get_blob_client(
                container=req.output_container,
                blob=output_blob_name,
            )
            logger.info(
                f"Uploading output JSON to container={req.output_container}, "
                f"blob={output_blob_name}"
            )
            with open(local_output_path, "rb") as f:
                output_blob_client.upload_blob(
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"),
                )
            logger.info("Upload complete")
        except Exception as e:
            logger.exception("Output upload failed")
            return {
                "status": "Error",
                "message": f"Failed to upload output JSON: {e}",
            }

        # ---------- Generate SAS URL for the output blob (NO FALLBACK) ----------
        try:
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=req.output_container,
                blob_name=output_blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365),  # 1 year validity
            )

            if not sas_token:
                raise RuntimeError("generate_blob_sas returned empty token")

            output_url = f"{output_blob_client.url}?{sas_token}"
            logger.info("Generated SAS URL for output blob")
        except Exception as e:
            logger.exception("Failed to generate SAS URL")
            return {
                "status": "Error",
                "message": f"Failed to generate SAS URL: {e}",
            }

        # ---------- Post-process: question labels & counts ----------
        # result is expected to be: { "Screener": [...], "Main Survey": [...], ... }
        if isinstance(result, dict):
            questions_extracted = sum(
                len(v) for v in result.values() if isinstance(v, list)
            )
        else:
            questions_extracted = None

        questions_labels = (
            extract_questions_labels(result) if isinstance(result, dict) else []
        )

        # ---------- SUCCESS RESPONSE ----------
        return {
            "status": "success",
            "input_blob": req.input_blob_url,
            "output_blob": output_url,  # ALWAYS SAS URL on success
            "questions_extracted": questions_extracted,
            "questions_labels": questions_labels,
        }

    except Exception as e:
        logger.exception("Unexpected error during processing")
        return {
            "status": "Error",
            "message": f"Unexpected error: {e}",
        }

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Temporary folder cleaned up")


# ============== Health Check Endpoint ==============


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="document-parser-audience-generator",
        version="1.0.0",
    )


# ============== Audience Generation Helpers ==============


def process_audience_generation_background(
    task_id: str,
    req: GenerateAudienceRequest,
    conn_str: str,
    account_name: str,
    account_key: str,
) -> None:
    """
    Background task to process audience generation.

    Args:
        task_id: Unique task identifier
        req: The original request
        conn_str: Azure storage connection string
        account_name: Azure storage account name
        account_key: Azure storage account key
    """
    start_time = time.time()
    input_data = req.input_data
    tmp_dir = None

    try:
        # Create BlobServiceClient
        blob_service = BlobServiceClient.from_connection_string(conn_str)

        # Create a temp dir (unique per request)
        tmp_dir = tempfile.mkdtemp(prefix="audience_gen_")

        # Initialize Azure OpenAI client
        try:
            client, deployment = _create_azure_client()
        except ValueError as e:
            logger.error(f"Task {task_id}: Azure OpenAI not configured: {e}")
            return

        # Get audiences from input data
        audiences = input_data.get("audiences", [])
        if not audiences:
            logger.error(f"Task {task_id}: No audiences found in input data")
            return

        # Run generation
        normalized_audiences = [
            {
                "persona": aud.get("persona", {}),
                "screenerQuestions": aud.get("screenerQuestions", []),
                "sampleSize": aud.get("sampleSize", 1),
            }
            for aud in audiences
        ]

        async def run_generation():
            tasks = [
                generate_audience_characteristics(
                    client=client,
                    deployment=deployment,
                    audience_data=aud,
                    audience_index=idx,
                    max_concurrent=req.max_concurrent,
                )
                for idx, aud in enumerate(normalized_audiences)
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_generation())
        # Each result is a tuple of (audience_dict, progress_tracker)
        enriched_audiences = [result[0] for result in results]

        total_generated = sum(
            aud["metadata"]["generation_stats"]["total_generated"]
            for aud in enriched_audiences
        )
        total_failed = sum(
            aud["metadata"]["generation_stats"]["total_failed"] for aud in enriched_audiences
        )

        processing_time = time.time() - start_time

        # Build output response
        project_id = input_data.get("projectId")
        user_id = input_data.get("userId")
        json_file_url = input_data.get("json_file_url")
        output_data = {
            "project_name": input_data.get("projectName"),
            "project_description": input_data.get("projectDescription"),
            "project_id": str(project_id) if project_id is not None else None,
            "user_id": str(user_id) if user_id is not None else None,
            "request_id": input_data.get("requestId"),
            "json_file_url": json_file_url,
            "generation_model": f"azure:{deployment}",
            "provider": "azure",
            "total_audiences": len(enriched_audiences),
            "total_members_processed": total_generated + total_failed,
            "total_successfully_generated": total_generated,
            "total_failed": total_failed,
            "processing_time_seconds": round(processing_time, 2),
            "audiences": enriched_audiences,
        }

        # Write output to temp file
        output_blob_name = generate_output_blob_name(req.output_blob_prefix)
        local_output_path = os.path.join(tmp_dir, output_blob_name)

        with open(local_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Task {task_id}: Generation completed, output written to temp file"
        )

        # Ensure output container exists
        try:
            container_client = blob_service.get_container_client(
                AUDIENCE_OUTPUT_CONTAINER
            )
            try:
                container_client.get_container_properties()
            except Exception:
                container_client.create_container()
                logger.info(f"Created container '{AUDIENCE_OUTPUT_CONTAINER}'")
        except Exception as e:
            logger.error(f"Task {task_id}: Failed to ensure container exists: {e}")
            return

        # Upload output JSON
        try:
            output_blob_client = blob_service.get_blob_client(
                container=AUDIENCE_OUTPUT_CONTAINER,
                blob=output_blob_name,
            )
            with open(local_output_path, "rb") as f:
                output_blob_client.upload_blob(
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"),
                )
            logger.info(f"Task {task_id}: Upload complete")
        except Exception as e:
            logger.error(f"Task {task_id}: Output upload failed: {e}")
            return

        # Generate SAS URL for the output blob
        try:
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=AUDIENCE_OUTPUT_CONTAINER,
                blob_name=output_blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365),
            )

            if not sas_token:
                raise RuntimeError("generate_blob_sas returned empty token")

            output_url = f"{output_blob_client.url}?{sas_token}"
            logger.info(f"Task {task_id}: Generated SAS URL for output blob")
        except Exception as e:
            logger.error(f"Task {task_id}: Failed to generate SAS URL: {e}")
            return

        # Print completion summary
        logger.info(
            f"\n{'='*60}\n"
            f"TASK COMPLETED SUCCESSFULLY\n"
            f"{'='*60}\n"
            f"task_id: {task_id}\n"
            f"output_blob: {output_url}\n"
            f"json_file_url: {json_file_url}\n"
            f"project_id: {project_id}\n"
            f"{'='*60}"
        )

    except Exception as e:
        logger.exception(f"Task {task_id}: Unexpected error during processing: {e}")

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Task {task_id}: Temporary folder cleaned up")


# ============== Audience Generation Endpoint ==============


@app.post(
    "/generate_audience/",
    response_model=GenerateAudienceAsyncResponse,
    tags=["Audience Generation"],
    summary="Generate audience characteristics from JSON data (async)",
    description="Accepts JSON data, returns task_id immediately, and processes in background.",
)
@app.post(
    "/generate_audience",
    response_model=GenerateAudienceAsyncResponse,
    tags=["Audience Generation"],
    include_in_schema=False,  # Hide duplicate from docs
)
async def generate_audience(
    req: GenerateAudienceRequest,
    background_tasks: BackgroundTasks,
) -> GenerateAudienceAsyncResponse:
    """
    Generate audience characteristics from inline JSON data (async).

    - Accepts JSON data directly in request body (must include projectId and userId)
    - Returns task_id and status immediately
    - Processes audiences in background using Azure OpenAI
    - Uploads output to 'generated-synthetic-audience' container
    """
    task_id = str(uuid4())
    logger.info(f"Received audience generation request, task_id: {task_id}")

    # Get connection string from env
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AZURE_STORAGE_CONNECTION_STRING not set",
        )

    # Parse account info (needed for SAS)
    account_name, account_key = parse_account_from_connection_string(conn_str)
    if not account_name or not account_key:
        logger.error(
            "AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING. "
            "SAS URL generation requires a connection string with AccountKey."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING (cannot generate SAS).",
        )

    # Use input_data directly from request
    input_data = req.input_data

    # Validate required structure
    audiences = input_data.get("audiences", [])
    if not audiences:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input JSON must contain 'audiences' array with at least one audience",
        )

    # Validate projectId is provided (required for update-personas API)
    project_id = input_data.get("projectId")
    if project_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input JSON must contain 'projectId' for update-personas API callback",
        )

    # Schedule background task
    background_tasks.add_task(
        process_audience_generation_background,
        task_id=task_id,
        req=req,
        conn_str=conn_str,
        account_name=account_name,
        account_key=account_key,
    )

    logger.info(f"Task {task_id}: Scheduled for background processing")

    # Return immediately with task_id and status
    return GenerateAudienceAsyncResponse(
        task_id=task_id,
        status="processing",
    )
