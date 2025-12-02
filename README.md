# Document Parser & Audience Generator API

A FastAPI service that provides two main functionalities:

1.**Document Parsing**: Downloads a document from Azure Blob Storage, extracts survey questions using Azure OpenAI, and uploads the resulting JSON back to a blob container.

2.**Synthetic Audience Generation**: Generates detailed audience member characteristics using persona templates and screener questions via Azure OpenAI.

## Setup

### Install dependencies:

pip install -r requirements.txt

## Environment Variables

AZURE_STORAGE_CONNECTION_STRING=

AZURE_OPENAI_API_KEY=

AZURE_OPENAI_ENDPOINT=

AZURE_OPENAI_DEPLOYMENT=gpt-4o

OPENAI_API_VERSION=2025-01-01-preview

LOG_LEVEL=INFO

## Run the **API**

### Start the server (development):

### Example Request

**POST** /process

Request body (**JSON**):

{

*input_blob_url*: "[https://`<account>`.blob.core.windows.net/samplefiles/survey.docx&#34;,](https://<account>.blob.core.windows.net/samplefiles/survey.docx*,)

*output_container*: *json-output-container*,

*output_blob_prefix*: *survey_extracted*

}

### Example Response

{

*status*: *success*,

*input_blob*: *[https://`<account>`.blob.core.windows.net/samplefiles/survey.docx&#34;,](https://<account>.blob.core.windows.net/samplefiles/survey.docx*,)

*output_blob*: *[https://`<account>`.blob.core.windows.net/json-output-container/survey_extracted_20251119T073453Z_ab12cd34.json&#34;,](https://<account>.blob.core.windows.net/json-output-container/survey_extracted_20251119T073453Z_ab12cd34.json*,)

    *questions_extracted":**148**

}

## Quick test

**POST** [http://localhost:**8000**/process](http://localhost:**8000**/process) Content-Type: application/json

{

*input_blob_url*: "[https://`<account>`.blob.core.windows.net/samplefiles/survey.docx&#34;,](https://<account>.blob.core.windows.net/samplefiles/survey.docx*,)

*output_container*: *json-output-container*,

*output_blob_prefix*: *survey_extracted"

}

---

## Endpoint 2: Audience Generation

### POST /generate_audience

Generate synthetic audience characteristics from a JSON file containing persona templates and screener questions.

**Request body (JSON):**

```json

{

"input_blob_url": "https://<account>.blob.core.windows.net/container/personas_input.json",

"output_blob_prefix": "audience_output",

"max_concurrent": 10

}

```

**Input JSON structure:**

```json

{

"projectName": "My Project",

"projectDescription": "Project description",

"projectId": "123",

"userId": "456",

"requestId": "req-789",

"audiences": [

        {

"persona": {

"personaName": "Tech Enthusiast",

"about": "A tech-savvy individual...",

"goalsAndMotivations": "Stay updated with latest tech...",

"frustrations": "Information overload...",

"needState": "Seeking reliable sources...",

"occasions": "During commute, lunch breaks..."

            },

"screenerQuestions": [

                {"question": "What is your age?", "answer": "25-34"},

                {"question": "What is your occupation?", "answer": "Software Engineer"}

            ],

"sampleSize": 5

        }

    ]

}

```

**Response:**

```json

{

"status": "success",

"input_blob": "https://<account>.blob.core.windows.net/container/personas_input.json",

"output_blob": "https://<account>.blob.core.windows.net/generated-synthetic-audience/audience_output_20251202T120000Z_ab12cd34.json?<sas_token>",

"total_audiences": 1,

"total_members_processed": 5,

"total_successfully_generated": 5,

"total_failed": 0,

"processing_time_seconds": 12.34

}

```

---

## Endpoint 3: Health Check

### GET /health

Returns the health status of the service.

**Response:**

```json

{

"status": "healthy",

"service": "document-parser-audience-generator",

"version": "1.0.0"

}

```

---

## API Endpoints Summary

| Endpoint | Method | Description |

|----------|--------|-------------|

| `/process` | POST | Extract survey questions from DOCX/PDF documents |

| `/generate_audience` | POST | Generate synthetic audience characteristics |

| `/health` | GET | Health check endpoint |
