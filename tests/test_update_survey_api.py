"""
Test script for UPDATE_SURVEY_API_URL endpoint.

Usage:
    python test_update_survey_api.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

UPDATE_SURVEY_API_URL = os.getenv(
    "UPDATE_SURVEY_API_URL",
    "https://sample-agument-middleware-dev.azurewebsites.net/sample-enrichment/api/projects/update-survey-file-url",
)


def test_update_survey_api():
    """Test the update-survey-file-url API with sample data."""
    
    payload = {
        "projectId": 10,
        "taskId": "ABD123",
        "surveyFileUrl": "https://storage/doc1.json",
        "status": "Survey File Generated",
    }

    print(f"Testing UPDATE_SURVEY_API_URL: {UPDATE_SURVEY_API_URL}")
    print(f"Payload: {payload}")
    print("-" * 50)

    try:
        response = requests.post(
            UPDATE_SURVEY_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        print("-" * 50)

        if response.status_code == 200:
            data = response.json()
            api_status = data.get("status", "").lower()
            if api_status == "success":
                print("✅ API call SUCCESS")
            else:
                print(f"❌ API returned non-success status: {data}")
        else:
            print(f"❌ HTTP error: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response body: {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    test_update_survey_api()
