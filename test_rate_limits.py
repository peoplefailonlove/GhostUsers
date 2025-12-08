"""
Test script to explore Azure OpenAI rate limits and metrics monitoring.

Run: python test_rate_limits.py
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Setup client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_5", "gpt-5")


def test_single_call_with_usage():
    """Test a single call and inspect usage stats."""
    print("\n" + "=" * 60)
    print("TEST 1: Single Call - Token Usage")
    print("=" * 60)

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "user", "content": "Say hello in 5 words."}
        ],
        max_completion_tokens=20,
    )

    # Token usage from response
    usage = response.usage
    print(f"\nResponse: {response.choices[0].message.content}")
    print(f"\n📊 Token Usage:")
    print(f"   Prompt tokens:     {usage.prompt_tokens}")
    print(f"   Completion tokens: {usage.completion_tokens}")
    print(f"   Total tokens:      {usage.total_tokens}")

    return response


def test_rate_limit_headers():
    """
    Test to check rate limit headers.
    
    Note: The OpenAI Python SDK wraps responses, so we need to use
    the `with_raw_response` method to access headers.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Rate Limit Headers")
    print("=" * 60)

    # Use with_raw_response to get headers
    raw_response = client.chat.completions.with_raw_response.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "user", "content": "Hi"}
        ],
        max_completion_tokens=5,
    )

    # Access headers
    headers = raw_response.headers
    print(f"\n📋 Response Headers:")
    
    # Common rate limit headers
    rate_limit_headers = [
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "retry-after",
        "x-ms-region",
    ]

    for header in rate_limit_headers:
        value = headers.get(header)
        if value:
            print(f"   {header}: {value}")

    # Also print all headers for discovery
    print(f"\n📋 All Headers:")
    for key, value in headers.items():
        print(f"   {key}: {value}")

    # Parse the actual response
    response = raw_response.parse()
    print(f"\nResponse: {response.choices[0].message.content}")

    return headers


def test_multiple_calls_tracking():
    """Make multiple calls and track cumulative usage."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Calls - Cumulative Tracking")
    print("=" * 60)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    call_count = 5

    print(f"\nMaking {call_count} API calls...")

    for i in range(call_count):
        start = time.time()
        
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "user", "content": f"Count to {i + 1}"}
            ],
            max_completion_tokens=50,
        )
        
        elapsed = time.time() - start
        usage = response.usage
        
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens

        print(f"   Call {i + 1}: {usage.total_tokens} tokens, {elapsed:.2f}s")

    print(f"\n📊 Cumulative Usage:")
    print(f"   Total prompt tokens:     {total_prompt_tokens}")
    print(f"   Total completion tokens: {total_completion_tokens}")
    print(f"   Grand total:             {total_prompt_tokens + total_completion_tokens}")


def test_429_behavior():
    """
    Attempt to trigger rate limiting by making rapid calls.
    
    WARNING: This may consume quota quickly!
    """
    print("\n" + "=" * 60)
    print("TEST 4: Rate Limit Behavior (429 Detection)")
    print("=" * 60)

    print("\n⚠️  This test makes rapid API calls to observe rate limiting.")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Skipped.")
        return

    success_count = 0
    rate_limited_count = 0
    max_calls = 20

    print(f"\nMaking {max_calls} rapid calls...")

    for i in range(max_calls):
        try:
            raw_response = client.chat.completions.with_raw_response.create(
                model=DEPLOYMENT,
                messages=[{"role": "user", "content": "Hi"}],
                max_completion_tokens=1,
            )
            
            remaining_requests = raw_response.headers.get("x-ratelimit-remaining-requests", "?")
            remaining_tokens = raw_response.headers.get("x-ratelimit-remaining-tokens", "?")
            
            print(f"   Call {i + 1}: ✅ OK | Remaining: {remaining_requests} req, {remaining_tokens} tokens")
            success_count += 1

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                rate_limited_count += 1
                print(f"   Call {i + 1}: ⛔ RATE LIMITED")
                
                # Check for retry-after
                if hasattr(e, 'response') and e.response:
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        print(f"            Retry after: {retry_after}s")
            else:
                print(f"   Call {i + 1}: ❌ Error: {error_str[:100]}")

    print(f"\n📊 Results:")
    print(f"   Successful calls:   {success_count}")
    print(f"   Rate limited calls: {rate_limited_count}")


def main():
    print("🔍 Azure OpenAI Rate Limit & Metrics Test")
    print(f"   Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"   Deployment: {DEPLOYMENT}")
    print(f"   Time: {datetime.now().isoformat()}")

    # Run tests
    test_single_call_with_usage()
    test_rate_limit_headers()
    test_multiple_calls_tracking()
    test_429_behavior()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
