"""
Test script to explore Azure OpenAI rate limits and metrics monitoring.
Uses LangChain with Azure OpenAI.

Run: python test_rate_limits.py
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback

load_dotenv()

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_5", "gpt-5")

# Setup LangChain Azure OpenAI client
llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    max_tokens=50,
)


def test_single_call_with_usage():
    """Test a single call and inspect usage stats using LangChain callback."""
    print("\n" + "=" * 60)
    print("TEST 1: Single Call - Token Usage (LangChain)")
    print("=" * 60)

    # Use get_openai_callback to track token usage
    with get_openai_callback() as cb:
        response = llm.invoke([HumanMessage(content="Say hello in 5 words.")])

    print(f"\nResponse: {response.content}")
    print(f"\n📊 Token Usage (via callback):")
    print(f"   Prompt tokens:     {cb.prompt_tokens}")
    print(f"   Completion tokens: {cb.completion_tokens}")
    print(f"   Total tokens:      {cb.total_tokens}")
    print(f"   Total cost (USD):  ${cb.total_cost:.6f}")

    return response


def test_rate_limit_headers():
    """
    Test to check rate limit headers.
    
    LangChain doesn't expose raw headers directly, so we use the underlying
    OpenAI client to access them.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Rate Limit Headers")
    print("=" * 60)

    # Access the underlying OpenAI client from LangChain
    from openai import AzureOpenAI
    
    raw_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )

    # Use with_raw_response to get headers
    raw_response = raw_client.chat.completions.with_raw_response.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": "Hi"}],
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
    """Make multiple calls and track cumulative usage with LangChain callback."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Calls - Cumulative Tracking (LangChain)")
    print("=" * 60)

    call_count = 5
    print(f"\nMaking {call_count} API calls...")

    # Track all calls within a single callback context
    with get_openai_callback() as cb:
        for i in range(call_count):
            start = time.time()
            response = llm.invoke([HumanMessage(content=f"Count to {i + 1}")])
            elapsed = time.time() - start
            print(f"   Call {i + 1}: {elapsed:.2f}s - {response.content[:50]}...")

    print(f"\n📊 Cumulative Usage (all {call_count} calls):")
    print(f"   Total prompt tokens:     {cb.prompt_tokens}")
    print(f"   Total completion tokens: {cb.completion_tokens}")
    print(f"   Total tokens:            {cb.total_tokens}")
    print(f"   Total cost (USD):        ${cb.total_cost:.6f}")
    print(f"   Successful requests:     {cb.successful_requests}")


def test_429_behavior():
    """
    Attempt to trigger rate limiting by making rapid calls using LangChain.
    
    WARNING: This may consume quota quickly!
    """
    print("\n" + "=" * 60)
    print("TEST 4: Rate Limit Behavior (429 Detection - LangChain)")
    print("=" * 60)

    print("\n⚠️  This test makes rapid API calls to observe rate limiting.")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Skipped.")
        return

    success_count = 0
    rate_limited_count = 0
    error_count = 0
    max_calls = 20

    print(f"\nMaking {max_calls} rapid calls...")

    # Create a minimal LLM for fast calls
    fast_llm = AzureChatOpenAI(
        azure_deployment=DEPLOYMENT,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        max_tokens=1,
    )

    for i in range(max_calls):
        try:
            with get_openai_callback() as cb:
                response = fast_llm.invoke([HumanMessage(content="Hi")])
            print(f"   Call {i + 1}: ✅ OK | Tokens used: {cb.total_tokens}")
            success_count += 1

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                rate_limited_count += 1
                print(f"   Call {i + 1}: ⛔ RATE LIMITED")
            else:
                error_count += 1
                print(f"   Call {i + 1}: ❌ Error: {error_str[:80]}")

    print(f"\n📊 Results:")
    print(f"   Successful calls:   {success_count}")
    print(f"   Rate limited calls: {rate_limited_count}")
    print(f"   Other errors:       {error_count}")


def main():
    print("🔍 Azure OpenAI Rate Limit & Metrics Test (LangChain)")
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
