#!/bin/bash

# Test LLM Judge with curl commands
# Set your AZURE_OPENAI_KEY environment variable first:
# export AZURE_OPENAI_KEY="your-api-key-here"

echo "=== Testing LLM Judge ==="
echo

# Test 1: Good alignment (should pass)
echo "Test 1: Good alignment (should pass with high scores)"
curl -X POST "https://syncbillopenai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-12-01-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: $AZURE_OPENAI_KEY" \
  -d '{
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "messages": [
      {
        "role": "user",
        "content": "IGNORE name, age, gender, location, ethnicity.\n\nPARENT PERSONA:\nAbout: Creative professional who loves innovation and technology\nGoals: To scale business operations; To learn emerging trends; To create meaningful impact\nFrustrations: Managing limited resources; Meeting tight deadlines; Limited access to premium tools\nNeed State: Driven and resourceful, seeking growth opportunities\nOccasions: Engages with content during morning planning and evening wind-down\n\nSCREENER ANSWERS (must match 100%):\nQ: What is your primary role?\nA: Senior Marketing Manager\nQ: How do you prefer to spend your weekends?\nA: Hiking and outdoor activities\n\nGENERATED PERSONA:\nAbout: A senior marketing manager at a tech startup who combines creative strategy with analytical thinking, passionate about innovative campaigns and team leadership\nGoals: To scale marketing operations while maintaining quality; To continuously learn emerging digital marketing trends; To create lasting impact through meaningful brand storytelling\nFrustrations: Managing marketing workflows with limited team resources; Maintaining quality standards under tight campaign deadlines; Limited access to premium marketing analytics tools\nNeed State: Driven and resourceful, seeking growth opportunities in the competitive tech marketing landscape\nOccasions: Engages with content during morning planning sessions and evening wind-down after campaign launches\n\nReturn only JSON:\n{\n  \"screener_score\": 0.0-1.0,\n  \"parent_score\": 0.0-1.0,\n  \"passed\": true/false,\n  \"explanation\": \"short reason\"\n}\nBe very strict. Pass only if both scores \u003e= 0.88."
      }
    ]
  }' | jq .

echo
echo "----------------------------------------"
echo

# Test 2: Poor alignment (should fail)
echo "Test 2: Poor alignment (should fail with low scores)"
curl -X POST "https://syncbillopenai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-12-01-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: $AZURE_OPENAI_KEY" \
  -d '{
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "messages": [
      {
        "role": "user",
        "content": "IGNORE name, age, gender, location, ethnicity.\n\nPARENT PERSONA:\nAbout: Creative professional who loves innovation and technology\nGoals: To scale business operations; To learn emerging trends; To create meaningful impact\nFrustrations: Managing limited resources; Meeting tight deadlines; Limited access to premium tools\nNeed State: Driven and resourceful, seeking growth opportunities\nOccasions: Engages with content during morning planning and evening wind-down\n\nSCREENER ANSWERS (must match 100%):\nQ: What is your primary role?\nA: Senior Marketing Manager\nQ: How do you prefer to spend your weekends?\nA: Hiking and outdoor activities\n\nGENERATED PERSONA:\nAbout: A retired teacher who enjoys gardening and reading books\nGoals: To maintain a peaceful garden; To read classic literature; To spend time with grandchildren\nFrustrations: Bad weather affecting plants; Finding new books to read; Health issues limiting activities\nNeed State: Content and relaxed, enjoying retirement\nOccasions: Engages with content during quiet afternoons with tea\n\nReturn only JSON:\n{\n  \"screener_score\": 0.0-1.0,\n  \"parent_score\": 0.0-1.0,\n  \"passed\": true/false,\n  \"explanation\": \"short reason\"\n}\nBe very strict. Pass only if both scores \u003e= 0.88."
      }
    ]
  }' | jq .

echo
echo "----------------------------------------"
echo

# Test 3: Mixed alignment (edge case)
echo "Test 3: Mixed alignment (edge case - might pass one but not both)"
curl -X POST "https://syncbillopenai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-12-01-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: $AZURE_OPENAI_KEY" \
  -d '{
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "messages": [
      {
        "role": "user",
        "content": "IGNORE name, age, gender, location, ethnicity.\n\nPARENT PERSONA:\nAbout: Creative professional who loves innovation and technology\nGoals: To scale business operations; To learn emerging trends; To create meaningful impact\nFrustrations: Managing limited resources; Meeting tight deadlines; Limited access to premium tools\nNeed State: Driven and resourceful, seeking growth opportunities\nOccasions: Engages with content during morning planning and evening wind-down\n\nSCREENER ANSWERS (must match 100%):\nQ: What is your primary role?\nA: Senior Marketing Manager\nQ: How do you prefer to spend your weekends?\nA: Hiking and outdoor activities\n\nGENERATED PERSONA:\nAbout: A marketing professional focused on creative campaigns and digital innovation\nGoals: To scale marketing operations; To learn emerging digital trends; To create impact through storytelling\nFrustrations: Limited budget constraints; Tight project deadlines; Outdated marketing tools\nNeed State: Ambitious and growth-oriented\nOccasions: Checks marketing content during work hours\n\nReturn only JSON:\n{\n  \"screener_score\": 0.0-1.0,\n  \"parent_score\": 0.0-1.0,\n  \"passed\": true/false,\n  \"explanation\": \"short reason\"\n}\nBe very strict. Pass only if both scores \u003e= 0.88."
      }
    ]
  }' | jq .

echo
echo "=== Test completed ==="
echo
echo "Expected results:"
echo "Test 1: Both scores >= 0.88 (PASS)"
echo "Test 2: Both scores < 0.88 (FAIL - poor alignment)"
echo "Test 3: One or both scores around 0.85-0.90 (EDGE CASE)"
