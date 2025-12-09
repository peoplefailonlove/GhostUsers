import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# =========================
# Load environment variables
# =========================

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_5 = os.getenv("AZURE_OPENAI_DEPLOYMENT_5", "gpt-5")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT_5:
    raise RuntimeError(
        "Missing Azure OpenAI config. Please set AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_5 in your environment or .env file."
    )

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# =========================
# System prompt
# =========================

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


# =========================
# Utility
# =========================

def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    """Split a list into chunks of at most `size` elements."""
    return [items[i:i + size] for i in range(0, len(items), size)]


# =========================
# Questions helpers
# =========================

def load_questionnaire(path: str) -> Dict[str, Any]:
    """
    Load the full questionnaire JSON.

    Expected shape:
    {
      "Screener": [...],
      "Functional Profiling & Needs": [...],
      "Concept Test & Value Story": [...],
      ...
    }
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_non_screener_questions(questionnaire: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract a flat list of questions from all segments EXCEPT 'Screener'.

    Each question normalized to:
    {
      "id": ...,
      "text": ...,
      "type": ...,
      "options": [...],
      "required": bool,
      "notes": [...],
      "logic_validations": {...},
      "category": "<segment name>"
    }
    """
    questions: List[Dict[str, Any]] = []

    for section_name, section_questions in questionnaire.items():
        if section_name.strip().lower() == "screener":
            continue

        if not isinstance(section_questions, list):
            continue

        for q in section_questions:
            questions.append(
                {
                    "id": q.get("id"),
                    "text": q.get("text"),
                    "type": q.get("type"),
                    "options": q.get("options", []),
                    "required": q.get("required", False),
                    "notes": q.get("instructions", []),
                    "logic_validations": q.get("conditions", {}),
                    "category": q.get("category", section_name),
                }
            )

    return questions


# =========================
# Audience / persona helpers
# =========================

def load_project_json(path: str) -> Dict[str, Any]:
    """
    Load the project JSON that contains 'audiences' and 'generated_audience'.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_personas_from_generated_audience(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten all members from all audiences into a list of persona dicts:

    {
      "member_id": "<member_id>",
      "audience_index": <int>,
      "persona_json": {
         "member_id": ...,
         "name": ...,
         "about": ...,
         "goals_and_motivations": [...],
         "frustrations": [...],
         "need_state": ...,
         "occasions": ...
      }
    }
    """
    personas: List[Dict[str, Any]] = []

    for aud_idx, audience in enumerate(data.get("audiences", [])):
        meta = audience.get("metadata", {})
        audience_index = meta.get("audience_index", aud_idx)

        for member in audience.get("generated_audience", []):
            persona_json = {
                "member_id": member.get("member_id"),
                "name": member.get("name"),
                "about": member.get("about", ""),
                "goals_and_motivations": member.get("goals_and_motivations", []),
                "frustrations": member.get("frustrations", []),
                "need_state": member.get("need_state", ""),
                "occasions": member.get("occasions", ""),
            }

            personas.append(
                {
                    "member_id": member.get("member_id"),
                    "audience_index": audience_index,
                    "persona_json": persona_json,
                }
            )

    return personas


# =========================
# LLM call helpers
# =========================

def extract_json_array(text: str) -> str:
    """
    Try to extract the first top-level JSON array from the text.
    This is a fallback in case the model wraps JSON with extra text/markdown.
    """
    if not text:
        return ""

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start : end + 1]


def call_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Call Azure OpenAI chat completions and parse the JSON array of answers.
    """
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_5,
        messages=messages,
    )

    content = resp.choices[0].message.content

    if content is None or not str(content).strip():
        raise ValueError(
            "Empty or null content returned from LLM.\n"
            f"Full response object:\n{resp}"
        )

    # Try direct JSON parse first
    try:
        answers = json.loads(content)
        if isinstance(answers, list):
            return answers
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract array from the text
    maybe_json = extract_json_array(content)
    if maybe_json:
        try:
            answers = json.loads(maybe_json)
            if isinstance(answers, list):
                return answers
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "LLM did not return valid JSON.\n"
        f"Raw content:\n{content}\n\n"
        f"Full response object:\n{resp}"
    )


def build_messages(
    persona_json: Dict[str, Any], questions_batch: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build the messages payload for Azure OpenAI.

    User content is a JSON object:
      {
        "persona_json": {...},
        "questions_json": [...]
      }
    """
    user_payload = {
        "persona_json": persona_json,
        "questions_json": questions_batch,
    }

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]


# ADDED FOR MVP: start - alignment check helper
def _check_alignment_with_llm(
    persona_json: Dict[str, Any],
    answers_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Check if survey answers align with persona traits using LLM.
    Returns dict with 'aligned' (bool), 'score' (0.0-1.0), and 'reasons' (list of strings).
    """
    # Extract key traits - include about for richer context
    traits = {
        "about": persona_json.get("about", ""),
        "goals_and_motivations": persona_json.get("goals_and_motivations", []),
        "frustrations": persona_json.get("frustrations", []),
        "need_state": persona_json.get("need_state", ""),
        "occasions": persona_json.get("occasions", ""),
    }
    
    # If no meaningful traits, assume aligned
    if not traits["about"] and not any([traits["goals_and_motivations"], traits["frustrations"], traits["need_state"]]):
        return {"aligned": True, "score": 1.0, "reasons": []}
    
    # Build answers summary - use more answers for better analysis
    answers_text = json.dumps(answers_list[:50], ensure_ascii=False)
    
    prompt = f"""Analyze if these survey answers align with the persona's unique traits and characteristics.

Persona Profile:
- About: {traits['about']}
- Goals & Motivations: {traits['goals_and_motivations']}
- Frustrations: {traits['frustrations']}
- Need State: {traits['need_state']}
- Occasions: {traits['occasions']}

Survey answers:
{answers_text}

Analyze the alignment between the persona's unique characteristics and their survey responses.
Consider:
1. Do the answers reflect the persona's stated goals and motivations?
2. Are the frustrations and pain points consistent with their responses?
3. Does the overall tone match their need state?
4. Are there any contradictions between who they are and how they answered?

Return ONLY a JSON object with:
- "aligned": true/false (true if mostly aligned, false if significant misalignments)
- "score": a number 0.0-1.0 representing alignment degree (1.0 = perfect alignment)
- "reasons": list of specific observations about alignment or misalignment

Example: {{"aligned": true, "score": 0.85, "reasons": ["Answers reflect entrepreneurial mindset", "Response to Q5 slightly inconsistent with stated frustration"]}}"""

    try:
        messages = [
            {"role": "system", "content": "You are an expert alignment analyzer. Evaluate how well survey responses match persona characteristics. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_5,
            messages=messages,
        )
        content = resp.choices[0].message.content
        if content:
            # Try to extract JSON from response
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                import re
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise
            
            return {
                "aligned": result.get("aligned", True),
                "score": float(result.get("score", 1.0 if result.get("aligned", True) else 0.5)),
                "reasons": result.get("reasons", []),
            }
    except Exception as e:
        logger.warning(f"Alignment check failed: {e}")
    
    return {"aligned": True, "score": 1.0, "reasons": ["Alignment check could not be performed"]}
# ADDED FOR MVP: end - alignment check helper


# =========================
# Core: run survey for one persona
# =========================

def answer_questions_for_persona(
    member_id: str,
    audience_index: int,
    persona_json: Dict[str, Any],
    questions: List[Dict[str, Any]],
    batch_size: int = 30,
) -> Dict[str, Any]:
    """
    For a single audience member, run through all questions (batched) and collect answers.
    """
    all_answers: Dict[str, Any] = {}

    batches = chunk_list(questions, batch_size)
    logger.info(
        "Answering survey for member %s (audience %s) in %d batches",
        member_id,
        audience_index,
        len(batches),
    )

    for q_batch in batches:
        messages = build_messages(persona_json, q_batch)
        batch_answers = call_llm(messages)

        for ans in batch_answers:
            qid = ans.get("id")
            if qid is None:
                continue
            all_answers[qid] = ans.get("answer")

    answers_list = [
        {"question_id": q["id"], "value": all_answers.get(q["id"])}
        for q in questions
    ]

    # ADDED FOR MVP: start - alignment check using LLM
    alignment = _check_alignment_with_llm(persona_json, answers_list)
    # ADDED FOR MVP: end

    return {
        "member_id": member_id,
        "audience_index": audience_index,
        "answers": answers_list,
        "alignment": alignment,  # ADDED FOR MVP
    }


# =========================
# Run for all personas
# =========================

def run_survey_for_all_members(
    questionnaire_path: str,
    project_path: str,
    batch_size: int = 30,
) -> List[Dict[str, Any]]:
    """
    End-to-end:

    - Load questionnaire JSON
    - Extract non-screener questions
    - Load project JSON (audiences + generated_audience)
    - Extract personas (one per member)
    - For each member, call LLM and collect answers

    Returns:
        List of member-level answer dicts.
    """
    questionnaire = load_questionnaire(questionnaire_path)
    questions = extract_non_screener_questions(questionnaire)
    logger.info("Loaded %d non-screener questions", len(questions))

    project_data = load_project_json(project_path)
    personas = extract_personas_from_generated_audience(project_data)
    logger.info("Loaded %d personas", len(personas))

    results: List[Dict[str, Any]] = []

    for p in personas:
        result = answer_questions_for_persona(
            member_id=p["member_id"],
            audience_index=p["audience_index"],
            persona_json=p["persona_json"],
            questions=questions,
            batch_size=batch_size,
        )
        results.append(result)

    return results
