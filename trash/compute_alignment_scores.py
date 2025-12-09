"""
Compute alignment scores (screener_score and parent_score) for generated personas.

Usage:
    python compute_alignment_scores.py \
        --audience-file data/generate_persona.json \
        --answers-file data/survey_output.json \
        --questions-file data/questions_extracted.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

# Thresholds (for dot product - OpenAI embeddings are normalized, so range is similar to cosine)
SCREENER_THRESHOLD = 0.45
PARENT_THRESHOLD = 0.50


def make_azure_client() -> AzureOpenAI:
    """Create Azure OpenAI client from environment variables."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint or not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY")
    
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


def embed_text(client: AzureOpenAI, text: str) -> np.ndarray:
    """Get embedding for text using Azure OpenAI."""
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    response = client.embeddings.create(model=deployment, input=text)
    return np.array(response.data[0].embedding)


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product between two vectors."""
    return float(np.dot(a, b))


def load_json_from_url(url: str) -> Dict:
    """Download and parse JSON from URL."""
    print(f"Downloading: {url[:80]}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def load_json_from_file(path: str) -> Dict:
    """Load JSON from local file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_parent_text(persona: Dict, screener_questions: List[Dict] = None) -> str:
    """Extract text from parent persona including screener Q&A."""
    fields = [
        persona.get("about", ""),
        persona.get("goalsAndMotivations", ""),
        persona.get("frustrations", ""),
        persona.get("needState", ""),
        persona.get("occasions", ""),
    ]
    text = "\n".join([f for f in fields if f])
    
    # Add screener questions if provided
    if screener_questions:
        qa_pairs = []
        for sq in screener_questions:
            q = sq.get("question", "")
            a = sq.get("answer", [])
            if isinstance(a, list):
                a = "; ".join(a)
            if q and a:
                qa_pairs.append(f"Q: {q}\nA: {a}")
        if qa_pairs:
            text += "\n\n" + "\n\n".join(qa_pairs)
    
    return text


def extract_generated_text(member: Dict) -> str:
    """Extract text from generated member."""
    goals = member.get("goals_and_motivations", [])
    if isinstance(goals, list):
        goals = "; ".join(goals)
    
    frustrations = member.get("frustrations", [])
    if isinstance(frustrations, list):
        frustrations = "; ".join(frustrations)
    
    fields = [
        member.get("about", ""),
        goals,
        frustrations,
        member.get("need_state", ""),
        member.get("occasions", ""),
    ]
    return "\n".join([f for f in fields if f])


def extract_screener_text(answers: List[Dict], questions_lookup: Dict) -> str:
    """Extract Q&A text from screener answers using questions lookup."""
    qa_pairs = []
    for ans in answers:
        qid = ans.get("question_id", "")
        value = ans.get("value")
        
        # Skip null values
        if value is None:
            continue
        
        # Get question text from lookup
        q_text = questions_lookup.get(qid, {}).get("text", qid)
        
        # Format answer value
        if isinstance(value, list):
            # Look up option texts
            options = questions_lookup.get(qid, {}).get("options", [])
            opt_map = {o.get("value"): o.get("text", o.get("value")) for o in options}
            a_text = "; ".join([opt_map.get(v, str(v)) for v in value])
        elif isinstance(value, dict):
            # For ranked items, just join keys
            a_text = "; ".join([str(k) for k in value.keys()])
        else:
            # Look up single option text
            options = questions_lookup.get(qid, {}).get("options", [])
            opt_map = {o.get("value"): o.get("text", o.get("value")) for o in options}
            a_text = opt_map.get(value, str(value))
        
        if q_text and a_text:
            qa_pairs.append(f"Q: {q_text}\nA: {a_text}")
    
    return "\n\n".join(qa_pairs)


def find_screener_answers(survey_results: List, member_id: str) -> List[Dict]:
    """Find screener answers for a member from survey results list."""
    # survey_results is a list of respondent objects
    for resp in survey_results:
        resp_id = resp.get("member_id", "")
        if resp_id == member_id:
            return resp.get("answers", [])
    return []


def build_questions_lookup(questions_data: Dict) -> Dict:
    """Build a lookup dict from questions data: {question_id: {text, options}}."""
    lookup = {}
    # questions_data has categories as keys, each with a list of questions
    for category, questions in questions_data.items():
        if not isinstance(questions, list):
            continue
        for q in questions:
            qid = q.get("id", "")
            if qid:
                lookup[qid] = {
                    "text": q.get("text", ""),
                    "options": q.get("options", [])
                }
    return lookup


def compute_scores(
    questions_data: Dict,
    audience_data: Dict,
    answers_data: List,
    output_path: str = "audience_with_scores.json"
) -> None:
    """Main function to compute alignment scores."""
    
    client = make_azure_client()
    
    # Build questions lookup for answer text resolution
    questions_lookup = build_questions_lookup(questions_data)
    print(f"Loaded {len(questions_lookup)} questions")
    
    # Extract audiences
    audiences = audience_data.get("audiences", [])
    if not audiences:
        raise ValueError("No audiences found in audience data")
    
    all_screener_scores = []
    all_parent_scores = []
    failed_members = []
    
    for aud_idx, audience in enumerate(audiences):
        # Get parent persona from metadata
        metadata = audience.get("metadata", {})
        parent_persona = metadata.get("persona", {})
        screener_questions = metadata.get("screener_questions", [])
        parent_text = extract_parent_text(parent_persona, screener_questions)
        
        if not parent_text:
            print(f"Warning: No parent text for audience {aud_idx}")
            continue
        
        # Embed parent text
        print(f"\nProcessing Audience {aud_idx + 1}...")
        parent_embedding = embed_text(client, parent_text)
        
        # Process generated members
        generated = audience.get("generated_audience", [])
        
        for mem_idx, member in enumerate(generated):
            member_id = member.get("member_id", f"AUD{aud_idx}_{mem_idx:04d}")
            
            # Extract generated text
            generated_text = extract_generated_text(member)
            if not generated_text:
                print(f"  {member_id}: skipped (no text)")
                continue
            
            # Find screener answers for this member
            screener_answers = find_screener_answers(answers_data, member_id)
            screener_text = extract_screener_text(screener_answers, questions_lookup)
            
            # Compute embeddings
            generated_embedding = embed_text(client, generated_text)
            
            # parent_score: dot_product(parent, generated)
            parent_score = dot_product(parent_embedding, generated_embedding)
            
            # screener_score: dot_product(parent + screener, generated)
            if screener_text:
                combined_text = parent_text + "\n\n" + screener_text
                combined_embedding = embed_text(client, combined_text)
                screener_score = dot_product(combined_embedding, generated_embedding)
            else:
                # If no screener data, use parent score
                screener_score = parent_score
            
            # Determine pass/fail
            screener_passed = screener_score >= SCREENER_THRESHOLD
            parent_passed = parent_score >= PARENT_THRESHOLD
            validation_passed = screener_passed and parent_passed
            
            # Add scores to member
            member["screener_score"] = round(screener_score, 4)
            member["parent_score"] = round(parent_score, 4)
            member["screener_passed"] = screener_passed
            member["parent_passed"] = parent_passed
            member["validation_passed"] = validation_passed
            
            all_screener_scores.append(screener_score)
            all_parent_scores.append(parent_score)
            
            if not validation_passed:
                failed_members.append({
                    "member_id": member_id,
                    "screener_score": round(screener_score, 4),
                    "parent_score": round(parent_score, 4)
                })
            
            status = "✓" if validation_passed else "✗"
            print(f"  {member_id}: screener={screener_score:.4f}, parent={parent_score:.4f} {status}")
    
    # Save enriched output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audience_data, f, indent=2)
    print(f"\n✓ Saved enriched output to: {output_path}")
    
    # Save failed members
    if failed_members:
        failed_path = "failed_members.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_members, f, indent=2)
        print(f"✓ Saved failed members to: {failed_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if all_screener_scores:
        avg_screener = np.mean(all_screener_scores)
        avg_parent = np.mean(all_parent_scores)
        passed_count = sum(1 for s, p in zip(all_screener_scores, all_parent_scores) 
                         if s >= SCREENER_THRESHOLD and p >= PARENT_THRESHOLD)
        total = len(all_screener_scores)
        pass_rate = (passed_count / total) * 100
        
        print(f"Total members:       {total}")
        print(f"Avg screener_score:  {avg_screener:.4f} (threshold: {SCREENER_THRESHOLD})")
        print(f"Avg parent_score:    {avg_parent:.4f} (threshold: {PARENT_THRESHOLD})")
        print(f"Passed:              {passed_count}/{total} ({pass_rate:.1f}%)")
        
        # Top 5 worst offenders
        if failed_members:
            print("\nTop 5 worst offenders:")
            sorted_failed = sorted(failed_members, key=lambda x: min(x["screener_score"], x["parent_score"]))
            for m in sorted_failed[:5]:
                print(f"  {m['member_id']}: screener={m['screener_score']}, parent={m['parent_score']}")


def main():
    parser = argparse.ArgumentParser(description="Compute alignment scores for generated personas")
    
    # Required file arguments
    parser.add_argument("--audience-file", required=True, help="Path to generated audience JSON (e.g., data/generate_persona.json)")
    parser.add_argument("--answers-file", required=True, help="Path to survey answers JSON (e.g., data/survey_output.json)")
    parser.add_argument("--questions-file", required=True, help="Path to questions JSON (e.g., data/questions_extracted.json)")
    
    # Output
    parser.add_argument("--output", default="audience_with_scores.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Load data from files
    print("Loading data...")
    questions_data = load_json_from_file(args.questions_file)
    audience_data = load_json_from_file(args.audience_file)
    answers_data = load_json_from_file(args.answers_file)
    
    compute_scores(questions_data, audience_data, answers_data, args.output)


if __name__ == "__main__":
    main()
