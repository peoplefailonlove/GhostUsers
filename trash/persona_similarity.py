"""
Compute dot products between a parent persona and each generated child persona
using Azure OpenAI text embeddings.

Usage:
    python persona_similarity.py \
        --parent data/parent_persona.json \
        --children data/child_personas.json

Required environment variables:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION (e.g. 2024-08-01-preview)
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT (deployment name for your embedding model)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI


def make_azure_client() -> AzureOpenAI:
    """Create an Azure OpenAI client from environment variables."""
    load_dotenv()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    missing = [
        name
        for name, val in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_API_VERSION", api_version),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            "Missing required Azure OpenAI environment variables: " + ", ".join(missing)
        )

    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


def embed_text(client: AzureOpenAI, text: str) -> List[float]:
    """Get an embedding for the given text from Azure OpenAI."""
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to your embedding deployment name.")

    response = client.embeddings.create(model=deployment, input=text)
    return response.data[0].embedding


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parent_persona_text(parent_data: Dict) -> str:
    """Extract a representative text block for the parent persona."""
    audiences = parent_data.get("audiences") or []
    if not audiences:
        raise ValueError("No audiences found in parent persona JSON.")
    persona = audiences[0].get("persona") or {}

    fields = [
        persona.get("generatedTitle"),
        persona.get("personaName"),
        persona.get("personaType"),
        persona.get("about"),
        persona.get("goalsAndMotivations"),
        persona.get("frustrations"),
        persona.get("needState"),
        persona.get("occasions"),
    ]
    return "\n".join([f for f in fields if f])


def child_persona_text(child: Dict) -> str:
    """Combine key text fields from a child persona into a single string."""
    fields = [
        child.get("about"),
        "; ".join(child.get("goals_and_motivations") or []),
        "; ".join(child.get("frustrations") or []),
        child.get("need_state"),
        child.get("occasions"),
    ]
    return "\n".join([f for f in fields if f])


def compute_similarities(parent_path: Path, children_path: Path) -> None:
    client = make_azure_client()

    parent_data = load_json(parent_path)
    children_data = load_json(children_path)

    parent_text = parent_persona_text(parent_data)
    parent_embedding = np.array(embed_text(client, parent_text))

    audiences = children_data.get("audiences") or []
    if not audiences:
        raise ValueError("No audiences found in child personas JSON.")

    print(f"Parent persona source: {parent_path}")
    for audience in audiences:
        generated = audience.get("generated_audience") or []
        audience_index = audience.get("metadata", {}).get("audience_index", "unknown")
        print(f"\nAudience {audience_index}:")
        for child in generated:
            member_id = child.get("member_id", "unknown")
            child_text = child_persona_text(child)
            if not child_text:
                print(f"  {member_id}: skipped (no text fields present)")
                continue
            child_embedding = np.array(embed_text(client, child_text))
            dot_product = float(np.dot(parent_embedding, child_embedding))
            print(f"  {member_id}: dot_product={dot_product:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dot products between parent and child personas.")
    parser.add_argument("--parent", required=True, type=Path, help="Path to parent_persona.json")
    parser.add_argument("--children", required=True, type=Path, help="Path to child_personas.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_similarities(args.parent, args.children)
