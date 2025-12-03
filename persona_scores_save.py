"""
Script A: Compute persona embedding scores and save to CSV.

Per audience:
- Run a separate PCA on [parent + that audience's members]
- 'score' = distance in that group's 3D PCA space from PARENT
- 'similarity_score' = cosine similarity to parent in original embedding space

CSV columns:
- score
- member
- similarity_score
- audience
- pc1
- pc2
- pc3
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA

# === Setup ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"


def safe_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def extract_persona_text(
    persona_data: Dict, member_id: str = "", audience_label: str = ""
) -> str:
    def _norm(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, dict):
            items = [f"{k}: {v}" for k, v in val.items() if v not in (None, "", [], {})]
            return " | ".join(items)
        if isinstance(val, (list, tuple)):
            return " | ".join(str(v) for v in val if v not in (None, "", [], {}))
        return safe_str(val)

    parts = []
    if member_id:
        parts.append(f"ID: {member_id}")
    if audience_label:
        parts.append(f"AUDIENCE: {audience_label}")

    for k in ["personaName", "generatedTitle", "about", "personaType"]:
        if persona_data.get(k):
            parts.append(safe_str(persona_data[k]))
    for k in ["age", "gender", "location", "job_title", "income_range"]:
        if persona_data.get(k):
            parts.append(f"{k}: {safe_str(persona_data[k])}")
    if attrs := persona_data.get("attributes"):
        if isinstance(attrs, dict) and any(attrs.values()):
            parts.append(f"Attributes: {_norm(attrs)}")

    parts.extend(
        [
            f"Goals: {_norm(persona_data.get('goals_and_motivations') or persona_data.get('goalsAndMotivations'))}",
            f"Frustrations: {_norm(persona_data.get('frustrations'))}",
            f"Need State: {_norm(persona_data.get('need_state') or persona_data.get('needState'))}",
        ]
    )

    text = "  ||  ".join([p for p in parts if p.strip()])
    return text or "EMPTY PERSONA"


def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def load_parent(path: Path) -> Tuple[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    persona = data["audiences"][0]["persona"]
    text = extract_persona_text(persona, "PARENT", "PARENT")
    emb = get_embedding(text)
    print(f"Parent embedded ({len(text)} chars)")
    return text, emb


def load_children(path: Path) -> List[Tuple[str, str, str, np.ndarray]]:
    """
    Returns list of tuples: (audience_label, member_id, text, embedding)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    results: List[Tuple[str, str, str, np.ndarray]] = []

    for audience in data.get("audiences", []):
        aud_idx = (
            audience.get("audience_index")
            or audience.get("metadata", {}).get("audience_index", 0)
        )
        label = f"Audience {aud_idx}"
        members = (
            audience.get("members")
            or audience.get("generated_audience")
            or audience.get("generated")
            or []
        )
        for child in members:
            mid = child.get("member_id") or child.get("id") or "unknown"
            wrapper = child.get("persona_template") or child
            text = extract_persona_text(wrapper, mid, label)
            if len(text) < 50:
                continue
            emb = get_embedding(text)
            results.append((label, mid, text, emb))

    print(f"Loaded {len(results)} child personas")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument(
        "--csv-out", type=Path, default=Path("persona_scores.csv")
    )
    args = parser.parse_args()

    print("Loading parent persona...")
    parent_text, parent_emb = load_parent(args.parent)

    print("Loading child personas...")
    children = load_children(args.children)
    if not children:
        raise ValueError("No valid child personas found with sufficient text length.")

    # Group children by audience
    children_by_aud: Dict[str, List[Tuple[str, str, str, np.ndarray]]] = {}
    for aud_label, member_id, text, emb in children:
        children_by_aud.setdefault(aud_label, []).append(
            (aud_label, member_id, text, emb)
        )

    # Prepare CSV
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "score",
                "member",
                "similarity_score",
                "audience",
                "pc1",
                "pc2",
                "pc3",
            ],
        )
        writer.writeheader()

        # For each audience, run separate PCA on [parent + that group's members]
        for aud_label, group in children_by_aud.items():
            group_embs = [emb for (_, _, _, emb) in group]
            all_embs_group = np.vstack([parent_emb] + group_embs)

            pca = PCA(n_components=3, random_state=42)
            coords_group = pca.fit_transform(all_embs_group)
            explained = pca.explained_variance_ratio_.sum()
            print(
                f"[{aud_label}] PCA explained variance: {explained:.1%} "
                f"for {len(group)} members"
            )

            parent_coord = coords_group[0]
            child_coords_group = coords_group[1:]

            for (aud, member_id, text, emb), coord in zip(group, child_coords_group):
                pc1, pc2, pc3 = map(float, coord)
                score = float(np.linalg.norm(coord - parent_coord))
                similarity = float(np.dot(parent_emb, emb))

                writer.writerow(
                    {
                        "score": score,
                        "member": member_id,
                        "similarity_score": similarity,
                        "audience": aud,
                        "pc1": pc1,
                        "pc2": pc2,
                        "pc3": pc3,
                    }
                )

    print(f"Scores CSV (with PCA 3D coords) saved to: {args.csv_out.resolve()}")


if __name__ == "__main__":
    main()
