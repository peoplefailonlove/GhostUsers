"""
3D Persona Embedding Visualization — Fixed & Beautiful
OpenAI text-embedding-3-small + PCA → Interactive 3D Plot
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.decomposition import PCA

# === Setup ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"

# Valid 3D symbols in Plotly
VALID_SYMBOLS = [
    "circle", "circle-open", "square", "square-open",
    "diamond", "diamond-open", "cross", "x"
]

def safe_str(value: Any) -> str:
    if value is None or value == "": return ""
    if isinstance(value, (list, dict)): return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

def extract_persona_text(persona_data: Dict, member_id: str = "", audience_label: str = "") -> str:
    def _norm(val: Any) -> str:
        if val is None: return ""
        if isinstance(val, dict):
            items = [f"{k}: {v}" for k, v in val.items() if v not in (None, "", [], {})]
            return " | ".join(items)
        if isinstance(val, (list, tuple)):
            return " | ".join(str(v) for v in val if v not in (None, "", [], {}))
        return safe_str(val)

    parts = []
    if member_id: parts.append(f"ID: {member_id}")
    if audience_label: parts.append(f"AUDIENCE: {audience_label}")

    for k in ["personaName", "generatedTitle", "about", "personaType"]:
        if persona_data.get(k): parts.append(safe_str(persona_data[k]))
    for k in ["age", "gender", "location", "job_title", "income_range"]:
        if persona_data.get(k): parts.append(f"{k}: {safe_str(persona_data[k])}")
    if attrs := persona_data.get("attributes"):
        if isinstance(attrs, dict) and any(attrs.values()):
            parts.append(f"Attributes: {_norm(attrs)}")

    parts.extend([
        f"Goals: {_norm(persona_data.get('goals_and_motivations') or persona_data.get('goalsAndMotivations'))}",
        f"Frustrations: {_norm(persona_data.get('frustrations'))}",
        f"Need State: {_norm(persona_data.get('need_state') or persona_data.get('needState'))}",
    ])

    text = "  ||  ".join([p for p in parts if p.strip()])
    return text or "EMPTY PERSONA"

def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb

# === Load Data ===
def load_parent(path: Path) -> Tuple[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    persona = data["audiences"][0]["persona"]
    text = extract_persona_text(persona, "PARENT", "PARENT")
    emb = get_embedding(text)
    print(f"Parent embedded ({len(text)} chars)")
    return text, emb

def load_children(path: Path) -> List[Tuple[str, str, str, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for audience in data.get("audiences", []):
        aud_idx = audience.get("audience_index") or audience.get("metadata", {}).get("audience_index", 0)
        label = f"Audience {aud_idx}"
        members = audience.get("members") or audience.get("generated_audience") or audience.get("generated") or []
        for child in members:
            mid = child.get("member_id") or child.get("id") or "unknown"
            wrapper = child.get("persona_template") or child
            text = extract_persona_text(wrapper, mid, label)
            if len(text) < 50: continue
            emb = get_embedding(text)
            results.append((label, mid, text, emb))
    print(f"Loaded {len(results)} child personas")
    return results

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--children", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("persona_3d_fixed.html"))
    args = parser.parse_args()

    print("Loading parent persona...")
    parent_text, parent_emb = load_parent(args.parent)

    print("Loading child personas...")
    children = load_children(args.children)

    # Combine all embeddings
    all_embs = np.vstack([parent_emb] + [c[3] for c in children])

    # Reduce to 3D with PCA
    pca = PCA(n_components=3, random_state=42)
    coords_3d = pca.fit_transform(all_embs)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained:.1%}")

    parent_coord = coords_3d[0]
    child_coords = coords_3d[1:]

    # Color per audience
    audiences = sorted({c[0] for c in children})
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#FF6692", "#19D3F3", "#FF97A6"]
    color_map = {aud: colors[i % len(colors)] for i, aud in enumerate(audiences)}

    fig = go.Figure()

    # Add child points
    for (label, mid, text, emb), (x, y, z) in zip(children, child_coords):
        sim = float(np.dot(parent_emb, emb))
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers",
            name=label,
            marker=dict(
                size=7,
                color=color_map[label],
                opacity=0.8,
                line=dict(width=1, color="black")
            ),
            text=mid,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Member: {mid}<br>"
                f"Similarity to Parent: {sim:.3f}<extra></extra>"
            ),
            legendgroup=label,
            showlegend=(mid == children[0][1])  # Show legend only once per audience
        ))

    # Add PARENT — BIG, RED, DIAMOND
    px, py, pz = parent_coord
    fig.add_trace(go.Scatter3d(
        x=[px], y=[py], z=[pz],
        mode="markers+text",
        name="PARENT PERSONA",
        marker=dict(
            size=18,
            color="red",
            symbol="diamond",           # Valid 3D symbol
            line=dict(width=3, color="darkred")
        ),
        text="PARENT",
        textposition="bottom center",
        hovertemplate="<b>PARENT PERSONA</b><br>Reference point<extra></extra>",
        showlegend=True
    ))

    fig.update_layout(
        title=f"<b>3D Persona Embedding Space</b><br>"
              f"OpenAI text-embedding-3-small • PCA ({explained:.1%} variance explained)",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2))
        ),
        width=1200,
        height=800,
        legend=dict(title="Audience Groups"),
        margin=dict(l=0, r=0, t=100, b=0),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"\nSUCCESS! 3D plot saved: {args.out.resolve()}")
    print("Open it in your browser → rotate, zoom, hover → beautiful clusters!")

if __name__ == "__main__":
    main()