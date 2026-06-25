"""
embeddings/generate_jd_emb.py
Run ONCE offline to precompute:
  - jd_embedding.npy : the JD embedding vector
  - candidate_embeddings.npy : (N, dim) array for all candidates
  - candidate_ids.txt : candidate IDs in same order

This is the slow step (~90-120 seconds CPU). The rank.py script
loads these files and runs in <5 minutes total.

Usage:
  python embeddings/generate_jd_emb.py \
      --candidates ../candidates.jsonl \
      --out ./embeddings/
"""

import argparse
import json
import os
import time

import numpy as np


JD_TEXT = """
Senior AI Engineer founding team role. Production experience with embeddings-based retrieval 
systems using sentence-transformers, BGE, E5, or similar deployed to real users. 
Vector database experience with Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, 
Elasticsearch, FAISS. Hybrid search infrastructure. Strong Python. 
Evaluation frameworks for ranking systems: NDCG, MRR, MAP, offline to online correlation, 
A/B test interpretation. LLM fine-tuning experience with LoRA, QLoRA, PEFT. 
Learning to rank XGBoost neural. Candidate JD matching at scale. 
Retrieval ranking recommendation systems shipped to real users at product companies. 
Applied ML AI roles not consulting firms. Building evaluation infrastructure offline 
benchmarks online A/B testing. Semantic search dense retrieval BM25.
"""


def get_candidate_text(cand: dict) -> str:
    """Concatenate all career descriptions + title for embedding."""
    parts = []
    profile = cand.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))

    for role in cand.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))

    skills_text = " ".join(s.get("name", "") for s in cand.get("skills", []))
    parts.append(skills_text)

    return " ".join(p for p in parts if p).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="embeddings/")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    # Embed JD
    print("Embedding JD...")
    jd_emb = model.encode([JD_TEXT], normalize_embeddings=True)[0]
    np.save(os.path.join(args.out, "jd_embedding.npy"), jd_emb)
    print(f"JD embedding shape: {jd_emb.shape}")

    # Load and embed all candidates
    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    n = len(candidates)
    print(f"Loaded {n} candidates. Embedding...")

    texts = [get_candidate_text(c) for c in candidates]
    ids = [c["candidate_id"] for c in candidates]

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"Embedding done in {elapsed:.1f}s. Shape: {embeddings.shape}")

    np.save(os.path.join(args.out, "candidate_embeddings.npy"), embeddings)
    with open(os.path.join(args.out, "candidate_ids.txt"), "w") as f:
        for cid in ids:
            f.write(cid + "\n")

    print(f"Saved to {args.out}")
    print("Done. Pre-computation complete.")


if __name__ == "__main__":
    main()
