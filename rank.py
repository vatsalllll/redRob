#!/usr/bin/env python3
"""
rank.py — Redrob Intelligent Candidate Ranking System
======================================================

Produce a top-100 ranked CSV from candidates.jsonl for the Senior AI Engineer JD.

Usage (ranking step — must complete in <5 min, CPU only, no network):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pre-computation (run once offline, not counted in the 5-min budget):
    python embeddings/generate_jd_emb.py --candidates ./candidates.jsonl --out ./embeddings/

Architecture:
    Stage 1: Load precomputed embeddings + candidate data
    Stage 2: Compute 6-component base score for each of 100K candidates
    Stage 3: Apply behavioral multiplier (Redrob signals)
    Stage 4: Sort, take top 100, generate reasoning
    Stage 5: Write validated CSV

Scoring components:
    C1 career_narrative_fit       30%  (cosine sim vs JD embedding)
    C2 production_ai_experience   20%  (rule-based, anti-trap)
    C3 experience_seniority       15%
    C4 skill_depth                15%  (endorsement x duration trust)
    C5 location_logistics         10%
    C6 education_github           10%
    ×  behavioral_multiplier      [0.25, 1.20] (Redrob signals)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Local modules
from scorer import compute_base_score
from signals import compute_final_score
from reasoning import generate_reasoning


def load_candidates(path: str) -> list[dict]:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_embeddings(embeddings_dir: str) -> tuple:
    """Load precomputed JD + candidate embeddings."""
    jd_emb = np.load(os.path.join(embeddings_dir, "jd_embedding.npy"))
    cand_emb = np.load(os.path.join(embeddings_dir, "candidate_embeddings.npy"))
    with open(os.path.join(embeddings_dir, "candidate_ids.txt")) as f:
        emb_ids = [line.strip() for line in f if line.strip()]
    return jd_emb, cand_emb, emb_ids


def compute_cosine_batch(jd_emb: np.ndarray, cand_emb: np.ndarray) -> np.ndarray:
    """
    Batch cosine similarity. Both must be L2-normalized.
    jd_emb: (dim,)
    cand_emb: (N, dim)
    Returns: (N,) array of similarities in [-1, 1]
    """
    # Since embeddings are L2-normalized, cosine similarity = dot product
    sims = cand_emb @ jd_emb
    # Shift to [0, 1]
    sims = (sims + 1.0) / 2.0
    return sims.astype(float)


def rank_candidates(
    candidates: list[dict],
    jd_emb: np.ndarray,
    cand_emb: np.ndarray,
    emb_ids: list[str],
    verbose: bool = True,
) -> list[dict]:
    """Score all candidates and return sorted list."""

    # Build id→embedding index map
    id_to_idx = {cid: i for i, cid in enumerate(emb_ids)}

    if verbose:
        print(f"Computing cosine similarities for {len(candidates)} candidates...")

    # Batch cosine similarities for candidates that have embeddings
    has_emb = [c["candidate_id"] in id_to_idx for c in candidates]
    emb_order = [id_to_idx[c["candidate_id"]] for c in candidates if c["candidate_id"] in id_to_idx]
    cosine_sims_all = compute_cosine_batch(jd_emb, cand_emb)

    t0 = time.time()
    results = []

    for i, cand in enumerate(candidates):
        if i % 10000 == 0 and verbose:
            print(f"  [{i}/{len(candidates)}] {time.time()-t0:.1f}s elapsed...")

        cid = cand["candidate_id"]

        # Cosine similarity
        if cid in id_to_idx:
            cosine_sim = float(cosine_sims_all[id_to_idx[cid]])
        else:
            cosine_sim = 0.35  # fallback if embedding missing

        # Base score (6 components)
        scores = compute_base_score(cand, cosine_sim)

        # Final score with behavioral multiplier
        if scores["honeypot"]:
            final = 0.0
        else:
            final = compute_final_score(scores["base_score"], cand)

        results.append({
            "candidate": cand,
            "candidate_id": cid,
            "final_score": final,
            "scores": scores,
        })

    if verbose:
        print(f"Scoring done in {time.time()-t0:.1f}s")

    for r in results:
        r["rounded_score"] = round(r["final_score"], 4)
    # Triple-key sort: unrounded score first (preserves discrimination),
    # then rounded score (CSV consistency), then candidate_id for deterministic tie-break.
    results.sort(key=lambda x: (-x["final_score"], -x["rounded_score"], x["candidate_id"]))

    return results


def build_submission(ranked: list[dict], top_n: int = 100) -> list[dict]:
    """Build the submission rows with reasoning."""
    rows = []
    for rank_idx, item in enumerate(ranked[:top_n]):
        rank = rank_idx + 1
        cand = item["candidate"]
        final_score = item["final_score"]
        scores = item["scores"]

        reasoning = generate_reasoning(cand, rank, final_score, scores)

        rows.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": round(final_score, 4),
            "reasoning": reasoning,
        })

    return rows


def write_csv(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def validate_output(rows: list[dict]) -> bool:
    """Basic sanity checks matching submission_spec."""
    ok = True

    if len(rows) != 100:
        print(f"ERROR: Expected 100 rows, got {len(rows)}")
        ok = False

    ranks = [r["rank"] for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        print("ERROR: Ranks are not exactly 1..100")
        ok = False

    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != 100:
        print("ERROR: Duplicate candidate_ids in submission")
        ok = False

    scores = [r["score"] for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1] + 1e-6:
            print(f"ERROR: Score at rank {i+1} > score at rank {i}: {scores[i]} > {scores[i-1]}")
            ok = False
            break

    if ok:
        print("Validation PASSED: 100 rows, ranks 1-100, scores non-increasing, unique IDs")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--embeddings-dir", default="embeddings/", help="Dir with precomputed .npy files")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to rank")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    t_total = time.time()

    # --- Check embeddings exist ---
    emb_dir = args.embeddings_dir
    jd_emb_path = os.path.join(emb_dir, "jd_embedding.npy")
    cand_emb_path = os.path.join(emb_dir, "candidate_embeddings.npy")
    ids_path = os.path.join(emb_dir, "candidate_ids.txt")

    embeddings_available = all(os.path.exists(p) for p in [jd_emb_path, cand_emb_path, ids_path])

    if not embeddings_available:
        print("=" * 70)
        print("ERROR: Precomputed embeddings not found.")
        print(f"  Expected: {jd_emb_path}, {cand_emb_path}, {ids_path}")
        print()
        print("The ranking step requires embeddings generated offline.")
        print("Choose one of these options:")
        print()
        print("  1. Local generation (slow, ~30 min on CPU):")
        print(f"     python embeddings/generate_jd_emb.py --candidates {args.candidates} --out ./embeddings/")
        print()
        print("  2. HuggingFace Space (fast, ~2 min on GPU):")
        print("     Deploy hf-space/ to a Space, upload candidates.jsonl,")
        print("     download the zip, extract into ./embeddings/")
        print("     See: https://huggingface.co/spaces/VatsalHF30/redRob")
        print()
        print("  3. Download pre-computed embeddings (if available):")
        print("     Check the GitHub repo's Releases page.")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f"Loading embeddings from {emb_dir}...")
        jd_emb, cand_emb, emb_ids = load_embeddings(emb_dir)
        print(f"  JD embedding: {jd_emb.shape}")
        print(f"  Candidate embeddings: {cand_emb.shape}")

    # --- Load candidates ---
    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates")

    # --- Score & rank ---
    ranked = rank_candidates(candidates, jd_emb, cand_emb, emb_ids, verbose=args.verbose)

    # --- Build submission ---
    print("Building top-100 submission with reasoning...")
    rows = build_submission(ranked, top_n=args.top_n)

    # --- Validate ---
    validate_output(rows)

    # --- Honeypot audit ---
    from scorer import is_honeypot
    cand_by_id = {item["candidate_id"]: item["candidate"] for item in ranked}
    audit_top_n = min(100, len(rows))
    honeypots_in_top = [
        r["candidate_id"] for r in rows[:audit_top_n]
        if is_honeypot(cand_by_id[r["candidate_id"]])
    ]
    honeypot_rate = len(honeypots_in_top) / audit_top_n * 100
    print(f"\nHoneypot audit (top {audit_top_n}): {len(honeypots_in_top)} found ({honeypot_rate:.1f}%)")
    if honeypots_in_top:
        print(f"  WARNING: Honeypots in top {audit_top_n}: {honeypots_in_top}")
        if honeypot_rate > 10.0:
            print("  CRITICAL: Honeypot rate exceeds 10% — submission would be DISQUALIFIED at Stage 3")
    else:
        print(f"  PASS: No honeypots in top {audit_top_n} (well under 10% threshold)")

    # --- Write ---
    write_csv(rows, args.out)

    elapsed = time.time() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s")
    if elapsed > 300:
        print("WARNING: Exceeded 5-minute budget! Optimize batch size or scoring.")
    else:
        print(f"Within 5-minute budget. ({300 - elapsed:.1f}s remaining)")


if __name__ == "__main__":
    main()
