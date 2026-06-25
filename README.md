# Redrob Intelligent Candidate Ranker

A hybrid multi-signal ranking system for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. Ranks 100,000 candidates for a Senior AI Engineer (Founding Team) role using semantic + behavioral signals — not keyword matching.

**Runtime:** ~57 seconds for 100K candidates on CPU · **RAM:** ~4 GB peak · **Network:** None during ranking.

---

## How it works

```
Stage 1: Pre-computation (offline, once)
         ↓ Embed JD + all 100K candidates using all-MiniLM-L6-v2

Stage 2: Base scoring (6 components)
         C1 Career narrative fit       30%  ← cosine sim (JD vs career text)
         C2 Production AI experience   20%  ← rule-based, anti-trap
         C3 Experience seniority       15%
         C4 Skill depth trust          15%  ← endorsement × duration multiplier
         C5 Location & logistics       10%
         C6 Education + GitHub         10%

Stage 3: Behavioral multiplier [0.25, 1.20]
         ← Redrob platform signals: response rate, recency, open-to-work, etc.

Stage 4: Sort + top-100 + per-candidate reasoning
```

### What makes this different from keyword matching

- **Consulting-only career penalty** — TCS/Wipro/Infosys/Accenture full-career → 0.15× base score
- **Keyword stuffer trap** — "Marketing Manager" with 9 expert AI skills scores near 0 because C2 is gated by career narrative
- **Skill trust multiplier** — Skills with 0 endorsements AND 0 duration_months get near-zero credit
- **Career plausibility guard** — If C2 < 0.25 (no real AI career), C4 is capped proportionally. This is the key anti-trap for "Graphic Designer with Pinecone skills"
- **Behavioral ghost penalty** — Candidates inactive 6+ months with <20% response rate get ≤40% of base
- **Honeypot detection** — Impossible career duration math or 3+ expert skills with 0 months/endorsements → score 0
- **LangChain tourist detection** — All AI experience post-2022 and <24 months → 0.7× recency penalty
- **Plain-language AI engineers win** — Semantic embedding captures "built recommendation system" even without buzzwords

---

## Quick start

### 1. Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Requires Python 3.12. Sentence-transformers needs PyTorch, which is slow to install but only once.

### 2. Pre-compute embeddings (one-time)

You need 3 files in `embeddings/`: `jd_embedding.npy`, `candidate_embeddings.npy`, `candidate_ids.txt`.

**Option A — HuggingFace Space (recommended for large datasets):**

The `hf-space/` directory is a ready-to-deploy Gradio app:

```bash
cd hf-space
# Push to your own Space, or run locally
python app.py
```

Upload `candidates.jsonl`, click "Generate Embeddings", download the zip, extract into `embeddings/`.

**Option B — Local generation (will be slow on CPU):**

```bash
python embeddings/generate_jd_emb.py --candidates ./candidates.jsonl --out ./embeddings/
```

> ⚠️ 100K candidates takes ~30+ minutes on CPU. Use Option A with a GPU Space.

### 3. Rank candidates

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This is the reproducible command. Runs in **under 60 seconds on CPU** with 16 GB RAM. No network calls.

### 4. Validate

```bash
python validate_submission.py ./submission.csv
```

### 5. Run tests

```bash
pytest tests/ -v
```

17 unit + integration tests covering all scoring components, anti-trap measures, and edge cases.

### 6. Launch sandbox UI (optional)

```bash
streamlit run app.py
```

Upload a small JSONL sample, see scores in real time.

---

## File structure

```
redrob-ranker/
├── rank.py                       # Main entry point (reproduce command)
├── scorer.py                     # 6-component scoring engine
├── signals.py                    # Behavioral multiplier
├── reasoning.py                  # Per-candidate reasoning generator
├── app.py                        # Streamlit sandbox UI
├── embeddings/
│   └── generate_jd_emb.py        # Offline precomputation script
├── hf-space/                     # HuggingFace Space (Gradio) for cloud embedding
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── tests/
│   └── test_scorer.py            # 17 unit + integration tests
├── validate_submission.py        # Official submission validator
├── submission_metadata.yaml      # Hackathon metadata (fill in personal details)
├── submission_metadata_template.yaml
├── candidate_schema.json         # Input data schema
├── sample_candidates.json        # 10-candidate sample for testing
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Output format

`submission.csv` — exactly 100 rows, columns:

| Column | Description |
|--------|-------------|
| `candidate_id` | `CAND_XXXXXXX` (7 digits) |
| `rank` | Integer 1–100 |
| `score` | Float in [0.0, 1.0], non-increasing by rank |
| `reasoning` | 1–2 sentence factual summary specific to the candidate |

Score tie-breaking: when two candidates have identical scores (at 4-decimal precision), candidate_id sorts ascending.

---

## Methodology summary

Rule-based + semantic hybrid ranker. Six scoring components combined with weights tuned to the JD's explicit priorities (production AI at product companies, evaluation framework experience, behavioral availability). The semantic component (`all-MiniLM-L6-v2`, 384-dim) captures plain-language AI engineers who built retrieval/ranking systems without using modern buzzwords. The behavioral multiplier ensures that perfect-on-paper-but-unreachable candidates are down-weighted appropriately. Anti-trap measures explicitly penalize keyword stuffers, consulting lifers, behavioral ghosts, and honeypot profiles.

### Why this beats pure embedding cosine sim

Pure embedding similarity ranks "Marketing Manager" who pasted 9 AI tool names above a real ML engineer at Zomato — because their resume text *mentions* the right things. The 6-component base score gates C4 (skill depth) by C2 (real production AI evidence), so a fake profile's skill claims collapse when their career narrative shows zero AI work.

### Why this beats a pure rules-based approach

A pure rules-based approach can't tell "I built a hybrid retrieval system with BM25 and dense embeddings for legal contract search" from "I used Elasticsearch once." Semantic embeddings capture the former even when it uses non-buzzword vocabulary.

---

## Compute requirements

| Stage | Time | RAM | Network |
|-------|------|-----|---------|
| Embedding generation (100K, CPU) | ~30–40 min | ~4 GB | First run only (model download) |
| Embedding generation (100K, L4 GPU) | ~2 min | ~4 GB | First run only (model download) |
| Ranking (100K, CPU) | ~57 sec | ~4 GB | None |
| Ranking (100K, GPU) | N/A | N/A | Not supported (no benefit) |

The ranking step is CPU-bound on NumPy operations and string/regex matching. No GPU needed. The embedding pre-computation is the only expensive step, and it only runs once.

---

## AI tools declaration

This project was built with AI assistance (Claude / Sisyphus orchestrator) for:
- Code review and architectural discussion
- Debugging the tie-breaking edge case in `rank.py`
- Writing the README and methodology summary

No candidate data was fed to any LLM. The scoring engine, embeddings generation, and ranking pipeline are original work.

---

## License

MIT
