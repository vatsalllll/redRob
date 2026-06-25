"""
app.py — Streamlit sandbox for Redrob Ranker
Fulfills the submission_spec Section 10.5 "Sandbox / demo link" requirement.

Deploy to: streamlit.io/cloud or HuggingFace Spaces

Usage:
    streamlit run app.py
"""

import json
import csv
import io
import streamlit as st
import numpy as np
from scorer import compute_base_score
from signals import compute_final_score
from reasoning import generate_reasoning

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Senior AI Engineer — Founding Team · Redrob AI · Pune/Noida")

st.markdown("""
Upload a **JSONL file** with up to 100 candidates. The ranker applies a 6-component 
scoring model (career narrative, production AI experience, seniority, skill depth, 
location/logistics, education) multiplied by behavioral signals from the Redrob platform.
""")

uploaded = st.file_uploader("Upload candidates (JSONL)", type=["jsonl", "json"])

if uploaded:
    raw = uploaded.read().decode("utf-8")
    candidates = []
    
    # Support both JSONL and JSON array
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            candidates = data
        else:
            candidates = [data]
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass

    if not candidates:
        st.error("Could not parse any candidates from the uploaded file.")
        st.stop()

    st.success(f"Loaded {len(candidates)} candidates")

    if len(candidates) > 100:
        st.warning("Sandbox capped at 100 candidates. Using first 100.")
        candidates = candidates[:100]

    with st.spinner("Scoring candidates..."):
        results = []
        for cand in candidates:
            # No embeddings in sandbox — use 0.5 cosine sim as neutral fallback
            cosine_sim = 0.50
            scores = compute_base_score(cand, cosine_sim)
            final = compute_final_score(scores["base_score"], cand)
            results.append({
                "candidate": cand,
                "candidate_id": cand.get("candidate_id", "UNKNOWN"),
                "final_score": final,
                "scores": scores,
            })

        results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # Build output rows
    rows = []
    for i, item in enumerate(results):
        rank = i + 1
        reasoning = generate_reasoning(item["candidate"], rank, item["final_score"], item["scores"])
        rows.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": round(item["final_score"], 4),
            "reasoning": reasoning,
        })

    # Display table
    st.subheader(f"Top {min(len(rows), 20)} Results")
    
    import pandas as pd
    df = pd.DataFrame(rows[:20])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Component breakdown for top candidates
    st.subheader("Score Breakdown — Top 5")
    cols = st.columns(5)
    for col, item in zip(cols, results[:5]):
        sc = item["scores"]
        p = item["candidate"].get("profile", {})
        with col:
            st.metric("Rank", f"#{results.index(item)+1}")
            st.caption(f"**{p.get('current_title','?')}**")
            st.caption(f"{p.get('current_company','?')}")
            st.caption(f"{p.get('years_of_experience',0):.1f} yrs")
            st.progress(sc.get("c1", 0), text=f"Narrative {sc.get('c1',0):.2f}")
            st.progress(sc.get("c2", 0), text=f"Prod AI {sc.get('c2',0):.2f}")
            st.progress(sc.get("c3", 0), text=f"Seniority {sc.get('c3',0):.2f}")
            st.progress(sc.get("c4", 0), text=f"Skills {sc.get('c4',0):.2f}")
            st.progress(sc.get("c5", 0), text=f"Location {sc.get('c5',0):.2f}")
            st.progress(sc.get("c6", 0), text=f"Education {sc.get('c6',0):.2f}")
            st.metric("Final", f"{item['final_score']:.3f}")

    # Download button
    st.subheader("Download Results")
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    st.download_button(
        label="📥 Download ranked CSV",
        data=csv_buf.getvalue(),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )

    st.info(
        "**Note:** This sandbox runs without precomputed embeddings. "
        "The semantic narrative score (C1) uses a fixed neutral value (0.50) for speed. "
        "Full reproduction with embeddings available in the GitHub repo."
    )

else:
    st.info("Upload a JSONL file to begin ranking. You can use the `sample_candidates.json` from the hackathon bundle.")

    with st.expander("How the scoring works"):
        st.markdown("""
        | Component | Weight | What it measures |
        |-----------|--------|-----------------|
        | C1 Career Narrative (semantic) | 30% | Cosine similarity between career text and JD embedding |
        | C2 Production AI Experience | 20% | Real AI/ML work at product companies (not consulting) |
        | C3 Experience Seniority | 15% | Years in sweet spot (5-9 yrs preferred) |
        | C4 Skill Depth | 15% | Endorsement × duration trust multiplier on AI skills |
        | C5 Location & Logistics | 10% | City match, notice period, relocation willingness |
        | C6 Education + GitHub | 10% | Institution tier, field of study, GitHub activity |
        
        The base score is then multiplied by a **behavioral multiplier** (0.25–1.20) derived 
        from Redrob platform signals: recruiter response rate, days since last active, 
        open-to-work flag, interview completion rate, and profile completeness.
        
        **Anti-trap measures:** Consulting-only careers get 0.2× penalty. 
        Behavioral ghosts (inactive >90 days, <20% response rate) get heavy down-weighting. 
        Honeypot profiles (impossible duration math, unearned expert skills) score 0.
        """)
