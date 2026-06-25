"""
reasoning.py — Generate specific, non-hallucinated reasoning strings for each candidate.

The Stage 4 review checks:
1. Specific facts from the candidate's profile
2. JD connection
3. Honest concerns where applicable
4. No hallucination
5. Rank-consistent tone
"""

from __future__ import annotations
from datetime import date, datetime


def _parse_date(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _days_since(d) -> int:
    if d is None:
        return 9999
    return (date.today() - d).days


CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
}

def generate_reasoning(candidate: dict, rank: int, final_score: float, scores: dict) -> str:
    """Generate a 1-2 sentence reasoning string specific to this candidate."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    redrob = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    location = profile.get("location", "Unknown")

    rr = redrob.get("recruiter_response_rate", 0)
    last_active = _parse_date(redrob.get("last_active_date"))
    days_inactive = _days_since(last_active)
    notice = redrob.get("notice_period_days", 90)
    github = redrob.get("github_activity_score", -1)
    open_to_work = redrob.get("open_to_work_flag", False)

    # --- Build factual fragments ---
    parts = []

    # Core identity
    parts.append(f"{title} with {yoe:.1f} yrs total experience at {company}")

    # AI experience evidence (pick best career role)
    ai_roles = []
    for role in career:
        desc = role.get("description", "").lower()
        rtitle = role.get("title", "").lower()
        is_consulting = role.get("company", "").lower().strip() in CONSULTING_FIRMS
        ai_signals = ["embed", "vector", "retriev", "ranki", "recommend", "rag",
                      "fine-tun", "llm", "nlp", "search engine", "semantic"]
        if any(sig in desc or sig in rtitle for sig in ai_signals) and not is_consulting:
            ai_roles.append(role)

    if ai_roles:
        best_role = max(ai_roles, key=lambda r: r.get("duration_months", 0))
        btitle = best_role.get("title", "")
        bcomp = best_role.get("company", "")
        bdur = best_role.get("duration_months", 0)
        parts.append(f"hands-on AI/ML work as {btitle} at {bcomp} ({bdur} months)")

    # Top AI skills with evidence
    assessed = redrob.get("skill_assessment_scores", {})
    credible_ai_skills = []
    AI_KW = ["embed", "vector", "nlp", "rag", "fine-tun", "retriev", "lora",
             "ranking", "search", "bert", "faiss", "pinecone", "milvus",
             "sentence-transform", "recommend"]
    for s in skills:
        sname = s.get("name", "")
        sname_l = sname.lower()
        if not any(kw in sname_l for kw in AI_KW):
            continue
        endorse = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)
        if endorse >= 5 or dur >= 12:
            credible_ai_skills.append(sname)
    if credible_ai_skills:
        parts.append(f"credible AI skills: {', '.join(credible_ai_skills[:3])}")

    # Assessment scores
    if assessed:
        assess_str = "; ".join(f"{k}={v:.0f}" for k, v in list(assessed.items())[:2])
        parts.append(f"platform assessments: {assess_str}")

    # Location
    if "pune" in location.lower() or "noida" in location.lower():
        parts.append("Pune/Noida-based (preferred location)")
    elif "india" in profile.get("country", "").lower():
        parts.append(f"India-based ({location})")

    # Behavioral signals - positives
    if rr >= 0.6:
        parts.append(f"high recruiter engagement ({rr:.0%} response rate)")
    if notice <= 30:
        parts.append(f"available quickly ({notice}d notice)")
    if github > 50:
        parts.append(f"active GitHub ({github:.0f}/100)")

    # --- Build concerns (honest negatives) ---
    concerns = []

    if days_inactive > 90:
        concerns.append(f"inactive {days_inactive} days — reachability uncertain")
    if rr < 0.20:
        concerns.append(f"low response rate ({rr:.0%}) is a risk")
    if notice > 90:
        concerns.append(f"long notice period ({notice} days)")

    consulting_career = all(
        r.get("company", "").lower().strip() in CONSULTING_FIRMS
        for r in career
    )
    if consulting_career:
        concerns.append("consulting-only background (JD flags this as non-ideal)")

    if yoe < 4:
        concerns.append(f"below JD experience band ({yoe:.1f} yrs)")
    elif yoe > 12:
        concerns.append(f"above JD experience band ({yoe:.1f} yrs) — potential overqualification")

    if scores.get("honeypot", False):
        return "Flagged as likely honeypot — impossible profile signals detected; excluded from ranking."

    # --- Rank-appropriate tone ---
    if rank <= 10:
        opener = "Strong fit:"
    elif rank <= 30:
        opener = "Good fit:"
    elif rank <= 60:
        opener = "Moderate fit:"
    else:
        opener = "Weak fit:"

    sentence1 = f"{opener} {'; '.join(parts[:3])}."
    if len(parts) > 3:
        sentence1 += f" {'; '.join(parts[3:])}."

    if concerns:
        sentence2 = f"Concerns: {'; '.join(concerns[:2])}."
        return f"{sentence1} {sentence2}"

    return sentence1
