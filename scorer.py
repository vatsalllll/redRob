"""
scorer.py — Multi-signal scoring engine for Redrob Senior AI Engineer JD.

6 components:
  C1  career_narrative_fit       30%  (semantic embedding similarity)
  C2  production_ai_experience   20%  (rule-based, title+company+description)
  C3  experience_seniority       15%  (years, AI-specific years)
  C4  skill_depth                15%  (endorsement x duration trust multiplier)
  C5  location_logistics         10%  (city, notice period, relocation)
  C6  education_github           10%  (tier, field, github activity)

Final = base_score × behavioral_multiplier  (from signals.py)
"""

from __future__ import annotations
import re
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "hexaware", "niit technologies", "l&t infotech",
    "larsen & toubro infotech", "mindtree", "mindtree ltd",
}

PREFERRED_LOCATIONS = {
    "pune", "noida",
}
OK_LOCATIONS = {
    "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "gurgaon",
    "gurugram", "chennai", "kolkata", "delhi ncr", "ncr",
}

# Core skills the JD actually requires (production IR/ML work)
CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "vector search", "vector database",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
    "opensearch", "hybrid search", "dense retrieval", "sparse retrieval",
    "bm25", "retrieval", "ranking", "learning to rank", "ltr",
    "ndcg", "mrr", "map", "a/b testing", "information retrieval",
    "nlp", "natural language processing", "fine-tuning", "fine-tuning llms",
    "lora", "qlora", "peft", "rag", "retrieval augmented generation",
    "transformer", "bert", "sentence bert", "bge", "e5",
    "recommendation system", "recommender system", "search",
    "xgboost ranking", "lightgbm", "neural ranking",
}

# Skills that are red flags by themselves (framework tourists)
TOURIST_SKILLS = {
    "langchain", "llamaindex", "chatgpt api", "openai api",
}

AI_JD_KEYWORDS = [
    r"\bembed(ding|dings)?\b",
    r"\bvector (search|database|db|store|index)\b",
    r"\bhybrid (search|retrieval)\b",
    r"\bdense (retrieval|search)\b",
    r"\b(semantic|neural) (search|ranking|retrieval)\b",
    r"\b(sentence[- ]?transformer|sbert|bge|e5)\b",
    r"\b(pinecone|weaviate|qdrant|milvus|faiss|opensearch|elasticsearch)\b",
    r"\b(fine[- ]?tun|lora|qlora|peft)\b",
    r"\b(rag|retrieval[- ]?augmented)\b",
    r"\b(recommend(ation|er)?|ranking|ranker)\b",
    r"\b(ndcg|mrr|recall@|map|mean average precision)\b",
    r"\b(a/b test|offline eval|online eval|evaluation framework)\b",
    r"\b(production (ml|ai|model|pipeline|inference))\b",
    r"\b(information retrieval|ir )\b",
    r"\b(learning[- ]?to[- ]?rank|ltr)\b",
]

RESEARCH_KEYWORDS = [
    r"\b(phd|ph\.d|research (scientist|engineer|intern|fellow))\b",
    r"\b(paper|conference|workshop|arxiv|neurips|icml|iclr|cvpr|acl|emnlp)\b",
    r"\b(academic|university lab|research lab)\b",
]

AI_TITLES = {
    "ml engineer", "machine learning engineer", "senior ml engineer",
    "staff ml engineer", "ai engineer", "applied ml engineer",
    "applied scientist", "data scientist", "nlp engineer", "search engineer",
    "ranking engineer", "recommendation engineer", "research engineer",
    "senior ai engineer", "principal ml engineer", "junior ml engineer",
    "deep learning engineer",
}

# Titles that are definitely not AI Engineer fits
NON_AI_TITLES = {
    "marketing manager", "hr manager", "human resources", "content writer",
    "graphic designer", "sales executive", "accountant", "civil engineer",
    "mechanical engineer", "operations manager", "customer support",
    "project manager", "business analyst",
}

EDUCATION_TIER_SCORE = {
    "tier_1": 1.0,
    "tier_2": 0.8,
    "tier_3": 0.6,
    "tier_4": 0.35,
    "unknown": 0.5,
}

AI_FIELDS = {
    "computer science", "computer engineering", "information technology",
    "electronics", "electrical engineering", "mathematics", "statistics",
    "data science", "artificial intelligence", "machine learning",
    "computational linguistics",
}


def _now() -> date:
    return date.today()


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _norm(text: str) -> str:
    return text.lower().strip()


def _is_consulting_company(company: str) -> bool:
    return _norm(company) in CONSULTING_FIRMS


def _career_text(candidate: dict) -> str:
    parts = []
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
        parts.append(role.get("company", ""))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# C1: Career narrative fit  (embedding-based; score injected from outside)
# ---------------------------------------------------------------------------

def score_career_narrative(cosine_sim: float) -> float:
    """Pass in cosine_similarity(jd_emb, candidate_emb). Range 0-1."""
    return float(max(0.0, min(1.0, cosine_sim)))


# ---------------------------------------------------------------------------
# C2: Production AI experience
# ---------------------------------------------------------------------------

def score_production_ai_experience(candidate: dict) -> float:
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # --- Hard penalty: consulting-only career ---
    non_consulting_roles = [
        r for r in career if not _is_consulting_company(r.get("company", ""))
    ]
    all_consulting = len(career) > 0 and len(non_consulting_roles) == 0
    if all_consulting:
        return 0.15  # hard floor for consulting lifers

    # --- Current title signal ---
    current_title = _norm(profile.get("current_title", ""))
    title_in_ai = any(t in current_title for t in AI_TITLES)
    title_non_ai = any(t in current_title for t in NON_AI_TITLES)

    if title_non_ai and not title_in_ai:
        title_score = 0.1
    elif title_in_ai:
        title_score = 1.0
    else:
        title_score = 0.4  # ambiguous

    # --- Career history analysis ---
    ai_months = 0
    research_penalty = 0.0
    description_score = 0.0
    langchain_only_flag = True

    for role in career:
        title = _norm(role.get("title", ""))
        desc = _norm(role.get("description", ""))
        company = role.get("company", "")
        duration = role.get("duration_months", 0)
        start = _parse_date(role.get("start_date"))

        is_consulting = _is_consulting_company(company)

        # AI role detection
        role_is_ai = any(t in title for t in AI_TITLES)
        if role_is_ai and not is_consulting:
            ai_months += duration

        # Production AI keywords in description
        ai_kw_hits = sum(
            1 for pattern in AI_JD_KEYWORDS
            if re.search(pattern, desc, re.IGNORECASE)
        )
        if ai_kw_hits > 0 and not is_consulting:
            description_score += min(1.0, ai_kw_hits / 5) * min(1.0, duration / 24)
            # Check if this pre-dates the LLM era (before 2022)
            if start and start.year < 2022:
                langchain_only_flag = False
        elif ai_kw_hits > 0:
            # Consulting with AI keywords — partial credit
            description_score += min(0.3, ai_kw_hits / 10) * min(1.0, duration / 24)

        # Research penalty
        research_hits = sum(
            1 for pat in RESEARCH_KEYWORDS
            if re.search(pat, desc + " " + title, re.IGNORECASE)
        )
        if research_hits >= 2:
            research_penalty += 0.1

    description_score = min(1.0, description_score)

    # --- LangChain-only tourists: if all AI exp is post-2022 & < 24 months ---
    total_ai_months = ai_months
    langchain_penalty = 0.7 if (langchain_only_flag and total_ai_months < 24) else 1.0

    # --- Combine ---
    raw = (
        0.25 * title_score
        + 0.50 * description_score
        + 0.25 * min(1.0, total_ai_months / 48)  # 4 years of AI = max
    )
    raw -= min(0.3, research_penalty)
    raw = max(0.0, raw)
    return float(raw * langchain_penalty)


# ---------------------------------------------------------------------------
# C3: Experience seniority
# ---------------------------------------------------------------------------

def score_experience_seniority(candidate: dict) -> float:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0.0)

    # Peak band: 5-9 years
    if 5.0 <= yoe <= 9.0:
        return 1.0
    elif yoe < 3.0:
        return 0.2 + 0.1 * yoe  # 0 yrs → 0.2, 3 yrs → 0.5
    elif 3.0 <= yoe < 5.0:
        return 0.5 + 0.25 * (yoe - 3.0)  # ramp up 0.5→1.0
    elif 9.0 < yoe <= 12.0:
        return 1.0 - 0.05 * (yoe - 9.0)  # slight decline
    else:  # > 12 years
        return max(0.6, 1.0 - 0.04 * (yoe - 9.0))


# ---------------------------------------------------------------------------
# C4: Skill depth with trust multiplier
# ---------------------------------------------------------------------------

def score_skill_depth(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})
    assessment_scores = redrob.get("skill_assessment_scores", {})

    PROFICIENCY_W = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}

    total_score = 0.0
    ai_skill_count = 0
    tourist_count = 0

    for skill in skills:
        name = _norm(skill.get("name", ""))
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # Tourist skill flag
        if any(ts in name for ts in TOURIST_SKILLS):
            tourist_count += 1
            continue

        is_ai_skill = any(kw in name for kw in CORE_AI_SKILLS) or \
                      any(re.search(pat, name) for pat in AI_JD_KEYWORDS)

        if not is_ai_skill:
            continue

        ai_skill_count += 1
        prof_w = PROFICIENCY_W.get(proficiency, 0.25)
        # Trust multiplier: requires endorsements AND duration
        endorse_trust = min(1.0, endorsements / 15)
        duration_trust = min(1.0, duration / 18)
        trust = 0.3 + 0.7 * (0.5 * endorse_trust + 0.5 * duration_trust)

        skill_score = prof_w * trust

        # Bonus if assessment score exists for this skill
        for assess_name, assess_score in assessment_scores.items():
            if _norm(assess_name) in name or name in _norm(assess_name):
                skill_score *= (0.7 + 0.3 * assess_score / 100)
                break

        total_score += skill_score

    # Penalize if mostly tourists
    if tourist_count > ai_skill_count and ai_skill_count < 2:
        total_score *= 0.5

    # Normalize: 5 strong AI skills = ~1.0
    return float(min(1.0, total_score / 4.0))


# ---------------------------------------------------------------------------
# C5: Location & logistics
# ---------------------------------------------------------------------------

def score_location_logistics(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    redrob = candidate.get("redrob_signals", {})

    location = _norm(profile.get("location", ""))
    country = _norm(profile.get("country", ""))
    willing_to_relocate = redrob.get("willing_to_relocate", False)
    notice_period = redrob.get("notice_period_days", 90)

    # Location score
    if any(city in location for city in PREFERRED_LOCATIONS):
        loc_score = 1.0
    elif any(city in location for city in OK_LOCATIONS):
        loc_score = 0.8
    elif country == "india":
        loc_score = 0.6
    elif willing_to_relocate and country == "india":
        loc_score = 0.65
    elif willing_to_relocate:
        loc_score = 0.45
    else:
        loc_score = 0.3  # outside India, won't relocate

    # Notice period score
    if notice_period <= 15:
        np_score = 1.0
    elif notice_period <= 30:
        np_score = 0.9
    elif notice_period <= 60:
        np_score = 0.65
    elif notice_period <= 90:
        np_score = 0.45
    else:
        np_score = 0.2

    return float(0.6 * loc_score + 0.4 * np_score)


# ---------------------------------------------------------------------------
# C6: Education + GitHub
# ---------------------------------------------------------------------------

def score_education_github(candidate: dict) -> float:
    education = candidate.get("education", [])
    redrob = candidate.get("redrob_signals", {})
    github_score = redrob.get("github_activity_score", -1)

    edu_score = 0.0
    if education:
        best = max(
            education,
            key=lambda e: EDUCATION_TIER_SCORE.get(e.get("tier", "unknown"), 0.5),
        )
        edu_score = EDUCATION_TIER_SCORE.get(best.get("tier", "unknown"), 0.5)
        # Field bonus
        field = _norm(best.get("field_of_study", ""))
        if any(f in field for f in AI_FIELDS):
            edu_score = min(1.0, edu_score + 0.1)
    else:
        edu_score = 0.4  # no education listed

    # GitHub
    if github_score < 0:
        gh_score = 0.3  # no GitHub linked
    else:
        gh_score = github_score / 100

    return float(0.6 * edu_score + 0.4 * gh_score)


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

def is_honeypot(candidate: dict) -> bool:
    """Returns True if candidate shows impossible/fake-looking signals."""
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})

    # Check 1: total career months vs stated years_of_experience
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    if total_career_months > (yoe * 12 + 30):
        return True

    # Check 2: expert skill with 0 duration AND 0 endorsements (multiple)
    suspicious_skills = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 0) == 0
        and s.get("endorsements", 0) == 0
    )
    if suspicious_skills >= 3:
        return True

    # Check 3: skill duration exceeds stated years_of_experience
    max_skill_months = max((s.get("duration_months", 0) for s in skills), default=0)
    if max_skill_months > (yoe * 12 + 6):
        return True

    # Check 4: start_date in the future
    from datetime import date
    today = date.today()
    for r in career:
        start = _parse_date(r.get("start_date"))
        if start and start > today:
            return True
        end = _parse_date(r.get("end_date"))
        if end and end > today:
            return True

    # Check 5: salary range with min > max
    sal = redrob.get("expected_salary_range_inr_lpa", {})
    if isinstance(sal, dict) and sal.get("min", 0) > sal.get("max", 0):
        return True

    # Check 6: profile completeness < 5 (effectively empty)
    completeness = redrob.get("profile_completeness_score", 50)
    if completeness < 5:
        return True

    # Check 7: overlapping roles at different companies (same time period)
    if len(career) >= 2:
        intervals = []
        for r in career:
            s = _parse_date(r.get("start_date"))
            e = _parse_date(r.get("end_date")) or today
            if s:
                intervals.append((s, e, r.get("company", "")))
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                return True

    return False


# ---------------------------------------------------------------------------
# Combined base score (without behavioral multiplier)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "career_narrative": 0.30,
    "production_ai":    0.20,
    "seniority":        0.15,
    "skill_depth":      0.15,
    "location":         0.10,
    "education":        0.10,
}


def compute_base_score(candidate: dict, cosine_sim: float) -> dict:
    """
    Returns a dict with all component scores and the final base_score.
    cosine_sim: precomputed cosine similarity between JD embedding and candidate
    """
    if is_honeypot(candidate):
        return {
            "honeypot": True,
            "base_score": 0.0,
            "c1": 0.0, "c2": 0.0, "c3": 0.0,
            "c4": 0.0, "c5": 0.0, "c6": 0.0,
        }

    c1 = score_career_narrative(cosine_sim)
    c2 = score_production_ai_experience(candidate)
    c3 = score_experience_seniority(candidate)
    c4 = score_skill_depth(candidate)
    c5 = score_location_logistics(candidate)
    c6 = score_education_github(candidate)

    # Career plausibility guard: if a candidate has no real AI career (C2 < 0.25),
    # their listed AI skills are suspect keyword stuffing. Cap C4 proportionally.
    # This is the key anti-trap measure for "Graphic Designer with Pinecone skills."
    if c2 < 0.25:
        c4 = c4 * max(0.25, c2 / 0.25)

    base = (
        WEIGHTS["career_narrative"] * c1
        + WEIGHTS["production_ai"] * c2
        + WEIGHTS["seniority"] * c3
        + WEIGHTS["skill_depth"] * c4
        + WEIGHTS["location"] * c5
        + WEIGHTS["education"] * c6
    )

    return {
        "honeypot": False,
        "base_score": round(float(base), 6),
        "c1": round(c1, 4),
        "c2": round(c2, 4),
        "c3": round(c3, 4),
        "c4": round(c4, 4),
        "c5": round(c5, 4),
        "c6": round(c6, 4),
    }
