"""
signals.py — Behavioral multiplier from Redrob platform signals.

Converts 23 raw signals into a single multiplier in [0.25, 1.20] that
modifies the base score.

final_score = base_score × behavioral_multiplier
"""

from __future__ import annotations
import math
from datetime import date, datetime


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _days_since(d: date | None) -> int:
    if d is None:
        return 9999
    return (date.today() - d).days


def compute_behavioral_multiplier(candidate: dict) -> float:
    """
    Returns a float in [0.25, 1.20].

    Signal groups:
      A. Availability (40%): open_to_work, recency of activity
      B. Engagement quality (40%): response rate, response time, interview completion
      C. Platform credibility (20%): profile completeness, verification, linkedin
    """
    r = candidate.get("redrob_signals", {})

    # ---- A. Availability (40%) ----

    # open_to_work
    open_to_work = 1.0 if r.get("open_to_work_flag", False) else 0.4

    # recency of last activity
    last_active = _parse_date(r.get("last_active_date"))
    days_inactive = _days_since(last_active)
    if days_inactive <= 7:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.9
    elif days_inactive <= 60:
        recency = 0.75
    elif days_inactive <= 90:
        recency = 0.55
    elif days_inactive <= 180:
        recency = 0.35
    else:
        recency = 0.15  # ghost candidate

    # applications submitted recently (proxy for active job search)
    apps_30d = r.get("applications_submitted_30d", 0)
    apps_score = min(1.0, 0.5 + 0.1 * apps_30d)  # baseline 0.5, bonus for activity

    availability_score = 0.35 * open_to_work + 0.45 * recency + 0.20 * apps_score

    # ---- B. Engagement quality (40%) ----

    # Recruiter response rate — most important engagement signal
    rr = r.get("recruiter_response_rate", 0.0)
    if rr < 0.05:
        rr_score = 0.1  # almost never responds — near-useless
    elif rr < 0.20:
        rr_score = 0.3
    elif rr < 0.40:
        rr_score = 0.55
    elif rr < 0.60:
        rr_score = 0.75
    elif rr < 0.80:
        rr_score = 0.90
    else:
        rr_score = 1.0

    # Response time (lower = better)
    avg_rt = r.get("avg_response_time_hours", 168)
    if avg_rt <= 4:
        rt_score = 1.0
    elif avg_rt <= 12:
        rt_score = 0.9
    elif avg_rt <= 24:
        rt_score = 0.8
    elif avg_rt <= 48:
        rt_score = 0.65
    elif avg_rt <= 96:
        rt_score = 0.45
    else:
        rt_score = 0.25

    # Interview completion rate
    icr = r.get("interview_completion_rate", 0.5)
    icr_score = icr  # already 0-1

    # Offer acceptance rate (skip if -1)
    oar = r.get("offer_acceptance_rate", -1)
    if oar < 0:
        oar_score = 0.5  # neutral
    elif oar > 0.7:
        oar_score = 1.0
    elif oar > 0.4:
        oar_score = 0.7
    else:
        oar_score = 0.4

    engagement_score = (
        0.45 * rr_score
        + 0.20 * rt_score
        + 0.20 * icr_score
        + 0.15 * oar_score
    )

    # ---- C. Platform credibility (20%) ----
    completeness = r.get("profile_completeness_score", 50) / 100
    verified_email = 1.0 if r.get("verified_email", False) else 0.0
    verified_phone = 1.0 if r.get("verified_phone", False) else 0.0
    linkedin = 0.8 if r.get("linkedin_connected", False) else 0.5

    credibility_score = (
        0.40 * completeness
        + 0.25 * verified_email
        + 0.20 * verified_phone
        + 0.15 * linkedin
    )

    # ---- Combine ----
    raw = (
        0.40 * availability_score
        + 0.40 * engagement_score
        + 0.20 * credibility_score
    )

    # Map 0-1 raw score to multiplier range [0.25, 1.20]
    # 0.0 raw → 0.25 multiplier (ghost, unverified, no response)
    # 0.5 raw → ~0.70 (average)
    # 1.0 raw → 1.20 (perfect engagement)
    multiplier = 0.25 + raw * 0.95

    return float(round(multiplier, 4))


def compute_final_score(base_score: float, candidate: dict) -> float:
    mult = compute_behavioral_multiplier(candidate)
    final = base_score * mult
    return float(round(min(1.0, final), 6))
