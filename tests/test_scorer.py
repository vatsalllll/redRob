"""
tests/test_scorer.py — Unit tests for scoring components.

Run: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scorer import (
    score_production_ai_experience,
    score_experience_seniority,
    score_skill_depth,
    score_location_logistics,
    score_education_github,
    is_honeypot,
    compute_base_score,
    CONSULTING_FIRMS,
)
from signals import compute_behavioral_multiplier, compute_final_score


# ---- Fixtures ----

def make_candidate(
    current_title="ML Engineer",
    current_company="Startup AI",
    years_of_experience=7.0,
    location="Pune, Maharashtra",
    country="India",
    career_history=None,
    skills=None,
    education=None,
    redrob_signals=None,
):
    base_signals = {
        "profile_completeness_score": 80,
        "signup_date": "2025-01-01",
        "last_active_date": "2026-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 20,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.7,
        "avg_response_time_hours": 12,
        "skill_assessment_scores": {},
        "connection_count": 300,
        "endorsements_received": 50,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20, "max": 35},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 55,
        "search_appearance_30d": 100,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.85,
        "offer_acceptance_rate": 0.7,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    if redrob_signals:
        base_signals.update(redrob_signals)

    return {
        "candidate_id": "CAND_TEST",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": f"{current_title} at {current_company}",
            "summary": "Test candidate summary.",
            "location": location,
            "country": country,
            "years_of_experience": years_of_experience,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": "201-500",
            "current_industry": "Technology",
        },
        "career_history": career_history or [
            {
                "company": current_company,
                "title": current_title,
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Technology",
                "company_size": "201-500",
                "description": "Built production embedding-based retrieval system with Pinecone and sentence-transformers. Designed evaluation framework with NDCG and MRR metrics. Deployed hybrid search to real users.",
            }
        ],
        "education": education or [
            {
                "institution": "IIT Delhi",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2013,
                "end_year": 2017,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": skills or [
            {"name": "NLP", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
            {"name": "Vector Search", "proficiency": "advanced", "endorsements": 18, "duration_months": 30},
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 10, "duration_months": 24},
            {"name": "Sentence-Transformers", "proficiency": "advanced", "endorsements": 12, "duration_months": 20},
            {"name": "FAISS", "proficiency": "intermediate", "endorsements": 8, "duration_months": 18},
        ],
        "redrob_signals": base_signals,
    }


# ---- Tests: Production AI Experience ----

class TestProductionAIExperience:

    def test_ideal_ai_engineer(self):
        cand = make_candidate(current_title="ML Engineer")
        score = score_production_ai_experience(cand)
        assert score > 0.65, f"Expected > 0.65, got {score}"

    def test_marketing_manager_with_ai_skills(self):
        """Should be heavily penalized — this is a trap candidate."""
        cand = make_candidate(
            current_title="Marketing Manager",
            career_history=[{
                "company": "Some Corp",
                "title": "Marketing Manager",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Marketing",
                "company_size": "201-500",
                "description": "Managed marketing campaigns, ran social media, content strategy.",
            }]
        )
        score = score_production_ai_experience(cand)
        assert score < 0.30, f"Marketing manager should score < 0.30, got {score}"

    def test_consulting_lifer(self):
        """Full career at Wipro should be penalized."""
        cand = make_candidate(
            current_company="Wipro",
            career_history=[
                {
                    "company": "Wipro",
                    "title": "Senior Developer",
                    "start_date": "2018-01-01",
                    "end_date": None,
                    "duration_months": 96,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Development of enterprise applications.",
                }
            ]
        )
        score = score_production_ai_experience(cand)
        assert score <= 0.20, f"Consulting lifer should score <= 0.20, got {score}"

    def test_consulting_with_prior_product_experience(self):
        """Currently at Wipro but has prior product-company AI experience — should NOT be disqualified."""
        cand = make_candidate(
            current_company="Wipro",
            career_history=[
                {
                    "company": "Wipro",
                    "title": "Senior Engineer",
                    "start_date": "2023-01-01",
                    "end_date": None,
                    "duration_months": 18,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Enterprise application development.",
                },
                {
                    "company": "Razorpay",
                    "title": "ML Engineer",
                    "start_date": "2019-01-01",
                    "end_date": "2022-12-31",
                    "duration_months": 48,
                    "is_current": False,
                    "industry": "Fintech",
                    "company_size": "1001-5000",
                    "description": "Built recommendation and ranking systems using embeddings and hybrid retrieval.",
                },
            ]
        )
        score = score_production_ai_experience(cand)
        # Should get partial-to-good credit for the prior product experience
        assert score > 0.25, f"Consulting + prior product should score > 0.25, got {score}"


# ---- Tests: Experience Seniority ----

class TestExperienceSeniority:

    def test_sweet_spot(self):
        for yoe in [5.0, 6.5, 7.0, 8.0, 9.0]:
            score = score_experience_seniority({"profile": {"years_of_experience": yoe}})
            assert score == 1.0, f"yoe={yoe} should score 1.0, got {score}"

    def test_too_junior(self):
        score = score_experience_seniority({"profile": {"years_of_experience": 1.0}})
        assert score < 0.50, f"1 year should score < 0.50, got {score}"

    def test_overqualified(self):
        score = score_experience_seniority({"profile": {"years_of_experience": 18.0}})
        assert score < 0.80, f"18 years should score < 0.80, got {score}"
        assert score >= 0.60, f"18 years should not be completely penalized, got {score}"


# ---- Tests: Skill Depth ----

class TestSkillDepth:

    def test_trusted_ai_skills(self):
        """Skills with endorsements and duration should score well."""
        cand = make_candidate()
        score = score_skill_depth(cand)
        assert score > 0.4, f"Should score > 0.4, got {score}"

    def test_zero_trust_skills(self):
        """Expert skills with 0 endorsements and 0 duration are suspect."""
        cand = make_candidate(
            skills=[
                {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Weaviate", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Milvus", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            ]
        )
        score = score_skill_depth(cand)
        trusted_cand = make_candidate()
        trusted_score = score_skill_depth(trusted_cand)
        assert score < trusted_score, "Unendorsed 'expert' skills should score lower than endorsed skills"


# ---- Tests: Location ----

class TestLocation:

    def test_pune(self):
        cand = make_candidate(location="Pune, Maharashtra", country="India")
        score = score_location_logistics(cand)
        assert score > 0.7, f"Pune should score > 0.7, got {score}"

    def test_outside_india(self):
        cand = make_candidate(location="San Francisco, CA", country="United States",
                              redrob_signals={"willing_to_relocate": False, "notice_period_days": 60})
        score = score_location_logistics(cand)
        assert score < 0.5, f"Outside India should score < 0.5, got {score}"


# ---- Tests: Honeypot Detection ----

class TestHoneypot:

    def test_impossible_duration(self):
        """Career months >> stated years of experience."""
        cand = make_candidate(
            years_of_experience=3.0,
            career_history=[
                {"company": "A", "title": "Engineer", "start_date": "2015-01-01",
                 "end_date": "2020-12-31", "duration_months": 72, "is_current": False,
                 "industry": "Tech", "company_size": "201-500", "description": "work"},
                {"company": "B", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "201-500", "description": "work"},
            ]
        )
        # 132 career months vs 36 stated months — should flag
        assert is_honeypot(cand), "Should detect impossible career duration"

    def test_expert_skill_abuse(self):
        """Many expert skills with 0 endorsements and 0 duration."""
        cand = make_candidate(
            skills=[
                {"name": s, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
                for s in ["NLP", "Pinecone", "FAISS", "Weaviate", "Milvus",
                           "LoRA", "RAG", "Fine-tuning LLMs", "BM25", "Ranking"]
            ]
        )
        assert is_honeypot(cand), "Should detect expert skill abuse"

    def test_normal_candidate_not_flagged(self):
        cand = make_candidate()
        assert not is_honeypot(cand), "Normal candidate should not be flagged as honeypot"


# ---- Tests: Behavioral Signals ----

class TestBehavioralSignals:

    def test_ghost_candidate(self):
        cand = make_candidate(redrob_signals={
            "profile_completeness_score": 40,
            "last_active_date": "2025-01-01",  # ~17 months ago
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.03,
            "avg_response_time_hours": 500,
            "interview_completion_rate": 0.2,
            "offer_acceptance_rate": -1,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
            "applications_submitted_30d": 0,
            "notice_period_days": 120,
            "willing_to_relocate": False,
            "github_activity_score": -1,
        })
        mult = compute_behavioral_multiplier(cand)
        assert mult < 0.60, f"Ghost should have multiplier < 0.60, got {mult}"

    def test_engaged_candidate(self):
        mult = compute_behavioral_multiplier(make_candidate())
        assert mult > 0.80, f"Engaged candidate should have multiplier > 0.80, got {mult}"


# ---- Integration test ----

class TestIntegration:

    def test_ideal_beats_keyword_stuffer(self):
        """A real AI engineer should rank above a Marketing Manager with AI keywords."""
        ideal = make_candidate()
        stuffer = make_candidate(
            current_title="Marketing Manager",
            career_history=[{
                "company": "Corp",
                "title": "Marketing Manager",
                "start_date": "2018-01-01",
                "end_date": None,
                "duration_months": 84,
                "is_current": True,
                "industry": "Marketing",
                "company_size": "201-500",
                "description": "Marketing campaigns. Used ChatGPT for content.",
            }],
            skills=[
                {"name": "NLP", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Fine-tuning LLMs", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "RAG", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Vector Search", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "NDCG", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                {"name": "Weaviate", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            ]
        )

        cosine_sim = 0.5  # neutral
        ideal_scores = compute_base_score(ideal, cosine_sim)
        stuffer_scores = compute_base_score(stuffer, cosine_sim)

        ideal_final = compute_final_score(ideal_scores["base_score"], ideal)
        stuffer_final = compute_final_score(stuffer_scores["base_score"], stuffer)

        print(f"Ideal final: {ideal_final:.4f}, Stuffer final: {stuffer_final:.4f}")
        assert ideal_final > stuffer_final, (
            f"Real engineer ({ideal_final:.4f}) should beat keyword stuffer ({stuffer_final:.4f})"
        )


# ---- New tests: edge cases and gaps ----

class TestTiedScores:
    def test_tied_final_scores_tiebreak_by_candidate_id(self):
        from rank import rank_candidates
        import numpy as np
        c1 = make_candidate()
        c1["candidate_id"] = "CAND_9999999"
        c2 = make_candidate()
        c2["candidate_id"] = "CAND_1111111"
        jd_emb = np.zeros(384)
        cand_emb = np.zeros((2, 384))
        emb_ids = ["CAND_9999999", "CAND_1111111"]
        results = rank_candidates([c1, c2], jd_emb, cand_emb, emb_ids, verbose=False)
        assert results[0]["candidate_id"] == "CAND_1111111", "Lower ID should win tie-break"

    def test_rounded_score_tiebreak(self):
        from rank import rank_candidates
        import numpy as np
        c1 = make_candidate()
        c1["candidate_id"] = "CAND_0000005"
        c2 = make_candidate()
        c2["candidate_id"] = "CAND_0000003"
        jd_emb = np.zeros(384)
        cand_emb = np.zeros((2, 384))
        emb_ids = ["CAND_0000005", "CAND_0000003"]
        results = rank_candidates([c1, c2], jd_emb, cand_emb, emb_ids, verbose=False)
        assert results[0]["candidate_id"] in ("CAND_0000003", "CAND_0000005")


class TestMissingFields:
    def test_no_education(self):
        cand = make_candidate()
        cand["education"] = []
        score = score_education_github(cand)
        assert 0.2 <= score <= 0.5, f"No-education should score 0.2-0.5, got {score}"

    def test_no_skills(self):
        cand = make_candidate()
        cand["skills"] = []
        score = score_skill_depth(cand)
        assert score == 0.0, f"No skills should score 0.0, got {score}"

    def test_empty_career_history(self):
        cand = make_candidate()
        cand["career_history"] = []
        score = score_production_ai_experience(cand)
        assert score <= 0.2, f"Empty career should score <= 0.2, got {score}"


class TestBehavioralBoundary:
    def test_minimum_multiplier(self):
        cand = make_candidate(redrob_signals={
            "recruiter_response_rate": 0.0,
            "avg_response_time_hours": 999,
            "profile_completeness_score": 0,
            "last_active_date": "2020-01-01",
            "open_to_work_flag": False,
            "interview_completion_rate": 0.0,
            "offer_acceptance_rate": -1,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
            "applications_submitted_30d": 0,
            "notice_period_days": 180,
            "willing_to_relocate": False,
            "github_activity_score": -1,
        })
        mult = compute_behavioral_multiplier(cand)
        assert 0.40 <= mult <= 0.60, f"Minimum ghost should be near 0.40, got {mult}"

    def test_maximum_multiplier(self):
        cand = make_candidate(redrob_signals={
            "recruiter_response_rate": 0.95,
            "avg_response_time_hours": 2,
            "profile_completeness_score": 100,
            "last_active_date": "2026-06-25",
            "open_to_work_flag": True,
            "interview_completion_rate": 1.0,
            "offer_acceptance_rate": 0.9,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "applications_submitted_30d": 5,
        })
        mult = compute_behavioral_multiplier(cand)
        assert 1.05 <= mult <= 1.10, f"Maximum engaged should be near 1.10, got {mult}"


class TestHoneypotExpanded:
    def test_future_start_date(self):
        cand = make_candidate(
            career_history=[{
                "company": "Future Corp", "title": "AI Engineer",
                "start_date": "2099-01-01", "end_date": None,
                "duration_months": 12, "is_current": True,
                "industry": "Tech", "company_size": "201-500",
                "description": "work",
            }]
        )
        assert is_honeypot(cand), "Future start_date should be flagged"

    def test_skill_duration_exceeds_yoe(self):
        cand = make_candidate(
            years_of_experience=2.0,
            skills=[
                {"name": "Python", "proficiency": "expert",
                 "endorsements": 5, "duration_months": 120},
            ]
        )
        assert is_honeypot(cand), "Skill duration > YOE should be flagged"

    def test_salary_min_greater_than_max(self):
        cand = make_candidate(redrob_signals={
            "expected_salary_range_inr_lpa": {"min": 50, "max": 10}
        })
        assert is_honeypot(cand), "Salary min > max should be flagged"

    def test_empty_profile(self):
        cand = make_candidate(redrob_signals={"profile_completeness_score": 2})
        assert is_honeypot(cand), "Profile completeness < 5 should be flagged"

    def test_overlapping_roles(self):
        cand = make_candidate(
            career_history=[
                {"company": "A", "title": "Engineer", "start_date": "2020-01-01",
                 "end_date": "2023-01-01", "duration_months": 36, "is_current": False,
                 "industry": "Tech", "company_size": "201-500", "description": "work"},
                {"company": "B", "title": "Engineer", "start_date": "2022-06-01",
                 "end_date": None, "duration_months": 30, "is_current": True,
                 "industry": "Tech", "company_size": "201-500", "description": "work"},
            ]
        )
        assert is_honeypot(cand), "Overlapping roles at different companies should be flagged"


class TestSeniorityBoundary:
    def test_exact_sweet_spot_boundaries(self):
        for yoe in [4.99, 5.0, 9.0, 9.01]:
            s = score_experience_seniority({"profile": {"years_of_experience": yoe}})
            if yoe == 5.0 or yoe == 9.0:
                assert s == 1.0, f"yoe={yoe} should be 1.0, got {s}"
            else:
                assert s < 1.0, f"yoe={yoe} (outside band) should be < 1.0, got {s}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
