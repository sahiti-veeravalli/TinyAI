import pytest
from tinyai.optimization.pareto import CandidateEvaluation, ParetoAnalyzer


def test_pareto_analysis_dominance():
    # Cand A: High quality, higher latency, medium size
    cand_a = CandidateEvaluation(
        candidate_id="a",
        name="Candidate A",
        strategy="baseline",
        model_family="lightgbm",
        quality_metric_name="roc_auc",
        quality_score=0.92,
        latency_p50_ms=1.5,
        disk_size_kb=200.0,
        memory_rss_mb=50.0,
        compression_ratio=1.0,
        is_feasible=True
    )
    # Cand B: Slightly lower quality, but ultra-low latency and tiny size (non-dominated)
    cand_b = CandidateEvaluation(
        candidate_id="b",
        name="Candidate B",
        strategy="pruned",
        model_family="lightgbm",
        quality_metric_name="roc_auc",
        quality_score=0.90,
        latency_p50_ms=0.3,
        disk_size_kb=35.0,
        memory_rss_mb=40.0,
        compression_ratio=5.7,
        is_feasible=True
    )
    # Cand C: Strictly dominated by B (lower quality, higher latency, larger size than B)
    cand_c = CandidateEvaluation(
        candidate_id="c",
        name="Candidate C",
        strategy="slow_inferior",
        model_family="lightgbm",
        quality_metric_name="roc_auc",
        quality_score=0.85,
        latency_p50_ms=0.8,
        disk_size_kb=80.0,
        memory_rss_mb=45.0,
        compression_ratio=2.5,
        is_feasible=True
    )

    candidates = [cand_a, cand_b, cand_c]
    frontier = ParetoAnalyzer.extract_pareto_frontier(candidates, feasible_only=True)

    frontier_ids = [c.candidate_id for c in frontier]
    assert "a" in frontier_ids
    assert "b" in frontier_ids
    assert "c" not in frontier_ids  # Cand C must be dominated and excluded from frontier!


def test_feasibility_filtering():
    cand = CandidateEvaluation(
        candidate_id="x",
        name="Candidate X",
        strategy="heavy",
        model_family="mlp",
        quality_metric_name="roc_auc",
        quality_score=0.80,
        latency_p50_ms=3.5,
        disk_size_kb=500.0,
        memory_rss_mb=80.0,
        compression_ratio=1.0
    )
    constraints = {
        "max_disk_size_kb": 250.0,
        "max_latency_ms": 1.0,
        "min_quality_score": 0.85
    }
    is_feas, viols = ParetoAnalyzer.check_feasibility(cand, constraints)
    assert is_feas is False
    assert len(viols) == 3  # Violates all 3 constraints


def test_recommendation_ranking():
    cand1 = CandidateEvaluation(
        candidate_id="1",
        name="Balanced Model",
        strategy="quant",
        model_family="lightgbm",
        quality_metric_name="roc_auc",
        quality_score=0.91,
        latency_p50_ms=0.5,
        disk_size_kb=50.0,
        memory_rss_mb=40.0,
        compression_ratio=4.0,
        is_feasible=True
    )
    top, ranked = ParetoAnalyzer.rank_and_recommend([cand1], priority="balanced")
    assert top is not None
    assert top.candidate_id == "1"
    assert top.recommendation_rationale is not None
    assert "Selected under 'balanced' optimization strategy" in top.recommendation_rationale


def test_infeasible_candidate_ranking_honest_rationale():
    cand_viol = CandidateEvaluation(
        candidate_id="bad",
        name="Heavy Candidate",
        strategy="quant",
        model_family="lightgbm",
        quality_metric_name="roc_auc",
        quality_score=0.75,
        latency_p50_ms=5.0,
        disk_size_kb=800.0,
        memory_rss_mb=120.0,
        compression_ratio=1.0,
        is_feasible=False,
        constraint_violations=[
            "Disk size 800.0KB exceeds limit of 250.0KB",
            "Latency 5.000ms exceeds limit of 1.000ms"
        ]
    )
    top, ranked = ParetoAnalyzer.rank_and_recommend([cand_viol], priority="lowest_latency")
    assert top is not None
    assert top.is_feasible is False
    assert "No candidate met all hard operational constraints" in top.recommendation_rationale
    assert "Hard constraints were not relaxed" in top.recommendation_rationale
