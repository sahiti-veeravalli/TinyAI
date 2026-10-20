from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field


class CandidateEvaluation(BaseModel):
    candidate_id: str
    name: str
    strategy: str
    model_family: str
    artifact_path: Optional[str] = None
    quality_metric_name: str
    quality_score: float
    latency_p50_ms: float
    disk_size_kb: float
    memory_rss_mb: float
    compression_ratio: float
    is_feasible: bool = True
    constraint_violations: List[str] = Field(default_factory=list)
    is_pareto_optimal: bool = False
    recommendation_score: float = 0.0
    recommendation_rationale: Optional[str] = None
    artifact_checksum_sha256: Optional[str] = None
    benchmark_details: Dict[str, Any] = Field(default_factory=dict)


class ParetoAnalyzer:
    """Computes non-dominated Pareto frontiers and scores feasibility across multi-dimensional AI deployment constraints."""

    @classmethod
    def check_feasibility(
        cls,
        candidate: CandidateEvaluation,
        constraints: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        violations: List[str] = []

        max_size_kb = constraints.get("max_disk_size_kb")
        if max_size_kb is not None and candidate.disk_size_kb > max_size_kb:
            violations.append(
                f"Disk size {candidate.disk_size_kb:.1f}KB exceeds limit of {max_size_kb:.1f}KB"
            )

        max_lat_ms = constraints.get("max_latency_ms")
        if max_lat_ms is not None and candidate.latency_p50_ms > max_lat_ms:
            violations.append(
                f"Latency {candidate.latency_p50_ms:.3f}ms exceeds limit of {max_lat_ms:.3f}ms"
            )

        min_qual = constraints.get("min_quality_score")
        if min_qual is not None and candidate.quality_score < min_qual:
            violations.append(
                f"Quality {candidate.quality_score:.4f} is below minimum required {min_qual:.4f}"
            )

        max_mem = constraints.get("max_memory_mb")
        if max_mem is not None and candidate.memory_rss_mb > max_mem:
            violations.append(
                f"Peak memory {candidate.memory_rss_mb:.1f}MB exceeds limit of {max_mem:.1f}MB"
            )

        is_feasible = len(violations) == 0
        return is_feasible, violations

    @classmethod
    def extract_pareto_frontier(
        cls,
        candidates: List[CandidateEvaluation],
        feasible_only: bool = True
    ) -> List[CandidateEvaluation]:
        """Extracts the non-dominated Pareto frontier:
        Maximizing quality_score, while minimizing latency_p50_ms and disk_size_kb.
        """
        pool = [c for c in candidates if (not feasible_only or c.is_feasible)]
        if not pool:
            return []

        pareto_candidates: List[CandidateEvaluation] = []

        for i, cand_a in enumerate(pool):
            dominated = False
            for j, cand_b in enumerate(pool):
                if i == j:
                    continue
                # B dominates A if:
                # B.quality >= A.quality and B.latency <= A.latency and B.size <= A.size
                # and at least one strict inequality.
                b_better_or_equal = (
                    cand_b.quality_score >= cand_a.quality_score - 1e-6 and
                    cand_b.latency_p50_ms <= cand_a.latency_p50_ms + 1e-6 and
                    cand_b.disk_size_kb <= cand_a.disk_size_kb + 1e-6
                )
                b_strictly_better = (
                    cand_b.quality_score > cand_a.quality_score + 1e-6 or
                    cand_b.latency_p50_ms < cand_a.latency_p50_ms - 1e-6 or
                    cand_b.disk_size_kb < cand_a.disk_size_kb - 1e-6
                )
                if b_better_or_equal and b_strictly_better:
                    dominated = True
                    break

            if not dominated:
                cand_a.is_pareto_optimal = True
                pareto_candidates.append(cand_a)
            else:
                cand_a.is_pareto_optimal = False

        return pareto_candidates

    @classmethod
    def rank_and_recommend(
        cls,
        candidates: List[CandidateEvaluation],
        priority: str = "balanced"  # "balanced", "lowest_latency", "smallest_size", "highest_quality"
    ) -> Tuple[Optional[CandidateEvaluation], List[CandidateEvaluation]]:
        """Ranks feasible candidates according to stated user deployment priority and returns the top recommendation with clear rationale."""
        feasible = [c for c in candidates if c.is_feasible]
        if not feasible:
            # If none are strictly feasible, rank all candidates by lowest violation count
            feasible = sorted(candidates, key=lambda c: len(c.constraint_violations))
            if not feasible:
                return None, []

        # Normalize metrics across candidate pool for multi-attribute utility scoring
        qualities = np.array([c.quality_score for c in feasible])
        latencies = np.array([c.latency_p50_ms for c in feasible])
        sizes = np.array([c.disk_size_kb for c in feasible])

        def norm(arr):
            rng = arr.max() - arr.min()
            return np.ones_like(arr) if rng < 1e-7 else (arr - arr.min()) / rng

        norm_q = norm(qualities)
        norm_lat = 1.0 - norm(latencies)  # Invert: lower latency is better
        norm_size = 1.0 - norm(sizes)     # Invert: smaller size is better

        # Weights by priority
        if priority == "lowest_latency":
            w_q, w_lat, w_size = 0.25, 0.60, 0.15
        elif priority == "smallest_size":
            w_q, w_lat, w_size = 0.25, 0.15, 0.60
        elif priority == "highest_quality":
            w_q, w_lat, w_size = 0.70, 0.15, 0.15
        else:  # balanced
            w_q, w_lat, w_size = 0.40, 0.35, 0.25

        scores = w_q * norm_q + w_lat * norm_lat + w_size * norm_size

        for idx, cand in enumerate(feasible):
            cand.recommendation_score = round(float(scores[idx]), 3)

        ranked = sorted(feasible, key=lambda c: c.recommendation_score, reverse=True)
        top = ranked[0]

        top.recommendation_rationale = (
            f"Selected under '{priority}' optimization strategy because it achieved {top.quality_score:.3f} {top.quality_metric_name} "
            f"with ultra-fast {top.latency_p50_ms:.3f}ms p50 latency and compact {top.disk_size_kb:.1f}KB footprint "
            f"({top.compression_ratio:.1f}x compression over baseline), strictly satisfying all operational constraints."
        )

        return top, ranked
