"""Candidate per-scan uncertainty signal registry + scaling transforms.

Each candidate is a per-scan scalar derived from the v5 H5
``/uncertainty/`` group (post-Stage-0 repair). The registry maps a code
name to a function ``builder(uncertainty_group, vol_mean) -> np.ndarray``
of shape (n_scans,). The runner then selects per-scan values for a
specific patient/timepoint pair via the join keys in
``candidate_signals.csv``.

Scaling transforms convert a candidate to the "variance on log-volume"
scale before injection into the LMEHetero model so widths are
comparable across candidates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpec:
    """One per-scan uncertainty signal.

    Attributes:
        name: Code name (matches config and output paths).
        family: Taxonomy tag for plotting / discussion.
        builder: Function taking the H5 ``uncertainty/`` group (as a
            mapping from name → np.ndarray of shape (n_scans,)) and
            returning the per-scan candidate vector.
        description: One-line human-readable description.
    """

    name: str
    family: str
    builder: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    description: str


def _logvol_var(u: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray(u["logvol_std"], dtype=np.float64) ** 2


def _logvol_mad_var(u: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray(u["logvol_mad_scaled"], dtype=np.float64) ** 2


def _vol_cv2(u: Mapping[str, np.ndarray]) -> np.ndarray:
    vmean = np.asarray(u["vol_mean"], dtype=np.float64)
    vstd = np.asarray(u["vol_std"], dtype=np.float64)
    cv = vstd / np.maximum(vmean, 1.0)  # treat empty masks as no-information
    return cv**2


def _passthrough(key: str) -> Callable[[Mapping[str, np.ndarray]], np.ndarray]:
    def _b(u: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(u[key], dtype=np.float64)

    return _b


def _composite_logvol_x_boundary_entropy(
    u: Mapping[str, np.ndarray], beta: float = 1.0
) -> np.ndarray:
    """σ²_{log V} · (1 + β · H_{boundary, MEN})."""
    base = _logvol_var(u)
    bh = np.asarray(u["men_boundary_entropy"], dtype=np.float64)
    return base * (1.0 + beta * np.nan_to_num(bh, nan=0.0))


CANDIDATE_REGISTRY: dict[str, CandidateSpec] = {
    "logvol_var": CandidateSpec(
        name="logvol_var",
        family="epistemic_scalar",
        builder=_logvol_var,
        description="Variance of log-volume across the M=20 LoRA ensemble (current main-experiment σ²_v).",
    ),
    "logvol_mad_var": CandidateSpec(
        name="logvol_mad_var",
        family="epistemic_scalar_robust",
        builder=_logvol_mad_var,
        description="MAD-scaled robust variance of log-volume across the ensemble.",
    ),
    "vol_cv2": CandidateSpec(
        name="vol_cv2",
        family="relative",
        builder=_vol_cv2,
        description="Squared coefficient of variation of raw volume across the ensemble.",
    ),
    "mean_entropy": CandidateSpec(
        name="mean_entropy",
        family="total_voxel",
        builder=_passthrough("mean_entropy"),
        description="ROI-mean predictive entropy of the mean prob map (post-Stage-0 repair).",
    ),
    "mean_mi": CandidateSpec(
        name="mean_mi",
        family="epistemic_voxel",
        builder=_passthrough("mean_mi"),
        description="ROI-mean BALD / mutual information across ensemble members (post-Stage-0 repair).",
    ),
    "mean_var_voxel": CandidateSpec(
        name="mean_var_voxel",
        family="epistemic_voxel",
        builder=_passthrough("mean_var"),
        description="ROI-mean of voxelwise ensemble variance Var_m[p_{m,v}].",
    ),
    "men_entropy": CandidateSpec(
        name="men_entropy",
        family="total_men",
        builder=_passthrough("men_mean_entropy"),
        description="Predictive entropy averaged over MEN-region voxels (post-Stage-0 repair).",
    ),
    "men_mi": CandidateSpec(
        name="men_mi",
        family="epistemic_men",
        builder=_passthrough("men_mean_mi"),
        description="BALD / MI averaged over MEN-region voxels (post-Stage-0 repair).",
    ),
    "men_boundary_entropy": CandidateSpec(
        name="men_boundary_entropy",
        family="total_boundary",
        builder=_passthrough("men_boundary_entropy"),
        description="Predictive entropy averaged over the MEN boundary band (Kendall–Gal-style aleatoric proxy).",
    ),
    "men_boundary_mi": CandidateSpec(
        name="men_boundary_mi",
        family="epistemic_boundary",
        builder=_passthrough("men_boundary_mi"),
        description="BALD / MI averaged over the MEN boundary band (post-Stage-0 repair).",
    ),
    "composite_logvol_x_boundary_entropy": CandidateSpec(
        name="composite_logvol_x_boundary_entropy",
        family="composite",
        builder=_composite_logvol_x_boundary_entropy,
        description="σ²_{log V} multiplicatively modulated by MEN boundary entropy.",
    ),
}

CONTROL_NAMES: tuple[str, ...] = ("zero", "constant_mean", "permuted")


# ---------------------------------------------------------------------------
# Scaling transforms
# ---------------------------------------------------------------------------


def scale_raw(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Identity scaling — use the candidate value directly as σ²_v."""
    return np.asarray(candidate, dtype=np.float64)


def scale_mean_matched(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Affine rescaling so cohort mean equals the cohort mean of ``reference``.

    Preserves rank; isolates shape information from absolute scale.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    cm = float(candidate.mean())
    if cm <= 0.0 or not np.isfinite(cm):
        # All zeros (or NaN) → leave as-is; downstream floor will clip.
        return candidate
    return candidate * (float(reference.mean()) / cm)


SCALING_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "raw": scale_raw,
    "mean_matched": scale_mean_matched,
}


# ---------------------------------------------------------------------------
# Vector assembly + control samplers
# ---------------------------------------------------------------------------


def apply_floor(vec: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(np.asarray(vec, dtype=np.float64), float(floor))


def build_control(
    name: str,
    n_scans: int,
    empirical: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Build one of the negative-control σ²_v vectors."""
    if name == "zero":
        return np.zeros(n_scans, dtype=np.float64)
    if name == "constant_mean":
        return np.full(n_scans, float(np.asarray(empirical).mean()), dtype=np.float64)
    if name == "permuted":
        rng = np.random.default_rng(seed)
        return rng.permutation(np.asarray(empirical, dtype=np.float64))
    raise ValueError(f"Unknown control: {name!r}")


def candidate_value_per_scan(
    spec: CandidateSpec,
    uncertainty_group: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return per-scan candidate vector aligned with H5 row order."""
    return np.asarray(spec.builder(uncertainty_group), dtype=np.float64)
