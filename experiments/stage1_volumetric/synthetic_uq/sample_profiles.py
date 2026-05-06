"""Synthetic σ²_v profile samplers for the stress-test in
``docs/UQ_SYNTHETIC_VARIANCE_STRESSTEST.md``.

Each sampler returns a 1-D array ``sigma_v_sq`` of length ``n_scans`` whose
entries replace the empirical ``observation_variance`` column on every scan
of every trajectory. Cohort-mean σ²_v is held fixed at ``target_mean`` for
profiles B/C/D so we isolate *dispersion* from *magnitude*; profile A
varies magnitude explicitly; profile E reuses the empirical vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ProfileName = Literal["A", "B", "C", "D", "E"]

# Empirical reference numbers (from `MenGrowth.h5` `uncertainty/logvol_std**2`,
# documented in `docs/UQ_HETERO_CALIBRATION_ANSWER.md`).
EMPIRICAL_MEAN_VAR = 0.4165  # mean σ²_v across 179 scans
EMPIRICAL_MEDIAN_VAR = 1.18e-3  # median σ²_v
EMPIRICAL_HI_FRAC = 11 / 179  # 6.1% of scans with σ_v > 1
EMPIRICAL_HI_MEAN_VAR = 4.2  # mean σ²_v among the high tail (rough)
EMPIRICAL_LO_MEAN_VAR = 0.011  # mean σ²_v among the bulk


@dataclass(frozen=True)
class ProfileSpec:
    name: ProfileName
    level: str  # human-readable level tag, used in paths
    description: str
    target_mean: float | None  # cohort mean σ²_v we aim to hold; None = uncontrolled


def _renormalise(sigma_v_sq: np.ndarray, target_mean: float | None) -> np.ndarray:
    """Rescale array so its mean equals target_mean (skip if target_mean is None)."""
    if target_mean is None:
        return sigma_v_sq
    cur = float(np.mean(sigma_v_sq))
    if cur <= 0.0:
        return sigma_v_sq
    return sigma_v_sq * (target_mean / cur)


def sample_A_constant(n_scans: int, c: float, rng: np.random.Generator) -> np.ndarray:
    """Profile A: constant σ²_v = c (degenerate dispersion)."""
    return np.full(n_scans, float(c), dtype=np.float64)


def sample_B_matched(
    n_scans: int,
    rng: np.random.Generator,
    target_mean: float = EMPIRICAL_MEAN_VAR,
) -> np.ndarray:
    """Profile B: matched-empirical bimodal mixture.

    Bulk mass at small σ²_v matched to the empirical 25-90 pct (≈ 0.001-0.26),
    long right tail at σ²_v ≈ empirical max (~ 11). Cohort mean rescaled to
    ``target_mean``.
    """
    p_high = EMPIRICAL_HI_FRAC
    n_high = int(round(p_high * n_scans))
    n_low = n_scans - n_high

    # Bulk: log-normal centred on empirical median (1.18e-3) with mild scatter
    log_mu_lo, log_tau_lo = np.log(1e-3), 1.5
    bulk = rng.lognormal(mean=log_mu_lo, sigma=log_tau_lo, size=n_low)

    # Tail: log-normal centred on σ²_v ≈ 4 (matches empirical mean of σ_v>1 set)
    log_mu_hi, log_tau_hi = np.log(4.0), 0.5
    tail = rng.lognormal(mean=log_mu_hi, sigma=log_tau_hi, size=n_high)

    sigma_v_sq = np.concatenate([bulk, tail])
    rng.shuffle(sigma_v_sq)
    return _renormalise(sigma_v_sq, target_mean)


def sample_C_p_sweep(
    n_scans: int,
    rng: np.random.Generator,
    p: float,
    target_mean: float = EMPIRICAL_MEAN_VAR,
) -> np.ndarray:
    """Profile C: bimodal mixture with high-tail fraction ``p`` ∈ [0, 1].

    p=0 ⇒ all scans drawn from the low component (degenerate).
    p=EMPIRICAL_HI_FRAC ⇒ matches profile B in expectation.
    p=0.4 ⇒ adversarial high dispersion.

    Cohort mean rescaled to ``target_mean`` so dispersion is the only knob.
    """
    n_high = int(round(p * n_scans))
    n_low = n_scans - n_high

    log_mu_lo, log_tau_lo = np.log(1e-3), 1.5
    log_mu_hi, log_tau_hi = np.log(4.0), 0.5

    if n_low > 0:
        bulk = rng.lognormal(mean=log_mu_lo, sigma=log_tau_lo, size=n_low)
    else:
        bulk = np.empty(0)
    if n_high > 0:
        tail = rng.lognormal(mean=log_mu_hi, sigma=log_tau_hi, size=n_high)
    else:
        tail = np.empty(0)

    sigma_v_sq = np.concatenate([bulk, tail]) if (n_low + n_high) > 0 else np.zeros(n_scans)
    if len(sigma_v_sq) < n_scans:
        # Edge case: rounding to zero scans — fall back to bulk only.
        sigma_v_sq = rng.lognormal(mean=log_mu_lo, sigma=log_tau_lo, size=n_scans)

    rng.shuffle(sigma_v_sq)
    if p == 0.0:
        # Degenerate: still rescale so cohort mean equals target_mean (this puts
        # everyone at the same magnitude — hetero ≈ homo expected).
        return _renormalise(sigma_v_sq, target_mean)
    return _renormalise(sigma_v_sq, target_mean)


def sample_D_lognormal(
    n_scans: int,
    rng: np.random.Generator,
    tau: float,
    target_mean: float = EMPIRICAL_MEAN_VAR,
) -> np.ndarray:
    """Profile D: continuous log-normal dispersion sweep.

    log σ²_v ~ N(μ, τ²); μ chosen so that E[σ²_v] = target_mean exactly:
    μ = log(target_mean) − τ²/2.

    τ=0 ⇒ degenerate (constant σ²_v = target_mean) ≡ profile A at c=target_mean.
    Larger τ ⇒ smoother dispersion than the bimodal profiles, but with a
    log-normal heavy tail.
    """
    if tau == 0.0:
        return np.full(n_scans, float(target_mean), dtype=np.float64)
    mu = np.log(target_mean) - 0.5 * tau**2
    sigma_v_sq = rng.lognormal(mean=mu, sigma=tau, size=n_scans)
    return _renormalise(sigma_v_sq, target_mean)  # numerical pin


def sample_E_empirical(empirical_sigma_v_sq: np.ndarray) -> np.ndarray:
    """Profile E: pass-through of the empirical σ²_v (no synthetic perturbation).

    Sanity baseline; the result should match the existing
    ``conditional_calibration_last_from_rest.json`` to within MC noise.
    """
    return np.asarray(empirical_sigma_v_sq, dtype=np.float64).copy()


def get_default_profiles() -> list[ProfileSpec]:
    """Return the (profile, level) grid run by the stress test."""
    specs: list[ProfileSpec] = []

    # A: magnitude sweep
    for c in [1e-3, 1e-2, 1e-1, 1.0]:
        specs.append(
            ProfileSpec(
                name="A",
                level=f"c{c:g}",
                description=f"constant σ²_v = {c:g}",
                target_mean=None,
            )
        )

    # B: empirical-matched bimodal
    specs.append(
        ProfileSpec(
            name="B",
            level="matched",
            description=f"bimodal, p={EMPIRICAL_HI_FRAC:.3f}, mean={EMPIRICAL_MEAN_VAR:.3f}",
            target_mean=EMPIRICAL_MEAN_VAR,
        )
    )

    # C: p sweep at fixed cohort mean
    for p in [0.0, 0.05, 0.10, 0.20, 0.40]:
        specs.append(
            ProfileSpec(
                name="C",
                level=f"p{p:g}",
                description=f"bimodal, p={p}, mean={EMPIRICAL_MEAN_VAR:.3f}",
                target_mean=EMPIRICAL_MEAN_VAR,
            )
        )

    # D: tau sweep at fixed cohort mean
    for tau in [0.0, 0.5, 1.0, 1.5, 2.0]:
        specs.append(
            ProfileSpec(
                name="D",
                level=f"tau{tau:g}",
                description=f"log-normal, τ={tau}, mean={EMPIRICAL_MEAN_VAR:.3f}",
                target_mean=EMPIRICAL_MEAN_VAR,
            )
        )

    # E: pass-through empirical
    specs.append(
        ProfileSpec(
            name="E",
            level="empirical",
            description="empirical σ²_v (unchanged)",
            target_mean=None,
        )
    )

    return specs


def sample_profile(
    spec: ProfileSpec,
    n_scans: int,
    rng: np.random.Generator,
    empirical_sigma_v_sq: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to the right sampler given a ``ProfileSpec``."""
    if spec.name == "A":
        c = float(spec.level.lstrip("c"))
        return sample_A_constant(n_scans, c, rng)
    if spec.name == "B":
        return sample_B_matched(n_scans, rng, spec.target_mean or EMPIRICAL_MEAN_VAR)
    if spec.name == "C":
        p = float(spec.level.lstrip("p"))
        return sample_C_p_sweep(n_scans, rng, p, spec.target_mean or EMPIRICAL_MEAN_VAR)
    if spec.name == "D":
        tau = float(spec.level.lstrip("tau"))
        return sample_D_lognormal(n_scans, rng, tau, spec.target_mean or EMPIRICAL_MEAN_VAR)
    if spec.name == "E":
        if empirical_sigma_v_sq is None:
            raise ValueError("Profile E requires empirical_sigma_v_sq.")
        return sample_E_empirical(empirical_sigma_v_sq)
    raise ValueError(f"Unknown profile name: {spec.name}")
