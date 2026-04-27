"""Synthetic fixtures for inter-LoRA tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_SCANS: int = 10
N_MEMBERS: int = 5
RANKS: list[int] = [4, 8, 16]
SEED: int = 42


def _make_dice(rng: np.random.Generator, n: int, base: float = 0.75) -> np.ndarray:
    return np.clip(rng.normal(base, 0.08, size=n), 0.01, 0.99)


def _make_scan_ids(n: int) -> list[str]:
    return [f"BraTS-MEN-{i:05d}-000" for i in range(n)]


def _write_rank_dir(root: Path, rank: int, rng: np.random.Generator) -> None:
    """Create a synthetic rank directory with all required evaluation files."""
    name = f"r{rank}_M{N_MEMBERS}_s{SEED}"
    rank_dir = root / name
    ev = rank_dir / "evaluation"
    ev.mkdir(parents=True)

    scan_ids = _make_scan_ids(N_SCANS)
    base_offset = 0.01 * np.log2(rank)

    # baseline_test_dice.csv — identical across ranks
    bas_tc = _make_dice(np.random.default_rng(999), N_SCANS, 0.65)
    bas_wt = _make_dice(np.random.default_rng(999), N_SCANS, 0.60)
    bas_et = _make_dice(np.random.default_rng(999), N_SCANS, 0.64)
    bas_df = pd.DataFrame(
        {
            "scan_id": scan_ids,
            "dice_tc": bas_tc,
            "dice_wt": bas_wt,
            "dice_et": bas_et,
            "dice_mean": (bas_tc + bas_wt + bas_et) / 3,
            "volume_baseline": rng.integers(100, 50000, size=N_SCANS).astype(float),
            "volume_gt": rng.integers(100, 50000, size=N_SCANS).astype(float),
        },
    )
    bas_df.to_csv(ev / "baseline_test_dice.csv", index=False)

    # ensemble_test_dice.csv
    ens_tc = _make_dice(rng, N_SCANS, 0.75 + base_offset)
    ens_wt = _make_dice(rng, N_SCANS, 0.70 + base_offset)
    ens_et = _make_dice(rng, N_SCANS, 0.74 + base_offset)
    ens_df = pd.DataFrame(
        {
            "scan_id": scan_ids,
            "dice_tc": ens_tc,
            "dice_wt": ens_wt,
            "dice_et": ens_et,
            "dice_mean": (ens_tc + ens_wt + ens_et) / 3,
            "volume_ensemble": rng.integers(100, 50000, size=N_SCANS).astype(float),
            "volume_gt": bas_df["volume_gt"].values,
        },
    )
    ens_df.to_csv(ev / "ensemble_test_dice.csv", index=False)

    # per_member_test_dice.csv
    rows = []
    for m in range(N_MEMBERS):
        for i, sid in enumerate(scan_ids):
            rows.append(
                {
                    "member_id": m,
                    "scan_id": sid,
                    "dice_tc": float(np.clip(ens_tc[i] + rng.normal(0, 0.03), 0.01, 0.99)),
                    "dice_wt": float(np.clip(ens_wt[i] + rng.normal(0, 0.03), 0.01, 0.99)),
                    "dice_et": float(np.clip(ens_et[i] + rng.normal(0, 0.03), 0.01, 0.99)),
                    "dice_mean": 0.0,
                    "volume_pred": float(rng.integers(100, 50000)),
                },
            )
    pm_df = pd.DataFrame(rows)
    pm_df["dice_mean"] = (pm_df["dice_tc"] + pm_df["dice_wt"] + pm_df["dice_et"]) / 3
    pm_df.to_csv(ev / "per_member_test_dice.csv", index=False)

    # paired_differences.csv
    pd.DataFrame(
        {
            "scan_id": scan_ids,
            "dice_tc_delta": ens_tc - bas_tc,
            "dice_wt_delta": ens_wt - bas_wt,
            "dice_et_delta": ens_et - bas_et,
        },
    ).to_csv(ev / "paired_differences.csv", index=False)

    # calibration.json
    with open(ev / "calibration.json", "w") as f:
        json.dump(
            {
                "ece": float(rng.uniform(1e-4, 5e-3)),
                "brier_score": float(rng.uniform(1e-4, 5e-3)),
                "reliability": {
                    "bin_edges": list(np.linspace(0, 1, 16)),
                    "bin_accuracy": [float(rng.uniform(0, 1)) for _ in range(15)],
                    "bin_confidence": [float(rng.uniform(0, 1)) for _ in range(15)],
                    "bin_count": [int(rng.integers(10, 1000)) for _ in range(15)],
                },
            },
            f,
        )

    # calibration_coverage.csv
    pd.DataFrame(
        {
            "nominal_level": [0.5, 0.8, 0.9, 0.95],
            "t_multiplier": [0.68, 1.33, 1.73, 2.09],
            "n_scans": [N_SCANS] * 4,
            "n_covered": [
                int(N_SCANS * rng.uniform(0.1, 0.6)),
                int(N_SCANS * rng.uniform(0.15, 0.7)),
                int(N_SCANS * rng.uniform(0.2, 0.8)),
                int(N_SCANS * rng.uniform(0.25, 0.9)),
            ],
            "empirical_coverage": [0.0] * 4,
            "coverage_deficit": [0.0] * 4,
        },
    ).to_csv(ev / "calibration_coverage.csv", index=False)

    # bias_diagnostics.csv
    pd.DataFrame(
        {
            "scan_id": scan_ids,
            "n_members": [N_MEMBERS] * N_SCANS,
            "volume_gt": bas_df["volume_gt"].values,
            "volume_ensemble_mean": ens_df["volume_ensemble"].values,
            "volume_ensemble_std": rng.uniform(10, 200, size=N_SCANS),
            "bias": rng.normal(0, 100, size=N_SCANS),
            "abs_bias": rng.uniform(0, 200, size=N_SCANS),
            "bias_to_std_ratio": rng.uniform(0, 5, size=N_SCANS),
            "logvol_gt": np.log1p(bas_df["volume_gt"].values),
            "logvol_ensemble_mean": np.log1p(ens_df["volume_ensemble"].values),
            "logvol_ensemble_std": rng.uniform(0.001, 0.2, size=N_SCANS),
            "logvol_bias": rng.normal(0, 0.1, size=N_SCANS),
            "logvol_abs_bias": rng.uniform(0, 0.3, size=N_SCANS),
            "logvol_bias_to_std_ratio": rng.uniform(0, 10, size=N_SCANS),
        },
    ).to_csv(ev / "bias_diagnostics.csv", index=False)

    # bias_dominance_threshold.csv
    pd.DataFrame(
        {
            "scan_id": scan_ids,
            "n_members_actual": [N_MEMBERS] * N_SCANS,
            "k_star_raw": rng.integers(1, 50, size=N_SCANS),
            "k_star_logvol": rng.integers(1, 50, size=N_SCANS),
            "k_star_saturated": [False] * N_SCANS,
            "degenerate_ensemble": [False] * N_SCANS,
            "k_star_exceeds_M": (rng.uniform(size=N_SCANS) > 0.9).tolist(),
        },
    ).to_csv(ev / "bias_dominance_threshold.csv", index=False)

    # epistemic_taxonomy.json
    pct_k1 = float(rng.uniform(0.3, 0.9))
    pct_exceed = float(rng.uniform(0.01, 0.1))
    pct_degen = float(rng.uniform(0.01, 0.05))
    with open(ev / "epistemic_taxonomy.json", "w") as f:
        json.dump(
            {
                "config": {"rank": rank, "n_members": N_MEMBERS, "seed": SEED},
                "taxonomy": {
                    "estimation_bias": {
                        "status": "diagnosed",
                        "bias_dominance": {
                            "median_k_star_logvol": 1.0,
                            "pct_scans_k_star_eq_1": pct_k1,
                            "pct_scans_k_star_exceeds_M": pct_exceed,
                            "pct_scans_degenerate_ensemble": pct_degen,
                            "n_members_sampled": N_MEMBERS,
                        },
                    },
                },
                "calibration": {
                    "coverage_50": float(rng.uniform(0.1, 0.5)),
                    "coverage_80": float(rng.uniform(0.15, 0.7)),
                    "coverage_90": float(rng.uniform(0.2, 0.8)),
                    "coverage_95": float(rng.uniform(0.25, 0.9)),
                    "coverage_deficit_95": float(rng.uniform(0.1, 0.7)),
                },
            },
            f,
        )

    # statistical_summary.json
    ss: dict = {
        "ensemble_vs_baseline": {},
        "inter_member_agreement": {
            "icc_wt": float(rng.uniform(0.9, 0.99)),
            "icc_tc": float(rng.uniform(0.9, 0.99)),
            "icc_et": float(rng.uniform(0.9, 0.99)),
            "mean_pairwise_correlation_wt": float(rng.uniform(0.9, 0.99)),
        },
        "per_member_summary": [],
    }
    for label, col in [
        ("tc", ens_tc),
        ("wt", ens_wt),
        ("et", ens_et),
        ("mean", (ens_tc + ens_wt + ens_et) / 3),
    ]:
        m = float(np.mean(col))
        se = float(np.std(col, ddof=1) / np.sqrt(len(col)))
        bm = float(
            np.mean(
                bas_tc
                if label == "tc"
                else bas_wt
                if label == "wt"
                else bas_et
                if label == "et"
                else (bas_tc + bas_wt + bas_et) / 3
            )
        )
        ss["ensemble_vs_baseline"][label] = {
            "ensemble_mean": m,
            "ensemble_ci95": [m - 1.96 * se, m + 1.96 * se],
            "baseline_mean": bm,
            "baseline_ci95": [bm - 1.96 * se, bm + 1.96 * se],
            "p_value_wilcoxon": float(rng.uniform(1e-6, 0.05)),
            "cohens_d": float(rng.uniform(0.3, 1.5)),
            "best_member_mean": m + float(rng.uniform(0, 0.02)),
            "delta": m - bm,
            "ci_95_lower": (m - bm) - 0.05,
            "ci_95_upper": (m - bm) + 0.05,
        }
    with open(ev / "statistical_summary.json", "w") as f:
        json.dump(ss, f)


@pytest.fixture()
def three_rank_fixture(tmp_path: Path) -> Path:
    """Create a synthetic 3-rank directory structure."""
    rng = np.random.default_rng(SEED)
    for rank in RANKS:
        _write_rank_dir(tmp_path, rank, rng)
    return tmp_path


@pytest.fixture()
def two_rank_fixture(tmp_path: Path) -> Path:
    """Create a synthetic 2-rank directory (below MIN_RANKS threshold)."""
    rng = np.random.default_rng(SEED)
    for rank in [4, 8]:
        _write_rank_dir(tmp_path, rank, rng)
    return tmp_path
