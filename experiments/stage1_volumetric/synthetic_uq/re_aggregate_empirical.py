"""Re-aggregate the synthetic-UQ sweep stratifying by **empirical**
σ²_v_target (per patient), not by the injected synthetic value.

Why this script exists:
  ``run_synthetic_uq.py`` records per-fold ``sigma_v_sq_target`` as the
  *injected* value the LMEHetero model actually saw. Profile A holds
  σ²_v constant across all scans, so every patient's recorded value is
  identical and the per-tertile breakdown collapses to a single bucket.

  We want to evaluate calibration *on the same patients* across all
  profiles — i.e. the patient strata must be identical to those of the
  original observational analysis. So we stratify by the empirical
  σ²_v_target stored in ``cohort_meta.json``.

Inputs:
  ``synthetic_uq/cohort_meta.json``                     (patient_ids → empirical σ²_v_target, q33, q66)
  ``synthetic_uq/lme_baseline.json``                    (LME LOPO, used as observational baseline)
  ``synthetic_uq/runs/{profile}_{level}/seed{NNN}/lopo_results_lmehetero.json``

Outputs:
  ``synthetic_uq/aggregated/marginal_summary_empirical.csv``
  ``synthetic_uq/aggregated/conditional_summary_empirical.csv``
  ``synthetic_uq/aggregated/clean_propagation_table.csv``
      Δ vs ``LMEHetero @ Profile A / c=1e-3`` baseline (degenerate σ²_v —
      same model machinery, no propagation possible). This is the
      *clean propagation* comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from growth.shared.metrics import (
    compute_coverage_at_levels,
    compute_crps_gaussian,
    compute_interval_score,
    compute_log_score,
    compute_r2,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = (
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction/synthetic_uq"
)


def _calib(pred, actual, lower, upper, var):
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    sigma = np.sqrt(np.maximum(var, 1e-15))
    cov = compute_coverage_at_levels(actual, pred, sigma)
    return {
        "n": int(len(actual)),
        "r2_log": compute_r2(actual, pred),
        "ci_width_mean": float(np.mean(upper - lower)),
        "cov_50": float(cov[0.50]),
        "cov_80": float(cov[0.80]),
        "cov_90": float(cov[0.90]),
        "cov_95": float(cov[0.95]),
        "crps": compute_crps_gaussian(actual, pred, sigma),
        "is_95": compute_interval_score(actual, lower, upper, alpha=0.05),
        "nlpd": compute_log_score(actual, pred, np.maximum(var, 1e-15)),
    }


def _flatten_lopo(results_dict, protocol="last_from_rest"):
    """Yield (patient_id, pred_mean, actual, lower, upper, pred_var) per fold."""
    for fr in results_dict["fold_results"]:
        if protocol not in fr["predictions"]:
            continue
        for p in fr["predictions"][protocol]:
            yield (
                fr["patient_id"],
                p["pred_mean"],
                p["actual"],
                p["lower_95"],
                p["upper_95"],
                p["pred_var"],
            )


def stratify(rows, tertile_map):
    """Group rows by empirical tertile. ``rows`` is a list of tuples
    (patient_id, pred, actual, lower, upper, var)."""
    out = {"low": [], "mid": [], "high": []}
    for r in rows:
        t = tertile_map[r[0]]
        out[t].append(r)
    return out


def build_tertile_map(meta):
    q33, q66 = meta["tertile_cuts_empirical"]
    tertiles = {}
    for pid, sv2 in zip(meta["patient_ids"], meta["empirical_sigma_v_sq_target_per_patient"]):
        tertiles[pid] = "low" if sv2 <= q33 else ("mid" if sv2 <= q66 else "high")
    return tertiles, q33, q66


def metrics_for_run(rows, tertile_map):
    """Compute marginal + per-tertile calibration battery from a fold-rows list."""
    arrs = list(zip(*rows))
    pids, pm, pa, pl, pu, pv = arrs
    marginal = _calib(pm, pa, pl, pu, pv)
    cond = {}
    by_t = stratify(rows, tertile_map)
    for t, rs in by_t.items():
        if len(rs) == 0:
            cond[t] = {"n": 0}
            continue
        a = list(zip(*rs))
        cond[t] = _calib(a[1], a[2], a[3], a[4], a[5])
    return marginal, cond


def aggregate(output_dir: Path) -> dict:
    meta = json.loads((output_dir / "cohort_meta.json").read_text())
    tertile_map, q33, q66 = build_tertile_map(meta)
    logger.info("Empirical tertile cuts: q33=%.5g  q66=%.5g", q33, q66)

    # LME baseline (single run, deterministic)
    lme = json.loads((output_dir / "lme_baseline.json").read_text())
    lme_rows = list(_flatten_lopo(lme))
    lme_marg, lme_cond = metrics_for_run(lme_rows, tertile_map)

    # LMEHetero per (profile, level, seed)
    marg_rows = []
    cond_rows = []
    runs_dir = output_dir / "runs"
    profiles = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
    for prof_dir in profiles:
        prof_path = runs_dir / prof_dir
        # parse "{profile}_{level}" → split on first _
        prof, level = prof_dir.split("_", 1)
        for seed_dir in sorted(prof_path.iterdir()):
            if not seed_dir.is_dir():
                continue
            lopo_path = seed_dir / "lopo_results_lmehetero.json"
            if not lopo_path.exists():
                logger.warning("missing %s — skipping", lopo_path)
                continue
            lopo = json.loads(lopo_path.read_text())
            rows = list(_flatten_lopo(lopo))
            marg, cond = metrics_for_run(rows, tertile_map)
            seed = int(seed_dir.name.lstrip("seed"))
            base = {"model": "LMEHetero", "profile": prof, "level": level, "seed": seed}
            marg_rows.append({**base, **marg})
            for t, vals in cond.items():
                if vals.get("n", 0) == 0:
                    continue
                cond_rows.append({**base, "tertile": t, **vals})

    # Add LME rows replicated for each level so the table aligns
    for prof_dir in profiles:
        prof, level = prof_dir.split("_", 1)
        # one synthetic "seed=-1" row that flags this is the cached baseline
        marg_rows.append({"model": "LME", "profile": prof, "level": level, "seed": -1, **lme_marg})
        for t, vals in lme_cond.items():
            if vals.get("n", 0) == 0:
                continue
            cond_rows.append(
                {"model": "LME", "profile": prof, "level": level, "seed": -1, "tertile": t, **vals}
            )

    return {
        "marginal_rows": marg_rows,
        "conditional_rows": cond_rows,
        "tertile_cuts": (q33, q66),
        "patient_tertiles": tertile_map,
    }


def summarise(df, group_keys):
    """Mean / std of every numeric calibration metric across seeds."""
    metrics = [
        "r2_log",
        "ci_width_mean",
        "cov_50",
        "cov_80",
        "cov_90",
        "cov_95",
        "crps",
        "is_95",
        "nlpd",
    ]
    out = df.groupby(group_keys, as_index=False).agg(
        {**{m: ["mean", "std"] for m in metrics}, "n": "first", "seed": "count"}
    )
    out.columns = ["__".join(c).rstrip("_") if isinstance(c, tuple) else c for c in out.columns]
    out = out.rename(columns={"seed__count": "n_seeds"})
    return out


def clean_propagation_table(marg_summary: pd.DataFrame, cond_summary: pd.DataFrame) -> pd.DataFrame:
    """Δ each (profile, level) row vs LMEHetero @ Profile A / level=c0.001.

    The baseline is "LMEHetero with degenerate σ²_v" — same custom REML
    machinery, no per-scan dispersion. Δ = X(profile) − X(baseline)
    isolates the σ²_v propagation effect.
    """
    base = marg_summary[
        (marg_summary["model"] == "LMEHetero")
        & (marg_summary["profile"] == "A")
        & (marg_summary["level"] == "c0.001")
    ]
    if base.empty:
        raise RuntimeError("baseline LMEHetero @ A/c0.001 not found")
    b = base.iloc[0]
    delta_rows = []
    for _, r in marg_summary.iterrows():
        if r["model"] != "LMEHetero":
            continue
        delta_rows.append(
            {
                "profile": r["profile"],
                "level": r["level"],
                "tertile": "MARGINAL",
                "cov_95_mean": r["cov_95__mean"],
                "ci_width_mean_mean": r["ci_width_mean__mean"],
                "is_95_mean": r["is_95__mean"],
                "crps_mean": r["crps__mean"],
                "delta_cov_95": r["cov_95__mean"] - b["cov_95__mean"],
                "delta_ci_width": r["ci_width_mean__mean"] - b["ci_width_mean__mean"],
                "delta_is_95": r["is_95__mean"] - b["is_95__mean"],
                "delta_crps": r["crps__mean"] - b["crps__mean"],
            }
        )

    # Per-tertile baselines
    for tert in ["low", "mid", "high"]:
        bt = cond_summary[
            (cond_summary["model"] == "LMEHetero")
            & (cond_summary["profile"] == "A")
            & (cond_summary["level"] == "c0.001")
            & (cond_summary["tertile"] == tert)
        ]
        if bt.empty:
            continue
        bt = bt.iloc[0]
        for _, r in cond_summary[
            (cond_summary["model"] == "LMEHetero") & (cond_summary["tertile"] == tert)
        ].iterrows():
            delta_rows.append(
                {
                    "profile": r["profile"],
                    "level": r["level"],
                    "tertile": tert,
                    "cov_95_mean": r["cov_95__mean"],
                    "ci_width_mean_mean": r["ci_width_mean__mean"],
                    "is_95_mean": r["is_95__mean"],
                    "crps_mean": r["crps__mean"],
                    "delta_cov_95": r["cov_95__mean"] - bt["cov_95__mean"],
                    "delta_ci_width": r["ci_width_mean__mean"] - bt["ci_width_mean__mean"],
                    "delta_is_95": r["is_95__mean"] - bt["is_95__mean"],
                    "delta_crps": r["crps__mean"] - bt["crps__mean"],
                }
            )
    return pd.DataFrame(delta_rows).sort_values(["tertile", "profile", "level"])


def paired_bootstrap_propagation(
    cond_long: pd.DataFrame, n_boot: int = 10_000, seed: int = 11
) -> pd.DataFrame:
    """For each (profile, level), pair LMEHetero@(p,l) vs LMEHetero@A/c0.001 by seed
    and bootstrap Δ on the high tertile. This is the *clean* propagation test."""
    rng = np.random.default_rng(seed)
    base = cond_long[
        (cond_long["model"] == "LMEHetero")
        & (cond_long["profile"] == "A")
        & (cond_long["level"] == "c0.001")
        & (cond_long["tertile"] == "high")
    ].sort_values("seed")
    if base.empty:
        return pd.DataFrame()
    out = []
    for (prof, lvl), g in cond_long[
        (cond_long["model"] == "LMEHetero") & (cond_long["tertile"] == "high")
    ].groupby(["profile", "level"]):
        if (prof, lvl) == ("A", "c0.001"):
            continue
        g = g.sort_values("seed")
        merged = base.merge(g, on="seed", suffixes=("_base", "_x"))
        for metric in ["is_95", "crps", "cov_95", "ci_width_mean"]:
            a = merged[f"{metric}_x"].to_numpy(dtype=np.float64)
            b = merged[f"{metric}_base"].to_numpy(dtype=np.float64)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                continue
            d = a[mask] - b[mask]
            obs = float(d.mean())
            boots = np.empty(n_boot)
            for i in range(n_boot):
                boots[i] = float(np.mean(d[rng.integers(0, len(d), size=len(d))]))
            lo, hi = np.quantile(boots, [0.025, 0.975])
            p = 2.0 * float(min(np.mean(boots <= 0), np.mean(boots >= 0)))
            out.append(
                {
                    "profile": prof,
                    "level": lvl,
                    "metric": metric,
                    "n_seeds": int(mask.sum()),
                    "delta_x_minus_baseline": obs,
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                    "p_value": float(min(p, 1.0)),
                    "baseline_mean": float(b.mean()),
                    "x_mean": float(a.mean()),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    output_dir = Path(args.output_dir)
    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    res = aggregate(output_dir)
    df_marg = pd.DataFrame(res["marginal_rows"])
    df_cond = pd.DataFrame(res["conditional_rows"])
    df_marg.to_csv(agg_dir / "marginal_table_empirical.csv", index=False)
    df_cond.to_csv(agg_dir / "conditional_table_empirical.csv", index=False)

    # Restrict aggregation to LMEHetero rows (LME has only one "seed=-1" replica per level)
    df_marg_h = df_marg[df_marg["model"] == "LMEHetero"]
    df_cond_h = df_cond[df_cond["model"] == "LMEHetero"]

    marg_sum = summarise(df_marg_h, ["model", "profile", "level"])
    cond_sum = summarise(df_cond_h, ["model", "profile", "level", "tertile"])
    marg_sum.to_csv(agg_dir / "marginal_summary_empirical.csv", index=False)
    cond_sum.to_csv(agg_dir / "conditional_summary_empirical.csv", index=False)

    # LME observational reference (single row per tertile)
    lme_marg = df_marg[df_marg["model"] == "LME"].drop_duplicates(["model"]).iloc[0]
    lme_cond = df_cond[df_cond["model"] == "LME"].drop_duplicates(["tertile"])
    print("\n=== LME observational reference (cached) ===")
    print(
        f"  Marginal: cov_95={lme_marg['cov_95']:.4f}, ci_w={lme_marg['ci_width_mean']:.4f}, "
        f"IS@95={lme_marg['is_95']:.4f}, CRPS={lme_marg['crps']:.4f}, R²={lme_marg['r2_log']:.4f}"
    )
    for _, r in lme_cond.sort_values("tertile").iterrows():
        print(
            f"  {r['tertile']:>5s} (n={int(r['n'])}): cov_95={r['cov_95']:.4f}, "
            f"ci_w={r['ci_width_mean']:.4f}, IS@95={r['is_95']:.4f}, CRPS={r['crps']:.4f}"
        )

    # Clean propagation: LMEHetero@(p,l) - LMEHetero@A/c0.001
    delta_df = clean_propagation_table(marg_sum, cond_sum)
    delta_df.to_csv(agg_dir / "clean_propagation_table.csv", index=False)

    print("\n=== CLEAN PROPAGATION: LMEHetero @ profile X − LMEHetero @ Profile A/c=1e-3 ===")
    for tert in ["MARGINAL", "low", "mid", "high"]:
        sub = delta_df[delta_df["tertile"] == tert].copy()
        if sub.empty:
            continue
        print(f"\n--- {tert} ---")
        cols = [
            "profile",
            "level",
            "cov_95_mean",
            "ci_width_mean_mean",
            "is_95_mean",
            "crps_mean",
            "delta_cov_95",
            "delta_ci_width",
            "delta_is_95",
            "delta_crps",
        ]
        for c in cols:
            if sub[c].dtype.kind == "f":
                sub[c] = sub[c].round(4)
        print(sub[cols].to_string(index=False))

    paired = paired_bootstrap_propagation(df_cond_h, n_boot=args.n_boot)
    if not paired.empty:
        paired.to_csv(agg_dir / "paired_propagation_high.csv", index=False)
        print(
            "\n=== Paired bootstrap on HIGH tertile, "
            "Δ = LMEHetero@(profile/level) − LMEHetero@A/c0.001 ==="
        )
        for metric in ["is_95", "ci_width_mean", "cov_95"]:
            sub = paired[paired["metric"] == metric].copy()
            sub = sub[
                [
                    "profile",
                    "level",
                    "delta_x_minus_baseline",
                    "ci95_lo",
                    "ci95_hi",
                    "p_value",
                    "baseline_mean",
                    "x_mean",
                ]
            ]
            for c in sub.columns:
                if sub[c].dtype.kind == "f":
                    sub[c] = sub[c].round(4)
            print(f"\n  metric={metric}")
            print("  " + sub.to_string(index=False).replace("\n", "\n  "))

    logger.info("Re-aggregation done. CSVs in %s", agg_dir)


if __name__ == "__main__":
    main()
