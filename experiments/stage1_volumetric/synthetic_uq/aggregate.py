"""Aggregate the synthetic σ²_v stress-test sweep.

Inputs (under ``--output-dir``):
  - ``cohort_meta.json``
  - ``runs/{profile}_{level}/seed*/marginal.json``
  - ``lme_baseline.json``

Outputs (in the same directory):
  - ``aggregated/marginal_table.csv`` — per (model, profile, level, seed)
  - ``aggregated/marginal_summary.csv`` — mean ± 95% bootstrap CI per
    (model, profile, level), aggregated over seeds
  - ``aggregated/conditional_table.csv`` — per (model, profile, level,
    seed, tertile)
  - ``aggregated/conditional_summary.csv`` — aggregated over seeds
  - ``aggregated/paired_high_tertile.csv`` — paired bootstrap of
    ΔIS@95(LME → LMEHetero) on the high tertile, per (profile, level)
  - ``figures/`` — per-profile PDF/PNG
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

CALIB_KEYS = (
    "n",
    "r2_log",
    "ci_width_mean",
    "cov_50",
    "cov_80",
    "cov_90",
    "cov_95",
    "crps",
    "is_95",
    "nlpd",
    "pred_var_mean",
)
META_KEYS = (
    "model",
    "profile",
    "level",
    "seed",
    "wall_time_s",
    "injected_sigma_v_sq_mean",
    "injected_sigma_v_sq_p90",
    "injected_sigma_v_sq_high_frac",
)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_summary(output_dir: Path) -> pd.DataFrame:
    rows = json.loads((output_dir / "summary_rows.json").read_text())
    return pd.DataFrame(rows)


def explode_conditional(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        cond = r.get("conditional", {})
        for tertile, vals in cond.items():
            if not vals or vals.get("n", 0) == 0:
                continue
            row = {k: r.get(k) for k in META_KEYS}
            row["tertile"] = tertile
            for k in CALIB_KEYS:
                row[k] = vals.get(k, float("nan"))
            row["sigma_v_sq_mean_tertile"] = vals.get("sigma_v_sq_mean", float("nan"))
            out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _summarise(group: pd.DataFrame) -> pd.Series:
    out: dict[str, float] = {}
    out["n_seeds"] = int(len(group))
    for k in CALIB_KEYS:
        if k == "n":
            out[k] = int(group[k].iloc[0])
            continue
        vals = group[k].astype(np.float64).to_numpy()
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return pd.Series(out)


def aggregate_marginal(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "profile", "level"], as_index=False)
        .apply(_summarise, include_groups=False)
        .reset_index(drop=True)
    )


def aggregate_conditional(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "profile", "level", "tertile"], as_index=False)
        .apply(_summarise, include_groups=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Paired bootstrap on high-tertile IS@95
# ---------------------------------------------------------------------------


def _paired_bootstrap_delta(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, tuple[float, float], float]:
    """Return (mean Δ, 95% CI, two-sided p) for paired Δ = a − b under
    a paired bootstrap of the average."""
    diffs = a - b
    obs = float(np.nanmean(diffs))
    n = len(diffs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.nanmean(diffs[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    # Two-sided p-value: fraction of bootstrap means with sign opposite to obs
    if obs == 0.0:
        p = 1.0
    else:
        p = 2.0 * float(np.minimum(np.mean(boots <= 0), np.mean(boots >= 0)))
        p = float(min(p, 1.0))
    return obs, (float(lo), float(hi)), p


def paired_high_tertile_bootstrap(
    cond_df: pd.DataFrame, n_boot: int = 10_000, seed: int = 7
) -> pd.DataFrame:
    """For each (profile, level), pair LME vs LMEHetero on the high tertile
    across seeds and bootstrap Δ for IS@95, CRPS, cov_95, ci_width."""
    rng = np.random.default_rng(seed)
    high = cond_df[cond_df["tertile"] == "high"].copy()
    out = []
    for (profile, level), g in high.groupby(["profile", "level"]):
        # Pivot to (seed) x model
        try:
            pivot = g.pivot(
                index="seed", columns="model", values=["is_95", "crps", "cov_95", "ci_width_mean"]
            )
        except Exception as e:
            logger.warning(f"pivot failed for {profile}/{level}: {e}")
            continue
        # If LME has only one row (it's deterministic under our protocol), broadcast.
        n_seeds = len(pivot)
        for metric in ["is_95", "crps", "cov_95", "ci_width_mean"]:
            if metric not in pivot:
                continue
            sub = pivot[metric]
            if "LMEHetero" not in sub.columns or "LME" not in sub.columns:
                continue
            a = sub["LMEHetero"].to_numpy()
            b = sub["LME"].to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                continue
            obs, (lo, hi), p = _paired_bootstrap_delta(a[mask], b[mask], n_boot=n_boot, rng=rng)
            out.append(
                {
                    "profile": profile,
                    "level": level,
                    "metric": metric,
                    "n_seeds": int(mask.sum()),
                    "delta_mean_hetero_minus_homo": obs,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "p_value": p,
                    "lme_mean": float(np.nanmean(b[mask])),
                    "hetero_mean": float(np.nanmean(a[mask])),
                }
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_profile_C_p_sweep(
    cond_summary: pd.DataFrame, marginal_summary: pd.DataFrame, out_path: Path
) -> None:
    """Δ-cov95(homo − hetero) and IS@95 vs p, on the high tertile."""
    ch = cond_summary[(cond_summary["profile"] == "C") & (cond_summary["tertile"] == "high")].copy()
    if ch.empty:
        return
    ch["p"] = ch["level"].str.lstrip("p").astype(float)
    ch = ch.sort_values("p")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    for model, color, marker in [("LME", "C0", "o"), ("LMEHetero", "C3", "s")]:
        m = ch[ch["model"] == model].sort_values("p")
        axes[0].errorbar(
            m["p"],
            m["cov_95_mean"],
            yerr=m["cov_95_std"],
            fmt=marker + "-",
            color=color,
            label=model,
            capsize=3,
        )
        axes[1].errorbar(
            m["p"],
            m["is_95_mean"],
            yerr=m["is_95_std"],
            fmt=marker + "-",
            color=color,
            label=model,
            capsize=3,
        )
        axes[2].errorbar(
            m["p"],
            m["ci_width_mean_mean"],
            yerr=m["ci_width_mean_std"],
            fmt=marker + "-",
            color=color,
            label=model,
            capsize=3,
        )

    axes[0].axhline(0.95, color="k", ls=":", alpha=0.6, label="nominal")
    axes[0].set_ylabel("Coverage @ 95% (high σ²_v tertile)")
    axes[1].set_ylabel("IS@95 (high σ²_v tertile, lower=better)")
    axes[2].set_ylabel("Mean CI width (high σ²_v tertile)")
    for ax in axes:
        ax.set_xlabel("high-tail fraction p (profile C)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Profile C — bimodal p sweep on high-σ²_v tertile (cohort mean fixed)", fontsize=12
    )
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def plot_profile_D_tau_sweep(cond_summary: pd.DataFrame, out_path: Path) -> None:
    ch = cond_summary[(cond_summary["profile"] == "D") & (cond_summary["tertile"] == "high")].copy()
    if ch.empty:
        return
    ch["tau"] = ch["level"].str.lstrip("tau").astype(float)
    ch = ch.sort_values("tau")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model, color, marker in [("LME", "C0", "o"), ("LMEHetero", "C3", "s")]:
        m = ch[ch["model"] == model].sort_values("tau")
        axes[0].errorbar(
            m["tau"],
            m["cov_95_mean"],
            yerr=m["cov_95_std"],
            fmt=marker + "-",
            color=color,
            label=model,
            capsize=3,
        )
        axes[1].errorbar(
            m["tau"],
            m["is_95_mean"],
            yerr=m["is_95_std"],
            fmt=marker + "-",
            color=color,
            label=model,
            capsize=3,
        )
    axes[0].axhline(0.95, color="k", ls=":", alpha=0.6, label="nominal")
    axes[0].set_ylabel("Coverage @ 95% (high tertile)")
    axes[1].set_ylabel("IS@95 (high tertile)")
    for ax in axes:
        ax.set_xlabel(r"log-normal $\tau$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Profile D — log-normal τ sweep at fixed cohort mean", fontsize=12)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def plot_profile_A_constant(marginal_summary: pd.DataFrame, out_path: Path) -> None:
    """Sanity: profile A should give LME ≈ LMEHetero (degenerate dispersion)."""
    a = marginal_summary[marginal_summary["profile"] == "A"].copy()
    if a.empty:
        return
    a["c"] = a["level"].str.lstrip("c").astype(float)
    a = a.sort_values("c")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model, color, marker in [("LME", "C0", "o"), ("LMEHetero", "C3", "s")]:
        m = a[a["model"] == model].sort_values("c")
        axes[0].errorbar(
            m["c"],
            m["cov_95_mean"],
            yerr=m["cov_95_std"],
            fmt=marker + "-",
            color=color,
            label=model,
            capsize=3,
        )
        axes[1].errorbar(
            m["c"],
            m["is_95_mean"],
            yerr=m["is_95_std"],
            fmt=marker + "-",
            color=color,
            label=model,
            capsize=3,
        )
    axes[0].axhline(0.95, color="k", ls=":", alpha=0.6)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel("Marginal coverage @ 95%")
    axes[1].set_ylabel("Marginal IS@95")
    for ax in axes:
        ax.set_xlabel("constant σ²_v = c")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.suptitle("Profile A — constant σ²_v (LME ≡ LMEHetero expected)", fontsize=12)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
        "uncertainty_propagation_volume_prediction/synthetic_uq",
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    output_dir = Path(args.output_dir)
    agg_dir = output_dir / "aggregated"
    fig_dir = output_dir / "figures"
    agg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((output_dir / "summary_rows.json").read_text())
    df_marg = pd.DataFrame([{k: r.get(k) for k in (*META_KEYS, *CALIB_KEYS)} for r in rows])
    df_cond = explode_conditional(rows)

    df_marg.to_csv(agg_dir / "marginal_table.csv", index=False)
    df_cond.to_csv(agg_dir / "conditional_table.csv", index=False)

    marg_summary = aggregate_marginal(df_marg)
    cond_summary = aggregate_conditional(df_cond)
    marg_summary.to_csv(agg_dir / "marginal_summary.csv", index=False)
    cond_summary.to_csv(agg_dir / "conditional_summary.csv", index=False)

    paired = paired_high_tertile_bootstrap(df_cond, n_boot=args.n_boot)
    paired.to_csv(agg_dir / "paired_high_tertile.csv", index=False)

    # Headline rows for stdout
    print("\n=== Marginal IS@95, mean across seeds ===")
    pivot_is = marg_summary[
        marg_summary.columns.intersection(
            [
                "model",
                "profile",
                "level",
                "is_95_mean",
                "is_95_std",
                "cov_95_mean",
                "ci_width_mean_mean",
            ]
        )
    ].sort_values(["profile", "level", "model"])
    print(pivot_is.to_string(index=False))

    print("\n=== Conditional IS@95 on HIGH tertile ===")
    high = cond_summary[cond_summary["tertile"] == "high"].sort_values(
        ["profile", "level", "model"]
    )[["model", "profile", "level", "is_95_mean", "cov_95_mean", "ci_width_mean_mean"]]
    print(high.to_string(index=False))

    print("\n=== Paired Δ(LMEHetero − LME) on high tertile ===")
    print(paired.sort_values(["profile", "level"]).to_string(index=False))

    # Figures
    plot_profile_A_constant(marg_summary, fig_dir / "fig_A_constant.pdf")
    plot_profile_C_p_sweep(cond_summary, marg_summary, fig_dir / "fig_C_p_sweep_high.pdf")
    plot_profile_D_tau_sweep(cond_summary, fig_dir / "fig_D_tau_sweep_high.pdf")

    logger.info("Aggregation complete. Tables in %s, figures in %s", agg_dir, fig_dir)


if __name__ == "__main__":
    main()
