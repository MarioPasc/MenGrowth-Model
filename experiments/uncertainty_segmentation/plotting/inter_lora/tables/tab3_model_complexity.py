"""Model complexity vs performance table (Tab 3).

One row per adapter configuration (Frozen BSF + each LoRA rank).
Columns: Adapter, ΔW Params, Total Params, Dice (mean ± 95% CI),
Mean Variance (mean ± 95% CI).

Produces CSV, Markdown, and LaTeX (booktabs) renditions.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import (
    InterLoraData,
    RankRun,
)

logger = logging.getLogger(__name__)

_LORA_PARAMS_PER_RANK: int = 23_040
_BASE_MODEL_PARAMS: int = 62_191_941


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapter_label(rank: int) -> str:
    if rank == 0:
        return "Frozen BSF"
    return f"LoRA $r{{=}}{rank}$"


def _adapter_label_plain(rank: int) -> str:
    if rank == 0:
        return "Frozen BSF"
    return f"LoRA r={rank}"


def _fmt_params(n: int) -> str:
    if n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_pm(mean: float, ci_lo: float, ci_hi: float) -> str:
    """Format as 'mean ± half-width' using 95% CI bounds."""
    if any(math.isnan(v) for v in (mean, ci_lo, ci_hi)):
        return "—"
    hw = (ci_hi - ci_lo) / 2
    return f"{mean:.3f} ± {hw:.3f}"


def _fmt_pm_sci(mean: float, ci_lo: float, ci_hi: float) -> str:
    """Format as 'mean ± half-width' for small values (scientific)."""
    if any(math.isnan(v) for v in (mean, ci_lo, ci_hi)):
        return "—"
    hw = (ci_hi - ci_lo) / 2
    return f"{mean:.4f} ± {hw:.4f}"


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap mean and 95% CI for a 1-D array."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    mean = float(values.mean())
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def _compute_mean_variance(
    rank_run: RankRun,
) -> tuple[float, float, float]:
    """Compute mean inter-member Dice variance with bootstrap 95% CI.

    Returns (mean, ci_lo, ci_hi).  For baseline (no members), returns
    (0, 0, 0).
    """
    pm = rank_run.per_member_dice
    if pm.empty:
        return 0.0, 0.0, 0.0

    dice_cols = [c for c in ["dice_tc", "dice_wt", "dice_et"] if c in pm.columns]
    if not dice_cols:
        return float("nan"), float("nan"), float("nan")

    per_scan_var = pm.groupby("scan_id")[dice_cols].var()
    scan_mean_var = per_scan_var.mean(axis=1).values

    return _bootstrap_ci(scan_mean_var)


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def _build_table(data: InterLoraData) -> pd.DataFrame:
    """Build the complexity table with one row per configuration."""
    cm = data.compiled_metrics

    rows: list[dict] = []

    # Frozen BSF row (rank = 0)
    bsf_dice = cm[(cm["rank"] == 0) & (cm["label"] == "mean")]
    if not bsf_dice.empty:
        r = bsf_dice.iloc[0]
        dice_str = _fmt_pm(r["dice_mean"], r["dice_ci_lo"], r["dice_ci_hi"])
        dice_str_plain = dice_str
    else:
        dice_str = "—"
        dice_str_plain = "—"

    rows.append(
        {
            "rank_val": 0,
            "adapter": _adapter_label_plain(0),
            "adapter_latex": _adapter_label(0),
            "delta_w": 0,
            "delta_w_fmt": _fmt_params(0),
            "total_params": _BASE_MODEL_PARAMS,
            "total_params_fmt": _fmt_params(_BASE_MODEL_PARAMS),
            "dice": dice_str,
            "mean_var": "0.0000 ± 0.0000",
        }
    )

    # LoRA rank rows
    for rr in data.ranks:
        if rr.rank == 0:
            continue

        lora_params = rr.rank * _LORA_PARAMS_PER_RANK
        total = _BASE_MODEL_PARAMS + lora_params

        rank_dice = cm[(cm["rank"] == rr.rank) & (cm["label"] == "mean")]
        if not rank_dice.empty:
            r = rank_dice.iloc[0]
            dice_str = _fmt_pm(r["dice_mean"], r["dice_ci_lo"], r["dice_ci_hi"])
        else:
            dice_str = "—"

        mv_mean, mv_lo, mv_hi = _compute_mean_variance(rr)
        var_str = _fmt_pm_sci(mv_mean, mv_lo, mv_hi)

        rows.append(
            {
                "rank_val": rr.rank,
                "adapter": _adapter_label_plain(rr.rank),
                "adapter_latex": _adapter_label(rr.rank),
                "delta_w": lora_params,
                "delta_w_fmt": _fmt_params(lora_params),
                "total_params": total,
                "total_params_fmt": _fmt_params(total),
                "dice": dice_str,
                "mean_var": var_str,
            }
        )

    df = pd.DataFrame(rows).sort_values("rank_val").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------


def _write_csv(table: pd.DataFrame, out_dir: Path) -> None:
    out = table[["adapter", "delta_w", "total_params", "dice", "mean_var"]].copy()
    out.columns = ["Adapter", "Delta_W_Params", "Total_Params", "Dice", "Mean_Variance"]
    path = out_dir / "tab3_model_complexity.csv"
    out.to_csv(path, index=False)
    logger.info("CSV written to %s", path)


def _write_markdown(table: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "| Adapter | ΔW Params | Total Params | Dice (mean ± 95% CI) | Mean Variance (mean ± 95% CI) |",
        "|:--------|----------:|-------------:|:--------------------:|:-----------------------------:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['adapter']} | {row['delta_w_fmt']} | {row['total_params_fmt']} "
            f"| {row['dice']} | {row['mean_var']} |"
        )
    path = out_dir / "tab3_model_complexity.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Markdown written to %s", path)


def _write_latex(table: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Model complexity versus segmentation performance. "
        r"$\Delta W$ denotes the LoRA adapter parameters. "
        r"Dice and variance are reported as mean $\pm$ 95\% CI "
        r"(bootstrap, $B{=}10\,000$).}",
        r"\label{tab:model_complexity}",
        r"\begin{tabular}{l r r c c}",
        r"\toprule",
        r"Adapter & $\Delta W$ & Total & Dice & Mean Var. \\",
        r"\midrule",
    ]

    # Find best Dice row (excluding BSF)
    non_bsf = table[table["rank_val"] > 0]
    best_dice_idx = -1
    if not non_bsf.empty:
        best_dice_idx = int(non_bsf["rank_val"].iloc[0])
        best_dice_val = 0.0
        for _, row in non_bsf.iterrows():
            d = row["dice"]
            if d != "—":
                val = float(d.split("±")[0].strip())
                if val > best_dice_val:
                    best_dice_val = val
                    best_dice_idx = int(row["rank_val"])

    for i, (_, row) in enumerate(table.iterrows()):
        adapter = row["adapter_latex"]
        dw = row["delta_w_fmt"]
        tp = row["total_params_fmt"]
        dice = row["dice"].replace("±", r"$\pm$")
        mvar = row["mean_var"].replace("±", r"$\pm$")

        if int(row["rank_val"]) == best_dice_idx:
            dice = r"\textbf{" + dice + "}"

        line = f"{adapter} & {dw} & {tp} & {dice} & {mvar} \\\\"

        if i == 0:
            line += "\n" + r"\midrule"

        lines.append(line)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path = out_dir / "tab3_model_complexity.tex"
    path.write_text("\n".join(lines) + "\n")
    logger.info("LaTeX written to %s", path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render(data: InterLoraData, config: dict, out_dir: Path) -> None:
    """Render Tab 3: model complexity vs performance.

    Args:
        data: Aggregated inter-LoRA data container.
        config: Configuration dictionary (unused).
        out_dir: Output directory for table files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Rendering Tab 3 (model complexity) to %s", out_dir)

    table = _build_table(data)

    if table.empty:
        logger.warning("Tab 3: empty table — skipping.")
        return

    _write_csv(table, out_dir)
    _write_markdown(table, out_dir)
    _write_latex(table, out_dir)

    logger.info("Tab 3 complete: %d rows.", len(table))
