"""Table generation for TSI analysis.

Generates Table 1 from the spec: per-stage, per-condition TSI statistics
aggregated across N test scans, with Wilcoxon tests.

Outputs:
- tsi_table.csv (machine-readable)
- tsi_table.tex (LaTeX-formatted for thesis)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from .tsi_analysis import STAGE_META

logger = logging.getLogger(__name__)


def _aggregate_per_stage(
    df: pd.DataFrame,
    thresholds: list[float] = (1.5, 2.0),
) -> pd.DataFrame:
    """Aggregate per-scan TSI statistics by stage and condition.

    For each (stage, condition): compute mean ± SD of per-scan mean_tsi,
    mean of frac_above thresholds, and Wilcoxon test (H0: median TSI = 1)
    across scans.

    Args:
        df: Per-scan DataFrame with columns: scan_id, condition, stage,
            mean_tsi, std_tsi, frac_{tau}, wilcoxon_p.
        thresholds: TSI thresholds.

    Returns:
        Aggregated DataFrame with one row per (stage, condition).
    """
    rows = []
    for (stage, condition), group in df.groupby(["stage", "condition"]):
        meta = STAGE_META.get(int(stage), {})
        mean_tsi_values = group["mean_tsi"].dropna().values

        row = {
            "stage": int(stage),
            "condition": condition,
            "n_channels": meta.get("channels", "?"),
            "resolution": f"{192 // meta.get('downsample', 1)}^3"
            if meta
            else "?",
            "has_lora": meta.get("has_lora", False),
            "n_scans": len(mean_tsi_values),
            "mean_tsi": np.mean(mean_tsi_values) if len(mean_tsi_values) > 0 else np.nan,
            "sd_tsi": np.std(mean_tsi_values, ddof=1)
            if len(mean_tsi_values) > 1
            else 0.0,
        }

        # Aggregated Frac above thresholds
        for tau in thresholds:
            col = f"frac_{tau}"
            if col in group.columns:
                row[f"frac_{tau}"] = group[col].dropna().mean()

        # Wilcoxon across scans: H0: the per-scan mean TSI = 1
        if len(mean_tsi_values) >= 5:
            try:
                stat = scipy.stats.wilcoxon(
                    mean_tsi_values - 1.0,
                    alternative="greater",
                )
                row["wilcoxon_p_scans"] = stat.pvalue
            except ValueError:
                row["wilcoxon_p_scans"] = 1.0
        else:
            row["wilcoxon_p_scans"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _paired_wilcoxon_per_stage(
    frozen_df: pd.DataFrame,
    adapted_df: pd.DataFrame,
) -> pd.DataFrame:
    """Paired Wilcoxon test: adapted vs frozen, per stage.

    For each stage, pairs the per-scan mean TSI under both conditions
    and tests H0: adapted ≤ frozen (one-sided greater).

    Args:
        frozen_df: Per-scan frozen results.
        adapted_df: Per-scan adapted results.

    Returns:
        DataFrame with columns: stage, delta_mean, delta_sd, paired_p.
    """
    rows = []
    for stage in range(5):
        f_stage = frozen_df[frozen_df["stage"] == stage].set_index("scan_id")["mean_tsi"]
        a_stage = adapted_df[adapted_df["stage"] == stage].set_index("scan_id")["mean_tsi"]

        # Align by scan_id and drop pairs where either is NaN
        common = f_stage.index.intersection(a_stage.index)
        frozen_vals = f_stage.loc[common].values
        adapted_vals = a_stage.loc[common].values
        valid = ~(np.isnan(frozen_vals) | np.isnan(adapted_vals))
        frozen_vals = frozen_vals[valid]
        adapted_vals = adapted_vals[valid]

        n = len(frozen_vals)
        if n < 5:
            rows.append({"stage": stage, "n_paired": n, "delta_mean": np.nan, "delta_sd": np.nan, "paired_p": np.nan})
            continue

        delta = adapted_vals - frozen_vals

        try:
            stat = scipy.stats.wilcoxon(delta, alternative="greater")
            p_val = stat.pvalue
        except ValueError:
            p_val = 1.0

        rows.append({
            "stage": stage,
            "n_paired": n,
            "delta_mean": np.mean(delta),
            "delta_sd": np.std(delta, ddof=1),
            "paired_p": p_val,
        })

    return pd.DataFrame(rows)


def _format_pvalue(p: float) -> str:
    """Format p-value for LaTeX table.

    Args:
        p: p-value.

    Returns:
        Formatted string.
    """
    if np.isnan(p):
        return "---"
    if p < 0.001:
        return f"$<$0.001"
    return f"{p:.3f}"


def _generate_latex(
    agg_df: pd.DataFrame,
    paired_df: pd.DataFrame | None,
    rank: int | None,
) -> str:
    """Generate LaTeX table string.

    Args:
        agg_df: Aggregated TSI statistics.
        paired_df: Paired Wilcoxon results (optional).
        rank: LoRA rank (for caption).

    Returns:
        LaTeX table string.
    """
    rank_label = f"r={rank}" if rank else "all ranks"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Tumor Selectivity Index across SwinViT encoder stages (" + rank_label + r").}",
        r"\label{tab:tsi_" + (f"r{rank}" if rank else "cross") + r"}",
        r"\small",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Stage & Condition & $C_s$ & Resolution & Mean TSI $\pm$ SD & Frac($>$1.5) & Frac($>$2.0) & Wilcoxon $p$ \\",
        r"\midrule",
    ]

    for _, row in agg_df.iterrows():
        stage = int(row["stage"])
        cond = row["condition"]
        n_ch = int(row["n_channels"])
        res = row["resolution"]
        mean_sd = f"{row['mean_tsi']:.3f} $\\pm$ {row['sd_tsi']:.3f}"
        frac15 = f"{row.get('frac_1.5', 0):.1%}" if not np.isnan(row.get("frac_1.5", np.nan)) else "---"
        frac20 = f"{row.get('frac_2.0', 0):.1%}" if not np.isnan(row.get("frac_2.0", np.nan)) else "---"
        p_str = _format_pvalue(row.get("wilcoxon_p_scans", np.nan))

        # Bold LoRA stages
        stage_str = f"\\textbf{{{stage}}}" if STAGE_META[stage]["has_lora"] else str(stage)
        cond_clean = cond.replace("adapted_", "Adapted ").replace("frozen", "Frozen")

        lines.append(
            f"  {stage_str} & {cond_clean} & {n_ch} & ${res}$ & {mean_sd} & {frac15} & {frac20} & {p_str} \\\\"
        )

        # Add midrule between stages
        next_rows = agg_df[agg_df.index > _]
        if len(next_rows) > 0:
            next_stage = int(next_rows.iloc[0]["stage"])
            if next_stage != stage:
                lines.append(r"  \midrule")

    lines.append(r"\bottomrule")

    # Paired test footnote
    if paired_df is not None:
        lines.append(r"\end{tabular}")
        lines.append(r"\vspace{0.3em}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"\multicolumn{5}{l}{\textit{Paired Wilcoxon test (Adapted $>$ Frozen):}} \\")
        lines.append(r"Stage & 0 & 1 & 2 & 3 \& 4 \\")
        lines.append(r"\midrule")
        p_vals = []
        for _, row in paired_df.iterrows():
            p_vals.append(_format_pvalue(row["paired_p"]))
        lines.append(f"$p$-value & {' & '.join(p_vals[:3])} & {' / '.join(p_vals[3:])} \\\\")
        lines.append(r"\bottomrule")

    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_all_tables(
    frozen_df: pd.DataFrame,
    adapted_df: pd.DataFrame | None,
    rank: int | None,
    output_dir: Path,
    all_adapted: dict[int, pd.DataFrame] | None = None,
) -> None:
    """Generate CSV and LaTeX tables for one rank or cross-rank comparison.

    Args:
        frozen_df: Per-scan frozen results DataFrame.
        adapted_df: Per-scan adapted results for one rank (None for cross-rank).
        rank: LoRA rank (None for cross-rank).
        output_dir: Directory to write tables into.
        all_adapted: Dict of rank -> adapted_df for cross-rank comparison.
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if adapted_df is not None:
        # Single-rank table
        combined = pd.concat([frozen_df, adapted_df], ignore_index=True)
        agg = _aggregate_per_stage(combined)
        paired = _paired_wilcoxon_per_stage(frozen_df, adapted_df)

        suffix = f"_r{rank}" if rank else ""
        agg.to_csv(tables_dir / f"tsi_table{suffix}.csv", index=False)
        paired.to_csv(tables_dir / f"tsi_paired{suffix}.csv", index=False)

        latex = _generate_latex(agg, paired, rank)
        (tables_dir / f"tsi_table{suffix}.tex").write_text(latex)

        logger.info(f"Saved tables to {tables_dir}")

    elif all_adapted is not None:
        # Cross-rank comparison table
        rows = []
        for r, adf in sorted(all_adapted.items()):
            paired = _paired_wilcoxon_per_stage(frozen_df, adf)
            for _, row in paired.iterrows():
                rows.append({
                    "rank": r,
                    "stage": int(row["stage"]),
                    "delta_mean": row["delta_mean"],
                    "delta_sd": row["delta_sd"],
                    "paired_p": row["paired_p"],
                })

        cross_df = pd.DataFrame(rows)
        cross_df.to_csv(tables_dir / "tsi_cross_rank.csv", index=False)
        logger.info(f"Saved cross-rank table to {tables_dir}")
