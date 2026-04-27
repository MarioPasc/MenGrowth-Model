"""Paired contrasts vs frozen-BSF baseline table (Tab 2).

Produces CSV, Markdown, and LaTeX (booktabs) renditions of one row per
rank × label, reporting delta Dice, 95% CI, Wilcoxon p-values (raw and
Holm-Bonferroni adjusted), and Cohen's d. Cells with p_holm < 0.05 AND
|d| >= 0.5 are highlighted in the LaTeX output.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import InterLoraData

logger = logging.getLogger(__name__)

# Threshold constants for LaTeX cell highlighting.
_P_HOLM_THRESHOLD: float = 0.05
_COHENS_D_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_nan(value: float | None, fmt: str) -> str:
    """Return formatted string or '—' for NaN/None values.

    Args:
        value: Numeric value to format.
        fmt: Python format spec string (e.g. '.3f').

    Returns:
        Formatted string or em-dash on NaN/None.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return format(value, fmt)


def _fmt_p(value: float | None) -> str:
    """Format a p-value with appropriate precision.

    Values below 0.001 are shown as '<0.001'; others as '.3f'.

    Args:
        value: p-value in [0, 1].

    Returns:
        Formatted string or em-dash on NaN/None.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _fmt_delta_ci(lo: float, hi: float) -> str:
    """Format a 95% CI as '[lo, hi]'.

    Args:
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Formatted string, e.g. '[0.021, 0.054]', or '—' on NaN.
    """
    if math.isnan(lo) or math.isnan(hi):
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def _rank_label(rank: int) -> str:
    """Return 'r=N' display label for a rank.

    Args:
        rank: LoRA rank integer (must be > 0 for this table).

    Returns:
        Display string such as 'r=8'.
    """
    return f"r={rank}"


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def _build_table(data: InterLoraData) -> pd.DataFrame:
    """Extract paired-contrast rows from compiled_metrics.

    Excludes the baseline row (rank == 0) and retains only rows with
    TC / WT / ET labels.

    Args:
        data: Aggregated inter-LoRA data container.

    Returns:
        DataFrame with one row per (rank, label) pair, with display-ready
        string columns.
    """
    df = data.compiled_metrics.copy()

    # Retain only non-baseline rows with a recognised label.
    labels_of_interest = {"TC", "WT", "ET"}
    mask = (df["rank"] != 0) & (df["label"].isin(labels_of_interest))
    sub = df[mask].copy()

    if sub.empty:
        logger.warning("Tab 2: no non-baseline rows found in compiled_metrics.")
        return pd.DataFrame()

    rows: list[dict] = []
    for _, r in sub.iterrows():
        delta = r.get("delta_vs_baseline", float("nan"))
        ci_lo = r.get("delta_ci_lo", float("nan"))
        ci_hi = r.get("delta_ci_hi", float("nan"))
        p_raw = r.get("p_wilcoxon_raw", float("nan"))
        p_holm = r.get("p_wilcoxon_holm", float("nan"))
        d = r.get("cohens_d", float("nan"))

        rows.append(
            {
                "rank_val": int(r["rank"]),
                "rank": _rank_label(int(r["rank"])),
                "label": str(r["label"]),
                "delta_dice_mean": _fmt_nan(delta, "+.3f"),
                "delta_95ci": _fmt_delta_ci(
                    float(ci_lo)
                    if not (isinstance(ci_lo, float) and math.isnan(ci_lo))
                    else float("nan"),
                    float(ci_hi)
                    if not (isinstance(ci_hi, float) and math.isnan(ci_hi))
                    else float("nan"),
                ),
                "p_raw": _fmt_p(
                    float(p_raw) if not (isinstance(p_raw, float) and math.isnan(p_raw)) else None
                ),
                "p_holm": _fmt_p(
                    float(p_holm)
                    if not (isinstance(p_holm, float) and math.isnan(p_holm))
                    else None
                ),
                "cohens_d": _fmt_nan(d, ".2f"),
                # Raw numerics retained for LaTeX highlighting logic.
                "_p_holm_num": float(p_holm)
                if not (isinstance(p_holm, float) and math.isnan(p_holm))
                else float("nan"),
                "_d_num": float(d)
                if not (isinstance(d, float) and math.isnan(d))
                else float("nan"),
            }
        )

    result = pd.DataFrame(rows)
    # Sort by rank ascending, then label in TC/WT/ET order.
    label_order = {"TC": 0, "WT": 1, "ET": 2}
    result["_label_sort"] = result["label"].map(label_order).fillna(99).astype(int)
    result.sort_values(["rank_val", "_label_sort"], inplace=True)
    result.drop(columns=["rank_val", "_label_sort"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------

_DISPLAY_COLUMNS = [
    "rank",
    "label",
    "delta_dice_mean",
    "delta_95ci",
    "p_raw",
    "p_holm",
    "cohens_d",
]

_HEADER_MAP = {
    "rank": "Rank",
    "label": "Label",
    "delta_dice_mean": "ΔDice (mean)",
    "delta_95ci": "ΔDice 95% CI",
    "p_raw": "p (raw)",
    "p_holm": "p (Holm)",
    "cohens_d": "Cohen's d",
}

_LATEX_HEADER_MAP = {
    "rank": "Rank",
    "label": "Label",
    "delta_dice_mean": r"$\Delta$Dice",
    "delta_95ci": r"$\Delta$Dice 95\% CI",
    "p_raw": r"$p_\text{raw}$",
    "p_holm": r"$p_\text{Holm}$",
    "cohens_d": r"$d$",
}


def _write_csv(table: pd.DataFrame, out_dir: Path) -> Path:
    """Write plain CSV with display columns only.

    Args:
        table: Display-ready DataFrame (may include internal '_*' columns).
        out_dir: Output directory.

    Returns:
        Path to written file.
    """
    path = out_dir / "tab2_paired_vs_baseline.csv"
    display = table[_DISPLAY_COLUMNS].rename(columns=_HEADER_MAP)
    display.to_csv(path, index=False)
    logger.info("Wrote %s", path)
    return path


def _write_markdown(table: pd.DataFrame, out_dir: Path) -> Path:
    """Write Markdown table.

    Args:
        table: Display-ready DataFrame.
        out_dir: Output directory.

    Returns:
        Path to written file.
    """
    path = out_dir / "tab2_paired_vs_baseline.md"
    display = table[_DISPLAY_COLUMNS].rename(columns=_HEADER_MAP)
    headers = list(display.columns)
    md_lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in row) + " |")
    path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _is_highlighted(p_holm: float, d: float) -> bool:
    """Return True if the cell meets both highlighting criteria.

    Args:
        p_holm: Holm-adjusted p-value (may be NaN).
        d: Cohen's d (may be NaN).

    Returns:
        True iff p_holm < threshold AND |d| >= threshold.
    """
    if math.isnan(p_holm) or math.isnan(d):
        return False
    return p_holm < _P_HOLM_THRESHOLD and abs(d) >= _COHENS_D_THRESHOLD


def _write_latex(table: pd.DataFrame, out_dir: Path) -> Path:
    """Write LaTeX booktabs table with conditional cell highlighting.

    Rows where p_holm < 0.05 AND |Cohen's d| >= 0.5 are wrapped in
    \\cellcolor{green!15} on every cell. Requires the 'colortbl' and
    'xcolor' packages in the document preamble.

    Args:
        table: Display-ready DataFrame (must include '_p_holm_num' and
            '_d_num' columns for highlighting logic).
        out_dir: Output directory.

    Returns:
        Path to written file.
    """
    path = out_dir / "tab2_paired_vs_baseline.tex"
    cols = _DISPLAY_COLUMNS
    n_cols = len(cols)
    col_spec = "ll" + "c" * (n_cols - 2)

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Paired contrasts vs.\ frozen-BSF baseline. "
        r"Rows with $p_\text{Holm} < 0.05$ and $|d| \geq 0.5$ are "
        r"\colorbox{green!15}{highlighted}.}",
        r"  \label{tab:paired_vs_baseline}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    header_cells = [_LATEX_HEADER_MAP.get(c, c) for c in cols]
    lines.append("    " + " & ".join(header_cells) + r" \\")
    lines.append(r"    \midrule")

    prev_rank: str | None = None
    for _, row in table.iterrows():
        p_holm_num = row.get("_p_holm_num", float("nan"))
        d_num = row.get("_d_num", float("nan"))
        highlight = _is_highlighted(float(p_holm_num), float(d_num))

        cells: list[str] = []
        for col in cols:
            val = str(row[col])
            if highlight:
                cells.append(rf"\cellcolor{{green!15}}{val}")
            else:
                cells.append(val)

        # Insert thin rule between rank groups for readability.
        current_rank = str(row["rank"])
        if prev_rank is not None and current_rank != prev_rank:
            lines.append(r"    \midrule")
        prev_rank = current_rank

        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render(data: InterLoraData, config: dict, out_dir: Path) -> None:
    """Render Tab 2: paired contrasts vs frozen-BSF baseline.

    Writes tab2_paired_vs_baseline.{csv,md,tex} to out_dir.

    Args:
        data: Aggregated inter-LoRA data container.
        config: Plot configuration dictionary (unused; reserved for future
            styling overrides).
        out_dir: Directory where output files are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Rendering Tab 2 (paired contrasts) to %s", out_dir)

    table = _build_table(data)

    if table.empty:
        logger.warning("Tab 2: empty table produced — skipping file writes.")
        return

    _write_csv(table, out_dir)
    _write_markdown(table, out_dir)
    _write_latex(table, out_dir)

    n_highlighted = sum(
        _is_highlighted(float(row["_p_holm_num"]), float(row["_d_num"]))
        for _, row in table.iterrows()
    )
    logger.info(
        "Tab 2 complete: %d rows, %d highlighted (p_Holm<%.2f & |d|>=%.1f).",
        len(table),
        n_highlighted,
        _P_HOLM_THRESHOLD,
        _COHENS_D_THRESHOLD,
    )
