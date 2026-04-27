"""Summary metrics per LoRA rank table (Tab 1).

Produces CSV, Markdown, and LaTeX (booktabs + siunitx) renditions of one row
per rank, including the frozen-BSF baseline as rank=0.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import InterLoraData

logger = logging.getLogger(__name__)

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


def _fmt_dice(mean: float, lo: float, hi: float) -> str:
    """Format Dice as 'mean [lo, hi]'.

    Args:
        mean: Point estimate.
        lo: 95% CI lower bound.
        hi: 95% CI upper bound.

    Returns:
        Formatted string, e.g. '0.877 [0.843, 0.908]'.
    """
    if any(math.isnan(v) for v in (mean, lo, hi)):
        return "—"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _rank_label(rank: int) -> str:
    """Return display label for a rank value.

    Args:
        rank: LoRA rank integer; 0 denotes frozen BSF baseline.

    Returns:
        'BSF' for rank 0, else 'r=N'.
    """
    return "BSF" if rank == 0 else f"r={rank}"


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def _build_table(data: InterLoraData) -> pd.DataFrame:
    """Extract and pivot compiled_metrics into one row per rank.

    Args:
        data: Aggregated inter-LoRA data container.

    Returns:
        DataFrame with one row per rank and display-ready string columns.
    """
    df = data.compiled_metrics.copy()

    # Separate per-label Dice rows from aggregated scalar metrics
    dice_labels = ["TC", "WT", "ET"]
    dice_df = df[df["label"].isin(dice_labels)].copy()

    # Scalar columns expected once per rank (not per label): take from 'mean' or
    # first available row per rank to avoid duplication.
    scalar_cols = ["ece", "brier", "cov95_deficit", "pct_bias_dominated", "icc"]
    scalar_src = df.drop_duplicates(subset=["rank"])[["rank"] + scalar_cols].copy()

    rows: list[dict] = []
    for rank in sorted(df["rank"].unique()):
        row: dict = {"rank_val": int(rank), "rank": _rank_label(int(rank))}

        for label, col in [("TC", "dice_tc_mean"), ("WT", "dice_wt_mean"), ("ET", "dice_et_mean")]:
            lbl_mask = (dice_df["rank"] == rank) & (dice_df["label"] == label)
            sub = dice_df[lbl_mask]
            if sub.empty:
                row[col] = "—"
            else:
                r = sub.iloc[0]
                row[col] = _fmt_dice(
                    r.get("dice_mean", float("nan")),
                    r.get("dice_ci_lo", float("nan")),
                    r.get("dice_ci_hi", float("nan")),
                )

        sc = scalar_src[scalar_src["rank"] == rank]
        if sc.empty:
            for c in scalar_cols:
                row[c] = "—"
        else:
            sc_row = sc.iloc[0]
            row["ece"] = _fmt_nan(sc_row.get("ece"), ".2e")
            row["brier"] = _fmt_nan(sc_row.get("brier"), ".2e")
            row["cov95_deficit"] = _fmt_nan(sc_row.get("cov95_deficit"), ".2f")
            pbd = sc_row.get("pct_bias_dominated")
            row["pct_bias_dominated"] = (
                "—"
                if pbd is None or (isinstance(pbd, float) and math.isnan(pbd))
                else f"{pbd:.0f}%"
            )
            row["icc_wt"] = _fmt_nan(sc_row.get("icc"), ".3f")

        rows.append(row)

    result = pd.DataFrame(rows)
    result.sort_values("rank_val", inplace=True)
    result.drop(columns=["rank_val"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------

_DISPLAY_COLUMNS = [
    "rank",
    "dice_tc_mean",
    "dice_wt_mean",
    "dice_et_mean",
    "ece",
    "brier",
    "cov95_deficit",
    "pct_bias_dominated",
    "icc_wt",
]

_HEADER_MAP = {
    "rank": "Rank",
    "dice_tc_mean": "Dice TC (mean [95% CI])",
    "dice_wt_mean": "Dice WT (mean [95% CI])",
    "dice_et_mean": "Dice ET (mean [95% CI])",
    "ece": "ECE",
    "brier": "Brier",
    "cov95_deficit": "Cov-95 deficit",
    "pct_bias_dominated": "Bias-dominated %",
    "icc_wt": "ICC (WT)",
}

_LATEX_HEADER_MAP = {
    "rank": "Rank",
    "dice_tc_mean": r"Dice\textsubscript{TC} (mean [95\% CI])",
    "dice_wt_mean": r"Dice\textsubscript{WT} (mean [95\% CI])",
    "dice_et_mean": r"Dice\textsubscript{ET} (mean [95\% CI])",
    "ece": "ECE",
    "brier": "Brier",
    "cov95_deficit": r"Cov$_{95}$ deficit",
    "pct_bias_dominated": r"Bias-dom. (\%)",
    "icc_wt": r"ICC\textsubscript{WT}",
}

# Columns where higher is better (True) or lower is better (False).
# None = no bolding applied.
_BETTER_HIGH: dict[str, bool | None] = {
    "rank": None,
    "dice_tc_mean": True,
    "dice_wt_mean": True,
    "dice_et_mean": True,
    "ece": False,
    "brier": False,
    "cov95_deficit": False,
    "pct_bias_dominated": False,
    "icc_wt": True,
}


def _write_csv(table: pd.DataFrame, out_dir: Path) -> Path:
    """Write plain CSV.

    Args:
        table: Display-ready DataFrame.
        out_dir: Output directory.

    Returns:
        Path to written file.
    """
    path = out_dir / "tab1_summary_per_rank.csv"
    renamed = table.rename(columns=_HEADER_MAP)
    renamed.to_csv(path, index=False)
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
    path = out_dir / "tab1_summary_per_rank.md"
    renamed = table.rename(columns=_HEADER_MAP)
    md_lines: list[str] = []
    headers = list(renamed.columns)
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in renamed.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in row) + " |")
    path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _extract_numeric(value: str) -> float | None:
    """Parse the leading numeric from a formatted display string.

    Args:
        value: Formatted string like '0.877 [...]' or '3.1e-4' or '87%'.

    Returns:
        Float value, or None if not parseable or em-dash.
    """
    if value == "—":
        return None
    # Strip trailing % and leading whitespace
    clean = value.strip().rstrip("%").split()[0]
    try:
        return float(clean)
    except ValueError:
        return None


def _find_best_row_index(table: pd.DataFrame, col: str, higher_better: bool) -> int | None:
    """Return row index of the best non-baseline value in a column.

    The baseline row (rank == 'BSF') is excluded from consideration.

    Args:
        table: Display-ready DataFrame with a 'rank' column.
        col: Column name to search.
        higher_better: True if larger value is better.

    Returns:
        Integer position in the DataFrame, or None if no numeric values found.
    """
    non_baseline = table[table["rank"] != "BSF"]
    numerics = [(i, _extract_numeric(non_baseline.iloc[i][col])) for i in range(len(non_baseline))]
    valid = [(i, v) for i, v in numerics if v is not None]
    if not valid:
        return None
    best_local_idx = max(valid, key=lambda t: t[1] if higher_better else -t[1])[0]
    # Map local index to global DataFrame index position
    global_positions = list(non_baseline.index)
    return global_positions[best_local_idx]


def _write_latex(table: pd.DataFrame, out_dir: Path) -> Path:
    """Write LaTeX booktabs + siunitx table.

    Best value per column (excluding baseline) is wrapped in \\textbf{}.

    Args:
        table: Display-ready DataFrame.
        out_dir: Output directory.

    Returns:
        Path to written file.
    """
    path = out_dir / "tab1_summary_per_rank.tex"
    cols = _DISPLAY_COLUMNS

    # Pre-compute best positions
    best_positions: dict[str, int | None] = {}
    for col, better in _BETTER_HIGH.items():
        if better is None or col not in table.columns:
            best_positions[col] = None
        else:
            best_positions[col] = _find_best_row_index(table, col, better)

    n_cols = len(cols)
    col_spec = "l" + "c" * (n_cols - 1)

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Summary metrics per LoRA rank. Best value per column "
        r"(excluding BSF baseline) is \textbf{bolded}.}",
        r"  \label{tab:summary_per_rank}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    # Header row
    header_cells = [_LATEX_HEADER_MAP.get(c, c) for c in cols]
    lines.append("    " + " & ".join(header_cells) + r" \\")
    lines.append(r"    \midrule")

    # Data rows
    for pos, (_, row) in enumerate(table.iterrows()):
        cells: list[str] = []
        is_baseline = row["rank"] == "BSF"
        for col in cols:
            val = str(row[col])
            should_bold = (
                not is_baseline
                and best_positions.get(col) is not None
                and row.name == best_positions[col]
            )
            cells.append(rf"\textbf{{{val}}}" if should_bold else val)
        lines.append("    " + " & ".join(cells) + r" \\")
        # Thin rule after baseline row
        if is_baseline:
            lines.append(r"    \midrule")

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
    """Render Tab 1: summary metrics per rank.

    Writes tab1_summary_per_rank.{csv,md,tex} to out_dir.

    Args:
        data: Aggregated inter-LoRA data container.
        config: Plot configuration dictionary (unused; reserved for future
            styling overrides).
        out_dir: Directory where output files are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Rendering Tab 1 (summary per rank) to %s", out_dir)

    table = _build_table(data)

    if table.empty:
        logger.warning("Tab 1: compiled_metrics produced an empty table — skipping.")
        return

    _write_csv(table, out_dir)
    _write_markdown(table, out_dir)
    _write_latex(table, out_dir)

    logger.info("Tab 1 complete: %d rows, %d columns.", len(table), len(table.columns))
