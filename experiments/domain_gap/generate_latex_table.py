"""Generate LaTeX table from domain gap results (CPU-only).

Supports 2-domain (GLI + MEN) and 3-domain (GLI + MEN + MenGrowth) modes.

2-domain (``--datasets gli men``): Original table format unchanged.
3-domain (``--datasets gli men mengrowth``): Wider table with per-dataset
columns, pairwise metric rows, and Kruskal-Wallis column.

Usage:
    python -m experiments.domain_gap.generate_latex_table --output-dir <path> [--datasets gli men mengrowth]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Hardcoded MEN probe R² from baseline_frozen results
MEN_PROBE_R2 = {
    "volume": 0.179,
    "location": -0.257,
    "shape": -0.500,
}

_DOMAIN_DISPLAY = {"gli": "BraTS-GLI", "men": "BraTS-MEN", "mengrowth": "MenGrowth"}
_PAIR_DISPLAY = {
    "gli_men": r"GLI\(\leftrightarrow\)MEN",
    "gli_mengrowth": r"GLI\(\leftrightarrow\)MG",
    "men_mengrowth": r"MEN\(\leftrightarrow\)MG",
}


def _fmt_mean_std(mean: float, std: float) -> str:
    """Format mean +/- std with 3 decimal places."""
    return f"{mean:.3f} $\\pm$ {std:.2f}"


def _fmt_bold(text: str) -> str:
    """Wrap text in LaTeX bold."""
    return f"\\textbf{{{text}}}"


def _p_to_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def _subject_avg_dice(dice_data: dict) -> list[dict]:
    """Average per-scan Dice by subject_id (for longitudinal data)."""
    entries = dice_data["per_subject"]
    if not entries or "subject_id" not in entries[0]:
        return entries

    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        grouped[e["subject_id"]].append(e)

    averaged = []
    for sid, scans in sorted(grouped.items()):
        avg = {"subject_id": sid}
        for key in ["dice_TC", "dice_WT", "dice_ET"]:
            avg[key] = float(np.mean([s[key] for s in scans]))
        averaged.append(avg)
    return averaged


def _load_data(output_dir: Path, datasets: list[str]) -> dict:
    """Load all required data for table generation.

    Supports both new (pairwise_metrics.json) and legacy (domain_metrics.json)
    output formats.

    Args:
        output_dir: Root output directory.
        datasets: Dataset keys.

    Returns:
        Dict with dice, pairwise_metrics, per_dataset_metrics.
    """
    result: dict = {"dice": {}}

    for ds in datasets:
        dice_path = output_dir / "dice" / f"{ds}_dice.json"
        if dice_path.exists():
            with open(dice_path) as f:
                result["dice"][ds] = json.load(f)

    # Pairwise metrics
    pw_path = output_dir / "metrics" / "pairwise_metrics.json"
    legacy_path = output_dir / "metrics" / "domain_metrics.json"

    if pw_path.exists():
        with open(pw_path) as f:
            result["pairwise_metrics"] = json.load(f)
    elif legacy_path.exists():
        with open(legacy_path) as f:
            result["pairwise_metrics"] = {"gli_men": json.load(f)}
    else:
        result["pairwise_metrics"] = {}

    # Per-dataset metrics
    pd_path = output_dir / "metrics" / "per_dataset_metrics.json"
    if pd_path.exists():
        with open(pd_path) as f:
            result["per_dataset_metrics"] = json.load(f)
    elif legacy_path.exists():
        with open(legacy_path) as f:
            legacy = json.load(f)
        result["per_dataset_metrics"] = {
            "gli": {"effective_rank": legacy.get("effective_rank_gli", 0)},
            "men": {"effective_rank": legacy.get("effective_rank_men", 0)},
        }
    else:
        result["per_dataset_metrics"] = {}

    return result


# ============================================================================
# 2-domain table (original format)
# ============================================================================


def generate_table_2domain(output_dir: Path) -> str:
    """Generate 2-domain LaTeX table (original format).

    Args:
        output_dir: Root output directory from run_domain_gap.py.

    Returns:
        LaTeX table as a string.
    """
    data = _load_data(output_dir, ["gli", "men"])
    gli_dice = data["dice"]["gli"]
    men_dice = data["dice"]["men"]

    # Get metrics — support both new and legacy format
    pw = data["pairwise_metrics"]
    if "gli_men" in pw:
        metrics = pw["gli_men"]
        # Merge effective rank from per_dataset_metrics or legacy
        pd_metrics = data.get("per_dataset_metrics", {})
        metrics.setdefault("effective_rank_gli", pd_metrics.get("gli", {}).get("effective_rank", 0))
        metrics.setdefault("effective_rank_men", pd_metrics.get("men", {}).get("effective_rank", 0))
    else:
        metrics = pw

    gs = gli_dice["summary"]
    ms = men_dice["summary"]

    dice_p = {}
    for cls in ["dice_TC", "dice_WT", "dice_ET"]:
        gli_vals = [s[cls] for s in gli_dice["per_subject"]]
        men_vals = [s[cls] for s in men_dice["per_subject"]]
        _, p = stats.ttest_ind(gli_vals, men_vals, equal_var=False)
        dice_p[cls] = p

    mmd_p = metrics.get("mmd_pvalue", 1.0)
    mmd_p_str = f"p = {mmd_p:.3f}" if mmd_p >= 0.001 else "p < 0.001"

    rows = []

    for cls, label in [("dice_TC", "Dice TC"), ("dice_WT", "Dice WT"), ("dice_ET", "Dice ET")]:
        gli_str = _fmt_mean_std(gs[f"{cls}_mean"], gs[f"{cls}_std"])
        men_str = _fmt_mean_std(ms[f"{cls}_mean"], ms[f"{cls}_std"])
        stars = _p_to_stars(dice_p[cls])
        rows.append(f"    {label} & {gli_str} & {men_str} & {stars} \\\\")

    rows.append("    \\midrule")

    mmd_val = f"{metrics.get('mmd_sq', 0):.2f}"
    rows.append(
        f"    MMD$^2$ (RBF) & \\multicolumn{{3}}{{c}}{{{_fmt_bold(mmd_val)} ({mmd_p_str})}} \\\\"
    )

    cka_val = f"{metrics.get('cka', 0):.2f}"
    rows.append(f"    CKA & \\multicolumn{{3}}{{c}}{{{cka_val}}} \\\\")

    pad_val = f"{metrics.get('proxy_a_distance', 0):.2f}"
    rows.append(f"    Proxy A-distance & \\multicolumn{{3}}{{c}}{{{_fmt_bold(pad_val)}}} \\\\")

    clf_acc = f"{metrics.get('classifier_accuracy', 0) * 100:.1f}\\%"
    rows.append(f"    Classifier Acc. & \\multicolumn{{3}}{{c}}{{{_fmt_bold(clf_acc)}}} \\\\")

    eff_gli = f"{metrics.get('effective_rank_gli', 0):.1f}"
    eff_men = f"{metrics.get('effective_rank_men', 0):.1f}"
    rows.append(f"    Effective Rank & {eff_gli} & {eff_men} & \\\\")

    rows.append("    \\midrule")

    for key, label in [("volume", "vol"), ("location", "loc"), ("shape", "shape")]:
        r2 = MEN_PROBE_R2[key]
        r2_str = f"{r2:.3f}"
        rows.append(f"    Probe $R^2$ ({label}) & & {r2_str} & \\\\")

    row_block = "\n".join(rows)

    table = f"""\\begin{{table}}[t]
  \\centering
  \\caption{{Domain gap analysis: frozen BrainSegFounder on BraTS-GLI vs.\\ BraTS-MEN.
    Dice scores (mean $\\pm$ std) quantify segmentation degradation.
    Distribution metrics confirm significant feature-space separation.
    Probe $R^2$ values from frozen encoder baseline.}}
  \\label{{tab:domain-gap}}
  \\begin{{tabular}}{{lccc}}
    \\toprule
    Metric & BraTS-GLI & BraTS-MEN & $\\Delta$ / $p$ \\\\
    \\midrule
{row_block}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""

    return table


# ============================================================================
# 3-domain table
# ============================================================================


def generate_table_3domain(output_dir: Path, datasets: list[str]) -> str:
    """Generate 3-domain LaTeX table with Kruskal-Wallis column.

    Args:
        output_dir: Root output directory.
        datasets: Ordered list of dataset keys.

    Returns:
        LaTeX table string.
    """
    data = _load_data(output_dir, datasets)
    pw = data["pairwise_metrics"]
    pd_m = data["per_dataset_metrics"]

    # Column headers
    col_headers = " & ".join(_DOMAIN_DISPLAY[ds] for ds in datasets)
    n_data_cols = len(datasets)
    col_spec = "l" + "c" * n_data_cols + "c"  # +1 for KW column

    rows = []

    # --- Dice rows ---
    per_subj = {ds: _subject_avg_dice(data["dice"][ds]) for ds in datasets if ds in data["dice"]}

    for cls, label in [("dice_TC", "Dice TC"), ("dice_WT", "Dice WT"), ("dice_ET", "Dice ET")]:
        cells = []
        for ds in datasets:
            if ds in data["dice"]:
                s = data["dice"][ds]["summary"]
                cells.append(_fmt_mean_std(s[f"{cls}_mean"], s[f"{cls}_std"]))
            else:
                cells.append("---")

        # Kruskal-Wallis
        groups = [[e[cls] for e in per_subj[ds]] for ds in datasets if ds in per_subj]
        if len(groups) >= 2:
            _, kw_p = stats.kruskal(*groups)
            kw_str = _p_to_stars(kw_p)
        else:
            kw_str = ""

        cells_str = " & ".join(cells)
        rows.append(f"    {label} & {cells_str} & {kw_str} \\\\")

    rows.append("    \\midrule")

    # --- Pairwise metric rows ---
    pair_keys = []
    for a_idx in range(len(datasets)):
        for b_idx in range(a_idx + 1, len(datasets)):
            pair_keys.append(f"{datasets[a_idx]}_{datasets[b_idx]}")

    pair_col_headers = " & ".join(_PAIR_DISPLAY.get(pk, pk) for pk in pair_keys)

    # Sub-header for pairwise section
    rows.append(f"    & \\multicolumn{{{len(pair_keys) + 1}}}{{c}}{{}} \\\\")
    rows.append(f"    & {pair_col_headers} & \\\\")
    rows.append("    \\midrule")

    # MMD²
    mmd_cells = []
    for pk in pair_keys:
        if pk in pw:
            val = pw[pk]["mmd_sq"]
            p = pw[pk].get("mmd_pvalue", 1.0)
            stars = _p_to_stars(p)
            mmd_cells.append(f"{val:.2f}{stars}")
        else:
            mmd_cells.append("---")
    mmd_str = " & ".join(mmd_cells)
    rows.append(f"    MMD$^2$ (RBF) & {mmd_str} & \\\\")

    # CKA
    cka_cells = []
    for pk in pair_keys:
        if pk in pw:
            cka_cells.append(f"{pw[pk]['cka']:.2f}")
        else:
            cka_cells.append("---")
    cka_str = " & ".join(cka_cells)
    rows.append(f"    CKA & {cka_str} & \\\\")

    # PAD
    pad_cells = []
    for pk in pair_keys:
        if pk in pw:
            pad_cells.append(f"{pw[pk]['proxy_a_distance']:.2f}")
        else:
            pad_cells.append("---")
    pad_str = " & ".join(pad_cells)
    rows.append(f"    Proxy A-distance & {pad_str} & \\\\")

    # Classifier accuracy
    clf_cells = []
    for pk in pair_keys:
        if pk in pw:
            clf_cells.append(f"{pw[pk]['classifier_accuracy'] * 100:.1f}\\%")
        else:
            clf_cells.append("---")
    clf_str = " & ".join(clf_cells)
    rows.append(f"    Classifier Acc. & {clf_str} & \\\\")

    rows.append("    \\midrule")

    # Effective rank (per-dataset)
    eff_cells = []
    for ds in datasets:
        if ds in pd_m:
            eff_cells.append(f"{pd_m[ds]['effective_rank']:.1f}")
        else:
            eff_cells.append("---")
    eff_str = " & ".join(eff_cells)
    rows.append(f"    Effective Rank & {eff_str} & \\\\")

    row_block = "\n".join(rows)

    table = f"""\\begin{{table}}[t]
  \\centering
  \\caption{{Domain gap analysis: frozen BrainSegFounder across three cohorts.
    Dice scores (mean $\\pm$ std) quantify segmentation performance per domain.
    Kruskal-Wallis (KW) tests the null hypothesis of equal distributions.
    Pairwise metrics quantify feature-space separation between each domain pair.}}
  \\label{{tab:domain-gap}}
  \\begin{{tabular}}{{{col_spec}}}
    \\toprule
    Metric & {col_headers} & KW \\\\
    \\midrule
{row_block}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""

    return table


# ============================================================================
# Unified entry point
# ============================================================================


def generate_table(output_dir: Path, datasets: list[str] | None = None) -> str:
    """Generate LaTeX table, dispatching to 2- or 3-domain format.

    Args:
        output_dir: Root output directory.
        datasets: Dataset keys. Defaults to ["gli", "men"].

    Returns:
        LaTeX table string.
    """
    if datasets is None:
        datasets = ["gli", "men"]

    if len(datasets) <= 2:
        return generate_table_2domain(output_dir)
    else:
        return generate_table_3domain(output_dir, datasets)


def main() -> None:
    """Generate and save LaTeX table."""
    parser = argparse.ArgumentParser(description="Domain gap LaTeX table (CPU)")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory from run_domain_gap.py",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gli", "men"],
        choices=["gli", "men", "mengrowth"],
        help="Datasets to include in the table",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    table_str = generate_table(output_dir, datasets=args.datasets)

    # Save to file
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    out_path = table_dir / "domain_gap.tex"
    out_path.write_text(table_str)
    logger.info(f"Saved: {out_path}")

    # Print to stdout
    print(table_str)


if __name__ == "__main__":
    main()
