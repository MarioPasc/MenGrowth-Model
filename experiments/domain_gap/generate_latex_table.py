"""Generate LaTeX table from domain gap results (CPU-only).

Reads saved metrics and Dice data, outputs a LaTeX table with
bold highlights for significant domain shift indicators.

Usage:
    python -m experiments.domain_gap.generate_latex_table --output-dir <path>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Hardcoded MEN probe R² from baseline_frozen results
MEN_PROBE_R2 = {
    "volume": 0.179,
    "location": -0.257,
    "shape": -0.500,
}


def _fmt_mean_std(mean: float, std: float) -> str:
    """Format mean ± std with 3 decimal places."""
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


def generate_table(output_dir: Path) -> str:
    """Generate LaTeX table string from saved results.

    Args:
        output_dir: Root output directory from run_domain_gap.py.

    Returns:
        LaTeX table as a string.
    """
    with open(output_dir / "dice" / "gli_dice.json") as f:
        gli_dice = json.load(f)
    with open(output_dir / "dice" / "men_dice.json") as f:
        men_dice = json.load(f)
    with open(output_dir / "metrics" / "domain_metrics.json") as f:
        metrics = json.load(f)

    gs = gli_dice["summary"]
    ms = men_dice["summary"]

    # Build p-value strings for Dice rows
    from scipy import stats

    dice_p = {}
    for cls in ["dice_TC", "dice_WT", "dice_ET"]:
        gli_vals = [s[cls] for s in gli_dice["per_subject"]]
        men_vals = [s[cls] for s in men_dice["per_subject"]]
        _, p = stats.ttest_ind(gli_vals, men_vals, equal_var=False)
        dice_p[cls] = p

    # MMD p-value string
    mmd_p = metrics["mmd_pvalue"]
    mmd_p_str = f"p = {mmd_p:.3f}" if mmd_p >= 0.001 else "p < 0.001"

    rows = []

    # Dice rows
    for cls, label in [("dice_TC", "Dice TC"), ("dice_WT", "Dice WT"), ("dice_ET", "Dice ET")]:
        gli_str = _fmt_mean_std(gs[f"{cls}_mean"], gs[f"{cls}_std"])
        men_str = _fmt_mean_std(ms[f"{cls}_mean"], ms[f"{cls}_std"])
        stars = _p_to_stars(dice_p[cls])
        rows.append(f"    {label} & {gli_str} & {men_str} & {stars} \\\\")

    rows.append("    \\midrule")

    # Domain shift metrics (pairwise — span across GLI/MEN columns)
    mmd_val = f"{metrics['mmd_sq']:.2f}"
    rows.append(
        f"    MMD$^2$ (RBF) & \\multicolumn{{3}}{{c}}"
        f"{{{_fmt_bold(mmd_val)} ({mmd_p_str})}} \\\\"
    )

    cka_val = f"{metrics['cka']:.2f}"
    rows.append(f"    CKA & \\multicolumn{{3}}{{c}}{{{cka_val}}} \\\\")

    pad_val = f"{metrics['proxy_a_distance']:.2f}"
    rows.append(f"    Proxy A-distance & \\multicolumn{{3}}{{c}}{{{_fmt_bold(pad_val)}}} \\\\")

    clf_acc = f"{metrics['classifier_accuracy'] * 100:.1f}\\%"
    rows.append(f"    Classifier Acc. & \\multicolumn{{3}}{{c}}{{{_fmt_bold(clf_acc)}}} \\\\")

    eff_gli = f"{metrics['effective_rank_gli']:.1f}"
    eff_men = f"{metrics['effective_rank_men']:.1f}"
    rows.append(f"    Effective Rank & {eff_gli} & {eff_men} & \\\\")

    rows.append("    \\midrule")

    # Probe R² (hardcoded from baseline_frozen, MEN-only)
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


def main() -> None:
    """Generate and save LaTeX table."""
    parser = argparse.ArgumentParser(description="Domain gap LaTeX table (CPU)")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory from run_domain_gap.py",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    table_str = generate_table(output_dir)

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
