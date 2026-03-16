#!/usr/bin/env python
# experiments/lora/analysis/generate_tables.py
"""Comprehensive table generation for LoRA ablation study.

This module generates publication-ready tables in CSV and LaTeX formats,
consolidating all metrics from the ablation study.

Tables generated:
1. comprehensive_results.csv - All metrics in CSV format
2. comprehensive_table.tex - Publication-ready LaTeX table

Usage:
    python -m experiments.lora.analysis.generate_tables \
        --config experiments/lora_ablation/config/ablation.yaml
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ALL_SEMANTIC_FEATURES = ["volume", "location", "shape"]


def _detect_available_features(metrics: dict[str, dict]) -> list[str]:
    """Detect which semantic features have probe results across conditions.

    Args:
        metrics: Dict mapping condition name to metrics dict.

    Returns:
        Ordered list of feature names that have at least one non-None R2 value.
    """
    available = []
    for feat in ALL_SEMANTIC_FEATURES:
        key = f"r2_{feat}"
        if any(m.get(key) is not None for m in metrics.values()):
            available.append(feat)
    return available


def load_all_metrics(config: dict) -> dict[str, dict]:
    """Load all metrics for all conditions.

    Consolidates:
    - Training metrics (training_summary.yaml)
    - Probe metrics (metrics.json)
    - Test Dice (test_dice_men.json)

    Args:
        config: Experiment configuration.

    Returns:
        Dict mapping condition name to metrics dict.
    """
    output_dir = Path(config["experiment"]["output_dir"])
    all_metrics = {}

    for cond in config["conditions"]:
        condition_name = cond["name"]
        cond_dir = output_dir / "conditions" / condition_name

        metrics = {
            "condition": condition_name,
            "lora_rank": cond.get("lora_rank"),
            "use_dora": cond.get("use_dora", False),
            "skip_training": cond.get("skip_training", False),
        }

        # Load training summary
        summary_path = cond_dir / "training_summary.yaml"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = yaml.safe_load(f)
            metrics["val_dice"] = summary.get("best_val_dice")
            metrics["best_epoch"] = summary.get("best_epoch")
            metrics["training_time_minutes"] = summary.get("training_time_minutes")
            param_counts = summary.get("param_counts", {})
            metrics["params_encoder"] = param_counts.get("encoder", 0)
            metrics["params_decoder"] = param_counts.get("decoder", 0)
            metrics["params_semantic"] = param_counts.get("semantic_heads", 0)
            metrics["params_total"] = param_counts.get("total", 0)

        # Load probe metrics
        metrics_path = cond_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                probe_metrics = json.load(f)
            # Linear probe R2
            metrics["r2_volume"] = probe_metrics.get("r2_volume")
            metrics["r2_location"] = probe_metrics.get("r2_location")
            metrics["r2_shape"] = probe_metrics.get("r2_shape")
            metrics["r2_mean"] = probe_metrics.get("r2_mean")
            # GP-RBF probe R2
            metrics["r2_volume_rbf"] = probe_metrics.get("r2_volume_rbf")
            metrics["r2_location_rbf"] = probe_metrics.get("r2_location_rbf")
            metrics["r2_shape_rbf"] = probe_metrics.get("r2_shape_rbf")
            metrics["r2_mean_rbf"] = probe_metrics.get("r2_mean_rbf")
            # Feature quality
            metrics["variance_mean"] = probe_metrics.get("variance_mean")
            metrics["variance_min"] = probe_metrics.get("variance_min")

        # Load BraTS-MEN test Dice
        men_dice_path = cond_dir / "test_dice_men.json"
        if men_dice_path.exists():
            with open(men_dice_path) as f:
                men_dice = json.load(f)
            metrics["dice_men_mean"] = men_dice.get("dice_mean")
            metrics["dice_men_TC"] = men_dice.get("dice_TC")
            metrics["dice_men_WT"] = men_dice.get("dice_WT")
            metrics["dice_men_ET"] = men_dice.get("dice_ET")
            metrics["dice_men_std"] = men_dice.get("dice_std")

        all_metrics[condition_name] = metrics
        logger.info(f"Loaded metrics for {condition_name}")

    return all_metrics


def generate_comprehensive_csv(
    metrics: dict[str, dict],
    config: dict,
    output_path: Path,
) -> None:
    """Generate comprehensive CSV with all metrics.

    Columns:
    - Condition info: condition, lora_rank, use_dora
    - BraTS-MEN Dice: dice_men_mean, dice_men_TC, dice_men_WT, dice_men_ET
    - Linear probe R2: r2_volume, r2_location, r2_shape, r2_mean
    - GP-RBF probe R2: r2_volume_rbf, r2_location_rbf, r2_shape_rbf, r2_mean_rbf
    - Parameters: params_encoder, params_decoder, params_total
    """
    rows = []
    for cond in config["conditions"]:
        name = cond["name"]
        if name in metrics:
            rows.append(metrics[name])

    df = pd.DataFrame(rows)

    available_features = _detect_available_features(metrics)
    linear_r2_cols = [f"r2_{f}" for f in available_features] + ["r2_mean"]
    rbf_r2_cols = [f"r2_{f}_rbf" for f in available_features] + ["r2_mean_rbf"]

    column_order = [
        "condition",
        "lora_rank",
        "use_dora",
        "skip_training",
        # BraTS-MEN Dice
        "dice_men_mean",
        "dice_men_TC",
        "dice_men_WT",
        "dice_men_ET",
        "dice_men_std",
        # Linear probe R2 (dynamic)
        *linear_r2_cols,
        # GP-RBF probe R2 (dynamic)
        *rbf_r2_cols,
        # Parameters
        "params_encoder",
        "params_decoder",
        "params_semantic",
        "params_total",
        # Training info
        "val_dice",
        "best_epoch",
        "training_time_minutes",
        # Feature quality
        "variance_mean",
        "variance_min",
    ]

    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]

    df.to_csv(output_path, index=False)
    logger.info(f"Saved comprehensive CSV to {output_path}")


def generate_comprehensive_latex(
    metrics: dict[str, dict],
    config: dict,
    output_path: Path,
) -> None:
    """Generate publication-ready LaTeX table.

    Dynamically adjusts columns based on which semantic features (volume,
    location, shape) have probe results. Works for both v1 (all three) and
    v2 (volume only) configs.
    """
    available_features = _detect_available_features(metrics)

    FEATURE_LABELS = {"volume": "Vol", "location": "Loc", "shape": "Shp"}
    feat_headers = [FEATURE_LABELS[f] for f in available_features]
    n_r2_cols = len(available_features) + 1  # +1 for mean

    n_dice_cols = 4
    n_param_cols = 2
    col_spec = (
        "l|"
        + "c" * n_dice_cols
        + "|"
        + "c" * n_r2_cols
        + "|"
        + "c" * n_r2_cols
        + "|"
        + "r" * n_param_cols
    )

    feat_header_str = " & ".join(feat_headers)
    r2_sub_header = f"{feat_header_str} & $\\mu$"

    lines = [
        r"% Auto-generated by generate_tables.py",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        r"",
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Comprehensive comparison of encoder adaptation strategies. "
        r"BraTS-MEN Dice scores measure segmentation quality on the meningioma test set. "
        r"Linear and GP-RBF probe $R^2$ measure semantic feature "
        r"predictability from encoder representations.}",
        r"\label{tab:lora_ablation_comprehensive}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & \multicolumn{{{n_dice_cols}}}{{c|}}{{BraTS-MEN Dice}} "
        rf"& \multicolumn{{{n_r2_cols}}}{{c|}}{{Linear $R^2$}} "
        rf"& \multicolumn{{{n_r2_cols}}}{{c|}}{{GP-RBF $R^2$}} "
        rf"& \multicolumn{{{n_param_cols}}}{{c}}{{Params}} \\",
        rf"Condition & Mean & TC & WT & ET "
        rf"& {r2_sub_header} & {r2_sub_header} & Total & Train \\",
        r"\midrule",
    ]

    for cond in config["conditions"]:
        name = cond["name"]
        if name not in metrics:
            continue

        m = metrics[name]

        if name == "baseline_frozen":
            label = r"Frozen (original)"
        elif name == "baseline":
            label = r"Baseline"
        elif m.get("use_dora"):
            rank = cond.get("lora_rank", "?")
            label = f"DoRA $r={rank}$"
        else:
            rank = cond.get("lora_rank", "?")
            label = f"LoRA $r={rank}$"

        def fmt_dice(val: float | None) -> str:
            return f"{val:.3f}" if val is not None else "---"

        def fmt_r2(val: float | None) -> str:
            return f"{val:.3f}" if val is not None else "---"

        men_mean = fmt_dice(m.get("dice_men_mean"))
        men_tc = fmt_dice(m.get("dice_men_TC"))
        men_wt = fmt_dice(m.get("dice_men_WT"))
        men_et = fmt_dice(m.get("dice_men_ET"))

        linear_vals = [fmt_r2(m.get(f"r2_{f}")) for f in available_features]
        linear_vals.append(fmt_r2(m.get("r2_mean")))
        rbf_vals = [fmt_r2(m.get(f"r2_{f}_rbf")) for f in available_features]
        rbf_vals.append(fmt_r2(m.get("r2_mean_rbf")))

        total_params = m.get("params_total", 0) or 0
        train_params = (
            (m.get("params_encoder", 0) or 0)
            + (m.get("params_decoder", 0) or 0)
            + (m.get("params_semantic", 0) or 0)
        )

        def fmt_params(p: int | float) -> str:
            if p >= 1e6:
                return f"{p / 1e6:.1f}M"
            elif p >= 1e3:
                return f"{p / 1e3:.0f}K"
            else:
                return str(int(p))

        total_str = fmt_params(total_params) if total_params > 0 else "62M"
        train_str = fmt_params(train_params) if train_params > 0 else "0"

        linear_str = " & ".join(linear_vals)
        rbf_str = " & ".join(rbf_vals)

        row = (
            f"{label} & "
            f"{men_mean} & {men_tc} & {men_wt} & {men_et} & "
            f"{linear_str} & "
            f"{rbf_str} & "
            f"{total_str} & {train_str} \\\\"
        )
        lines.append(row)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved LaTeX table to {output_path}")


def generate_simplified_latex(
    metrics: dict[str, dict],
    config: dict,
    output_path: Path,
) -> None:
    """Generate simplified LaTeX table (main paper version).

    Focused on key results:
    - Condition, BraTS-MEN Dice (mean), Linear R2 (mean), Trainable Params
    """
    lines = [
        r"% Simplified table for main paper",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        r"",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Encoder adaptation comparison. Dice on BraTS-MEN test set, "
        r"linear probe $R^2$ for semantic feature prediction, and trainable parameters.}",
        r"\label{tab:lora_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Condition & Dice & $R^2_\text{mean}$ & Trainable \\",
        r"\midrule",
    ]

    for cond in config["conditions"]:
        name = cond["name"]
        if name not in metrics:
            continue

        m = metrics[name]

        # Format condition label
        if name == "baseline_frozen":
            label = "Frozen"
        elif name == "baseline":
            label = "Baseline"
        elif m.get("use_dora"):
            rank = cond.get("lora_rank", "?")
            label = f"DoRA-{rank}"
        else:
            rank = cond.get("lora_rank", "?")
            label = f"LoRA-{rank}"

        dice = m.get("dice_men_mean")
        dice_str = f"{dice:.3f}" if dice is not None else "---"

        r2 = m.get("r2_mean")
        r2_str = f"{r2:.3f}" if r2 is not None else "---"

        train_params = (
            (m.get("params_encoder", 0) or 0)
            + (m.get("params_decoder", 0) or 0)
            + (m.get("params_semantic", 0) or 0)
        )
        if train_params >= 1e6:
            params_str = f"{train_params / 1e6:.1f}M"
        elif train_params >= 1e3:
            params_str = f"{train_params / 1e3:.0f}K"
        else:
            params_str = str(int(train_params))

        lines.append(f"{label} & {dice_str} & {r2_str} & {params_str} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved simplified LaTeX table to {output_path}")


def main(config_path: str) -> None:
    """Generate all tables.

    Args:
        config_path: Path to ablation configuration.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["experiment"]["output_dir"])

    logger.info("=" * 60)
    logger.info("Generating Comprehensive Tables")
    logger.info("=" * 60)

    # Load all metrics
    metrics = load_all_metrics(config)

    if not metrics:
        logger.error("No metrics found. Run the experiment first.")
        return

    # Generate tables
    generate_comprehensive_csv(metrics, config, output_dir / "comprehensive_results.csv")
    generate_comprehensive_latex(metrics, config, output_dir / "comprehensive_table.tex")
    generate_simplified_latex(metrics, config, output_dir / "simplified_table.tex")

    logger.info("")
    logger.info("Generated files:")
    logger.info("  - comprehensive_results.csv")
    logger.info("  - comprehensive_table.tex")
    logger.info("  - simplified_table.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comprehensive tables for LoRA ablation study"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lora_ablation/config/ablation.yaml",
        help="Path to ablation configuration file",
    )

    args = parser.parse_args()
    main(args.config)
