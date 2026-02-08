# experiments/lora_ablation/compute_domain_metrics.py
"""Domain gap analysis for LoRA ablation experiments.

Computes domain shift metrics between glioma and meningioma feature
representations across all ablation conditions. Uses pre-extracted
.pt feature files (CPU only, no GPU needed).

Metrics computed per condition:
    - MMD with permutation test (distributional distance)
    - CKA between GLI and MEN features (representation similarity)
    - Domain classifier accuracy (linear separability of domains)
    - Proxy A-distance (domain divergence bound)
    - Effective rank per domain (feature space utilization)
    - CKA drift: CKA(frozen_MEN, adapted_MEN) measuring representation change

Outputs:
    - Per-condition domain_metrics.json
    - Experiment-wide domain_metrics_summary.csv
    - 5 paper-ready figures (PDF + PNG)

Usage:
    # Single experiment
    python -m experiments.lora_ablation.compute_domain_metrics \\
        --output-dir /path/to/results/lora_ablation_semantic_heads

    # All 4 variants
    python -m experiments.lora_ablation.compute_domain_metrics \\
        --output-dir /path/to/results/lora_ablation_semantic_heads --all

    # Via config file
    python -m experiments.lora_ablation.compute_domain_metrics \\
        --config experiments/lora_ablation/config/server/LoRA_semantic_heads_icai.yaml
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from growth.evaluation.latent_quality import (
    DomainShiftMetrics,
    compute_cka,
    compute_domain_shift_metrics,
    compute_effective_rank,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

logger = logging.getLogger(__name__)

# Canonical condition order
CONDITION_ORDER = [
    "baseline_frozen",
    "baseline",
    "lora_r2",
    "lora_r4",
    "lora_r8",
    "lora_r16",
    "lora_r32",
    "dora_r2",
    "dora_r4",
    "dora_r8",
    "dora_r16",
    "dora_r32",
]

CONDITION_COLORS = {
    "baseline_frozen": "#a0a0a0",
    "baseline": "#808080",
    "lora_r2": "#a6cee3",
    "lora_r4": "#1f78b4",
    "lora_r8": "#33a02c",
    "lora_r16": "#ff7f00",
    "lora_r32": "#e31a1c",
    "dora_r2": "#a6cee3",
    "dora_r4": "#1f78b4",
    "dora_r8": "#33a02c",
    "dora_r16": "#ff7f00",
    "dora_r32": "#e31a1c",
}

EXPERIMENT_LABELS = {
    "lora_ablation_semantic_heads": "LoRA + Sem",
    "lora_ablation_no_semantic_heads": "LoRA",
    "dora_ablation_semantic_heads": "DoRA + Sem",
    "dora_ablation_no_semantic_heads": "DoRA",
}

DOMAIN_COLORS = {
    "glioma": "#b2182b",
    "meningioma": "#2166ac",
}


# ─────────────────────────────────────────────────────────────────────
# Feature loading
# ─────────────────────────────────────────────────────────────────────

def load_features(cond_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load pre-extracted GLI and MEN features from a condition directory.

    Returns:
        (gli_features, men_features) as numpy arrays, or None if missing.
    """
    gli_path = cond_dir / "features_glioma.pt"
    men_path = cond_dir / "features_meningioma_subset.pt"

    if not gli_path.exists() or not men_path.exists():
        logger.warning(f"Missing feature files in {cond_dir}")
        return None

    gli_data = torch.load(gli_path, map_location="cpu", weights_only=False)
    men_data = torch.load(men_path, map_location="cpu", weights_only=False)

    gli_feat = gli_data["features"].numpy() if isinstance(gli_data["features"], torch.Tensor) else np.array(gli_data["features"])
    men_feat = men_data["features"].numpy() if isinstance(men_data["features"], torch.Tensor) else np.array(men_data["features"])

    logger.info(f"Loaded features: GLI {gli_feat.shape}, MEN {men_feat.shape} from {cond_dir.name}")
    return gli_feat, men_feat


def discover_conditions(output_dir: Path) -> List[str]:
    """Discover available conditions in output directory."""
    conds_dir = output_dir / "conditions"
    if not conds_dir.exists():
        return []
    conditions = []
    for d in sorted(conds_dir.iterdir()):
        if d.is_dir() and (d / "features_glioma.pt").exists():
            conditions.append(d.name)
    # Sort by canonical order
    return sorted(conditions, key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 999)


def discover_sibling_experiments(output_dir: Path) -> List[Path]:
    """Find sibling experiment directories (for --all mode)."""
    parent = output_dir.parent
    siblings = []
    for name in [
        "lora_ablation_semantic_heads",
        "lora_ablation_no_semantic_heads",
        "dora_ablation_semantic_heads",
        "dora_ablation_no_semantic_heads",
    ]:
        p = parent / name
        if p.exists() and (p / "conditions").exists():
            siblings.append(p)
    return siblings


# ─────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    output_dir: Path,
    conditions: List[str],
) -> Dict[str, dict]:
    """Compute domain shift metrics for all conditions.

    Returns:
        Dict[condition_name -> metrics_dict]
    """
    results = {}
    frozen_men_feat = None

    for cond in conditions:
        cond_dir = output_dir / "conditions" / cond
        loaded = load_features(cond_dir)
        if loaded is None:
            continue

        gli_feat, men_feat = loaded

        # Store frozen baseline MEN features for CKA drift
        if cond == "baseline_frozen":
            frozen_men_feat = men_feat.copy()

        # Core domain shift metrics
        logger.info(f"Computing domain shift metrics for {cond}...")
        dsm = compute_domain_shift_metrics(gli_feat, men_feat)

        metrics = asdict(dsm)

        # CKA drift: how much MEN representation changed from frozen baseline
        if frozen_men_feat is not None and cond != "baseline_frozen":
            n = min(len(frozen_men_feat), len(men_feat))
            cka_drift = compute_cka(frozen_men_feat[:n], men_feat[:n])
            metrics["cka_drift_from_frozen"] = cka_drift
        else:
            metrics["cka_drift_from_frozen"] = 1.0  # frozen vs itself

        # Load R² from metrics_enhanced.json for trade-off analysis
        enhanced_path = cond_dir / "metrics_enhanced.json"
        if enhanced_path.exists():
            with open(enhanced_path) as f:
                enhanced = json.load(f)
            metrics["r2_mean_linear"] = enhanced.get("r2_mean_linear", None)
            metrics["r2_volume_linear"] = enhanced.get("r2_volume_linear", None)

        results[cond] = metrics

        # Save per-condition JSON
        out_path = cond_dir / "domain_metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved {out_path}")

    return results


def save_summary_csv(results: Dict[str, dict], output_dir: Path) -> Path:
    """Save domain metrics summary as CSV."""
    import csv

    csv_path = output_dir / "domain_metrics_summary.csv"
    if not results:
        return csv_path

    # Gather all metric keys
    all_keys = set()
    for m in results.values():
        all_keys.update(m.keys())
    all_keys = sorted(all_keys)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition"] + all_keys)
        for cond in results:
            row = [cond]
            for k in all_keys:
                val = results[cond].get(k)
                if isinstance(val, float):
                    row.append(f"{val:.6f}")
                else:
                    row.append(val if val is not None else "")
            writer.writerow(row)

    logger.info(f"Saved {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────

def _short_label(cond: str) -> str:
    """Shorten condition name for plot labels."""
    return (
        cond.replace("baseline_frozen", "frozen")
        .replace("baseline", "base")
        .replace("lora_", "r")
        .replace("dora_", "dr")
    )


def _save_figure(fig, path: Path) -> None:
    """Save figure as both PDF and PNG."""
    for ext in ["pdf", "png"]:
        out = path.parent / f"{path.stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path.stem}.{{pdf,png}}")


def plot_domain_gap_metrics(
    results: Dict[str, dict],
    output_path: Path,
) -> None:
    """Fig 1: 4-panel bar chart of domain gap metrics."""
    if not HAS_MATPLOTLIB:
        return

    conditions = list(results.keys())
    x = np.arange(len(conditions))
    labels = [_short_label(c) for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#808080") for c in conditions]

    panels = [
        ("MMD²", "mmd", None, "lower = more similar"),
        ("CKA (GLI ↔ MEN)", "cka", 1.0, "higher = more similar"),
        ("Domain Classifier Acc.", "domain_classifier_accuracy", 0.5, "0.5 = chance"),
        ("Proxy A-Distance", "proxy_a_distance", 0.0, "0 = indistinguishable"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, (title, key, ref_val, ref_label) in zip(axes, panels):
        values = [results[c].get(key, 0) for c in conditions]
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

        if ref_val is not None:
            ax.axhline(y=ref_val, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.text(
                len(x) - 0.5, ref_val, ref_label,
                ha="right", va="bottom", fontsize=7, alpha=0.6,
            )

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Domain Gap Metrics Across Conditions", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_umap_grid(
    output_dir: Path,
    results: Dict[str, dict],
    output_path: Path,
) -> None:
    """Fig 2: UMAP grid for frozen baseline vs best adapted condition."""
    try:
        from experiments.lora_ablation.domain_visualizations import plot_domain_umap
    except ImportError:
        logger.warning("Cannot import plot_domain_umap, skipping UMAP grid")
        return

    conditions = list(results.keys())

    # Pick frozen + best adapted (highest CKA drift = most changed, or lora_r8 as default)
    targets = ["baseline_frozen"]
    adapted = [c for c in conditions if c.startswith(("lora_r", "dora_r"))]
    if adapted:
        # Pick the one with highest r2_mean_linear (best adapted)
        best = max(adapted, key=lambda c: results[c].get("r2_mean_linear", 0) or 0)
        targets.append(best)
    elif "baseline" in conditions:
        targets.append("baseline")

    for cond in targets:
        cond_dir = output_dir / "conditions" / cond
        loaded = load_features(cond_dir)
        if loaded is None:
            continue
        gli_feat, men_feat = loaded
        plot_domain_umap(
            men_feat, gli_feat,
            output_path.parent / f"domain_umap_{cond}",
            condition_name=cond,
        )


def plot_effective_rank(
    results: Dict[str, dict],
    output_path: Path,
) -> None:
    """Fig 3: Grouped bar chart of effective rank by domain."""
    if not HAS_MATPLOTLIB:
        return

    conditions = list(results.keys())
    x = np.arange(len(conditions))
    labels = [_short_label(c) for c in conditions]
    width = 0.35

    gli_ranks = [results[c].get("source_effective_rank", 0) for c in conditions]
    men_ranks = [results[c].get("target_effective_rank", 0) for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width / 2, gli_ranks, width, label="Glioma (GLI)", color=DOMAIN_COLORS["glioma"], alpha=0.8)
    bars2 = ax.bar(x + width / 2, men_ranks, width, label="Meningioma (MEN)", color=DOMAIN_COLORS["meningioma"], alpha=0.8)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_title("Effective Rank: Feature Space Utilization by Domain", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_cross_experiment_heatmap(
    all_results: Dict[str, Dict[str, dict]],
    output_path: Path,
) -> None:
    """Fig 4: 2x2 heatmap grid across experiment variants."""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        logger.warning("Matplotlib + Seaborn required for heatmap")
        return

    metric_keys = ["mmd", "cka", "proxy_a_distance", "domain_classifier_accuracy"]
    metric_labels = ["MMD²", "CKA", "PAD", "Clf Acc"]

    n_exp = len(all_results)
    if n_exp == 0:
        return

    ncols = 2
    nrows = (n_exp + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (exp_name, results) in enumerate(all_results.items()):
        ax = axes[idx // ncols, idx % ncols]
        conditions = list(results.keys())
        row_labels = [_short_label(c) for c in conditions]

        data = np.zeros((len(conditions), len(metric_keys)))
        for i, cond in enumerate(conditions):
            for j, key in enumerate(metric_keys):
                data[i, j] = results[cond].get(key, 0)

        sns.heatmap(
            data, ax=ax, annot=True, fmt=".3f",
            xticklabels=metric_labels, yticklabels=row_labels,
            cmap="YlOrRd", linewidths=0.5, linecolor="white",
        )
        label = EXPERIMENT_LABELS.get(exp_name, exp_name)
        ax.set_title(label, fontsize=11, fontweight="bold")

    # Hide unused axes
    for idx in range(len(all_results), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Domain Gap Metrics Across Experiment Variants", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_adaptation_tradeoff(
    results: Dict[str, dict],
    output_path: Path,
    experiment_label: str = "",
) -> None:
    """Fig 5: Scatter plot of R² improvement vs MMD change."""
    if not HAS_MATPLOTLIB:
        return

    frozen = results.get("baseline_frozen")
    if frozen is None:
        logger.warning("No baseline_frozen in results, skipping trade-off plot")
        return

    frozen_r2 = frozen.get("r2_mean_linear")
    frozen_mmd = frozen.get("mmd")
    if frozen_r2 is None or frozen_mmd is None:
        logger.warning("Missing r2_mean_linear or mmd in frozen baseline, skipping trade-off plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for cond, metrics in results.items():
        if cond == "baseline_frozen":
            continue

        r2 = metrics.get("r2_mean_linear")
        mmd = metrics.get("mmd")
        if r2 is None or mmd is None:
            continue

        delta_r2 = r2 - frozen_r2
        delta_mmd = mmd - frozen_mmd
        color = CONDITION_COLORS.get(cond, "#808080")

        ax.scatter(delta_r2, delta_mmd, color=color, s=80, zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(
            _short_label(cond),
            (delta_r2, delta_mmd),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color=color,
        )

    # Reference lines
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    # Quadrant labels
    ax.text(0.98, 0.98, "R²↑ MMD↑\n(adapt + diverge)", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, alpha=0.4)
    ax.text(0.02, 0.98, "R²↓ MMD↑\n(worse all around)", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, alpha=0.4)
    ax.text(0.98, 0.02, "R²↑ MMD↓\n(ideal)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, alpha=0.4)
    ax.text(0.02, 0.02, "R²↓ MMD↓\n(converge but degrade)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7, alpha=0.4)

    ax.set_xlabel("ΔR² (mean linear) from frozen baseline", fontsize=11)
    ax.set_ylabel("ΔMMD² from frozen baseline", fontsize=11)
    title = "Adaptation vs Domain Gap Trade-off"
    if experiment_label:
        title += f" ({experiment_label})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output_path)


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

def run_single_experiment(output_dir: Path) -> Dict[str, dict]:
    """Run domain metrics analysis for a single experiment."""
    output_dir = Path(output_dir)
    conditions = discover_conditions(output_dir)
    if not conditions:
        logger.error(f"No conditions with feature files found in {output_dir}")
        return {}

    logger.info(f"Found {len(conditions)} conditions: {conditions}")

    # Compute metrics
    results = compute_all_metrics(output_dir, conditions)
    if not results:
        logger.error("No metrics computed")
        return {}

    # Save summary CSV
    save_summary_csv(results, output_dir)

    # Generate figures
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    exp_label = EXPERIMENT_LABELS.get(output_dir.name, output_dir.name)

    plot_domain_gap_metrics(results, figures_dir / "domain_gap_metrics")
    plot_umap_grid(output_dir, results, figures_dir / "domain_umap_grid")
    plot_effective_rank(results, figures_dir / "domain_effective_rank")
    plot_adaptation_tradeoff(results, figures_dir / "domain_adaptation_tradeoff", exp_label)

    logger.info(f"Domain metrics analysis complete for {output_dir.name}")
    return results


def run_all_experiments(output_dir: Path) -> None:
    """Run domain metrics on all sibling experiment directories."""
    siblings = discover_sibling_experiments(output_dir)
    if not siblings:
        logger.error("No sibling experiment directories found")
        return

    logger.info(f"Running domain metrics on {len(siblings)} experiments: {[s.name for s in siblings]}")

    all_results = {}
    for exp_dir in siblings:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {exp_dir.name}")
        logger.info(f"{'=' * 60}")
        results = run_single_experiment(exp_dir)
        if results:
            all_results[exp_dir.name] = results

    # Cross-experiment heatmap (Fig 4)
    if len(all_results) > 1 and HAS_MATPLOTLIB:
        # Save in the parent directory
        parent_figures = output_dir.parent / "figures"
        parent_figures.mkdir(parents=True, exist_ok=True)
        plot_cross_experiment_heatmap(
            all_results,
            parent_figures / "domain_cross_experiment_heatmap",
        )
        logger.info(f"Cross-experiment heatmap saved to {parent_figures}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute domain gap metrics for LoRA ablation experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to experiment output directory (e.g., .../lora_ablation_semantic_heads)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to experiment config YAML (reads output_dir from it)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all 4 sibling experiment directories",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve output directory
    if args.config is not None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        output_dir = Path(cfg.experiment.output_dir)
    elif args.output_dir is not None:
        output_dir = args.output_dir
    else:
        parser.error("Must specify --output-dir or --config")

    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return

    if args.all:
        run_all_experiments(output_dir)
    else:
        run_single_experiment(output_dir)


if __name__ == "__main__":
    main()
