#!/usr/bin/env python
# scripts/sdp_diagnostics.py
"""Diagnostic analysis of encoder features for SDP optimization.

Establishes upper bounds on what the SDP can achieve:
- Linear probe ceilings (Ridge regression) from raw 768-dim features to targets
- PCA analysis of encoder features (effective dimensionality)
- Target distribution analysis (correlations between targets)
- Feature-to-target correlation matrix

Usage:
    python scripts/sdp_diagnostics.py \
        --features-dir /path/to/features/lora_semantic_heads_r16
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_split(features_dir: str, split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load features and targets from a split H5 file."""
    path = Path(features_dir) / f"{split}.h5"
    with h5py.File(path, "r") as f:
        h = np.array(f["features/encoder10"])
        targets = {
            "vol": np.array(f["targets/volume"]),
            "loc": np.array(f["targets/location"]),
            "shape": np.array(f["targets/shape"]),
        }
    return h, targets


def compute_linear_probe_ceiling(
    h_train: np.ndarray,
    targets_train: dict[str, np.ndarray],
    h_val: np.ndarray,
    targets_val: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Ridge regression from raw 768-dim features to each target group."""
    scaler = StandardScaler()
    h_train_s = scaler.fit_transform(h_train)
    h_val_s = scaler.transform(h_val)

    results = {}
    for key in ["vol", "loc", "shape"]:
        # Normalize targets
        t_scaler = StandardScaler()
        y_train = t_scaler.fit_transform(targets_train[key])
        y_val = t_scaler.transform(targets_val[key])

        best_r2 = -np.inf
        best_alpha = None

        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            model = Ridge(alpha=alpha)
            model.fit(h_train_s, y_train)
            y_pred = model.predict(h_val_s)
            r2 = r2_score(y_val, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha

        # Get per-dim R² at best alpha
        model = Ridge(alpha=best_alpha)
        model.fit(h_train_s, y_train)
        y_pred = model.predict(h_val_s)
        r2_per_dim = []
        for d in range(y_val.shape[1]):
            r2_d = r2_score(y_val[:, d], y_pred[:, d])
            r2_per_dim.append(r2_d)

        results[key] = {
            "r2_overall": float(best_r2),
            "r2_per_dim": r2_per_dim,
            "best_alpha": float(best_alpha),
            "n_dims": y_val.shape[1],
        }

    return results


def compute_pca_analysis(h_all: np.ndarray) -> dict:
    """PCA analysis of encoder features."""
    scaler = StandardScaler()
    h_s = scaler.fit_transform(h_all)

    pca = PCA(n_components=min(h_s.shape))
    pca.fit(h_s)

    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # Effective rank via entropy
    ev = pca.explained_variance_ratio_
    ev_nonzero = ev[ev > 1e-10]
    entropy = -np.sum(ev_nonzero * np.log(ev_nonzero))
    effective_rank = np.exp(entropy)

    # Dimensionality for 90%, 95%, 99% variance
    dim_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    dim_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    dim_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    return {
        "effective_rank": float(effective_rank),
        "dim_90pct": dim_90,
        "dim_95pct": dim_95,
        "dim_99pct": dim_99,
        "top_10_explained": cumulative[:10].tolist(),
        "top_50_explained": float(cumulative[49]) if len(cumulative) > 49 else None,
        "explained_variance_ratio": ev[:50].tolist(),
    }


def compute_target_analysis(targets: dict[str, np.ndarray]) -> dict:
    """Analyze target distributions and cross-target correlations."""
    # Stats per target
    stats = {}
    for key, t in targets.items():
        stats[key] = {
            "mean": t.mean(axis=0).tolist(),
            "std": t.std(axis=0).tolist(),
            "min": t.min(axis=0).tolist(),
            "max": t.max(axis=0).tolist(),
            "n_dims": t.shape[1],
        }

    # Cross-target correlation
    all_targets = np.concatenate([targets["vol"], targets["loc"], targets["shape"]], axis=1)
    labels = (
        [f"vol_{i}" for i in range(targets["vol"].shape[1])]
        + [f"loc_{i}" for i in range(targets["loc"].shape[1])]
        + [f"shape_{i}" for i in range(targets["shape"].shape[1])]
    )
    corr_matrix = np.corrcoef(all_targets.T)

    # Cross-group max correlations
    n_vol = targets["vol"].shape[1]
    n_loc = targets["loc"].shape[1]
    n_shape = targets["shape"].shape[1]

    vol_loc_corr = np.abs(corr_matrix[:n_vol, n_vol:n_vol + n_loc]).max()
    vol_shape_corr = np.abs(corr_matrix[:n_vol, n_vol + n_loc:]).max()
    loc_shape_corr = np.abs(corr_matrix[n_vol:n_vol + n_loc, n_vol + n_loc:]).max()

    return {
        "stats": stats,
        "cross_group_max_corr": {
            "vol_loc": float(vol_loc_corr),
            "vol_shape": float(vol_shape_corr),
            "loc_shape": float(loc_shape_corr),
        },
        "labels": labels,
        "corr_matrix": corr_matrix.tolist(),
    }


def compute_feature_target_correlation(
    h: np.ndarray, targets: dict[str, np.ndarray]
) -> dict:
    """Correlation between individual encoder dimensions and targets."""
    scaler = StandardScaler()
    h_s = scaler.fit_transform(h)

    results = {}
    for key, t in targets.items():
        t_s = StandardScaler().fit_transform(t)
        # Correlation: [768, n_targets]
        corr = np.array([
            [np.corrcoef(h_s[:, i], t_s[:, j])[0, 1]
             for j in range(t_s.shape[1])]
            for i in range(h_s.shape[1])
        ])
        # Max absolute correlation per target dim
        max_abs_corr = np.abs(corr).max(axis=0)
        # Number of features with |corr| > 0.3 for each target dim
        n_informative = (np.abs(corr) > 0.3).sum(axis=0)

        results[key] = {
            "max_abs_corr_per_target": max_abs_corr.tolist(),
            "n_informative_features": n_informative.tolist(),
            "mean_max_corr": float(max_abs_corr.mean()),
        }

    return results


def generate_report(
    probe_results: dict,
    pca_results: dict,
    target_analysis: dict,
    feature_target_corr: dict,
    n_train: int,
    n_val: int,
) -> str:
    """Generate markdown report."""
    lines = [
        "# SDP Diagnostic Analysis",
        "",
        "## Data Summary",
        f"- Training samples: {n_train}",
        f"- Validation samples: {n_val}",
        f"- Feature dim: 768 (encoder10)",
        "",
        "---",
        "",
        "## Iteration 0: Diagnostic Baselines",
        "",
        "### 1. Linear Probe Ceilings (Ridge Regression: raw 768-dim → targets)",
        "",
        "These represent the **upper bound** on what a linear SDP can achieve.",
        "",
        "| Target | R² (val) | Per-dim R² | Best α |",
        "|--------|----------|------------|--------|",
    ]

    for key in ["vol", "loc", "shape"]:
        r = probe_results[key]
        per_dim_str = ", ".join(f"{d:.3f}" for d in r["r2_per_dim"])
        lines.append(f"| {key} | {r['r2_overall']:.4f} | [{per_dim_str}] | {r['best_alpha']:.1f} |")

    lines.extend([
        "",
        "### 2. PCA Analysis of Encoder Features",
        "",
        f"- **Effective rank**: {pca_results['effective_rank']:.1f} / 768",
        f"- Dims for 90% variance: {pca_results['dim_90pct']}",
        f"- Dims for 95% variance: {pca_results['dim_95pct']}",
        f"- Dims for 99% variance: {pca_results['dim_99pct']}",
        "",
        "Cumulative explained variance (top 10 PCs):",
        "```",
    ])
    for i, v in enumerate(pca_results["top_10_explained"]):
        lines.append(f"  PC{i+1:2d}: {v:.4f}")
    lines.extend(["```", ""])

    lines.extend([
        "### 3. Target Distribution Analysis",
        "",
        "Cross-group max absolute correlations between targets:",
        f"- vol ↔ loc: {target_analysis['cross_group_max_corr']['vol_loc']:.4f}",
        f"- vol ↔ shape: {target_analysis['cross_group_max_corr']['vol_shape']:.4f}",
        f"- loc ↔ shape: {target_analysis['cross_group_max_corr']['loc_shape']:.4f}",
        "",
    ])

    if max(target_analysis['cross_group_max_corr'].values()) > 0.3:
        lines.append("**WARNING**: Targets have non-trivial cross-group correlation. "
                      "Perfect disentanglement may be geometrically impossible.")
        lines.append("")

    lines.extend([
        "### 4. Feature-Target Correlation",
        "",
        "| Target | Mean max |r| | Max |r| per dim | Informative features (|r|>0.3) |",
        "|--------|-------------|------------------|-------------------------------|",
    ])

    for key in ["vol", "loc", "shape"]:
        r = feature_target_corr[key]
        max_str = ", ".join(f"{d:.3f}" for d in r["max_abs_corr_per_target"])
        info_str = ", ".join(str(d) for d in r["n_informative_features"])
        lines.append(f"| {key} | {r['mean_max_corr']:.4f} | [{max_str}] | [{info_str}] |")

    lines.extend([
        "",
        "---",
        "",
    ])

    return "\n".join(lines)


def main(features_dir: str) -> str:
    """Run full diagnostic analysis.

    Args:
        features_dir: Path to directory with per-split feature H5 files.

    Returns:
        Markdown report string.
    """
    logger.info(f"Loading features from {features_dir}")

    # Load all splits
    h_train1, targets_train1 = load_split(features_dir, "lora_train")
    h_train2, targets_train2 = load_split(features_dir, "sdp_train")
    h_val, targets_val = load_split(features_dir, "lora_val")

    # Combine training
    h_train = np.concatenate([h_train1, h_train2], axis=0)
    targets_train = {
        k: np.concatenate([targets_train1[k], targets_train2[k]], axis=0)
        for k in ["vol", "loc", "shape"]
    }

    logger.info(f"Train: {h_train.shape}, Val: {h_val.shape}")

    # 1. Linear probe ceilings
    logger.info("Computing linear probe ceilings...")
    probe_results = compute_linear_probe_ceiling(h_train, targets_train, h_val, targets_val)
    for key, r in probe_results.items():
        logger.info(f"  {key}: R²={r['r2_overall']:.4f} (alpha={r['best_alpha']})")

    # 2. PCA analysis
    logger.info("Computing PCA analysis...")
    h_all = np.concatenate([h_train, h_val], axis=0)
    pca_results = compute_pca_analysis(h_all)
    logger.info(f"  Effective rank: {pca_results['effective_rank']:.1f}")

    # 3. Target analysis
    logger.info("Computing target distribution analysis...")
    target_analysis = compute_target_analysis(targets_train)

    # 4. Feature-target correlation
    logger.info("Computing feature-target correlations...")
    feature_target_corr = compute_feature_target_correlation(h_train, targets_train)

    # Generate report
    report = generate_report(
        probe_results, pca_results, target_analysis, feature_target_corr,
        n_train=h_train.shape[0], n_val=h_val.shape[0],
    )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDP Diagnostic Analysis")
    parser.add_argument(
        "--features-dir",
        type=str,
        default="/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/features/lora_semantic_heads_r16",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/growth-related/misc/sdp_findings.md",
    )

    args = parser.parse_args()
    report = main(args.features_dir)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info(f"Report saved to {output_path}")
    print(report)
