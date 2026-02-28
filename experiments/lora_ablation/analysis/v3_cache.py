#!/usr/bin/env python
# experiments/lora_ablation/v3_cache.py
"""Precompute figure data from raw condition outputs to a JSON/NPZ cache.

Reads per-condition .pt, .json, and .csv files and produces a
``results/figure_cache/`` directory that v3_figures.py can render without
a GPU.

Usage:
    from experiments.lora_ablation.v3_cache import precompute_all
    precompute_all(config, output_dir)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from experiments.utils.settings import V3_CONDITIONS

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, returning None on failure."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    """Load CSV as list of dicts, returning None on failure."""
    if not path.exists():
        return None
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def _save_json(data: Any, path: Path) -> None:
    """Save data as JSON with human-readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logger.info(f"Saved {path}")


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ============================================================================
# Per-cache builders
# ============================================================================

def _build_training_logs(
    config: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """Consolidate all training_log.csv into a single dict.

    Returns:
        Dict mapping condition name to list of epoch records.
    """
    logs: Dict[str, Any] = {}
    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        csv_path = output_dir / "conditions" / name / "training_log.csv"
        rows = _load_csv_rows(csv_path)
        if rows is not None:
            # Convert numeric fields
            for row in rows:
                for key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            logs[name] = rows
            logger.debug(f"  training_log: {name} ({len(rows)} epochs)")
    return logs


def _build_dice_data(
    config: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """Consolidate test_dice_men.json and test_dice_gli.json per condition.

    Returns:
        Dict mapping condition name to ``{"men": {...}, "gli": {...}}``.
        Falls back to flat MEN-only format if no GLI data exists.
    """
    data: Dict[str, Any] = {}
    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        men_path = output_dir / "conditions" / name / "test_dice_men.json"
        gli_path = output_dir / "conditions" / name / "test_dice_gli.json"
        men = _load_json(men_path)
        gli = _load_json(gli_path)
        if men is not None:
            data[name] = {"men": men}
            if gli is not None:
                data[name]["gli"] = gli
    return data


def _build_feature_quality_data(
    config: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """Consolidate all feature_quality.json into a single dict."""
    data: Dict[str, Any] = {}
    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        fq_path = output_dir / "conditions" / name / "feature_quality.json"
        fq = _load_json(fq_path)
        if fq is not None:
            data[name] = fq
    return data


def _build_probe_metrics(
    config: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """Consolidate all metrics.json / metrics_enhanced.json into a single dict."""
    data: Dict[str, Any] = {}
    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        # Prefer enhanced metrics
        metrics_path = output_dir / "conditions" / name / "metrics_enhanced.json"
        if not metrics_path.exists():
            metrics_path = output_dir / "conditions" / name / "metrics.json"
        metrics = _load_json(metrics_path)
        if metrics is not None:
            data[name] = metrics
    return data


def _build_predictions(
    config: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """Find the best condition and consolidate its predictions.

    Best condition = highest r2_mean from probe metrics.
    """
    best_name: Optional[str] = None
    best_r2: float = -float("inf")

    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        metrics_path = output_dir / "conditions" / name / "metrics_enhanced.json"
        if not metrics_path.exists():
            metrics_path = output_dir / "conditions" / name / "metrics.json"
        metrics = _load_json(metrics_path)
        if metrics is not None:
            r2 = metrics.get("r2_mean", metrics.get("r2_mean_mlp", -1.0))
            if r2 is not None and r2 > best_r2:
                best_r2 = r2
                best_name = name

    if best_name is None:
        return {}

    pred_path = output_dir / "conditions" / best_name / "predictions_enhanced.json"
    if not pred_path.exists():
        pred_path = output_dir / "conditions" / best_name / "predictions.json"
    preds = _load_json(pred_path)

    return {
        "best_condition": best_name,
        "best_r2_mean": best_r2,
        "predictions": preds,
    }


def _build_variance_spectrum(
    config: dict,
    output_dir: Path,
    cache_dir: Path,
) -> None:
    """Precompute sorted variance per dimension for each condition.

    Saves as NPZ with keys = condition names, values = sorted variance arrays.
    """
    arrays: Dict[str, np.ndarray] = {}
    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        feat_path = output_dir / "conditions" / name / "features_test.pt"
        if not feat_path.exists():
            # Try level-specific name
            level = config.get("feature_extraction", {}).get("level", "encoder10")
            feat_path = output_dir / "conditions" / name / f"features_test_{level}.pt"
        if not feat_path.exists():
            continue

        features = torch.load(feat_path, map_location="cpu", weights_only=True)
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        variance = np.sort(features.var(axis=0))[::-1]
        arrays[name] = variance.astype(np.float32)
        logger.debug(f"  variance: {name} ({len(variance)} dims)")

    if arrays:
        out_path = cache_dir / "variance_spectrum.npz"
        np.savez_compressed(out_path, **arrays)
        logger.info(f"Saved {out_path}")


def _build_umap_embedding(
    config: dict,
    output_dir: Path,
    cache_dir: Path,
    max_samples_per_cond: int = 100,
    pca_dims: int = 50,
) -> None:
    """Precompute UMAP embedding (PCA -> 50d first) for all conditions.

    Saves:
      - umap_embedding.npz: 'embedding' [N, 2]
      - umap_targets.npz: 'conditions' [N], 'volume' [N], 'shape_0' [N]
    """
    try:
        from sklearn.decomposition import PCA
        from umap import UMAP
    except ImportError:
        logger.warning("sklearn or umap not available; skipping UMAP cache.")
        return

    all_features: List[np.ndarray] = []
    all_conditions: List[str] = []
    all_volumes: List[np.ndarray] = []
    all_shape0: List[np.ndarray] = []

    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        feat_path = output_dir / "conditions" / name / "features_test.pt"
        if not feat_path.exists():
            level = config.get("feature_extraction", {}).get("level", "encoder10")
            feat_path = output_dir / "conditions" / name / f"features_test_{level}.pt"
        tgt_path = output_dir / "conditions" / name / "targets_test.pt"

        if not feat_path.exists():
            continue

        features = torch.load(feat_path, map_location="cpu", weights_only=True)
        if isinstance(features, torch.Tensor):
            features = features.numpy()

        # Subsample
        n = min(max_samples_per_cond, len(features))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(features), n, replace=False)
        features = features[idx]

        all_features.append(features)
        all_conditions.extend([name] * n)

        # Load targets for coloring
        if tgt_path.exists():
            targets = torch.load(tgt_path, map_location="cpu", weights_only=True)
            if isinstance(targets, dict):
                vol = targets.get("volume", torch.zeros(len(features) + max_samples_per_cond))
                vol = vol.numpy() if isinstance(vol, torch.Tensor) else np.array(vol)
                vol = vol[idx] if len(vol) > max(idx) else vol[:n]
                # Use first volume dim (total volume)
                if vol.ndim > 1:
                    vol = vol[:, 0]
                all_volumes.append(vol[:n])

                shp = targets.get("shape", torch.zeros(len(features) + max_samples_per_cond))
                shp = shp.numpy() if isinstance(shp, torch.Tensor) else np.array(shp)
                shp = shp[idx] if len(shp) > max(idx) else shp[:n]
                if shp.ndim > 1:
                    shp = shp[:, 0]
                all_shape0.append(shp[:n])
            else:
                all_volumes.append(np.zeros(n))
                all_shape0.append(np.zeros(n))
        else:
            all_volumes.append(np.zeros(n))
            all_shape0.append(np.zeros(n))

    if not all_features:
        logger.warning("No features found for UMAP computation.")
        return

    X = np.vstack(all_features)
    logger.info(f"UMAP: {X.shape[0]} samples, {X.shape[1]} dims")

    # PCA reduction first
    actual_pca_dims = min(pca_dims, X.shape[1], X.shape[0])
    pca = PCA(n_components=actual_pca_dims, random_state=42)
    X_pca = pca.fit_transform(X)

    # UMAP
    umap = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = umap.fit_transform(X_pca)

    np.savez_compressed(
        cache_dir / "umap_embedding.npz",
        embedding=embedding.astype(np.float32),
    )
    np.savez_compressed(
        cache_dir / "umap_targets.npz",
        conditions=np.array(all_conditions),
        volume=np.concatenate(all_volumes).astype(np.float32),
        shape_0=np.concatenate(all_shape0).astype(np.float32),
    )
    logger.info(f"Saved UMAP embedding ({X.shape[0]} points)")


def _build_domain_umap_grid(
    config: dict,
    output_dir: Path,
    cache_dir: Path,
    max_samples_per_domain: int = 150,
    pca_dims: int = 50,
) -> None:
    """Precompute a shared UMAP over GLI+MEN features for all conditions.

    For each condition, loads ``features_glioma.pt`` and
    ``features_meningioma_subset.pt``, concatenates all conditions into a
    single PCA -> UMAP space, then stores per-condition coordinates and
    silhouette scores.

    Saves ``domain_umap_grid.npz`` with keys:
      - ``embedding`` [N_total, 2]
      - ``domains`` [N_total]  (``"glioma"`` / ``"meningioma"``)
      - ``conditions`` [N_total]
      - ``silhouette_<cond>`` float per condition
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        from umap import UMAP
    except ImportError:
        logger.warning("sklearn or umap not available; skipping domain UMAP grid.")
        return

    all_features: List[np.ndarray] = []
    all_domains: List[str] = []
    all_conditions: List[str] = []

    for cond_cfg in config["conditions"]:
        name = cond_cfg["name"]
        cond_dir = output_dir / "conditions" / name

        gli_path = cond_dir / "features_glioma.pt"
        men_path = cond_dir / "features_meningioma_subset.pt"
        if not gli_path.exists() or not men_path.exists():
            logger.debug(f"  domain_umap: skipping {name} (missing domain features)")
            continue

        gli_feat = torch.load(gli_path, map_location="cpu", weights_only=True)
        men_feat = torch.load(men_path, map_location="cpu", weights_only=True)
        if isinstance(gli_feat, torch.Tensor):
            gli_feat = gli_feat.numpy()
        if isinstance(men_feat, torch.Tensor):
            men_feat = men_feat.numpy()

        rng = np.random.RandomState(42)
        n_gli = min(max_samples_per_domain, len(gli_feat))
        n_men = min(max_samples_per_domain, len(men_feat))
        gli_idx = rng.choice(len(gli_feat), n_gli, replace=False)
        men_idx = rng.choice(len(men_feat), n_men, replace=False)

        all_features.append(gli_feat[gli_idx])
        all_features.append(men_feat[men_idx])
        all_domains.extend(["glioma"] * n_gli + ["meningioma"] * n_men)
        all_conditions.extend([name] * (n_gli + n_men))

    if not all_features:
        logger.warning("No domain features found for UMAP grid computation.")
        return

    X = np.vstack(all_features)
    domains_arr = np.array(all_domains)
    conditions_arr = np.array(all_conditions)
    logger.info(f"Domain UMAP grid: {X.shape[0]} samples, {X.shape[1]} dims")

    # Shared PCA -> UMAP
    actual_pca_dims = min(pca_dims, X.shape[1], X.shape[0])
    pca = PCA(n_components=actual_pca_dims, random_state=42)
    X_pca = pca.fit_transform(X)

    umap = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = umap.fit_transform(X_pca)

    # Per-condition silhouette scores
    save_dict: Dict[str, Any] = {
        "embedding": embedding.astype(np.float32),
        "domains": domains_arr,
        "conditions": conditions_arr,
    }

    unique_conds = sorted(set(all_conditions))
    for cond in unique_conds:
        mask = conditions_arr == cond
        cond_domains = domains_arr[mask]
        if len(set(cond_domains)) < 2:
            continue
        domain_labels = (cond_domains == "glioma").astype(int)
        sil = silhouette_score(embedding[mask], domain_labels)
        save_dict[f"silhouette_{cond}"] = np.float32(sil)
        logger.debug(f"  domain_umap: {cond} silhouette={sil:.3f}")

    out_path = cache_dir / "domain_umap_grid.npz"
    np.savez_compressed(out_path, **save_dict)
    logger.info(f"Saved domain UMAP grid ({len(unique_conds)} conditions)")


# ============================================================================
# Main entry point
# ============================================================================

def precompute_all(config: dict, output_dir: Path) -> Path:
    """Precompute all figure data from raw condition outputs.

    Creates ``{output_dir}/results/figure_cache/`` with JSON and NPZ files
    that v3_figures.py reads to produce thesis-quality figures.

    Args:
        config: Loaded experiment configuration dict.
        output_dir: Experiment output directory (containing ``conditions/``).

    Returns:
        Path to the figure_cache directory.
    """
    cache_dir = output_dir / "results" / "figure_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Precomputing figure cache...")

    # JSON caches
    training_logs = _build_training_logs(config, output_dir)
    _save_json(training_logs, cache_dir / "training_logs.json")

    dice_data = _build_dice_data(config, output_dir)
    _save_json(dice_data, cache_dir / "dice_data.json")

    fq_data = _build_feature_quality_data(config, output_dir)
    _save_json(fq_data, cache_dir / "feature_quality_data.json")

    probe_data = _build_probe_metrics(config, output_dir)
    _save_json(probe_data, cache_dir / "probe_metrics.json")

    pred_data = _build_predictions(config, output_dir)
    _save_json(pred_data, cache_dir / "predictions.json")

    # NPZ caches (heavier computation)
    _build_variance_spectrum(config, output_dir, cache_dir)
    _build_umap_embedding(config, output_dir, cache_dir)
    _build_domain_umap_grid(config, output_dir, cache_dir)

    logger.info(f"Figure cache ready: {cache_dir}")
    return cache_dir
