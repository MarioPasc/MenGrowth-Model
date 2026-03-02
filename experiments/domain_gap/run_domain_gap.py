"""Domain gap analysis pipeline (GPU).

Extracts encoder10 features and computes segmentation Dice for
BraTS-GLI (glioma), BraTS-MEN (meningioma), and optionally MenGrowth
using the frozen BrainSegFounder model, then computes domain shift metrics.

If ``paths.mengrowth_root`` is null in config, the pipeline behaves
identically to the two-domain version.  When present it computes all
three datasets and all C(3,2)=3 pairwise comparisons.

Usage:
    python -m experiments.domain_gap.run_domain_gap --config <path>
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset

from src.growth.data.bratsmendata import BraTSMENDatasetH5
from src.growth.data.transforms import (
    get_h5_val_transforms,
    get_val_transforms,
)
from src.growth.evaluation.latent_quality import (
    compute_cka,
    compute_domain_classifier_accuracy,
    compute_effective_rank,
    compute_proxy_a_distance,
    mmd_permutation_test,
)
from src.growth.losses.segmentation import DiceMetric3Ch
from src.growth.models.encoder.feature_extractor import FeatureExtractor
from src.growth.models.encoder.swin_loader import load_full_swinunetr, load_swin_encoder
from src.growth.utils.seed import set_seed

logger = logging.getLogger(__name__)

# ============================================================================
# BraTS-GLI Dataset (standalone, no cross-experiment import)
# ============================================================================

GLIOMA_MODALITY_SUFFIXES = {
    "t1c": "-t1c.nii.gz",
    "t1n": "-t1n.nii.gz",
    "t2f": "-t2f.nii.gz",
    "t2w": "-t2w.nii.gz",
}
GLIOMA_SEG_SUFFIX = "-seg.nii.gz"


class BraTSGLIDataset(Dataset):
    """BraTS-GLI (Glioma) dataset for feature extraction and Dice evaluation.

    Loads NIfTI volumes from the standard BraTS-GLI directory structure.
    Always includes segmentation when available.

    Args:
        data_root: Path to BraTS-GLI data root directory.
        subject_ids: Optional list of subject IDs. If None, discovers all.
        transform: MONAI transform pipeline.
    """

    def __init__(
        self,
        data_root: str | Path,
        subject_ids: list[str] | None = None,
        transform=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform or get_val_transforms()

        if subject_ids is None:
            self.subject_ids = self._discover_subjects()
        else:
            self.subject_ids = subject_ids

        logger.info(f"BraTSGLIDataset initialized with {len(self.subject_ids)} subjects")

    def _discover_subjects(self) -> list[str]:
        """Discover all subject IDs in the glioma dataset."""
        subject_ids = []
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name.startswith("BraTS-GLI-"):
                subject_ids.append(item.name)
        return sorted(subject_ids)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict:
        subject_id = self.subject_ids[idx]
        subject_dir = self.data_root / subject_id

        data = {}
        for modality, suffix in GLIOMA_MODALITY_SUFFIXES.items():
            path = subject_dir / f"{subject_id}{suffix}"
            if path.exists():
                data[modality] = str(path)
            else:
                raise FileNotFoundError(f"Missing {modality}: {path}")

        seg_path = subject_dir / f"{subject_id}{GLIOMA_SEG_SUFFIX}"
        if seg_path.exists():
            data["seg"] = str(seg_path)

        transformed = self.transform(data)

        result = {
            "image": transformed["image"],
            "subject_id": subject_id,
            "domain": "glioma",
        }

        if "seg" in transformed:
            result["seg"] = transformed["seg"]

        return result


# ============================================================================
# MenGrowth Dataset
# ============================================================================

MENGROWTH_MODALITY_KEYS = ["t2f", "t1c", "t1n", "t2w"]


def _find_nifti(scan_dir: Path, name: str) -> Path:
    """Find a NIfTI file in *scan_dir* by modality/seg name.

    Supports two naming conventions:
      1. Bare names:     ``{name}.nii.gz``  (e.g. ``t2f.nii.gz``)
      2. Prefixed names: ``{scan_id}-{name}.nii.gz``
         (e.g. ``MenGrowth-0001-000-t2f.nii.gz``)

    The scan_id prefix is taken from the directory name.

    Args:
        scan_dir: Directory containing NIfTI files for one scan.
        name: Modality key (``t2f``, ``t1c``, …) or ``seg``.

    Returns:
        Path to the matching NIfTI file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    # Try bare name first (current convention)
    bare = scan_dir / f"{name}.nii.gz"
    if bare.exists():
        return bare

    # Try prefixed name: {scan_dir.name}-{name}.nii.gz
    prefixed = scan_dir / f"{scan_dir.name}-{name}.nii.gz"
    if prefixed.exists():
        return prefixed

    raise FileNotFoundError(f"Missing {name} in {scan_dir}: tried {bare.name} and {prefixed.name}")


class MenGrowthDataset(Dataset):
    """MenGrowth dataset for feature extraction and Dice evaluation.

    Directory layout::

        data_root/
            MenGrowth-XXXX/
                MenGrowth-XXXX-YYY/
                    {t2f,t1c,t1n,t2w,seg}.nii.gz   (bare)
                    OR
                    MenGrowth-XXXX-YYY-{t2f,...}.nii.gz  (prefixed)

    Args:
        data_root: Path to MenGrowth data root directory.
        first_timepoint_only: If True, keep only the first scan per subject
            (``-000`` suffix). Gives N=33 independent samples.
        transform: MONAI transform pipeline.
    """

    def __init__(
        self,
        data_root: str | Path,
        first_timepoint_only: bool = False,
        transform=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform or get_val_transforms()
        self.scans = self._discover_scans(first_timepoint_only)
        logger.info(
            f"MenGrowthDataset initialized: {len(self.scans)} scans, "
            f"{len(set(s[0] for s in self.scans))} subjects"
        )

    def _discover_scans(self, first_timepoint_only: bool) -> list[tuple[str, str, Path]]:
        """Discover (subject_id, scan_id, scan_dir) tuples.

        Returns:
            Sorted list of (subject_id, scan_id, scan_dir).
        """
        scans: list[tuple[str, str, Path]] = []
        subject_pattern = re.compile(r"^MenGrowth-\d+$")

        for subject_dir in sorted(self.data_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            if not subject_pattern.match(subject_dir.name):
                continue

            subject_id = subject_dir.name

            for scan_dir in sorted(subject_dir.iterdir()):
                if not scan_dir.is_dir():
                    continue
                if not scan_dir.name.startswith(subject_id):
                    continue

                scan_id = scan_dir.name

                if first_timepoint_only:
                    # Keep only -000 scans
                    suffix = scan_id[len(subject_id) :]  # e.g. "-000"
                    if suffix != "-000":
                        continue

                scans.append((subject_id, scan_id, scan_dir))

        return scans

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int) -> dict:
        subject_id, scan_id, scan_dir = self.scans[idx]

        data = {}
        for modality in MENGROWTH_MODALITY_KEYS:
            data[modality] = str(_find_nifti(scan_dir, modality))

        seg_path = _find_nifti(scan_dir, "seg")
        data["seg"] = str(seg_path)

        transformed = self.transform(data)

        result = {
            "image": transformed["image"],
            "subject_id": subject_id,
            "scan_id": scan_id,
            "domain": "mengrowth",
        }

        if "seg" in transformed:
            result["seg"] = transformed["seg"]

        return result


# ============================================================================
# Feature Extraction
# ============================================================================


def extract_features(
    encoder: torch.nn.Module,
    dataset: Dataset,
    level: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Extract encoder features from a dataset.

    Args:
        encoder: SwinUNETR encoder model.
        dataset: Dataset returning dict with "image" and "subject_id".
        level: Feature extraction level (e.g. "encoder10").
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        device: Torch device.

    Returns:
        Tuple of (features [N, D], subject_ids [N]).
    """
    extractor = FeatureExtractor(encoder, level=level)
    extractor.eval()
    extractor.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    all_features = []
    all_ids = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["image"].to(device)
            features = extractor(images)  # [B, D]
            all_features.append(features.cpu().numpy())

            if isinstance(batch["subject_id"], (list, tuple)):
                all_ids.extend(batch["subject_id"])
            else:
                all_ids.append(batch["subject_id"])

            if (i + 1) % 10 == 0:
                logger.info(f"  Extracted {(i + 1) * batch_size}/{len(dataset)}")

    features_arr = np.concatenate(all_features, axis=0)
    assert features_arr.ndim == 2, f"Expected [N, D], got {features_arr.shape}"
    logger.info(f"  Final features shape: {features_arr.shape}")
    return features_arr, all_ids


# ============================================================================
# Dice Evaluation
# ============================================================================


def compute_dice_scores(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict]:
    """Compute per-subject Dice scores using frozen full model.

    Args:
        model: Full SwinUNETR model (encoder + decoder).
        dataset: Dataset returning "image" and "seg" keys.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        device: Torch device.

    Returns:
        List of dicts with subject_id, dice_TC, dice_WT, dice_ET per subject.
    """
    model.eval()
    model.to(device)
    metric = DiceMetric3Ch(threshold=0.5, reduction="none")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["image"].to(device)
            segs = batch["seg"].to(device)

            logits = model(images)  # [B, 3, D, H, W]
            dice = metric(logits, segs)  # [B, 3]

            for j in range(dice.shape[0]):
                sid = (
                    batch["subject_id"][j]
                    if isinstance(batch["subject_id"], (list, tuple))
                    else batch["subject_id"]
                )
                results.append(
                    {
                        "subject_id": sid,
                        "dice_TC": float(dice[j, 0]),
                        "dice_WT": float(dice[j, 1]),
                        "dice_ET": float(dice[j, 2]),
                    }
                )

            if (i + 1) % 10 == 0:
                logger.info(f"  Dice computed for {(i + 1) * batch_size}/{len(dataset)}")

    logger.info(f"  Dice results for {len(results)} subjects")
    return results


# ============================================================================
# Subject-Level Averaging (for MenGrowth longitudinal data)
# ============================================================================


def average_features_by_subject(
    features: np.ndarray,
    subject_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Average features per subject for datasets with repeated measures.

    Args:
        features: Feature array [N_scans, D].
        subject_ids: Subject ID for each scan.

    Returns:
        Tuple of (averaged_features [N_subjects, D], unique_subject_ids).
    """
    unique_ids = sorted(set(subject_ids))
    id_to_rows: dict[str, list[int]] = {uid: [] for uid in unique_ids}
    for i, sid in enumerate(subject_ids):
        id_to_rows[sid].append(i)

    averaged = np.zeros((len(unique_ids), features.shape[1]), dtype=features.dtype)
    for k, uid in enumerate(unique_ids):
        averaged[k] = features[id_to_rows[uid]].mean(axis=0)

    logger.info(f"  Averaged {features.shape[0]} scans -> {averaged.shape[0]} subjects")
    return averaged, unique_ids


# ============================================================================
# Per-Dimension KS Test
# ============================================================================


def compute_per_dim_ks(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Kolmogorov-Smirnov test per feature dimension.

    Args:
        feat_a: Features from domain A [N1, D].
        feat_b: Features from domain B [N2, D].

    Returns:
        Tuple of (ks_statistics [D], ks_pvalues [D]).
    """
    n_dims = feat_a.shape[1]
    ks_statistics = np.zeros(n_dims)
    ks_pvalues = np.zeros(n_dims)

    for d in range(n_dims):
        stat, pval = stats.ks_2samp(feat_a[:, d], feat_b[:, d])
        ks_statistics[d] = stat
        ks_pvalues[d] = pval

    logger.info(
        f"  KS test: max D={ks_statistics.max():.3f}, "
        f"min p={ks_pvalues.min():.2e}, "
        f"significant dims (p<0.05): {(ks_pvalues < 0.05).sum()}/{n_dims}"
    )
    return ks_statistics, ks_pvalues


# ============================================================================
# Domain Shift Metrics (pairwise)
# ============================================================================


def compute_pairwise_metrics(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    n_perm: int = 1000,
    cv_folds: int = 5,
    groups_a: np.ndarray | None = None,
    groups_b: np.ndarray | None = None,
) -> dict:
    """Compute domain shift metrics for one pair of domains.

    Args:
        feat_a: Features from domain A [N1, D].
        feat_b: Features from domain B [N2, D].
        n_perm: Number of permutations for MMD test.
        cv_folds: Cross-validation folds for classifier accuracy.
        groups_a: Group labels for domain A (for GroupKFold). Optional.
        groups_b: Group labels for domain B (for GroupKFold). Optional.

    Returns:
        Dict with pairwise domain shift metrics.
    """
    logger.info("  Computing MMD²...")
    mmd_sq, mmd_pvalue = mmd_permutation_test(feat_a, feat_b, n_perm=n_perm)
    logger.info(f"    MMD² = {mmd_sq:.4f}, p = {mmd_pvalue:.4f}")

    logger.info("  Computing CKA...")
    # CKA requires equal sample sizes — subsample the larger set
    n_a, n_b = feat_a.shape[0], feat_b.shape[0]
    if n_a == n_b:
        cka = compute_cka(feat_a, feat_b)
    else:
        rng = np.random.RandomState(42)
        n_min = min(n_a, n_b)
        fa = feat_a[rng.choice(n_a, n_min, replace=False)] if n_a > n_min else feat_a
        fb = feat_b[rng.choice(n_b, n_min, replace=False)] if n_b > n_min else feat_b
        cka = compute_cka(fa, fb)
        logger.info(f"    (subsampled to N={n_min} for CKA)")
    logger.info(f"    CKA = {cka:.4f}")

    logger.info("  Computing PAD...")
    pad = compute_proxy_a_distance(feat_a, feat_b)
    logger.info(f"    PAD = {pad:.4f}")

    logger.info("  Computing classifier accuracy...")
    if groups_a is not None or groups_b is not None:
        clf_acc = _domain_classifier_with_groups(feat_a, feat_b, cv_folds, groups_a, groups_b)
    else:
        clf_acc = compute_domain_classifier_accuracy(feat_a, feat_b, n_splits=cv_folds)
    logger.info(f"    Clf Acc = {clf_acc:.4f}")

    return {
        "mmd_sq": float(mmd_sq),
        "mmd_pvalue": float(mmd_pvalue),
        "cka": float(cka),
        "proxy_a_distance": float(pad),
        "classifier_accuracy": float(clf_acc),
    }


def _domain_classifier_with_groups(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    n_splits: int,
    groups_a: np.ndarray | None,
    groups_b: np.ndarray | None,
) -> float:
    """Domain classifier using GroupKFold when groups are available.

    Prevents same-subject scans from leaking across folds.

    Args:
        feat_a: Features from domain A.
        feat_b: Features from domain B.
        n_splits: Number of CV folds.
        groups_a: Group IDs for domain A (or None).
        groups_b: Group IDs for domain B (or None).

    Returns:
        Mean cross-validated accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([feat_a, feat_b])
    y = np.array([0] * len(feat_a) + [1] * len(feat_b))

    # Build groups: use provided group IDs, or assign unique IDs per sample
    ga = groups_a if groups_a is not None else np.arange(len(feat_a))
    gb = groups_b if groups_b is not None else np.arange(len(feat_a), len(feat_a) + len(feat_b))
    groups = np.concatenate([ga, gb])

    # Ensure enough groups for the number of splits
    n_unique = len(np.unique(groups))
    actual_splits = min(n_splits, n_unique)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv = GroupKFold(n_splits=actual_splits)
    scores = cross_val_score(clf, X, y, cv=cv, groups=groups)
    return float(scores.mean())


def compute_all_pairwise_metrics(
    feat_dict: dict[str, np.ndarray],
    groups_dict: dict[str, np.ndarray | None],
    n_perm: int = 1000,
    cv_folds: int = 5,
) -> dict[str, dict]:
    """Compute metrics for all C(k,2) domain pairs.

    Args:
        feat_dict: Mapping of domain name -> feature array.
        groups_dict: Mapping of domain name -> group IDs (or None).
        n_perm: MMD permutations.
        cv_folds: Classifier CV folds.

    Returns:
        Dict keyed by ``"{a}_{b}"`` with pairwise metrics.
    """
    domains = sorted(feat_dict.keys())
    results = {}

    for a, b in itertools.combinations(domains, 2):
        pair_key = f"{a}_{b}"
        logger.info(f"Computing pairwise metrics: {a} <-> {b}")
        results[pair_key] = compute_pairwise_metrics(
            feat_dict[a],
            feat_dict[b],
            n_perm=n_perm,
            cv_folds=cv_folds,
            groups_a=groups_dict.get(a),
            groups_b=groups_dict.get(b),
        )

    return results


def compute_all_pairwise_ks(
    feat_dict: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute per-dimension KS test for all domain pairs.

    Args:
        feat_dict: Mapping of domain name -> feature array.

    Returns:
        Dict keyed by ``"{a}_{b}"`` -> (ks_statistics, ks_pvalues).
    """
    domains = sorted(feat_dict.keys())
    results = {}

    for a, b in itertools.combinations(domains, 2):
        pair_key = f"{a}_{b}"
        logger.info(f"Computing KS stats: {a} <-> {b}")
        results[pair_key] = compute_per_dim_ks(feat_dict[a], feat_dict[b])

    return results


# ============================================================================
# Save Results
# ============================================================================


def _dice_summary(results: list[dict]) -> dict:
    """Compute summary statistics from per-subject Dice results."""
    tc = [r["dice_TC"] for r in results]
    wt = [r["dice_WT"] for r in results]
    et = [r["dice_ET"] for r in results]
    mean_dice = [(r["dice_TC"] + r["dice_WT"] + r["dice_ET"]) / 3 for r in results]

    return {
        "dice_TC_mean": float(np.mean(tc)),
        "dice_TC_std": float(np.std(tc)),
        "dice_WT_mean": float(np.mean(wt)),
        "dice_WT_std": float(np.std(wt)),
        "dice_ET_mean": float(np.mean(et)),
        "dice_ET_std": float(np.std(et)),
        "dice_mean_mean": float(np.mean(mean_dice)),
        "dice_mean_std": float(np.std(mean_dice)),
        "n_subjects": len(results),
    }


def save_results(
    output_dir: Path,
    features_dict: dict[str, np.ndarray],
    ids_dict: dict[str, list[str]],
    dice_dict: dict[str, list[dict]],
    pairwise_metrics: dict[str, dict],
    ks_dict: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save all results to structured output directory.

    Supports both 2-domain and 3-domain modes transparently.

    Args:
        output_dir: Root output directory.
        features_dict: Mapping domain -> feature array [N, D].
        ids_dict: Mapping domain -> subject/scan ID list.
        dice_dict: Mapping domain -> per-scan Dice results.
        pairwise_metrics: Mapping ``"{a}_{b}"`` -> metric dict.
        ks_dict: Mapping ``"{a}_{b}"`` -> (stats, pvals).
    """
    feat_dir = output_dir / "features"
    dice_dir = output_dir / "dice"
    metrics_dir = output_dir / "metrics"

    for d in [feat_dir, dice_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Features
    for domain, features in features_dict.items():
        np.savez(
            feat_dir / f"{domain}_features.npz",
            features=features,
            subject_ids=np.array(ids_dict[domain], dtype=object),
        )
        logger.info(f"Saved {domain} features: {features.shape}")

    # Dice
    for domain, dice_results in dice_dict.items():
        dice_out = {"per_subject": dice_results, "summary": _dice_summary(dice_results)}
        with open(dice_dir / f"{domain}_dice.json", "w") as f:
            json.dump(dice_out, f, indent=2)
        logger.info(f"Saved {domain} Dice: {len(dice_results)} entries")

    # Pairwise metrics
    with open(metrics_dir / "pairwise_metrics.json", "w") as f:
        json.dump(pairwise_metrics, f, indent=2)
    logger.info(f"Saved pairwise metrics: {list(pairwise_metrics.keys())}")

    # Per-dataset metrics (effective rank)
    per_dataset = {}
    for domain, features in features_dict.items():
        per_dataset[domain] = {
            "effective_rank": float(compute_effective_rank(features)),
        }
    with open(metrics_dir / "per_dataset_metrics.json", "w") as f:
        json.dump(per_dataset, f, indent=2)
    logger.info(f"Saved per-dataset metrics: {list(per_dataset.keys())}")

    # Backward-compat: also write domain_metrics.json for 2-domain case
    if set(features_dict.keys()) == {"gli", "men"}:
        pair = pairwise_metrics["gli_men"]
        compat = {
            "mmd_sq": pair["mmd_sq"],
            "mmd_pvalue": pair["mmd_pvalue"],
            "cka": pair["cka"],
            "proxy_a_distance": pair["proxy_a_distance"],
            "classifier_accuracy": pair["classifier_accuracy"],
            "effective_rank_gli": per_dataset["gli"]["effective_rank"],
            "effective_rank_men": per_dataset["men"]["effective_rank"],
        }
        with open(metrics_dir / "domain_metrics.json", "w") as f:
            json.dump(compat, f, indent=2)

    # KS statistics
    for pair_key, (ks_stats, ks_pvals) in ks_dict.items():
        np.savez(
            metrics_dir / f"ks_stats_{pair_key}.npz",
            ks_statistics=ks_stats,
            ks_pvalues=ks_pvals,
        )
        logger.info(f"Saved KS stats ({pair_key}): {ks_stats.shape}")

    # Backward-compat: also write ks_stats.npz for 2-domain case
    if "gli_men" in ks_dict:
        ks_s, ks_p = ks_dict["gli_men"]
        np.savez(
            metrics_dir / "ks_stats.npz",
            ks_statistics=ks_s,
            ks_pvalues=ks_p,
        )


# ============================================================================
# PCA Outlier Report
# ============================================================================


def write_pca_outlier_report(
    output_dir: Path,
    features_dict: dict[str, np.ndarray],
    ids_dict: dict[str, list[str]],
    sigma_threshold: float = 3.0,
    top_k: int = 10,
) -> None:
    """Detect PCA outliers per dataset and write a diagnostic report.

    Fits PCA on all domains combined, then flags samples that exceed
    ``sigma_threshold`` standard deviations on either PC, plus lists
    the top-k most extreme points by Euclidean distance.

    Args:
        output_dir: Root output directory.
        features_dict: Mapping domain -> features [N, D].
        ids_dict: Mapping domain -> ID list (same length as features).
        sigma_threshold: Number of std devs to flag as outlier.
        top_k: Number of top outliers to list.
    """
    from sklearn.decomposition import PCA

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Combine all features, track domain + id per row
    all_feat = []
    all_domains = []
    all_ids = []
    for domain in sorted(features_dict.keys()):
        feat = features_dict[domain]
        ids = ids_dict[domain]
        all_feat.append(feat)
        all_domains.extend([domain] * len(feat))
        all_ids.extend(ids)

    combined = np.vstack(all_feat)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)
    var_pct = pca.explained_variance_ratio_ * 100

    lines: list[str] = []
    lines.append("PCA Outlier Report")
    lines.append("=" * 72)
    lines.append(f"PC1 explains {var_pct[0]:.1f}%, PC2 explains {var_pct[1]:.1f}%")
    lines.append(f"Threshold: {sigma_threshold}σ")
    lines.append(f"Total samples: {len(combined)}")
    lines.append("")

    # Per-PC outlier detection
    for pc_idx in range(2):
        vals = proj[:, pc_idx]
        mean, std = vals.mean(), vals.std()
        mask = np.abs(vals - mean) > sigma_threshold * std

        lines.append(f"PC{pc_idx + 1} outliers (>{sigma_threshold}σ, mean={mean:.2f}, σ={std:.2f})")
        lines.append("-" * 72)

        if not mask.any():
            lines.append("  (none)")
        else:
            for i in np.where(mask)[0]:
                lines.append(
                    f"  {all_domains[i]:<12s}  {all_ids[i]:<30s}  "
                    f"PC1={proj[i, 0]:8.2f}  PC2={proj[i, 1]:8.2f}"
                )
        lines.append("")

    # Top-k by Euclidean distance, grouped by dataset
    dists = np.sqrt(proj[:, 0] ** 2 + proj[:, 1] ** 2)
    top_idx = np.argsort(dists)[::-1][:top_k]

    lines.append(f"Top {top_k} by Euclidean distance from PCA origin")
    lines.append("-" * 72)
    for rank, i in enumerate(top_idx, 1):
        lines.append(
            f"  {rank:2d}. {all_domains[i]:<12s}  {all_ids[i]:<30s}  "
            f"PC1={proj[i, 0]:8.2f}  PC2={proj[i, 1]:8.2f}  "
            f"dist={dists[i]:.2f}"
        )
    lines.append("")

    # Per-dataset breakdown: top outliers within each domain
    for domain in sorted(features_dict.keys()):
        domain_mask = np.array([d == domain for d in all_domains])
        domain_idx = np.where(domain_mask)[0]
        domain_dists = dists[domain_idx]
        top_in_domain = domain_idx[np.argsort(domain_dists)[::-1][:5]]

        lines.append(f"Top 5 outliers in {domain}")
        lines.append("-" * 72)
        for rank, i in enumerate(top_in_domain, 1):
            lines.append(
                f"  {rank}. {all_ids[i]:<30s}  "
                f"PC1={proj[i, 0]:8.2f}  PC2={proj[i, 1]:8.2f}  "
                f"dist={dists[i]:.2f}"
            )
        lines.append("")

    report = "\n".join(lines)

    out_path = diag_dir / "pca_outliers.txt"
    out_path.write_text(report)
    logger.info(f"Saved PCA outlier report: {out_path}")

    # Also log a summary
    n_outliers = sum(
        1
        for pc in range(2)
        for v in proj[:, pc]
        if abs(v - proj[:, pc].mean()) > sigma_threshold * proj[:, pc].std()
    )
    logger.info(f"  {n_outliers} samples exceed {sigma_threshold}σ threshold")


# ============================================================================
# Main Pipeline
# ============================================================================


def main() -> None:
    """Run the full domain gap analysis pipeline."""
    parser = argparse.ArgumentParser(description="Domain gap analysis (GPU)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    cfg = OmegaConf.load(args.config)
    seed = cfg.experiment.seed
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ckpt_path = cfg.paths.checkpoint
    h5_file = cfg.paths.h5_file
    glioma_root = cfg.paths.glioma_root
    mengrowth_root = OmegaConf.select(cfg, "paths.mengrowth_root", default=None)
    n_men = cfg.data.n_men_subjects
    mengrowth_first_only = OmegaConf.select(cfg, "data.mengrowth_first_only", default=False)
    roi_size = tuple(cfg.data.feature_roi_size)
    level = cfg.feature_extraction.level

    has_mengrowth = mengrowth_root is not None

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    logger.info("Loading frozen encoder...")
    encoder = load_swin_encoder(ckpt_path, include_encoder10=True, freeze=True, device=str(device))

    logger.info("Loading frozen full SwinUNETR...")
    full_model = load_full_swinunetr(
        ckpt_path, freeze_encoder=True, freeze_decoder=True, device=str(device)
    )

    # ------------------------------------------------------------------
    # 2. Create datasets
    # ------------------------------------------------------------------
    logger.info("Creating GLI dataset...")
    gli_dataset = BraTSGLIDataset(
        data_root=glioma_root,
        transform=get_val_transforms(roi_size=roi_size),
    )
    logger.info(f"  GLI subjects: {len(gli_dataset)}")

    logger.info(f"Creating MEN dataset (sampling {n_men} subjects)...")
    men_full = BraTSMENDatasetH5(
        h5_path=h5_file,
        split=None,
        transform=get_h5_val_transforms(roi_size=roi_size),
    )

    # Sample from lora_train + test splits for balanced representation
    splits = BraTSMENDatasetH5.load_splits_from_h5(h5_file)
    available_indices = np.concatenate(
        [
            splits.get("lora_train", np.array([], dtype=int)),
            splits.get("sdp_train", np.array([], dtype=int)),  # legacy compat
            splits.get("test", np.array([], dtype=int)),
        ]
    ).astype(int)
    # Deduplicate in case of overlap between legacy and new splits
    available_indices = np.unique(available_indices)

    if len(available_indices) == 0:
        # Fallback: use all subjects
        logger.warning("No train/test splits found, using all subjects")
        available_indices = np.arange(len(men_full))

    rng = np.random.RandomState(seed)
    n_sample = min(n_men, len(available_indices))
    sampled_indices = rng.choice(available_indices, size=n_sample, replace=False)
    men_dataset = Subset(men_full, sampled_indices.tolist())
    logger.info(f"  MEN subjects: {len(men_dataset)} (sampled from {len(available_indices)})")

    mg_dataset = None
    if has_mengrowth:
        logger.info("Creating MenGrowth dataset...")
        mg_dataset = MenGrowthDataset(
            data_root=mengrowth_root,
            first_timepoint_only=mengrowth_first_only,
            transform=get_val_transforms(roi_size=roi_size),
        )
        logger.info(f"  MenGrowth scans: {len(mg_dataset)}")

    # ------------------------------------------------------------------
    # 3. Extract features
    # ------------------------------------------------------------------
    logger.info("Extracting GLI features...")
    gli_features, gli_ids = extract_features(
        encoder,
        gli_dataset,
        level=level,
        batch_size=cfg.feature_extraction.batch_size,
        num_workers=cfg.feature_extraction.num_workers,
        device=device,
    )

    logger.info("Extracting MEN features...")
    men_features, men_ids = extract_features(
        encoder,
        men_dataset,
        level=level,
        batch_size=cfg.feature_extraction.batch_size,
        num_workers=cfg.feature_extraction.num_workers,
        device=device,
    )

    mg_features = None
    mg_ids = None
    mg_subject_ids = None
    if mg_dataset is not None:
        logger.info("Extracting MenGrowth features...")
        mg_features, mg_ids = extract_features(
            encoder,
            mg_dataset,
            level=level,
            batch_size=cfg.feature_extraction.batch_size,
            num_workers=cfg.feature_extraction.num_workers,
            device=device,
        )
        # Record subject IDs for averaging
        mg_subject_ids = [mg_dataset.scans[i][0] for i in range(len(mg_dataset))]

    # Validate
    for name, feat in [("GLI", gli_features), ("MEN", men_features)]:
        assert not np.isnan(feat).any(), f"{name} features contain NaN"
        assert not np.isinf(feat).any(), f"{name} features contain Inf"
    if mg_features is not None:
        assert not np.isnan(mg_features).any(), "MenGrowth features contain NaN"
        assert not np.isinf(mg_features).any(), "MenGrowth features contain Inf"

    # ------------------------------------------------------------------
    # 4. Compute Dice scores
    # ------------------------------------------------------------------
    logger.info("Computing GLI Dice scores...")
    gli_dice = compute_dice_scores(
        full_model,
        gli_dataset,
        batch_size=cfg.dice_evaluation.batch_size,
        num_workers=cfg.dice_evaluation.num_workers,
        device=device,
    )

    logger.info("Computing MEN Dice scores...")
    men_dice = compute_dice_scores(
        full_model,
        men_dataset,
        batch_size=cfg.dice_evaluation.batch_size,
        num_workers=cfg.dice_evaluation.num_workers,
        device=device,
    )

    mg_dice = None
    if mg_dataset is not None:
        logger.info("Computing MenGrowth Dice scores...")
        mg_dice = compute_dice_scores(
            full_model,
            mg_dataset,
            batch_size=cfg.dice_evaluation.batch_size,
            num_workers=cfg.dice_evaluation.num_workers,
            device=device,
        )

    # ------------------------------------------------------------------
    # 5. Compute domain metrics
    # ------------------------------------------------------------------
    # Build feature dict using subject-averaged MenGrowth features for metrics
    feat_dict: dict[str, np.ndarray] = {
        "gli": gli_features,
        "men": men_features,
    }
    groups_dict: dict[str, np.ndarray | None] = {
        "gli": None,
        "men": None,
    }

    if mg_features is not None:
        mg_avg_features, mg_avg_ids = average_features_by_subject(mg_features, mg_subject_ids)
        feat_dict["mengrowth"] = mg_avg_features
        groups_dict["mengrowth"] = None  # Already averaged, no groups needed

    logger.info("Computing pairwise domain shift metrics...")
    pairwise_metrics = compute_all_pairwise_metrics(
        feat_dict,
        groups_dict,
        n_perm=cfg.metrics.mmd_permutations,
        cv_folds=cfg.metrics.classifier_cv_folds,
    )

    logger.info("Computing per-pair KS statistics...")
    ks_dict = compute_all_pairwise_ks(feat_dict)

    # ------------------------------------------------------------------
    # 6. Save everything
    # ------------------------------------------------------------------
    # Build save dicts (use all scans for MenGrowth features/dice on disk)
    save_features: dict[str, np.ndarray] = {
        "gli": gli_features,
        "men": men_features,
    }
    save_ids: dict[str, list[str]] = {
        "gli": gli_ids,
        "men": men_ids,
    }
    save_dice: dict[str, list[dict]] = {
        "gli": gli_dice,
        "men": men_dice,
    }

    if mg_features is not None:
        save_features["mengrowth"] = mg_features  # All scans
        save_ids["mengrowth"] = mg_ids  # Scan IDs
        save_dice["mengrowth"] = mg_dice

    logger.info(f"Saving results to {output_dir}")
    save_results(
        output_dir,
        save_features,
        save_ids,
        save_dice,
        pairwise_metrics,
        ks_dict,
    )

    # ------------------------------------------------------------------
    # 7. PCA outlier report
    # ------------------------------------------------------------------
    logger.info("Writing PCA outlier report...")
    write_pca_outlier_report(output_dir, save_features, save_ids)

    logger.info("Domain gap analysis complete.")


if __name__ == "__main__":
    main()
