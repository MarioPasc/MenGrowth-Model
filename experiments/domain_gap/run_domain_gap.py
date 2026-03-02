"""Domain gap analysis pipeline (GPU).

Extracts encoder10 features and computes segmentation Dice for both
BraTS-GLI (glioma) and BraTS-MEN (meningioma) using the frozen
BrainSegFounder model, then computes domain shift metrics.

Usage:
    python -m experiments.domain_gap.run_domain_gap --config <path>
"""

from __future__ import annotations

import argparse
import json
import logging
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
# Per-Dimension KS Test
# ============================================================================


def compute_per_dim_ks(
    gli_feat: np.ndarray,
    men_feat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Kolmogorov-Smirnov test per feature dimension.

    Args:
        gli_feat: GLI features [N1, D].
        men_feat: MEN features [N2, D].

    Returns:
        Tuple of (ks_statistics [D], ks_pvalues [D]).
    """
    n_dims = gli_feat.shape[1]
    ks_statistics = np.zeros(n_dims)
    ks_pvalues = np.zeros(n_dims)

    for d in range(n_dims):
        stat, pval = stats.ks_2samp(gli_feat[:, d], men_feat[:, d])
        ks_statistics[d] = stat
        ks_pvalues[d] = pval

    logger.info(
        f"  KS test: max D={ks_statistics.max():.3f}, "
        f"min p={ks_pvalues.min():.2e}, "
        f"significant dims (p<0.05): {(ks_pvalues < 0.05).sum()}/{n_dims}"
    )
    return ks_statistics, ks_pvalues


# ============================================================================
# Domain Shift Metrics
# ============================================================================


def compute_domain_metrics(
    gli_feat: np.ndarray,
    men_feat: np.ndarray,
    n_perm: int = 1000,
    cv_folds: int = 5,
) -> dict:
    """Compute comprehensive domain shift metrics.

    Args:
        gli_feat: GLI features [N1, D].
        men_feat: MEN features [N2, D].
        n_perm: Number of permutations for MMD test.
        cv_folds: Cross-validation folds for classifier accuracy.

    Returns:
        Dict with all domain shift metrics.
    """
    logger.info("Computing MMD² with permutation test...")
    mmd_sq, mmd_pvalue = mmd_permutation_test(gli_feat, men_feat, n_perm=n_perm)
    logger.info(f"  MMD² = {mmd_sq:.4f}, p = {mmd_pvalue:.4f}")

    logger.info("Computing CKA...")
    cka = compute_cka(gli_feat, men_feat)
    logger.info(f"  CKA = {cka:.4f}")

    logger.info("Computing Proxy A-distance...")
    pad = compute_proxy_a_distance(gli_feat, men_feat)
    logger.info(f"  PAD = {pad:.4f}")

    logger.info("Computing domain classifier accuracy...")
    clf_acc = compute_domain_classifier_accuracy(gli_feat, men_feat, n_splits=cv_folds)
    logger.info(f"  Classifier Acc = {clf_acc:.4f}")

    logger.info("Computing effective rank...")
    eff_rank_gli = compute_effective_rank(gli_feat)
    eff_rank_men = compute_effective_rank(men_feat)
    logger.info(f"  Effective rank: GLI={eff_rank_gli:.1f}, MEN={eff_rank_men:.1f}")

    return {
        "mmd_sq": float(mmd_sq),
        "mmd_pvalue": float(mmd_pvalue),
        "cka": float(cka),
        "proxy_a_distance": float(pad),
        "classifier_accuracy": float(clf_acc),
        "effective_rank_gli": float(eff_rank_gli),
        "effective_rank_men": float(eff_rank_men),
    }


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
    gli_features: np.ndarray,
    men_features: np.ndarray,
    gli_ids: list[str],
    men_ids: list[str],
    gli_dice: list[dict],
    men_dice: list[dict],
    domain_metrics: dict,
    ks_statistics: np.ndarray,
    ks_pvalues: np.ndarray,
) -> None:
    """Save all results to structured output directory.

    Args:
        output_dir: Root output directory.
        gli_features: GLI feature array [N, 768].
        men_features: MEN feature array [N, 768].
        gli_ids: GLI subject IDs.
        men_ids: MEN subject IDs.
        gli_dice: Per-subject GLI Dice results.
        men_dice: Per-subject MEN Dice results.
        domain_metrics: Domain shift metrics dict.
        ks_statistics: Per-dimension KS statistics [D].
        ks_pvalues: Per-dimension KS p-values [D].
    """
    feat_dir = output_dir / "features"
    dice_dir = output_dir / "dice"
    metrics_dir = output_dir / "metrics"

    for d in [feat_dir, dice_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Features
    np.savez(
        feat_dir / "gli_features.npz",
        features=gli_features,
        subject_ids=np.array(gli_ids, dtype=object),
    )
    np.savez(
        feat_dir / "men_features.npz",
        features=men_features,
        subject_ids=np.array(men_ids, dtype=object),
    )
    logger.info(f"Saved features: GLI {gli_features.shape}, MEN {men_features.shape}")

    # Dice
    gli_dice_out = {"per_subject": gli_dice, "summary": _dice_summary(gli_dice)}
    men_dice_out = {"per_subject": men_dice, "summary": _dice_summary(men_dice)}

    with open(dice_dir / "gli_dice.json", "w") as f:
        json.dump(gli_dice_out, f, indent=2)
    with open(dice_dir / "men_dice.json", "w") as f:
        json.dump(men_dice_out, f, indent=2)
    logger.info(f"Saved Dice: GLI {len(gli_dice)} subjects, MEN {len(men_dice)} subjects")

    # Domain metrics
    with open(metrics_dir / "domain_metrics.json", "w") as f:
        json.dump(domain_metrics, f, indent=2)
    logger.info(f"Saved domain metrics: {list(domain_metrics.keys())}")

    # KS statistics
    np.savez(
        metrics_dir / "ks_stats.npz",
        ks_statistics=ks_statistics,
        ks_pvalues=ks_pvalues,
    )
    logger.info(f"Saved KS stats: {ks_statistics.shape}")


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
    n_men = cfg.data.n_men_subjects
    roi_size = tuple(cfg.data.feature_roi_size)
    level = cfg.feature_extraction.level

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

    logger.info("Creating MEN dataset (sampling {n_men} subjects)...")
    men_full = BraTSMENDatasetH5(
        h5_path=h5_file,
        split=None,
        transform=get_h5_val_transforms(roi_size=roi_size),
    )

    # Sample from sdp_train + test splits for balanced representation
    splits = BraTSMENDatasetH5.load_splits_from_h5(h5_file)
    available_indices = np.concatenate(
        [
            splits.get("sdp_train", np.array([], dtype=int)),
            splits.get("test", np.array([], dtype=int)),
        ]
    ).astype(int)

    if len(available_indices) == 0:
        # Fallback: use all subjects
        logger.warning("No sdp_train/test splits found, using all subjects")
        available_indices = np.arange(len(men_full))

    rng = np.random.RandomState(seed)
    n_sample = min(n_men, len(available_indices))
    sampled_indices = rng.choice(available_indices, size=n_sample, replace=False)
    men_dataset = Subset(men_full, sampled_indices.tolist())
    logger.info(f"  MEN subjects: {len(men_dataset)} (sampled from {len(available_indices)})")

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

    # Validate
    assert not np.isnan(gli_features).any(), "GLI features contain NaN"
    assert not np.isnan(men_features).any(), "MEN features contain NaN"
    assert not np.isinf(gli_features).any(), "GLI features contain Inf"
    assert not np.isinf(men_features).any(), "MEN features contain Inf"

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

    # ------------------------------------------------------------------
    # 5. Compute domain metrics
    # ------------------------------------------------------------------
    logger.info("Computing domain shift metrics...")
    domain_metrics = compute_domain_metrics(
        gli_features,
        men_features,
        n_perm=cfg.metrics.mmd_permutations,
        cv_folds=cfg.metrics.classifier_cv_folds,
    )

    logger.info("Computing per-dimension KS statistics...")
    ks_statistics, ks_pvalues = compute_per_dim_ks(gli_features, men_features)

    # ------------------------------------------------------------------
    # 6. Save everything
    # ------------------------------------------------------------------
    logger.info(f"Saving results to {output_dir}")
    save_results(
        output_dir,
        gli_features,
        men_features,
        gli_ids,
        men_ids,
        gli_dice,
        men_dice,
        domain_metrics,
        ks_statistics,
        ks_pvalues,
    )

    logger.info("Domain gap analysis complete.")


if __name__ == "__main__":
    main()
