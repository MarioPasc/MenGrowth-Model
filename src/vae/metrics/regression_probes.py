"""Regression probes for latent space analysis.

Provides functions to extract tumor characteristics from segmentation masks
and test predictability of these characteristics from latent codes using
ridge regression with proper cross-validation.

Functions:
    extract_segmentation_targets: Extract tumor features from segmentation masks
    ridge_probe_cv: Ridge regression with cross-validation
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, Any
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import logging

logger = logging.getLogger(__name__)


def extract_segmentation_targets(
    seg_batch: torch.Tensor,
    label_map: Dict[str, int],
    spacing: Tuple[float, float, float] = (1.875, 1.875, 1.875)
) -> pd.DataFrame:
    """Extract tumor characteristics from segmentation masks.

    Computes volumes, centroids, and ratios for different tumor compartments
    (e.g., necrotic core, edema, enhancing tumor).

    Args:
        seg_batch: Segmentation masks [B, 1, D, H, W] or [B, D, H, W]
        label_map: Dict mapping compartment names to label values
                   e.g., {"ncr": 1, "ed": 2, "et": 3}
        spacing: Voxel spacing in mm (D, H, W) for volume calculation

    Returns:
        DataFrame with columns:
        - logV_<compartment>: Log volumes in mm³ (using log1p for zeros)
        - logV_total: Log total tumor volume
        - cx_total, cy_total, cz_total: Centroid coordinates in voxel units
        - r_<compartment>: Volume ratios (compartment_volume / total_volume)

    Example:
        >>> seg = torch.randint(0, 4, (10, 1, 128, 128, 128))
        >>> label_map = {"ncr": 1, "ed": 2, "et": 3}
        >>> df = extract_segmentation_targets(seg, label_map)
        >>> print(df.columns)
    """
    # Handle [B, 1, D, H, W] or [B, D, H, W]
    if seg_batch.ndim == 5:
        seg_batch = seg_batch.squeeze(1)  # [B, D, H, W]

    B = seg_batch.shape[0]
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    # Extract compartment labels
    compartments = list(label_map.keys())
    ncr_label = label_map.get("ncr", 1)
    ed_label = label_map.get("ed", 2)
    et_label = label_map.get("et", 3)

    targets = []

    for i in range(B):
        seg = seg_batch[i]  # [D, H, W]

        # Volumes (voxel counts)
        V_ncr = (seg == ncr_label).sum().item()
        V_ed = (seg == ed_label).sum().item()
        V_et = (seg == et_label).sum().item()
        V_total = V_ncr + V_ed + V_et

        # Log volumes in mm³ (with offset to handle zeros)
        logV_ncr = np.log1p(V_ncr * voxel_volume_mm3)
        logV_ed = np.log1p(V_ed * voxel_volume_mm3)
        logV_et = np.log1p(V_et * voxel_volume_mm3)
        logV_total = np.log1p(V_total * voxel_volume_mm3)

        # Centroid (only for total tumor mask)
        if V_total > 0:
            mask_total = seg > 0
            coords = torch.nonzero(mask_total, as_tuple=False).float()  # [N_voxels, 3]
            centroid = coords.mean(dim=0)  # [3]: (D_idx, H_idx, W_idx)
            cz, cy, cx = centroid.tolist()
        else:
            cz, cy, cx = np.nan, np.nan, np.nan

        # Ratios (handle division by zero)
        if V_total > 0:
            r_ncr = V_ncr / V_total
            r_ed = V_ed / V_total
            r_et = V_et / V_total
        else:
            r_ncr = r_ed = r_et = np.nan

        targets.append(
            {
                "logV_ncr": logV_ncr,
                "logV_ed": logV_ed,
                "logV_et": logV_et,
                "logV_total": logV_total,
                "cz_total": cz,
                "cy_total": cy,
                "cx_total": cx,
                "r_ncr": r_ncr,
                "r_ed": r_ed,
                "r_et": r_et,
            }
        )

    df = pd.DataFrame(targets)
    return df


def ridge_probe_cv(
    z: np.ndarray,
    targets_df: pd.DataFrame,
    n_folds: int = 5,
    alpha: float = 1.0,
    random_state: int = 42
) -> Dict[str, Any]:
    """Ridge regression probe with proper cross-validation.

    Uses Pipeline + TransformedTargetRegressor to ensure scalers are fit
    ONLY on training folds, preventing test data from leaking into preprocessing.
    Reports mean ± std of held-out R² across folds.

    Reference: sklearn best practices for preprocessing in CV.

    Args:
        z: Latent codes [N, z_dim] (numpy array)
        targets_df: DataFrame with target columns
                    (logV_*, c*_total, r_*)
        n_folds: Number of cross-validation folds
        alpha: Ridge regression regularization parameter
        random_state: Random seed for CV splits

    Returns:
        Dictionary with keys:
        - r2_<target>_mean: Mean R² across folds
        - r2_<target>_std: Std R² across folds
        - top5dims_<target>: Top 5 predictive dimensions (for key targets)
        - n_empty_<compartment>: Number of empty (NaN) samples per compartment

    Example:
        >>> z = np.random.randn(100, 128)
        >>> targets_df = pd.DataFrame({
        ...     'logV_total': np.random.randn(100),
        ...     'cx_total': np.random.randn(100)
        ... })
        >>> results = ridge_probe_cv(z, targets_df)
        >>> print(f"R²(logV_total): {results['r2_logV_total_mean']:.3f}")
    """
    X = z  # [N, z_dim]

    results = {}

    # Define targets to probe
    target_cols = [
        "logV_ncr",
        "logV_ed",
        "logV_et",
        "logV_total",
        "cz_total",
        "cy_total",
        "cx_total",
        "r_ncr",
        "r_ed",
        "r_et",
    ]

    # Track empty counts
    empty_counts = {"ncr": 0, "ed": 0, "et": 0, "total": 0}

    for target_col in target_cols:
        if target_col not in targets_df.columns:
            logger.warning(f"Target column '{target_col}' not found in targets_df, skipping")
            continue

        y = targets_df[target_col].values  # [N]

        # Count invalid samples
        valid_mask = np.isfinite(y)
        n_valid = valid_mask.sum()
        n_empty = len(y) - n_valid

        # Update empty counts for volume targets
        if target_col.startswith("logV_"):
            compartment = target_col.replace("logV_", "")
            empty_counts[compartment] = int(n_empty)

        # Skip if too few valid samples for CV (need at least 10 for 5-fold)
        if n_valid < 10:
            logger.warning(
                f"Target {target_col} has only {n_valid} valid samples (need ≥10 for {n_folds}-fold CV), skipping"
            )
            results[f"r2_{target_col}_mean"] = np.nan
            results[f"r2_{target_col}_std"] = np.nan
            # Only add top5dims for key targets
            if target_col in ["logV_total", "cx_total", "cy_total", "cz_total"]:
                results[f"top5dims_{target_col}"] = ""
            continue

        # Filter to valid samples
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Pipeline: StandardScaler fit ONLY on train fold, then Ridge
        # TransformedTargetRegressor: same for y scaling
        # This prevents preprocessing leakage from test fold
        model = TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=alpha, random_state=random_state))
            ]),
            transformer=StandardScaler()
        )

        # Cross-validation with proper preprocessing
        cv_scores = cross_val_score(
            model, X_valid, y_valid,
            cv=n_folds,
            scoring='r2',
            n_jobs=1  # Single-threaded for determinism
        )

        # Log mean and std across folds
        results[f"r2_{target_col}_mean"] = float(np.mean(cv_scores))
        results[f"r2_{target_col}_std"] = float(np.std(cv_scores))

        # Top-5 coefficient dimensions (fit on all valid data for interpretability)
        if target_col in ["logV_total", "cx_total", "cy_total", "cz_total"]:
            # Fit full model to get coefficients (for interpretability only)
            model_full = TransformedTargetRegressor(
                regressor=Pipeline([
                    ('scaler', StandardScaler()),
                    ('ridge', Ridge(alpha=alpha, random_state=random_state))
                ]),
                transformer=StandardScaler()
            )
            model_full.fit(X_valid, y_valid)
            # Extract coefficients from the ridge regressor in the pipeline
            coef_abs = np.abs(model_full.regressor_.named_steps['ridge'].coef_)
            top5_idx = np.argsort(coef_abs)[::-1][:5]
            results[f"top5dims_{target_col}"] = f"[{','.join(map(str, top5_idx))}]"

    # Add empty counts
    for compartment, count in empty_counts.items():
        results[f"n_empty_{compartment}"] = count

    return results
