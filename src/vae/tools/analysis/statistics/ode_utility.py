"""Neural ODE utility metrics.

Computes factor independence, cross-partition correlations, and
ODE readiness scores for downstream Neural ODE training assessment.
"""

import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ..schemas import ODEUtilityMetrics, ODEReadinessGrade

logger = logging.getLogger(__name__)


# Thresholds for ODE readiness
VOL_R2_THRESHOLD = 0.85
LOC_R2_THRESHOLD = 0.90
SHAPE_R2_THRESHOLD = 0.35
CROSS_CORR_THRESHOLD = 0.30


def compute_ode_utility_metrics(
    data: Dict[str, Any],
    epoch: Optional[int] = None,
    vol_r2: Optional[float] = None,
    loc_r2: Optional[float] = None,
    shape_r2: Optional[float] = None,
) -> ODEUtilityMetrics:
    """Compute ODE utility metrics for a specific epoch.

    Args:
        data: Dictionary from load_experiment_data()
        epoch: Specific epoch (None = use final)
        vol_r2: Override volume R² (from PerformanceMetrics)
        loc_r2: Override location R² (from PerformanceMetrics)
        shape_r2: Override shape R² (from PerformanceMetrics)

    Returns:
        ODEUtilityMetrics dataclass
    """
    metrics_df = data.get("metrics")
    cross_corr_df = data.get("cross_correlation")
    semantic_df = data.get("semantic_quality")

    result = ODEUtilityMetrics()

    # Determine epoch
    if epoch is None and metrics_df is not None and not metrics_df.empty:
        epoch = int(metrics_df["epoch"].max())

    result.epoch = epoch if epoch is not None else -1

    # Get semantic R² values
    if semantic_df is not None and not semantic_df.empty:
        # Handle long format (partition column) vs wide format (r2_vol columns)
        if "partition" in semantic_df.columns:
            # Long format: each row is one partition
            sem_epoch_data = semantic_df[semantic_df["epoch"] == epoch] if epoch else None
            if sem_epoch_data is None or sem_epoch_data.empty:
                available = semantic_df["epoch"].unique()
                if len(available) > 0:
                    if epoch is not None:
                        closest = min(available, key=lambda x: abs(x - epoch))
                    else:
                        closest = max(available)
                    sem_epoch_data = semantic_df[semantic_df["epoch"] == closest]

            if not sem_epoch_data.empty:
                for partition, attr in [("z_vol", "vol_r2"), ("z_loc", "loc_r2"), ("z_shape", "shape_r2")]:
                    part_row = sem_epoch_data[sem_epoch_data["partition"] == partition]
                    if not part_row.empty and "r2" in part_row.columns:
                        val = part_row["r2"].iloc[0]
                        if pd.notna(val):
                            if attr == "vol_r2" and vol_r2 is None:
                                vol_r2 = float(val)
                            elif attr == "loc_r2" and loc_r2 is None:
                                loc_r2 = float(val)
                            elif attr == "shape_r2" and shape_r2 is None:
                                shape_r2 = float(val)
        else:
            # Wide format: columns like r2_vol, r2_loc, etc.
            sem_epoch = semantic_df[semantic_df["epoch"] == epoch] if epoch else semantic_df.iloc[[-1]]
            if sem_epoch.empty and epoch is not None:
                available = semantic_df["epoch"].unique()
                if len(available) > 0:
                    closest = min(available, key=lambda x: abs(x - epoch))
                    sem_epoch = semantic_df[semantic_df["epoch"] == closest]

            if not sem_epoch.empty:
                row = sem_epoch.iloc[0]
                if vol_r2 is None and "r2_vol" in row.index:
                    vol_r2 = float(row["r2_vol"]) if pd.notna(row["r2_vol"]) else 0.0
                if loc_r2 is None and "r2_loc" in row.index:
                    loc_r2 = float(row["r2_loc"]) if pd.notna(row["r2_loc"]) else 0.0
                if shape_r2 is None and "r2_shape" in row.index:
                    shape_r2 = float(row["r2_shape"]) if pd.notna(row["r2_shape"]) else 0.0

    # Defaults
    vol_r2 = vol_r2 or 0.0
    loc_r2 = loc_r2 or 0.0
    shape_r2 = shape_r2 or 0.0

    # Get cross-partition correlations
    if cross_corr_df is not None and not cross_corr_df.empty:
        # Handle long format (partition_i, partition_j columns)
        if "partition_i" in cross_corr_df.columns and "partition_j" in cross_corr_df.columns:
            corr_epoch = cross_corr_df[cross_corr_df["epoch"] == epoch] if epoch else None
            if corr_epoch is None or corr_epoch.empty:
                # Use closest available epoch
                available = cross_corr_df["epoch"].unique()
                if len(available) > 0:
                    if epoch is not None:
                        closest = min(available, key=lambda x: abs(x - epoch))
                    else:
                        closest = max(available)
                    corr_epoch = cross_corr_df[cross_corr_df["epoch"] == closest]

            if not corr_epoch.empty:
                # Extract correlations between supervised partitions
                pairs = [
                    ("corr_vol_loc", "z_vol", "z_loc"),
                    ("corr_vol_shape", "z_vol", "z_shape"),
                    ("corr_loc_shape", "z_loc", "z_shape"),
                ]
                for attr, p1, p2 in pairs:
                    # Find row matching this pair (order may vary)
                    mask = ((corr_epoch["partition_i"] == p1) & (corr_epoch["partition_j"] == p2)) | \
                           ((corr_epoch["partition_i"] == p2) & (corr_epoch["partition_j"] == p1))
                    pair_row = corr_epoch[mask]
                    if not pair_row.empty:
                        corr_col = "abs_correlation" if "abs_correlation" in pair_row.columns else "correlation"
                        val = pair_row[corr_col].iloc[0]
                        if pd.notna(val):
                            setattr(result, attr, abs(float(val)))
        else:
            # Wide format: columns like z_vol_z_loc
            corr_epoch = cross_corr_df[cross_corr_df["epoch"] == epoch] if epoch else cross_corr_df.iloc[[-1]]
            if corr_epoch.empty and epoch is not None:
                available = cross_corr_df["epoch"].unique()
                if len(available) > 0:
                    closest = min(available, key=lambda x: abs(x - epoch))
                    corr_epoch = cross_corr_df[cross_corr_df["epoch"] == closest]

            if not corr_epoch.empty:
                row = corr_epoch.iloc[0]
                corr_pairs = [
                    ("corr_vol_loc", "z_vol_z_loc"),
                    ("corr_vol_shape", "z_vol_z_shape"),
                    ("corr_loc_shape", "z_loc_z_shape"),
                ]
                for attr, col in corr_pairs:
                    if col in row.index and pd.notna(row[col]):
                        setattr(result, attr, abs(float(row[col])))

    # Alternatively, get from metrics if cross_part/ columns exist
    if metrics_df is not None and not metrics_df.empty:
        epoch_data = metrics_df[metrics_df["epoch"] == epoch] if epoch else metrics_df.iloc[[-1]]
        if not epoch_data.empty:
            row = epoch_data.iloc[0]
            # Check for cross_part/ columns
            cross_cols = {
                "corr_vol_loc": ["cross_part/z_vol_z_loc", "val_cross_part/z_vol_z_loc"],
                "corr_vol_shape": ["cross_part/z_vol_z_shape", "val_cross_part/z_vol_z_shape"],
                "corr_loc_shape": ["cross_part/z_loc_z_shape", "val_cross_part/z_loc_z_shape"],
            }
            for attr, cols in cross_cols.items():
                for col in cols:
                    if col in row.index and pd.notna(row[col]):
                        setattr(result, attr, abs(float(row[col])))
                        break

    # Compute max cross-correlation
    result.max_cross_corr = max(
        result.corr_vol_loc,
        result.corr_vol_shape,
        result.corr_loc_shape,
    )

    # Independence score: 1 - max_cross_corr
    result.independence_score = max(0.0, 1.0 - result.max_cross_corr)

    # Store R² values in result for history tracking
    result.vol_r2 = vol_r2
    result.loc_r2 = loc_r2
    result.shape_r2 = shape_r2

    # ODE readiness composite score
    # 50% vol, 25% loc, 25% independence
    result.ode_readiness = (
        0.50 * vol_r2 +
        0.25 * loc_r2 +
        0.25 * result.independence_score
    )

    # Individual readiness flags
    result.vol_ready = vol_r2 >= VOL_R2_THRESHOLD
    result.loc_ready = loc_r2 >= LOC_R2_THRESHOLD
    result.shape_ready = shape_r2 >= SHAPE_R2_THRESHOLD
    result.factors_independent = result.max_cross_corr < CROSS_CORR_THRESHOLD

    # Compute grade
    result.grade = _compute_grade(
        vol_r2=vol_r2,
        loc_r2=loc_r2,
        factors_independent=result.factors_independent,
    )

    return result


def _compute_grade(
    vol_r2: float,
    loc_r2: float,
    factors_independent: bool,
) -> ODEReadinessGrade:
    """Compute ODE readiness grade.

    Args:
        vol_r2: Volume R²
        loc_r2: Location R²
        factors_independent: Whether factors are independent

    Returns:
        ODEReadinessGrade enum value
    """
    vol_ready = vol_r2 >= VOL_R2_THRESHOLD
    loc_ready = loc_r2 >= LOC_R2_THRESHOLD

    if vol_ready and loc_ready and factors_independent:
        return ODEReadinessGrade.A
    elif vol_ready and loc_ready:
        return ODEReadinessGrade.B
    elif vol_ready:
        return ODEReadinessGrade.C
    elif vol_r2 >= 0.70:
        return ODEReadinessGrade.D
    else:
        return ODEReadinessGrade.F


def compute_ode_utility_history(
    data: Dict[str, Any],
    epochs: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute ODE utility metrics over training history.

    Args:
        data: Dictionary from load_experiment_data()
        epochs: Specific epochs (None = all available)

    Returns:
        DataFrame with ODE utility metrics per epoch
    """
    cross_corr_df = data.get("cross_correlation")
    semantic_df = data.get("semantic_quality")

    # Determine epochs from available data
    if cross_corr_df is not None and not cross_corr_df.empty:
        all_epochs = sorted(cross_corr_df["epoch"].unique())
    elif semantic_df is not None and not semantic_df.empty:
        all_epochs = sorted(semantic_df["epoch"].unique())
    else:
        return pd.DataFrame()

    if epochs is not None:
        all_epochs = [e for e in all_epochs if e in epochs]

    records = []
    for epoch in all_epochs:
        metrics = compute_ode_utility_metrics(data, epoch=epoch)
        records.append({
            "epoch": metrics.epoch,
            "vol_r2": metrics.vol_r2,
            "loc_r2": metrics.loc_r2,
            "shape_r2": metrics.shape_r2,
            "corr_vol_loc": metrics.corr_vol_loc,
            "corr_vol_shape": metrics.corr_vol_shape,
            "corr_loc_shape": metrics.corr_loc_shape,
            "max_cross_corr": metrics.max_cross_corr,
            "independence_score": metrics.independence_score,
            "ode_readiness": metrics.ode_readiness,
            "vol_ready": metrics.vol_ready,
            "loc_ready": metrics.loc_ready,
            "shape_ready": metrics.shape_ready,
            "factors_independent": metrics.factors_independent,
            "grade": metrics.grade.value,
        })

    return pd.DataFrame(records)
