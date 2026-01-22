"""Collapse detection metrics.

Computes active units, variance analysis, and decoder bypass detection
to identify posterior collapse or decoder ignoring latent codes.
"""

import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ..schemas import CollapseMetrics

logger = logging.getLogger(__name__)


# Thresholds for collapse detection
AU_RESIDUAL_COLLAPSE_THRESHOLD = 0.10  # AU_frac < 10% = collapsed
DECODER_BYPASS_THRESHOLD = 0.20  # (z0_mse - mu_mse) / mu_mse < 20% = bypassed
KL_FLOOR_DEFAULT = 0.2  # nats per dim


def compute_collapse_metrics(
    data: Dict[str, Any],
    epoch: Optional[int] = None,
    kl_floor: float = KL_FLOOR_DEFAULT,
) -> CollapseMetrics:
    """Compute collapse detection metrics for a specific epoch.

    Args:
        data: Dictionary from load_experiment_data()
        epoch: Specific epoch (None = use final)
        kl_floor: KL free bits threshold for comparison

    Returns:
        CollapseMetrics dataclass
    """
    metrics_df = data.get("metrics")
    au_df = data.get("au_history")
    partition_df = data.get("partition_stats")

    result = CollapseMetrics()

    # Determine epoch
    if epoch is None and metrics_df is not None and not metrics_df.empty:
        epoch = int(metrics_df["epoch"].max())

    result.epoch = epoch if epoch is not None else -1

    # Active units from au_history.csv
    if au_df is not None and not au_df.empty:
        if epoch is not None:
            au_epoch_data = au_df[au_df["epoch"] == epoch]
            if au_epoch_data.empty:
                # Use closest
                available = au_df["epoch"].unique()
                closest = min(available, key=lambda x: abs(x - epoch))
                au_epoch_data = au_df[au_df["epoch"] == closest]
        else:
            au_epoch_data = au_df.iloc[[-1]]  # Last row

        if not au_epoch_data.empty:
            row = au_epoch_data.iloc[0]
            if "au_count" in row.index:
                result.au_count_total = int(row["au_count"])
            if "au_frac" in row.index:
                result.au_frac_total = float(row["au_frac"])

    # Get residual AU from partition_stats or metrics
    if partition_df is not None and not partition_df.empty:
        res_data = partition_df[partition_df["partition"] == "z_residual"]
        if not res_data.empty and epoch is not None:
            res_epoch = res_data[res_data["epoch"] == epoch]
            if res_epoch.empty:
                available = res_data["epoch"].unique()
                if len(available) > 0:
                    closest = min(available, key=lambda x: abs(x - epoch))
                    res_epoch = res_data[res_data["epoch"] == closest]

            if not res_epoch.empty:
                row = res_epoch.iloc[0]
                if "au_count" in row.index:
                    result.au_count_residual = int(row["au_count"])
                if "au_frac" in row.index:
                    result.au_frac_residual = float(row["au_frac"])
                if "mu_var_mean" in row.index:
                    result.z_residual_var = float(row["mu_var_mean"])

    # Get per-partition variance from partition_stats
    if partition_df is not None and not partition_df.empty:
        for part_name, attr_name in [
            ("z_vol", "z_vol_var"),
            ("z_loc", "z_loc_var"),
            ("z_shape", "z_shape_var"),
        ]:
            part_data = partition_df[partition_df["partition"] == part_name]
            if not part_data.empty and epoch is not None:
                part_epoch = part_data[part_data["epoch"] == epoch]
                if part_epoch.empty:
                    available = part_data["epoch"].unique()
                    if len(available) > 0:
                        closest = min(available, key=lambda x: abs(x - epoch))
                        part_epoch = part_data[part_data["epoch"] == closest]

                if not part_epoch.empty:
                    row = part_epoch.iloc[0]
                    if "mu_var_mean" in row.index:
                        setattr(result, attr_name, float(row["mu_var_mean"]))

    # Decoder bypass test from metrics
    if metrics_df is not None and not metrics_df.empty:
        epoch_data = metrics_df[metrics_df["epoch"] == epoch] if epoch else metrics_df.iloc[[-1]]
        if not epoch_data.empty:
            row = epoch_data.iloc[0]

            if "diag/recon_mu_mse" in row.index and pd.notna(row["diag/recon_mu_mse"]):
                result.recon_mu_mse = float(row["diag/recon_mu_mse"])
            if "diag/recon_z0_mse" in row.index and pd.notna(row["diag/recon_z0_mse"]):
                result.recon_z0_mse = float(row["diag/recon_z0_mse"])

            # Compute bypass ratio
            if result.recon_mu_mse > 1e-8:
                gap = result.recon_z0_mse - result.recon_mu_mse
                result.decoder_bypass_ratio = gap / result.recon_mu_mse
                result.decoder_bypassed = result.decoder_bypass_ratio < DECODER_BYPASS_THRESHOLD

            # KL per dimension
            if "val_epoch/kl_raw" in row.index and pd.notna(row["val_epoch/kl_raw"]):
                kl_raw = float(row["val_epoch/kl_raw"])
                # Get residual dim from config
                config = data.get("config", {})
                z_res_dim = config.get("model", {}).get("latent_partitioning", {}).get(
                    "z_residual", {}
                ).get("dim", 80)
                if z_res_dim > 0:
                    result.kl_raw_per_dim = kl_raw / z_res_dim
                    result.kl_below_floor = result.kl_raw_per_dim < kl_floor

    # Set collapse flags
    result.residual_collapsed = result.au_frac_residual < AU_RESIDUAL_COLLAPSE_THRESHOLD

    return result


def compute_collapse_history(
    data: Dict[str, Any],
    epochs: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute collapse metrics over training history.

    Args:
        data: Dictionary from load_experiment_data()
        epochs: Specific epochs (None = all available)

    Returns:
        DataFrame with collapse metrics per epoch
    """
    au_df = data.get("au_history")
    if au_df is None or au_df.empty:
        return pd.DataFrame()

    all_epochs = sorted(au_df["epoch"].unique())
    if epochs is not None:
        all_epochs = [e for e in all_epochs if e in epochs]

    records = []
    for epoch in all_epochs:
        metrics = compute_collapse_metrics(data, epoch=epoch)
        records.append({
            "epoch": metrics.epoch,
            "au_count_total": metrics.au_count_total,
            "au_frac_total": metrics.au_frac_total,
            "au_count_residual": metrics.au_count_residual,
            "au_frac_residual": metrics.au_frac_residual,
            "z_vol_var": metrics.z_vol_var,
            "z_loc_var": metrics.z_loc_var,
            "z_shape_var": metrics.z_shape_var,
            "z_residual_var": metrics.z_residual_var,
            "decoder_bypass_ratio": metrics.decoder_bypass_ratio,
            "decoder_bypassed": metrics.decoder_bypassed,
            "residual_collapsed": metrics.residual_collapsed,
        })

    return pd.DataFrame(records)


def analyze_variance_trajectory(
    data: Dict[str, Any],
    partition: str = "z_residual",
    deflation_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Analyze variance trajectory for a partition.

    Detects patterns like variance "deflation" where variance peaks
    then declines significantly, indicating the partition is being
    underutilized in late training.

    Args:
        data: Dictionary from load_experiment_data()
        partition: Partition to analyze (e.g., "z_residual")
        deflation_threshold: Decline fraction to flag as deflating

    Returns:
        Dictionary with trajectory analysis:
        - var_peak: Maximum variance reached
        - var_peak_epoch: Epoch of peak variance
        - var_final: Final variance value
        - var_decline_pct: (peak - final) / peak
        - deflating: Whether decline exceeds threshold
        - trajectory: List of (epoch, variance) tuples
    """
    partition_df = data.get("partition_stats")

    result = {
        "var_peak": 0.0,
        "var_peak_epoch": 0,
        "var_final": 0.0,
        "var_decline_pct": 0.0,
        "deflating": False,
        "trajectory": [],
    }

    if partition_df is None or partition_df.empty:
        return result

    # Filter to partition
    part_data = partition_df[partition_df["partition"] == partition]
    if part_data.empty:
        return result

    # Get variance column
    var_col = "mu_var_mean" if "mu_var_mean" in part_data.columns else None
    if var_col is None:
        return result

    # Sort by epoch
    part_data = part_data.sort_values("epoch")

    # Build trajectory
    trajectory = []
    for _, row in part_data.iterrows():
        if pd.notna(row[var_col]):
            trajectory.append((int(row["epoch"]), float(row[var_col])))

    if not trajectory:
        return result

    result["trajectory"] = trajectory

    # Find peak
    epochs, variances = zip(*trajectory)
    peak_idx = np.argmax(variances)
    result["var_peak"] = variances[peak_idx]
    result["var_peak_epoch"] = epochs[peak_idx]
    result["var_final"] = variances[-1]

    # Compute decline
    if result["var_peak"] > 1e-8:
        decline = (result["var_peak"] - result["var_final"]) / result["var_peak"]
        result["var_decline_pct"] = max(0.0, decline)
        result["deflating"] = result["var_decline_pct"] > deflation_threshold

    return result


def compute_collapse_metrics_with_trajectory(
    data: Dict[str, Any],
    epoch: Optional[int] = None,
    kl_floor: float = KL_FLOOR_DEFAULT,
) -> CollapseMetrics:
    """Compute collapse metrics including variance trajectory analysis.

    This is an enhanced version of compute_collapse_metrics that adds
    variance trajectory analysis for z_residual.

    Args:
        data: Dictionary from load_experiment_data()
        epoch: Specific epoch (None = use final)
        kl_floor: KL free bits threshold for comparison

    Returns:
        CollapseMetrics with trajectory fields populated
    """
    # Get base metrics
    result = compute_collapse_metrics(data, epoch=epoch, kl_floor=kl_floor)

    # Add trajectory analysis for z_residual
    traj = analyze_variance_trajectory(data, partition="z_residual")
    result.residual_var_peak = traj["var_peak"]
    result.residual_var_peak_epoch = traj["var_peak_epoch"]
    result.residual_var_decline_pct = traj["var_decline_pct"]
    result.residual_deflating = traj["deflating"]

    return result
