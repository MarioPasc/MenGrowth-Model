"""CLI orchestrator for TSI Explainability Analysis.

Computes Tumor Selectivity Index across SwinViT encoder stages, comparing
frozen BrainSegFounder against LoRA-adapted models for each rank ablation.

Usage:
    python -m experiments.uncertainty_segmentation.explainability.run_tsi \\
        --config experiments/uncertainty_segmentation/config.yaml \\
        --tsi-config experiments/uncertainty_segmentation/explainability/config.yaml \\
        --device cuda:0

    # Single rank (r4 only):
    python -m experiments.uncertainty_segmentation.explainability.run_tsi \\
        --config experiments/uncertainty_segmentation/config.yaml \\
        --tsi-config experiments/uncertainty_segmentation/explainability/config.yaml \\
        --rank 4

    # Validation only (1 scan, shape checks):
    python -m experiments.uncertainty_segmentation.explainability.run_tsi \\
        --config experiments/uncertainty_segmentation/config.yaml \\
        --tsi-config experiments/uncertainty_segmentation/explainability/config.yaml \\
        --validate-only
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from growth.data.bratsmendata import BraTSDatasetH5
from growth.data.transforms import get_h5_val_transforms

from .model_loader import (
    get_checkpoint_path,
    get_run_dir,
    load_adapted_model,
    load_frozen_model,
)
from .tsi_analysis import (
    STAGE_META,
    ScanTSIResult,
    compute_tsi_single_scan,
    extract_hidden_states,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(output_dir: Path) -> None:
    """Configure logging to both console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tsi_analysis.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s — %(message)s")
    )
    root.addHandler(fh)


def _load_test_dataset(
    config: DictConfig,
    roi_size: tuple[int, int, int] = (128, 128, 128),
) -> tuple[BraTSDatasetH5, list[str], np.ndarray]:
    """Load the BraTS-MEN test dataset and resolve scan IDs.

    Args:
        config: Parent config with paths and data sections.
        roi_size: Spatial ROI for transforms (128³ fits 8GB GPU).

    Returns:
        Tuple of (dataset, all_scan_ids, test_indices).
    """
    h5_path = config.paths.men_h5_file
    transform = get_h5_val_transforms(roi_size=roi_size)
    dataset = BraTSDatasetH5(
        h5_path=h5_path,
        split=config.data.test_split,
        transform=transform,
        compute_semantic=False,
    )

    # Resolve scan IDs from H5 metadata (BUG-2 fix pattern)
    with h5py.File(h5_path, "r") as f:
        all_scan_ids = [
            s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]
        ]
    splits = BraTSDatasetH5.load_splits_from_h5(h5_path)
    test_indices = splits.get(
        config.data.test_split, np.arange(len(all_scan_ids))
    )

    return dataset, all_scan_ids, test_indices


def _get_wt_mask(seg: torch.Tensor) -> torch.Tensor:
    """Convert integer segmentation to binary WT mask.

    MEN domain: WT = (seg == 1) | (seg == 2) | (seg == 3) = (seg > 0).

    Args:
        seg: Integer labels [1, D, H, W].

    Returns:
        Binary mask [D, H, W].
    """
    return (seg.squeeze(0) > 0).float()


def _select_scan_indices(
    n_available: int,
    n_scans: int,
    selection: str,
    seed: int,
) -> list[int]:
    """Select which test scans to use for TSI analysis.

    Args:
        n_available: Total number of test scans available.
        n_scans: Desired number of scans.
        selection: "first" or "random".
        seed: Random seed for reproducible selection.

    Returns:
        List of dataset indices to use.
    """
    n_use = min(n_scans, n_available)
    if selection == "first":
        return list(range(n_use))
    elif selection == "random":
        rng = np.random.RandomState(seed)
        return sorted(rng.choice(n_available, size=n_use, replace=False).tolist())
    else:
        raise ValueError(f"Unknown scan_selection: {selection}")


def _results_to_dataframe(results: list[ScanTSIResult]) -> pd.DataFrame:
    """Convert a list of ScanTSIResult to a per-scan, per-stage DataFrame.

    Args:
        results: List of ScanTSIResult.

    Returns:
        DataFrame with columns: scan_id, condition, stage, n_channels,
        resolution, mean_tsi, std_tsi, frac_1.5, frac_2.0, wilcoxon_p.
    """
    rows = []
    for scan_res in results:
        for sr in scan_res.stages:
            row = {
                "scan_id": scan_res.scan_id,
                "condition": scan_res.condition,
                "stage": sr.stage,
                "n_channels": sr.n_channels,
                "resolution": f"{sr.resolution[0]}^3",
                "mean_tsi": sr.mean_tsi,
                "std_tsi": sr.std_tsi,
                "wilcoxon_p": sr.wilcoxon_p,
            }
            for tau, frac in sr.frac_above.items():
                row[f"frac_{tau}"] = frac
            rows.append(row)
    return pd.DataFrame(rows)


def _save_channel_data(
    results: list[ScanTSIResult],
    output_path: Path,
) -> None:
    """Save per-channel TSI arrays for all scans/stages to NPZ.

    Args:
        results: List of ScanTSIResult.
        output_path: Path to .npz file.
    """
    arrays = {}
    for scan_res in results:
        for sr in scan_res.stages:
            key = f"{scan_res.scan_id}_stage{sr.stage}"
            arrays[key] = sr.tsi_per_channel
    np.savez_compressed(output_path, **arrays)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def run_validation(
    config: DictConfig,
    tsi_config: DictConfig,
    rank: int,
    device: str,
) -> bool:
    """Run validation checks on 1 scan before the full analysis.

    Checks:
    1. Hidden state shapes match expected values.
    2. Stages 0-2 are numerically identical between frozen and adapted.
    3. Stages 3-4 differ between frozen and adapted.
    4. TSI values are computed without errors.

    Args:
        config: Parent config.
        tsi_config: TSI config.
        rank: LoRA rank to validate.
        device: Compute device.

    Returns:
        True if all checks pass.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION RUN (1 scan)")
    logger.info("=" * 60)

    roi_size = tuple(tsi_config.analysis.roi_size)
    logger.info(f"ROI size: {roi_size}")
    dataset, all_scan_ids, test_indices = _load_test_dataset(config, roi_size=roi_size)
    sample = dataset[0]
    images = sample["image"].unsqueeze(0).to(device)
    gt_mask = _get_wt_mask(sample["seg"])
    sid = all_scan_ids[test_indices[0]]

    logger.info(f"Validation scan: {sid}")
    logger.info(f"Input shape: {images.shape}")
    logger.info(f"GT mask shape: {gt_mask.shape}, tumor voxels: {gt_mask.sum().item():.0f}")

    # Load models
    frozen_model = load_frozen_model(config, device=device)
    adapted_model = load_adapted_model(
        config, tsi_config, rank=rank,
        member_id=tsi_config.analysis.member_id, device=device,
    )

    # Extract hidden states
    with torch.amp.autocast("cuda", enabled=(device != "cpu")):
        hs_frozen = extract_hidden_states(frozen_model, images)
        hs_adapted = extract_hidden_states(adapted_model, images)

    # Free GPU models
    del frozen_model, adapted_model
    torch.cuda.empty_cache() if device != "cpu" else None

    all_ok = True

    # Check 1: shapes
    logger.info("\n--- Check 1: Hidden state shapes ---")
    for s in range(5):
        meta = STAGE_META[s]
        expected_c = meta["channels"]
        expected_spatial = images.shape[2] // meta["downsample"]
        f_shape = hs_frozen[s].shape
        a_shape = hs_adapted[s].shape
        ok = (
            f_shape[1] == expected_c
            and f_shape[2] == expected_spatial
            and f_shape == a_shape
        )
        status = "OK" if ok else "FAIL"
        logger.info(f"  Stage {s}: frozen={list(f_shape)}, adapted={list(a_shape)} [{status}]")
        if not ok:
            all_ok = False

    # Check 2: stages 0-2 identical
    logger.info("\n--- Check 2: Stages 0-2 should be identical ---")
    for s in range(3):
        max_diff = (hs_frozen[s] - hs_adapted[s]).abs().max().item()
        ok = max_diff < 1e-5
        status = "OK" if ok else "FAIL"
        logger.info(f"  Stage {s}: max_diff={max_diff:.2e} [{status}]")
        if not ok:
            all_ok = False

    # Check 3: stages 3-4 differ
    logger.info("\n--- Check 3: Stages 3-4 should differ (LoRA applied) ---")
    for s in range(3, 5):
        max_diff = (hs_frozen[s] - hs_adapted[s]).abs().max().item()
        ok = max_diff > 1e-3
        status = "OK" if ok else "FAIL"
        logger.info(f"  Stage {s}: max_diff={max_diff:.2e} [{status}]")
        if not ok:
            all_ok = False

    # Check 4: TSI computation
    logger.info("\n--- Check 4: TSI computation ---")
    thresholds = list(tsi_config.analysis.tsi_thresholds)
    for label, hs in [("frozen", hs_frozen), ("adapted", hs_adapted)]:
        result = compute_tsi_single_scan(
            hs, gt_mask, sid, label,
            thresholds=thresholds,
            top_k=tsi_config.analysis.top_k,
        )
        for sr in result.stages:
            logger.info(
                f"  {label} stage {sr.stage}: "
                f"mean_TSI={sr.mean_tsi:.3f} ± {sr.std_tsi:.3f}, "
                f"Frac(>1.5)={sr.frac_above.get(1.5, 0):.1%}, "
                f"p={sr.wilcoxon_p:.2e}"
            )

    del hs_frozen, hs_adapted

    if all_ok:
        logger.info("\n[OK] All validation checks passed.")
    else:
        logger.error("\n[FAIL] Some validation checks failed — review output above.")

    return all_ok


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def run_condition(
    model: torch.nn.Module,
    dataset: BraTSDatasetH5,
    all_scan_ids: list[str],
    test_indices: np.ndarray,
    scan_indices: list[int],
    condition: str,
    tsi_config: DictConfig,
    device: str,
) -> list[ScanTSIResult]:
    """Run TSI analysis for one condition (frozen or adapted) across N scans.

    Args:
        model: Loaded model (frozen or adapted).
        dataset: BraTS-MEN test dataset.
        all_scan_ids: All scan ID strings from H5 metadata.
        test_indices: Indices into H5 for the test split.
        scan_indices: Which dataset indices to process.
        condition: "frozen" or "adapted".
        tsi_config: TSI analysis configuration.
        device: Compute device.

    Returns:
        List of ScanTSIResult, one per scan.
    """
    thresholds = list(tsi_config.analysis.tsi_thresholds)
    top_k = tsi_config.analysis.top_k
    epsilon = tsi_config.analysis.epsilon
    results = []

    for idx_i, ds_idx in enumerate(scan_indices):
        sample = dataset[ds_idx]
        images = sample["image"].unsqueeze(0).to(device)
        gt_mask = _get_wt_mask(sample["seg"])
        sid = all_scan_ids[test_indices[ds_idx]]

        t0 = time.time()

        # Extract hidden states
        with torch.amp.autocast("cuda", enabled=(device != "cpu")):
            hidden_states = extract_hidden_states(model, images)

        # Free GPU input
        del images
        if device != "cpu":
            torch.cuda.empty_cache()

        # Compute TSI on CPU
        scan_result = compute_tsi_single_scan(
            hidden_states,
            gt_mask,
            scan_id=sid,
            condition=condition,
            thresholds=thresholds,
            top_k=top_k,
            return_maps=(idx_i == 0),  # Only store maps for first scan
            epsilon=epsilon,
        )
        results.append(scan_result)

        elapsed = time.time() - t0
        logger.info(
            f"  [{condition}] scan {idx_i + 1}/{len(scan_indices)} "
            f"({sid}): {elapsed:.1f}s — "
            f"mean_TSI=[{', '.join(f'{sr.mean_tsi:.2f}' for sr in scan_result.stages)}]"
        )

        # Free hidden states
        del hidden_states

    return results


def run_full_analysis(
    config: DictConfig,
    tsi_config: DictConfig,
    ranks: list[int],
    device: str,
    skip_frozen: bool = False,
) -> None:
    """Run the complete TSI analysis for all requested ranks.

    Args:
        config: Parent uncertainty_segmentation config.
        tsi_config: TSI analysis config.
        ranks: List of LoRA ranks to analyze.
        device: Compute device.
        skip_frozen: If True, load cached frozen results instead of recomputing.
    """
    output_dir = Path(tsi_config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config snapshot
    OmegaConf.save(tsi_config, output_dir / "config_snapshot.yaml", resolve=True)

    # Load dataset
    roi_size = tuple(tsi_config.analysis.roi_size)
    logger.info(f"ROI size: {roi_size}")
    dataset, all_scan_ids, test_indices = _load_test_dataset(config, roi_size=roi_size)
    n_available = len(dataset)
    scan_indices = _select_scan_indices(
        n_available,
        tsi_config.analysis.n_scans,
        tsi_config.analysis.scan_selection,
        tsi_config.analysis.seed,
    )
    n_scans = len(scan_indices)
    logger.info(f"Using {n_scans}/{n_available} test scans for TSI analysis")

    # =====================================================================
    # FROZEN condition (run once, shared across ranks)
    # =====================================================================
    frozen_data_dir = output_dir / "frozen" / "data"
    frozen_csv_path = frozen_data_dir / "tsi_frozen_per_scan.csv"
    frozen_npz_path = frozen_data_dir / "tsi_frozen_channels.npz"

    frozen_results: list[ScanTSIResult] | None = None

    if skip_frozen and frozen_csv_path.exists():
        logger.info(f"Skipping frozen condition (cached at {frozen_csv_path})")
    else:
        logger.info("=" * 60)
        logger.info("FROZEN CONDITION")
        logger.info("=" * 60)

        frozen_model = load_frozen_model(config, device=device)
        frozen_results = run_condition(
            frozen_model, dataset, all_scan_ids, test_indices,
            scan_indices, "frozen", tsi_config, device,
        )
        del frozen_model
        if device != "cpu":
            torch.cuda.empty_cache()

        # Save frozen results
        frozen_data_dir.mkdir(parents=True, exist_ok=True)
        frozen_df = _results_to_dataframe(frozen_results)
        frozen_df.to_csv(frozen_csv_path, index=False)
        _save_channel_data(frozen_results, frozen_npz_path)
        logger.info(f"Saved frozen results to {frozen_data_dir}")

    # =====================================================================
    # ADAPTED conditions (one per rank)
    # =====================================================================
    all_rank_results: dict[int, list[ScanTSIResult]] = {}

    for rank in ranks:
        run_dir = get_run_dir(config, tsi_config, rank)
        if not run_dir.exists():
            logger.warning(f"Run directory not found for rank={rank}: {run_dir} — skipping")
            continue

        logger.info("=" * 60)
        logger.info(f"ADAPTED CONDITION: rank={rank}")
        logger.info("=" * 60)

        adapted_model = load_adapted_model(
            config, tsi_config, rank=rank,
            member_id=tsi_config.analysis.member_id, device=device,
        )
        adapted_results = run_condition(
            adapted_model, dataset, all_scan_ids, test_indices,
            scan_indices, f"adapted_r{rank}", tsi_config, device,
        )
        del adapted_model
        if device != "cpu":
            torch.cuda.empty_cache()

        all_rank_results[rank] = adapted_results

        # Save per-rank results
        rank_data_dir = output_dir / f"r{rank}" / "data"
        rank_data_dir.mkdir(parents=True, exist_ok=True)
        adapted_df = _results_to_dataframe(adapted_results)
        adapted_df.to_csv(rank_data_dir / "tsi_adapted_per_scan.csv", index=False)
        _save_channel_data(adapted_results, rank_data_dir / "tsi_adapted_channels.npz")
        logger.info(f"Saved adapted results (r={rank}) to {rank_data_dir}")

    # =====================================================================
    # TABLE + FIGURE generation
    # =====================================================================
    # Reload frozen results from CSV if we skipped recomputation
    if frozen_results is None:
        logger.info("Loading cached frozen results for table/figure generation")
        frozen_df = pd.read_csv(frozen_csv_path)
    else:
        frozen_df = _results_to_dataframe(frozen_results)

    from .figure_tsi import generate_all_figures
    from .table_tsi import generate_all_tables

    # Generate tables
    for rank, adapted_results in all_rank_results.items():
        adapted_df = _results_to_dataframe(adapted_results)
        rank_dir = output_dir / f"r{rank}"

        generate_all_tables(
            frozen_df=frozen_df,
            adapted_df=adapted_df,
            rank=rank,
            output_dir=rank_dir,
        )

    # Generate cross-rank table
    if len(all_rank_results) > 1:
        generate_all_tables(
            frozen_df=frozen_df,
            adapted_df=None,
            rank=None,
            output_dir=output_dir / "cross_rank",
            all_adapted={r: _results_to_dataframe(res) for r, res in all_rank_results.items()},
        )

    # Generate figures
    for rank, adapted_results in all_rank_results.items():
        rank_dir = output_dir / f"r{rank}"

        generate_all_figures(
            frozen_results=frozen_results,
            adapted_results=adapted_results,
            frozen_df=frozen_df,
            adapted_df=_results_to_dataframe(adapted_results),
            dataset=dataset,
            scan_indices=scan_indices,
            all_scan_ids=all_scan_ids,
            test_indices=test_indices,
            rank=rank,
            config=tsi_config,
            output_dir=rank_dir,
        )

    # Cross-rank comparison figure
    if len(all_rank_results) > 1:
        from .figure_tsi import generate_cross_rank_figure

        all_adapted_dfs = {
            r: _results_to_dataframe(res)
            for r, res in all_rank_results.items()
        }
        generate_cross_rank_figure(
            frozen_df=frozen_df,
            all_adapted_dfs=all_adapted_dfs,
            config=tsi_config,
            output_dir=output_dir / "cross_rank",
        )

    # Save statistical summary JSON for quick inspection
    _save_summary(frozen_df, all_rank_results, output_dir)

    logger.info("=" * 60)
    logger.info("TSI ANALYSIS COMPLETE")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


def _save_summary(
    frozen_df: pd.DataFrame,
    all_rank_results: dict[int, list[ScanTSIResult]],
    output_dir: Path,
) -> None:
    """Save a JSON summary of key findings for quick inspection.

    Args:
        frozen_df: Frozen results DataFrame.
        all_rank_results: Dict of rank -> adapted results.
        output_dir: Output directory.
    """
    import json

    summary: dict = {"frozen": {}, "adapted": {}}
    for stage in range(5):
        vals = frozen_df[frozen_df["stage"] == stage]["mean_tsi"].dropna()
        summary["frozen"][f"stage{stage}"] = {
            "mean_tsi": round(float(vals.mean()), 4),
            "n_valid": int(len(vals)),
        }

    for rank, results in all_rank_results.items():
        adf = _results_to_dataframe(results)
        summary["adapted"][f"r{rank}"] = {}
        for stage in range(5):
            vals = adf[adf["stage"] == stage]["mean_tsi"].dropna()
            summary["adapted"][f"r{rank}"][f"stage{stage}"] = {
                "mean_tsi": round(float(vals.mean()), 4) if len(vals) > 0 else None,
                "n_valid": int(len(vals)),
            }

    path = output_dir / "tsi_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {path}")


# ---------------------------------------------------------------------------
# Figure regeneration (offline, no GPU)
# ---------------------------------------------------------------------------


def regenerate_figures(
    tsi_config: DictConfig,
    ranks: list[int],
) -> None:
    """Regenerate all figures from saved data without model inference.

    Requires that the full analysis has been run at least once so that
    the per-scan CSVs, per-channel NPZs, and viz_*.npz files exist.

    Args:
        tsi_config: TSI config.
        ranks: Ranks to regenerate figures for.
    """
    from .figure_tsi import (
        generate_cross_rank_figure,
        generate_delta_figure,
        generate_panel_figure_from_saved,
        load_viz_data,
    )
    from .table_tsi import generate_all_tables

    output_dir = Path(tsi_config.paths.output_dir)
    save_fmt = tsi_config.figure.save_format
    save_dpi = tsi_config.figure.save_dpi

    # Load frozen data
    frozen_csv = output_dir / "frozen" / "data" / "tsi_frozen_per_scan.csv"
    if not frozen_csv.exists():
        logger.error(f"Frozen data not found: {frozen_csv}")
        return
    frozen_df = pd.read_csv(frozen_csv)

    logger.info("=" * 60)
    logger.info("REGENERATING FIGURES FROM SAVED DATA")
    logger.info("=" * 60)

    all_adapted_dfs: dict[int, pd.DataFrame] = {}

    for rank in ranks:
        rank_dir = output_dir / f"r{rank}"
        adapted_csv = rank_dir / "data" / "tsi_adapted_per_scan.csv"
        if not adapted_csv.exists():
            logger.warning(f"Adapted data not found for r={rank}: {adapted_csv} — skipping")
            continue

        adapted_df = pd.read_csv(adapted_csv)
        all_adapted_dfs[rank] = adapted_df

        figures_dir = rank_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Panel figures from saved viz data
        viz_frozen_path = rank_dir / "data" / "viz_frozen.npz"
        viz_adapted_path = rank_dir / "data" / f"viz_adapted_r{rank}.npz"

        if viz_frozen_path.exists():
            viz_data = load_viz_data(viz_frozen_path)
            fig = generate_panel_figure_from_saved(viz_data, tsi_config)
            path = figures_dir / f"tsi_frozen.{save_fmt}"
            fig.savefig(path, dpi=save_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Regenerated: {path}")

        if viz_adapted_path.exists():
            viz_data = load_viz_data(viz_adapted_path)
            fig = generate_panel_figure_from_saved(
                viz_data, tsi_config, title_suffix=f" (r={rank})",
            )
            path = figures_dir / f"tsi_adapted_r{rank}.{save_fmt}"
            fig.savefig(path, dpi=save_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Regenerated: {path}")

        # Delta figure (from CSVs — always available)
        fig_delta = generate_delta_figure(frozen_df, adapted_df, tsi_config, rank)
        path = figures_dir / f"tsi_delta_r{rank}.{save_fmt}"
        fig_delta.savefig(path, dpi=save_dpi, bbox_inches="tight")
        plt.close(fig_delta)
        logger.info(f"Regenerated: {path}")

        # Re-generate tables too
        generate_all_tables(
            frozen_df=frozen_df,
            adapted_df=adapted_df,
            rank=rank,
            output_dir=rank_dir,
        )

    # Cross-rank figures
    if len(all_adapted_dfs) > 1:
        generate_cross_rank_figure(
            frozen_df=frozen_df,
            all_adapted_dfs=all_adapted_dfs,
            config=tsi_config,
            output_dir=output_dir / "cross_rank",
        )
        generate_all_tables(
            frozen_df=frozen_df,
            adapted_df=None,
            rank=None,
            output_dir=output_dir / "cross_rank",
            all_adapted=all_adapted_dfs,
        )

    logger.info("Figure regeneration complete.")


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def preflight(
    config: DictConfig,
    tsi_config: DictConfig,
    ranks: list[int],
) -> bool:
    """Verify all required files exist before starting analysis.

    Args:
        config: Parent config.
        tsi_config: TSI config.
        ranks: Ranks to check.

    Returns:
        True if all files exist.
    """
    ok = True

    # Checkpoint
    ckpt = get_checkpoint_path(config)
    if not ckpt.exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        ok = False
    else:
        logger.info(f"[OK] Checkpoint: {ckpt}")

    # H5 file
    h5 = Path(config.paths.men_h5_file)
    if not h5.exists():
        logger.error(f"H5 file not found: {h5}")
        ok = False
    else:
        logger.info(f"[OK] H5 file: {h5}")

    # Run directories
    member_id = tsi_config.analysis.member_id
    for rank in ranks:
        run_dir = get_run_dir(config, tsi_config, rank)
        adapter_dir = run_dir / "adapters" / f"member_{member_id}" / "adapter"
        decoder_path = run_dir / "adapters" / f"member_{member_id}" / "decoder.pt"

        if not run_dir.exists():
            logger.warning(f"[SKIP] Run dir not found: {run_dir}")
        elif not adapter_dir.exists():
            logger.error(f"Adapter not found: {adapter_dir}")
            ok = False
        elif not decoder_path.exists():
            logger.error(f"Decoder not found: {decoder_path}")
            ok = False
        else:
            logger.info(f"[OK] rank={rank}: {run_dir}")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for TSI analysis."""
    parser = argparse.ArgumentParser(
        description="TSI Explainability Analysis for LoRA-Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to parent uncertainty_segmentation config.yaml",
    )
    parser.add_argument(
        "--tsi-config", default=None,
        help="Path to TSI-specific config.yaml (default: explainability/config.yaml)",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Compute device (default: cuda:0)",
    )
    parser.add_argument(
        "--rank", type=int, nargs="+", default=None,
        help="Which LoRA ranks to analyze (default: all from tsi-config)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Run validation checks on 1 scan only, then exit",
    )
    parser.add_argument(
        "--skip-frozen", action="store_true",
        help="Skip frozen condition (use cached results)",
    )
    parser.add_argument(
        "--regenerate-figures", action="store_true",
        help="Regenerate figures from saved data (no model inference)",
    )
    args = parser.parse_args()

    # Load configs
    config = OmegaConf.load(args.config)

    tsi_config_path = args.tsi_config
    if tsi_config_path is None:
        tsi_config_path = (
            Path(args.config).parent / "explainability" / "config.yaml"
        )
    tsi_config = OmegaConf.load(tsi_config_path)

    # Resolve ranks
    ranks = args.rank or list(tsi_config.ranks)

    # Setup logging
    output_dir = Path(tsi_config.paths.output_dir)
    _setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("TSI EXPLAINABILITY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Parent config: {args.config}")
    logger.info(f"TSI config: {tsi_config_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Ranks: {ranks}")
    logger.info(f"Output: {output_dir}")

    # Pre-flight
    if not preflight(config, tsi_config, ranks):
        logger.error("Pre-flight checks failed — aborting")
        return

    # Validation mode
    if args.validate_only:
        validation_rank = ranks[0]
        run_validation(config, tsi_config, validation_rank, args.device)
        return

    # Regenerate figures mode (no GPU needed)
    if args.regenerate_figures:
        regenerate_figures(tsi_config, ranks)
        return

    # Full analysis
    t_start = time.time()
    run_full_analysis(
        config, tsi_config, ranks, args.device,
        skip_frozen=args.skip_frozen,
    )
    elapsed = time.time() - t_start
    logger.info(f"Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
