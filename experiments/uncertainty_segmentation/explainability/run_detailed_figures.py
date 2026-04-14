"""Detailed per-metric figure generation (visual example + aggregate stats).

Produces three figures, one per explainability metric, each combining a
GPU-rendered visual example of a single representative scan (row 1) with
aggregate statistics computed from the cached raw CSVs/NPZs (rows 2-3):

- ``tsi_detailed_{condition}.{fmt}``  — 3 × 5  (stages 0-4)
- ``asi_detailed_{condition}.{fmt}``  — 3 × 4  (stages 1-4)
- ``dad_detailed_{condition}.{fmt}``  — 3 × 4  (stages 1-4)

The visual rows require one extra GPU forward pass per metric, but no
backprop and no LoRA loading when ``--condition frozen``. The aggregate
rows read the existing artefacts that ``run_analysis.py`` writes to
``{data_dir}/raw/``, so the figures stay consistent with the production
metrics.

Usage
-----
::

    python -m experiments.uncertainty_segmentation.explainability.run_detailed_figures \\
        --config experiments/uncertainty_segmentation/config.yaml \\
        --analysis-config experiments/uncertainty_segmentation/explainability/config.yaml \\
        --data-dir /path/to/explainability \\
        --output /path/to/explainability \\
        --condition frozen \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from .engine.asi import (
    downsample_mask_to_stage,
    partition_mask_windows,
    select_boundary_windows,
)
from .engine.brain_mask import derive_brain_mask
from .engine.dad import DADScanAccumulator
from .engine.data_loader import (
    build_gli_loader,
    build_men_loader,
    get_wt_mask,
    select_scan_indices,
)
from .engine.hooks import AttentionCapture, CapturedAttention
from .engine.model_loader import load_adapted_model, load_frozen_model
from .engine.tsi import compute_tsi_single_scan, extract_hidden_states
from .figures.fig_detailed import (
    render_asi_detailed,
    render_dad_detailed,
    render_tsi_detailed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------


def _pick_slice(wt_mask: np.ndarray, mode: str = "max_tumor") -> int:
    """Choose an axial slice for the visualisation row.

    ``"max_tumor"`` returns the slice with the most WT voxels; falls back
    to the volume centre when the mask is empty.
    """
    if mode == "centre":
        return wt_mask.shape[0] // 2
    if mode != "max_tumor":
        raise ValueError(f"Unknown slice_selection: {mode}")
    counts = wt_mask.reshape(wt_mask.shape[0], -1).sum(axis=1)
    if counts.max() == 0:
        return wt_mask.shape[0] // 2
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# ASI visual: pick one boundary window per stage with tumor-first ordering
# ---------------------------------------------------------------------------


def _build_asi_visual(
    captured: dict[str, CapturedAttention],
    gt_mask: torch.Tensor,
    target_stages: tuple[int, ...] = (1, 2, 3, 4),
    min_tumor: int = 5,
    min_nontumor: int = 5,
    prefer_block: int = 0,
) -> dict[int, dict]:
    """Pluck one boundary window per stage, reorder tokens tumor-first.

    The renderer plots ``attn[H, N, N]`` with the first ``n_tumor`` rows
    and columns being the tumor tokens, so the boundary line at
    ``n_tumor`` separates tumor-from-background sub-blocks.

    Parameters
    ----------
    captured : dict[str, CapturedAttention]
        ``AttentionCapture(mode="store").get_attention_maps()`` output.
    gt_mask : torch.Tensor
        Whole-tumor binary mask ``[D, H, W]``.
    target_stages : tuple[int, ...]
        Stages to visualise.
    min_tumor, min_nontumor : int
        Boundary-window thresholds.
    prefer_block : int
        Try the non-shifted block first; falls back to the other block
        if no boundary windows exist there.

    Returns
    -------
    dict[int, dict]
        ``stage -> {attn_matrix [H,N,N], n_tumor, block}``. Stages with
        no boundary windows are absent (renderer plots a placeholder).
    """
    visual: dict[int, dict] = {}
    for stage in target_stages:
        chosen: dict | None = None
        block_order = [prefer_block, 1 - prefer_block]
        for block in block_order:
            key = f"stage_{stage}_block_{block}"
            cap = captured.get(key)
            if cap is None:
                continue
            attn = cap.attn_weights  # [n_windows*B, H, N, N]
            n_tokens = attn.shape[-1]
            wd = round(n_tokens ** (1.0 / 3.0))
            if wd ** 3 != n_tokens:
                logger.warning("ASI visual: non-cubic window at %s", key)
                continue
            window_size = (wd, wd, wd)

            ds_mask = downsample_mask_to_stage(gt_mask, stage)
            d, h, w = ds_mask.shape
            if block == 0:
                shift_size = (0, 0, 0)
            else:
                shift_size = tuple(wd // 2 if dim_sz > wd else 0 for dim_sz in (d, h, w))
            mask_windows = partition_mask_windows(ds_mask, window_size, shift_size)

            n_win_attn = attn.shape[0]
            if n_win_attn % mask_windows.shape[0] != 0:
                logger.warning(
                    "ASI visual: %s windows %d not divisible by mask windows %d",
                    key, n_win_attn, mask_windows.shape[0],
                )
                continue
            bsz = n_win_attn // mask_windows.shape[0]
            boundary_idx = select_boundary_windows(
                mask_windows, min_tumor=min_tumor, min_nontumor=min_nontumor,
            )
            if boundary_idx.numel() == 0:
                continue

            # Pick the window with the most balanced tumor/non-tumor split.
            n_tum = mask_windows.sum(dim=1)
            n_total = mask_windows.shape[1]
            balance = torch.minimum(n_tum, n_total - n_tum)
            best_local = int(torch.argmax(balance[boundary_idx]).item())
            w_idx = int(boundary_idx[best_local].item())
            mask_w = mask_windows[w_idx]
            tumor_idx = torch.nonzero(mask_w > 0.5, as_tuple=False).squeeze(-1)
            bg_idx = torch.nonzero(mask_w <= 0.5, as_tuple=False).squeeze(-1)
            order = torch.cat([tumor_idx, bg_idx])

            # Use the first batch element (bsz=1 in the orchestrator).
            attn_w = attn[w_idx]  # [H, N, N]
            attn_reord = attn_w.index_select(1, order).index_select(2, order)
            chosen = {
                "attn_matrix": attn_reord.numpy().astype(np.float32),
                "n_tumor": int(tumor_idx.numel()),
                "block": block,
            }
            logger.info(
                "ASI visual stage %d: block=%d window=%d (n_tumor=%d / N=%d, bsz=%d)",
                stage, block, w_idx, tumor_idx.numel(), n_total, bsz,
            )
            break
        if chosen is not None:
            visual[stage] = chosen
        else:
            logger.warning("ASI visual: no boundary window for stage %d", stage)
    return visual


# ---------------------------------------------------------------------------
# DAD visual: row-averaged attention for one MEN + one GLI scan
# ---------------------------------------------------------------------------


def _build_dad_visual(
    men_row_avg: dict[str, np.ndarray],
    gli_row_avg: dict[str, np.ndarray],
    target_stages: tuple[int, ...] = (1, 2, 3, 4),
    block: int = 0,
    head: int = 0,
) -> dict[int, dict]:
    """Build per-stage MEN/GLI row-averaged attention dicts for the renderer."""
    visual: dict[int, dict] = {}
    for stage in target_stages:
        key = f"stage_{stage}_block_{block}"
        if key not in men_row_avg or key not in gli_row_avg:
            logger.warning("DAD visual: missing key %s in one cohort", key)
            continue
        visual[stage] = {
            "men": men_row_avg[key].astype(np.float32),
            "gli": gli_row_avg[key].astype(np.float32),
            "head": head,
            "block": block,
        }
    return visual


# ---------------------------------------------------------------------------
# Single-scan TSI + ASI capture
# ---------------------------------------------------------------------------


def _capture_tsi_and_asi(
    model: torch.nn.Module,
    sample: dict,
    scan_id: str,
    condition: str,
    analysis_cfg: DictConfig,
    device: str,
) -> tuple:
    """Run one forward; return (ScanTSIResult, captured_attn, gt_mask, brain_mask, mri_volume).

    ``mri_volume`` is the channel-1 (T1ce) slice indexed for the
    visualisation, returned as a NumPy array.
    """
    images = sample["image"].unsqueeze(0).to(device)
    gt_mask = get_wt_mask(sample["seg"])
    bm_threshold = analysis_cfg.brain_mask_threshold
    if bm_threshold is not None:
        bm_threshold = float(bm_threshold)
    brain_mask = derive_brain_mask(images.cpu(), threshold=bm_threshold)

    asi_stages = set(int(s) for s in analysis_cfg.asi_stages)
    with AttentionCapture(model, mode="store", target_stages=asi_stages) as cap:
        with torch.amp.autocast("cuda", enabled=(device != "cpu")):
            hidden_states = extract_hidden_states(model, images)
        captured = cap.get_attention_maps()

    scan_tsi = compute_tsi_single_scan(
        hidden_states=hidden_states,
        gt_mask=gt_mask,
        scan_id=scan_id,
        condition=condition,
        thresholds=list(analysis_cfg.tsi_thresholds),
        top_k=int(analysis_cfg.tsi_top_k),
        return_maps=True,
        epsilon=float(analysis_cfg.tsi_epsilon),
        brain_mask=brain_mask,
    )
    mri = images[0].cpu().numpy()  # [C, D, H, W]
    return scan_tsi, captured, gt_mask, brain_mask, mri


# ---------------------------------------------------------------------------
# Single-scan DAD row-averaged attention
# ---------------------------------------------------------------------------


def _capture_dad_row_avg(
    model: torch.nn.Module,
    sample: dict,
    target_stages: set[int],
    device: str,
) -> dict[str, np.ndarray]:
    """Run a forward through ``model`` and return per-(stage, block) row-avg attn."""
    images = sample["image"].unsqueeze(0).to(device)
    acc = DADScanAccumulator(target_stages=target_stages)
    with AttentionCapture(model, mode="callback", process_fn=acc, target_stages=target_stages):
        with torch.amp.autocast("cuda", enabled=(device != "cpu")):
            with torch.no_grad():
                _ = extract_hidden_states(model, images)
    del images
    if device != "cpu":
        torch.cuda.empty_cache()
    return acc.result().row_avg


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate detailed per-metric explainability figures",
    )
    parser.add_argument("--config", required=True, help="Parent config.yaml")
    parser.add_argument(
        "--analysis-config", default=None,
        help="Analysis config (default: explainability/config.yaml)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing the raw/ subfolder with cached CSV/NPZ "
             "(default: analysis_config.paths.output_dir)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for figures (default: --data-dir)",
    )
    parser.add_argument(
        "--condition", default="frozen", choices=["frozen", "adapted"],
        help="Which model variant to use for the visual rows",
    )
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank for --condition adapted")
    parser.add_argument("--member-id", type=int, default=0, help="LoRA member for --condition adapted")
    parser.add_argument("--run-dir", default=None, help="Override LoRA run directory")
    parser.add_argument("--scan-idx", type=int, default=0, help="MEN scan index (within selection)")
    parser.add_argument("--gli-scan-idx", type=int, default=0, help="GLI scan index (within selection)")
    parser.add_argument("--metrics", nargs="+", default=["tsi", "asi", "dad"],
                        choices=["tsi", "asi", "dad"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--format", default=None, help="Output format (default: from analysis_config)")
    parser.add_argument("--dpi", type=int, default=None)
    args = parser.parse_args()

    _setup_logging()

    config = OmegaConf.load(args.config)
    analysis_path = args.analysis_config or (
        Path(args.config).parent / "explainability" / "config.yaml"
    )
    analysis_config = OmegaConf.load(analysis_path)
    analysis_cfg = analysis_config.analysis

    data_dir = Path(args.data_dir or analysis_config.paths.output_dir)
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw/ directory not found under {data_dir}")
    output_dir = Path(args.output or data_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format or str(analysis_config.figure.save_format)
    dpi = args.dpi or int(analysis_config.figure.save_dpi)
    mri_channel = int(analysis_config.figure.mri_channel)
    slice_mode = str(analysis_config.figure.slice_selection)
    lora_stages = set(int(s) for s in config.lora.target_stages)
    target_stages_tuple = tuple(int(s) for s in analysis_cfg.asi_stages)
    asi_target_set = set(target_stages_tuple)

    # ------------------------------------------------------------------
    # Load model (one-shot for the visual rows)
    # ------------------------------------------------------------------
    if args.condition == "frozen":
        logger.info("Loading frozen BrainSegFounder for visual rows")
        model = load_frozen_model(config, device=args.device)
        condition_label = "frozen"
    else:
        logger.info(
            "Loading LoRA-adapted model rank=%d member=%d for visual rows",
            args.rank, args.member_id,
        )
        run_dir_override = Path(args.run_dir) if args.run_dir else None
        model = load_adapted_model(
            config, analysis_config,
            rank=args.rank, member_id=args.member_id,
            device=args.device, run_dir=run_dir_override,
        )
        condition_label = f"adapted_r{args.rank}_m{args.member_id}"

    # ------------------------------------------------------------------
    # Load MEN test scan (used by TSI + ASI + DAD-MEN)
    # ------------------------------------------------------------------
    roi_size = tuple(int(x) for x in analysis_cfg.roi_size)
    men_loader = build_men_loader(config, roi_size=roi_size)
    n_scans = int(analysis_cfg.n_scans_tsi)
    men_indices = select_scan_indices(
        len(men_loader.dataset), n_scans,
        str(analysis_cfg.scan_selection), int(analysis_cfg.seed),
    )
    if not 0 <= args.scan_idx < len(men_indices):
        raise ValueError(
            f"--scan-idx {args.scan_idx} out of range [0, {len(men_indices)})",
        )
    ds_idx = men_indices[args.scan_idx]
    men_sample = men_loader.dataset[ds_idx]
    men_sid = men_loader.all_scan_ids[men_loader.test_indices[ds_idx]]
    logger.info("MEN visual scan: idx=%d (%s)", ds_idx, men_sid)

    # ------------------------------------------------------------------
    # TSI + ASI: shared forward pass on the MEN scan
    # ------------------------------------------------------------------
    if "tsi" in args.metrics or "asi" in args.metrics:
        scan_tsi, captured, gt_mask, _bm, mri = _capture_tsi_and_asi(
            model, men_sample, scan_id=men_sid, condition=condition_label,
            analysis_cfg=analysis_cfg, device=args.device,
        )
        if mri_channel >= mri.shape[0]:
            raise ValueError(f"mri_channel {mri_channel} out of range for {mri.shape[0]} channels")
        mri_volume = mri[mri_channel]
        wt_volume = gt_mask.numpy().astype(np.float32)
        slice_idx = _pick_slice(wt_volume, mode=slice_mode)
        logger.info("Visual axial slice: %d (mode=%s)", slice_idx, slice_mode)

    # ---- TSI figure ----
    if "tsi" in args.metrics:
        channels_npz = raw_dir / f"tsi_{condition_label.split('_')[0]}_channels.npz"
        out_path = output_dir / f"tsi_detailed_{condition_label}.{fmt}"
        render_tsi_detailed(
            scan_tsi=scan_tsi, mri_volume=mri_volume, wt_mask=wt_volume,
            slice_idx=slice_idx, channels_npz_path=channels_npz,
            condition=condition_label, out_path=out_path,
            lora_stages=lora_stages, dpi=dpi,
        )

    # ---- ASI figure ----
    if "asi" in args.metrics:
        asi_visual = _build_asi_visual(
            captured=captured, gt_mask=gt_mask,
            target_stages=target_stages_tuple,
            min_tumor=int(analysis_cfg.asi_min_tumor_tokens),
            min_nontumor=int(analysis_cfg.asi_min_nontumor_tokens),
        )
        asi_csv = raw_dir / f"asi_{condition_label.split('_')[0]}_per_scan.csv"
        if asi_csv.exists():
            asi_df = pd.read_csv(asi_csv)
        else:
            logger.warning("ASI CSV missing (%s); aggregate rows will be empty", asi_csv)
            asi_df = pd.DataFrame(columns=["stage", "block", "head", "asi_value"])
        out_path = output_dir / f"asi_detailed_{condition_label}.{fmt}"
        render_asi_detailed(
            asi_visual=asi_visual, asi_per_scan_df=asi_df, scan_id=men_sid,
            condition=condition_label, out_path=out_path,
            stages=target_stages_tuple, lora_stages=lora_stages, dpi=dpi,
        )

    # ------------------------------------------------------------------
    # DAD: needs one MEN + one GLI scan
    # ------------------------------------------------------------------
    if "dad" in args.metrics:
        gli_loader = build_gli_loader(config, roi_size=roi_size)
        gli_indices = select_scan_indices(
            len(gli_loader.dataset), int(analysis_cfg.n_scans_dad),
            str(analysis_cfg.scan_selection), int(analysis_cfg.seed),
        )
        if not 0 <= args.gli_scan_idx < len(gli_indices):
            raise ValueError(
                f"--gli-scan-idx {args.gli_scan_idx} out of range [0, {len(gli_indices)})",
            )
        gli_ds_idx = gli_indices[args.gli_scan_idx]
        gli_sample = gli_loader.dataset[gli_ds_idx]
        gli_sid = gli_loader.all_scan_ids[gli_loader.test_indices[gli_ds_idx]]
        logger.info("DAD GLI visual scan: idx=%d (%s)", gli_ds_idx, gli_sid)

        dad_stages_set = set(int(s) for s in analysis_cfg.dad_stages)
        men_row_avg = _capture_dad_row_avg(model, men_sample, dad_stages_set, args.device)
        gli_row_avg = _capture_dad_row_avg(model, gli_sample, dad_stages_set, args.device)

        # Map our condition_label to the DAD CSV's condition values.
        # ``run_analysis.py`` uses "frozen" or "adapted_r{rank}".
        if condition_label == "frozen":
            dad_condition_filter = "frozen"
        else:
            dad_condition_filter = f"adapted_r{args.rank}"

        dad_visual = _build_dad_visual(
            men_row_avg=men_row_avg, gli_row_avg=gli_row_avg,
            target_stages=tuple(sorted(dad_stages_set)),
            block=0, head=0,
        )
        dad_csv = raw_dir / "dad_per_head.csv"
        if dad_csv.exists():
            dad_df = pd.read_csv(dad_csv)
        else:
            logger.warning("DAD CSV missing (%s); rows 2-3 will be empty", dad_csv)
            dad_df = pd.DataFrame(
                columns=["condition", "stage", "block", "head", "dad", "p_value",
                         "null_mean", "null_std"],
            )
        null_npz = raw_dir / "dad_permutation_null.npz"
        out_path = output_dir / f"dad_detailed_{condition_label}.{fmt}"
        render_dad_detailed(
            dad_visual=dad_visual, dad_df=dad_df, null_npz_path=null_npz,
            condition=dad_condition_filter, out_path=out_path,
            stages=tuple(sorted(dad_stages_set)),
            lora_stages=lora_stages, dpi=dpi,
        )

    logger.info("Detailed figures written to %s", output_dir)


if __name__ == "__main__":
    main()
