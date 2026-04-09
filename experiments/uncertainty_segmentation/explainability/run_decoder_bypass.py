"""Decoder Bypass Test: Does the decoder rely on LoRA features?

Tests whether the 54M-parameter decoder co-adapted with LoRA or learned
to bypass the LoRA perturbations via high-resolution skip connections
from stages 0-2.

Three conditions using the SAME trained decoder weights:
  A: Full model — LoRA encoder + trained decoder (normal inference)
  B: LoRA zeroed — Frozen encoder + trained decoder (bypass test)
  C: Baseline — Frozen encoder + pretrained decoder (frozen reference)

The critical comparison is A vs B:
  If Dice_A ≈ Dice_B: decoder bypasses LoRA (LoRA is decorative)
  If Dice_A >> Dice_B: decoder co-adapted with LoRA (LoRA contributes)

Usage:
    python -m experiments.uncertainty_segmentation.explainability.run_decoder_bypass \\
        --config experiments/uncertainty_segmentation/config.yaml \\
        --rank 4 8 16 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from growth.data.bratsmendata import BraTSDatasetH5
from growth.data.transforms import get_h5_val_transforms
from growth.inference.sliding_window import sliding_window_segment
from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_full_swinunetr
from growth.models.segmentation.original_decoder import (
    LoRAOriginalDecoderModel,
    OriginalDecoderWrapper,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results"
    "/uncertainty_segmentation/decoder_bypass"
)


# -------------------------------------------------------------------------
# Dice computation (reused from evaluate_members.py)
# -------------------------------------------------------------------------


def _convert_seg_to_binary(seg: torch.Tensor) -> torch.Tensor:
    """Convert MEN integer labels to 3-channel binary masks (TC/WT/ET)."""
    seg = seg.squeeze(0).long()
    tc = ((seg == 1) | (seg == 2)).float()
    wt = ((seg == 1) | (seg == 2) | (seg == 3)).float()
    et = (seg == 1).float()
    return torch.stack([tc, wt, et], dim=0)


def _compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute per-channel Dice between binary masks [C, D, H, W]."""
    C = pred.shape[0]
    dice = torch.zeros(C)
    for c in range(C):
        p, t = pred[c].flatten(), target[c].flatten()
        intersection = (p * t).sum()
        dice[c] = (2 * intersection + smooth) / (p.sum() + t.sum() + smooth)
    return dice


# -------------------------------------------------------------------------
# Model builders for each condition
# -------------------------------------------------------------------------


class BypassModel(nn.Module):
    """Condition B: Frozen encoder + trained decoder (no LoRA).

    Uses the base SwinUNETR encoder (identical to pretrained, no LoRA)
    paired with the decoder weights trained alongside LoRA. If the decoder
    co-adapted with LoRA, this configuration will produce worse Dice than
    the full model because the decoder expects LoRA-modified features at
    stages 3-4 but receives pristine ones.
    """

    def __init__(self, base_model: nn.Module, decoder_wrapper: OriginalDecoderWrapper) -> None:
        super().__init__()
        self.swinViT = base_model.swinViT
        self.normalize = base_model.normalize
        self.decoder = decoder_wrapper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen encoder → trained decoder."""
        hidden_states = self.swinViT(x, self.normalize)
        return self.decoder(x, hidden_states)


def load_condition_a(
    ckpt_path: Path,
    member_dir: Path,
    device: str,
) -> nn.Module:
    """Condition A: Full LoRA + trained decoder (normal inference).

    Args:
        ckpt_path: BrainSegFounder checkpoint.
        member_dir: Member directory with adapter/ and decoder.pt.
        device: Target device.

    Returns:
        LoRAOriginalDecoderModel in eval mode.
    """
    full_model = load_full_swinunetr(
        ckpt_path, freeze_encoder=True, freeze_decoder=True,
        out_channels=3, device="cpu",
    )
    lora_encoder = LoRASwinViT.load_lora(
        base_encoder=full_model,
        adapter_path=str(member_dir / "adapter"),
        device="cpu", trainable=False,
    )
    model = LoRAOriginalDecoderModel(
        lora_encoder=lora_encoder, freeze_decoder=True,
        out_channels=3, use_semantic_heads=False,
    )
    decoder_state = torch.load(member_dir / "decoder.pt", map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(decoder_state)
    return model.to(device).eval()


def load_condition_b(
    ckpt_path: Path,
    member_dir: Path,
    device: str,
) -> nn.Module:
    """Condition B: Frozen encoder + SAME trained decoder (no LoRA).

    The encoder is the original BrainSegFounder with zero LoRA contribution.
    The decoder is the exact same weights from decoder.pt that were trained
    alongside LoRA. This tests whether the decoder depends on LoRA features.

    Args:
        ckpt_path: BrainSegFounder checkpoint.
        member_dir: Member directory with decoder.pt.
        device: Target device.

    Returns:
        BypassModel in eval mode.
    """
    # Fresh base model (no LoRA ever applied)
    base_model = load_full_swinunetr(
        ckpt_path, freeze_encoder=True, freeze_decoder=True,
        out_channels=3, device="cpu",
    )
    # Create decoder wrapper and load trained weights
    decoder_wrapper = OriginalDecoderWrapper(base_model, freeze_decoder=True)
    decoder_state = torch.load(member_dir / "decoder.pt", map_location="cpu", weights_only=True)
    decoder_wrapper.load_state_dict(decoder_state)

    model = BypassModel(base_model, decoder_wrapper)
    return model.to(device).eval()


def load_condition_c(
    ckpt_path: Path,
    device: str,
) -> nn.Module:
    """Condition C: Frozen encoder + pretrained decoder (no training at all).

    This is the pure BrainSegFounder baseline.

    Args:
        ckpt_path: BrainSegFounder checkpoint.
        device: Target device.

    Returns:
        SwinUNETR in eval mode.
    """
    model = load_full_swinunetr(
        ckpt_path, freeze_encoder=True, freeze_decoder=True,
        out_channels=3, device=device,
    )
    return model.eval()


# -------------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------------


def evaluate_condition(
    model: nn.Module,
    dataset: BraTSDatasetH5,
    all_scan_ids: list[str],
    test_indices: np.ndarray,
    condition: str,
    device: str,
    sw_roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 2,
) -> pd.DataFrame:
    """Evaluate one condition on all test scans.

    Args:
        model: Model for this condition.
        dataset: Test dataset.
        all_scan_ids: Scan ID strings from H5.
        test_indices: Test split indices.
        condition: Condition label (A, B, or C).
        device: Compute device.
        sw_roi_size: Sliding window patch size.

    Returns:
        DataFrame with columns: scan_id, condition, dice_tc, dice_wt, dice_et.
    """
    rows = []
    n = len(dataset)

    for i in range(n):
        sample = dataset[i]
        images = sample["image"].unsqueeze(0).to(device)
        seg_gt = sample["seg"]
        sid = all_scan_ids[test_indices[i]]

        t0 = time.time()

        with torch.amp.autocast("cuda", enabled=(device != "cpu")):
            logits = sliding_window_segment(
                model, images,
                roi_size=sw_roi_size,
                sw_batch_size=sw_batch_size,
                overlap=0.5,
                mode="gaussian",
            )

        probs = torch.sigmoid(logits).float().squeeze(0).cpu()
        pred_binary = (probs > 0.5).float()
        gt_binary = _convert_seg_to_binary(seg_gt)
        dice = _compute_dice(pred_binary, gt_binary)

        elapsed = time.time() - t0

        rows.append({
            "scan_id": sid,
            "condition": condition,
            "dice_tc": dice[0].item(),
            "dice_wt": dice[1].item(),
            "dice_et": dice[2].item(),
            "dice_mean": dice.mean().item(),
        })

        if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
            logger.info(
                f"  [{condition}] {i+1}/{n} ({sid}): "
                f"TC={dice[0]:.3f} WT={dice[1]:.3f} ET={dice[2]:.3f} "
                f"({elapsed:.1f}s)"
            )

        del images, logits
        if device != "cpu":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Statistical analysis
# -------------------------------------------------------------------------


def analyze_bypass(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    rank: int,
) -> dict:
    """Compute bypass statistics for one rank.

    Args:
        df_a: Condition A results (full model).
        df_b: Condition B results (LoRA zeroed).
        df_c: Condition C results (baseline).
        rank: LoRA rank.

    Returns:
        Dict with bypass statistics.
    """
    channels = ["dice_tc", "dice_wt", "dice_et", "dice_mean"]
    stats: dict = {"rank": rank}

    for ch in channels:
        a_vals = df_a.sort_values("scan_id")[ch].values
        b_vals = df_b.sort_values("scan_id")[ch].values
        c_vals = df_c.sort_values("scan_id")[ch].values
        n = len(a_vals)

        # A vs B: bypass test (paired)
        delta_ab = a_vals - b_vals
        try:
            w_ab = scipy.stats.wilcoxon(delta_ab, alternative="greater")
            p_ab = w_ab.pvalue
        except ValueError:
            p_ab = 1.0

        # A vs C: LoRA+decoder vs pure baseline (paired)
        delta_ac = a_vals - c_vals
        try:
            w_ac = scipy.stats.wilcoxon(delta_ac, alternative="greater")
            p_ac = w_ac.pvalue
        except ValueError:
            p_ac = 1.0

        # B vs C: trained decoder (no LoRA) vs pure baseline
        delta_bc = b_vals - c_vals
        try:
            w_bc = scipy.stats.wilcoxon(delta_bc, alternative="greater")
            p_bc = w_bc.pvalue
        except ValueError:
            p_bc = 1.0

        stats[ch] = {
            "mean_A": float(np.mean(a_vals)),
            "mean_B": float(np.mean(b_vals)),
            "mean_C": float(np.mean(c_vals)),
            "delta_AB": float(np.mean(delta_ab)),
            "delta_AB_sd": float(np.std(delta_ab, ddof=1)),
            "p_AB": float(p_ab),
            "delta_AC": float(np.mean(delta_ac)),
            "p_AC": float(p_ac),
            "delta_BC": float(np.mean(delta_bc)),
            "p_BC": float(p_bc),
            "n_scans": int(n),
        }

    return stats


def print_results(stats: dict) -> None:
    """Print a clear results table for one rank."""
    rank = stats["rank"]
    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"DECODER BYPASS TEST — rank={rank}")
    logger.info(f"{'='*70}")

    for ch_label, ch_key in [("TC", "dice_tc"), ("WT", "dice_wt"), ("ET", "dice_et"), ("Mean", "dice_mean")]:
        s = stats[ch_key]
        logger.info(f"\n  {ch_label} Dice (N={s['n_scans']} scans):")
        logger.info(f"    Condition A (LoRA + decoder):  {s['mean_A']:.4f}")
        logger.info(f"    Condition B (no LoRA + decoder): {s['mean_B']:.4f}")
        logger.info(f"    Condition C (baseline):         {s['mean_C']:.4f}")
        logger.info(f"    ---")
        logger.info(f"    Δ(A-B) bypass:  {s['delta_AB']:+.4f} ± {s['delta_AB_sd']:.4f}  p={s['p_AB']:.2e}")
        logger.info(f"    Δ(A-C) total:   {s['delta_AC']:+.4f}  p={s['p_AC']:.2e}")
        logger.info(f"    Δ(B-C) decoder: {s['delta_BC']:+.4f}  p={s['p_BC']:.2e}")

    # Interpretation
    s = stats["dice_mean"]
    logger.info(f"\n  INTERPRETATION (rank={rank}):")
    total_gain = s["delta_AC"]
    lora_gain = s["delta_AB"]
    decoder_gain = s["delta_BC"]

    if total_gain > 0.001:
        lora_pct = (lora_gain / total_gain * 100) if total_gain > 0 else 0
        decoder_pct = (decoder_gain / total_gain * 100) if total_gain > 0 else 0
        logger.info(f"    Total improvement (A-C): {total_gain:+.4f}")
        logger.info(f"    LoRA contribution (A-B): {lora_gain:+.4f} ({lora_pct:.1f}% of total)")
        logger.info(f"    Decoder contribution (B-C): {decoder_gain:+.4f} ({decoder_pct:.1f}% of total)")

        if s["p_AB"] < 0.05:
            logger.info(f"    → LoRA contribution is SIGNIFICANT (p={s['p_AB']:.2e})")
            logger.info(f"      The decoder co-adapted with LoRA features.")
        else:
            logger.info(f"    → LoRA contribution is NOT significant (p={s['p_AB']:.2e})")
            logger.info(f"      The decoder may bypass LoRA features.")
    else:
        logger.info(f"    No meaningful improvement over baseline.")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    """Run the decoder bypass experiment."""
    parser = argparse.ArgumentParser(description="Decoder Bypass Test")
    parser.add_argument("--config", required=True, help="Parent config.yaml")
    parser.add_argument("--rank", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--member-id", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Base dir containing r{rank}_M20_s42 run dirs (default: config output_dir)",
    )
    parser.add_argument(
        "--roi-size", type=int, nargs=3, default=None,
        help="Override ROI size for transforms (default: from config inference_roi_size)",
    )
    parser.add_argument(
        "--sw-batch-size", type=int, default=2,
        help="Sliding window batch size (reduce for low VRAM, default: 2)",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "decoder_bypass.log", mode="w"),
        ],
    )

    ckpt_path = Path(config.paths.checkpoint_dir) / config.paths.checkpoint_filename
    h5_path = config.paths.men_h5_file

    logger.info("=" * 70)
    logger.info("DECODER BYPASS EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"H5: {h5_path}")
    logger.info(f"Ranks: {args.rank}")
    logger.info(f"Member: {args.member_id}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}")

    # Load test dataset
    if args.roi_size:
        roi_size = tuple(args.roi_size)
    else:
        roi_size = tuple(config.data.get("inference_roi_size", config.data.val_roi_size))
    sw_batch_size = args.sw_batch_size
    logger.info(f"ROI size: {roi_size}, SW batch size: {sw_batch_size}")
    transform = get_h5_val_transforms(roi_size=roi_size)
    dataset = BraTSDatasetH5(
        h5_path=h5_path,
        split=config.data.test_split,
        transform=transform,
        compute_semantic=False,
    )
    with h5py.File(h5_path, "r") as f:
        all_scan_ids = [
            s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]
        ]
    splits = BraTSDatasetH5.load_splits_from_h5(h5_path)
    test_indices = splits.get(config.data.test_split, np.arange(len(all_scan_ids)))

    n_test = len(dataset)
    logger.info(f"Test scans: {n_test}")

    # Run Condition C once (same for all ranks)
    logger.info("\n" + "=" * 70)
    logger.info("CONDITION C: Frozen baseline (pretrained encoder + decoder)")
    logger.info("=" * 70)

    model_c = load_condition_c(ckpt_path, args.device)
    df_c = evaluate_condition(
        model_c, dataset, all_scan_ids, test_indices,
        "C_baseline", args.device,
        sw_batch_size=sw_batch_size,
    )
    del model_c
    torch.cuda.empty_cache()
    df_c.to_csv(output_dir / "condition_C_baseline.csv", index=False)
    logger.info(f"  Mean Dice: TC={df_c['dice_tc'].mean():.4f} WT={df_c['dice_wt'].mean():.4f} ET={df_c['dice_et'].mean():.4f}")

    # Run per rank
    all_stats = []
    if args.results_dir:
        base_results_dir = Path(args.results_dir)
    else:
        base_results_dir = Path(config.experiment.output_dir)

    for rank in args.rank:
        run_dir = base_results_dir / f"r{rank}_M20_s42"
        member_dir = run_dir / "adapters" / f"member_{args.member_id}"

        if not member_dir.exists():
            logger.warning(f"Member dir not found: {member_dir} — skipping rank={rank}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"RANK={rank}: member {args.member_id}")
        logger.info(f"{'='*70}")

        # Condition A: Full LoRA + trained decoder
        logger.info(f"\nCondition A: LoRA (rank={rank}) + trained decoder")
        model_a = load_condition_a(ckpt_path, member_dir, args.device)
        df_a = evaluate_condition(
            model_a, dataset, all_scan_ids, test_indices,
            f"A_lora_r{rank}", args.device,
            sw_batch_size=sw_batch_size,
        )
        del model_a
        torch.cuda.empty_cache()
        df_a.to_csv(output_dir / f"condition_A_lora_r{rank}.csv", index=False)

        # Condition B: Frozen encoder + trained decoder (NO LoRA)
        logger.info(f"\nCondition B: Frozen encoder + trained decoder (r={rank} decoder, NO LoRA)")
        model_b = load_condition_b(ckpt_path, member_dir, args.device)
        df_b = evaluate_condition(
            model_b, dataset, all_scan_ids, test_indices,
            f"B_bypass_r{rank}", args.device,
            sw_batch_size=sw_batch_size,
        )
        del model_b
        torch.cuda.empty_cache()
        df_b.to_csv(output_dir / f"condition_B_bypass_r{rank}.csv", index=False)

        # Analysis
        stats = analyze_bypass(df_a, df_b, df_c, rank)
        all_stats.append(stats)
        print_results(stats)

    # Save all statistics
    with open(output_dir / "bypass_statistics.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # Combined CSV
    all_dfs = [df_c]
    for rank in args.rank:
        a_path = output_dir / f"condition_A_lora_r{rank}.csv"
        b_path = output_dir / f"condition_B_bypass_r{rank}.csv"
        if a_path.exists():
            all_dfs.append(pd.read_csv(a_path))
        if b_path.exists():
            all_dfs.append(pd.read_csv(b_path))
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_dir / "all_conditions.csv", index=False)

    # Summary table
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'='*70}")
    logger.info(f"{'Condition':<30} {'TC':>8} {'WT':>8} {'ET':>8} {'Mean':>8}")
    logger.info("-" * 70)

    for df_path in sorted(output_dir.glob("condition_*.csv")):
        df = pd.read_csv(df_path)
        cond = df["condition"].iloc[0]
        logger.info(
            f"  {cond:<28} {df['dice_tc'].mean():>8.4f} {df['dice_wt'].mean():>8.4f} "
            f"{df['dice_et'].mean():>8.4f} {df['dice_mean'].mean():>8.4f}"
        )

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
