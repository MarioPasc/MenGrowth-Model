#!/usr/bin/env python
"""Merge LoRA adapter weights into base SwinUNETR encoder.

Standalone script for post-Phase 1 checkpoint preparation:
1. Load best LoRA adapter checkpoint from a completed ablation condition
2. Load base SwinUNETR + inject LoRA + load adapter weights
3. Merge LoRA weights into base model
4. Extract encoder-only keys (discard decoder, out, aux)
5. Save as phase1_encoder_merged.pt

Usage:
    python -m experiments.lora_ablation.merge_lora_checkpoint \
        --adapter-dir outputs/lora_ablation/LoRA_semantic/lora_r8/adapter \
        --checkpoint /path/to/finetuned_model_fold_0.pt \
        --output phase1_encoder_merged.pt

    # Or use defaults from ablation config:
    python -m experiments.lora_ablation.merge_lora_checkpoint \
        --adapter-dir outputs/lora_ablation/LoRA_semantic/lora_r8/adapter
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Ensure project root is on path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.growth.models.encoder.lora_adapter import LoRASwinViT
from src.growth.models.encoder.swin_loader import create_swinunetr, load_full_swinunetr
from src.growth.utils.checkpoint import (
    extract_encoder_weights,
    get_checkpoint_stats,
    print_checkpoint_summary,
)

logger = logging.getLogger(__name__)


def merge_and_save(
    adapter_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    rank: int = 8,
    alpha: int = 16,
    use_dora: bool = False,
) -> None:
    """Merge LoRA weights and save encoder-only checkpoint.

    Args:
        adapter_dir: Directory containing saved PEFT adapter.
        checkpoint_path: Path to base BrainSegFounder checkpoint.
        output_path: Path for output merged encoder checkpoint.
        rank: LoRA rank (must match adapter).
        alpha: LoRA alpha (must match adapter).
        use_dora: Whether the adapter used DoRA.
    """
    # Step 1: Load base model with pretrained weights
    logger.info(f"Loading base SwinUNETR from {checkpoint_path}")
    base_model = load_full_swinunetr(checkpoint_path, device="cpu")

    # Step 2: Load LoRA adapter onto base model
    logger.info(f"Loading LoRA adapter from {adapter_dir}")
    lora_model = LoRASwinViT.load_lora(
        base_model, adapter_dir, device="cpu", trainable=False
    )

    # Step 3: Merge LoRA weights into base
    logger.info("Merging LoRA weights into base model")
    merged_model = lora_model.merge_lora()

    # Step 4: Extract encoder-only state dict
    full_state_dict = merged_model.state_dict()
    encoder_state_dict = extract_encoder_weights(full_state_dict)

    # Step 5: Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state_dict, output_path)

    # Print summary
    print(f"\nMerged encoder checkpoint saved to: {output_path}")
    print(f"  Keys: {len(encoder_state_dict)}")
    total_params = sum(v.numel() for v in encoder_state_dict.values())
    print(f"  Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    print("\nEncoder key prefixes:")
    stats = get_checkpoint_stats(encoder_state_dict)
    for prefix in sorted(stats):
        info = stats[prefix]
        print(f"  {prefix}: {info['count']} keys, {info['params_m']:.2f}M params")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base SwinUNETR encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Path to saved PEFT adapter directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to base BrainSegFounder checkpoint (default: from adapter config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for merged checkpoint (default: adapter_dir/../phase1_encoder_merged.pt)",
    )
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--use-dora", action="store_true", help="Adapter uses DoRA"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.adapter_dir.exists():
        parser.error(f"Adapter directory not found: {args.adapter_dir}")

    if args.checkpoint is None:
        parser.error("--checkpoint is required (path to base BrainSegFounder .pt file)")

    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    if args.output is None:
        args.output = args.adapter_dir.parent / "phase1_encoder_merged.pt"

    merge_and_save(
        adapter_dir=args.adapter_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        rank=args.rank,
        alpha=args.alpha,
        use_dora=args.use_dora,
    )


if __name__ == "__main__":
    main()
