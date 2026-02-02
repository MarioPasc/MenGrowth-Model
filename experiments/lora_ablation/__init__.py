# experiments/lora_ablation/__init__.py
"""LoRA ablation experiment for meningioma encoder adaptation.

This unified experiment compares:
- Baseline (frozen encoder with configurable decoder)
- LoRA rank 2, 4, 8, 16, 32 adaptations

Decoder types (via decoder_type config):
- "lightweight": Custom SegmentationHead (~2M params)
- "original": Full SwinUNETR decoder (~30M params, recommended)

Primary metric: Linear probe R² for semantic features.
Secondary metrics: MLP probe R², Segmentation Dice.

Usage:
    growth-exp-lora-ablation run-all --config path/to/ablation.yaml
"""
