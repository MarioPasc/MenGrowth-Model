# experiments/lora_ablation/__init__.py
"""LoRA ablation experiment for meningioma encoder adaptation.

This experiment compares:
- Baseline (frozen glioma-trained encoder)
- LoRA rank 4, 8, 16 adaptations

Primary metric: Linear probe R² for semantic features.
Secondary metric: Segmentation Dice.
"""
