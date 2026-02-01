# src/growth/training/train_lora.py
"""
Phase 1 training entry point: LoRA segmentation adaptation.

Loads BrainSegFounder checkpoint, applies LoRA to Stages 3-4,
and fine-tunes on BraTS-MEN with segmentation objective.
"""
