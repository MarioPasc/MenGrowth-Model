# src/growth/training/lit_modules/lora_module.py
"""
LightningModule for Phase 1: LoRA segmentation adaptation.

Manages SwinViT encoder with LoRA, segmentation head, and Dice+CE loss.
Handles separate learning rates for LoRA and segmentation head parameters.
"""
