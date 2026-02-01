# src/growth/models/encoder/lora_adapter.py
"""
LoRA adapter for SwinViT encoder.

Applies Low-Rank Adaptation to Q, K, V projections in Stages 3-4.
Freezes Stages 0-2 to preserve low-level anatomy features.
"""
