# experiments/segment_based_approach/__init__.py
"""Ablation A0: Segment-Based Baseline for meningioma growth prediction.

Frozen BrainSegFounder -> segment WT -> extract volume (mm^3) -> log1p ->
scalar GP (Matern 5/2, LOPO-CV).

Establishes the empirical lower bound that the latent-space pipeline
(LoRA -> SDP -> GP) must beat to justify its added complexity.
"""
