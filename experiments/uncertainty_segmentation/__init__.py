# experiments/uncertainty_segmentation/__init__.py
"""LoRA-Ensemble for uncertainty-aware meningioma segmentation.

Trains M independent LoRA adapters on a shared frozen BrainSegFounder backbone,
then aggregates predictions to produce mean segmentation maps, per-voxel epistemic
uncertainty, and volume estimates with uncertainty bounds.

Reference:
    Mühlematter et al. (2024). LoRA-Ensemble: Efficient Uncertainty Modelling
    for Self-Attention Networks. arXiv:2405.14438.
"""
