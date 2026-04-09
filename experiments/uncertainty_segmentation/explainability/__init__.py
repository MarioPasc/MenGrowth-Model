"""TSI Explainability Analysis for LoRA-Ensemble uncertainty segmentation.

Computes Tumor Selectivity Index across SwinViT encoder stages to demonstrate
that LoRA adaptation amplifies tumor selectivity in stages 3-4 while preserving
general anatomical representations in stages 0-2.

Reference: Bau et al. (CVPR 2017) Network Dissection — TSI is the continuous,
medical-domain analogue of unit-level IoU interpretability.
"""
