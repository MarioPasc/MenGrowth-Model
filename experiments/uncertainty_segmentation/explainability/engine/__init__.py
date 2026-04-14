"""Compute engine for the SwinViT explainability analysis.

Submodules
----------
- ``brain_mask``     : Derive a brain-vs-background mask from MRI intensities.
- ``tsi``            : Brain-masked Tumor Selectivity Index.
- ``asi``            : Attention Selectivity Index from WindowAttention weights.
- ``dad``            : Domain Attention Divergence between BraTS-GLI and BraTS-MEN.
- ``hooks``          : Context manager that captures attention weights from
                       MONAI's WindowAttention modules without breaking the
                       network's normal forward pass.
- ``model_loader``   : Load frozen and LoRA-adapted BrainSegFounder variants.
- ``data_loader``    : Open BraTS H5 files, resolve splits, sample scans.
"""
