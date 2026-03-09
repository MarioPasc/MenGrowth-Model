# src/growth/models/growth/pamogp_model.py
"""
Multi-Output Gaussian Process (MOGP) growth model (Model C).

Multi-output GP on z_vol ∈ ℝ³² with Matérn-5/2 temporal kernel.
Location centroid is used as a static GP covariate, not as a temporal
latent dimension.

Methodology Revision R1: PA-MOGP → MOGP. Partition-specific kernels and
cross-partition coupling removed (no location/shape temporal dynamics).
See module_5_growth_prediction.md Section 2.3.
"""
