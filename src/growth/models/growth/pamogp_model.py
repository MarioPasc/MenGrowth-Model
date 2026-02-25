# src/growth/models/growth/pamogp_model.py
"""
Partition-Aware Multi-Output Gaussian Process (PA-MOGP) growth model (Model C).

Novel contribution: multi-output GP on z_active ∈ ℝ⁴⁴ (vol+loc+shape) with:
- Partition-specific temporal kernels (Matérn-5/2 vol, SE loc, Matérn-3/2 shape)
- Rank-1 cross-partition coupling via ICM (D20)
- 95 total shared hyperparameters

The coupling term encodes the mechanistic hypothesis that volume growth drives
secondary changes in shape and location.
See module_5_growth_prediction.md Section 2.3.
"""
