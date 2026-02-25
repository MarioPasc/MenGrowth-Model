# src/growth/models/growth/hgp_model.py
"""
Hierarchical Gaussian Process (H-GP) growth model (Model B).

Per-dimension GP on z_vol ∈ ℝ²⁴ with:
- Population linear mean from LME (D18)
- Matérn-5/2 temporal kernel (default; SE and Matérn-3/2 as ablation A9)
- Hierarchical hyperparameter sharing across patients (empirical Bayes, D19)

Posterior conditioning provides calibrated uncertainty that grows with
extrapolation distance. See module_5_growth_prediction.md Section 2.2.
"""
