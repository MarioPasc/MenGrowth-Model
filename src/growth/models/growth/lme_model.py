# src/growth/models/growth/lme_model.py
"""
Linear Mixed-Effects (LME) baseline growth model (Model A).

Per-dimension LME on z_vol ∈ ℝ²⁴:
  z_d(t) = (β₀ + b₀ᵢ) + (β₁ + b₁ᵢ) · t + ε

Fitted via REML (statsmodels.MixedLM). Patient-specific predictions via BLUP.
Automatic shrinkage for patients with few observations (n_i = 2).
See module_5_growth_prediction.md Section 2.1.
"""
