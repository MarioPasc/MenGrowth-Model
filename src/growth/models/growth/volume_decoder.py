# src/growth/models/growth/volume_decoder.py
"""
Volume decoder for growth prediction.

Decodes z_vol predictions through frozen semantic head π_vol (ℝ²⁴ → ℝ⁴),
with uncertainty propagation through the linear mapping.

Since π_vol is linear:
  V_mean = W · z_vol_mean + b
  V_cov  = W · diag(z_vol_var) · Wᵀ

Handles denormalization using vol_mean/vol_std from SDP training (D14).
"""
