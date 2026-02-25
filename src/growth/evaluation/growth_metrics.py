# src/growth/evaluation/growth_metrics.py
"""
Growth prediction metrics for Phase 4 evaluation.

Implements:
- Volume prediction R² (LOPO-CV)
- Volume MAE in physical units
- Latent MSE on volume partition
- Calibration: fraction of true values within 95% prediction intervals
- Per-patient trajectory Pearson r (for patients with n_i >= 3)
"""
