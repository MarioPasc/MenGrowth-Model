# src/growth/evaluation/growth_figures.py
"""
Publication-quality figure generation for Phase 4 growth prediction.

Generates figures 9-13:
  9. Volume prediction scatter (predicted vs actual ΔV, colored by model)
 10. Trajectory prediction with uncertainty (mean ± 95% CI, 3-4 patients)
 11. Model comparison bar chart (R², MAE, calibration per model)
 12. Learned kernel hyperparameters (length-scales per partition)
 13. Cross-partition coupling weights (heatmap of wwᵀ from PA-MOGP)
"""
