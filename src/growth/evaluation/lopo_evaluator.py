# src/growth/evaluation/lopo_evaluator.py
"""
Leave-One-Patient-Out (LOPO) cross-validation evaluator.

Runs LOPO-CV (33 folds) for a given growth model class, computes all metrics
(vol R², MAE, latent MSE, calibration, per-patient correlation), and returns
the full results dict matching the module_5 output contract.
"""
