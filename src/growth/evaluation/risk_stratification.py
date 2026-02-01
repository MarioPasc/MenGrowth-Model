# src/growth/evaluation/risk_stratification.py
"""
Patient-specific risk stratification.

Extracts per-patient Gompertz parameters (alpha, K) from trained ODE.
Computes risk scores and patient stratification based on growth dynamics.
"""
