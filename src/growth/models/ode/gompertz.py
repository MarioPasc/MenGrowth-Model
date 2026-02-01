# src/growth/models/ode/gompertz.py
"""
Gompertz growth dynamics for volume partition.

Implements: dz_vol/dt = alpha * z_vol * ln(K / (z_vol + eps))
Learnable parameters: alpha (growth rate), K (carrying capacity).
"""
