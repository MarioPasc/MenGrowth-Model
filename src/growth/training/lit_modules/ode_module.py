# src/growth/training/lit_modules/ode_module.py
"""
LightningModule for Phase 4: Neural ODE training.

Manages Gompertz-informed ODE, trajectory loss, and smoothness regularization.
Handles integration with torchdiffeq ODE solvers.
"""
