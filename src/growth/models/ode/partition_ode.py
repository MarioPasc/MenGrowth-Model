# src/growth/models/ode/partition_ode.py
"""
Partition-aware Neural ODE for full latent space.

Combines Gompertz physics (volume) with learned dynamics (loc, shape, residual).
Each partition has its own MLP with scaling factor (eta).
"""
