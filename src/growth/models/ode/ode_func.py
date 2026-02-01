# src/growth/models/ode/ode_func.py
"""
ODEFunc wrapper for torchdiffeq integration.

Wraps partition ODE as nn.Module compatible with torchdiffeq.odeint.
Handles time encoding and numerical stability.
"""
