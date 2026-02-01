# src/growth/losses/ode_loss.py
"""
Neural ODE training losses for Phase 4.

Implements:
- Trajectory MSE between predicted and actual latent states
- Weight regularization
- Jerk regularization (smoothness of dynamics)
"""
