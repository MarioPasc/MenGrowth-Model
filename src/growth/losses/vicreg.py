# src/growth/losses/vicreg.py
"""
VICReg-style regularization losses.

Implements:
- Cross-partition covariance loss (linear decorrelation)
- Variance hinge loss (collapse prevention)

Adapted from Bardes et al., ICLR 2022.
"""
