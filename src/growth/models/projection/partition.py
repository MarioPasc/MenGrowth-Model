# src/growth/models/projection/partition.py
"""
Latent space partitioning utilities.

Defines partition boundaries and provides slicing operations:
- z_vol: dims 0-23 (24 dims)
- z_loc: dims 24-31 (8 dims)
- z_shape: dims 32-43 (12 dims)
- z_residual: dims 44-127 (84 dims)
"""
