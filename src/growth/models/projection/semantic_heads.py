# src/growth/models/projection/semantic_heads.py
"""
Semantic prediction heads for SDP.

Lightweight MLPs that map latent partitions to semantic targets:
- pi_vol: z_vol (24) -> volumes (4)
- pi_loc: z_loc (8) -> centroid (3)
- pi_shape: z_shape (12) -> shape features (3: sphericity, surface_area_log, solidity)
"""
