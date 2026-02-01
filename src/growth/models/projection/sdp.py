# src/growth/models/projection/sdp.py
"""
Supervised Disentangled Projection (SDP) network.

2-layer MLP with spectral normalization: 768 -> 512 -> 128.
Maps foundation model features to partitioned latent space.
"""
