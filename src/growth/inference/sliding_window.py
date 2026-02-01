# src/growth/inference/sliding_window.py
"""
Sliding window encoding for full-resolution volumes.

Handles volumes larger than 96^3 via overlapping patches.
Implements tumor-weighted pooling for patch aggregation.
"""
