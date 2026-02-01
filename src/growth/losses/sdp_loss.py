# src/growth/losses/sdp_loss.py
"""
Combined SDP loss for Phase 2 training.

Aggregates: semantic regression + covariance + variance + dCor losses.
Provides the complete objective for disentangled projection learning.
"""
