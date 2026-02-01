# src/growth/training/train_sdp.py
"""
Phase 2 training entry point: SDP training.

Loads merged encoder from Phase 1, trains SDP network with
semantic regression and VICReg-style disentanglement losses.
"""
