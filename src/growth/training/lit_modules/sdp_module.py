# src/growth/training/lit_modules/sdp_module.py
"""
LightningModule for Phase 2: SDP training.

Manages frozen encoder, SDP network, semantic heads, and disentanglement losses.
Optionally supports curriculum scheduling for loss terms.
"""
