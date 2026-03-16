# src/growth/stages/__init__.py
"""Stage-organized code for the 3-stage complexity ladder.

- ``stage1_volumetric``: Segmentation → volume → GP/LME growth prediction (PRIMARY)
- ``stage2_severity``: Latent severity NLME model (SECONDARY)
- ``stage3_latent``: BrainSegFounder → LoRA → SDP → PCA → GP+ARD (TERTIARY)

Each stage must earn its place by outperforming the previous one under LOPO-CV.
"""
