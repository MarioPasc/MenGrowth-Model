"""BraTS meningioma segmentation benchmark.

Benchmarks 5 state-of-the-art models (BraTS25 1st-2nd, BraTS23 1st-3rd)
against our BSF+LoRA ensemble on the 150-patient BraTS-MEN test split.
Models run as Singularity containers on Picasso (DGX / A100).

Reference:
    BraTS Orchestrator — Kofler et al. (2025), arXiv:2506.13807
    BraTS-MEN Dataset — LaBella et al. (2024), Scientific Data 11:496
"""
