---
name: explore
description: Deep codebase exploration in isolated context
---

# MenGrowth Codebase Exploration

Thoroughly explore the MenGrowth-Model codebase to answer the query.

## Key Locations
- `src/growth/models/encoder/` -- SwinUNETR loader, LoRA adapter, feature extractor
- `src/growth/models/projection/` -- SDP network, partition, semantic heads
- `src/growth/models/growth/` -- GP models (LME, H-GP, PA-MOGP, ScalarGP)
- `src/growth/data/` -- BraTS-MEN/GLI loaders, transforms, semantic features
- `src/growth/losses/` -- Dice/CE segmentation, SDP composite, VICReg, dCor
- `src/growth/evaluation/` -- GP probes, latent quality, LOPO evaluator
- `experiments/lora/` -- Phase 1 LoRA adaptation (complete)
- `experiments/sdp/` -- Phase 2 SDP training
- `experiments/segment_based_approach/` -- Ablation A0 baseline
- `docs/growth-related/claude_files_BSGNeuralODE/` -- Per-module specifications
- `docs/growth-related/methodology_refined.md` -- Master methodology document

## Critical Conventions
- Channel order: `["t2f", "t1c", "t1n", "t2w"]` (FLAIR, T1ce, T1, T2)
- ROI: 128^3 (training), 192^3 (feature extraction)
- Segmentation: TC/WT/ET (3-ch sigmoid, hierarchical overlapping)
- GP models replace Neural ODE (see DECISIONS.md D16)

$ARGUMENTS
