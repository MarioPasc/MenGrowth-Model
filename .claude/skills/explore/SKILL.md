---
name: explore
description: Deep codebase exploration in isolated context
---

# MenGrowth Codebase Exploration

Thoroughly explore the MenGrowth-Model codebase to answer the query.

## Framework: 3-Stage Complexity Ladder

- **Stage 1 (Primary):** Volume-only growth prediction (ScalarGP, LME, HGP)
- **Stage 2 (Secondary):** Latent severity model (NLME, quantile transform)
- **Stage 3 (Tertiary):** Representation learning (LoRA → SDP → PCA → GP+ARD)

## Key Locations
- `src/growth/models/growth/` — Growth models: scalar_gp, lme_model, hgp_model, severity_model
- `src/growth/models/encoder/` — SwinUNETR loader, LoRA adapter, feature extractor (Stage 3)
- `src/growth/models/projection/` — SDP network, partition, semantic heads (Stage 3)
- `src/growth/data/` — BraTS-MEN/GLI loaders, transforms, semantic features
- `src/growth/losses/` — Dice/CE segmentation, SDP composite, VICReg, dCor
- `src/growth/evaluation/` — GP probes, latent quality, LOPO evaluator, variance decomposition
- `experiments/segment_based_approach/` — Stage 1 pipeline
- `experiments/severity_model/` — Stage 2 pipeline
- `experiments/lora/` — Stage 3: LoRA adaptation (complete)
- `experiments/sdp/` — Stage 3: SDP training
- `docs/stages/` — Self-contained stage specifications
- `docs/RESEACH_PLAN.md` — Literature synthesis and research foundation
- `docs/PLAN_OF_ACTION_v1.md` — Implementation plan for all 3 stages

## Critical Conventions
- Channel order: `["t2f", "t1c", "t1n", "t2w"]` (FLAIR, T1ce, T1, T2)
- ROI: 128³ (training), 192³ (feature extraction)
- Segmentation: TC/WT/ET (3-ch sigmoid, hierarchical overlapping)
- N=31-58 patients, max 2-3 predictor params at N=31

$ARGUMENTS
