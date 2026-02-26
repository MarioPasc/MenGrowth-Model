---
name: explore
description: Deep codebase exploration in isolated context
---

# MenGrowth Codebase Exploration

Thoroughly explore the MenGrowth-Model codebase to answer the query.

## Key Locations
- `src/growth/models/` — Encoder (swin_loader, lora_adapter), SDP, ODE
- `src/growth/data/` — BraTS-MEN loader, transforms, semantic features
- `src/growth/losses/` — Dice/CE segmentation, SDP composite
- `src/growth/training/` — Lightning modules, training entry points
- `experiments/lora_ablation/` — Phase 1 ablation experiment (complete)
- `docs/Methods/claude_files_BSGNeuralODE/` — Per-module specifications

## Critical Conventions
- Channel order: `["t2f", "t1c", "t1n", "t2w"]` (FLAIR, T1ce, T1, T2)
- ROI: 128³, segmentation: TC/WT/ET (3-ch sigmoid)

$ARGUMENTS
