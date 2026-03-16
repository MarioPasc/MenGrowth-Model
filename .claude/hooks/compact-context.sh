#!/usr/bin/env bash
# Compaction recovery hook — re-injects critical context after /compact

cat <<'CONTEXT'
=== MENGROWTH COMPACTION RECOVERY ===

CHANNEL ORDER (CRITICAL):
  [FLAIR, T1ce, T1, T2] = ["t2f", "t1c", "t1n", "t2w"]
  Wrong order → Dice ~0.00. See transforms.py MODALITY_KEYS.

ROI SIZE: 128³ (training), 192³ (feature extraction/eval).
  encoder10 → [B, 768, 4, 4, 4] with 128³ input.

SEGMENTATION: 3-ch sigmoid — Ch0=TC, Ch1=WT, Ch2=ET (overlapping).
  Input labels: 0=BG, 1=NCR, 2=ED, 3=ET.

ENVIRONMENT:
  Conda: ~/.conda/envs/growth/bin
  Tests: ~/.conda/envs/growth/bin/python -m pytest tests/ -v
  Safe default: pytest -m "not slow and not real_data" -v --tb=short

PIPELINE PHASES:
  [X] Phase 0: Data Infrastructure (BraTSDatasetH5, HDF5 v2.0)
  [X] Phase 1: LoRA Adaptation (experiments/lora/, Dice WT ~0.87)
  [X] Phase 2: SDP (128-d latent, vol/loc/shape/residual partitions)
  [X] Phase 3: Encoding + ComBat (infrastructure ready)
  [~] Phase 4: Growth Prediction (LME ✓, H-GP ✓, PA-MOGP STUBBED)
  [~] Phase 5: Evaluation (GP probes ✓, ablation A0 in progress)

GP HIERARCHY (replaces Neural ODE — D16):
  LME → H-GP → PA-MOGP, LOPO-CV 33 folds
  z_vol (24-dim), population linear mean from LME (D18)

KEY DECISIONS:
  D16: GP hierarchy (not ODE — overparameterization)
  D17: LOPO-CV (33 folds)
  D18: Population linear mean from LME
  D19: Hierarchical hyperparameter sharing
  D20: Rank-1 cross-partition coupling (PA-MOGP)

SLASH COMMANDS:
  /implement-phase N — implement a pipeline phase
  /run-tests N — run phase-specific tests
  /check-gate N — check if phase gate is OPEN
  /pre-flight <config> — validate before SLURM submission
  /analyze-run <dir> — analyze experiment results
  /dl-scientist — rigorous scientific analysis
  /explore — codebase exploration
  /test — quick test runner

KEY FILES:
  docs/growth-related/claude_files_BSGNeuralODE/ — Module specs
  docs/growth-related/methodology_refined.md — Master methodology
  docs/technical_report/main.tex — LaTeX technical report hub
  src/growth/losses/segmentation.py — TC/WT/ET conversion
  src/growth/data/transforms.py — Channel order, preprocessing

CODING RULES:
  1. Type hints on ALL function signatures
  2. Google-style docstrings on public functions/classes
  3. No magic numbers — config from YAML via OmegaConf
  4. Use MONAI transforms, einops.rearrange
  5. No BatchNorm — LayerNorm only
  6. Shape assertions at tensor function boundaries
  7. Logging via Python logging module (no print)

=== END COMPACTION RECOVERY ===
CONTEXT
