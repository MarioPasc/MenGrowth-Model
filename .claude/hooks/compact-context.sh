#!/usr/bin/env bash
# Compaction recovery hook — re-injects critical context after /compact

cat <<'CONTEXT'
=== MENGROWTH COMPACTION RECOVERY ===

CHANNEL ORDER (CRITICAL):
  [FLAIR, T1ce, T1, T2] = ["t2f", "t1c", "t1n", "t2w"]
  Wrong order → Dice ~0.00. See transforms.py MODALITY_KEYS.

ROI SIZE: 128³ (NOT 96³). encoder10 → [B, 768, 4, 4, 4].

SEGMENTATION: 3-ch sigmoid — Ch0=TC, Ch1=WT, Ch2=ET (overlapping).

ENVIRONMENT:
  Conda: ~/.conda/envs/growth/bin
  Tests: ~/.conda/envs/growth/bin/python -m pytest tests/ -v

PIPELINE STATUS:
  [X] Phase 1: LoRA Adaptation (complete, in experiments/lora_ablation/)
  [ ] Phase 2: SDP (stubbed, spec in module_3_sdp.md)
  [ ] Phase 3: Encoding + ComBat (not started, spec in module_4_encoding.md)
  [ ] Phase 4: Neural ODE (stubbed, spec in module_5_neural_ode.md)

CODING RULES:
  1. Type hints on ALL function signatures
  2. Google-style docstrings on public functions/classes
  3. No magic numbers — config from YAML via OmegaConf
  4. Use MONAI transforms, einops.rearrange
  5. No BatchNorm — LayerNorm only
  6. Shape assertions at tensor function boundaries

KEY FILES:
  src/growth/losses/segmentation.py — TC/WT/ET conversion
  src/growth/data/transforms.py — Channel order, preprocessing
  docs/Methods/claude_files_BSGNeuralODE/ — Module specifications

=== END COMPACTION RECOVERY ===
CONTEXT
