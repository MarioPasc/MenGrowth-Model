# MenGrowth-Model

Foundation model pipeline for **meningioma growth forecasting** from multi-modal 3D MRI.
BrainSegFounder (SwinUNETR, pretrained on 41K+ subjects) → LoRA adaptation → Supervised Disentangled Projection → Neural ODE.
B.Sc. thesis project.

>[!IMPORTANT] For full context of the project (only read if asked to gather all context for a complex task), read the file: `docs/growth-related/methodology_refined.md`

## Environment

- Conda: `~/.conda/envs/growth/bin`
- Python: `~/.conda/envs/growth/bin/python`
- Tests: `~/.conda/envs/growth/bin/python -m pytest tests/ -v`

## Pipeline Phases

Phase order: 1 → 2 → 3 → 4. Do NOT start a phase until predecessors pass BLOCKING tests.

- **Phase 1 (LoRA Adaptation)**: COMPLETE — code in `experiments/lora_ablation/`
- **Phase 2 (SDP)**: STUBBED — needs implementation per module_3_sdp spec
- **Phase 3 (Encoding + ComBat)**: NOT STARTED — per module_4_encoding spec
- **Phase 4 (Neural ODE)**: STUBBED — per module_5_neural_ode spec

## Critical Conventions (bugs if violated)

- **Channel order**: `["t2f", "t1c", "t1n", "t2w"]` = [FLAIR, T1ce, T1, T2]. Wrong order → Dice ~0.00. Defined in `MODALITY_KEYS` in `src/growth/data/transforms.py`.
- **ROI size**: 128³ (matching BrainSegFounder fine-tuning), NOT 96³. encoder10 output: `[B, 768, 4, 4, 4]`.
- **Segmentation output**: 3-ch sigmoid — Ch0=TC, Ch1=WT, Ch2=ET (hierarchical overlapping, NOT individual labels).
- **Input labels**: 0=background, 1=NCR, 2=ED, 3=ET. Conversion in `segmentation.py._convert_target()`.
- **Preprocessing**: Training: CropForeground → SpatialPad(128³) → RandSpatialCrop(128³). Validation: center crop. Inference: sliding window 128³ patches, 50% overlap.

## Key Libraries

PyTorch 2.0+, MONAI 1.3+, Lightning 2.0+, OmegaConf, peft (LoRA/DoRA), torchdiffeq, scipy

## Codebase Layout

```
src/growth/          # Main pipeline
  config/            # YAML configs (foundation.yaml + phase overrides)
  models/encoder/    # swin_loader, lora_adapter, feature_extractor
  models/projection/ # SDP (sdp.py, partition.py, semantic_heads.py)
  models/ode/        # Gompertz dynamics, partition ODE
  losses/            # Dice/CE segmentation, SDP composite, ODE loss
  data/              # BraTS-MEN loader, transforms, semantic features
  training/          # Lightning modules + entry points (train_lora, train_sdp, train_ode)
  evaluation/        # Probes, metrics, visualization
  inference/         # Sliding window, ComBat harmonization

experiments/lora_ablation/  # Phase 1 ablation (complete)
src/vae/                    # Legacy VAE code (Exp1-3, superseded)
```

## Module Dependency Chain

```
module_0 (Data) → module_1 (Domain Gap) → module_2 (LoRA) → module_3 (SDP)
→ module_4 (Encoding) → module_5 (Neural ODE) → module_6 (Evaluation)
```

## Error Recovery

- **BLOCKING** test fails: follow recovery steps in module spec, stop if all fail
- **DIAGNOSTIC** test fails: log warning, continue

## Legacy Note

The VAE approach (Exp1–3, `src/vae/`) is preserved for reference. It was abandoned due to posterior collapse, residual collapse, and KL distortion. The SDP approach directly optimizes for Neural ODE requirements.

## Detailed Specifications (read on demand)

@docs/growth-related/claude_files_BSGNeuralODE/DECISIONS.md
@docs/growth-related/claude_files_BSGNeuralODE/module_0_data.md
@docs/growth-related/claude_files_BSGNeuralODE/module_1_domain_gap.md
@docs/growth-related/claude_files_BSGNeuralODE/module_2_lora.md
@docs/growth-related/claude_files_BSGNeuralODE/module_3_sdp.md
@docs/growth-related/claude_files_BSGNeuralODE/module_4_encoding.md
@docs/growth-related/claude_files_BSGNeuralODE/module_5_neural_ode.md
@docs/growth-related/claude_files_BSGNeuralODE/module_6_evaluation.md
