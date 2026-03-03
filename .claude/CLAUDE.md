# MenGrowth-Model

Foundation model pipeline for **meningioma growth forecasting** from multi-modal 3D MRI.
BrainSegFounder (SwinUNETR, pretrained on 41K+ subjects) → LoRA adaptation → Supervised Disentangled Projection → GP-based growth prediction (LME → H-GP → PA-MOGP).
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
- **Phase 4 (Growth Prediction)**: STUBBED — per module_5_growth_prediction spec (LME → H-GP → PA-MOGP, LOPO-CV)

## Dataset: HDF5

The pipeline uses a single HDF5 backend (`BraTSDatasetH5`, alias `BraTSMENDatasetH5`) that supports two schemas:

- **Cross-sectional** (BraTS-MEN): `subject_ids` dataset, one scan per subject.
- **Longitudinal** (BraTS-GLI): `scan_ids` + `patient_ids` + CSR `longitudinal/` group. Patient-level splits are expanded to scan indices via CSR offsets.

Schema auto-detected by presence of `scan_ids` (longitudinal) vs `subject_ids` (cross-sectional).

All three datasets (BraTS-MEN, BraTS-GLI, MenGrowth) use `BraTSDatasetH5` — no NIfTI loaders remain.

Convert NIfTI → H5:
- MEN: `python scripts/convert_brats_men_to_h5.py --data-root /path/to/BraTS_Men_Train --output brats_men_train.h5`
- GLI: `python scripts/convert_brats_gli_to_h5.py --data-root /path/to/BraTS_GLI --output brats_gli.h5`
- MenGrowth: `python scripts/convert_mengrowth_to_h5.py --data-root /path/to/MenGrowth-2025 --output mengrowth.h5`

H5 schema (cross-sectional): `images [N,4,192,192,192]`, `segs [N,1,192,192,192]`, `subject_ids [N]`, `semantic/{volume,location,shape}`, `splits/{lora_train,lora_val,test}`, `metadata/{grade,age,sex}`.

H5 schema (longitudinal): adds `scan_ids [N]`, `patient_ids [N]`, `timepoint_idx [N]`, `longitudinal/{patient_offsets, patient_list}`. Splits reference `patient_list` indices.

Images are spatially preprocessed but NOT intensity-normalized (normalization at runtime).

H5 transforms: `get_h5_train_transforms()` / `get_h5_val_transforms()` in `transforms.py`. Config key: `paths.h5_file`.

## Critical Conventions (bugs if violated)

- **Channel order**: `["t2f", "t1c", "t1n", "t2w"]` = [FLAIR, T1ce, T1, T2]. Wrong order → Dice ~0.00. Defined in `MODALITY_KEYS` in `src/growth/data/transforms.py`.
- **ROI size**: 128³ (matching BrainSegFounder fine-tuning), NOT 96³. encoder10 output: `[B, 768, 4, 4, 4]`.
- **Segmentation output**: 3-ch sigmoid — Ch0=TC, Ch1=WT, Ch2=ET (hierarchical overlapping, NOT individual labels).
- **Input labels**: 0=background, 1=NCR, 2=ED, 3=ET (+ 4=RC for GLI, merged into NCR for semantic features). Conversion in `segmentation.py._convert_target()`.
- **Preprocessing (H5)**: Volumes pre-preprocessed to 192³. Runtime: NormalizeIntensity → optional RandSpatialCrop(128³) for training.

## Key Libraries

PyTorch 2.0+, MONAI 1.3+, Lightning 2.0+, OmegaConf, peft (LoRA/DoRA), GPy>=1.13, statsmodels>=0.14, scipy

## Codebase Layout

```
src/growth/          # Main pipeline
  config/            # YAML configs (foundation.yaml + phase overrides)
  models/encoder/    # swin_loader, lora_adapter, feature_extractor
  models/projection/ # SDP (sdp.py, partition.py, semantic_heads.py)
  models/growth/     # GP growth models (LME, H-GP, PA-MOGP, volume decoder)
  losses/            # Dice/CE segmentation, SDP composite
  data/              # BraTS-MEN loader, transforms, semantic features
  training/          # Lightning modules + entry points (train_lora, train_sdp)
  evaluation/        # Probes, metrics, visualization
  inference/         # Sliding window, ComBat harmonization

experiments/lora_ablation/  # Phase 1 ablation (complete)
experiments/sdp/            # Phase 2 SDP (feature extraction + training)
scripts/                    # Utilities (convert_brats_men_to_h5.py, convert_brats_gli_to_h5.py, convert_mengrowth_to_h5.py)
slurm/sdp/                  # SLURM jobs for Picasso cluster
src/vae/                    # Legacy VAE code (Exp1-3, superseded)
```

## Module Dependency Chain

```
module_0 (Data) → module_1 (Domain Gap) → module_2 (LoRA) → module_3 (SDP)
→ module_4 (Encoding) → module_5 (Growth Prediction) → module_6 (Evaluation)
```

## Error Recovery

- **BLOCKING** test fails: follow recovery steps in module spec, stop if all fail
- **DIAGNOSTIC** test fails: log warning, continue

## Legacy Note

The VAE approach (Exp1–3, `src/vae/`) is preserved for reference. It was abandoned due to posterior collapse, residual collapse, and KL distortion. The SDP approach directly optimizes for downstream growth prediction requirements. The Neural ODE approach (original Phase 4) was also abandoned due to catastrophic overparameterization with only 33 patients / 112 forward pairs — replaced by a GP hierarchy (D16). See `docs/growth-related/phase4_pivot_to_gp_models.md`.

## Detailed Specifications (read on demand)

@docs/growth-related/claude_files_BSGNeuralODE/DECISIONS.md
@docs/growth-related/claude_files_BSGNeuralODE/module_0_data.md
@docs/growth-related/claude_files_BSGNeuralODE/module_1_domain_gap.md
@docs/growth-related/claude_files_BSGNeuralODE/module_2_lora.md
@docs/growth-related/claude_files_BSGNeuralODE/module_3_sdp.md
@docs/growth-related/claude_files_BSGNeuralODE/module_4_encoding.md
@docs/growth-related/claude_files_BSGNeuralODE/module_5_growth_prediction.md
@docs/growth-related/claude_files_BSGNeuralODE/module_6_evaluation.md
