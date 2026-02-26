# CLAUDE.md — BSG Growth Prediction Pipeline

## Project Overview

This BSc thesis pipeline adapts BrainSegFounder (a glioma-pretrained 3D brain MRI foundation model) for meningioma growth forecasting. The pipeline has 4 phases: (1) LoRA-based encoder adaptation using segmentation as proxy task, (2) Supervised Disentangled Projection (SDP) mapping frozen encoder features to a structured 128-d latent space, (3) Cohort encoding and ComBat harmonization of longitudinal Andalusian cohort MRIs, and (4) Growth prediction via a three-model GP hierarchy (LME → Hierarchical GP → Partition-Aware Multi-Output GP) operating on the disentangled latent space, evaluated under LOPO-CV. The master methodology is in `methodology_refined.md`.

## Directory Structure

```
MenGrowth-Model/                          # /home/mpascual/research/code/MenGrowth-Model/
├── src/growth/
│   ├── models/
│   │   ├── encoder/
│   │   │   └── swin_loader.py            # BSF checkpoint loading (356 lines) ← REUSE
│   │   ├── projection/
│   │   │   └── sdp.py                    # SDP stub (8 lines) ← IMPLEMENT
│   │   ├── encoder/
│   │   │   ├── swin_loader.py            # BSF checkpoint loading ← REUSE
│   │   │   ├── lora_adapter.py           # LoRA/DoRA injection ← REUSE
│   │   │   └── feature_extractor.py      # GAP feature extraction ← REUSE
│   │   ├── segmentation/
│   │   │   ├── semantic_heads.py         # Auxiliary vol/loc/shape heads ← REUSE
│   │   │   └── original_decoder.py       # Full SwinUNETR decoder ← REUSE
│   │   ├── growth/                        # GP growth models (LME, H-GP, PA-MOGP) ← IMPLEMENT
│   ├── data/
│   │   ├── semantic_features.py          # 3 shape features via compute_shape_array() (445 lines) ← REUSE
│   │   ├── transforms.py                 # MONAI transforms ← REUSE
│   │   └── bratsmendata.py               # Dataset class ← REUSE/EXTEND
│   ├── losses/                           # SDP losses ← IMPLEMENT
│   ├── harmonization/                    # ComBat ← IMPLEMENT
│   └── utils/
│       └── checkpoint.py                 # extract_encoder_weights() ← REUSE
├── experiments/
│   ├── lora_ablation/                    # Full Phase 1 implementation (~9.4K lines) ← REFERENCE
│   │   ├── model_factory.py
│   │   ├── train_condition.py            # Custom training loop (not Lightning)
│   │   ├── data_splits.py               # Data split management
│   │   └── merge_lora_checkpoint.py      # LoRA merge script
│   └── pipeline/                         # Main pipeline runs ← IMPLEMENT
│   ├── config/
│   │   ├── foundation.yaml               # Base config
│   │   ├── phase1_lora.yaml
│   │   ├── phase2_sdp.yaml
│   │   ├── phase3_encode.yaml
│   │   └── phase4_growth.yaml
└── tests/                                # pytest test suite
```

## Module Dependency Chain

```
module_0 (Data) → module_1 (Domain Gap) → module_2 (LoRA) → module_3 (SDP) → module_4 (Encoding) → module_5 (Growth Prediction) → module_6 (Evaluation)
```

Each module's outputs are the next module's inputs. Do NOT start a module until all predecessor modules pass their BLOCKING tests.

## Coding Conventions

- **Reuse existing code:** Always import from `swin_loader.py`, `semantic_features.py`, `transforms.py` before writing new code
- **PyTorch Lightning** or custom training loops (Phase 1 uses custom loop in `experiments/lora_ablation/train_condition.py`)
- **OmegaConf** for hierarchical YAML configuration
- **MONAI** for medical imaging transforms, losses (`DiceCELoss`), and `SwinUNETR` architecture
- **All tensor shapes** documented in docstrings: `# shape: [B, C, H, W, D]`
- **Type hints** on all public functions
- **Tests** use `pytest` with fixtures; separate smoke tests (synthetic data) from integration tests (real data)
- **No magic numbers:** All hyperparameters in YAML configs, referenced via OmegaConf

## Environment

- Python 3.10+
- PyTorch 2.0+ (CUDA 12.x)
- MONAI 1.3+
- PyTorch Lightning 2.0+
- GPy>=1.13 (Gaussian Process models with ICM multi-output support)
- statsmodels>=0.14 (LME fitting via REML)
- OmegaConf 2.3+
- scipy, scikit-learn, umap-learn, neuroCombat

## Critical Warnings

### Channel Order
```
channel_order: [t2f, t1c, t1n, t2w]  # [FLAIR, T1ce, T1, T2]
```
**Wrong channel order causes near-zero Dice.** This order matches BrainSegFounder's convention. Verified in `swin_loader.py` line 32 and `transforms.py` line 39.

### GPU Memory
- Input: 128³ × 4ch, bf16-mixed precision
- Batch size: 2/GPU for Phase 1 training (3D volumes)
- Batch size: Full-batch (800) for Phase 2 SDP (precomputed features, trivial memory)
- Always use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` for forward passes on volumes

## Data Paths

```yaml
bratsmen_root: /path/to/BraTS_Men_Train          # 1000 subjects
andalusian_root: /path/to/andalusian_cohort       # 42 patients, 137 studies
bsf_checkpoint: /path/to/finetuned_model_fold_0.pt  # BSF-Tiny, 62M params
output_root: /path/to/outputs
```

## Data Splits

| Split | N | Use |
|-------|---|-----|
| `lora_train` | 525 | Phase 1 LoRA training |
| `lora_val` | 100 | Phase 1 early stopping |
| `sdp_train` | 225 | Phase 2 SDP training (never seen during LoRA) |
| `test` | 150 | Final held-out evaluation |

## Error Recovery Protocol

1. If a **BLOCKING** test fails:
   - Read the failure recovery steps documented in the module file
   - Try each recovery step in order
   - If all recovery steps fail, stop and report the failure with diagnostics
2. If a **DIAGNOSTIC** test fails:
   - Log the warning with full metrics
   - Continue to the next step
3. Never skip a BLOCKING test to proceed to the next module

## Reproducibility

```python
from lightning import seed_everything
seed_everything(42, workers=True)
```

Save the resolved OmegaConf config to the run directory for every experiment.

## Reference Documents

- **`DECISIONS.md`** — All resolved design choices (20 decisions with rationale; D8-D9 superseded, D16-D20 added for GP pivot)
- **`methodology_refined.md`** — Full scientific methodology (canonical reference)
- **`module_0_data.md`** through **`module_6_evaluation.md`** — Per-module task specifications
