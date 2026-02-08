# CLAUDE.md — MenGrowth-Model (Foundation Model → Disentangled Projection → Neural ODE)

This repository implements a **4-phase pipeline** for learning **disentangled latent state vectors** from **multi-modal 3D MRI** (4 channels, 128³) suitable for **continuous-time meningioma growth forecasting** via a **Neural ODE**. The pipeline is built on top of **BrainSegFounder**, a Swin-UNETR foundation model pretrained on 41,400+ brain MRI subjects.

> **Note:** The earlier VAE-based approach (Exp1–3, SemiVAE) has been **superseded** by this foundation model pipeline. See [Section 10](#10-legacy-vae-experiments-lessons-learned) for historical context and lessons learned.

---

## 1) Methodology Overview

### Why Foundation Model Instead of VAE?

The SemiVAE approach (Exp1–3) suffered from fundamental limitations:
- **Posterior collapse**: Persistent across 6+ runs despite curriculum learning, free bits, cyclical annealing
- **Residual collapse**: z_residual variance 0.015 vs target >0.10 — SBD decoder bypassed latent code
- **Factor entanglement**: z_vol↔z_shape correlation reached 0.78 despite cross-partition penalties
- **KL distortion**: KL regularization actively minimizes mutual information I(x; z), destroying rich encoder features
- **Reconstruction overhead**: Reconstructing 128³×4 volumes from 128-dim latents is unnecessary for Neural ODE

The revised pipeline **discards the VAE entirely** and uses a **Supervised Disentangled Projection (SDP)** on top of a pretrained foundation encoder. This eliminates posterior collapse risk, removes reconstruction overhead, and directly optimizes for the properties the Neural ODE needs.

### Pipeline Overview

```
Phase 1: Encoder Adaptation (BraTS-MEN, 1000 subjects)
┌──────────────┐    ┌────────────┐    ┌────────────┐
│ BraTS-MEN MRI│───→│ SwinViT    │───→│ Seg Head   │───→ L_dice + L_CE
│ [B,4,128³]   │    │ (LoRA r=8) │    │ (discard)  │
└──────────────┘    └────────────┘    └────────────┘

Phase 2: Disentangled Projection (BraTS-MEN, 1000 subjects)
┌──────────────┐    ┌────────────┐    ┌─────────┐
│ BraTS-MEN MRI│───→│ SwinViT    │───→│ SDP MLP │───→ z ∈ ℝ^128
│ [B,4,128³]   │    │ (frozen)   │    │ (train) │    │
└──────────────┘    └────────────┘    └─────────┘    ↓
                      L_sem + L_cov + L_var + L_dCor

Phase 3: Encoding + Harmonization (Private Cohort, 30 patients)
┌──────────────┐    ┌────────────┐    ┌─────────┐    ┌───────┐
│ Private MRI  │───→│ SwinViT    │───→│ SDP MLP │───→│ComBat │→ z*
│ [all t_k]    │    │ (frozen)   │    │(frozen) │    │       │
└──────────────┘    └────────────┘    └─────────┘    └───────┘

Phase 4: Neural ODE (Private Cohort trajectories)
z*(t₀) ──→ ODESolve(f_θ, z*(t₀), t₀, t₁) ──→ ẑ(t₁)
           Gompertz-informed dynamics
```

### Common Data / Tensor Contract
- Input per subject: 4 modalities `["t2f","t1c","t1n","t2w"]` (FLAIR, T1ce, T1, T2) stacked into `x ∈ ℝ^{B×4×128×128×128}`
- **Channel order is critical**: BrainSegFounder expects [FLAIR, T1ce, T1, T2] — wrong order causes near-zero Dice
- Z-score intensity normalization per subject (channel-wise, nonzero voxels)
- Spatial: isotropic 1mm spacing, RAS orientation, CropForeground + 128³ ROI (matching BrainSegFounder fine-tuning)
- Training: CropForeground + RandSpatialCrop(128³); Validation: CropForeground + center crop to 128³
- Inference (full-resolution): sliding window with 128³ patches, 50% overlap, Gaussian blending
- Segmentation labels (input): 0=background, 1=NCR, 2=ED, 3=ET
- Segmentation output (3-ch sigmoid): Ch0=TC, Ch1=WT, Ch2=ET (hierarchical overlapping)

### Module Dependency Chain

```
module_0 (Data) → module_1 (Domain Gap) → module_2 (LoRA) → module_3 (SDP) → module_4 (Encoding) → module_5 (Neural ODE) → module_6 (Evaluation)
```

Each module's outputs are the next module's inputs. Do NOT start a module until all predecessor modules pass their BLOCKING tests.

### Error Recovery Protocol

1. If a **BLOCKING** test fails:
   - Read the failure recovery steps documented in the module spec file
   - Try each recovery step in order
   - If all recovery steps fail, stop and report the failure with diagnostics
2. If a **DIAGNOSTIC** test fails:
   - Log the warning with full metrics
   - Continue to the next step
3. Never skip a BLOCKING test to proceed to the next module

### Reference Documents

- **`docs/Methods/claude_files_BSGNeuralODE/DECISIONS.md`** — 15 resolved design choices with rationale
- **`docs/Methods/claude_files_BSGNeuralODE/module_*.md`** — Per-module task specifications

---

## 2) Phase 1: LoRA Encoder Adaptation

**Goal:** Adapt BrainSegFounder encoder from glioma to meningioma morphology while preserving low-level anatomy features.

### BrainSegFounder Encoder

The Swin Vision Transformer encoder (`model.swinViT`) from BrainSegFounder provides:
- SSL pretraining on 41,400 UK Biobank subjects (brain anatomy)
- SSL pretraining on 1,251 BraTS 2021 subjects (tumor pathology)
- Supervised segmentation fine-tuning on BraTS 2021 (tumor discrimination)

Architecture (SwinUNETR from MONAI):
```
Input: [B, 4, 128, 128, 128]
Stage 0: [B, 48,  64, 64, 64]   ← Patch embed + 2× Swin Blocks
Stage 1: [B, 96,  32, 32, 32]   ← Patch merge + 2× Swin Blocks
Stage 2: [B, 192, 16, 16, 16]   ← Patch merge + 2× Swin Blocks
Stage 3: [B, 384,  8,  8,  8]   ← Patch merge + 2× Swin Blocks
Stage 4: [B, 768,  4,  4,  4]   ← 2× Swin Blocks (deepest)
```

Feature extraction: `h = GAP(Stage4(x)) ∈ ℝ^768` (encoder10 level).

### LoRA Configuration

Apply Low-Rank Adaptation to Stages 3–4 of the Swin Transformer:
- **Target modules**: Q, K, V projection matrices in all self-attention layers of Stages 3 and 4
- **Default rank**: r=8, alpha=16 (effective scaling α/r = 2.0)
- **Frozen**: Patch embedding, Stages 0–2 (preserve low-level anatomy features)
- **Trainable parameters**: ~197K (LoRA) + decoder parameters
- **DoRA variant**: Weight-Decomposed LoRA also supported (`use_dora: true`)

### BrainSegFounder Channel Convention (TC/WT/ET)

BrainSegFounder uses MONAI's standard BraTS 3-channel sigmoid output:
- **Ch0: TC** (Tumor Core) = NCR ∪ ET = `(label==1) | (label==3)`
- **Ch1: WT** (Whole Tumor) = NCR ∪ ED ∪ ET = `(label==1) | (label==2) | (label==3)`
- **Ch2: ET** (Enhancing Tumor) = `(label==3)`

These are **hierarchical overlapping regions**, NOT individual labels. The `_convert_target()` methods in `src/growth/losses/segmentation.py` handle this conversion.

### Decoder Options

Controlled by `decoder_type` in config:
- **`"original"`** (recommended): Full pretrained SwinUNETR decoder (~30M params). Provides stronger gradients for encoder adaptation.
- **`"lightweight"`**: Custom SegmentationHead (~2M params). Faster but limited gradient quality.

### Optional Auxiliary Semantic Heads

When `use_semantic_heads: true`, auxiliary regression heads predict volume, location, and shape features from encoder bottleneck during segmentation training. This provides multi-task learning signal that improves feature quality for downstream SDP.

Configuration:
```yaml
training:
  use_semantic_heads: true
  lambda_aux: 0.1
  aux_warmup_epochs: 5        # Delay semantic loss
  aux_warmup_duration: 10     # Ramp duration
loss:
  lambda_volume: 1.0
  lambda_location: 1.0
  lambda_shape: 0.5           # Shape is harder to predict
```

### Post-Phase 1

1. Merge LoRA weights into base weights: `W_merged = W_pretrained + B_r × A_r`
2. Discard the segmentation head entirely
3. Freeze all encoder parameters
4. Save the merged encoder checkpoint

---

## 3) Phase 2: Supervised Disentangled Projection (SDP)

**Goal:** Project frozen encoder features to a partitioned, disentangled latent space with semantic alignment — without any generative component.

### SDP Architecture

```
Frozen SwinViT encoder10: [B, 768, 4, 4, 4]
       ↓
AdaptiveAvgPool3d(1): [B, 768]
       ↓
LayerNorm(768)
       ↓
Linear(768, 512) → GELU → Dropout(0.1)
       ↓
SpectralNorm(Linear(512, 128))
       ↓
z ∈ ℝ^128 = [z_vol(24) | z_loc(8) | z_shape(12) | z_residual(84)]
       ↓                ↓               ↓
   π_vol(24→4)     π_loc(8→3)     π_shape(12→3)
```

Spectral normalization on ALL SDP linear layers ensures Lipschitz continuity (see DECISIONS.md D13), which propagates to Neural ODE dynamics.

Total trainable parameters: ~500K (projection MLP) + ~3K (semantic heads) ≈ 503K.

### Latent Space Partitioning

| Partition | Dims | Indices | Target Features |
|-----------|------|---------|-----------------|
| z_vol | 24 | 0–23 | V_total, V_NCR, V_ED, V_ET (log-transformed) |
| z_loc | 8 | 24–31 | Centroid c_x, c_y, c_z |
| z_shape | 12 | 32–43 | Sphericity, surface_area_log, solidity (3 targets) |
| z_residual | 84 | 44–127 | Unsupervised (texture, context, scanner) |

### Loss Function

```
L_SDP = Σ_p λ_p·L_sem_p + λ_cov·L_cov + λ_var·L_var + λ_dCor·Σ_{i<j} dCor(z_i, z_j)
```

Four complementary mechanisms replace all VAE-based disentanglement:

1. **Semantic Regression** (informativeness): `L_sem_p = (1/k_p) ||π_p(z_p) - y_p||²` for each partition p ∈ {vol, loc, shape}
2. **VICReg Covariance** (linear independence): Penalizes cross-partition off-diagonal covariance entries only. Within-partition correlations are free.
3. **Variance Preservation** (collapse prevention): Hinge loss ensuring each dimension maintains std dev ≥ γ=1.0
4. **Distance Correlation** (nonlinear independence): `dCor(z_i, z_j) = 0 ⟺ z_i ⊥⊥ z_j` — strictly stronger than zero covariance

### Semantic Target Features (from segmentation masks)

Computed by `src/growth/data/semantic_features.py`:
- **Volumes** (mm³): Total tumor, NCR, ED, ET — log-transformed: log(V+1)
- **Location**: Center of mass (3D), normalized or physical coordinates
- **Shape**: Sphericity, surface_area_log, solidity (3 features) — via scipy. See DECISIONS.md D12.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| λ_vol | 20.0 | Highest priority (Gompertz proxy) |
| λ_loc | 12.0 | Moderate priority |
| λ_shape | 15.0 | High priority |
| λ_cov | 5.0 | VICReg default scale |
| λ_var | 5.0 | VICReg default scale |
| λ_dCor | 2.0 | Moderate nonlinear penalty |
| Epochs | 100 | Single-phase, converges fast |
| Batch size | 64 | Larger batches improve covariance/dCor estimates |

### Quality Targets

| Metric | Target |
|--------|--------|
| Vol R² | ≥ 0.90 |
| Loc R² | ≥ 0.95 |
| Shape R² | ≥ 0.40 |
| Max cross-partition correlation | < 0.20 |
| Variance per dimension | > 0.5 |
| dCor(vol, loc) | < 0.10 |

---

## 4) Phase 3: Encoding + Harmonization

### Inference Protocol

After Phase 2, freeze both encoder and SDP. For each volume: `z = g_φ(GAP(SwinViT(x))) ∈ ℝ^128` (deterministic, no sampling).

### Sliding-Window Encoding (for volumes larger than 128³)

1. Extract overlapping 128³ patches with stride 64
2. Encode each patch through frozen SwinViT
3. Pool features with tumor-weighted averaging: weight each patch proportionally to its overlap with the segmentation mask
4. Project pooled features through frozen SDP

### ComBat Harmonization

Standard ComBat at the latent level to correct scanner/site effects. Site correction parameters are constant across timepoints, so intra-patient temporal dynamics are preserved: `Δz* = Δz / δ_site`.

**Recommendation**: Before applying ComBat, visualize BraTS-MEN and private cohort latent distributions via UMAP. If they overlap substantially, skip ComBat.

---

## 5) Phase 4: Neural ODE

**Goal:** Continuous-time evolution in latent space: `dz(t)/dt = f_θ(z(t), t)`

### Architecture (Gompertz-informed, partition-aware)

**Volume partition** (dims 0–23):
```
dz_vol/dt = α · z_vol ⊙ ln(K / (z_vol + ε)) + η_vol · h_θ(z)
```
- α ∈ ℝ⁺ (softplus): learned growth rate
- K ∈ ℝ²⁴₊ (softplus): carrying capacities
- η_vol = 0.01: neural correction scale
- h_θ: 2-layer MLP (128 → 64 → 24)

**Location partition** (dims 24–31): `dz_loc/dt = η_loc · MLP_loc(z_vol, z_loc)`, η_loc = 0.01

**Shape partition** (dims 32–43): `dz_shape/dt = η_shape · MLP_shape(z_vol, z_shape)`, η_shape = 0.01

**Residual partition** (dims 44–127): `dz_res/dt = η_res · MLP_res(z)`, η_res = 0.001

### Data Augmentation

Construct all pairwise temporal combinations from each patient's trajectory (forward + reverse). With 30 patients averaging 3.5 timepoints: ~270 training pairs.

### Training Objective

```
L_ODE = Σ ||z*(t₁) - ẑ(t₁)||² + λ_reg·||θ_ODE||² + λ_smooth·∫||d²z/dt²||² dt
```

Uses `torchdiffeq` adjoint method for memory-efficient backprop.

---

## 6) Architecture Overview

### Model Instantiation Flow

```
YAML Config (foundation.yaml + phase-specific override)
    ↓
Phase 1: train_lora.py
    ↓
    LoRALitModule
        ├─ swin_loader.load_swin_encoder() or load_full_swinunetr()
        ├─ lora_adapter.LoRASwinViT (LoRA/DoRA on stages 3-4)
        ├─ Decoder: original_decoder or seg_head
        └─ Optional: semantic_heads (auxiliary vol/loc/shape regression)

Phase 2: train_sdp.py
    ↓
    SDPLitModule
        ├─ Frozen merged encoder (from Phase 1)
        ├─ feature_extractor.FeatureExtractor (GAP on encoder10)
        ├─ projection/sdp.SDP (768 → 512 → 128, spectral norm)
        ├─ projection/partition.LatentPartition (split z into partitions)
        └─ projection/semantic_heads (partition → target regression)

Phase 4: train_ode.py
    ↓
    ODELitModule
        ├─ ode/gompertz.GompertzDynamics (volume partition)
        ├─ ode/partition_ode.PartitionODE (full partition-aware dynamics)
        └─ ode/ode_func.ODEFunc (torchdiffeq wrapper)
```

### Feature Extraction Levels

| Level | Dimensions | Source |
|-------|-----------|--------|
| `encoder10` | 768 | Bottleneck (default) |
| `layers4` | 768 | Raw SwinViT stage 4 |
| `multi_scale` | 1344 | layers2(192) + layers3(384) + layers4(768) |

---

## 7) Codebase Map

```
src/
├── growth/                              # Main pipeline (current approach)
│   ├── config/
│   │   ├── foundation.yaml              # Base config (paths, encoder, training)
│   │   ├── phase1_lora.yaml             # Phase 1: LoRA adaptation
│   │   ├── phase2_sdp.yaml              # Phase 2: SDP training
│   │   ├── phase3_encode.yaml           # Phase 3: Encoding cohort
│   │   ├── phase4_ode.yaml              # Phase 4: Neural ODE
│   │   └── server/
│   │       └── foundation_icai.yaml     # Server-specific paths
│   ├── models/
│   │   ├── encoder/
│   │   │   ├── swin_loader.py           # BrainSegFounder checkpoint loading
│   │   │   ├── lora_adapter.py          # LoRA/DoRA application to SwinViT
│   │   │   └── feature_extractor.py     # Multi-scale feature extraction (GAP)
│   │   ├── segmentation/
│   │   │   ├── original_decoder.py      # Full SwinUNETR decoder (Phase 1)
│   │   │   ├── seg_head.py              # Lightweight segmentation head
│   │   │   └── semantic_heads.py        # Auxiliary vol/loc/shape heads
│   │   ├── projection/
│   │   │   ├── sdp.py                   # SDP network (2-layer MLP, spectral norm)
│   │   │   ├── partition.py             # Latent space partitioning
│   │   │   └── semantic_heads.py        # SDP semantic heads
│   │   ├── ode/
│   │   │   ├── gompertz.py              # Gompertz growth dynamics
│   │   │   ├── partition_ode.py         # Partition-aware Neural ODE
│   │   │   └── ode_func.py             # torchdiffeq wrapper
│   │   └── factory.py                   # Model creation factory
│   ├── losses/
│   │   ├── segmentation.py              # Dice + CE loss (MONAI DiceCELoss)
│   │   ├── semantic.py                  # MSE regression losses
│   │   ├── sdp_loss.py                  # SDP composite loss
│   │   ├── vicreg.py                    # VICReg disentanglement
│   │   ├── dcor.py                      # Distance correlation
│   │   └── ode_loss.py                  # Trajectory matching + smoothness
│   ├── data/
│   │   ├── bratsmendata.py              # BraTS-MEN dataset loader
│   │   ├── semantic_features.py         # Feature extraction (vol, loc, shape)
│   │   ├── transforms.py               # MONAI data augmentation
│   │   ├── trajectory.py               # Trajectory pairs for ODE
│   │   └── longitudinal.py             # Longitudinal cohort loader
│   ├── training/
│   │   ├── train_lora.py                # Phase 1 entry point
│   │   ├── train_sdp.py                 # Phase 2 entry point
│   │   ├── train_ode.py                 # Phase 4 entry point
│   │   ├── lit_modules/
│   │   │   ├── lora_module.py           # Phase 1 Lightning module
│   │   │   ├── sdp_module.py            # Phase 2 Lightning module
│   │   │   └── ode_module.py            # Phase 4 Lightning module
│   │   └── callbacks/
│   │       ├── semantic_metrics.py      # Semantic feature tracking
│   │       ├── disentanglement.py       # Disentanglement metrics
│   │       └── latent_viz.py            # Latent space visualization
│   ├── evaluation/
│   │   ├── enhanced_probes.py           # Ridge + MLP probing
│   │   ├── latent_quality.py            # Linear separability, AU metrics
│   │   ├── segmentation_metrics.py      # Dice, Hausdorff
│   │   ├── ode_metrics.py               # ODE trajectory quality
│   │   ├── risk_stratification.py       # Clinical outcome prediction
│   │   ├── statistics.py                # Statistical tests
│   │   └── visualization.py             # Result visualization
│   ├── inference/
│   │   ├── encode.py                    # Batch encoding of new subjects
│   │   ├── harmonization.py             # ComBat scanner harmonization
│   │   └── sliding_window.py            # Inference on large volumes
│   └── utils/
│       ├── config.py                    # Config loading
│       ├── checkpoint.py                # Checkpoint utilities
│       ├── seed.py                      # Reproducibility seeding
│       ├── reproducibility.py           # Reproducibility tracking
│       ├── paths.py                     # Path utilities
│       ├── model_card.py                # Model metadata
│       └── logging.py                   # Logging utilities
│
├── engine/                              # Legacy VAE entry points
│   ├── train.py                         # VAE routing (Exp1-3)
│   ├── model_factory.py                 # VAE model instantiation
│   └── plot_training_dashboard.py       # Dashboard generation
│
├── vae/                                 # Legacy VAE code (Exp1-3)
│   ├── config/                          # vae.yaml, dipvae.yaml, semivae.yaml
│   ├── data/                            # MONAI transforms, dataloaders
│   ├── losses/                          # ELBO, DIP-VAE, SemiVAE losses
│   ├── models/                          # VAE architectures (BaselineVAE, VAESBD)
│   ├── metrics/                         # PSNR, SSIM, AU, probes
│   └── training/                        # VAE LitModules + callbacks
│
experiments/
├── lora_ablation/                       # LoRA rank comparison experiment
│   ├── config/
│   │   ├── ablation.yaml                # Unified ablation config
│   │   └── server/                      # Server-specific configs
│   │       ├── LoRA_semantic_heads_icai.yaml
│   │       ├── DoRA_semantic_heads_icai.yaml
│   │       ├── LoRA_no_semantic_heads_icai.yaml
│   │       └── DoRA_no_semantic_heads_icai.yaml
│   ├── run_ablation.py                  # Main orchestrator
│   ├── train_condition.py               # Single condition training loop
│   ├── model_factory.py                 # Experiment model instantiation
│   ├── extract_features.py              # Feature extraction from trained models
│   ├── evaluate_probes.py               # Linear + MLP probe evaluation
│   ├── evaluate_dice.py                 # Dice metric evaluation
│   ├── analyze_results.py              # Cross-condition analysis
│   ├── post_hoc_analysis.py             # Detailed result breakdown
│   ├── generate_tables.py              # Publication-ready CSV/LaTeX
│   ├── visualizations.py               # Figure generation
│   ├── domain_visualizations.py        # Glioma vs meningioma analysis
│   ├── extract_domain_features.py      # Domain shift evaluation
│   ├── enhanced_diagnostics.py         # Model diagnostics
│   ├── statistical_analysis.py         # Significance testing
│   ├── diagnose_frozen_gli.py          # Frozen model diagnostic (GLI/MEN)
│   ├── data_splits.py                  # Data split management
│   └── output_paths.py                 # Consistent directory structure
└── utils/
    └── settings.py

slurm/
├── execute_experiment1.sh               # Legacy VAE Exp1
└── execute_experiment2_dip.sh           # Legacy DIP-VAE
```

---

## 8) LoRA Ablation Experiment

The `experiments/lora_ablation/` directory contains a systematic ablation study to determine the optimal LoRA rank and validate that domain adaptation is necessary for meningioma-specific features.

### Conditions

| Condition | Trainable Params | Description |
|-----------|-----------------|-------------|
| baseline_frozen | 0 | Original BrainSegFounder (test only) |
| baseline | ~30M (decoder only) | Frozen encoder + trainable decoder |
| lora_r2 | ~49K + decoder | LoRA rank 2 |
| lora_r4 | ~98K + decoder | LoRA rank 4 |
| lora_r8 | ~197K + decoder | LoRA rank 8 (recommended) |
| lora_r16 | ~393K + decoder | LoRA rank 16 |
| lora_r32 | ~786K + decoder | LoRA rank 32 |

### Configuration Variants

- **v1**: Lightweight decoder, encoder10 features, linear probes only
- **v2** (recommended): Original SwinUNETR decoder, multi-scale features, Ridge + MLP probes, semantic heads

### Evaluation

- **Primary metric**: Linear probe R² on volume, location, shape features
- **Secondary**: MLP probe R² (captures nonlinear encoding), Dice scores, feature variance
- **Diagnostic**: Nonlinearity gap (MLP R² - Linear R²), UMAP visualizations

### Data Splits

```yaml
# Ablation experiment (v2 server configs):
lora_train: 250      # LoRA training
lora_val: 50         # Early stopping
probe_train: 200     # Probe training (separate from LoRA)
test: 500            # Final evaluation
```

Note: actual server configs in `experiments/lora_ablation/config/server/` may use different splits (e.g., 525/100/225/150 for the main pipeline).

### Key Finding

The frozen BrainSegFounder encoder (trained on gliomas) produces features where meningioma semantics are not linearly accessible (negative R²). LoRA adaptation with r=8–16 is necessary and sufficient for making semantic information linearly predictable.

---

## 9) Libraries and Best Practices

### Libraries

| Library | Role |
|---------|------|
| **PyTorch 2.0+** | Models, losses, AMP-safe kernels |
| **MONAI 1.3+** | NIfTI loading, transforms, SwinUNETR, DiceCELoss |
| **PyTorch Lightning 2.0+** | Training loop, checkpointing, logging |
| **OmegaConf 2.3+** | Hierarchical configuration |
| **torchdiffeq** | Neural ODE solvers (adjoint method) |
| **scipy** | Convex hull, center of mass for semantic features |

Guiding rule: **MONAI for data + SwinUNETR**, **PyTorch for core ML**, **Lightning for training loop + logging**.

### Configuration Management

- Base config: `src/growth/config/foundation.yaml` (paths, encoder, training defaults)
- Phase-specific overrides: `phase1_lora.yaml`, `phase2_sdp.yaml`, `phase3_encode.yaml`, `phase4_ode.yaml`
- Server-specific: `config/server/` subdirectories
- **Required encoder params**: `feature_size: 48`, `in_channels: 4`, `depths: [2,2,2,2]`, `num_heads: [3,6,12,24]`
- **ROI size**: 128×128×128 (matching BrainSegFounder fine-tuning)

### Numerical Stability

- Gradient clipping: `gradient_clip_val: 1.0`
- Mixed precision: `bf16-mixed` (preferred) or `16-mixed`
- Spectral normalization on ALL SDP linear layers for Lipschitz continuity (D13)
- Semantic targets normalized to μ=0, σ=1 during training

### Logging Keys

- Phase 1: `train/dice`, `train/ce`, `val/dice`, `train/vol_loss`, `train/loc_loss`, `train/shape_loss`
- Phase 2: `train/sem_vol`, `train/sem_loc`, `train/sem_shape`, `train/cov`, `train/var`, `train/dcor`
- Phase 4: `train/trajectory_mse`, `train/smoothness`, `val/volume_r2`

### Reproducibility

- `seed_everything(seed, workers=True)` (default seed: 42)
- Save resolved config in run directory
- Model cards with training metadata
- Separate data splits for training vs. probe evaluation

---

## 10) Legacy VAE Experiments (Lessons Learned)

The VAE approach (Exp1–3, `src/vae/`) is preserved for reference but is no longer the active methodology. Key lessons that informed the current pipeline:

### Exp1 (Baseline VAE)
- Stable reconstruction but entangled latents. Posterior collapse mitigated via cyclical annealing + free bits.

### Exp2 (DIP-VAE-II)
- SBD decoder caused complete posterior collapse in 3D medical imaging (decoder ignores z, relies on coordinates).
- Standard decoder with delayed DIP penalties worked better, but unsupervised disentanglement is insufficient.

### Exp3 (SemiVAE, 6 runs)
- **Run 1** (300 epochs): Residual collapse (1.3% AU), SBD bypass, shape R²=0.22
- **Run 4** (800 epochs, curriculum): 109 AU total but residual variance 0.015. z_vol↔z_shape correlation reached 0.78.
- **Key insight**: Reconstruction objective and KL regularization are fundamentally misaligned with Neural ODE requirements. The VAE actively minimizes mutual information I(x; z), destroying the encoder's learned features.
- **Decision**: Discard VAE entirely in favor of SDP approach that directly optimizes for ODE-relevant properties.

---

## 11) Key References

**Foundation Model Pipeline:**
- Cox et al. (2024). "BrainFounder: Towards Brain Foundation Models for Neuroimage Analysis." Medical Image Analysis.
- Hatamizadeh et al. (2022). "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." BrainLes, MICCAI.
- Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
- Bardes et al. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR.
- Miyato et al. (2018). "Spectral Normalization for Generative Adversarial Networks." ICLR.
- Székely et al. (2007). "Measuring and Testing Dependence by Correlation of Distances." Annals of Statistics.

**Neural ODE:**
- Chen et al. (2018). "Neural Ordinary Differential Equations." NeurIPS.
- Rubanova et al. (2019). "Latent ODEs for Irregularly-Sampled Time Series." NeurIPS.
- Benzekry et al. (2014). "Classical Mathematical Models for Description and Prediction of Experimental Tumor Growth." PLOS Computational Biology.

**Disentanglement Theory:**
- Locatello et al. (2019). "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations." ICML.
- Locatello et al. (2020). "Disentangling Factors of Variation Using Few Labels." ICLR.
- Higgins et al. (2018). "Towards a Definition of Disentangled Representations." arXiv:1812.02230.

**Data:**
- LaBella et al. (2024). "The ASNR-MICCAI BraTS Meningioma Challenge." arXiv.
- Johnson et al. (2007). "Adjusting batch effects in microarray expression data using empirical Bayes methods." Biostatistics.
