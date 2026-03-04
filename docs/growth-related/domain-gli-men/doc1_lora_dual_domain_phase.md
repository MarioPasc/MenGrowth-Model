# Phase 1: Dual-Domain Mixed-Batch LoRA Adaptation

## Implementation Specification for BrainSegFounder Encoder Adaptation

**Project:** MenGrowth-Model  
**Date:** March 2026  
**Status:** Active implementation task  
**Audience:** Local coding agent with full codebase access  
**Scope:** Everything required to implement, train, and evaluate the dual-domain LoRA adaptation. Self-contained.

---

## 1. Objective

Adapt the BrainSegFounder encoder (SwinUNETR, BSF-Tiny, 62M parameters) to produce semantically structured latent representations for both meningioma and glioma brain tumours simultaneously. The encoder is adapted via LoRA (rank 8) with mixed-batch training on BraTS-MEN (meningioma) and BraTS-GLI (glioma) data drawn from HDF5 files.

The purpose of dual-domain training is twofold:

1. **Maintain glioma competence** in the encoder while acquiring meningioma specialisation. The downstream temporal growth model (Phase 4) will train on glioma longitudinal trajectories and transfer to meningioma. If LoRA adaptation erases glioma features, this transfer will fail.

2. **Increase effective training diversity.** The combined dataset (~1700 studies per epoch) exposes the encoder to substantially more anatomical variation than meningioma alone (750 studies), producing more robust features.

The key scientific question this phase answers: *Can a single LoRA-adapted encoder produce semantically structured features for both tumour types without catastrophic forgetting of either domain?*

---

## 2. Data Infrastructure

### 2.1 HDF5 File Schemas

All data has been converted from NIfTI to HDF5. NIfTI loading is deprecated. All three H5 files use the **unified v2.0 schema**:

#### Unified v2.0 H5 Schema

Every H5 file — regardless of whether the underlying data is cross-sectional or longitudinal — stores the same top-level datasets:

```
attrs:           {n_scans, n_patients, roi_size, spacing, channel_order,
                  version="2.0", dataset_type, domain}
images           [N_scans, 4, 192, 192, 192]  float32
segs             [N_scans, 1, 192, 192, 192]  int8
scan_ids         [N_scans] str
patient_ids      [N_scans] str
timepoint_idx    [N_scans] int32
semantic/        {volume [N,4], location [N,3], shape [N,3]}
longitudinal/    {patient_offsets [N_patients+1] int32 (CSR), patient_list [N_patients] str}
splits/          {lora_train, lora_val, test}  int32 (patient-level indices into patient_list)
metadata/        {grade [int8], age [float32], sex [str]}
```

For cross-sectional data (BraTS-MEN), the longitudinal structure is trivial: each patient has exactly 1 scan at timepoint 0, so `n_scans == n_patients` and `patient_offsets = [0, 1, 2, ..., N]`.

#### Per-Dataset Details

**BraTS_MEN.h5** (meningioma, cross-sectional):

```
attrs: {n_scans=1000, n_patients=1000, roi_size=[192,192,192],
        spacing=[1,1,1], channel_order=[t2f,t1c,t1n,t2w],
        version="2.0", dataset_type="cross-sectional", domain="MEN"}
scan_ids = patient_ids = subject IDs  (1:1 mapping)
timepoint_idx: all zeros
longitudinal/patient_offsets: [0, 1, 2, ..., 1000]  (trivial CSR)
segs: labels {0, 1, 2, 3}
splits/: {lora_train [750], lora_val [100], test [150]}  (patient-level)
```

Label mapping for MEN: 0=background, 1=ET, 2=NET, 3=Cyst.

**BraTS_GLI.h5** (glioma, longitudinal):

```
attrs: {n_scans=1350, n_patients=613, roi_size=[192,192,192],
        spacing=[1,1,1], channel_order=[t2f,t1c,t1n,t2w],
        version="2.0", dataset_type="longitudinal", domain="GLI"}
segs: labels {0, 1, 2, 3, 4}
longitudinal/patient_offsets: CSR with variable scans per patient (1–10)
splits/: {lora_train [430], lora_val [60], test [123]}  (patient-level)
```

Label mapping for GLI: 0=background, 1=NETC, 2=SNFH, 3=ET, 4=RC (Resection Cavity).
Note: label 4 is GLI-specific. Semantic features in the H5 were computed with `merge_rc_into_ncr=True` (label 4 → 1 for feature extraction purposes).

**MenGrowth.h5** (meningioma longitudinal, Andalusian cohort — not used in this phase):

```
attrs: {n_scans=100, n_patients=33, ...,
        dataset_type="longitudinal", domain="MenGrowth"}
segs: labels {0, 1, 2, 3} (no label 4)
longitudinal/patient_offsets: CSR with 2–6 scans per patient
splits/: {lora_train [23], lora_val [3], test [7]}  (patient-level)
```

### 2.2 Remaining Per-Dataset Differences

Since all H5 files now share the unified v2.0 schema, the **only** differences between datasets are in the data values, not the schema structure:

| Property | BraTS_MEN.h5 | BraTS_GLI.h5 |
|---|---|---|
| `dataset_type` attr | `"cross-sectional"` | `"longitudinal"` |
| `domain` attr | `"MEN"` | `"GLI"` |
| Scans per patient | Always 1 (trivial CSR) | 1–10 (variable CSR) |
| Label set | {0, 1, 2, 3} | {0, 1, 2, 3, 4} |
| `n_scans` / `n_patients` | 1000 / 1000 | 1350 / 613 |

The schema is identical: both have `scan_ids`, `patient_ids`, `timepoint_idx`, `longitudinal/`, and patient-level `splits/`. The `BraTSDatasetH5` reader handles both without branching.

### 2.3 Resolving Patient-Level Splits to Scan Indices

All H5 files store splits as patient indices into `patient_list`, not scan indices into `images`. To get the scan indices for a split:

```
patient_indices = f["splits/lora_train"][:]   # e.g. [0, 3, 7, ...]
offsets = f["longitudinal/patient_offsets"][:]  # CSR: [0, 2, 5, 8, ...]
scan_indices = []
for pi in patient_indices:
    scan_indices.extend(range(offsets[pi], offsets[pi + 1]))
```

For cross-sectional MEN data this is a trivial identity mapping (patient i → scan i), but the code path is the same. This ensures all timepoints of a patient go to the same split (no patient-level leakage).

### 2.4 Dataset Sizes (after split expansion)

| Split | BraTS-MEN studies | BraTS-GLI scans (est.) | Combined |
|---|---|---|---|
| lora_train | 750 | ~950 (430 patients × ~2.2 tp) | ~1700 |
| lora_val | 100 | ~130 (60 patients × ~2.2 tp) | ~230 |
| test | 150 | ~270 (123 patients × ~2.2 tp) | ~420 |

---

## 3. GPU Memory and Batch Size

### 3.1 Hardware

NVIDIA A100 40 GB VRAM. Training with bf16 mixed precision (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`).

### 3.2 Memory Budget Analysis

| Component | Estimated VRAM |
|---|---|
| SwinUNETR weights (bf16) | ~124 MB |
| LoRA adapters (rank 8, bf16) | ~1 MB |
| Decoder weights (bf16) | ~60 MB |
| Optimizer states (Adam, fp32 for LoRA + decoder) | ~500 MB |
| CUDA context + buffers | ~500 MB |
| **Subtotal (fixed)** | **~1.2 GB** |
| Per-sample activations (128³, bf16, SwinViT 4 stages) | ~2.0 GB |
| **batch_size=2 activations** | **~4.0 GB** |
| **Total at batch_size=2** | **~5.2 GB** |
| **Headroom** | **~34.8 GB** |

The headroom is large, but **peak memory spikes during the backward pass** (gradient checkpointing is not used by default in SwinUNETR) can consume 2–3× the forward activation memory. Conservative estimate for peak: ~15–20 GB at batch_size=2.

### 3.3 Recommendation

- **Physical batch size: 2** (unchanged from current configuration). This is the safe, validated configuration. Attempting batch_size=3 or 4 is possible but risks OOM on certain volumes and has not been validated.
- **Gradient accumulation steps: 4** (already supported by the v3 training infrastructure).
- **Effective batch size: 8.** This means VICReg statistics (variance, covariance) are computed on 8-sample mini-batches after accumulation, which is adequate for the 768-dim feature space.
- The `grad_accum_steps` parameter is already present in `ablation_v3.yaml`. Set `grad_accum_steps: 4`.

### 3.4 VICReg Batch Size Sensitivity

VICReg variance and covariance estimation improves with batch size (Bardes et al., ICLR 2022). With effective batch_size=8 across two domains, each VICReg step sees approximately 4 MEN + 4 GLI samples. This is sufficient because:

- The variance hinge loss only requires `batch_size ≥ 2` to compute meaningful per-dimension std.
- The covariance penalty computes a 768×768 matrix from 8 samples. The off-diagonal estimates will be noisy but directionally correct. The gradient signal is averaged over many steps per epoch (~212 steps with 1700 samples / effective_bs 8).

---

## 4. Dual-Domain DataLoader

### 4.1 Architecture

The training DataLoader must combine BraTS-MEN and BraTS-GLI into a single iterator that yields mixed-domain batches. The chosen strategy is **ConcatDataset + WeightedRandomSampler** with domain-balanced sampling.

### 4.2 Dataset Class (Unified)

No new dataset class is needed. The existing `BraTSDatasetH5` (in `src/growth/data/bratsmendata.py`) already handles all three H5 files via the unified v2.0 schema:

- It reads `scan_ids`, `patient_ids`, `timepoint_idx`, and `longitudinal/` from every H5 file.
- Patient-level splits are expanded to scan indices via CSR offsets at `__init__` time.
- The `domain` field (read from `f.attrs["domain"]`) is included in every `__getitem__` output: `"domain": "MEN"` or `"domain": "GLI"`.
- The lazy per-worker H5 file handle pattern (`threading.local()`) is already implemented.

For legacy v1.0 MEN files (with `subject_ids` instead of `scan_ids`), the reader falls back to the cross-sectional code path via auto-detection. New MEN H5 files use the v2.0 schema and take the same code path as GLI.

### 4.3 Sampling Strategy

Given ~750 MEN studies and ~950 GLI scans in training, the WeightedRandomSampler assigns inverse-dataset-size weights so each domain contributes ~50% of samples per epoch. The `num_samples` parameter should be set to `2 * max(n_men, n_gli)` to ensure full coverage.

### 4.4 Segmentation Label Unification

Both domains must produce the same 3-channel binary output `[TC, WT, ET]` for the segmentation loss. A domain-aware conversion is required:

**MEN** (labels 1=ET, 2=NET, 3=Cyst):
- TC = ET ∪ NET = (seg == 1) | (seg == 2)
- WT = ET ∪ NET ∪ Cyst = (seg == 1) | (seg == 2) | (seg == 3)
- ET = (seg == 1)

**GLI** (labels 1=NETC, 2=SNFH, 3=ET, 4=RC):
- TC = NETC ∪ ET ∪ RC = (seg == 1) | (seg == 3) | (seg == 4)
- WT = NETC ∪ SNFH ∪ ET ∪ RC = (seg > 0)
- ET = (seg == 3)

This conversion must happen either inside the dataset `__getitem__` (preferred, so the DataLoader always yields [3, H, W, D] segs) or as a domain-conditioned transform.

### 4.5 Semantic Feature Targets

The H5 files already contain pre-computed semantic features (`semantic/volume`, `semantic/location`, `semantic/shape`) computed with the correct label mappings (GLI uses `merge_rc_into_ncr=True`). No recomputation is needed at runtime. Both domains use the same 10-dimensional target space: [vol_total, vol_1, vol_2, vol_3, loc_z, loc_y, loc_x, sphericity, enhancement_ratio, infiltration_index].

### 4.6 Target Normalisation

The `compute_target_statistics()` function iterates over the training DataLoader and computes mean/std for volume (4), location (3), and shape (3). When the DataLoader is already mixed (MEN + GLI), these statistics naturally reflect both domains. No code change is required beyond using the mixed DataLoader.

---

## 5. Training Configuration

### 5.1 Experimental Conditions

The dual-domain experiment should test the following conditions:

| Condition | LoRA Rank | Data | VICReg | λ_aux | Purpose |
|---|---|---|---|---|---|
| `baseline_frozen` | — | — | — | — | Reference (no training) |
| `men_only_r8` | 8 | MEN only | Yes | 0.3 | Single-domain baseline |
| `dual_r8` | 8 | MEN + GLI | Yes | 0.3 | Primary dual-domain condition |
| `dual_r16` | 16 | MEN + GLI | Yes | 0.3 | Higher rank dual-domain |

The `men_only_r8` condition reproduces the best v3 single-domain configuration as a direct comparator.

### 5.2 Unchanged Hyperparameters

These remain identical to the v3 configuration:

- `lr_encoder: 1.0e-4`, `lr_decoder: 5.0e-4`, `weight_decay: 1.0e-5`
- `max_epochs: 150`, `early_stopping_patience: 25`
- `lora_dropout: 0.1`, `gradient_clip: 1.0`
- `aux_warmup_epochs: 15`, `aux_warmup_duration: 10`
- `lambda_dice: 1.0`, `lambda_ce: 1.0`
- `lambda_volume: 1.0`, `lambda_location: 0.3`, `lambda_shape: 0.5`
- `lambda_var_enc: 5.0`, `lambda_cov_enc: 1.0`
- ROI: `[128, 128, 128]` for training, `[192, 192, 192]` for feature extraction
- Mixed precision: bf16

### 5.3 New Configuration Keys

```yaml
# Dual-domain configuration
dual_domain:
  enabled: true
  men_h5_path: <path_to_BraTS_MEN.h5>
  gli_h5_path: <path_to_BraTS_GLI.h5>
  mixing_strategy: weighted_random
  domain_balance: 0.5
  grad_accum_steps: 4

# Per-domain validation
validation:
  track_per_domain: true
  early_stopping_metric: combined_dice_mean  # arithmetic mean of MEN + GLI val Dice
```

### 5.4 Validation

Validation must be split by domain. After each epoch, report:

- `val/men_dice_tc`, `val/men_dice_wt`, `val/men_dice_et`, `val/men_dice_mean`
- `val/gli_dice_tc`, `val/gli_dice_wt`, `val/gli_dice_et`, `val/gli_dice_mean`
- `val/combined_dice_mean` = 0.5 × (men_dice_mean + gli_dice_mean)

Early stopping monitors `val/combined_dice_mean`. The validation DataLoaders are domain-separated (one MEN val loader, one GLI val loader), not mixed.

---

## 6. Enhanced Evaluation Protocol

This section defines a comprehensive evaluation battery that must be executed after training completes. Each evaluation answers a specific scientific question. Results are organised into three tiers: (A) online metrics logged during training, (B) post-training quantitative evaluations, and (C) post-training qualitative visualisations.

### 6.1 Tier A: Online Training Metrics (Logged Every Epoch)

These metrics are recorded during training and written to `training_log.csv`.

#### A1. Per-Domain Training Loss Decomposition

For each epoch, log the following **separately for MEN and GLI batches**:

| Metric | Key | Purpose |
|---|---|---|
| Segmentation loss (Dice + CE) | `train/men_seg_loss`, `train/gli_seg_loss` | Track per-domain segmentation learning |
| Auxiliary semantic loss | `train/men_aux_loss`, `train/gli_aux_loss` | Track semantic head convergence per domain |
| VICReg total loss | `train/vicreg_total` | Track encoder regularisation (domain-agnostic) |
| VICReg variance component | `train/vicreg_var_loss` | Dimensional collapse detection |
| VICReg covariance component | `train/vicreg_cov_loss` | Inter-dimension decorrelation |
| Total loss | `train/total_loss` | Aggregate |

**Question answered:** Is one domain dominating the loss landscape?

**Red flag:** If `train/men_seg_loss` decreases monotonically while `train/gli_seg_loss` increases (or vice versa), there is a domain interference problem.

#### A2. Per-Domain Validation Dice (Logged Every Epoch)

As specified in §5.4. Six Dice scores per domain (TC, WT, ET, mean).

**Question answered:** Are we forgetting gliomas while learning meningiomas?

**Red flag:** If `val/gli_dice_mean` drops below the frozen-baseline Dice for GLI, catastrophic forgetting is occurring.

**Threshold reference:** From the domain gap experiment, the frozen BrainSegFounder achieves approximately Dice ~0.65–0.70 on BraTS-GLI using 3-channel BraTS-MEN label convention (expected to be lower than on BraTS 2021 glioma because the label conventions differ). The adapted model should exceed or maintain this level.

#### A3. Gradient Norm Monitoring (Logged Every N Batches)

Log per-parameter-group gradient L2 norms:

| Group | Key | Purpose |
|---|---|---|
| LoRA A matrices | `grad/lora_A` | Encoder adaptation signal |
| LoRA B matrices | `grad/lora_B` | Encoder adaptation signal |
| Decoder | `grad/decoder` | Decoder gradient health |
| Semantic heads | `grad/semantic_heads` | Auxiliary loss gradient health |

Log separately for MEN-sourced and GLI-sourced batches when possible (requires per-batch domain tracking during gradient accumulation).

**Question answered:** Are both domains providing balanced gradient signal?

**Red flag:** Order-of-magnitude asymmetry in gradient norms between domains suggests one domain dominates the other.

---

### 6.2 Tier B: Post-Training Quantitative Evaluation

These are computed after training completes, on the test sets.

#### B1. Per-Domain Segmentation Dice (Sliding Window)

Run full sliding-window inference on both test sets:

| Metric | Dataset | Samples |
|---|---|---|
| `test/men_dice_{tc,wt,et,mean}` | BraTS-MEN test | 150 subjects |
| `test/gli_dice_{tc,wt,et,mean}` | BraTS-GLI test | ~270 scans |

Use the same sliding-window evaluator already implemented in `experiments/lora_ablation/pipeline/evaluate_dice.py`.

**Question answered:** Does dual-domain training maintain segmentation performance on both domains?

**Success criterion:** `test/men_dice_mean ≥ men_only_r8 test Dice` (dual-domain does not regress meningioma performance). `test/gli_dice_mean ≥ frozen baseline GLI Dice` (glioma competence is preserved).

#### B2. Per-Domain GP Semantic Probes (Linear + RBF)

Extract `encoder10` features (768-dim, GAP-pooled over [192, 192, 192]) for all test samples from both domains. Run the existing `GPSemanticProbes` evaluation separately per domain:

| Metric | Kernel | Domain | Description |
|---|---|---|---|
| `probe/men_r2_vol_{linear,rbf}` | Linear / RBF | MEN | Volume predictability |
| `probe/men_r2_loc_{linear,rbf}` | Linear / RBF | MEN | Location predictability |
| `probe/men_r2_shape_{linear,rbf}` | Linear / RBF | MEN | Shape predictability |
| `probe/gli_r2_vol_{linear,rbf}` | Linear / RBF | GLI | Volume predictability |
| `probe/gli_r2_loc_{linear,rbf}` | Linear / RBF | GLI | Location predictability |
| `probe/gli_r2_shape_{linear,rbf}` | Linear / RBF | GLI | Shape predictability |

Also compute:
- `probe/men_r2_mean_linear`, `probe/men_r2_mean_rbf` (average across semantic targets)
- `probe/gli_r2_mean_linear`, `probe/gli_r2_mean_rbf`
- `probe/nonlinearity_evidence_{men,gli}` = Δlog-marginal-likelihood (RBF − Linear) per domain

**Question answered:** Are the GP probes working? Is the latent space linearly decodable for both tumour types?

**Success criterion:** R² > 0 for all semantic targets on both domains. R²(volume) should be highest. Nonlinearity evidence > 0 means RBF captures structure beyond linear.

#### B3. Cross-Domain GP Probes (Transfer Readiness Test)

This is a novel evaluation designed to predict whether the downstream GP temporal transfer will succeed. Train GP probes on one domain's features and test on the other:

| Evaluation | Train On | Test On | Key |
|---|---|---|---|
| GLI→MEN volume | GLI test features | MEN test features | `probe/cross_gli2men_r2_vol` |
| MEN→GLI volume | MEN test features | GLI test features | `probe/cross_men2gli_r2_vol` |
| GLI→MEN location | GLI test features | MEN test features | `probe/cross_gli2men_r2_loc` |
| GLI→MEN shape | GLI test features | MEN test features | `probe/cross_gli2men_r2_shape` |

**Question answered:** Are the semantic mappings domain-invariant? If a GP probe trained on GLI features can predict MEN semantic targets (and vice versa), the two domains share a compatible latent space — a prerequisite for temporal dynamics transfer.

**Success criterion:** `cross_gli2men_r2_vol > 0`. Any positive R² in cross-domain probing is evidence of domain invariance.

**Statistical note:** Cross-domain R² will be lower than within-domain R² due to distributional shift. The relevant comparison is `cross_r2 > 0` (not `cross_r2 ≈ within_r2`).

#### B4. Domain Discrimination in Adapted Feature Space

Compute the same domain gap metrics used in the domain gap experiment, but now on the **LoRA-adapted** features rather than frozen features:

| Metric | Description |
|---|---|
| `gap/mmd2_gli_men` | Maximum Mean Discrepancy² (permutation test, 1000 permutations) |
| `gap/pad_gli_men` | Proxy-A-Distance (2× (1 − 2ε) where ε = linear classifier error) |
| `gap/classifier_acc_gli_men` | Linear domain classifier accuracy |
| `gap/cka_frozen_vs_adapted_men` | CKA between frozen and adapted encoder features on MEN |
| `gap/cka_frozen_vs_adapted_gli` | CKA between frozen and adapted encoder features on GLI |
| `gap/effective_rank_men` | Effective rank of MEN feature matrix |
| `gap/effective_rank_gli` | Effective rank of GLI feature matrix |
| `gap/effective_rank_combined` | Effective rank of combined feature matrix |

These functions already exist in `src/growth/evaluation/latent_quality.py`.

**Question answered:** Is the latent space well-represented by both tumours? Has LoRA adaptation reduced or increased the domain gap?

**Interpretation guide:**
- **MMD² decrease** (adapted < frozen): LoRA adaptation aligned the domains. Good for transfer.
- **MMD² increase** (adapted > frozen): LoRA pushed domains apart. Bad — revisit sampling balance.
- **CKA(frozen, adapted)** close to 1.0: LoRA made minimal changes to the representation. Expected for low-rank adaptation. If CKA < 0.8, substantial restructuring occurred.
- **Effective rank**: Should not decrease compared to frozen baseline. Rank collapse indicates dimensional collapse (VICReg should prevent this).

#### B5. VICReg Effectiveness Diagnostics

Compute the following on the extracted encoder10 features:

| Metric | Description |
|---|---|
| `vicreg/variance_per_dim_mean` | Mean per-dimension std across 768 dimensions |
| `vicreg/variance_per_dim_min` | Minimum per-dimension std (collapse indicator) |
| `vicreg/n_dead_dims` | Number of dimensions with std < 0.01 (dead dimension count) |
| `vicreg/off_diag_cov_mean` | Mean absolute off-diagonal covariance element |
| `vicreg/effective_rank` | Same as gap/effective_rank but interpreted for VICReg |

Compare these across conditions: `baseline_frozen` vs `men_only_r8` vs `dual_r8`.

**Question answered:** Is VICReg preventing dimensional collapse?

**Success criterion:** `n_dead_dims = 0` (no collapsed dimensions). `variance_per_dim_min > 0.1`. `effective_rank(dual_r8) ≥ effective_rank(frozen)`.

#### B6. Forgetting Index (Quantitative)

Define a per-domain forgetting index as the normalised performance change relative to the frozen baseline:

```
FI_GLI = (Dice_adapted_GLI - Dice_frozen_GLI) / Dice_frozen_GLI
FI_MEN = (Dice_adapted_MEN - Dice_frozen_MEN) / Dice_frozen_MEN
```

| Condition | FI_GLI | FI_MEN | Interpretation |
|---|---|---|---|
| `men_only_r8` | Negative (expected) | Positive (expected) | MEN-only adaptation forgets GLI |
| `dual_r8` | ≥ 0 (goal) | Positive (goal) | Dual-domain preserves both |

Also compute the **probe forgetting index** (same formula but using GP probe R² instead of Dice).

**Question answered:** Precisely how much glioma performance is lost and meningioma performance is gained?

---

### 6.3 Tier C: Post-Training Qualitative Visualisations

#### C1. Dual-Domain UMAP

Generate a 2D UMAP embedding of `encoder10` features coloured by domain (MEN vs GLI) with separate marker shapes. Overlay semantic target values (e.g., volume) as a colour gradient on a second panel.

**Panels (4 total):**
1. UMAP coloured by domain (MEN=blue, GLI=orange)
2. UMAP coloured by log(total_volume) (continuous colourmap, viridis)
3. UMAP coloured by tumour location (centroid z-coordinate)
4. UMAP coloured by sphericity

Produce this for: (a) frozen encoder, (b) `men_only_r8`, (c) `dual_r8`.

**Question answered:** Do glioma and meningioma features form a single continuous manifold or separate clusters? Are semantic gradients smooth across both domains?

**Desired outcome:** In `dual_r8`, the two domains should partially overlap (shared anatomy manifold) but maintain distinct subregions (tumour-type-specific features). If the domains are completely separated, the cross-domain probe R² (B3) will be near zero.

#### C2. Per-Domain Training Curve Comparison

Plot the following over epochs for all conditions:

1. `val/men_dice_mean` and `val/gli_dice_mean` on the same axes (2 curves per condition, 4 conditions = 8 curves).
2. `train/men_seg_loss` and `train/gli_seg_loss` on the same axes.
3. VICReg components (`vicreg_var_loss`, `vicreg_cov_loss`) over epochs.

**Question answered:** Is there a point during training where one domain starts regressing while the other improves? (Early stopping should prevent this, but the visualisation documents it.)

#### C3. Variance Spectrum Plot

Plot the per-dimension variance of the 768-dim features, sorted in descending order, for frozen vs. `men_only_r8` vs. `dual_r8`. This is already implemented in `experiments/lora_ablation/analysis/visualizations.py`.

**Question answered:** Has LoRA adaptation changed the variance distribution? Is dimensional collapse occurring?

#### C4. Cross-Correlation Matrix

For the `dual_r8` condition, compute the 768×768 correlation matrix of the test features and visualise it as a heatmap. Also compute the inter-domain correlation matrix (GLI features × MEN features) if sample sizes permit.

**Question answered:** Are the feature dimensions decorrelated (VICReg is working) or highly redundant?

#### C5. GP Probe Sausage Plots (Predictions vs Ground Truth)

For each semantic target (volume, location, shape), for each domain, plot:
- x-axis: ground truth value
- y-axis: GP posterior mean prediction
- Shaded region: ±2σ posterior predictive interval
- Diagonal: perfect prediction line

This uses the `predictive_std` field from `GPProbeResults`.

**Question answered:** Are the GP probes calibrated? Are they confident where they should be?

---

## 7. Output Artefacts

After training and evaluation, the following directory structure is produced:

```
results/dual_domain_lora/
├── conditions/
│   ├── baseline_frozen/
│   │   ├── features/          # encoder10 features .npy (MEN + GLI test)
│   │   ├── probes/            # GP probe results .json
│   │   └── dice/              # Dice scores .json
│   ├── men_only_r8/
│   │   ├── checkpoints/       # best_model.pt
│   │   ├── training_log.csv   # All Tier A metrics
│   │   ├── features/
│   │   ├── probes/
│   │   └── dice/
│   ├── dual_r8/
│   │   ├── checkpoints/
│   │   ├── training_log.csv
│   │   ├── features/
│   │   │   ├── men_test_features.npy     # [150, 768]
│   │   │   ├── men_test_semantics.npy    # [150, 10]
│   │   │   ├── gli_test_features.npy     # [~270, 768]
│   │   │   └── gli_test_semantics.npy    # [~270, 10]
│   │   ├── probes/
│   │   │   ├── men_probes.json
│   │   │   ├── gli_probes.json
│   │   │   └── cross_domain_probes.json
│   │   ├── dice/
│   │   │   ├── men_test_dice.json
│   │   │   └── gli_test_dice.json
│   │   └── domain_gap/
│   │       ├── mmd2.json
│   │       ├── cka.json
│   │       └── effective_rank.json
│   └── dual_r16/
│       └── ...
├── figures/
│   ├── umap_dual_domain.png          # C1
│   ├── training_curves.png           # C2
│   ├── variance_spectrum.png         # C3
│   ├── correlation_matrix.png        # C4
│   └── sausage_plots/               # C5
├── tables/
│   ├── comprehensive_results.csv
│   └── comprehensive_results.tex
└── report.md                         # Auto-generated analysis
```

The features stored under `features/` are critical inputs to Phase 2 (SDP). These must be saved as float32 NumPy arrays.

---

## 8. Implementation Checklist

### 8.1 New Modules

- [ ] `src/growth/data/dual_domain.py` — Factory function to create the ConcatDataset + WeightedRandomSampler mixed DataLoader
- [ ] `src/growth/data/label_conversion.py` — Domain-aware 3-channel `[TC, WT, ET]` label conversion
- [ ] `experiments/dual_domain_lora/` — New experiment directory with YAML configs, adapted pipeline scripts

Note: No separate `BraTSGLIDatasetH5` class is needed. The existing `BraTSDatasetH5` already handles both MEN and GLI H5 files via the unified v2.0 schema (scan_ids, patient_ids, longitudinal/ CSR, domain attr). The `domain` field is already included in `__getitem__` output.

### 8.2 Modifications to Existing Modules

- [ ] `experiments/lora_ablation/pipeline/train_condition.py` → `train_epoch()` must handle the `domain` field for label conversion; `validate_epoch()` must support per-domain metric reporting
- [ ] `experiments/lora_ablation/pipeline/extract_features.py` → Must extract features separately for MEN and GLI test sets, saving to separate files
- [ ] `experiments/lora_ablation/pipeline/evaluate_probes.py` → Must run probes per-domain and cross-domain (B3)
- [ ] `experiments/lora_ablation/pipeline/evaluate_dice.py` → Must evaluate on both MEN and GLI test sets

### 8.3 New Configuration File

Create `experiments/dual_domain_lora/config/dual_domain_v1.yaml` with all the parameters from §5, inheriting from `ablation_v3.yaml` structure.

### 8.4 Unchanged Components

- `src/growth/models/encoder/` — No changes (LoRA injection, SwinUNETR loading)
- `src/growth/losses/` — No changes (DiceCELoss, SemanticRegressionLoss, EncoderVICRegLoss are all domain-agnostic after label conversion)
- `src/growth/evaluation/gp_probes.py` — No changes (the `GPSemanticProbes` class is data-agnostic)
- `src/growth/evaluation/latent_quality.py` — No changes (MMD, CKA, PAD functions are generic)

---

## 9. Potential Failure Modes and Mitigations

### 9.1 GLI Dominates MEN in Training Loss

**Symptom:** `val/gli_dice_mean` improves rapidly while `val/men_dice_mean` stagnates or regresses.

**Cause:** GLI contributes ~950 scans vs. MEN's 750 studies, and GLI tumours are typically larger (more segmentation signal per sample).

**Mitigation:** The WeightedRandomSampler balances domain representation at 50/50. If the problem persists, increase MEN weight to 0.6 or reduce GLI weight. Also verify that the segmentation loss per sample is similar across domains (GLI tumours may produce larger absolute Dice+CE loss due to larger tumour volume — normalisation by tumour size may be needed).

### 9.2 Label 4 (Resection Cavity) Causes Training Instability

**Symptom:** GLI segmentation loss spikes or oscillates.

**Cause:** The resection cavity (label 4) is a large empty region that the model may struggle with using the [TC, WT, ET] convention (RC is merged into TC).

**Mitigation:** This is handled by the label conversion in §4.4 which merges RC into TC. If instability occurs, an alternative is to convert RC to background (label 0) instead. Monitor `test/gli_dice_tc` specifically — if TC Dice is very low while WT and ET are reasonable, the RC→TC merging may be problematic.

### 9.3 VICReg Variance Term Overwhelms Segmentation

**Symptom:** `train/vicreg_var_loss` is orders of magnitude larger than segmentation loss.

**Mitigation:** The v3 configuration uses `lambda_var_enc=5.0, lambda_cov_enc=1.0`. If these dominate, reduce to `lambda_var_enc=1.0, lambda_cov_enc=0.5`. VICReg should be a regulariser, not the primary loss.

### 9.4 Gradient Accumulation Breaks VICReg Statistics

**Symptom:** VICReg loss is unstable or ineffective.

**Cause:** If VICReg is computed on physical batch_size=2 and accumulated (rather than computed on the effective batch of 8), the variance/covariance estimates are extremely noisy.

**Mitigation:** VICReg should be computed on the physical batch (2 samples). The variance hinge loss works with any batch_size ≥ 2. The covariance penalty with 2 samples is noisy but accumulated over 4 steps provides a reasonable gradient. If this is insufficient, an alternative is to accumulate features across accumulation steps in a buffer and compute VICReg on the full effective batch — but this adds memory and complexity.

---

## 10. References

- Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.
- Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175.
- Cox, J., et al. (2024). BrainSegFounder: Towards Foundation Models for Neuroimage Segmentation. *Medical Image Analysis*.
- De Verdier, M. C., et al. (2024). The 2024 BraTS Challenge: Glioma Segmentation on Post-treatment MRI. arXiv:2405.18368.
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
