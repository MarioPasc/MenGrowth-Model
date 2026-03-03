# Dual-Domain Mixed-Batch LoRA: Cross-Tumor Transfer for Longitudinal Growth Prediction

## Standalone Context Document for Implementation

**Project:** MenGrowth-Model  
**Date:** March 2026  
**Scope:** This document describes a methodological pivot from single-domain (meningioma-only) LoRA adaptation to dual-domain (glioma + meningioma) mixed-batch training across the full pipeline. It is self-contained: an implementing agent needs no other document to understand what must change and why.

---

## 1. Why This Pivot Exists

The MenGrowth-Model pipeline predicts meningioma growth trajectories from longitudinal MRI. The pipeline has four phases: (1) LoRA adaptation of BrainSegFounder, (2) Supervised Disentangled Projection (SDP), (3) ComBat harmonisation, and (4) Gaussian Process temporal modelling.

The original design trained and evaluated all phases exclusively on meningioma data. The fundamental problem was **sample size**: the Andalusian longitudinal cohort has only 31 patients (~100 studies, ~62 temporal pairs). The PA-MOGP temporal model alone requires 95 hyperparameters, yielding a ratio of ~0.65 observations per parameter — far below any reasonable statistical threshold.

Two discoveries change the situation:

1. **The BraTS-GLI 2024 dataset is longitudinal.** It is built primarily from the UCSF-ALPTDG collection (Fields et al., *Radiology: Artificial Intelligence*, 2024): 298 patients with 2 consecutive post-treatment timepoints each (596 scans), full BraTS preprocessing, expert voxelwise segmentations (ET, SNFH, NETC, RC), and 4-channel MRI (T2-FLAIR, T1c, T1n, T2w). This dataset is already in the project's data storage and was used for the domain gap experiment.

2. **The domain gap between glioma and meningioma is moderate, not catastrophic.** Empirical measurements from the frozen BrainSegFounder `encoder10` features (768-dim) show:

| Metric | GLI ↔ MEN | GLI ↔ MG | MEN ↔ MG |
|---|---|---|---|
| MMD² | 0.10*** | 0.05* | 0.02 n.s. |
| PAD | 0.82 | 1.0 | 1.4 |
| Classifier Accuracy | 0.70 | 0.75 | 0.84 |

PC1 captures 94.3% of variance across all three domains. The distributional shift is real but resides in the remaining ~5.7% of the variance spectrum.

The pivot: **train the LoRA-adapted encoder and SDP jointly on glioma and meningioma, then train the GP temporal model on glioma longitudinal trajectories and transfer/fine-tune on meningioma.** This is a cross-tumor transfer learning paradigm.

---

## 2. What Changes and What Does Not

### 2.1 What Does NOT Change

- **BrainSegFounder architecture** (SwinUNETR, BSF-Tiny, 62M params). No model architecture changes.
- **LoRA configuration**: rank 8, alpha 16, dropout 0.1, applied to Stages 3–4 of SwinViT. These hyperparameters remain the same.
- **SDP architecture**: 2-layer MLP (768 → 512 → 128) with spectral normalisation, GELU, dropout 0.1. Partition layout: vol=24, loc=8, shape=12, residual=84.
- **SDP loss structure**: SemanticRegressionLoss + CovarianceLoss + VarianceHingeLoss + DistanceCorrelationLoss with 4-phase curriculum.
- **GP model hierarchy**: LME → H-GP → PA-MOGP.
- **Feature extraction level**: `encoder10` (768-dim GAP-pooled).
- **Spatial configuration**: ROI `[128, 128, 128]` for training, `[192, 192, 192]` for feature extraction, 1mm isotropic.
- **Channel order**: `[t2f, t1c, t1n, t2w]` — identical for both datasets.

### 2.2 What Changes

| Component | Previous (Single-Domain) | New (Dual-Domain) |
|---|---|---|
| **Phase 1 LoRA training data** | BraTS-MEN only (1000 subjects) | BraTS-MEN + BraTS-GLI mixed batches |
| **Phase 1 LoRA loss** | `L_seg^MEN + λ_aux·L_aux^MEN + λ_vicreg·L_vicreg` | `L_seg^MEN + L_seg^GLI + λ_aux·(L_aux^MEN + L_aux^GLI) + λ_vicreg·L_vicreg` |
| **Phase 1 LoRA DataLoader** | Single-dataset loader | Domain-aware mixed-batch sampler |
| **Phase 1 segmentation labels** | 3-channel BraTS-MEN (TC, WT, ET) | Domain-conditioned: 3-ch for MEN, 3-ch for GLI (see §3.4) |
| **Phase 1 validation** | MEN-only val set | Separate MEN val and GLI val, tracked independently |
| **Phase 2 SDP training data** | BraTS-MEN encoder features only | BraTS-MEN + BraTS-GLI encoder features |
| **Phase 3 ComBat** | MenGrowth harmonisation only | Three-domain harmonisation (GLI, MEN, MG) |
| **Phase 4 GP training** | MenGrowth only (31 patients, ~62 pairs) | Train on GLI (298 patients, 298 pairs), evaluate/fine-tune on MenGrowth |
| **Evaluation** | LOPO-CV on MenGrowth | Three-condition ablation (§7) |

---

## 3. Phase 1: Dual-Domain Mixed-Batch LoRA

### 3.1 Training Objective

The new composite loss for each training batch is:

```
L_total = L_seg(pred, seg) + λ_aux · L_aux(encoder_features, semantic_targets) + λ_vicreg · L_vicreg(encoder_features)
```

This is structurally identical to the current loss — the change is that **each batch contains samples from both BraTS-MEN and BraTS-GLI**. The segmentation loss, auxiliary semantic loss, and VICReg loss are computed per-sample with domain-appropriate labels and targets, then averaged across the batch.

There is no domain-specific loss weighting or adversarial term. The mixed-batch approach treats glioma and meningioma as a single heterogeneous training set.

### 3.2 Mixed-Batch DataLoader

The DataLoader must alternate between the two datasets within each batch. The implementation requires a **domain-aware batch sampler**.

**Strategy: interleaved sampling with a configurable mixing ratio.**

Given batch size `B=2` (constrained by GPU memory for 128³ volumes), each batch should contain samples drawn proportionally from both domains. Since `B=2` is small, the practical implementation is an **alternating-epoch** or **interleaved-iteration** scheme:

- **Option A (recommended): ConcatDataset + WeightedRandomSampler.** Concatenate BraTS-MEN and BraTS-GLI into a single dataset. Assign sampling weights inversely proportional to dataset size so both domains are equally represented. This is the simplest approach and works with the existing training loop.
  
- **Option B: Alternating batches.** Odd iterations sample from MEN, even iterations sample from GLI. This is equivalent to Option A in expectation but introduces periodic gradient bias.

**Option A is preferred.** The implementation uses `torch.utils.data.ConcatDataset` and `torch.utils.data.WeightedRandomSampler`.

Each sample in the concatenated dataset must carry a `domain` field (`"MEN"` or `"GLI"`) so the training loop can:
1. Route samples to the correct segmentation label mapping.
2. Log per-domain loss components separately.
3. Apply domain-specific semantic target statistics for normalisation.

### 3.3 BraTS-GLI Dataset Class

A `BraTSGLIDataset` already exists in `experiments/domain_gap/run_domain_gap.py`. It needs to be extracted into a proper module under `src/growth/data/` with the following interface matching `BraTSMENDataset`:

```python
class BraTSGLIDataset(Dataset):
    """BraTS-GLI (Glioma) dataset for dual-domain LoRA training.
    
    Must return the same dict structure as BraTSMENDataset:
        - "image": [4, H, W, D] float32 tensor
        - "seg": [1, H, W, D] int tensor (raw BraTS labels)
        - "semantic_features": {"volume": [4], "location": [3], "shape": [3]}
        - "subject_id": str
        - "domain": "GLI"  # NEW FIELD
    """
```

**Critical:** the BraTS-GLI dataset follows the same directory structure and NIfTI naming convention as BraTS-MEN. The modality suffixes are identical: `-t1c.nii.gz`, `-t1n.nii.gz`, `-t2f.nii.gz`, `-t2w.nii.gz`, `-seg.nii.gz`. The channel order `[t2f, t1c, t1n, t2w]` is the same.

**Data path references** (from `foundation.yaml`):
- Local: `paths.brats_gli_root: /media/mpascual/PortableSSD/BraTS_GLI/source/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2`
- Server: `paths.brats_gli_root: /media/hddb/mario/data/BraTS-GLI-100`

The server path currently contains only 100 subjects (the subset used for domain gap analysis). The **full BraTS-GLI 2024 dataset** must be made available at a new path. The configuration should add a new key:

```yaml
paths:
  brats_gli_full_root: <path_to_full_BraTS_GLI_2024>
```

### 3.4 Segmentation Label Convention

This is a critical implementation detail. The two datasets use different segmentation label conventions:

**BraTS-MEN (meningioma):**
- Raw labels: 1 = ET (enhancing tumour), 2 = NET (non-enhancing tumour), 3 = Cyst
- 3-channel conversion: TC (Tumour Core = ET ∪ NET), WT (Whole Tumour = ET ∪ NET ∪ Cyst), ET

**BraTS-GLI 2024 post-treatment (glioma):**
- Raw labels: 1 = NETC (non-enhancing tumour core), 2 = SNFH (surrounding non-enhancing FLAIR hyperintensity), 3 = ET (enhancing tissue), 4 = RC (resection cavity)
- 3-channel conversion: TC = NETC ∪ ET ∪ RC, WT = NETC ∪ SNFH ∪ ET ∪ RC, ET

**Resolution:** both datasets must be converted to the **same 3-channel output** format `[TC, WT, ET]` before computing the segmentation loss. The existing `ConvertToMultiChannelBasedOnBratsClassesd` MONAI transform handles the standard BraTS-GLI convention. A **domain-aware label conversion** function is required:

```python
def convert_to_3channel(seg: Tensor, domain: str) -> Tensor:
    """Convert raw segmentation labels to 3-channel [TC, WT, ET].
    
    Args:
        seg: Raw segmentation tensor [1, H, W, D] with integer labels.
        domain: "MEN" or "GLI".
    
    Returns:
        3-channel binary tensor [3, H, W, D].
    """
    if domain == "MEN":
        # MEN labels: 1=ET, 2=NET, 3=Cyst
        et = (seg == 1)
        net = (seg == 2)
        cyst = (seg == 3)
        tc = et | net          # Tumour Core
        wt = et | net | cyst   # Whole Tumour
        return torch.stack([tc, wt, et], dim=0).float()
    
    elif domain == "GLI":
        # GLI 2024 labels: 1=NETC, 2=SNFH, 3=ET, 4=RC
        netc = (seg == 1)
        snfh = (seg == 2)
        et = (seg == 3)
        rc = (seg == 4)
        tc = netc | et | rc           # Tumour Core
        wt = netc | snfh | et | rc    # Whole Tumour
        return torch.stack([tc, wt, et], dim=0).float()
```

This function must be called **inside the dataset `__getitem__`** or as a MONAI transform, before the segmentation is passed to the loss function. The 3-channel Dice + CE loss (`DiceCELoss` with sigmoid activation, 3-channel output) works identically for both domains after this conversion.

### 3.5 Semantic Feature Computation for GLI

The auxiliary semantic heads require `volume`, `location`, and `shape` targets computed from the segmentation mask. The existing `extract_semantic_features()` function in `src/growth/data/semantic_features.py` computes:

- **Volume** (4 values): log(1 + voxel_count) for each of the 3 subregions + total.
- **Location** (3 values): centroid coordinates (cz, cy, cx) of the whole tumour, normalised to [0, 1].
- **Shape** (3 values): sphericity, log(surface_area), solidity.

For BraTS-GLI, the semantic features should be computed from the **3-channel [TC, WT, ET]** representation (after label conversion), so the volume subregions are: TC volume, WT volume, ET volume, total volume. This makes the semantic target space identical across domains — the auxiliary heads do not need to know which domain a sample comes from.

**Implementation note:** the `extract_semantic_features()` function currently takes a raw segmentation tensor and BraTS-MEN label mapping. It must be generalised to accept the 3-channel binary representation directly, or a `domain` parameter must be added to handle the label mapping internally.

### 3.6 Target Normalisation Statistics

The current LoRA training computes per-feature normalisation statistics (mean, std) from the training set for the auxiliary loss. Under dual-domain training, these statistics must be computed from the **combined** MEN + GLI training set. This ensures the auxiliary loss operates in a unified normalised space.

The `compute_target_statistics()` function in `experiments/lora_ablation/pipeline/train_condition.py` iterates over the training DataLoader and computes `volume_mean`, `volume_std`, `location_mean`, `location_std`, `shape_mean`, `shape_std`. No code change is needed if the DataLoader already contains mixed MEN + GLI samples — the statistics will naturally reflect both domains.

### 3.7 Validation Protocol

Validation must be tracked **per-domain** to monitor whether the LoRA adaptation maintains glioma competence while acquiring meningioma competence. The validation function should report:

- `val/men_dice_mean`, `val/men_dice_tc`, `val/men_dice_wt`, `val/men_dice_et`
- `val/gli_dice_mean`, `val/gli_dice_tc`, `val/gli_dice_wt`, `val/gli_dice_et`
- `val/combined_dice_mean` (for early stopping)

Early stopping should monitor `val/combined_dice_mean` (arithmetic mean of both domain Dice scores) or, if meningioma performance is the priority, a weighted combination with higher weight on MEN.

### 3.8 Data Splits

**BraTS-MEN splits (unchanged from current configuration):**
- LoRA train: 750 subjects
- LoRA val: 100 subjects
- Test: 150 subjects (held out, never seen during training)

**BraTS-GLI splits (new):**
- LoRA train: ~240 subjects (~80% of 298)
- LoRA val: ~58 subjects (~20% of 298)

The GLI split must be **patient-level** (no leakage between train/val). Since each patient has exactly 2 timepoints, both timepoints of the same patient must go to the same split.

**Important**: When creating the GLI training set for LoRA, each study (timepoint) is an independent training sample for segmentation. The 240 train patients contribute 480 studies; the 58 val patients contribute 116 studies. The mixed training set is then 750 (MEN) + 480 (GLI) = **1230 studies per epoch**.

### 3.9 Sampling Weights

To ensure balanced domain representation despite unequal dataset sizes, the `WeightedRandomSampler` should assign weights:

```python
n_men = len(men_dataset)  # 750
n_gli = len(gli_dataset)  # 480
w_men = 1.0 / n_men       # ~0.00133
w_gli = 1.0 / n_gli       # ~0.00208

# Each sample gets its domain's weight
weights = [w_men] * n_men + [w_gli] * n_gli
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

This ensures that in expectation, ~50% of training samples come from each domain per epoch. The `num_samples` parameter should equal `2 * max(n_men, n_gli)` to ensure both datasets are fully traversed per epoch.

### 3.10 Configuration Changes

A new YAML configuration file should be created at `experiments/lora_ablation/config/server/LoRA_dual_domain_icai.yaml` (and a local variant). Key additions relative to `LoRA_semantic_heads_icai.yaml`:

```yaml
experiment:
  name: lora_dual_domain
  seed: 42

paths:
  checkpoint: <path_to_finetuned_model_fold_0.pt>
  men_data_root: <path_to_BraTS-MEN>
  gli_data_root: <path_to_BraTS-GLI-2024-full>

dual_domain:
  enabled: true
  mixing_strategy: weighted_random   # "weighted_random" | "alternating"
  domain_balance: 0.5                # Target fraction of MEN samples per epoch
  gli_train_fraction: 0.8            # Fraction of GLI patients for training
  gli_val_fraction: 0.2
  
  # Per-domain validation tracking
  track_per_domain_metrics: true
  early_stopping_metric: combined_dice_mean  # or "men_dice_mean"

# All other training params remain identical to LoRA_semantic_heads_icai.yaml
```

---

## 4. Phase 2: Dual-Domain SDP Training

### 4.1 What Changes

The SDP module maps frozen (LoRA-adapted) encoder features to the structured 128-dim latent space. Under the dual-domain paradigm, the SDP training set expands from BraTS-MEN features to BraTS-MEN + BraTS-GLI features.

**Feature extraction:** After Phase 1 completes, run the LoRA-adapted encoder on both BraTS-MEN and BraTS-GLI datasets to extract `encoder10` features (768-dim, GAP-pooled). Store these as:

```
features/
├── men_train_features.npy    # [N_men_train, 768]
├── men_train_semantics.npy   # [N_men_train, 10] (4 vol + 3 loc + 3 shape)
├── gli_train_features.npy    # [N_gli_train, 768]
├── gli_train_semantics.npy   # [N_gli_train, 10]
├── men_val_features.npy
├── men_val_semantics.npy
├── gli_val_features.npy
└── gli_val_semantics.npy
```

### 4.2 SDP Training Data

Concatenate MEN and GLI features for SDP training:

```python
train_features = np.concatenate([men_train_features, gli_train_features], axis=0)
train_semantics = np.concatenate([men_train_semantics, gli_train_semantics], axis=0)
```

The SDP loss (semantic regression + VICReg + dCor) operates identically on the concatenated set. No domain label is needed at this stage — the SDP learns a shared projection that maps both tumour types to the same structured latent space.

### 4.3 SDP Configuration

No changes to `phase2_sdp.yaml` architecture or loss weights. The only change is the training data source (concatenated features from both domains).

### 4.4 Verification

After SDP training, run the semantic probe evaluation on both domains separately:

- `probe_r2_vol_men`, `probe_r2_loc_men`, `probe_r2_shape_men`
- `probe_r2_vol_gli`, `probe_r2_loc_gli`, `probe_r2_shape_gli`

Both should show positive R² values. If GLI probe R² values are substantially lower than MEN, the SDP is not learning a domain-invariant mapping, and the GP transfer will be compromised.

---

## 5. Phase 3: Three-Domain ComBat Harmonisation

### 5.1 What Changes

ComBat harmonisation (Johnson et al., *Biostatistics*, 2007) in the SDP latent space now operates over three domains instead of one:

- **Domain 1:** BraTS-GLI (glioma, multi-institutional)
- **Domain 2:** BraTS-MEN (meningioma, multi-institutional)
- **Domain 3:** MenGrowth (meningioma, Andalusian cohort, ~10 scanners)

ComBat removes scanner/acquisition batch effects while preserving biological variation. The implementation uses `neuroCombat` or equivalent.

### 5.2 Batch Variable

The ComBat batch variable should encode the **dataset of origin** (GLI, MEN, MG), not individual scanners (which are unknown for most subjects). This is a coarse-grained harmonisation that removes the dataset-level distributional shift while preserving within-dataset variation.

If scanner metadata is available for some datasets (MenGrowth has ~10 scanners), a hierarchical ComBat approach is possible: first harmonise within MenGrowth across scanners, then across datasets. However, for the initial implementation, dataset-level batching is sufficient.

---

## 6. Phase 4: Cross-Tumour GP Temporal Training

This is the phase that benefits most from the pivot. The GP temporal model transitions from being trained on 31 meningioma patients to being trained on 298 glioma patients and transferred to meningioma.

### 6.1 Glioma Latent Trajectory Extraction

After Phases 1–3, process all BraTS-GLI 2024 timepoints through the pipeline:

```
x_GLI(t_k) → φ_LoRA(x) → GAP → π_SDP → ComBat → z_GLI(t_k) ∈ ℝ^128
```

For each patient `i` with two timepoints `(t_1, t_2)`:

```
trajectory_i = {
    "patient_id": str,
    "domain": "GLI",
    "timepoints": [t_1, t_2],           # In months (from clinical metadata)
    "latent_vectors": [z(t_1), z(t_2)], # Each ℝ^128
    "delta_t": t_2 - t_1,              # In months
}
```

**Clinical metadata requirement:** The BraTS-GLI 2024 / UCSF-ALPTDG dataset includes clinical metadata with time between scans. This `delta_t` is essential for GP training. If exact dates are unavailable, the `delta_t` may need to be estimated from the dataset documentation or assumed uniform.

### 6.2 GP Training on Glioma

Train the three-tier GP hierarchy on glioma trajectories:

**Model A (LME):** Linear Mixed-Effects baseline on `z_vol` (24-dim). Fit population slope and intercept + per-patient random effects using REML.

**Model B (H-GP):** Hierarchical GP on `z_vol` with Matérn-5/2 kernel and linear mean function (initialised from LME). Hyperparameters optimised via marginal likelihood.

**Model C (PA-MOGP):** Partition-Aware Multi-Output GP on `z_active = [z_vol, z_loc, z_shape]` (44-dim) with partition-specific kernels and rank-1 cross-partition coupling. 95 shared hyperparameters.

All three models are trained on the 298 glioma patient trajectories (298 two-timepoint transitions, 596 observations total).

The MAP hyperparameters from glioma training are:

```
Θ_S* = argmax_Θ Σ_{i ∈ GLI} log p(z_i | t_i, Θ)
```

### 6.3 Transfer Strategies

**Strategy A — Direct transfer (zero-shot):**

Apply `Θ_S*` directly to the 31 MenGrowth patients without any re-estimation. Evaluate predictive R² for volume, latent MSE, and 95% calibration coverage. This tests whether glioma temporal dynamics transfer to meningioma.

**Strategy B — Transfer + fine-tune (few-shot):**

Initialise GP hyperparameters at `Θ_S*`, then refine via LOPO-CV on MenGrowth:

```
Θ_T* = argmax_Θ Σ_{j ∈ MG} log p(z_j | t_j, Θ),   Θ^(0) = Θ_S*
```

**Partial transfer variant:** Transfer kernel hyperparameters only, re-estimate mean function parameters from meningioma data:

```
θ_kernel^(0) = θ_kernel,S*    (transferred from glioma)
φ_mean^(0)   = MEN-specific   (re-initialised for meningioma)
```

This is motivated by the treatment-status confound: post-treatment glioma growth dynamics differ from untreated meningioma growth, but the temporal correlation structure (smoothness, lengthscale) may transfer.

### 6.4 Sample Size Improvement

| Statistic | Previous (MEN only) | New (GLI → MEN) |
|---|---|---|
| GP training patients | 31 | 298 |
| GP training studies | ~100 | 596 |
| GP training temporal pairs | ~62 | 298 |
| Observations per PA-MOGP parameter | 0.65 | 3.79 |
| Transfer test patients | — | 31 (all MenGrowth) |

---

## 7. Evaluation Protocol

### 7.1 Three-Condition Ablation

The decisive experiment compares three conditions:

| Condition | GP Training Data | GP Test Data | What It Tests |
|---|---|---|---|
| **(i) MEN-only** | MenGrowth (31 patients) | MenGrowth (LOPO-CV) | Baseline without transfer |
| **(ii) GLI → MEN (direct)** | BraTS-GLI (298 patients) | MenGrowth (all 31) | Whether glioma dynamics transfer directly |
| **(iii) GLI → MEN (fine-tune)** | BraTS-GLI → MenGrowth | MenGrowth (LOPO-CV) | Value of transfer + adaptation |

**Success criteria:**
- If R²(iii) > R²(i): cross-tumour transfer provides measurable benefit.
- If R²(ii) > 0: temporal dynamics transfer without meningioma-specific adaptation — a strong result.
- If R²(ii) ≈ 0 but R²(iii) > R²(i): partial transfer (kernel only) is effective.

### 7.2 Per-Phase Verification

At each phase, verify dual-domain competence:

**Phase 1 (LoRA):**
- MEN validation Dice ≥ 0.70 (mean across TC, WT, ET)
- GLI validation Dice ≥ 0.65 (mean across TC, WT, ET)
- Neither domain Dice should drop below frozen-baseline level

**Phase 2 (SDP):**
- Semantic probe R² > 0 for volume, location, shape on both MEN and GLI
- Cross-domain R² should not be dramatically asymmetric (|R²_MEN - R²_GLI| < 0.3)

**Phase 4 (GP):**
- Glioma GP training should achieve positive R² on held-out glioma patients (internal validation)
- Three-condition ablation as described in §7.1

---

## 8. Risks and Mitigations

### 8.1 Post-Treatment vs. Pre-Treatment Confound

The BraTS-GLI 2024 data is post-operative (resection cavities, radiation effects, pseudoprogression). MenGrowth is predominantly untreated watch-and-wait.

**Mitigation:** The GP transfer strategy B (partial transfer) separates kernel hyperparameters (which encode temporal smoothness — likely transferable) from mean function parameters (which encode absolute growth trajectories — likely domain-specific). Transfer only kernels; re-estimate means from meningioma.

### 8.2 Label Convention Mismatch

Post-treatment glioma labels (ET, SNFH, NETC, RC) differ from meningioma labels (ET, NET, Cyst).

**Mitigation:** Convert both to unified 3-channel [TC, WT, ET] before any loss computation. Semantic features are computed from this unified representation. See §3.4 for the conversion function.

### 8.3 Catastrophic Forgetting of Glioma Competence

LoRA adaptation on meningioma-only data erases glioma-specific features. Under mixed-batch training, this risk is inherently mitigated because glioma samples provide ongoing gradient signal.

**Mitigation:** The low-rank constraint (rank 8) itself acts as an implicit regulariser — the perturbation ΔW = BA can only modify the encoder in 8 directions per weight matrix, limiting the magnitude of domain shift. Mixed-batch training with balanced sampling ensures neither domain dominates.

### 8.4 Scanner Heterogeneity

BraTS-GLI is multi-institutional (7 academic centres). MenGrowth uses ~10 different scanners.

**Mitigation:** BraTS preprocessing (SRI24 atlas, 1mm iso, skull-stripping) substantially reduces acquisition variance. ComBat harmonisation in Phase 3 removes residual dataset-level batch effects.

---

## 9. Data Leakage Warnings

From Abbad Andaloussi et al. (*Neuro-Oncology Advances*, 2025):

- **BraTS 2024 Post-Treatment ⊃ UCSF-ALPTDG**: The BraTS-GLI 2024 data was contributed from UCSF-ALPTDG. Do not combine both independently.
- **BraTS 2021 ⊃ TCGA-GBM (subset)**: 102 TCGA-GBM patients overlap with BraTS 2021 training data, which is part of BrainSegFounder's Stage 3 fine-tuning set. Since BrainSegFounder was trained on BraTS 2021 glioma data, using the same patients for LoRA evaluation would be a soft leak. This is acceptable because LoRA does not re-train from scratch — it adapts an already-glioma-trained model — but it should be acknowledged.
- **Our BraTS-GLI 100 subset**: The 100-subject subset used for domain gap analysis is drawn from BraTS-GLI 2024. These subjects must be tracked to ensure they appear in the GLI training set (not held out) or excluded from evaluation.

---

## 10. Implementation Checklist

### 10.1 New Code Required

- [ ] `src/growth/data/bratsglidata.py` — BraTSGLI dataset class (extract from `experiments/domain_gap/run_domain_gap.py`, add `domain` field and `compute_semantic` support)
- [ ] `src/growth/data/dual_domain_loader.py` — Mixed-batch DataLoader factory (ConcatDataset + WeightedRandomSampler)
- [ ] `src/growth/data/label_conversion.py` — Domain-aware 3-channel label conversion (§3.4)
- [ ] Update `experiments/lora_ablation/pipeline/train_condition.py`:
  - `train_epoch()` must handle per-sample domain routing for label conversion
  - `validate_epoch()` must report per-domain metrics
- [ ] New YAML configs: `LoRA_dual_domain_icai.yaml` (server + local variants)
- [ ] Update `src/growth/data/semantic_features.py` to accept 3-channel binary input or domain parameter

### 10.2 Existing Code That Needs Modification

- [ ] `BraTSMENDataset.__getitem__()` — Add `"domain": "MEN"` to returned dict
- [ ] `compute_target_statistics()` — No change needed if DataLoader is already mixed
- [ ] `evaluate_feature_quality_inline()` — Add domain-stratified reporting
- [ ] Feature extraction scripts — Run on both MEN and GLI after LoRA training
- [ ] SDP training script — Accept concatenated features from both domains
- [ ] ComBat script — Three-domain batch variable
- [ ] GP training script — Accept glioma trajectories as training data, meningioma as test

### 10.3 New Configuration Keys

```yaml
# In foundation.yaml
paths:
  brats_gli_full_root: <path>  # Full BraTS-GLI 2024 dataset

# In LoRA experiment config
dual_domain:
  enabled: true
  mixing_strategy: weighted_random
  domain_balance: 0.5
  gli_train_fraction: 0.8
  gli_val_fraction: 0.2
  track_per_domain_metrics: true
  early_stopping_metric: combined_dice_mean
```

### 10.4 Unchanged Components

- `src/growth/models/encoder/` — No changes (LoRA injection, SwinUNETR loading)
- `src/growth/models/projection/` — No changes (SDP architecture, partitions, semantic heads)
- `src/growth/losses/` — No changes (all losses are domain-agnostic after label conversion)
- `src/growth/models/growth/` — No changes to GP model architecture (LME, H-GP, PA-MOGP)
- `src/growth/evaluation/` — GP probes, growth figures remain structurally identical

---

## 11. References

- Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175.
- Benzekry, S., et al. (2014). Classical Mathematical Models for Description and Prediction of Experimental Tumor Growth. *PLOS Computational Biology*, 10(8), e1003800.
- Cox, J., et al. (2024). BrainSegFounder: Towards Foundation Models for Neuroimage Segmentation. *Medical Image Analysis*.
- De Verdier, M. C., et al. (2024). The 2024 BraTS Challenge: Glioma Segmentation on Post-treatment MRI. arXiv:2405.18368.
- Fields, B. K. K., et al. (2024). The UCSF Adult Longitudinal Post-Treatment Diffuse Glioma MRI Dataset. *Radiology: Artificial Intelligence*.
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Isensee, F., et al. (2024). nnU-Net Revisited. *Nature Methods*.
- Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.
- Abbad Andaloussi, M., et al. (2025). Exploring adult glioma through MRI. *Neuro-Oncology Advances*, 7(1), vdae197.
