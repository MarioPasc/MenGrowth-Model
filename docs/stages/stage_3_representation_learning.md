# Stage 3 -- Representation-Learning Augmented Prediction

## Overview

Stage 3 is the full deep-learning pipeline: BrainSegFounder encoder (frozen or LoRA-adapted) produces 768-dim features, a Supervised Disentangled Projection (SDP) maps them to a structured 128-dim latent space, PCA compresses the residual partition, and a GP with ARD kernel performs growth prediction on the combined volume + residual features.

Previously the PRIMARY approach, Stage 3 is now TERTIARY. It is only justified if it demonstrably outperforms Stages 1 and 2 under LOPO-CV. At N=31, the bias-variance tradeoff strongly favors parsimony, making this the highest-risk stage.

---

## Objective

Test whether deep features from BrainSegFounder capture growth-relevant information **beyond volume**. Specifically:
1. Do residual (non-volume) latent dimensions improve growth prediction?
2. Do deep features improve severity estimation (Stage 2 integration)?
3. Which residual dimensions are selected by ARD, and what do they represent?

---

## Full Pipeline

```
MenGrowth MRI [4, D, H, W]
  --> BSF Encoder (frozen or LoRA-adapted, stages 0-4 + encoder10)
  --> AdaptiveAvgPool3d(1) --> h in R^768
  --> SDP Network (768 --> 512 --> 128, LayerNorm + GELU + SN)
  --> z in R^128 = [z_vol(32) | z_residual(96)]
  --> PCA on z_residual (fit per LOPO fold) --> z_tilde_res in R^k
  --> Concatenate: [log(V+1), z_tilde_res_1, ..., z_tilde_res_k]
  --> GP with Matern-5/2 ARD kernel
  --> Growth prediction +/- uncertainty
```

**Feature extraction:** encoder10 output `[B, 768, 4, 4, 4]` (with 128^3 input) or `[B, 768, 6, 6, 6]` (with 192^3 input) --> GAP normalizes to `[B, 768]`.

---

## Input Contract

```python
# Phase 1 encoder checkpoint
encoder_checkpoint: str  # path to phase1_encoder_merged.pt (or frozen BSF)

# Phase 2 SDP checkpoint
sdp_checkpoint: str      # path to phase2_sdp.pt

# MenGrowth H5 file
h5_path: str

# Stage 1 and Stage 2 results (for comparison)
stage1_best_r2: float
stage1_per_patient_errors: np.ndarray   # [N_patients]
stage2_best_r2: float
stage2_per_patient_errors: np.ndarray   # [N_patients]
```

## Output Contract

```python
# Deep feature prediction results
deep_results: dict = {
    "model_name": str,
    "lora_rank_selected": int,              # Best rank by GP probe R^2
    "sdp_quality": {
        "vol_r2": float,
        "dcor_vol_residual": float,
        "pca_k_components": int,            # k for 90% variance
        "pca_explained_variance": list[float],
    },
    "lopo_metrics": {
        "last_from_rest/r2_log": float,
        "last_from_rest/mae_log": float,
        "last_from_rest/calibration_95": float,
    },
    "bootstrap_ci": {
        "r2_log_point": float,
        "r2_log_lower_95": float,
        "r2_log_upper_95": float,
    },
    "ard_lengthscales": dict[str, float],   # Per-dim lengthscales
    "ard_relevance_ranking": list[str],     # Dims sorted by 1/lengthscale
    "comparison_to_stage1": {
        "delta_r2": float,
        "permutation_p_value": float,
    },
    "comparison_to_stage2": {
        "delta_r2": float,
        "permutation_p_value": float,
    },
    "severity_integration": {
        "clinical_only_loo_r2": float,
        "clinical_plus_deep_loo_r2": float,
        "deep_only_loo_r2": float,
    },
}

# Output files
# experiments/deep_features/results/
#   deep_prediction_results.json
#   lopo_results.json
#   ard_lengthscale_analysis.png
#   pca_variance_explained.png
#   severity_deep_vs_clinical.png
```

---

## LoRA Selection Protocol

Selection is based on **downstream utility for growth prediction**, not segmentation Dice.

**Metric hierarchy:**
1. GP probe R^2 on held-out volume prediction (primary)
2. SDP validation loss (secondary)
3. Segmentation Dice (tertiary -- sanity check only)

**Protocol:**
1. For each rank $r \in \{2, 4, 8, 16, 32\}$, extract features from MenGrowth data.
2. Train SDP on BraTS-MEN features (800 subjects, encoder frozen).
3. Run GP probes on MenGrowth features: `GPProbe(kernel='linear')` and `GPProbe(kernel='rbf')`.
4. Select rank with highest volume probe R^2.

**Existing code:** `src/growth/evaluation/gp_probes.py` (`GPProbe`, `GPSemanticProbes`).

**Recommendation:** At N=31, prefer lower ranks (r=4 or r=8). Higher ranks risk overfitting the adapter without improving downstream representation quality.

---

## SDP Module

### Architecture

```
h in R^768
  --> LayerNorm(768)
  --> SpectralNorm(Linear(768, 512)) --> GELU --> Dropout(0.1)
  --> SpectralNorm(Linear(512, 128))
  --> z in R^128
```

Spectral normalization on ALL linear layers (D13).

### Partition Structure

| Partition | Dimensions | Indices | Supervised Targets |
|---|---|---|---|
| `z_vol` | 32 | 0--31 | 4 log-volumes (total, NCR, ED, ET) |
| `z_residual` | 96 | 32--127 | Unsupervised (VICReg + dCor) |

### Proposed Improvement: Supervised Residual Targets

Add weak supervision on 8 residual dimensions using features not highly correlated with volume:

| Target | Feature | Source | Expected $|r|$ with volume |
|---|---|---|---|
| Sphericity | Shape compactness | `compute_sphericity()` | ~0.3 |
| Enhancement ratio | $V_{\text{ET}} / V_{\text{total}}$ | `compute_composition_features()` | ~0.2 |
| Infiltration index | $V_{\text{ED}} / (V_{\text{NCR}} + V_{\text{ET}})$ | `compute_composition_features()` | ~0.15 |

**Updated partition:**
```
z = [z_vol(32) | z_res_supervised(8) | z_res_free(88)]
```

**Where:** `src/growth/models/projection/sdp.py` and `src/growth/config/phase2_sdp.yaml`.

**Caveat:** This requires SDP retraining on BraTS-MEN (N=800, trivial compute). The SDP is trained on cross-sectional BraTS-MEN data, NOT on the longitudinal MenGrowth data.

---

## PCA Compression

### Why PCA on Residuals

The volume partition (32 dims) is already low-dimensional and supervised. The 96-dim residual partition at N=31 creates a 96:31 dimension-to-patient disaster for any GP. PCA reduces to the effective dimensionality.

### Protocol

```python
def compress_residuals_pca(
    z_residual_train: np.ndarray,    # [N_train_obs, 96]
    z_residual_test: np.ndarray,     # [n_test_obs, 96]
    variance_threshold: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, int]:
    """PCA compression of residual partition.

    Fit PCA on training data only (inside each LOPO fold).
    Transform both train and test.

    Returns:
        (z_train_reduced, z_test_reduced, k) where k = n_components.
    """
```

**Critical:** PCA must be fit INSIDE each LOPO fold on training patients only. Fitting on the full dataset is data leakage.

**Expected:** k = 5--15 components for 90% variance.

**Package:** `sklearn.decomposition.PCA(n_components=0.9)`.

---

## GP with ARD Kernel

### Kernel

$$k(\mathbf{x}, \mathbf{x}') = \sigma^2_f \prod_{d=1}^{D} k_{\text{Mat52}}\!\left(\frac{|x_d - x'_d|}{\ell_d}\right) + \sigma^2_n \delta(\mathbf{x}, \mathbf{x}')$$

where $D = 1 (\text{time}) + 1 (\text{volume}) + k (\text{PCA residuals})$. ARD assigns per-dimension lengthscale $\ell_d$. Irrelevant dimensions get large $\ell_d$ (soft feature selection).

**Parameter count:** $D$ lengthscales + 1 signal variance + 1 noise variance + mean function params. For D = 2 + k with k = 5: total ~10 hyperparameters. At ~112 observations, feasible with careful regularization.

**Package:** `GPy.kern.Matern52(input_dim=D, ARD=True)`.

### Mean Function

Linear in time only: $m(t) = \beta_0 + \beta_1 t$. Covariates (volume, PCA dims) enter through the kernel, not the mean.

### Implementation

```python
class DeepFeatureGP(GrowthModel):
    """GP growth model with deep feature inputs via ARD kernel.

    Input to GP: [time, log_volume, pca_res_1, ..., pca_res_k]
    Kernel: Matern-5/2 with per-dimension ARD lengthscales.
    Output: log(V + 1) at predicted times.

    Args:
        pca_variance_threshold: Cumulative variance threshold for PCA.
        kernel_type: Base kernel for ARD.
        n_restarts: Random restarts for hyperparameter optimization.
    """

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit GP with ARD on [time, volume, PCA_residuals].

        Steps:
          1. Extract z_residual from all training observations
          2. Fit PCA on training residuals
          3. Build input matrix X = [time, volume, PCA(residual)]
          4. Fit GP with ARD kernel via marginal likelihood
        """

    def predict(
        self, patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict by conditioning GP on patient's (time, volume, PCA_res) data."""
```

**Where:** `src/growth/models/growth/deep_feature_gp.py` (NEW).

---

## Integration with Stage 2 Severity Model

Deep features can improve severity estimation by replacing hand-crafted baseline features:

```
Baseline MRI --> BSF Encoder --> GAP --> h in R^768
  --> PCA --> h_tilde in R^10
  --> Severity regression: s_hat = sigma(w^T h_tilde + b)
  --> Feed s_hat into Stage 2 severity growth model
```

### Comparison Matrix

| Severity Input | Features | # Params |
|---|---|---|
| Clinical only | [log_vol, sphericity] | 3 |
| Clinical + demographics | [log_vol, age, sex, sphericity] | 5 |
| Deep features (PCA) | [PCA_1, ..., PCA_5] | 6 |
| Clinical + deep | [log_vol, sphericity, PCA_1, ..., PCA_3] | 6 |

Compare via leave-one-out R^2 of severity prediction.

---

## Existing Code References

This stage draws heavily on the existing module specifications:

| Module Spec | Relevant Content |
|---|---|
| `module_2_lora.md` | LoRA injection, checkpoint handling, merge protocol |
| `module_3_sdp.md` | SDP architecture, partition structure, loss functions |
| `module_4_encoding.md` | Cohort encoding, ComBat harmonization, trajectory building |

| Existing File | Purpose |
|---|---|
| `src/growth/models/encoder/swin_loader.py` | `load_swin_encoder()`, `create_swinunetr()` |
| `src/growth/models/encoder/feature_extractor.py` | `FeatureExtractor` (GAP pooling on encoder10) |
| `src/growth/models/encoder/lora_adapter.py` | `LoRASwinViT` (LoRA injection + merge) |
| `src/growth/models/projection/sdp.py` | SDP network (stub -- needs implementation) |
| `src/growth/evaluation/gp_probes.py` | `GPProbe`, `GPSemanticProbes` |
| `experiments/lora/engine/extract_features.py` | Feature extraction pipeline |
| `experiments/sdp/` | SDP training scripts |

---

## New Files to Create

1. **`src/growth/models/growth/deep_feature_gp.py`** -- GP with ARD on deep features.
2. **`experiments/deep_features/run_deep_prediction.py`** -- Stage 3 orchestrator.
3. **`experiments/deep_features/config.yaml`** -- Configuration.

---

## Configuration

```yaml
# experiments/deep_features/config.yaml
experiment:
  name: stage3_deep_features
  seed: 42

paths:
  mengrowth_h5: /path/to/MenGrowth.h5
  bratsmen_h5: /path/to/brats_men_train.h5
  encoder_checkpoint: /path/to/phase1_encoder_merged.pt
  sdp_checkpoint: /path/to/phase2_sdp.pt
  output_dir: experiments/deep_features/results
  stage1_results: experiments/segment_based_approach/results
  stage2_results: experiments/severity_model/results

encoder:
  lora_ranks_to_test: [2, 4, 8, 16, 32]
  selection_metric: gp_probe_r2_volume
  feature_extraction:
    roi_size: [192, 192, 192]  # FEATURE_ROI_SIZE for complete tumor containment
    deterministic: true

sdp:
  in_dim: 768
  hidden_dim: 512
  out_dim: 128
  dropout: 0.1
  partition:
    vol_dim: 32
    residual_dim: 96
    # Optional supervised residual (requires SDP retraining):
    # residual_supervised_dim: 8
    # residual_free_dim: 88

pca:
  variance_threshold: 0.9   # Retain 90% variance
  fit_per_fold: true         # CRITICAL: fit inside each LOPO fold

gp_ard:
  kernel: matern52
  ard: true
  mean_function: linear      # Linear in time only
  n_restarts: 5
  max_iter: 1000
  lengthscale_bounds: [0.01, 100.0]
  signal_var_bounds: [0.001, 10.0]
  noise_var_bounds: [0.000001, 5.0]

severity_integration:
  enabled: true
  pca_dims_for_severity: 5
  compare_to_clinical: true

patients:
  exclude: [MenGrowth-0028]
  min_timepoints: 2

bootstrap:
  enabled: true
  n_samples: 2000
  method: bca
  seed: 42

comparison:
  permutation_test_n: 10000
```

---

## Critical Caveat

At N=31 with ~112 total observations, the GP with ARD on D=2+k inputs (k ~ 5--15) has D+2 kernel hyperparameters (7--17). The parameter-to-observation ratio is 1:7 to 1:16, which is feasible but requires:

1. Informative priors on lengthscales (or tight bounds)
2. Multiple random restarts (n_restarts >= 5)
3. Regularization via noise variance lower bound
4. Monitoring for lengthscale collapse (all $\ell_d \to \infty$ means no signal)

If ARD assigns large lengthscales to ALL residual PCA dimensions, this is the model's way of saying deep features add no information beyond volume -- a scientifically honest and important result.

---

## Verification Tests

```
S3-T1: GP probe R^2 per LoRA rank computed [BLOCKING]
  - Extract features for each LoRA rank on MenGrowth data
  - Run GPProbe with linear and RBF kernels for volume prediction
  - Report R^2 per rank; identify best rank
  - Assert at least one rank produces R^2 > 0
  Recovery: Check encoder loading; verify feature dimensions are 768

S3-T2: SDP quality: dCor(vol, residual) < 0.15 [BLOCKING]
  - Compute distance correlation between z_vol and z_residual
  - Assert dCor < 0.15 (partitions are decorrelated)
  - If violated, increase lambda_dcor and retrain SDP
  Recovery: Retrain SDP with lambda_dcor = 5.0; check for collapsed dims

S3-T3: PCA on residual: k components for 90% variance [DIAGNOSTIC]
  - Fit PCA on training-fold residuals
  - Report k and per-component explained variance
  - Assert k < 96 (PCA actually compresses)
  - Report k across LOPO folds (should be stable, std(k) < 3)
  Note: DIAGNOSTIC -- k is informative, not pass/fail

S3-T4: LOPO-CV with [volume + PCA_residual] >= Stage 1 R^2 [BLOCKING for Stage 3 justification]
  - Run full LOPO-CV with DeepFeatureGP
  - Compare R^2_log to Stage 1 best and Stage 2
  - Compute paired permutation test p-values
  - If p > 0.05 vs both Stage 1 and Stage 2, Stage 3 does NOT justify complexity
  Recovery: Try fewer PCA components; try volume-only GP as sanity check

S3-T5: Severity estimation: deep features vs clinical [DIAGNOSTIC]
  - Compare LOO R^2 for severity prediction from:
    clinical-only, clinical+deep, deep-only
  - Report delta R^2 for each comparison
  Note: DIAGNOSTIC -- may reveal that deep features are redundant with volume

S3-T6: ARD lengthscales reveal which residual dims matter [DIAGNOSTIC]
  - Report per-dimension lengthscales from fitted GP
  - Compute relevance = 1 / lengthscale for each dim
  - Rank dimensions by relevance
  - If all residual dims have relevance < 0.01 * time_relevance,
    deep features add negligible signal
  Note: DIAGNOSTIC -- this is the key scientific finding regardless of outcome
```

---

## References

- BrainSegFounder: SSL pretrained SwinUNETR on 41K+ brain MRIs.
- Module 2 spec: `docs/growth-related/claude_files_BSGNeuralODE/module_2_lora.md`
- Module 3 spec: `docs/growth-related/claude_files_BSGNeuralODE/module_3_sdp.md`
- Module 4 spec: `docs/growth-related/claude_files_BSGNeuralODE/module_4_encoding.md`
- Rasmussen, C. E. & Williams, C. K. I. *Gaussian Processes for Machine Learning*, MIT Press, 2006.
- Neal, R. M. "Bayesian Learning for Neural Networks," Lecture Notes in Statistics 118, Springer, 1996. (ARD priors)
- Wipf, D. P. & Nagarajan, S. S. "A new view of automatic relevance determination," *NeurIPS*, 2008.
