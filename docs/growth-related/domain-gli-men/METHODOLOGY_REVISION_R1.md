# Methodology Revision R1: Volume-Focused SDP and Simplified GP Hierarchy

**Date:** 2026-03-09  
**Author:** Mario Pascual-González  
**Status:** Approved for implementation  
**Scope:** Phase 1 (LoRA), Phase 2 (SDP), Phase 4 (GP) code and documentation  

---

## 0. Executive Summary

Following the dual-domain LoRA adaptation experiment (Dual LoRA r=8), we revise the
methodology based on empirical findings. Three changes:

1. **Drop shape and location partitions from the SDP latent space.** Shape R² ≤ 0.11
   across all conditions (negative cross-domain). Location is temporally static for
   meningiomas and belongs as a GP covariate, not a temporal latent.
2. **Reduce the volume supervision target from 4 sub-volumes to 1 (whole-tumor
   log-volume).** Sub-compartment volumes (NCR, ED, ET) are label-dependent and
   incompatible across tumour types. Whole-tumour volume is the clinically relevant
   growth endpoint.
3. **Simplify the GP hierarchy from PA-MOGP to standard MOGP.** With a single
   supervised partition, partition-aware coupling is unnecessary.

Location (centroid) and optional sub-volumes are retained as **static covariates** to
the GP mean function, not as temporal latent dimensions.

---

## 1. Empirical Motivation

### 1.1 Shape partition failure

| Condition       | Shape R² (Linear) | Shape R² (RBF) | Cross-domain GLI→MEN |
|-----------------|-------------------|----------------|----------------------|
| Baseline        | 0.08              | ≈ 0            | −2.00                |
| MEN LoRA r=8    | 0.11              | ≈ 0            | −2.00                |
| Dual LoRA r=8   | −0.06             | ≈ 0.15         | −2.00                |

Shape features (sphericity, enhancement_ratio, infiltration_index) are not linearly
decodable from the encoder across any condition. The negative R² values indicate
predictions with higher MSE than a constant-mean baseline. Cross-domain shape transfer
is uniformly catastrophic (R² = −2.00), making shape incompatible with the dual-domain
GP transfer strategy. Biologically, shape semantics differ qualitatively between
glioma (infiltrative, irregular) and meningioma (well-circumscribed, dural-based),
making a shared representation a category error.

### 1.2 Location partition: temporally static

Meningiomas are extra-axial, dural-attached tumours. They do not migrate. The centroid
coordinates at t₀ are essentially identical at t₁, t₂, … (modulo asymmetric growth).
Modelling location as a time-varying GP process fits a constant function, which is
statistically vacuous and wastes degrees of freedom. Location is better treated as a
static covariate in the GP mean function (see Section 4).

Within-domain location R² is decent (0.35 MEN, 0.68 GLI), so the information is
retained — just relocated from temporal latent to static covariate.

### 1.3 Sub-volume target incompatibility

The current volume target is a 4-vector: `[log(V_total+1), log(V_NCR+1),
log(V_ED+1), log(V_ET+1)]`. This is problematic:

- **Label semantics differ across tumour types.** BraTS-MEN label 1 (NCR) is rare;
  most meningioma volume is ET (label 3). BraTS-GLI label 2 (ED) represents
  infiltrative oedema; in MEN it represents reactive oedema. Forcing a shared
  sub-volume representation across domains conflates biologically distinct entities.
- **Segmentation quality varies per subregion.** Our Dual LoRA model achieves TC Dice
  0.40 (MEN) and ET Dice 0.62 (MEN) — the sub-compartment labels used to compute
  NCR/ED/ET volumes are themselves noisy.
- **Whole-tumour volume is the clinical endpoint.** Growth prediction in meningioma
  monitoring uses total tumour volume or maximum diameter. No clinical decision is
  based on NCR or ED sub-volumes for meningiomas.
- **The Dual encoder already achieves CV R² = 0.712 for total volume (MEN).** This
  is already a strong signal from the frozen encoder features alone.

### 1.4 Dual-domain adaptation success (justifies Dual LoRA r=8 as encoder)

| Metric               | Base   | MEN LoRA | Dual LoRA | Change (Base→Dual) |
|----------------------|--------|----------|-----------|-------------------|
| MEN Mean Dice        | 0.52   | 0.48     | 0.57      | +9.6%             |
| Volume R² (MEN)      | 0.56   | 0.57     | 0.62      | +10.7%            |
| Volume CV R² (MEN)   | —      | —        | 0.712     | Best              |
| MMD²                 | 0.118  | 0.116    | 0.037     | −68.6%            |
| CKA (MEN vs GLI)     | 0.002  | 0.002    | 0.017     | +750%             |
| Effective Rank       | 47.6   | 45.8     | 25.4      | −46.6%            |

The Dual LoRA r=8 encoder is selected as the frozen encoder for all downstream phases.

---

## 2. Revised Architecture Overview

### 2.1 Before (3+1 partition SDP)

```
encoder10 features ∈ ℝ⁷⁶⁸
    ↓ SDP (768 → 512 → 128)
    ↓
z ∈ ℝ¹²⁸ = [z_vol ∈ ℝ²⁴ | z_loc ∈ ℝ⁸ | z_shape ∈ ℝ¹² | z_res ∈ ℝ⁸⁴]
    ↓                ↓              ↓
π_vol → ℝ⁴       π_loc → ℝ³    π_shape → ℝ³
(log-vols)        (centroid)     (shape feats)
```

### 2.2 After (1+1 partition SDP)

```
encoder10 features ∈ ℝ⁷⁶⁸
    ↓ SDP (768 → 512 → 128)
    ↓
z ∈ ℝ¹²⁸ = [z_vol ∈ ℝ³² | z_res ∈ ℝ⁹⁶]
    ↓
π_vol → ℝ¹
(log whole-tumour volume)
```

Location centroid ∈ ℝ³ is computed from the segmentation mask at inference time and
injected as a static covariate into the GP mean function.

---

## 3. Refactoring Goals

Each goal is an atomic, verifiable unit of work. Goals are ordered by dependency.
Goals within the same phase can be parallelized if they touch disjoint files.

---

### GOAL 1: Update LoRA semantic heads to volume-only

**Phase:** 1 (LoRA)  
**Rationale:** The auxiliary semantic heads during LoRA training currently predict
volume (4), location (3), and shape (3). We simplify to volume-only with a single
whole-tumour volume target.

**Files to modify:**

1. `src/growth/models/segmentation/semantic_heads.py`
   - `AuxiliarySemanticHeads.__init__`: Remove `location_head` and `shape_head`.
     Keep only `volume_head`. Change `volume_dim` default from 4 to 1.
   - `AuxiliarySemanticHeads.forward`: Return only `{'pred_volume': ...}`.
   - `AuxiliarySemanticHeads.dims`: Only `{'volume': 1}`.
   - `AuxiliarySemanticLoss.__init__`: Remove `lambda_location`, `lambda_shape`.
     Remove `location_mean/std`, `shape_mean/std` buffers.
   - `AuxiliarySemanticLoss.update_statistics`: Accept only `volume: Tensor`.
   - `AuxiliarySemanticLoss.forward`: Compute MSE only for volume. Remove
     location and shape branches.
   - `MultiScaleSemanticHeads`: If it exists and references location/shape, apply
     the same simplification. Otherwise leave untouched.

2. `experiments/lora/config/local/LoRA_semantic_heads_icai.yaml`  
   `experiments/lora/config/server/LoRA_semantic_heads_icai.yaml`
   - Under `loss:`: Remove `lambda_location` and `lambda_shape` keys.
   - Rename `lambda_volume` to `lambda_volume` (keep, value 1.0).
   - Add a comment: `# Volume target: log(V_WT + 1), scalar`.

3. Any LoRA training engine files that construct the `targets` dict with
   `'volume'`, `'location'`, `'shape'` keys — modify to only provide `'volume'`.
   Search for all callers of `AuxiliarySemanticLoss.forward()` and
   `AuxiliarySemanticHeads.forward()`. The volume target tensor must change shape
   from `[B, 4]` to `[B, 1]`.

**How volume target is computed:**  
In the dataset (`src/growth/data/bratsmendata.py`), semantic features are loaded from
HDF5 as `f["semantic/volume"][idx]` which returns shape `[4]`. The new target should
be `volume[0:1]` (the first element is `log1p(V_total)`). Do NOT change the HDF5
schema — just index at the point of use.

**Verification:**
- [ ] `AuxiliarySemanticHeads(input_dim=768)` constructs with only a volume head.
- [ ] `heads(torch.randn(4, 768))` returns dict with single key `'pred_volume'`
      and value shape `[4, 1]`.
- [ ] `AuxiliarySemanticLoss` computes a scalar loss from volume-only predictions.
- [ ] LoRA config YAML files parse without error and contain no location/shape keys.
- [ ] Running one training step on a dummy batch completes without error.

---

### GOAL 2: Update SDP partition layout to volume + residual only

**Phase:** 2 (SDP)  
**Rationale:** Remove shape and location partitions. Expand volume to 32 dims and
residual to 96 dims. Reduce volume target dimensionality from 4 to 1.

**Files to modify:**

1. `src/growth/models/projection/partition.py`
   - Replace `DEFAULT_PARTITIONS` dict:
     ```python
     DEFAULT_PARTITIONS: dict[str, PartitionSpec] = {
         "vol": PartitionSpec(name="vol", start=0, end=32, target_dim=1),
         "residual": PartitionSpec(name="residual", start=32, end=128, target_dim=None),
     }
     ```
   - Replace `SUPERVISED_PARTITIONS`:
     ```python
     SUPERVISED_PARTITIONS: list[str] = ["vol"]
     ```
   - Update `LatentPartition.from_config()`: Remove `loc_dim`, `shape_dim`,
     `n_loc`, `n_shape` parameters. New signature:
     ```python
     @classmethod
     def from_config(
         cls,
         vol_dim: int = 32,
         residual_dim: int = 96,
         n_vol: int = 1,
     ) -> "LatentPartition":
     ```
     Build only `"vol"` and `"residual"` partition specs.
   - Update module docstring to reflect 2-partition layout.

2. `src/growth/models/projection/semantic_heads.py`
   - Rename class to `SemanticHead` (singular) or keep `SemanticHeads` with only
     a volume head. Remove `loc_head` and `shape_head`.
   - Constructor: only `vol_in: int = 32, vol_out: int = 1`.
   - `forward()`: Accept `partitions` dict, return `{"vol": self.vol_head(partitions["vol"])}`.

3. `src/growth/models/projection/sdp.py`
   - `SDPWithHeads.from_config()`: Remove `loc_dim`, `shape_dim`, `n_loc`,
     `n_shape` parameters. New defaults:
     ```python
     vol_dim: int = 32,
     residual_dim: int = 96,
     n_vol: int = 1,
     ```
   - Update the `from_config` body to only construct volume head.

4. `src/growth/models/projection/__init__.py`
   - No structural change needed, but verify `__all__` exports are still valid.

5. `src/growth/config/phase2_sdp.yaml`
   - Update partition section:
     ```yaml
     partition:
       vol_dim: 32
       residual_dim: 96
     ```
   - Update targets section:
     ```yaml
     targets:
       n_vol: 1    # log(V_WT + 1), whole-tumour volume only
     ```
   - Update loss section: Remove `lambda_loc` and `lambda_shape`.

**Verification:**
- [ ] `LatentPartition()` creates 2 partitions: vol (0–32), residual (32–128).
- [ ] `LatentPartition().split(torch.randn(8, 128))` returns dict with keys
      `{"vol", "residual"}` and shapes `[8, 32]`, `[8, 96]`.
- [ ] `SDPWithHeads.from_config()` constructs and `forward(torch.randn(8, 768))`
      returns `(z, partitions, predictions)` with `predictions["vol"].shape == [8, 1]`.
- [ ] YAML config parses and matches new defaults.
- [ ] `32 + 96 == 128` (partition dimensions sum to output dim).

---

### GOAL 3: Update SDP loss functions for single supervised partition

**Phase:** 2 (SDP)  
**Depends on:** Goal 2

**Rationale:** With only one supervised partition, the cross-partition covariance loss
and distance correlation loss reduce to vol↔residual independence only. The semantic
regression loss simplifies to a single MSE term.

**Files to modify:**

1. `src/growth/losses/semantic.py`
   - `SemanticRegressionLoss.__init__`: Remove `lambda_loc` and `lambda_shape`.
     Keep only `lambda_vol`.
   - `SemanticRegressionLoss.lambdas`: `{"vol": lambda_vol}`.
   - `SemanticRegressionLoss.forward`: Iterate over `["vol"]` only.

2. `src/growth/losses/vicreg.py`
   - `CovarianceLoss.forward`: The `partition_names` default should become
     `("vol",)`. With a single supervised partition, the cross-partition covariance
     is between `vol` and `residual`. Update the default to `("vol", "residual")`.
     This ensures the covariance penalty still enforces independence between the
     supervised volume subspace and the residual.
   - `VarianceHingeLoss`: No change needed (operates on full z).

3. `src/growth/losses/dcor.py`
   - `DistanceCorrelationLoss.__init__`: Update default `partition_names` from
     `("vol", "loc", "shape")` to `("vol", "residual")`.
   - With only 1 pair (vol, residual), C(2,2) = 1 pair. The mean dCor is just
     the single pairwise distance correlation.

4. `src/growth/losses/sdp_loss.py`
   - `SDPLoss.__init__`: Remove `lambda_loc` and `lambda_shape` parameters.
     Update `cov_partition_names` default to `("vol", "residual")`.
   - `SDPLoss.__init__` → `SemanticRegressionLoss` constructor: pass only
     `lambda_vol`.
   - Docstring: Update to reflect 2-partition architecture.
   - `CurriculumSchedule`: No change needed (schedule logic is independent of
     partition count).

5. `src/growth/config/phase2_sdp.yaml`
   - Already addressed in Goal 2 (loss section).

**Verification:**
- [ ] `SemanticRegressionLoss(lambda_vol=20.0)` constructs without error.
- [ ] `SemanticRegressionLoss.forward({"vol": randn(8,1)}, {"vol": randn(8,1)})`
      returns valid (loss, details) tuple.
- [ ] `CovarianceLoss().forward({"vol": randn(8,32), "residual": randn(8,96)},
      partition_names=["vol", "residual"])` returns scalar loss.
- [ ] `DistanceCorrelationLoss().forward({"vol": randn(8,32), "residual": randn(8,96)})`
      returns (mean_dcor, details) with 1 pair.
- [ ] `SDPLoss()` composes all sub-losses and `forward()` returns valid total.

---

### GOAL 4: Update SDP dataset / data loading for WT-only volume target

**Phase:** 2 (SDP)  
**Depends on:** Goal 2

**Rationale:** The SDP training loop must provide `targets = {"vol": tensor[B, 1]}`
instead of `{"vol": tensor[B, 4], "loc": tensor[B, 3], "shape": tensor[B, 3]}`.

**Files to modify:**

1. `src/growth/data/bratsmendata.py`
   - In `__getitem__`, the `semantic_features` dict currently provides
     `"volume"` as shape `[4]`, `"location"` as `[3]`, `"shape"` as `[3]`.
   - Change `"volume"` to return only the first element (log total volume):
     ```python
     volume = f["semantic/volume"][h5_idx]  # [4] from HDF5
     output["semantic_features"] = {
         "volume": torch.tensor([volume[0]], dtype=torch.float32),  # [1] — log(V_WT + 1)
         "location": torch.from_numpy(location.astype(np.float32)),  # [3] — KEEP for static covariate
         "all": torch.tensor([volume[0]], dtype=torch.float32),
     }
     ```
   - **Important:** Keep loading `location` from H5 and include it in the output
     dict. It will be used as a static GP covariate (Goal 6). Do NOT load shape.
   - Remove the `"shape"` key from the semantic features dict.

2. Any SDP training script that constructs `targets` from the dataset output —
   ensure it maps `semantic_features["volume"]` → `targets["vol"]`.
   Search for references to `targets["loc"]` and `targets["shape"]` in training
   loops and remove them.

3. `src/growth/training/callbacks/semantic_metrics.py`
   - If this callback logs R² for location and shape, remove those metrics.
   - Keep only volume R² logging.

**Do NOT modify:**
- `src/growth/data/semantic_features.py` — the feature extraction functions are
  general-purpose and should remain capable of computing all features.
- HDF5 files — no schema changes.

**Verification:**
- [ ] `dataset[0]["semantic_features"]["volume"].shape == torch.Size([1])`.
- [ ] `dataset[0]["semantic_features"]` has no `"shape"` key.
- [ ] `dataset[0]["semantic_features"]["location"].shape == torch.Size([3])`.
- [ ] SDP training loop constructs `targets = {"vol": [B, 1]}` correctly.

---

### GOAL 5: Update LoRA probe evaluation for volume-only

**Phase:** 1 (LoRA)  
**Depends on:** Goal 1

**Rationale:** Probe evaluation during LoRA experiments currently evaluates linear
and RBF probes for volume, location, and shape. Simplify to volume-only probing
(WT log-volume). Keep location probing as a diagnostic but remove shape probing.

**Files to modify:**

1. Any probe evaluation scripts under `experiments/lora/` that iterate over
   `["volume", "location", "shape"]` targets. Change to `["volume"]` (or
   `["volume", "location"]` if location diagnostics are desired).

2. The volume probe target must be `[N, 1]` (log total volume) instead of
   `[N, 4]`. At the point where `semantic/volume` is loaded from H5,
   index `[:, 0:1]`.

3. Probe evaluation configs — remove shape-related config entries.

4. Report generation scripts (e.g., `experiments/lora/report/narrative.py`) —
   if they reference shape R², update narrative to reflect volume-only evaluation.

**Verification:**
- [ ] Probe evaluation runs on extracted features and reports only volume R².
- [ ] Probe target shape is `[N, 1]` for volume.
- [ ] No references to shape probing remain in evaluation code.

---

### GOAL 6: Update GP hierarchy documentation (Phase 4)

**Phase:** 4 (GP) — docs only, no code changes yet  
**Depends on:** Goals 2–4

**Rationale:** The GP hierarchy must be updated to reflect the simplified SDP output
and the reclassification of location from temporal latent to static covariate. Code
changes to Phase 4 models will happen when Phase 4 implementation begins; this goal
updates the design documents and configuration to ensure consistency.

**Files to modify:**

1. `src/growth/config/phase4_growth.yaml`
   - Under `pamogp:`: Remove `loc_kernel` and `shape_kernel`. Remove
     `coupling_rank` (no cross-partition coupling with single partition).
   - Rename `pamogp` section to `mogp` (multi-output GP, no longer
     partition-aware).
   - Add `static_covariates` section:
     ```yaml
     static_covariates:
       location: true      # centroid [cz, cy, cx] from initial scan
       # Future: age, sex, tumour grade if available
     ```
   - The `mogp` kernel becomes a single Matérn-5/2 for all 32 volume dimensions.

2. `src/growth/models/growth/pamogp_model.py`
   - Update module docstring to reflect the PA-MOGP → MOGP simplification.
   - Document that the model now operates on `z_vol ∈ ℝ³²` only.
   - Note: code refactoring of this file is deferred to Phase 4 implementation.
     For now, update comments and docstrings.

3. `src/growth/models/growth/hgp_model.py`
   - Update docstring: H-GP now operates on `z_vol ∈ ℝ³²` (was `z_vol ∈ ℝ²⁴`).

4. `src/growth/evaluation/growth_figures.py`
   - Update figure descriptions: remove references to cross-partition coupling
     heatmap (Figure 13). Replace with calibration plot or similar.

5. `src/growth/evaluation/growth_metrics.py`
   - Verify that metrics reference volume prediction only. No changes expected
     since metrics already focus on volume.

6. `src/growth/data/trajectory_dataset.py`
   - Update docstring to note that trajectories now consist of 32-dim volume
     latent vectors + static covariates.

**Revised GP mean function specification:**

For the LME (Model A):
```
m_i(t) = β₀ + β₁ · t + γᵀ · c_i
```
where `c_i = [cz_i, cy_i, cx_i]` is the centroid from patient i's initial scan,
`β₀, β₁` are fixed effects (intercept + slope), and `γ ∈ ℝ³` captures location
effects on baseline volume trajectory.

For the H-GP (Model B):
```
f_i(t) | c_i ~ GP(m_i(t), k(t, t'; θ))
m_i(t) = β₀_i + β₁_i · t + γᵀ · c_i     (LME-derived, with random slopes)
k(t, t'; θ) = σ² · Matérn_{5/2}(|t - t'| / ℓ) + σ²_n · δ(t, t')
```

For the MOGP (Model C, formerly PA-MOGP):
```
f_i(t) ∈ ℝ³² ~ MOGP(m_i(t), K(t, t'))
K(t, t') = (B ⊗ k_t(t, t'))         [ICM kernel, B ∈ ℝ³²ˣ³² PSD]
k_t(t, t') = σ² · Matérn_{5/2}(|t - t'| / ℓ)
```
where `B = W Wᵀ + diag(κ)` is the output coregionalization matrix (Álvarez et al.,
2012, *Foundations and Trends in Machine Learning*).

**Verification:**
- [ ] `phase4_growth.yaml` parses without error and contains no shape/loc kernel refs.
- [ ] All GP model docstrings reference `z_vol ∈ ℝ³²` and mention static covariates.
- [ ] No references to "partition-aware" remain in Phase 4 documentation.

---

### GOAL 7: Update `__init__.py` exports and clean up dead imports

**Phase:** All  
**Depends on:** Goals 1–6

**Rationale:** After removing location and shape from multiple modules, any `__init__.py`
that re-exports removed classes/constants must be updated. Dead imports in training
scripts and evaluation code must be cleaned.

**Files to check:**

1. `src/growth/models/projection/__init__.py` — verify exports match updated module contents.
2. `src/growth/losses/__init__.py` (if exists) — verify exports.
3. `src/growth/models/segmentation/__init__.py` (if exists).
4. All `experiments/lora/` scripts — grep for `shape`, `location`, `loc_head`,
   `shape_head`, `lambda_loc`, `lambda_shape`, `n_loc`, `n_shape` and remove
   dead references.
5. All `experiments/sdp/` scripts (if they exist yet).

**Verification:**
- [ ] `python -c "from growth.models.projection import SDPWithHeads, LatentPartition"` succeeds.
- [ ] `python -c "from growth.losses.sdp_loss import SDPLoss"` succeeds.
- [ ] `grep -r "lambda_shape\|lambda_loc\|n_shape\|n_loc\|shape_head\|loc_head" src/ experiments/` returns zero results (excluding this document and git history).

---

## 4. Files NOT Modified (Intentional Preservation)

| File | Reason |
|------|--------|
| `src/growth/data/semantic_features.py` | General-purpose feature extraction. Retains ability to compute all features. |
| HDF5 files (`BraTS_MEN.h5`, `BraTS_GLI.h5`, `MenGrowth.h5`) | Schema unchanged. Indexing at point of use. |
| `src/growth/losses/encoder_vicreg.py` | Operates on raw 768-dim encoder features. Independent of partition layout. |
| `src/growth/models/segmentation/model.py` (or equivalent) | Segmentation architecture unchanged. |
| Phase 3 (ComBat) code | Not yet implemented; will be designed with volume-only assumption. |

---

## 5. Mathematical Summary of Revised SDP

### 5.1 Projection

$$
\mathbf{z} = f_\theta(\mathbf{h}), \quad f_\theta: \mathbb{R}^{768} \to \mathbb{R}^{128}
$$

with `f_θ = Linear(512, 128) ∘ GELU ∘ Linear(768, 512) ∘ LayerNorm`, spectral
normalization on both linear layers.

### 5.2 Partition

$$
\mathbf{z} = [\mathbf{z}_\text{vol} \in \mathbb{R}^{32} \;|\; \mathbf{z}_\text{res} \in \mathbb{R}^{96}]
$$

### 5.3 Semantic head

$$
\hat{v} = \pi_\text{vol}(\mathbf{z}_\text{vol}) = \mathbf{W}_v \mathbf{z}_\text{vol} + \mathbf{b}_v, \quad \mathbf{W}_v \in \mathbb{R}^{1 \times 32}, \; \hat{v} \in \mathbb{R}^1
$$

Target: $v^* = \log(V_\text{WT} + 1)$, where $V_\text{WT} = V_\text{NCR} + V_\text{ED} + V_\text{ET}$ in mm³.

### 5.4 Loss function

$$
\mathcal{L} = \underbrace{\lambda_\text{vol} \cdot \text{MSE}(\hat{v}, v^*)}_{\text{Informativeness}} + \underbrace{\lambda_\text{var} \cdot \mathcal{L}_\text{var}(\mathbf{z})}_{\text{Collapse prevention}} + \underbrace{\lambda_\text{cov} \cdot \mathcal{L}_\text{cov}(\mathbf{z}_\text{vol}, \mathbf{z}_\text{res})}_{\text{Linear independence}} + \underbrace{\lambda_\text{dcor} \cdot \text{dCor}(\mathbf{z}_\text{vol}, \mathbf{z}_\text{res})}_{\text{Nonlinear independence}}
$$

With curriculum schedule (unchanged):
- Epochs 0–9: $\mathcal{L}_\text{var}$ only
- Epochs 10–39: + $\mathcal{L}_\text{vol}$
- Epochs 40–59: + $\mathcal{L}_\text{cov}$, $\mathcal{L}_\text{dcor}$
- Epochs 60+: All at full strength

### 5.5 Hyperparameters

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `vol_dim` | 24 | 32 | Absorbs freed capacity |
| `residual_dim` | 84 | 96 | Absorbs freed capacity |
| `n_vol` | 4 | 1 | WT-only volume |
| `lambda_vol` | 20.0 | 25.0 | Increased weight — single target needs stronger signal |
| `lambda_loc` | 12.0 | **removed** | |
| `lambda_shape` | 15.0 | **removed** | |
| `lambda_cov` | 5.0 | 5.0 | Unchanged |
| `lambda_var` | 5.0 | 5.0 | Unchanged |
| `lambda_dcor` | 2.0 | 2.0 | Unchanged |

**Note on `lambda_vol = 25.0`:** With a single scalar target, the MSE gradient has
lower magnitude than with a 4-vector. Increasing lambda compensates. This should
be validated empirically — if training is unstable, reduce to 20.0.

---

## 6. Literature References

- Bardes et al. (2022). "VICReg: Variance-Invariance-Covariance Regularization for
  Self-Supervised Learning." *ICLR 2022*.
- Locatello et al. (2019). "Challenging Common Assumptions in the Unsupervised
  Learning of Disentangled Representations." *ICML 2019*.
- Székely et al. (2007). "Measuring and Testing Dependence by Correlation of
  Distances." *Annals of Statistics* 35(6): 2769–2794.
- Álvarez et al. (2012). "Kernels for Vector-Valued Functions: A Review."
  *Foundations and Trends in Machine Learning* 4(3): 195–266.
- Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
- Ben-David et al. (2010). "A Theory of Learning from Different Domains."
  *Machine Learning* 79(1–2): 151–175.
- Hashimoto et al. (2012). "Natural History of Incidental Meningiomas."
  *Neurologia Medico-Chirurgica* 52(6): 374–378.
- Ius et al. (2020). "Management of Incidental Meningiomas."
  *World Neurosurgery* 141: e501–e509.

---

## 7. Implementation Order

```
GOAL 1 (LoRA semantic heads) ──→ GOAL 5 (LoRA probes)
                                        │
GOAL 2 (SDP partitions)  ──→ GOAL 3 (SDP losses) ──→ GOAL 7 (cleanup)
                          ──→ GOAL 4 (SDP data)   ──↗
                                        │
                                  GOAL 6 (GP docs)
```

**Critical path:** GOAL 2 → GOAL 3 → GOAL 7 (SDP partition changes gate everything).
GOAL 1 can proceed in parallel with GOAL 2 since they touch disjoint file sets.
