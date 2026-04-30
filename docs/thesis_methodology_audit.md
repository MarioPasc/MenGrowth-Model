# Thesis Methodology Audit — Segmentation & Pre-segmentation

Cross-reference of `sections/methods/segmentation.tex` and `sections/methods/mengroth_preprocessing.tex` (pre-segmentation only) against the actual codebase.
Findings are ranked by severity: **HIGH** = a reviewer would flag this; **MEDIUM** = reproducibility gap; **LOW** = minor omission.

---

## HIGH — Would be flagged in peer review

### 1. Output head is re-initialised, not "unfrozen"

**Thesis (line 67):** "a single output head [...] is unfrozen."

**Code reality:** The pretrained BrainSegFounder has 4 output channels (BraTS-GLI). Our model uses 3 (TC/WT/ET). Because the channel count differs, the output head is **replaced** with a fresh `nn.Sequential(nn.Conv3d(48, 3, kernel_size=1))` using PyTorch default Kaiming uniform initialisation — it is not "unfrozen" from pretrained weights; it is created from scratch.

**Why it matters:** The output head starts from random init while LoRA adapters start from zero-init (Hu et al. 2022). This asymmetry is the stated rationale for the 10x higher learning rate, but the thesis frames this as "smaller parameter count" (line 138). A reviewer familiar with LoRA would want to know the head is freshly initialised.

**Code path:** `src/growth/models/segmentation/original_decoder.py:344-348`

---

### 2. Training-time validation uses 128^3 centre crop, not sliding window

**Thesis (line 156):** "Early stopping halts training when the validation Dice score [...] does not improve."

**Code reality:** Training-time validation runs a **direct forward pass** on 128^3 centre-cropped volumes (`ResizeWithPadOrCropd` to 128^3). Test-time inference uses **192^3 sliding window** with 50% overlap and Gaussian blending. No test-time augmentation is applied during validation.

**Why it matters:** The training validation Dice is computed on a single 128^3 crop that may miss peripheral tumour tissue (the H5 data is 192^3). Test Dice uses the full volume via sliding window. These are not directly comparable, and the gap should be disclosed so the reader understands what the early-stopping criterion actually optimises.

**Code path:** `experiments/uncertainty_segmentation/engine/train_member.py:188-258` (validation loop); config `val_roi_size: [128,128,128]`

---

### 3. Pre-segmentation model is unspecified

**Thesis (line 338):** "initial tumour segmentation masks are generated using the winning model of the BraTS 2025 Meningioma Segmentation Challenge."

**Code reality:** No pre-segmentation code exists in the repository. No reference to a BraTS 2025 winning model's architecture, training data, or inference protocol. The pre-segmentation was performed externally.

**Why it matters:** The ground-truth labels for the MenGrowth cohort depend on this model's output + manual refinement. A reviewer would expect: (a) a citation or architectural description of the winning model, (b) its reported performance on the BraTS-MEN benchmark, and (c) a description of the manual refinement protocol (who annotated, how many annotators, inter-rater agreement, which tool). Without these, the label quality of the private cohort is unverifiable.

**Code path:** N/A — no implementation found in the repository

---

### 4. Per-member connected-component cleanup before volume computation

**Thesis (line 178):** "A connected-component cleanup then removes spurious predictions smaller than 64 voxels."

**Code reality:** Connected-component cleanup is applied **twice**: once **per-member** (before computing each member's volume), and again to the final ensemble mask. The thesis describes it only in the postprocessing paragraph after ensemble aggregation, implying it applies only to the ensemble mask.

**Why it matters:** Per-member CC cleanup affects the per-member volumes and therefore the volume variance estimate sigma^2_{v,k}. If a member has a small spurious island that gets removed, its volume drops, increasing inter-member variance. This is a deliberate design choice that should be stated.

**Code path:** `experiments/uncertainty_segmentation/engine/ensemble_inference.py:299-301` (per-member), `ensemble_inference.py:327-330` (ensemble)

---

## MEDIUM — Reproducibility gaps

### 5. Augmentation hyperparameters not specified

**Thesis (lines 150-151):** "(vii) additive Gaussian noise, and (viii) Gaussian smoothing for simulated resolution variation."

**Code reality:** These augmentations have specific hyperparameters:
- Gaussian noise: sigma = 0.05, probability = 0.15
- Gaussian smoothing: sigma range [0.5, 1.0] (per axis), probability = 0.1

**Why it matters:** These parameters affect the degree of regularisation and domain robustness. A reader attempting to reproduce the method cannot set them from the current text.

**Code path:** `src/growth/data/transforms.py:87-93` (defaults); `experiments/uncertainty_segmentation/config.yaml` (augmentation flags)

---

### 6. LoRA adapts the combined QKV projection, not separate Q/K/V

**Thesis (line 74):** "`qkv`: the combined query-key-value projection in the multi-head self-attention module"

**Assessment:** The thesis does say "combined" — so this is technically described. However, it does not explain the **implication**: a rank-r adapter on the combined QKV matrix (dimensions d x 3d) allocates capacity jointly across Q, K, and V, unlike adapting three separate d x d projections each at rank r. At rank 16, the combined QKV adapter has 16 x (d + 3d) = 16 x 4d parameters, whereas three separate rank-16 adapters would have 3 x 16 x 2d = 96d parameters. This is a known design choice in LoRA literature (Hu et al. 2022 use separate projections; PEFT wraps whatever `nn.Linear` it finds).

**Code path:** `src/growth/models/encoder/lora_adapter.py:43-51` (suffix mapping), `lora_adapter.py:281` (PEFT wrapping)

---

### 7. Robust volume statistics (median, MAD) are computed but not described

**Thesis (lines 192-199):** Only sample mean and sample variance of log-volumes are described.

**Code reality:** The inference pipeline also computes: raw volume median, volume MAD (median absolute deviation), log-volume median, log-volume MAD, and scaled MAD (1.4826 x MAD, the Gaussian-consistent estimator). These are written to the output CSV.

**Why it matters:** If downstream analyses use robust estimators instead of the sample mean/variance (e.g., for outlier-resistant growth modelling), the methods should describe them. If they are only diagnostic, a brief mention suffices.

**Code path:** `experiments/uncertainty_segmentation/engine/ensemble_inference.py:346-349`

---

### 8. Dice smoothing constant not specified

**Thesis (line 117):** "where epsilon is a smoothing constant that prevents division by zero."

**Code reality:** epsilon = 1e-5 (set in `SegmentationLoss3Ch.__init__`, `smooth=1e-5`).

**Why it matters:** Minor for reproducibility, but the specific value should be stated in the appendix or inline, as different values can affect training dynamics on small structures (e.g., NETC with only 39 voxels in some slices).

**Code path:** `src/growth/losses/segmentation.py:389`

---

## LOW — Minor omissions

### 9. Uncertainty maps are per-channel

**Thesis (line 171-174):** Describes predictive entropy, aleatoric entropy, and MI without specifying spatial structure.

**Code reality:** All three maps are computed **per-channel** (i.e., independently for TC, WT, ET), yielding tensors of shape [3, D, H, W]. Scalar summaries (spatial mean) are computed only when writing CSV.

**Code path:** `experiments/uncertainty_segmentation/engine/uncertainty_metrics.py:27-60`

---

### 10. Implementation uses HuggingFace PEFT library

**Thesis:** Does not name the LoRA implementation library.

**Code reality:** Uses `peft.get_peft_model` with `LoraConfig`. PEFT has specific implementation details: lora_dropout is applied before `lora_A` (not after), bias adaptation is disabled (`bias="none"`), and the output head is NOT registered via PEFT's `modules_to_save` (saved separately as `decoder.pt`).

**Why it matters:** PEFT is the de facto standard, but naming it in the methods (as "implemented via HuggingFace PEFT v0.x") aids reproducibility and clarifies that the LoRA formulation matches PEFT's conventions.

**Code path:** `src/growth/models/encoder/lora_adapter.py:30` (import), `lora_adapter.py:269-281` (config + wrapping)

---

### 11. Aleatoric uncertainty map not stored as a named output

**Thesis (line 173):** "the mean per-member entropy E_m[H[p^(m)]], which captures aleatoric uncertainty"

**Code reality:** The mean per-member entropy is tracked internally during Welford aggregation but is NOT stored as a named field in the `EnsemblePrediction` dataclass. Only `predictive_entropy` and `mutual_information` are persisted. The aleatoric component can only be recovered as `predictive_entropy - mutual_information`.

**Code path:** `experiments/uncertainty_segmentation/engine/ensemble_inference.py:295` (accumulation), `ensemble_inference.py:355-360` (output dataclass)

---

## Summary

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | Output head re-initialised, not unfrozen from pretrained | HIGH | Revise §3.2 "Freezing strategy" |
| 2 | Validation uses 128^3 crop, not sliding window | HIGH | Add sentence to §3.3 "Training configuration" |
| 3 | Pre-segmentation model unspecified | HIGH | Add citation + architecture + annotation protocol to §3.2.7 |
| 4 | Per-member CC cleanup before volume computation | HIGH | Clarify in §3.4 "Volume extraction" |
| 5 | Gaussian noise/smoothing hyperparameters missing | MEDIUM | Add to §3.3 "Data augmentation" or appendix |
| 6 | Combined QKV implication not discussed | MEDIUM | Optional sentence in §3.2 "LoRA configuration" |
| 7 | Robust statistics (median, MAD) not described | MEDIUM | Mention if used downstream |
| 8 | Dice smoothing constant not specified | MEDIUM | Add epsilon = 1e-5 to Eq. 4 |
| 9 | Uncertainty maps are per-channel | LOW | Add "(per-channel)" to §3.4 |
| 10 | PEFT library not named | LOW | Add implementation reference |
| 11 | Aleatoric map not a named output | LOW | Clarify or omit from methods |
