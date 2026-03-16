# LoRA Pipeline — Methodological Fixes (Q1 Review)

Date: 2026-02-27

A Q1-level methodological review of the LoRA adaptation pipeline identified
8 bugs/flaws and 2 config issues. All fixes target correctness and downstream
SDP/GP quality. No algorithmic changes to the core LoRA method.

## Impact on Existing Results

- **SDP production results remain valid**: BUG-2 (double normalization) was
  compensated by lambda tuning. Centroid normalization (FLAW-5) affects location
  R² but not primary vol/shape metrics.
- **H5 re-conversion required** after FLAW-5 (centroid normalization fix).
- **SDP lambda retuning may be needed** for future runs after BUG-2 fix.

---

## Bugs Fixed

### BUG-1: Baseline double forward pass (CRITICAL)
**File**: `experiments/lora_ablation/pipeline/model_factory.py`
**Severity**: Performance bug (2× swinViT computation per step)

`BaselineOriginalDecoderModel.forward_with_semantics()` called `self.model(x)`
and then `self.model.get_bottleneck_features(x)`, resulting in two separate
swinViT forward passes. Fixed to compute hidden states once and reuse for
both decoding and feature extraction (matching the LoRA model pattern).

### BUG-2: SDP semantic loss double normalization (MODERATE)
**File**: `src/growth/losses/semantic.py`

`torch.mean((pred - target) ** 2) / k_p` double-divided by the partition
dimensionality — `torch.mean()` already averages over B × k_p elements.
Result was `sum / (B * k_p²)` instead of `sum / (B * k_p)`. Removed the
extra `/ k_p`.

**Note**: Changes SDP loss magnitude. Future SDP runs need lambda retuning.

### BUG-3: Buffer overwrite in AuxiliarySemanticLoss (MODERATE)
**File**: `src/growth/models/segmentation/semantic_heads.py`

Direct assignment `self.volume_mean = ...` replaced registered buffers with
plain tensors, breaking `state_dict()` serialization and `.to(device)` behavior.
Fixed with `.copy_()` in both `update_statistics()` and the caller in
`train_condition.py`.

---

## Flaws Fixed

### FLAW-1: VICReg feature accumulation (MODERATE)
**Files**: `src/growth/losses/encoder_vicreg.py`, `train_condition.py`

VICReg was computed per micro-batch (B=4 with grad_accum=2), giving degenerate
covariance estimates with only 4 samples in 768 dimensions. Fixed to:
1. Add batch_size < 2 guard returning zero loss.
2. Buffer features across micro-batches and compute VICReg at optimizer step
   boundaries on the concatenated batch (B=8).

### FLAW-2: Feature-quality checkpoint selection (MINOR → MODERATE)
**File**: `train_condition.py`

Checkpoint selection used only Dice, ignoring feature quality for downstream SDP.
Added:
- Variance hinge tracking every epoch.
- Inline Ridge-probe R² evaluation every epoch (~30s overhead).
- Checkpoint score: `probe_mean_r2` (Dice used as fallback only when
  semantic heads are disabled).

### FLAW-3: ReduceLROnPlateau with warmup (MINOR)
**File**: `train_condition.py`

Replaced `CosineAnnealingLR` with warmup (5 epochs, LinearLR from 0.01×) followed
by `ReduceLROnPlateau` (mode='max' on val Dice, patience=10, factor=0.5). This
avoids premature LR decay when training converges faster/slower than `T_max`.

### FLAW-4: encoder10 unfreezing documentation (MINOR)
**File**: `src/growth/models/segmentation/original_decoder.py`

Added comment documenting that `encoder10` is unfrozen by design — it is
architecturally part of the decoder pathway (processes bottleneck hidden_states[4])
and its output becomes the 768-dim features used for SDP.

### FLAW-5: Centroid normalization (MODERATE)
**File**: `scripts/convert_nifti_to_h5.py`

Centroids were computed from native-resolution segmentation masks (variable
dimensions per subject). Now computed from preprocessed 192³ segmentation,
ensuring all subjects share the same spatial reference frame.

**Action required**: Re-run `convert_nifti_to_h5.py` to regenerate H5 file.

### FLAW-6: DCI cross-validated R² (MINOR)
**File**: `src/growth/evaluation/latent_quality.py`

Training-set R² in DCI informativeness was inflated (especially with n_dims >> n_samples).
Replaced with 5-fold cross-validated R² via `cross_val_score`.

---

## Config Changes

### Split sizes alignment
**File**: `experiments/lora_ablation/config/ablation_v3.yaml`

Updated to match production config: lora_train=525, lora_val=100, sdp_train=225, test=150.

### New training parameters
```yaml
training:
  lr_warmup_epochs: 5
  lr_reduce_factor: 0.5
  lr_reduce_patience: 10
```
