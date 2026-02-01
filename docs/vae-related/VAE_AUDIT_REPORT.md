# VAE Training Stack Audit Report

**Date**: 2025-12-26
**Auditor**: Claude Opus 4.5
**Scope**: Exp1 Baseline VAE + Exp2 DIP-VAE-SBD

---

## Executive Summary

The audit identified that the **posterior collapse is REAL** (not a wiring bug). The DIP-VAE-SBD architecture exhibits a known failure mode where the Spatial Broadcast Decoder (SBD) learns to ignore the latent vector `z` and reconstructs images using only the coordinate channels.

**Key Evidence**:
- AU = 0.0 across all 85 epochs (correct measurement)
- KL_raw = 1.2-2.0 nats TOTAL (0.01 nats/dim)
- logvar_mean ≈ 0, z_std ≈ 1.0 (prior-like)
- Reconstructions look like smooth templates

**Root Cause**: SBD provides explicit coordinates, so the decoder can produce reasonable reconstructions without encoding subject-specific information in `z`.

---

## Task A: Baseline VAE Correctness & Wiring (Exp1)

### A1. Entry Point Correctness ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| `train.py` routing | CORRECT | Lines 100-110: checks `cfg.model.variant`, defaults to baseline |
| `slurm/execute_experiment1.sh` | CORRECT | Uses `exp1_baseline_vae.yaml` |
| Config → LitModule | CORRECT | `VAELitModule.from_config()` instantiates `BaselineVAE` |

### A2. Config → Model → LitModule Pipeline ✅

**Path verified**:
```
exp1_baseline_vae.yaml → VAELitModule.from_config() → BaselineVAE → compute_elbo()
```

**Loss (ELBO)**:
- Reconstruction: MSE with `reduction="mean"` (line 105 in elbo.py)
- KL: closed-form, sum over dims, mean over batch (correct)
- Free bits: implemented with `batch_mean` mode (correct)
- Cyclical annealing: implemented (lines 169-263 in elbo.py)

### A3. Callback Completeness ✅

| Callback | Attached | Monitor Key | Notes |
|----------|----------|-------------|-------|
| ModelCheckpoint | ✅ | `val_epoch/loss` | Correct key exists |
| ReconstructionCallback | ✅ | - | Saves stochastic recon only |
| TrainingLoggingCallback | ✅ | - | Replaces tqdm |
| TidyEpochCSVCallback | ✅ | - | One row per epoch |
| ActiveUnitsCallback | ✅ | `diag/au_count` | Canonical AU metric |
| LatentDiagnosticsCallback | ✅ | - | Correlation, shift sensitivity |
| GradNormCallback | ✅ | `opt/grad_norm` | When `log_grad_norm: true` |
| WandbLogger | ✅ | - | Offline mode for cluster |

### A4. Baseline Sanity Checks

**Already present**:
- `recon_per_voxel` ✅ (line 225 in lit_modules.py)
- `kl_raw` ✅ (line 220)
- `kl_per_dim` ✅ (line 226)
- Periodic reconstructions ✅ (ReconstructionCallback)

**Missing** (addressed in patch):
- Deterministic reconstruction (decode with μ, no sampling)
- z=0 ablation reconstruction

**Verdict**: Exp1 baseline is CORRECTLY WIRED.

---

## Task B: DIP-VAE-SBD Correctness (Exp2) - CRITICAL

### B1. Config Routing ✅

| Check | Result | Evidence |
|-------|--------|----------|
| `exp2_dipvae_sbd.yaml` sets `variant: "dipvae_sbd"` | ✅ | Line 14 |
| `train.py` routes to `DIPVAELitModule` | ✅ | Lines 104-105 |
| `slurm/execute_experiment2_dip.sh` uses correct config | ✅ | Line 26 |

**CONFIRMED**: We are training DIP-VAE-SBD, not TC-VAE.

### B2. DIP-VAE Loss Math ✅

**Covariance estimator** (`dipvae.py:27-67`):
```python
# CORRECT DIP-VAE-II formula:
Cov_q(z) = Cov_batch(μ) + mean_batch(diag(exp(logvar)))
```
This is the aggregated posterior covariance, not sampled-z covariance.

**Penalties** (`dipvae.py:236-247`):
- Off-diagonal: `λ_od × ||Cov_offdiag||_F²` ✅
- Diagonal: `λ_d × ||diag(Cov) - 1||_2²` ✅

**FP32 Safety** (`dipvae.py:50-67`):
- Covariance computed in FP32 when `compute_in_fp32=True` ✅
- Penalties computed on FP32 tensors ✅
- No premature cast back to FP16 ✅

**DDP Gather** (`dipvae.py:70-112`):
- `use_ddp_gather=True` enables all-gather of μ/logvar across ranks ✅

**KL Scaling** (`dipvae.py:188-226`):
- Per-dim KL, mean over batch, sum over dims ✅
- Free bits with `batch_mean` mode ✅
- No double division by batch size ✅

### B3. Posterior Collapse Diagnostics ⚠️

**Current state**:
- `diag/au_count`, `diag/au_frac` logged ✅
- `logvar_mean`, `z_std` logged ✅
- `kl_raw`, `kl_constrained` logged ✅
- Per-modality SSIM/PSNR logged ✅

**Missing diagnostics** (critical for collapse debugging):
1. **Deterministic recon**: decode z = μ (no sampling noise)
2. **z=0 ablation**: decode z = 0 to test if decoder ignores z
3. **||x̂(μ) - x̂(sampled)|| stats**: quantify decoder's z-dependence
4. **μ variance per-dim**: identify which dimensions are active

### B4. DIP Regularizer Schedule ⚠️

**Current schedule**:
- Linear warmup from 0 → λ_target over `lambda_cov_annealing_epochs` (40 epochs)
- Starts at epoch 0

**Issue**: The covariance penalty is introduced before the VAE has learned to use latents. By epoch 40, the model may have already learned to ignore z.

**Recommended changes**:
1. Add `lambda_start_epoch` config key for delayed start
2. Add `lambda_d_ratio` to allow λ_d=0 (off-diagonal only ablation)
3. Consider longer warmup or VAE pre-training phase

### B5. SBD Correctness ✅

**Implementation** (`sbd.py`):
- Coordinate grids in [-1, 1] ✅ (lines 114-133)
- `indexing='ij'` for correct axis alignment ✅ (line 124)
- `resize_conv` mode available ✅ (lines 159-169)
- Shape progression correct: 8³ → 16³ → 32³ → 64³ → 128³ ✅

**Config** (`exp2_dipvae_sbd.yaml`):
- `sbd_upsample_mode: "resize_conv"` ✅
- `sbd_grid_size: [8, 8, 8]` ✅

### B6. Checkpointing & Metric Keys ✅

| Check | Status | Evidence |
|-------|--------|----------|
| Monitor key `val_epoch/loss` exists | ✅ | Logged at line 998 |
| Filename template uses valid key | ⚠️ | Uses `{val_loss:.4f}` but key is `val_epoch/loss` |

**Issue**: The checkpoint filename template at `exp2_dipvae_sbd.yaml:80`:
```yaml
filename: "dipvae-{epoch:04d}-{val_loss:.4f}"
```
Should be:
```yaml
filename: "dipvae-{epoch:04d}"
```
(The `{val_loss}` placeholder references a key that doesn't exist with that exact name)

---

## Task C: AU Computation Correctness ✅

### C1. AU Implementation (`au_callbacks.py`)

**Definition used** (line 249-254):
```python
var_per_dim = mu_mat.var(dim=0, unbiased=False)  # [z_dim]
active_mask = var_per_dim > self.eps_au
au_count = active_mask.sum().item()
```

This is the **correct** canonical AU definition:
- Compute variance of μ across the dataset (not z samples)
- Count dimensions where Var(μ_j) > ε
- Uses FP32 on CPU for stability

### C2. Schedule ✅

- Dense phase: every epoch for epochs 0..`au_dense_until` (15)
- Sparse phase: every `au_sparse_interval` (5) epochs after

### C3. Logging ✅

- Keys: `diag/au_count`, `diag/au_frac`
- Appears in `epoch_metrics.csv` ✅
- Callback is attached for exp2_dip (config has `au_dense_until: 15`)

### C4. Why AU = 0?

**This is a REAL measurement, not a bug.**

From the metrics:
| Epoch | AU | kl_raw | logvar_mean | z_std |
|-------|-----|--------|-------------|-------|
| 0 | 0 | 2.09 | -0.004 | 1.02 |
| 40 | 0 | 1.21 | 0.0 | 1.00 |
| 85 | 0 | 1.19 | 0.0 | 1.00 |

**Interpretation**:
- `eps_au = 0.01` → threshold for active dimension
- Var(μ) across samples < 0.01 for ALL 128 dimensions
- This means the encoder outputs nearly constant means across all subjects
- `logvar ≈ 0` means variance ≈ 1 (prior variance)
- The encoder has collapsed to outputting the prior: q(z|x) ≈ N(0, I)

---

## Collapse Root Cause Analysis

### The SBD "Decoder Ignores Z" Failure Mode

The Spatial Broadcast Decoder provides explicit coordinate channels to every layer. For 3D medical images with consistent spatial structure (brain in same orientation, tumor in similar locations), the decoder can learn a "template" that it modulates with coordinates.

**Evidence**:
1. Reconstructions look like smooth templates (per user report)
2. μ has near-zero variance → encoder doesn't encode subject-specific info
3. KL is tiny → no information flows through the bottleneck
4. SSIM/PSNR are reasonable (0.72-0.75 SSIM) → reconstruction works despite collapse

### Why Didn't Covariance Penalties Help?

DIP-VAE penalties encourage **factorized** posteriors but don't prevent **uninformative** posteriors. If q(z|x) ≈ N(0, I) for all x:
- Cov_batch(μ) ≈ 0 (means don't vary)
- mean(diag(exp(logvar))) ≈ I (prior variance)
- Total Cov_q(z) ≈ I (already factorized!)
- Covariance penalties are near-minimal

The penalties don't force the encoder to be informative—they just encourage factorization.

---

## Recommendations

### Immediate Fixes (Patch Included)

1. **Add collapse diagnostics** to DIPVAELitModule:
   - `recon_mu`: MSE of x̂(μ) (deterministic recon)
   - `recon_z0`: MSE of x̂(0) (z=0 ablation)
   - `mu_var_mean`: mean variance of μ per dim
   - `recon_delta`: ||x̂(μ) - x̂(z)|| to measure decoder's z-dependence

2. **Fix checkpoint filename** in config

3. **Add delayed start** for covariance penalties

### Training Recipe Recommendations

1. **Pre-train vanilla VAE** for N epochs before enabling SBD or DIP penalties
2. **Increase KL weight** (β > 1) to force information through bottleneck
3. **Use stronger free bits** (0.2-0.5 nats/dim instead of 0.05)
4. **Consider removing SBD** for an initial experiment to verify encoder learning

### Research Considerations

The SBD architecture may be fundamentally problematic for this task:
- SBD was designed for simple 2D images with movable objects (Watters et al.)
- 3D medical images have strong spatial priors
- Consider alternatives: learned position embeddings, FiLM conditioning

---

## File-by-File Summary

| File | Status | Key Findings |
|------|--------|--------------|
| `train.py` | ✅ CORRECT | Routing and callback attachment verified |
| `lit_modules.py` | ✅ CORRECT | DIPVAELitModule correctly implements loss |
| `dipvae.py` | ✅ CORRECT | DIP-VAE-II covariance formula correct |
| `elbo.py` | ✅ CORRECT | Free bits and cyclical annealing work |
| `tcvae_sbd.py` | ✅ CORRECT | Model architecture correct |
| `sbd.py` | ✅ CORRECT | Coordinate grids and upsampling correct |
| `au_callbacks.py` | ✅ CORRECT | AU computation is canonical |
| `exp2_dipvae_sbd.yaml` | ⚠️ MINOR | Checkpoint filename should be fixed |
| `callbacks.py` | ✅ CORRECT | Reconstruction callback works |

---

## Appendix: Metrics Evidence

### From `/home/mpascual/epoch_metrics.csv`

**Epoch 0**:
- `val_epoch/kl_raw`: 2.09 nats
- `val_epoch/kl_constrained`: 6.43 nats (free bits floor active)
- `val_epoch/logvar_mean`: -0.004
- `val_epoch/z_std`: 1.02
- `diag/au_count`: 0
- `diag/au_frac`: 0.0

**Epoch 85**:
- `val_epoch/kl_raw`: 1.19 nats
- `val_epoch/kl_constrained`: 6.40 nats
- `val_epoch/logvar_mean`: 0.0001
- `val_epoch/z_std`: 1.00
- `diag/au_count`: 0
- `diag/au_frac`: 0.0

**Expected KL floor**: 128 × 0.05 = 6.4 nats (matches kl_constrained)

**Conclusion**: The encoder is outputting q(z|x) ≈ N(0, I) for all inputs, and the free bits mechanism is enforcing a minimum KL. This is textbook posterior collapse.
