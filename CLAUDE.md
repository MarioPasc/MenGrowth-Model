# CLAUDE.md — MenGrowth-Model (VAE → Disentangled Latents → Neural ODE)

This repository implements a staged methodology to learn **disentangled latent state vectors** from **multi-modal 3D MRI** (4 channels, 128³) suitable for **continuous-time tumor growth forecasting** via a **Neural ODE**.

---

## 1) Methodology Overview (Experiments + Neural ODE)

### Common data / tensor contract (all experiments)
- Input per subject: 4 modalities `["t1c","t1n","t2f","t2w"]` stacked into:
  - `x ∈ ℝ^{B×4×128×128×128}`
- Z-score intensity normalization per subject (channel-wise) is mandatory to reduce scanner variability.
- All models use a **3D encoder** and **GroupNorm** (not BatchNorm) because feasible batch sizes are small for 3D volumes.

---

### Experiment 1 (Exp1): Baseline 3D VAE (ELBO)
**Goal:** stable reconstruction + a workable latent manifold (not disentangled yet).

**Architecture**
- Encoder: 3D ResNet-style downsampling to a compact feature volume (e.g., 8³) then flatten.
- Latent: diagonal Gaussian posterior `q(z|x)=N(μ, diag(exp(logσ²)))`, with `z_dim=128`.
- Decoder: standard 3D transposed-convolution upsampling to `x_hat ∈ ℝ^{B×4×128×128×128}`.

**Loss (negative ELBO with Posterior Collapse Mitigation)**
- Reconstruction: Gaussian likelihood → MSE. Use **`reduction="mean"`** over all voxels/channels to keep scale comparable to KL in 128³ volumes.
- KL: closed-form KL between diagonal Gaussian and `N(0,I)`.
- **Posterior Collapse Mitigations** (default configuration):
  - **Cyclical Annealing** (Fu et al., 2019): Beta oscillates in cycles (0 → 1 → 0...) to periodically relieve KL pressure, allowing the model to explore the latent space. Default: 4 cycles over 160 epochs.
  - **Free Bits** (Kingma et al., 2016): Per-dimension KL threshold (0.1 nats/dim) prevents optimizer from collapsing any latent dimension below the information bottleneck. Uses **batch-mean clamping** (Pelsmaeker & Aziz, 2020) which is more appropriate for small batch sizes (B=2) and heterogeneous data, allowing capacity allocation to informative subsets of latent dimensions rather than enforcing uniformity across all samples.
  - **Capacity Control** (Burgess et al., 2018): Available but disabled by default. Gradually increases KL capacity from 0 to target over training.

**Expected outcome**
- Good reconstructions, but **entangled** latents (not suitable as ODE state without further constraints).

---

### Experiment 2 (Exp2): DIP-VAE-II with Configurable Decoder
**Goal:** **unsupervised disentanglement** via covariance matching.

**Architecture (decoder is configurable)**
The decoder type is controlled by `model.use_sbd` flag:
- **`use_sbd: false`** (recommended starting point): Standard transposed-conv decoder (BaselineVAE)
- **`use_sbd: true`**: Spatial Broadcast Decoder (VAESBD)
  - Broadcast `z` to 8×8×8 grid
  - Concatenate coordinate channels `(D,H,W) ∈ [-1,1]`
  - Decode with `resize_conv` upsampling to 128³

**Rationale for switchable decoder**:
- Start with `use_sbd: false` to verify the encoder learns informative latents
- SBD can cause collapse in 3D medical imaging (decoder ignores z, relies only on coordinates)
- If collapse persists with standard decoder, the problem is in training dynamics, not architecture

**Loss (DIP-VAE-II covariance matching)**
```
L = L_recon + KL(q||p) + λ_od × ||Cov_offdiag||_F² + λ_d × ||diag(Cov) - I||_2²
```
- **Covariance estimator** (DIP-VAE-II):
  - `Cov_q(z) = Cov_batch(μ) + mean_batch(diag(exp(logvar)))`
  - Computed in FP32 for AMP stability
- **λ schedule**: Linear warmup from 0 → λ_target over `lambda_cov_annealing_epochs`
- **Delayed start**: `lambda_start_epoch=20` pre-trains vanilla VAE before DIP regularization
- **DDP support**: `use_ddp_gather=True` enables all-gather of μ/logvar across ranks

**Recommended config** (`exp2_dipvae_sbd.yaml`):
```yaml
model:
  use_sbd: false            # Start with standard decoder
loss:
  lambda_od: 10.0
  lambda_d: 5.0
  lambda_start_epoch: 20    # Pre-train VAE for 20 epochs
  lambda_cov_annealing_epochs: 40
train:
  kl_free_bits: 0.2         # 128 dims × 0.2 = 25.6 nats floor
```

**References:**
- Kumar et al. (2018). "Variational Inference of Disentangled Latent Concepts from Unlabeled Observations." ICLR 2018 (arXiv:1711.00848).
- Watters et al. (2019). "Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled Representations in VAEs." (arXiv:1901.07017).

---

### Experiment 3 (planned): Semi-supervised, physics-aligned latents
**Goal:** disentanglement **and** semantic alignment of latent subspaces to ODE-relevant state variables.

**Latent partition (example)**
- `z_vol ∈ ℝ¹` (log-volume),
- `z_loc ∈ ℝ³` (centroid),
- `z_shape` (morphology),
- remaining dims for background/anatomy.

**Regularization**
- DIP-VAE covariance penalties + supervised losses:
  - `||z_vol - log(V_gt)||²`, `||z_loc - c_gt||²`.

---

### Neural ODE interface (planned downstream)
**Goal:** continuous-time evolution in latent space:
```
dz(t)/dt = f_θ(z(t), t)
```
- Physics-informed volume dynamics: Gompertz-like growth on the volume latent.
- Use `torchdiffeq` adjoint method for memory-efficient backprop.

---

## 2) Known Failure Modes and Diagnostics

### Posterior Collapse (Decoder Ignores z)
**Symptoms:**
- `diag/au_count = 0` (no active units).
- `kl_raw` << `kl_constrained` (free bits floor is active).
- `logvar_mean ≈ 0`, `z_std ≈ 1.0` (prior-like).
- Reconstructions look like smooth "templates" without subject-specific detail.

**Root cause (SBD failure mode):**
- The SBD provides explicit coordinates to every decoder layer.
- For 3D medical images with consistent spatial structure, the decoder can learn a "template" modulated only by coordinates.
- No gradient pressure on encoder to produce informative latents.

**Diagnostic metrics:**
| Metric | Healthy | Collapsed | How to check |
|--------|---------|-----------|--------------|
| `diag/au_count` | >10 | 0 | Variance of μ across dataset |
| `kl_raw` | >10 nats | <2 nats | Should exceed free bits floor |
| `diag/mu_var_mean` | >0.1 | <0.01 | Encoder mean diversity |
| `diag/recon_z0_mse` | >> `recon_mse` | ≈ `recon_mse` | z=0 ablation |
| `diag/recon_delta_mu_vs_sampled` | >0 | ≈0 | Decoder z-dependence |

**How to interpret:**
- **If `recon_z0_mse ≈ recon_mse`**: Decoder is ignoring z entirely.
- **If `mu_var_mean < 0.01`**: Encoder outputs near-constant means.
- **If `kl_raw < free_bits_floor`**: Encoder has collapsed to prior.

### Training Recipes for Collapse Prevention

**Recipe 1: Recommended starting point (DEFAULT in exp2_dipvae_sbd.yaml)**
```yaml
model:
  use_sbd: false            # Start with standard decoder
loss:
  lambda_start_epoch: 20    # Pre-train VAE for 20 epochs
  lambda_cov_annealing_epochs: 40
train:
  kl_free_bits: 0.2         # 128 × 0.2 = 25.6 nats floor
```
This combination pre-trains the VAE with a standard decoder before introducing DIP penalties.

**Recipe 2: Try SBD after verifying encoder works**
If Recipe 1 succeeds (AU > 10, KL > 20 nats):
```yaml
model:
  use_sbd: true             # Switch to SBD
loss:
  lambda_start_epoch: 20
train:
  kl_free_bits: 0.2
```

**Recipe 3: Stronger KL pressure (if still collapsing)**
```yaml
train:
  kl_free_bits: 0.3         # 128 × 0.3 = 38.4 nats floor
  kl_free_bits_mode: "per_sample"  # Stronger than batch_mean
```

**Recipe 4: Off-diagonal only (ablation)**
```yaml
loss:
  lambda_od: 10.0
  lambda_d: 0.0             # Disable diagonal penalty
```

---

## 3) Codebase Map

```
src/
├── engine/
│   ├── train.py                    # Entry point (routes to LitModules)
│   ├── model_factory.py            # NEW: Model instantiation (handles use_sbd logic)
│   └── plot_training_dashboard.py  # Dashboard generation
├── vae/
│   ├── config/
│   │   ├── exp1_baseline_vae.yaml
│   │   └── exp2_dipvae_sbd.yaml    # DIP-VAE-II with configurable decoder
│   ├── data/                        # MONAI transforms, dataloaders
│   ├── losses/
│   │   ├── elbo.py                  # Exp1: ELBO, free bits, cyclical annealing
│   │   └── dipvae.py                # Exp2: DIP-VAE-II covariance penalties
│   ├── models/
│   │   ├── vae/
│   │   │   ├── baseline.py          # BaselineVAE (standard decoder)
│   │   │   └── vae_sbd.py           # VAESBD (SBD decoder)
│   │   └── components/
│   │       └── sbd.py               # Spatial Broadcast Decoder
│   └── training/
│       ├── lit_modules.py           # VAELitModule, DIPVAELitModule
│       ├── callbacks.py             # Reconstruction saving
│       ├── au_callbacks.py          # Active Units computation
│       └── metrics_callbacks.py     # Tidy CSV logging
slurm/
├── execute_experiment1.sh
└── execute_experiment2_dip.sh       # DIP-VAE
```

**Model routing** (`train.py` → `model_factory.py`):
- `cfg.model.variant == "dipvae_sbd"` → `DIPVAELitModule`
  - `cfg.model.use_sbd == true` → instantiates `VAESBD`
  - `cfg.model.use_sbd == false` → instantiates `BaselineVAE`
- default → `VAELitModule` (baseline)

---

## 4) Libraries and Responsibilities

| Library | Role |
|---------|------|
| **PyTorch 2.0+** | Models, losses, AMP-safe kernels |
| **MONAI 1.3+** | NIfTI loading, transforms, caching |
| **PyTorch Lightning 2.0+** | Training loop, checkpointing, logging |
| **OmegaConf 2.3+** | Hierarchical configuration |
| **Weights & Biases** | Experiment tracking (offline mode for cluster) |

Guiding rule: **MONAI for data**, **PyTorch for core ML**, **Lightning for training loop + logging**.

---

## 5) Best Practices

### Numerical Stability
- Compute covariance penalties in **FP32** even under AMP (`compute_in_fp32: true`).
- Use `reduction="mean"` for MSE loss (prevents overflow with 128³ volumes).
- Gradient clipping: `gradient_clip_val: 5.0` (recommended).
- Posterior variance floor: `posterior_logvar_min: -6.0`.

### Logging Keys
- Training: `train_epoch/loss`, `train_epoch/recon`, `train_epoch/kl_raw`, `train_epoch/cov_penalty`
- Validation: `val_epoch/loss`, `val_epoch/ssim_*`, `val_epoch/psnr_*`
- Schedules: `sched/lambda_od`, `sched/lambda_d`, `sched/expected_kl_floor`
- Diagnostics: `diag/au_count`, `diag/au_frac`, `diag/recon_mu_mse`, `diag/recon_z0_mse`

### Reproducibility
- `seed_everything(seed, workers=True)`
- Save resolved config in run directory.
- Wipe cache if transforms change.

---

## 6) Key References

- **Free Bits**: Kingma et al. (2016). "Improved Variational Inference with Inverse Autoregressive Flow." NeurIPS.
- **Cyclical Annealing**: Fu et al. (2019). "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing." NAACL.
- **DIP-VAE**: Kumar et al. (2018). "Variational Inference of Disentangled Latent Concepts from Unlabeled Observations." ICLR (arXiv:1711.00848).
- **SBD**: Watters et al. (2019). "Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled Representations." arXiv:1901.07017.
- **β-TCVAE**: Chen et al. (2018). "Isolating Sources of Disentanglement in Variational Autoencoders." NeurIPS.
- **Disentanglement Identifiability**: Locatello et al. (2019). "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations." ICML (caution: unsupervised disentanglement is not guaranteed without inductive biases).
