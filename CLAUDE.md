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

**References:**
- Fu et al. (2019). "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing." NAACL-HLT 2019.
- Kingma et al. (2016). "Improved Variational Inference with Inverse Autoregressive Flow." NeurIPS 2016.
- Burgess et al. (2018). "Understanding disentangling in β-VAE." ICLR 2018.
- Pelsmaeker & Aziz (2020). "Effective Estimation of Deep Generative Language Models." EMNLP 2020. 

**Expected outcome**
- Good reconstructions, but **entangled** latents (not suitable as ODE state without further constraints).
---

### Experiment 2 (Exp2): Candidate 1 — β-TCVAE + Spatial Broadcast Decoder (SBD)
**Goal:** **unsupervised disentanglement**, especially removing positional information from `z` and encouraging factor independence

**Architecture change (decoder)**
- Replace dense projection decoder with **Spatial Broadcast Decoder (SBD)**:
  1) broadcast `z` to a small grid (e.g., 8×8×8),
  2) concatenate fixed coordinate channels `(D,H,W) ∈ [-1,1]`,
  3) decode with conv / transposed-conv to 128³.
- Rationale: coordinates are supplied explicitly, so `z` is pressured to encode “what” not “where”. 

**Loss change (β-TCVAE decomposition)**
- Decompose the KL into:
  - Mutual Information (MI),
  - Total Correlation (TC),
  - Dimension-wise KL (DWKL),
- and upweight TC with **`β_tc > 1`** (typical target ≈ 6) to penalize dependence among latent dims. 
- Use the **minibatch-weighted sampling (MWS)** estimator for `q(z)` / marginals; compute these terms in FP32 for numeric stability (especially under AMP).
- Practical note: batch-size limits in 3D can bias TC estimation; mitigation includes larger effective batches (checkpointing / accumulation) or memory-bank extensions (optional).

**Schedule**
- Warm-up `β_tc` from 0 → target over a fixed fraction of training (linear warm-up minimum). 

---

### Experiment 3 (planned): Candidate 2 — Semi-supervised, physics-aligned latents (DIP-VAE + auxiliary regressions)
**Goal:** disentanglement **and** semantic alignment of latent subspaces to ODE-relevant state variables.

**Latent partition (example)**
- `z_vol ∈ ℝ¹` (log-volume),
- `z_loc ∈ ℝ³` (centroid),
- `z_shape` (morphology),
- remaining dims for background/anatomy.

**Regularization**
- Prefer **DIP-VAE** moment-matching (covariance penalties) in small-batch regimes; add supervised losses:
  - `||z_vol - log(V_gt)||²`, `||z_loc - c_gt||²`.
- Weighting must account for the voxel-sum recon term magnitude; uncertainty-weighting is an option. 
---

### Neural ODE interface (planned downstream)
**Goal:** continuous-time evolution in latent space:
\[
\frac{d z(t)}{dt} = f_\theta(z(t), t)
\]
- Physics-informed volume dynamics: Gompertz-like growth on the volume latent, with parameters predicted from phenotype latents (`z_shape`).
- Use `torchdiffeq` adjoint method for memory-efficient backprop through the solver.

---

## 2) Codebase map (what to touch)
Current layout (core):
- `src/vae_dynamics/data/`: dataset indexing, MONAI transforms, dataloaders/collate.
- `src/vae_dynamics/models/vae/`:
  - `baseline.py` (Exp1),
  - `tcvae_sbd.py` (Exp2).
- `src/vae_dynamics/models/components/sbd.py`: SBD coordinate grid + broadcast logic.
- `src/vae_dynamics/losses/`:
  - `elbo.py` (Exp1),
  - `tcvae.py` (Exp2).
- `src/vae_dynamics/training/`:
  - `lit_modules.py`: LightningModules for Exp1/Exp2,
  - `callbacks.py`: recon snapshot logging, run artifacts.
- `src/vae_dynamics/config/`:
  - `exp1_baseline_vae.yaml`,
  - `exp2_tcvae_sbd.yaml`.
- Entry point: `scripts/train.py`.

This file supersedes the older autogenerated notes where they conflict with the PDF specs (e.g., ELBO scaling and Exp2 decomposition).

---

## 3) Libraries and responsibilities
- **PyTorch 2.0+**: Deep learning framework (models, losses, tensor math, AMP-safe kernels).
- **MONAI 1.3+**: Medical imaging transforms and datasets (NIfTI loading pipelines, spacing/orientation, caching via `PersistentDataset`).
- **PyTorch Lightning 2.0+**: Training orchestration (Trainer, checkpointing, loggers, callbacks, multi-GPU/AMP plumbing).
- **OmegaConf 2.3+**: Hierarchical configuration (experiment YAMLs, overrides).
- **NiBabel 5.0+**: NIfTI file I/O (backend used by MONAI; keep for utilities if needed).
- **Python ≥3.11**: Type hints, dataclasses, modern stdlib features.

Guiding rule: **MONAI for data**, **PyTorch for core ML**, **Lightning for training loop + logging**.

---

## 4) Best Python / research-engineering practices (required)
- **No “AI slop”**:
  - avoid unnecessary try/except wrappers,
  - avoid needless helper abstractions,
  - prefer simple readable functions with clear names.
- **Docstrings and typing everywhere**:
  - module docstring: intent + boundaries,
  - function docstrings: inputs/outputs + tensor shapes,
  - type annotations for public APIs.
- **Keep contracts explicit**:
  - tensor shapes always `[B,C,D,H,W]`,
  - modality order is fixed and documented,
  - model forward signatures are stable (Exp1 returns `(x_hat, mu, logvar)`, Exp2 additionally returns `z`).
- **Numerical stability policy**:
  - compute TC-VAE density terms in **FP32** even under AMP,
  - log and assert finiteness (`torch.isfinite`) for loss terms in debug/test paths.
- **Reproducibility**:
  - seed everything (`seed_everything(seed, workers=True)`),
  - save resolved config and train/val split CSVs in each run directory,
  - treat cached datasets as invalid if transforms change (wipe cache).
- **Modularization**:
  - data/indexing/transforms separate from model,
  - losses as pure functions returning dicts of terms,
  - LightningModules should orchestrate calls, not own core math.

Keep this file concise and actionable. If you add Exp3/Neural ODE code, update only the methodology bullets and the module map (do not add long tutorials).
