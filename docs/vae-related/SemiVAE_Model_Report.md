# Semi-Supervised Variational Autoencoder (SemiVAE) Model Report

**MenGrowth-Model Project**
**Date:** January 2026
**Model Version:** Run 6 (Structural Independence)

---

## Table of Contents

1. [Overview and Scientific Motivation](#1-overview-and-scientific-motivation)
2. [Model Architecture](#2-model-architecture)
   - [2.1 Encoder (Encoder3D)](#21-encoder-encoder3d)
   - [2.2 Spatial Broadcast Decoder (SBD)](#22-spatial-broadcast-decoder-sbd)
   - [2.3 Semantic Projection Heads](#23-semantic-projection-heads)
   - [2.4 Auxiliary Residual Decoder](#24-auxiliary-residual-decoder)
3. [Latent Space Partitioning](#3-latent-space-partitioning)
4. [Semantic Feature Extraction](#4-semantic-feature-extraction)
   - [4.1 Volume Features](#41-volume-features)
   - [4.2 Location Features](#42-location-features)
   - [4.3 Shape Features](#43-shape-features)
   - [4.4 Normalization and Residualization](#44-normalization-and-residualization)
5. [Loss Functions](#5-loss-functions)
   - [5.1 Reconstruction Loss](#51-reconstruction-loss)
   - [5.2 KL Divergence on Residual Dimensions](#52-kl-divergence-on-residual-dimensions)
   - [5.3 Semantic Regression Losses](#53-semantic-regression-losses)
   - [5.4 Distance Correlation Loss](#54-distance-correlation-loss)
   - [5.5 Auxiliary Reconstruction Loss](#55-auxiliary-reconstruction-loss)
   - [5.6 Total Loss Formulation](#56-total-loss-formulation)
6. [Training Procedure](#6-training-procedure)
   - [6.1 Curriculum Learning](#61-curriculum-learning)
   - [6.2 Gradient Isolation](#62-gradient-isolation)
   - [6.3 Multi-GPU Training (DDP)](#63-multi-gpu-training-ddp)
7. [Numerical Stability Features](#7-numerical-stability-features)
8. [Configuration Reference](#8-configuration-reference)
9. [File Structure](#9-file-structure)
10. [References](#10-references)

---

## 1. Overview and Scientific Motivation

The SemiVAE model is a Semi-Supervised Variational Autoencoder designed for learning disentangled latent representations of 3D meningioma tumors from MRI scans. The model serves as the encoder stage for the MenGrowth-Model project, which aims to predict individualized tumor growth trajectories using Neural Ordinary Differential Equations (Neural ODEs).

### Scientific Objective

The primary goal is to learn a partitioned latent space where specific dimensions encode interpretable, semantically meaningful tumor characteristics:

- **Volume dimensions**: Encode tumor size (critical for Gompertz growth modeling)
- **Location dimensions**: Encode spatial position in the brain
- **Shape dimensions**: Encode morphological characteristics
- **Residual dimensions**: Capture remaining variation (texture, contrast, artifacts)

This disentanglement is essential because the downstream Neural ODE must learn Gompertz growth dynamics:

$$\frac{dV}{dt} = \alpha V \ln\left(\frac{K}{V}\right)$$

where $V$ is the tumor volume, $\alpha$ is the growth rate, and $K$ is the carrying capacity. Without disentanglement, the ODE would need to learn complex, entangled dynamics across all latent dimensions, requiring far more longitudinal training data than available.

### Key Innovations (Run 6)

1. **Distance Correlation (dCor)**: Replaces Pearson correlation for detecting both linear and non-linear dependencies between partitions
2. **Gradient Isolation**: Prevents cross-partition gradient leakage during semantic supervision
3. **Auxiliary Residual Decoder**: Prevents collapse of unsupervised dimensions
4. **Shape Residualization**: Removes natural volume-shape confounds via OLS regression

---

## 2. Model Architecture

### 2.1 Encoder (Encoder3D)

**Location:** `src/vae/models/components/encoder.py`

The encoder is a 3D ResNet-style convolutional network that maps input MRI volumes to posterior distribution parameters.

#### Architecture Stages

```
Input: [B, 4, 160, 160, 160]
    |
    v
Initial Conv (stride=2): 4 -> 32 channels
    [B, 32, 80, 80, 80]
    |
    v
Layer 1: 2x BasicBlock3d (no stride)
    [B, 32, 80, 80, 80]
    |
    v
Layer 2: 2x BasicBlock3d (stride=2)
    [B, 64, 40, 40, 40]
    |
    v
Layer 3: 2x BasicBlock3d (stride=2)
    [B, 128, 20, 20, 20]
    |
    v
Layer 4: 2x BasicBlock3d (stride=2)
    [B, 256, 10, 10, 10]
    |
    v
AdaptiveAvgPool3d(1)
    [B, 256, 1, 1, 1]
    |
    v
Flatten -> Linear heads
    mu: [B, 128]
    logvar: [B, 128] (clamped to min=-6.0)
```

#### Key Features

- **GroupNorm**: Uses 8 groups for batch-independent normalization (critical for small batch sizes)
- **Residual Connections**: BasicBlock3d with skip connections for stable gradient flow
- **Spectral Normalization**: Optional, applied to Conv3d layers (NOT fc_mu/fc_logvar) for Lipschitz continuity, enabling stable Neural ODE training
- **Logvar Clamping**: Posterior log-variance clamped at -6.0 to ensure minimum variance of ~0.0025

#### BasicBlock3d Structure (Post-Activation)

```
Input: x
    |
    v
Conv3d (k=3, padding=1, bias=False)
    |
    v
GroupNorm (8 groups)
    |
    v
ReLU
    |
    v
Conv3d (k=3, padding=1, bias=False)
    |
    v
GroupNorm (8 groups)
    |
    v
Add skip connection (x)
    |
    v
ReLU
    |
    v
Output
```

### 2.2 Spatial Broadcast Decoder (SBD)

**Location:** `src/vae/models/components/sbd.py`

The Spatial Broadcast Decoder (SBD) is the primary decoder architecture, chosen for its superior position-content disentanglement properties.

#### Motivation

Standard transposed-convolution decoders must learn to encode positional information implicitly in the latent space. The SBD provides explicit coordinate information, encouraging the latent space to encode content-only information and enabling better disentanglement.

#### Architecture

```
Latent Vector: z [B, 128]
    |
    v
Broadcast to spatial grid [B, 128, 10, 10, 10]
    |
    v
Concatenate with coordinate grids (normalized [-1, 1])
    [B, 131, 10, 10, 10]  (128 z + 3 coordinates)
    |
    v
Initial Conv (1x1): 131 -> 256
    [B, 256, 10, 10, 10]
    |
    v
UpBlock (2x): 256 -> 128, 10 -> 20
    [B, 128, 20, 20, 20]
    |
    v
UpBlock (2x): 128 -> 64, 20 -> 40
    [B, 64, 40, 40, 40]
    |
    v
UpBlock (2x): 64 -> 32, 40 -> 80
    [B, 32, 80, 80, 80]
    |
    v
UpBlock (2x): 32 -> 16, 80 -> 160
    [B, 16, 160, 160, 160]
    |
    v
Final Conv: 16 -> 4
    [B, 4, 160, 160, 160]
    |
    v
Tanh activation (bounds output to [-1, 1])
```

#### Coordinate Grid Creation

```python
def _create_coordinate_grid(grid_size):
    """Create normalized [-1, 1] coordinate grids.

    Returns: [1, 3, D, H, W] where each channel is x, y, z coordinates
    """
    coords = [torch.linspace(-1, 1, s) for s in grid_size]
    grids = torch.meshgrid(*coords, indexing='ij')
    return torch.stack(grids, dim=0).unsqueeze(0)
```

#### Upsampling Method: resize_conv (Default)

```
Interpolate (trilinear, scale=2) -> Conv3d -> GroupNorm -> ReLU
```

This avoids checkerboard artifacts common with transposed convolutions (Odena et al., 2016).

### 2.3 Semantic Projection Heads

**Location:** `src/vae/models/vae/semivae.py`

Each supervised latent partition has an associated MLP head that maps latent dimensions to predicted semantic features.

#### Architecture

```
Input: z_subset [B, partition_dim]
    |
    v
Linear: partition_dim -> 2 * partition_dim
    |
    v
ReLU
    |
    v
Dropout (p=0.1)
    |
    v
Linear: 2 * partition_dim -> num_features
    |
    v
Output: predictions [B, num_features]
```

#### Configuration (Run 6)

| Partition | Input Dims | Output Features | Features |
|-----------|------------|-----------------|----------|
| z_vol | 24 | 4 | vol_total, vol_ncr, vol_ed, vol_et |
| z_loc | 8 | 3 | loc_x, loc_y, loc_z |
| z_shape | 12 | 6 | sphericity_total, surface_area_total, solidity_total, aspect_xy_total, sphericity_ncr, surface_area_ncr |

#### Weight Initialization

- **Xavier normal** for Linear weights
- **Zero** for biases

### 2.4 Auxiliary Residual Decoder

**Location:** `src/vae/models/components/aux_decoder.py`

A lightweight decoder that reconstructs only from residual dimensions, providing explicit gradient signal to prevent z_residual collapse.

#### Architecture

```
z_residual [B, 84]
    |
    v
Linear: 84 -> 128 * 4^3 = 8192
    Reshape: [B, 128, 4, 4, 4]
    |
    v
UpBlock: 4->8, 128->64
    |
    v
UpBlock: 8->16, 64->32
    |
    v
UpBlock: 16->32, 32->16
    |
    v
UpBlock: 32->64, 16->16
    |
    v
Final Conv: 16 -> 4
    [B, 4, 64, 64, 64]  (low resolution output)
```

**Parameters:** ~2M (vs ~15M for main decoder)

**Purpose:** Forces z_residual to encode reconstructable information, preventing complete deflation under TC/manifold regularization.

---

## 3. Latent Space Partitioning

The 128-dimensional latent space is partitioned into four non-overlapping regions:

### Run 6 Configuration

| Partition | Dimensions | Indices | Supervision | Purpose |
|-----------|------------|---------|-------------|---------|
| **z_vol** | 24 | 0-23 | Regression | Tumor volume encoding |
| **z_loc** | 8 | 24-31 | Regression | Tumor location encoding |
| **z_shape** | 12 | 32-43 | Regression | Morphology encoding |
| **z_residual** | 84 | 44-127 | None (KL only) | Unsupervised factors |

### Visual Representation

```
z = [z_vol (24) | z_loc (8) | z_shape (12) | z_residual (84)]
     dims 0-23    24-31       32-43          44-127
     |           |           |              |
     v           v           v              v
  Supervised   Supervised  Supervised    Unsupervised
  (volumes)    (centroid)  (morphology)  (VAE prior)
```

### Design Rationale

- **z_vol (24 dims)**: 4 features x 6 dims redundancy = 24 dims. Volume is critical for Gompertz modeling.
- **z_loc (8 dims)**: 3 features x ~2.7 dims redundancy = 8 dims. Location may influence growth rate.
- **z_shape (12 dims)**: 6 residualized features x 2 dims redundancy = 12 dims. Tracks morphological evolution.
- **z_residual (84 dims)**: Captures texture, contrast, artifacts, and other factors not explicitly supervised.

---

## 4. Semantic Feature Extraction

### 4.1 Volume Features

**Location:** `src/vae/data/semantic_features.py`

Four volume features are extracted from segmentation masks:

| Feature | Definition | Transformation |
|---------|------------|----------------|
| vol_total | Union of all tumor labels (NCR + ED + ET) | log(vol + 1.0) |
| vol_ncr | Necrotic core volume (label 1) | log(vol + 1.0) |
| vol_ed | Peritumoral edema volume (label 2) | log(vol + 1.0) |
| vol_et | Enhancing tumor volume (label 3) | log(vol + 1.0) |

**Computation:**
```python
vol = voxel_count * (spacing[0] * spacing[1] * spacing[2])  # in mm^3
log_vol = np.log(vol + 1.0)  # Log-scale for numerical stability
```

### 4.2 Location Features

Three normalized centroid coordinates:

| Feature | Definition | Range |
|---------|------------|-------|
| loc_x | Tumor centroid X / ROI width | [0, 1] |
| loc_y | Tumor centroid Y / ROI height | [0, 1] |
| loc_z | Tumor centroid Z / ROI depth | [0, 1] |

**Computation:**
```python
centroid = scipy.ndimage.center_of_mass(tumor_mask)
loc_x = centroid[2] / roi_size[2]
loc_y = centroid[1] / roi_size[1]
loc_z = centroid[0] / roi_size[0]
```

### 4.3 Shape Features

Six shape descriptors extracted for the total tumor and NCR component:

| Feature | Definition | Range |
|---------|------------|-------|
| sphericity_total | $(36\pi V^2)^{1/3} / A$ for total tumor | [0, 1] |
| surface_area_total | log(marching cubes surface area + 1) | unbounded |
| solidity_total | Volume / Convex hull volume | [0, 1] |
| aspect_xy_total | Bounding box W/H | unbounded |
| sphericity_ncr | Sphericity for NCR component | [0, 1] |
| surface_area_ncr | Surface area for NCR component | unbounded |

### 4.4 Normalization and Residualization

#### Z-Score Normalization

All features are z-score normalized using statistics computed on the training set only (to prevent data leakage):

```python
normalized = (value - mean) / std
```

**Precomputation:** Run `vae.utils.precompute_semantic_normalizer` before training.

#### Shape Residualization

Shape features naturally correlate with volume (e.g., surface_area ~ V^(2/3)). To enable independent Neural ODE dynamics, shape features are residualized against volume:

```python
residual_shape = raw_shape - (slope * vol_total + intercept)
```

Coefficients are precomputed via OLS regression and stored in `shape_residual_params.json`.

---

## 5. Loss Functions

### 5.1 Reconstruction Loss

**Location:** `src/vae/losses/elbo.py`

Mean Squared Error between input and reconstruction:

$$L_{recon} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$

where $N = B \times C \times D \times H \times W$ (total elements).

**Reduction:** `mean` for numerical stability with BF16 mixed precision.

### 5.2 KL Divergence on Residual Dimensions

**Location:** `src/vae/losses/elbo.py`

KL divergence between the posterior and standard Gaussian prior, applied only to residual dimensions:

$$L_{KL} = \beta(t) \sum_{j \in \text{residual}} \max\left(\text{KL}_j, \delta_{fb}\right)$$

where:
$$\text{KL}_j = \frac{1}{2}\left(\exp(\log\sigma_j^2) + \mu_j^2 - 1 - \log\sigma_j^2\right)$$

#### Free Bits Mechanism

To prevent posterior collapse, a per-dimension KL floor is enforced:

- **kl_free_bits:** 0.2 nats/dimension
- **kl_free_bits_mode:** "batch_mean" (clamps batch-averaged KL per dimension)

#### Cyclical Annealing

The KL weight $\beta(t)$ follows cyclical annealing (Fu et al., 2019):

- **Number of cycles:** 4
- **Annealing epochs:** 240
- **Annealing ratio:** 0.5 (ramp up for first half of each cycle, plateau for second half)

### 5.3 Semantic Regression Losses

**Location:** `src/vae/training/lit_modules/semivae.py`

MSE losses between predicted and ground-truth semantic features:

$$L_{vol} = \lambda_{vol}(t) \cdot \text{MSE}(\hat{y}_{vol}, y_{vol})$$
$$L_{loc} = \lambda_{loc}(t) \cdot \text{MSE}(\hat{y}_{loc}, y_{loc})$$
$$L_{shape} = \lambda_{shape}(t) \cdot \text{MSE}(\hat{y}_{shape}, y_{shape})$$

#### Curriculum Schedule (Run 6)

| Partition | Target Lambda | Start Epoch | Annealing Epochs |
|-----------|--------------|-------------|------------------|
| Volume | 20.0 | 30 | 40 |
| Location | 12.0 | 80 | 40 |
| Shape | 15.0 | 120 | 50 |

Semantic lambdas follow a linear warmup from the start epoch:

$$\lambda_p(t) = \begin{cases}
0 & \text{if } t < t_{start} \\
\lambda_{target} \cdot \frac{t - t_{start}}{T_{anneal}} & \text{if } t_{start} \leq t < t_{start} + T_{anneal} \\
\lambda_{target} & \text{if } t \geq t_{start} + T_{anneal}
\end{cases}$$

#### Lambda Decay Phase

After epoch 400, semantic lambdas decay to 50% over 150 epochs:

$$\lambda_p(t) = \lambda_{target} \cdot (1 - 0.5 \cdot \text{progress})$$

This releases encoder capacity in late training, allowing residual dimensions to expand.

### 5.4 Distance Correlation Loss

**Location:** `src/vae/losses/dcor.py`

Distance correlation (Szekely & Rizzo, 2007) detects both linear and non-linear statistical dependencies between partitions. Unlike Pearson correlation, dCor = 0 if and only if the variables are independent.

$$L_{dCor} = \lambda_{dCor}(t) \sum_{i < j} \text{dCor}^2(\text{partition}_i, \text{partition}_j)$$

#### U-Centered Distance Correlation

For distance matrices $A_{kl} = \|X_k - X_l\|_2$ and $B_{kl} = \|Y_k - Y_l\|_2$:

1. **U-center (unbiased estimator):**
$$\tilde{A}_{kl} = A_{kl} - \frac{1}{n-2}\sum_m A_{km} - \frac{1}{n-2}\sum_m A_{ml} + \frac{1}{(n-1)(n-2)}\sum_{mn} A_{mn}$$

2. **Distance covariance:**
$$V^2(X,Y) = \frac{1}{n(n-3)}\sum_{k,l} \tilde{A}_{kl} \tilde{B}_{kl}$$

3. **Distance correlation:**
$$\text{dCor}^2(X,Y) = \frac{V^2(X,Y)}{\sqrt{V^2(X,X) \cdot V^2(Y,Y)}}$$

#### EMA Buffer

To increase effective sample size beyond per-batch N=2, an EMA buffer accumulates samples:

- **Buffer size:** 256 samples
- **Effective N:** batch_size + buffer_size = 2 + 254 = 256
- **Gradient flow:** Only current batch samples have gradients; buffer samples are detached

#### Configuration (Run 6)

- **lambda_dcor:** 5.0
- **dcor_start_epoch:** 150
- **dcor_annealing_epochs:** 50

### 5.5 Auxiliary Reconstruction Loss

**Location:** `src/vae/training/lit_modules/semivae.py`

Low-resolution reconstruction from residual dimensions only:

$$L_{aux} = \lambda_{aux} \cdot \text{MSE}(\hat{x}_{64}, \text{downsample}(x, 64^3))$$

#### Configuration (Run 6)

- **lambda_aux_recon:** 0.5
- **aux_recon_target_size:** 64

### 5.6 Total Loss Formulation

The complete training loss is:

$$L_{total} = L_{recon} + \beta(t) \cdot L_{KL,residual} + L_{vol} + L_{loc} + L_{shape} + L_{dCor} + L_{aux}$$

#### Disabled Loss Components (Run 6)

- **Total Correlation (TC):** Disabled ($\lambda_{TC} = 0$) due to gradient instability and limited benefit
- **Cross-Partition Pearson:** Disabled ($\lambda_{cross} = 0$), replaced by dCor
- **Manifold Density:** Disabled ($\lambda_{manifold} = 0$), contributes to residual collapse
- **KL on Supervised Partitions:** Disabled ($\beta_{supervised} = 0$), gradient isolation handles separation

---

## 6. Training Procedure

### 6.1 Curriculum Learning

Training follows a three-phase curriculum:

#### Phase 1: VAE Warmup (Epochs 0-29)

- Pure VAE training (reconstruction + KL only)
- All semantic/disentanglement losses inactive
- Establishes reconstruction baseline

#### Phase 2: Semantic Curriculum (Epochs 30-169)

Semantic losses are introduced sequentially:

| Epochs | Active Losses |
|--------|---------------|
| 30-69 | Volume ramps 0 -> 20 |
| 70-79 | Volume plateau (20) |
| 80-119 | Location ramps 0 -> 12, Volume holds |
| 120-149 | Shape ramps 0 -> 15, Location/Volume hold |
| 150-169 | dCor ramps 0 -> 5, All semantic at target |

#### Phase 3: Full Training + Decay (Epochs 170-600)

- Epochs 170-399: All losses at full strength
- Epochs 400-549: Semantic lambdas decay to 50%
- Epochs 550-600: Final phase with reduced semantic pressure

### 6.2 Gradient Isolation

**Location:** `src/vae/models/vae/semivae.py`

When enabled (`gradient_isolation: true`), each semantic head only receives gradients for its own partition dimensions:

```python
def predict_semantic_features(self, mu, gradient_isolation=False):
    predictions = {}
    for name, head in self.semantic_heads.items():
        if gradient_isolation:
            start, end = self._partition_bounds[name]
            # Detach all except own partition
            z_subset = torch.cat([
                mu[:, :start].detach(),     # Before: no gradient
                mu[:, start:end],            # Own partition: gradient flows
                mu[:, end:].detach(),        # After: no gradient
            ], dim=1)[:, start:end]
        else:
            z_subset = self.get_latent_subset(mu, name)
        predictions[name] = head(z_subset)
    return predictions
```

This prevents cross-partition gradient leakage where, e.g., the volume loss would otherwise update shape-encoding dimensions.

### 6.3 Multi-GPU Training (DDP)

The system supports distributed training across multiple GPUs:

```python
trainer = pl.Trainer(
    devices=4,
    strategy="ddp",
    sync_batchnorm=True,
    precision="bf16-mixed",
)
```

#### DDP-Aware Computations

- **dCor buffer:** Accumulated on CPU, shared across processes
- **Metric logging:** `sync_dist=True` for proper reduction
- **Batch normalization:** Synchronized across GPUs

---

## 7. Numerical Stability Features

### 7.1 Posterior Logvar Clamping

```python
logvar = torch.clamp(logvar, min=-6.0)
```

Ensures minimum variance of exp(-6)/2 ≈ 0.0012, preventing numerical instability from extreme certainty.

### 7.2 Free Bits for KL

```python
kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
```

Prevents posterior collapse by enforcing minimum KL of 0.2 nats/dimension.

### 7.3 Spectral Normalization

When enabled (`use_spectral_norm: true`):

- Applied to all Conv3d layers in encoder and decoder
- NOT applied to encoder fc_mu/fc_logvar heads (logvar needs large negative values)
- Ensures Lipschitz continuity: $\sigma(W) \leq 1$
- Critical for stable Neural ODE training downstream

### 7.4 Gradient Clipping

```python
gradient_clip_val: 1.0
gradient_clip_algorithm: norm
```

L2 norm clipping prevents gradient explosion during training.

### 7.5 BF16 Mixed Precision

```python
precision: bf16-mixed
```

BF16 provides faster training with larger dynamic range than FP16, reducing overflow risk.

### 7.6 FP32 Covariance Computation

```python
compute_in_fp32: true
```

Distance correlation and covariance computations are performed in FP32 for numerical stability.

---

## 8. Configuration Reference

### 8.1 Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| z_dim | 128 | Total latent dimensionality |
| input_channels | 4 | MRI modalities (T1c, T1n, T2f, T2w) |
| base_filters | 32 | Initial filter count |
| num_groups | 8 | Groups for GroupNorm |
| use_sbd | true | Use Spatial Broadcast Decoder |
| sbd_grid_size | [10, 10, 10] | SBD initial grid |
| use_spectral_norm | true | Enable spectral normalization |
| posterior_logvar_min | -6.0 | Minimum logvar clamp |

### 8.2 Loss Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| lambda_vol | 20.0 | Volume regression weight |
| lambda_loc | 12.0 | Location regression weight |
| lambda_shape | 15.0 | Shape regression weight |
| lambda_dcor | 5.0 | Distance correlation weight |
| lambda_aux_recon | 0.5 | Auxiliary reconstruction weight |
| lambda_tc | 0.0 | Total correlation (disabled) |
| lambda_cross_partition | 0.0 | Pearson correlation (disabled) |
| lambda_manifold | 0.0 | Manifold density (disabled) |

### 8.3 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_epochs | 600 | Total training epochs |
| lr | 5e-5 | Learning rate |
| weight_decay | 0.01 | AdamW regularization |
| batch_size | 2 | Per-GPU batch size |
| devices | 4 | Number of GPUs |
| kl_beta | 1.0 | Target KL weight |
| kl_free_bits | 0.2 | Per-dimension KL floor |
| gradient_clip_val | 1.0 | Gradient L2 norm clip |

### 8.4 Expected Outcomes

| Metric | Target | Description |
|--------|--------|-------------|
| Vol R² | > 0.85 | Volume prediction quality |
| Loc R² | > 0.91 | Location prediction quality |
| Shape R² | > 0.40 | Shape prediction quality |
| max\|dCor\| | < 0.35 | Inter-partition independence |
| z_residual AU | > 0.60 | Active units in residual |

---

## 9. File Structure

```
src/vae/
├── config/
│   ├── semivae_run6.yaml         # Current configuration
│   └── shape_residual_params.json # Shape residualization coefficients
│
├── data/
│   ├── semantic_features.py       # Feature extraction
│   ├── transforms.py              # Data transforms
│   ├── datasets.py                # Dataset classes
│   └── compute_shape_residuals.py # Shape residualization script
│
├── models/
│   ├── vae/
│   │   ├── semivae.py             # SemiVAE model
│   │   └── vae_sbd.py             # Base VAE with SBD
│   └── components/
│       ├── encoder.py             # Encoder3D
│       ├── sbd.py                 # SpatialBroadcastDecoder
│       ├── decoder.py             # Decoder3D
│       ├── aux_decoder.py         # AuxDecoder3D
│       └── basic.py               # BasicBlock3d
│
├── losses/
│   ├── elbo.py                    # ELBO loss with free bits
│   ├── tc.py                      # Total Correlation
│   ├── dcor.py                    # Distance Correlation
│   └── cross_partition.py         # Pearson correlation (legacy)
│
├── training/
│   ├── lit_modules/
│   │   └── semivae.py             # SemiVAELitModule
│   ├── engine/
│   │   ├── train.py               # Training script
│   │   └── model_factory.py       # Model instantiation
│   └── callbacks/
│       └── semivae_callbacks.py   # Diagnostics callbacks
│
└── utils/
    └── precompute_semantic_normalizer.py
```

---

## 10. References

### Primary Literature

1. **Semi-Supervised VAE:**
   Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014). Semi-Supervised Learning with Deep Generative Models. *NeurIPS*.

2. **Disentangled Representations:**
   Locatello, F., Bauer, S., Lucic, M., Raetsch, G., Gelly, S., Scholkopf, B., & Bachem, O. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. *ICML*.

3. **Distance Correlation:**
   Szekely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and Testing Dependence by Correlation of Distances. *Annals of Statistics*.

4. **Spatial Broadcast Decoder:**
   Watters, N., Matthey, L., Bosnjak, M., Burgess, C. P., & Lerchner, A. (2019). COBRA: Data-Efficient Model-Based RL through Unsupervised Object Discovery and Curiosity-Driven Exploration. *arXiv*.

5. **Neural ODE:**
   Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.

6. **Gompertz Growth:**
   Benzekry, S., Lamont, C., Beheshti, A., Tracz, A., Ebos, J. M. L., Hlatky, L., & Hahnfeldt, P. (2014). Classical Mathematical Models for Description and Prediction of Experimental Tumor Growth. *PLOS Computational Biology*.

### Technical References

7. **Free Bits:**
   Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., & Welling, M. (2016). Improved Variational Inference with Inverse Autoregressive Flow. *NeurIPS*.

8. **Cyclical Annealing:**
   Fu, H., Li, C., Liu, X., Gao, J., Celikyilmaz, A., & Carin, L. (2019). Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing. *NAACL*.

9. **Total Correlation (beta-TCVAE):**
   Chen, T. Q., Li, X., Grosse, R. B., & Duvenaud, D. K. (2018). Isolating Sources of Disentanglement in Variational Autoencoders. *NeurIPS*.

10. **Checkerboard Artifacts:**
    Odena, A., Dumoulin, V., & Olah, C. (2016). Deconvolution and Checkerboard Artifacts. *Distill*.

---

*This report documents the SemiVAE model as implemented in the MenGrowth-Model project, Experiment 3, Run 6 configuration.*
