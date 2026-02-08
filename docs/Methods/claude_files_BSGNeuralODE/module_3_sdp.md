# Module 3: Supervised Disentangled Projection (Phase 2)

## Overview
Train a lightweight projection network mapping frozen encoder features h ∈ ℝ^768 to a structured, disentangled latent space z ∈ ℝ^128 with semantic partitions for volume, location, shape, and residual.

## Input
- `phase1_encoder_merged.pt` from Module 2 (frozen encoder)
- BraTS-MEN `train_pool` (800 subjects) — precomputed features h ∈ ℝ^768
- Semantic features cache from Module 0

## Input Contract
```python
# Precomputed encoder features (frozen encoder, cached)
h: torch.Tensor          # shape [800, 768], dtype float32 (train_pool)
h_val: torch.Tensor      # shape [100, 768], dtype float32 (val set)

# Semantic targets (from semantic_features_cache/)
y_vol: torch.Tensor      # shape [N, 4], log-volumes (normalized μ=0, σ=1)
y_loc: torch.Tensor      # shape [N, 3], centroid coords (normalized)
y_shape: torch.Tensor    # shape [N, 6], all 6 shape features (normalized)
```

## Output
- `phase2_sdp.pt` — Trained SDP network + semantic heads
- `phase2_training_log.csv` — Per-epoch losses (each term individually)
- `phase2_quality_report.json` — R² per target, dCor between partitions, per-dimension variance
- `latent_umap.png` — UMAP colored by semantics

## Output Contract
```python
# SDP output
z: torch.Tensor           # shape [N, 128], dtype float32

# Partitioned latent space
z_vol: torch.Tensor       # shape [N, 24], indices 0-23
z_loc: torch.Tensor       # shape [N, 8], indices 24-31
z_shape: torch.Tensor     # shape [N, 12], indices 32-43
z_residual: torch.Tensor  # shape [N, 84], indices 44-127

# Semantic head predictions
y_hat_vol: torch.Tensor   # shape [N, 4]
y_hat_loc: torch.Tensor   # shape [N, 3]
y_hat_shape: torch.Tensor # shape [N, 6]

# Quality report
report: dict = {
    "r2_vol": float,       # ≥ 0.80 (BLOCKING minimum)
    "r2_loc": float,       # ≥ 0.85 (BLOCKING minimum)
    "r2_shape": float,     # ≥ 0.30 (BLOCKING minimum)
    "dcor_vol_loc": float, # < 0.20 target
    "dcor_vol_shape": float,
    "dcor_loc_shape": float,
    "max_cross_partition_corr": float,  # < 0.30
    "pct_dims_variance_gt_05": float,   # ≥ 85%
}
```

## Reuse Directives

| Existing File | What to Import | Path |
|---------------|----------------|------|
| `swin_loader.py` | `load_swin_encoder()` | `src/growth/models/encoder/swin_loader.py` |
| `semantic_features.py` | `compute_shape_features()` | `src/growth/data/semantic_features.py` |
| `sdp.py` | Stub — implement fully | `src/growth/models/projection/sdp.py` |

## Architecture

```
Frozen SwinViT encoder10: [B, 768, 4, 4, 4]
       ↓
AdaptiveAvgPool3d(1): [B, 768]
       ↓
LayerNorm(768)
       ↓
SpectralNorm(Linear(768, 512)) → GELU → Dropout(0.1)
       ↓
SpectralNorm(Linear(512, 128))
       ↓
z ∈ ℝ^128 = [z_vol(24) | z_loc(8) | z_shape(12) | z_residual(84)]
       ↓                ↓               ↓
   π_vol(24→4)     π_loc(8→3)     π_shape(12→6)
       ↓                ↓               ↓
   ŷ_vol ∈ ℝ^4    ŷ_loc ∈ ℝ^3    ŷ_shape ∈ ℝ^6
```

**Spectral normalization on ALL linear layers** (see DECISIONS.md D13).

## Code Requirements

1. **`SDP`** — 2-layer MLP with LayerNorm, GELU, Dropout, SN on all linear layers.
   ```python
   class SDP(nn.Module):
       def __init__(self, in_dim: int = 768, hidden_dim: int = 512, out_dim: int = 128, dropout: float = 0.1):
           self.norm = nn.LayerNorm(in_dim)
           self.fc1 = nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim))
           self.act = nn.GELU()
           self.drop = nn.Dropout(dropout)
           self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, out_dim))
       def forward(self, h: torch.Tensor) -> torch.Tensor:
           """h: [B, 768] → z: [B, 128]"""
   ```

2. **`LatentPartition`** — Splits z into named partitions.
   ```python
   class LatentPartition:
       partitions = {"vol": (0, 24), "loc": (24, 32), "shape": (32, 44), "residual": (44, 128)}
       def split(self, z: torch.Tensor) -> dict:
           """Returns {name: z_partition} dict."""
   ```

3. **`SemanticHeads`** — Per-partition linear projection heads.
   ```python
   class SemanticHeads(nn.Module):
       def __init__(self):
           self.vol_head = nn.Linear(24, 4)
           self.loc_head = nn.Linear(8, 3)
           self.shape_head = nn.Linear(12, 6)
   ```

4. **`SDPLoss`** — Composite loss with configurable weights.
5. **`dCorLoss`** — Differentiable distance correlation between partition tensors.
6. **`VICRegLoss`** — Cross-partition covariance + variance hinge loss.
7. **`SDPLitModule`** (PyTorch Lightning) — Encoder frozen; only SDP + heads trained.

## Loss Function

$$\mathcal{L}_{SDP} = \sum_p \lambda_p \mathcal{L}_p^{sem} + \lambda_{cov} \mathcal{L}_{cov} + \lambda_{var} \mathcal{L}_{var} + \lambda_{dCor} \sum_{i<j} dCor(z_i, z_j)$$

## Normalization Scope
**Compute μ,σ on train_pool (800) only.** Apply same parameters to val (100), test (100), and Andalusian cohort without recomputation. See DECISIONS.md D14.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| λ_vol | 20.0 |
| λ_loc | 12.0 |
| λ_shape | 15.0 |
| λ_cov | 5.0 |
| λ_var | 5.0 |
| λ_dCor | 2.0 |
| Learning rate | 1e-3 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Epochs | 100 |
| Batch size | Full-batch (800) |
| Scheduler | Cosine decay, 5-epoch warmup |

## Curriculum (4-phase) — see DECISIONS.md D5

| Phase | Epochs | Active Losses |
|-------|--------|---------------|
| Warm-up | 0–9 | L_var only |
| Semantic | 10–39 | + L_vol, L_loc, L_shape |
| Independence | 40–59 | + L_cov, L_dCor |
| Full | 60–100 | All at full strength |

## Configuration Snippet
```yaml
# configs/phase2_sdp.yaml
sdp:
  in_dim: 768
  hidden_dim: 512
  out_dim: 128
  dropout: 0.1
  spectral_norm: all  # on all linear layers

partition:
  vol_dim: 24
  loc_dim: 8
  shape_dim: 12
  residual_dim: 84

targets:
  n_vol: 4       # log-volumes
  n_loc: 3       # centroid coordinates
  n_shape: 6     # ALL 6 shape features (see D12)

loss:
  lambda_vol: 20.0
  lambda_loc: 12.0
  lambda_shape: 15.0
  lambda_cov: 5.0
  lambda_var: 5.0
  lambda_dcor: 2.0
  gamma_var: 1.0

training:
  epochs: 100
  lr: 1e-3
  optimizer: adamw
  weight_decay: 0.01
  batch_size: 800  # full-batch
  scheduler: cosine
  warmup_epochs: 5
  curriculum: true
```

## Smoke Test
```python
import torch

# Synthetic features
h = torch.randn(800, 768)
sdp = SDP(in_dim=768, hidden_dim=512, out_dim=128)
z = sdp(h)
assert z.shape == (800, 128)

# Check spectral norm
for name, module in sdp.named_modules():
    if isinstance(module, nn.Linear):
        # Spectral norm should constrain singular values ≈ 1
        sigma = torch.linalg.svdvals(module.weight)[0]
        assert sigma <= 1.1, f"SN not applied to {name}"
```

## Verification Tests

```
TEST_3.1: SDP forward pass [BLOCKING]
  - Input: h ∈ ℝ^{800×768} (full batch)
  - Assert output z shape == [800, 128]
  - Assert z partitions sum to 128 dimensions
  - Assert semantic head outputs: vol [800,4], loc [800,3], shape [800,6]
  Recovery: Check SDP architecture dimensions

TEST_3.2: Spectral normalization [BLOCKING]
  - Assert spectral norm of BOTH linear layer weights ≈ 1.0
  - After 10 training steps, assert spectral norm still ≈ 1.0
  Recovery: Verify nn.utils.spectral_norm applied to both fc1 and fc2

TEST_3.3: Loss computation [BLOCKING]
  - Compute each loss term individually, assert all finite and ≥ 0
  - Assert dCor ∈ [0, 1] for each partition pair
  - Assert L_var = 0 when all dimensions have std > γ
  Recovery: Check loss implementation, esp. dCor denominator ε

TEST_3.4: Training convergence [BLOCKING]
  - After 100 epochs, assert total loss decreased by ≥ 50%
  - Assert no NaN losses during training
  Recovery: Reduce learning rate to 5e-4, check for gradient explosion

TEST_3.5: Semantic quality [BLOCKING]
  - On val set: Vol R² ≥ 0.80 (target: 0.90)
  - On val set: Loc R² ≥ 0.85 (target: 0.95)
  - On val set: Shape R² ≥ 0.30 (target: 0.40)
  Recovery steps (in order):
    1. Increase epochs to 200
    2. Reduce dCor weight to 1.0
    3. Disable curriculum (flat schedule with all losses from epoch 0)
    4. If still failing, report R² values and stop

TEST_3.6: Disentanglement quality [BLOCKING]
  - Max cross-partition Pearson correlation < 0.30
  - dCor(vol, loc) < 0.20
  - Per-dimension variance > 0.3 for ≥ 90% of dimensions
  Recovery: Increase λ_cov to 10.0 and λ_dCor to 5.0

TEST_3.7: Latent space smoothness (Lipschitz check) [DIAGNOSTIC]
  - For 100 pairs of similar inputs (||h1 - h2|| < ε):
    Assert ||z1 - z2|| / ||h1 - h2|| < L for some bounded L
  Note: DIAGNOSTIC — bound may be loose due to GELU
```
