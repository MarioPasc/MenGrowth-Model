# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MenGrowth-Model** is a deep learning framework for meningioma tumor growth prediction from multi-modal MRI. It implements two VAE variants:

- **Experiment 1 (Exp1)**: Baseline 3D VAE with ELBO loss
- **Experiment 2 (Exp2)**: β-TCVAE with Spatial Broadcast Decoder (SBD)

The goal is to learn disentangled representations that separate anatomical position from tumor content characteristics.

## Development Commands

### Setup and Installation

```bash
# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

The project uses a conda environment named `vae-dynamics` with Python ≥3.11.

### Running Experiments

```bash
# Experiment 1: Baseline VAE
python scripts/train.py --config src/vae_dynamics/config/exp1_baseline_vae.yaml

# Experiment 2: β-TCVAE with SBD
python scripts/train.py --config src/vae_dynamics/config/exp2_tcvae_sbd.yaml

# Resume from checkpoint
python scripts/train.py --config path/to/config.yaml --resume path/to/checkpoint.ckpt
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_shapes.py
pytest tests/test_shapes_exp2.py

# Run with coverage
pytest --cov=src/vae_dynamics

# Run specific test function
pytest tests/test_shapes.py::test_baseline_vae_shapes
```

### SLURM Cluster Execution

```bash
# Submit Experiment 1 to cluster
sbatch slurm/execute_experiment1.sh

# Submit Experiment 2 to cluster
sbatch slurm/execute_experiment2.sh
```

SLURM scripts handle:
- Dynamic GPU assignment (finds available GPU 0-7)
- Conda environment activation
- Config file modification for cluster paths
- Results directory creation

## Architecture Overview

### Core Design Patterns

**Encoder-Decoder Architecture:**
- **Encoder**: 3D ResNet with GroupNorm (not BatchNorm due to small batch sizes 2-8)
  - Downsamples 128³ → 64³ → 32³ → 16³ → 8³
  - Outputs μ and log(σ²) for posterior q(z|x)
- **Exp1 Decoder**: Standard transposed convolutions
- **Exp2 Decoder**: Spatial Broadcast Decoder that explicitly provides position coordinates

**Why Spatial Broadcast Decoder?**
The SBD removes positional encoding from latent space by:
1. Broadcasting latent z to spatial grid [B, 128] → [B, 128, 8, 8, 8]
2. Concatenating normalized coordinates (D, H, W) ∈ [-1, 1]
3. Upsampling concatenated tensor to full resolution

This forces the model to encode *content* (tumor characteristics) in z while position is handled by explicit coordinates.

### Loss Functions

**Exp1 - ELBO Loss:**
```
Loss = MSE(x, x̂) + β·KL(q(z|x) || p(z))
```
- β anneals from 0 → 1.0 over 40 epochs (prevents posterior collapse)

**Exp2 - β-TCVAE Loss:**
```
Loss = Recon + α·MI + β_tc·TC + γ·DWKL
```
Where:
- **MI** (Mutual Information): How much z explains x
- **TC** (Total Correlation): Statistical dependence among latent dimensions
- **DWKL** (Dimension-wise KL): Regularization per dimension

**Key Implementation Detail:** Uses Minibatch-Weighted Sampling (MWS) to estimate marginal q(z):
```
log q(z_i) = logsumexp_j(log q(z_i|x_j)) - log(N·M)
```

### Data Pipeline

**Input Format:** NIfTI files (.nii.gz) with structure:
```
BraTS_Men_Train/
  ├── BraTS-MEN-00001-000/
  │   ├── *-t1c.nii.gz
  │   ├── *-t1n.nii.gz
  │   ├── *-t2f.nii.gz
  │   ├── *-t2w.nii.gz
  │   └── *-seg.nii.gz
  └── ...
```

**MONAI Transform Pipeline:**
1. LoadImaged → load with metadata
2. EnsureChannelFirstd → [C, D, H, W] format
3. Orientationd → reorient to "RAS"
4. Spacingd → resample to isotropic 1.875³ mm³
5. NormalizeIntensityd → z-score normalize per modality
6. ConcatItemsd → stack 4 modalities into [4, D, H, W]
7. ResizeWithPadOrCropd → deterministic pad/crop to 128³
8. ToTensord → convert to PyTorch tensors

**Critical Implementation:** Uses `safe_collate()` function in DataLoader that:
- Handles both numpy arrays and tensors (prevents `AttributeError` when PersistentDataset returns cached numpy arrays)
- **Filters to only training keys** ("image", "seg", "id") to exclude MONAI metadata
- Prevents `TypeError: iteration over a 0-d array` by avoiding MONAI MetaTensor metadata keys that contain scalar values

### Module Interaction Flow

```
train.py
  ├── Load config (OmegaConf)
  ├── data.build_subject_index() → scan directories for NIfTI files
  ├── data.create_train_val_split() → deterministic 90/10 split
  ├── data.get_dataloaders() → PersistentDataset + safe_collate
  ├── Instantiate model: BaselineVAE or TCVAESBD
  ├── Wrap in Lightning module: VAELitModule or TCVAELitModule
  ├── Configure callbacks: ModelCheckpoint, ReconstructionCallback, TrainingLoggingCallback
  └── pl.Trainer.fit() → executes training loop
```

### Configuration Schema

Experiments are fully configured via YAML. Key parameters:

**Data:**
- `root_dir`: Path to BraTS dataset
- `modalities`: ["t1c", "t1n", "t2f", "t2w"] (order matters)
- `roi_size`: [128, 128, 128] (target volume size)
- `spacing`: [1.875, 1.875, 1.875] (isotropic resampling)
- `batch_size`: 2-8 (small due to 3D volume memory)
- `val_split`: 0.1

**Model:**
- `z_dim`: 128 (latent dimension)
- `input_channels`: 4 (stacked modalities)
- `base_filters`: 32 (channel scaling)
- `norm`: "GROUP" (GroupNorm for small batches)
- `sbd_grid_size`: [8, 8, 8] (Exp2 only - broadcast resolution)

**Training:**
- `precision`: "16-mixed" (FP16 with FP32 fallback)
- `gradient_clip_val`: 1.0 (prevent FP16 overflow)
- `loss_reduction`: "mean" (more stable than "sum" in FP16)

### Numerical Stability Considerations

The codebase handles FP16 mixed precision carefully:

1. **Gradient Clipping:** L2 norm clipped to 1.0 to prevent explosion
2. **Mean Reduction:** Loss normalized by batch size (not summed over millions of voxels)
3. **FP32 Computation:** TC-VAE loss computed in FP32 when `compute_in_fp32: true`
4. **GroupNorm:** Used instead of BatchNorm (stable for small batches)

## Critical Implementation Details

### Preventing Experiment 1 Breakage

When modifying data loading or model code:
- **Never remove** the `safe_collate()` function in `datasets.py`
- **Always test both experiments** after changes to shared code
- **Preserve tensor shape contracts**: [B, C, D, H, W] throughout pipeline
- **Maintain modality order**: ["t1c", "t1n", "t2f", "t2w"]

### Model Output Signatures

**Exp1 (BaselineVAE):** Returns 3 values
```python
x_hat, mu, logvar = model(x)
```

**Exp2 (TCVAESBD):** Returns 4 values
```python
x_hat, mu, logvar, z = model(x)
```

Code that handles both must use indexing: `x_hat = model(x)[0]`

### Persistent Dataset Caching

PersistentDataset caches preprocessed volumes to `run_dir/cache/`.

**Important:** If transforms change, delete cache directories manually:
```bash
rm -rf experiments/runs/*/cache/
```

Otherwise, stale cached data may be used.

### Random Seed Management

Reproducibility requires seeding:
- PyTorch RNG: `torch.manual_seed(seed)`
- NumPy RNG: `np.random.seed(seed)`
- DataLoader workers: `set_seed(seed, workers=True)`
- Train/val split: Uses seeded `torch.Generator`

All seeds set from `cfg.train.seed` (default: 42).

## Output Directory Structure

Training creates timestamped runs:
```
experiments/runs/exp1_baseline_vae_YYYYMMDD_HHMMSS/
  ├── config.yaml                  # Resolved config
  ├── checkpoints/
  │   ├── vae-epoch=XXX-val_loss=X.XXXX.ckpt
  │   └── last.ckpt
  ├── logs/
  │   └── version_0/
  │       └── metrics.csv          # Training metrics
  ├── recon/
  │   └── epoch_XXXX/
  │       └── sample_XX_*.png      # Reconstruction visualizations
  ├── splits/
  │   ├── train_split.csv
  │   └── val_split.csv
  └── cache/
      ├── train/                   # Cached preprocessed volumes
      └── val/
```

## Testing Strategy

Tests verify:
- **Shape correctness**: All tensors have expected dimensions
- **Loss finiteness**: No NaN/Inf values
- **Gradient flow**: Backward pass produces gradients
- **Model compatibility**: Both Exp1 and Exp2 work correctly

Run tests before committing changes to models, losses, or data pipeline.

## Technology Stack

- **PyTorch 2.0+**: Deep learning framework
- **MONAI 1.3+**: Medical imaging transforms and datasets
- **PyTorch Lightning 2.0+**: Training orchestration
- **OmegaConf 2.3+**: Hierarchical configuration
- **NiBabel 5.0+**: NIfTI file I/O
- **Python ≥3.11**: Required for type hints and performance

## Common Issues and Solutions

**Issue:** `AttributeError: 'numpy.ndarray' object has no attribute 'numel'`
**Solution:** Ensure `safe_collate()` is used in DataLoader (already implemented)

**Issue:** `TypeError: iteration over a 0-d array` during validation
**Root Cause:** MONAI's `LoadImaged(image_only=False)` creates MetaTensor objects with metadata keys containing scalar values. PyTorch Lightning's batch size extraction recursively iterates through the batch and fails when encountering 0-d arrays in metadata.
**Solution:** `safe_collate()` now filters to only process training keys ("image", "seg", "id"), excluding MONAI metadata (already implemented)

**Issue:** `RuntimeError: CUDA out of memory`
**Solution:** Reduce `batch_size` in config (try 2 for 16GB GPU, 4-8 for 24GB+)

**Issue:** Loss becomes NaN during training
**Solution:** Check `gradient_clip_val` is enabled; verify `loss_reduction: "mean"` in config

**Issue:** PersistentDataset using stale cached data
**Solution:** Delete cache directories after changing transforms

**Issue:** Experiment 1 works but Experiment 2 fails
**Solution:** Check model returns 4 values (x_hat, mu, logvar, z); verify `sbd_grid_size` in config
