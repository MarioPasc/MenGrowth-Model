# Code Review Changes Report

**Date**: 2026-01-15
**Reviewed by**: Claude Opus 4.5

## Summary

This report documents methodological flaws and code issues identified in the MenGrowth-Model codebase, along with the applied fixes.

---

## 1. Methodological Flaws Fixed

### 1.1 Missing `loss` Section in `vae.yaml` Causes Runtime Error

**File**: `src/vae/data/datasets.py:232`

**Issue**: The code accessed `cfg.loss.get("use_ddp_gather", False)` but `vae.yaml` (Exp1 baseline) did not define a `loss` section, causing an `AttributeError` when running baseline VAE training.

**Fix**:
- Added safe access pattern in `datasets.py`
- Added minimal `loss` section to `vae.yaml` for API consistency

```python
# Before (would fail for vae.yaml)
use_ddp_gather = cfg.loss.get("use_ddp_gather", False)

# After (handles missing loss section)
loss_cfg = cfg.get("loss", {})
use_ddp_gather = loss_cfg.get("use_ddp_gather", False) if loss_cfg else False
```

---

## 2. Code Bugs Fixed

### 2.1 `kl_free_bits` Default Mismatch

**Files**: `src/vae/training/lit_modules/vae.py:57,133`

**Issue**: `VAELitModule.__init__` defaulted `kl_free_bits=0.5` while `vae.yaml` used `0.1`. This inconsistency could cause unexpected behavior if config didn't explicitly set the value.

**Fix**: Changed default from `0.5` to `0.1` to match config convention.

```python
# Before
kl_free_bits: float = 0.5,

# After
kl_free_bits: float = 0.1,  # Match config default (0.1 nats/dim)
```

### 2.2 `lambda_phase` Calculation Ignores `lambda_start_epoch`

**File**: `src/vae/training/lit_modules/dipvae.py:324`

**Issue**: The logged `sched/lambda_phase` metric was calculated from `current_epoch` directly, ignoring `lambda_start_epoch`. This meant the phase was incorrect during the delayed start period.

**Fix**: Calculate phase from effective epoch after delayed start.

```python
# Before (incorrect)
phase = min(1.0, self.current_epoch / self.lambda_cov_annealing_epochs)

# After (correct)
effective_epoch = max(0, self.current_epoch - self.lambda_start_epoch)
phase = min(1.0, effective_epoch / self.lambda_cov_annealing_epochs)
```

---

## 3. Code Smells Fixed

### 3.1 `au_subset_fraction=0.99` Defeats Subsetting Purpose

**File**: `src/vae/config/dipvae.yaml:165`

**Issue**: Setting `au_subset_fraction: 0.99` meant using 99% of validation data, which defeats the purpose of subsetting for computational efficiency during Active Units (AU) computation.

**Fix**: Changed to `0.5` (50% of validation set).

```yaml
# Before (inefficient)
au_subset_fraction: 0.99

# After (reasonable subset)
au_subset_fraction: 0.5
```

### 3.2 `weight_decay` Not Configurable in AdamW

**Files**:
- `src/vae/training/lit_modules/vae.py`
- `src/vae/training/lit_modules/dipvae.py`
- `src/vae/config/vae.yaml`
- `src/vae/config/dipvae.yaml`

**Issue**: AdamW optimizer used the default `weight_decay=0.01` without exposing it as a configurable parameter. This limits flexibility for hyperparameter tuning.

**Fix**:
- Added `weight_decay` parameter to both `VAELitModule` and `DIPVAELitModule`
- Wired to config files with default `0.01`
- Updated `configure_optimizers()` to use the parameter

```python
# Added to both lit modules
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.lr,
        weight_decay=self.weight_decay,  # Now configurable
    )
    return optimizer
```

---

## 4. Documentation Created

### 4.1 `CLAUDE.md` - Project Guide

Created comprehensive project documentation including:
- Directory structure reference
- Two experimental configurations comparison
- Architecture details
- Loss function explanations
- Critical implementation notes (posterior collapse prevention, numerical stability)
- Config parameter reference
- Debugging tips and common issues

---

## Files Modified

| File | Changes |
|------|---------|
| `CLAUDE.md` | Created (new file) |
| `CHANGES_REPORT.md` | Created (this file) |
| `src/vae/config/vae.yaml` | Added `loss` section, `weight_decay` |
| `src/vae/config/dipvae.yaml` | Fixed `au_subset_fraction`, added `weight_decay` |
| `src/vae/data/datasets.py` | Safe access for `cfg.loss` |
| `src/vae/training/lit_modules/vae.py` | Fixed `kl_free_bits` default, added `weight_decay` |
| `src/vae/training/lit_modules/dipvae.py` | Fixed `lambda_phase` calculation, added `weight_decay` |

---

## Recommendations for Future Work

1. **Learning Rate Scheduler**: Consider adding cosine annealing or ReduceLROnPlateau for long training runs (1000 epochs in DIP-VAE).

2. **Deterministic Evaluation**: Currently, validation still samples `z = μ + ε·σ`. Consider option to use `z = μ` for deterministic reconstruction quality metrics.

3. **Data Augmentation**: The current pipeline is geometrically deterministic. Consider adding intensity augmentation (brightness, contrast) while preserving spatial consistency for ODE compatibility.

4. **Multi-GPU Testing**: The DDP gather logic should be tested with `devices > 1` to verify covariance computation correctness across ranks.
