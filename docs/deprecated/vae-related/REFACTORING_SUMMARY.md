# Refactoring Summary: TC-VAE Removal & SBD Switchable Architecture

**Date**: 2025-12-26
**Changes**: Removed unused TC-VAE code, made SBD switchable, updated recommended training recipe

---

## Changes Made

### 1. Removed Unused TC-VAE Code

**Deleted files:**
- `src/vae/losses/tcvae.py` (TC-VAE loss decomposition)
- `src/vae/config/exp2_tcvae_sbd.yaml` (TC-VAE config)
- `slurm/execute_experiment2.sh` (TC-VAE slurm script)
- `src/vae/models/vae/tcvae_sbd.py` (old model file, replaced by vae_sbd.py)

**Removed from code:**
- `TCVAELitModule` class from `lit_modules.py` (lines 326-697)
- `compute_tcvae_loss` import
- `get_beta_tc_schedule` import

### 2. Renamed Files & Classes

| Old Name | New Name | Rationale |
|----------|----------|-----------|
| `tcvae_sbd.py` | `vae_sbd.py` | Model is used by DIP-VAE, not TC-VAE |
| `TCVAESBD` class | `VAESBD` class | Generic name (SBD is optional now) |

**Backward compatibility**: Added `TCVAESBD = VAESBD` alias in `__init__.py`

### 3. Created Model Factory (`src/engine/model_factory.py`)

**Purpose**: Separate model instantiation from training logic

**Features**:
- `create_vae_model(cfg)` function builds either:
  - `BaselineVAE` (standard transposed-conv decoder) when `use_sbd=false`
  - `VAESBD` (Spatial Broadcast Decoder) when `use_sbd=true`
- Handles all config parsing and parameter extraction
- Keeps model building logic in one place

**Integration**:
- `DIPVAELitModule.from_config()` now calls `create_vae_model(cfg)`
- Removed manual model instantiation code (27 lines → 3 lines)

### 4. Updated Configuration (`exp2_dipvae_sbd.yaml`)

**New parameters:**
```yaml
model:
  use_sbd: false              # NEW: Switchable decoder type
```

**Updated recommended values:**
```yaml
loss:
  lambda_start_epoch: 20      # Was: 0 (pre-train VAE before DIP)
train:
  kl_free_bits: 0.2           # Was: 0.05 (stronger collapse prevention)
```

### 5. Updated Documentation (`CLAUDE.md`)

**Major changes:**
- Removed "Experiment 2 (Alternative): β-TCVAE + SBD" section
- Updated Exp2 to "DIP-VAE-II with Configurable Decoder"
- Added `use_sbd` flag documentation
- Updated training recipes with new recommended defaults
- Updated codebase map to show `model_factory.py`
- Updated model routing documentation

---

## Rationale

### Why Remove TC-VAE?

1. **Not recommended**: TC-VAE's MWS estimator is biased with small batch sizes (B=2-8)
2. **DIP-VAE is superior**: Moment matching is more robust than density estimation in low-batch regimes
3. **Code simplification**: Remove 372 lines of unused code

### Why Make SBD Switchable?

1. **Collapse diagnosis**: SBD can cause the decoder to ignore `z` in 3D medical imaging
2. **Empirical testing**: Start with standard decoder to verify the encoder learns
3. **Flexibility**: If collapse persists with standard decoder, the problem is training dynamics, not architecture

### Why Delayed DIP Start (`lambda_start_epoch=20`)?

1. **Evidence from audit**: Covariance penalties introduced too early (epoch 0) may prevent encoder from learning
2. **Training recipe**: Pre-train a vanilla VAE for 20 epochs, then introduce DIP regularization
3. **Gradual introduction**: Allows the model to establish a latent representation before factorization pressure

### Why Stronger Free Bits (`kl_free_bits=0.2`)?

1. **Evidence from metrics**: Previous value (0.05) was too weak; AU=0, KL_raw=1.2 nats
2. **Expected floor**: 128 × 0.2 = 25.6 nats (vs 6.4 nats previously)
3. **Stronger pressure**: Forces encoder to allocate at least 0.2 nats/dim

---

## Migration Guide

### For Existing Experiments

**If you have old configs with `variant: "tcvae_sbd"`:**
```yaml
# Old (no longer supported)
model:
  variant: "tcvae_sbd"

# New (replace with)
model:
  variant: "dipvae_sbd"
  use_sbd: true  # to keep SBD behavior
```

**If you reference `TCVAESBD` in custom code:**
- The alias `TCVAESBD = VAESBD` provides backward compatibility
- Update to `VAESBD` for clarity

### For New Experiments

**Recommended starting config:**
```yaml
model:
  variant: "dipvae_sbd"
  use_sbd: false            # Start with standard decoder
loss:
  lambda_start_epoch: 20
  lambda_cov_annealing_epochs: 40
train:
  kl_free_bits: 0.2
```

**If encoder learns well (AU > 10, KL > 20 nats):**
```yaml
model:
  use_sbd: true             # Try SBD for better disentanglement
```

---

## Testing

All refactored code passes syntax checks:
```bash
python3 -m py_compile src/vae/training/lit_modules.py  # OK
python3 -m py_compile src/vae/models/vae/vae_sbd.py    # OK
python3 -m py_compile src/engine/model_factory.py       # OK
```

---

## Summary Statistics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Lines of code (lit_modules.py) | 1095 | 723 | -372 (-34%) |
| Number of LitModules | 3 | 2 | -1 |
| Model variants | 2 | 2 | 0 (but switchable) |
| Config files | 3 | 2 | -1 |
| Loss files | 3 | 2 | -1 |

**Code quality improvements:**
- Separation of concerns (model factory)
- Reduced duplication (one decoder choice)
- Clearer intent (use_sbd flag)
- Better documented (CLAUDE.md updated)
