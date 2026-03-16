# BrainSegFounder Alignment Verification Findings

Date: 2026-02-13

## Audit Result

Comprehensive comparison of BrainSegFounder's vendored code, its paper, and our pipeline.

### Confirmed Correct (no changes needed)
- Channel order: `[t2f, t1c, t1n, t2w]` = [FLAIR, T1ce, T1, T2]
- ROI size: 128^3 (from `launch.sh --roi_x=128`)
- Label conversion: TC/WT/ET from labels {0,1,2,3} with GLI remap
- 3-channel sigmoid output (TC, WT, ET hierarchical overlapping)
- Architecture: feature_size=48, depths=(2,2,2,2), num_heads=(3,6,12,24)
- Z-score normalization: NormalizeIntensityd(nonzero=True, channel_wise=True)
- Weight loading: fine-tuned checkpoint has clean keys (no `module.` prefix)
- encoder10 -> 768-dim with GAP for downstream features

### Discrepancies Fixed
1. **Missing `k_divisible` in CropForegroundd** (transforms.py) — added `k_divisible=list(roi_size)` to match BrainSegFounder
2. **Wrong augmentation probabilities** (transforms.py) — `RandScaleIntensityd` and `RandShiftIntensityd` changed from `prob=0.5` to `prob=1.0`
3. **Stale docstrings** (feature_extractor.py) — updated hidden_states dimensions for 128^3 input with MONAI 1.5+
4. **Wrong example sizes** (swin_loader.py) — docstring examples updated from 96^3 to 128^3
5. **Extra RandRotate90d** (transforms.py) — documented as beyond BrainSegFounder's pipeline (kept for domain adaptation)

### Not a Bug (despite plan)
- `img_size` parameter: Not accepted in newer MONAI SwinUNETR — no change needed.

### Test Status
All 259 tests pass after changes.

---

## Spatial Analysis: 128^3 Center Crop Loses Tumor Information

Date: 2026-02-13

### Motivation

Feature extraction (Phase 2+) uses center crop for deterministic patching.
Empirical analysis of 49 BraTS-MEN and 50 BraTS-GLI samples revealed severe tumor clipping.

### Results

| Crop Size | MEN bbox fully contained | GLI bbox fully contained | MEN worst-case voxel coverage |
|-----------|--------------------------|--------------------------|-------------------------------|
| **128^3** | **38.8%** | **30.0%** | **24.1%** (catastrophic) |
| 160^3 | 89.8% | 82.0% | 98.1% |
| **192^3** | **100.0%** | **100.0%** | **100.0%** |
| 224^3 | 100.0% | 100.0% | 100.0% |

### Key Observations

- Brain volumes after CropForeground: ~240x256x256 (MEN), ~251x256x253 (GLI)
- Meningiomas are more peripheral (0.241) than gliomas (0.211)
- BrainSegFounder itself never uses center crop for validation (uses full volumes + sliding window)
- Both datasets are natively 1.0mm isotropic
- SwinUNETR handles arbitrary input sizes divisible by 32 (GAP normalizes spatial dims)

### Decision

- **192^3 for feature extraction** (Phase 2+): guarantees 100% tumor containment
- **128^3 for LoRA training** (Phase 1): matches BrainSegFounder fine-tuning, random crop covers tumors stochastically
- A100 80GB can handle 192^3 inference (~22-26 GB at batch_size=1)

### Reproducibility

Analysis script: `experiments/analysis/analyze_brats_spatial.py`
