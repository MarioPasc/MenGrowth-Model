# Module 0: Data Infrastructure & Preprocessing

## Overview
Set up data loading, semantic feature computation, data splitting, and transform pipelines for the entire pipeline.

## Input
- Raw NIfTI files: BraTS-MEN 2024 (1000 subjects), Andalusian cohort (42 patients, 137 studies), BraTS-GLI 2021 (1251 subjects)
- Paths configured in `configs/foundation.yaml`

## Output
- `data_splits.json` — Deterministic split assignment (seed=42)
- `semantic_features_cache/` — Precomputed volume, location, shape features per subject
- Validated `BraTSMENDataset` and `LongitudinalDataset` classes

## Output Contract
```python
# data_splits.json structure
splits: dict = {
    "lora_train": List[str],    # 525 subject IDs
    "lora_val": List[str],      # 100 subject IDs
    "sdp_train": List[str],     # 225 subject IDs
    "test": List[str],          # 150 subject IDs
}

# Semantic features per subject
features: dict = {
    "volumes": np.ndarray,      # shape [4], log(V+1) for [total, NCR, ED, ET]
    "centroid": np.ndarray,     # shape [3], physical mm coordinates
    "shape": np.ndarray,        # shape [3], [sphericity, surface_area_log, solidity]
}
```

## Data Splits

| Split | N | Use |
|-------|---|-----|
| `lora_train` | 525 | Phase 1 LoRA training |
| `lora_val` | 100 | Phase 1 early stopping |
| `sdp_train` | 225 | Phase 2 SDP training (never seen during LoRA) |
| `test` | 150 | Final held-out evaluation |

## Reuse Directives

| Existing File | What to Import | Path |
|---------------|----------------|------|
| `semantic_features.py` | `compute_shape_array()` (returns 3 features: sphericity, surface_area_log, solidity) | `src/growth/data/semantic_features.py` |
| `transforms.py` | `get_train_transforms()`, `get_val_transforms()` | `src/growth/data/transforms.py` |
| `bratsmendata.py` | `BraTSMENDataset` (extend if needed) | `src/growth/data/bratsmendata.py` |

**Important:** Use `compute_shape_array()` (3 features: sphericity, surface_area_log, solidity). See DECISIONS.md D12.

## Code Requirements

1. **`BraTSMENDataset`** — MONAI-based dataset class loading 4-channel MRI + segmentation + semantic features.
   - Channel order: `[t2f, t1c, t1n, t2w]` (see DECISIONS.md D6)
   - Caches semantic features to disk on first computation
   - Supports both train and val transforms

2. **`LongitudinalDataset`** — Dataset class for Andalusian cohort with temporal metadata.
   - Returns `(volume, label, patient_id, timepoint_index, acquisition_date)`
   - Computes Δt between timepoints in months

3. **`DataSplitter`** — Deterministic split generation.
   ```python
   def create_splits(subject_ids: List[str], seed: int = 42) -> dict:
       """Returns split dict with lora_train, lora_val, sdp_train, test."""
   ```

4. **Transform pipelines** — MONAI transform chains matching BrainSegFounder convention:
   - Training: `CropForeground` + `RandSpatialCrop(128³)` + augmentations
   - Validation: `CropForeground` + `CenterSpatialCrop(128³)`
   - Inference: Sliding window 128³, stride 64, Gaussian blending

## Configuration Snippet
```yaml
# configs/foundation.yaml (data section)
data:
  channel_order: [t2f, t1c, t1n, t2w]
  label_mapping: {0: background, 1: NCR, 2: ED, 3: ET}
  seed: 42
  splits:
    lora_train: 525
    lora_val: 100
    sdp_train: 225
    test: 150
```

## Smoke Test (synthetic data)
```python
# Generate synthetic tensors for testing without full dataset
import torch
x = torch.randn(2, 4, 128, 128, 128)  # 2 subjects, 4 channels, 128³
y = torch.randint(0, 4, (2, 128, 128, 128))  # segmentation labels
```

## Verification Tests

```
TEST_0.1: Dataset loading [BLOCKING]
  - Load 5 random BraTS-MEN subjects
  - Assert output shape == [4, 128, 128, 128] (float32)
  - Assert label shape == [128, 128, 128] (int)
  - Assert unique label values ⊆ {0, 1, 2, 3}
  - Assert all 4 channels have nonzero variance
  Recovery: Check channel_order in config, verify NIfTI file paths

TEST_0.2: Semantic features [BLOCKING]
  - For each subject, compute semantic features
  - Assert V_total > 0 (tumor exists)
  - Assert centroid within volume bounds
  - Assert 0 < sphericity ≤ 1.0
  - Assert surface_area > 0
  - Assert shape vector has length 3 (sphericity, surface_area_log, solidity)
  Recovery: Check that compute_shape_array() is used (see DECISIONS.md D12)

TEST_0.3: Data splits [BLOCKING]
  - Load splits, assert no overlap between any pair of sets
  - Assert |lora_train| + |lora_val| + |sdp_train| + |test| == 1000
  - Assert splits are deterministic (reload and compare)
  Recovery: Verify seed=42 is set before splitting

TEST_0.4: Transform consistency [BLOCKING]
  - Apply train transform 10 times to same subject
  - Assert output shapes are always [4, 128, 128, 128]
  - Apply val transform, assert deterministic output
  Recovery: Check MONAI transform pipeline ordering

TEST_0.5: Longitudinal dataset [DIAGNOSTIC]
  - Load 3 patients from Andalusian cohort
  - Assert each has ≥ 2 timepoints
  - Assert temporal metadata (dates) are monotonically ordered
  - Assert all timepoints have shape [4, 128, 128, 128]
  Note: DIAGNOSTIC — skip if Andalusian data not yet available
```
