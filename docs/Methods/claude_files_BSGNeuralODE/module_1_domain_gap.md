# Module 1: Domain Gap Analysis

## Overview
Quantify the distributional shift between BrainSegFounder's glioma training domain and the meningioma target domain. Establishes empirical necessity of LoRA adaptation (Module 2).

## Input
- Frozen BSF encoder (checkpoint: `finetuned_model_fold_0.pt`)
- `data_splits.json` from Module 0
- BraTS-GLI 2021 data (200 random subjects)
- BraTS-MEN 2024 data (`sdp_train` + `test` split subjects)

## Input Contract
```python
# BSF checkpoint
checkpoint: str  # path to finetuned_model_fold_0.pt

# Data splits
splits: dict  # from data_splits.json (Module 0 output)
```

## Output
- `domain_gap_report.json` — MMD, CKA, per-feature linear probe R²
- `domain_umap.png` — 2D UMAP with GLI and MEN colored separately
- `domain_feature_variance.png` — Per-dimension variance comparison

## Output Contract
```python
# Feature extraction output
h_gli: torch.Tensor   # shape [200, 768], dtype float32
h_men: torch.Tensor   # shape [200, 768], dtype float32

# Report
report: dict = {
    "mmd_squared": float,
    "mmd_pvalue": float,      # permutation test, n_perm=1000
    "cka": float,
    "linear_probe_r2": {
        "volume": float,
        "location": float,
        "shape": float,
    }
}
```

## Reuse Directives

| Existing File | What to Import | Path |
|---------------|----------------|------|
| `swin_loader.py` | `load_swin_encoder()`, `create_swinunetr()` | `src/growth/models/encoder/swin_loader.py` |
| `feature_extractor.py` | `FeatureExtractor` (already implements GAP pooling on encoder10) | `src/growth/models/encoder/feature_extractor.py` |
| `latent_quality.py` | `compute_mmd()`, `DomainShiftMetrics`, `compute_domain_shift_metrics()` | `src/growth/evaluation/latent_quality.py` |
| `latent_quality.py` | `compute_cka()`, `mmd_permutation_test()` (NEW — added in B1) | `src/growth/evaluation/latent_quality.py` |
| `semantic_features.py` | `compute_shape_array()` | `src/growth/data/semantic_features.py` |

## Code Requirements

1. **`DomainGapAnalyzer`** — Computes MMD (Gaussian kernel, σ=median distance, permutation test n=1000), CKA, and linear probe metrics. Note: `compute_mmd()` and `compute_cka()` already exist in `src/growth/evaluation/latent_quality.py`. Only `mmd_permutation_test()` needs to be newly added there (see Part B1).
   ```python
   class DomainGapAnalyzer:
       def compute_mmd(self, h1: Tensor, h2: Tensor, n_perm: int = 1000) -> Tuple[float, float]:
           """Returns (mmd_squared, p_value). Uses existing compute_mmd() + mmd_permutation_test()."""
       def compute_cka(self, h1: Tensor, h2: Tensor) -> float:
           """Returns CKA score in [0, 1]. Uses existing compute_cka()."""
       def linear_probe_r2(self, h: Tensor, targets: Tensor) -> float:
           """Ridge regression R² on held-out fold."""
   ```

2. **`FeatureExtractor`** — Already exists at `src/growth/models/encoder/feature_extractor.py`. Extracts GAP-pooled features from frozen BSF encoder (encoder10 output → AdaptiveAvgPool3d(1) → h ∈ ℝ^768).

3. **`DomainVisualizer`** — UMAP plots, variance comparison plots.

## Feature Extraction Pipeline
```
Input: [B, 4, 128, 128, 128]
  → SwinViT (frozen) → Stage 4: [B, 768, 4, 4, 4]
  → encoder10 (frozen) → [B, 768, 4, 4, 4]
  → AdaptiveAvgPool3d(1) → [B, 768]
  → h ∈ ℝ^768
```

## Configuration Snippet
```yaml
# configs/phase0_domain_gap.yaml
domain_gap:
  n_gli_subjects: 200
  n_men_subjects: 200  # from sdp_train + test splits
  mmd_n_permutations: 1000
  mmd_kernel: gaussian  # σ = median pairwise distance
  umap_n_neighbors: 15
  umap_min_dist: 0.1
  feature_cache_dir: ${paths.output_root}/features_cache
```

## Smoke Test
```python
import torch
# Synthetic features for testing without full dataset
h_gli = torch.randn(50, 768)
h_men = torch.randn(50, 768) + 0.5  # shifted distribution
analyzer = DomainGapAnalyzer()
mmd, pval = analyzer.compute_mmd(h_gli, h_men, n_perm=100)
assert 0 < mmd < 10
assert 0 <= pval <= 1
```

## Verification Tests

```
TEST_1.1: Feature extraction [BLOCKING]
  - Extract features from 10 GLI and 10 MEN subjects
  - Assert shape == [N, 768] for each
  - Assert no NaN or Inf values
  - Assert feature variance > 0 for all dimensions
  Recovery: Verify BSF checkpoint loaded correctly via swin_loader.py

TEST_1.2: MMD computation [BLOCKING]
  - Compute MMD between two identical sets → assert MMD ≈ 0
  - Compute MMD between GLI and random noise → assert MMD >> 0
  - Compute MMD between GLI and MEN → assert 0 < MMD < MMD(GLI, noise)
  Recovery: Check kernel bandwidth σ computation

TEST_1.3: Linear probe [DIAGNOSTIC]
  - Fit Ridge regression from frozen features → semantic targets
  - Assert returned R² values are finite
  - For frozen GLI encoder on MEN data: expect R² < 0.3 for volume
  Note: R² thresholds are DIAGNOSTIC — low values confirm domain gap

TEST_1.4: Permutation test [BLOCKING]
  - MMD permutation test produces p-value ∈ [0, 1]
  - With n_perm=100 (fast check), p-value for GLI vs MEN should be < 0.05
  Recovery: If p > 0.05, increase n_perm to 1000 and re-run
```
