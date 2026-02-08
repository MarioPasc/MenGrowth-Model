# Module 4: Cohort Encoding & Harmonization (Phase 3)

## Overview
Encode all BraTS-MEN and Andalusian cohort MRI volumes through the frozen encoder + SDP pipeline, assess and apply ComBat harmonization if needed, and construct per-patient temporal trajectories.

## Input
- `phase1_encoder_merged.pt` from Module 2 (frozen encoder)
- `phase2_sdp.pt` from Module 3 (frozen SDP)
- BraTS-MEN data (1000 subjects)
- Andalusian cohort data (42 patients, 137 studies)
- Scanner metadata for Andalusian cohort

## Input Contract
```python
# Frozen pipeline
encoder: nn.Module     # frozen, from phase1_encoder_merged.pt
sdp: SDP               # frozen, from phase2_sdp.pt

# Andalusian cohort metadata
metadata: List[dict] = [
    {
        "patient_id": str,
        "timepoints": [
            {"study_id": str, "date": str, "scanner_id": str, "path": str},
            ...
        ]
    },
    ...
]
```

## Output
- `latent_bratsmen.pt` — z ∈ ℝ^{1000×128} (BraTS-MEN latent vectors)
- `latent_andalusian.pt` — z ∈ ℝ^{137×128} with patient/timepoint metadata
- `latent_andalusian_harmonized.pt` — z* ∈ ℝ^{137×128} (after ComBat, if applied)
- `combat_assessment.json` — MMD before/after, UMAP, decision rationale
- `trajectories.json` — Per-patient latent trajectories

## Output Contract
```python
# BraTS-MEN latents
latent_bratsmen: torch.Tensor   # shape [1000, 128], dtype float32

# Andalusian latents (with metadata)
latent_andalusian: dict = {
    "z": torch.Tensor,            # shape [137, 128]
    "patient_ids": List[str],     # length 137
    "timepoint_idx": List[int],   # length 137
    "dates": List[str],           # length 137
    "scanner_ids": List[str],     # length 137
}

# Trajectories (input to Module 5)
trajectories: List[dict] = [
    {
        "patient_id": str,
        "timepoints": [
            {"z": List[float],  # length 128
             "t": float,        # time in months from first scan
             "date": str},
            ...
        ]
    },
    ...  # 42 patients
]
```

## Reuse Directives

| Existing File | What to Import | Path |
|---------------|----------------|------|
| `swin_loader.py` | `load_swin_encoder()` | `src/growth/models/encoder/swin_loader.py` |
| `sdp.py` | `SDP` class | `src/growth/models/projection/sdp.py` |

## Code Requirements

1. **`CohortEncoder`** — Batch encoding pipeline.
   ```python
   class CohortEncoder:
       def __init__(self, encoder: nn.Module, sdp: SDP):
           """Both encoder and sdp are frozen (eval mode, no grad)."""
       def encode_dataset(self, dataset: Dataset) -> torch.Tensor:
           """Returns z ∈ ℝ^{N×128}. Deterministic (no dropout)."""
   ```

2. **`SlidingWindowEncoder`** — For volumes larger than 128³.
   ```python
   class SlidingWindowEncoder:
       def encode(self, volume: Tensor, mask: Tensor) -> Tensor:
           """
           Sliding window 128³, stride 64, tumor-weighted averaging:
           h = Σ(w_p · GAP(encoder(x_p))) / Σ(w_p)
           w_p = |mask_p ∩ tumor| / |patch_p|
           Returns h ∈ ℝ^768
           """
   ```

3. **`LatentComBat`** — Wrapper around neuroCombat.
   ```python
   class LatentComBat:
       def fit_transform(self, z: Tensor, site_labels: List[str],
                         covariates: Optional[pd.DataFrame] = None) -> Tensor:
           """Apply ComBat harmonization. Returns z* ∈ ℝ^{N×128}."""
   ```

4. **`HarmonizationAssessor`** — Pre/post ComBat assessment.
   ```python
   class HarmonizationAssessor:
       def assess(self, z_ref: Tensor, z_target: Tensor) -> dict:
           """Returns MMD, UMAP coords, KS test results."""
       def recommend(self, assessment: dict) -> str:
           """Returns 'harmonize' or 'skip' with rationale."""
   ```

5. **`TrajectoryBuilder`** — Constructs per-patient temporal trajectories.
   ```python
   class TrajectoryBuilder:
       def build(self, latents: dict, metadata: List[dict]) -> List[dict]:
           """
           Groups latent vectors by patient, orders by date,
           computes Δt in months. Returns trajectories list.
           """
   ```

## Scanner Verification

**TEST_4.6 checks scanner consistency per patient.** If any patient has scanner changes between visits:
- Log warning with affected patient IDs
- Use LongComBat (Beer et al., NeuroImage, 2020) instead of standard ComBat
- See DECISIONS.md and methodology Section 4.1

## Configuration Snippet
```yaml
# configs/phase3_encoding.yaml
encoding:
  batch_size: 8              # for volume encoding (GPU limited)
  sliding_window_stride: 64
  deterministic: true        # no dropout during encoding

harmonization:
  method: combat             # or longcombat if scanner changes detected
  assess_before: true
  mmd_threshold: 0.05        # p-value threshold for harmonization decision
  covariates: [age, sex]     # if available

trajectories:
  time_unit: months
  min_timepoints: 2
```

## Smoke Test
```python
import torch

# Synthetic latents
z_bratsmen = torch.randn(100, 128)
z_andalusian = torch.randn(20, 128) + 0.3  # scanner shift

# Test trajectory building
trajectories = [
    {"patient_id": "P001",
     "timepoints": [
         {"z": torch.randn(128).tolist(), "t": 0.0, "date": "2020-01-15"},
         {"z": torch.randn(128).tolist(), "t": 6.5, "date": "2020-08-01"},
         {"z": torch.randn(128).tolist(), "t": 14.2, "date": "2021-03-20"},
     ]}
]
assert len(trajectories[0]["timepoints"]) == 3
assert trajectories[0]["timepoints"][1]["t"] > trajectories[0]["timepoints"][0]["t"]
```

## Verification Tests

```
TEST_4.1: Encoding determinism [BLOCKING]
  - Encode same volume twice through frozen pipeline
  - Assert ||z1 - z2|| < 1e-6 (deterministic, no dropout)
  Recovery: Ensure model.eval() and torch.no_grad()

TEST_4.2: Encoding shape [BLOCKING]
  - Encode 5 Andalusian volumes
  - Assert output shape [5, 128]
  - Assert no NaN values
  Recovery: Check encoder + SDP pipeline connectivity

TEST_4.3: ComBat temporal preservation [BLOCKING]
  - For a patient with 3 timepoints:
    Δz = z(t2) - z(t1), Δz* = z*(t2) - z*(t1)
  - Assert Δz* = Δz / δ_site (up to numerical precision)
  - Assert ratio is constant across all dimension pairs
  Recovery: Verify same scanner assumption holds (see TEST_4.6)

TEST_4.4: Trajectory construction [BLOCKING]
  - For each patient, assert timepoints are temporally ordered
  - Assert Δt > 0 for all consecutive pairs
  - Assert trajectory length matches patient metadata
  Recovery: Check date parsing and sorting

TEST_4.5: Distribution assessment [DIAGNOSTIC]
  - Compute MMD between BraTS-MEN and Andalusian latents
  - Assert computation completes without error
  - Report result (no pass/fail — this is diagnostic)

TEST_4.6: Scanner consistency verification [DIAGNOSTIC]
  - For each patient, check scanner IDs across all timepoints
  - Report fraction of patients with consistent scanner
  - If any patient has scanner changes, flag for LongComBat
  Note: DIAGNOSTIC — does not block pipeline, but affects ComBat choice
```
