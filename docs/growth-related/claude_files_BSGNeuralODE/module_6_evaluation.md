# Module 6: End-to-End Evaluation

## Overview
Consolidate all pipeline outputs, run ablation experiments, produce publication-quality figures, and generate the final evaluation report.

## Input
All prior module outputs:
- Module 0: `data_splits.json`, `semantic_features_cache/`
- Module 1: `domain_gap_report.json`, UMAP plots
- Module 2: `phase1_encoder_merged.pt`, Dice metrics
- Module 3: `phase2_sdp.pt`, `phase2_quality_report.json`, latent UMAP
- Module 4: `latent_bratsmen.pt`, `latent_andalusian_harmonized.pt`, `trajectories.json`, `combat_assessment.json`
- Module 5: `lme_results.json`, `hgp_results.json`, `pamogp_results.json`, `model_comparison.json`, `growth_figures/`

## Input Contract
```python
# All module outputs loaded as Python objects
module_outputs: dict = {
    "splits": dict,                      # data_splits.json
    "domain_gap": dict,                  # domain_gap_report.json
    "dice_metrics": dict,                # phase1_best_dice.json
    "sdp_quality": dict,                 # phase2_quality_report.json
    "combat_assessment": dict,           # combat_assessment.json
    "trajectories": List[dict],          # trajectories.json
    "lme_results": dict,                 # lme_results.json
    "hgp_results": dict,                 # hgp_results.json
    "pamogp_results": dict,              # pamogp_results.json
    "model_comparison": dict,            # model_comparison.json
}
```

## Output
- `final_report.json` — All quality targets, pass/fail, ablation results
- 13 publication-quality figures (see Figure List below)
- `ablation_results.json` — Full ablation matrix results

## Quality Targets

### Phase 1 (Encoder Adaptation)

| Metric | Target | Minimum |
|--------|--------|---------|
| Dice (WT) on `lora_val` | ≥ 0.85 | ≥ 0.80 |
| Dice improvement over frozen BSF | ≥ 0.05 | ≥ 0.02 |
| Linear probe Vol R² (adapted) | ≥ 0.50 | ≥ 0.30 |

### Phase 2 (SDP)

| Metric | Target | Minimum |
|--------|--------|---------|
| Vol R² on `val` | ≥ 0.90 | ≥ 0.80 |
| Loc R² on `val` | ≥ 0.95 | ≥ 0.85 |
| Shape R² on `val` | ≥ 0.40 | ≥ 0.25 |
| Max cross-partition correlation | < 0.20 | < 0.30 |
| Per-dimension variance > 0.5 | ≥ 95% | ≥ 85% |
| dCor(vol, loc) | < 0.10 | < 0.20 |

### Phase 4 (Growth Prediction)

| Metric | Target | Minimum |
|--------|--------|---------|
| Volume prediction R² (LOPO) — best model | ≥ 0.70 | ≥ 0.50 |
| Calibration (95% CI coverage) — GP models | 0.90–0.98 | ≥ 0.80 |
| Per-patient trajectory r (patients n_i ≥ 3) | ≥ 0.80 | ≥ 0.60 |
| PA-MOGP R² > H-GP R² (coupling improvement) | > 0 | — |

## Ablation Study Matrix

| Experiment | Variable | Conditions | Primary Metric |
|------------|----------|------------|----------------|
| A1: LoRA rank | r | {2, 4, 8, 16, 32} | Dice, probe R² |
| A2: LoRA vs DoRA | Adapter type | {LoRA, DoRA} at r=8 | Dice, probe R² |
| A3: Aux semantic heads | Phase 1 aux | {with, without} | Phase 2 R² |
| A4: SDP dimension | d | {64, 128, 256} | R², dCor |
| A5: VICReg + dCor | Regularization | {full, no cov, no dCor, no both} | Cross-partition corr |
| A6: Growth model comparison | Model | {LME, H-GP, PA-MOGP} | Vol R² (LOPO) |
| A7: ComBat effect on prediction | Harmonization | {with, without} | Vol R² (LOPO) |
| A8: GP mean function | Mean function | {zero, linear (from LME), Gompertz fit} | Vol R² (LOPO) |
| A9: GP kernel selection | Kernel (H-GP) | {Matérn-3/2, Matérn-5/2, SE} | Vol R² (LOPO) |
| A10: Cross-partition coupling | PA-MOGP structure | {with coupling, without coupling} | Vol R² (LOPO), calibration |

## Figure List (13 publication-quality figures)

1. **Pipeline overview diagram** — Full 4-phase architecture
2. **Domain gap UMAP** — GLI vs MEN features, frozen encoder (from Module 1)
3. **LoRA ablation** — Dice and probe R² vs. rank (from Module 2)
4. **Phase 2 training curves** — Individual loss terms over epochs (from Module 3)
5. **Disentanglement matrix** — Cross-partition correlation heatmap (from Module 3)
6. **Latent UMAP colored by semantics** — Volume, location, shape (from Module 3)
7. **Cohort distribution comparison** — BraTS-MEN vs Andalusian UMAP (from Module 4)
8. **Patient trajectories in latent space** — 2D UMAP with temporal arrows for 5–10 patients with n_i ≥ 3 (from Module 4)
9. **Volume prediction scatter** — Predicted vs actual ΔV for all LOPO-CV test pairs, colored by model (LME/H-GP/PA-MOGP) (from Module 5)
10. **Trajectory prediction with uncertainty** — 3–4 example patients showing predicted trajectory (mean ± 95% CI) overlaid on actual observations, for all three models (from Module 5)
11. **Model comparison bar chart** — R², MAE, calibration for each model (from Module 5)
12. **Learned kernel hyperparameters** — Length-scales ℓ per partition (PA-MOGP), revealing characteristic timescales of volume/location/shape dynamics (from Module 5)
13. **Cross-partition coupling weights** — Heatmap of ww^T showing which volume dimensions most strongly drive location/shape changes (from Module 5)

## Figure Specifications
```python
# Common settings for all figures
figure_config = {
    "dpi": 300,
    "format": "pdf",  # and "png" for quick viewing
    "font_size": 12,
    "font_family": "serif",
    "figsize_single": (6, 4),      # single panel
    "figsize_double": (12, 4),     # two panels
    "figsize_matrix": (8, 8),      # correlation matrix
    "colormap": "viridis",
    "scatter_alpha": 0.6,
    "line_width": 1.5,
}
```

## Code Requirements

1. **`EvaluationPipeline`** — Loads all module outputs and computes consolidated metrics.
   ```python
   class EvaluationPipeline:
       def load_all_outputs(self, output_dir: str) -> dict:
           """Load all module outputs."""
       def compute_quality_targets(self) -> dict:
           """Evaluate all metrics against targets/minimums."""
       def generate_report(self) -> dict:
           """Generate final_report.json."""
   ```

2. **`AblationRunner`** — Runs ablation experiments.
   ```python
   class AblationRunner:
       def run_ablation(self, experiment: str, conditions: List) -> dict:
           """Run one ablation experiment across conditions."""
       def run_all(self) -> dict:
           """Run all A1-A10 experiments."""
   ```

3. **`FigureGenerator`** — Creates all 13 figures.
   ```python
   class FigureGenerator:
       def generate_all(self, data: dict, output_dir: str) -> List[str]:
           """Generate all 13 figures, return list of file paths."""
   ```

## Configuration Snippet
```yaml
# configs/evaluation.yaml
evaluation:
  output_dir: ${paths.output_root}/evaluation
  figure_format: [pdf, png]
  figure_dpi: 300

quality_targets:
  phase1:
    dice_wt_min: 0.80
    dice_wt_target: 0.85
    dice_improvement_min: 0.02
  phase2:
    vol_r2_min: 0.80
    loc_r2_min: 0.85
    shape_r2_min: 0.25
    max_cross_corr: 0.30
  phase4:
    vol_prediction_r2_min: 0.50
    calibration_95_min: 0.80
    per_patient_r_min: 0.60

ablation:
  experiments: [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]
  a1_ranks: [2, 4, 8, 16, 32]
  a4_dims: [64, 128, 256]
```

## Smoke Test
```python
import json

# Verify report structure
report = {
    "phase1": {"dice_wt": 0.87, "dice_improvement": 0.06, "pass": True},
    "phase2": {"vol_r2": 0.91, "loc_r2": 0.96, "shape_r2": 0.42, "pass": True},
    "phase4": {"vol_pred_r2": 0.65, "calibration_95": 0.93, "best_model": "PA-MOGP", "pass": True},
    "ablations": {"A1": {...}, "A2": {...}, ...},
    "overall_pass": True,
}
# Validate all required fields present
assert all(k in report for k in ["phase1", "phase2", "phase4", "ablations"])
```

## Verification Tests

All evaluation tests are **DIAGNOSTIC** — they log results but do not block.

```
TEST_6.1: Quality targets assessment [DIAGNOSTIC]
  - Load all module outputs
  - Evaluate each metric against target and minimum thresholds
  - Generate pass/fail table
  - Log warnings for any metric below minimum
  Note: DIAGNOSTIC — log but don't block; results are what they are

TEST_6.2: Ablation completeness [DIAGNOSTIC]
  - Verify all 10 ablation experiments have results
  - Check that each experiment has all specified conditions
  - Report any missing conditions
  Note: Some ablations may be skipped due to compute constraints

TEST_6.3: Figure generation [DIAGNOSTIC]
  - Generate all 13 figures
  - Verify each figure file exists and has size > 0
  - Visual inspection is manual (not automated)
  Note: Figure quality is subjective; automated check only verifies generation

TEST_6.4: Report completeness [DIAGNOSTIC]
  - Verify final_report.json contains all required fields
  - Verify ablation_results.json is valid JSON
  - Check for NaN or null values in metrics
```
