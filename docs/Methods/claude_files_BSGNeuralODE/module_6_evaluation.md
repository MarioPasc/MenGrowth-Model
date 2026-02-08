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
- Module 5: `ode_model.pt`, `ode_trajectories.pt`, `gompertz_params.json`, `risk_stratification.json`

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
    "ode_predictions": torch.Tensor,     # ode_trajectories.pt
    "gompertz_params": dict,             # gompertz_params.json
    "risk_scores": List[dict],           # risk_stratification.json
}
```

## Output
- `final_report.json` — All quality targets, pass/fail, ablation results
- 11 publication-quality figures (see Figure List below)
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

### Phase 4 (Neural ODE)

| Metric | Target | Minimum |
|--------|--------|---------|
| Volume prediction R² (LOPO-CV) | ≥ 0.70 | ≥ 0.50 |
| Trajectory MSE | Monotonically decreasing | — |
| Gompertz α > 0 for all patients | 100% | ≥ 90% |

## Ablation Study Matrix

| Experiment | Variable | Conditions | Primary Metric |
|------------|----------|------------|----------------|
| A1: LoRA rank | r | {2, 4, 8, 16, 32} | Dice, probe R² |
| A2: LoRA vs DoRA | Adapter type | {LoRA, DoRA} at r=8 | Dice, probe R² |
| A3: Aux semantic heads | Phase 1 aux | {with, without} | Phase 2 R² |
| A4: SDP dimension | d | {64, 128, 256} | R², dCor |
| A5: VICReg + dCor | Regularization | {full, no cov, no dCor, no both} | Cross-partition corr |
| A6: Gompertz prior | ODE architecture | {Gompertz+MLP, MLP only, Gompertz only} | Trajectory MSE, Vol R² |
| A7: ComBat | Harmonization | {with, without} | Phase 4 trajectory MSE |
| A8: Residual dynamics | ODE residual | {frozen (default), learned with η=0.001} | Trajectory MSE, overfitting |

## Figure List (11 publication-quality figures)

1. **Pipeline overview diagram** — Full 4-phase architecture
2. **Domain gap UMAP** — GLI vs MEN features, frozen encoder (from Module 1)
3. **LoRA ablation** — Dice and probe R² vs. rank (from Module 2)
4. **Phase 2 training curves** — Individual loss terms over epochs (from Module 3)
5. **Disentanglement matrix** — Cross-partition correlation heatmap (from Module 3)
6. **Latent UMAP colored by semantics** — Volume, location, shape (from Module 3)
7. **Cohort distribution comparison** — BraTS-MEN vs Andalusian UMAP (from Module 4)
8. **Patient trajectories** — 2D latent space with temporal arrows (from Module 4/5)
9. **Volume prediction** — Predicted vs actual volume change scatter plot (from Module 5)
10. **Gompertz parameter distribution** — Histogram of growth rates (from Module 5)
11. **Risk stratification** — Ranked patients by growth rate (from Module 5)

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
           """Run all A1-A8 experiments."""
   ```

3. **`FigureGenerator`** — Creates all 11 figures.
   ```python
   class FigureGenerator:
       def generate_all(self, data: dict, output_dir: str) -> List[str]:
           """Generate all 11 figures, return list of file paths."""
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
    gompertz_positive_pct_min: 0.90

ablation:
  experiments: [A1, A2, A3, A4, A5, A6, A7, A8]
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
    "phase4": {"vol_pred_r2": 0.65, "gompertz_positive_pct": 1.0, "pass": True},
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
  - Verify all 8 ablation experiments have results
  - Check that each experiment has all specified conditions
  - Report any missing conditions
  Note: Some ablations may be skipped due to compute constraints

TEST_6.3: Figure generation [DIAGNOSTIC]
  - Generate all 11 figures
  - Verify each figure file exists and has size > 0
  - Visual inspection is manual (not automated)
  Note: Figure quality is subjective; automated check only verifies generation

TEST_6.4: Report completeness [DIAGNOSTIC]
  - Verify final_report.json contains all required fields
  - Verify ablation_results.json is valid JSON
  - Check for NaN or null values in metrics
```
