# SPEC: Plotting Module Refactoring and Missing Figures

**Date:** 2026-04-04  
**Target Agent:** Claude Code Opus 4.6  
**Input:** The monolithic `plot_ensemble_results.py` (attached alongside this document)  
**Location:** `experiments/uncertainty_segmentation/plotting/`

---

## 0. Objective

Refactor the monolithic `plot_ensemble_results.py` into a modular plotting subpackage, add a YAML-driven configuration, and implement the 6 missing figures described in Section 4. The existing 8 figures are correct and must not change visually — this is a structural refactoring plus additions.

---

## 1. Target directory structure

```
experiments/uncertainty_segmentation/plotting/
├── __init__.py                          # Package init, public API
├── config.yaml                          # Plot settings (colors, sizes, output format)
├── style.py                             # _setup_style(), color constants, bracket/annotation helpers
├── data_loader.py                       # Loads all CSVs/JSONs from a run_dir into a dataclass
├── orchestrator.py                      # CLI: loads config + data, calls each figure module
│
└── figures/
    ├── __init__.py                      # Registry of all figure functions
    │
    │   # --- EXISTING (refactored from plot_ensemble_results.py) ---
    ├── fig_training_curves.py           # Fig 1: Loss + validation Dice over epochs
    ├── fig_performance_comparison.py    # Fig 2: Box plot baseline vs members vs ensemble
    ├── fig_paired_comparison.py         # Fig 3: Scatter + ΔDice histogram
    ├── fig_forest_plot.py              # Fig 4: Per-member CIs, ensemble/baseline refs
    ├── fig_convergence.py               # Fig 5: Running mean ± SE, 1/√k overlay
    ├── fig_calibration.py               # Fig 6: Reliability diagram
    ├── fig_best_worst.py                # Fig 7: Top-N best and worst Δ cases
    ├── fig_dice_compartments.py         # Fig 8: Grouped bars TC/WT/ET with brackets
    │
    │   # --- NEW (described in Section 4) ---
    ├── fig_inter_member_agreement.py    # Fig 9: Pairwise Dice heatmap + dendrogram
    ├── fig_volume_bland_altman.py       # Fig 10: Bland-Altman (ensemble vol vs GT vol)
    ├── fig_volume_trajectories.py       # Fig 11: Per-patient longitudinal vol ± uncertainty
    ├── fig_volume_uncertainty.py        # Fig 12: σ_v vs V scatter (heteroscedasticity check)
    ├── fig_boundary_disagreement.py     # Fig 13: Axial slice with per-member contours (NIfTI)
    └── fig_uncertainty_overlay.py       # Fig 14: MI/entropy heatmap on MRI slice (NIfTI)
```

---

## 2. Refactoring rules

### 2.1 `style.py` — Shared constants and helpers

Extract from `plot_ensemble_results.py`:
- `_setup_style()` → `setup_style()` (drop leading underscore, it is now public)
- Color constants: `C_BASELINE`, `C_ENSEMBLE`, `C_MEMBERS`, `C_BEST`, `C_DELTA_POS`, `C_DELTA_NEG`, `C_FILL`
- `_significance_label(p)` → `significance_label(p)`
- `_add_bracket(ax, x1, x2, y, h, text, fontsize)` → `add_stat_bracket(...)`

All figure modules import from `style.py`. No figure module defines its own colors or annotation helpers.

### 2.2 `config.yaml` — Plot configuration

```yaml
# experiments/uncertainty_segmentation/plotting/config.yaml

style:
  font_family: "serif"
  font_serif: ["CMU Serif", "DejaVu Serif", "Times New Roman"]
  font_size: 9
  axes_title_size: 10
  tick_size: 8
  legend_size: 8
  figure_dpi: 150        # Screen display
  save_dpi: 300          # Saved files
  save_format: "pdf"     # pdf | png | svg
  transparent: false
  pdf_fonttype: 42       # TrueType (editable in Illustrator)

colors:
  baseline: "#999999"
  ensemble: "#0072B2"
  members: "#E69F00"
  best_member: "#009E73"
  delta_positive: "#0072B2"
  delta_negative: "#D55E00"

figures:
  training_curves:
    enabled: true
    figsize: [7, 2.8]
  performance_comparison:
    enabled: true
    figsize: [4.5, 3.5]
    show_individual_points: true
    point_size: 6
    point_alpha: 0.3
  paired_comparison:
    enabled: true
    figsize: [7, 3.2]
    n_bins_histogram: 30
  forest_plot:
    enabled: true
    figsize: [4.5, 3.5]
  convergence:
    enabled: true
    figsize: [9, 2.8]
    show_theoretical: true    # 1/√k overlay
  calibration:
    enabled: true
    figsize: [3.5, 3.5]
    min_bin_count: 50         # Suppress bins with fewer samples
  best_worst:
    enabled: true
    figsize: [5, 4.5]
    n_cases: 5
  dice_compartments:
    enabled: true
    figsize: [4.5, 3]
  inter_member_agreement:
    enabled: true
    figsize: [5, 4.5]
  volume_bland_altman:
    enabled: true
    figsize: [4.5, 4]
  volume_trajectories:
    enabled: true
    figsize: [7, 5]
    n_patients: 6             # Number of patients to show (2×3 grid)
    patient_selection: "diverse"  # "diverse" | "first" | explicit list
  volume_uncertainty:
    enabled: true
    figsize: [4.5, 3.5]
  boundary_disagreement:
    enabled: true
    figsize: [8, 4]
    scan_id: null             # null = auto-select from sample scans
    slice_axis: "axial"       # axial | coronal | sagittal
  uncertainty_overlay:
    enabled: true
    figsize: [7, 3]
    scan_id: null
    metric: "mutual_information"  # mutual_information | entropy
```

### 2.3 `data_loader.py` — Centralised data loading

Create a dataclass that holds all loaded data, so figure modules receive a single object instead of N separate DataFrames:

```python
@dataclasses.dataclass
class EnsembleResultsData:
    """All data from a single experiment run."""
    run_dir: Path
    
    # Training
    training_curves: pd.DataFrame          # aggregated_training_curves.csv
    
    # Evaluation
    per_member_dice: pd.DataFrame          # per_member_test_dice.csv
    ensemble_dice: pd.DataFrame            # ensemble_test_dice.csv
    baseline_dice: pd.DataFrame            # baseline_test_dice.csv
    paired_differences: pd.DataFrame       # paired_differences.csv
    convergence_wt: pd.DataFrame           # convergence_dice_wt.csv
    convergence_tc: pd.DataFrame           # convergence_dice_tc.csv
    convergence_et: pd.DataFrame           # convergence_dice_et.csv
    statistical_summary: dict              # statistical_summary.json
    calibration: dict                      # calibration.json
    
    # Volumes (may be None if inference hasn't run)
    mengrowth_volumes: pd.DataFrame | None  # mengrowth_ensemble_volumes.csv
    
    # Predictions directory (may be None)
    predictions_dir: Path | None           # predictions/
    
    @property
    def n_members(self) -> int:
        return self.per_member_dice["member_id"].nunique()
    
    @property
    def n_test_scans(self) -> int:
        return len(self.ensemble_dice)
    
    @property
    def has_volumes(self) -> bool:
        return self.mengrowth_volumes is not None
    
    @property
    def has_predictions(self) -> bool:
        return self.predictions_dir is not None and self.predictions_dir.exists()


def load_results(run_dir: Path) -> EnsembleResultsData:
    """Load all results from a run directory.
    
    Gracefully handles missing files (sets to None).
    """
```

The loader must:
- Accept a `run_dir` (e.g., `results/uncertainty_segmentation/r8_M10_s42/`)
- Look for `evaluation/` and `volumes/` subdirectories
- Return `None` for files that do not exist (volumes CSV, predictions dir)
- Log which files were found vs missing

### 2.4 Each figure module — Standard interface

Every figure module in `figures/` must follow this contract:

```python
"""Fig N: <Title>.

<One-line description of what the figure shows.>
"""

def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the figure.
    
    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml (the dict under figures.<name>).
        ax: Optional pre-created axes. If None, create a new figure.
    
    Returns:
        The Figure object (for saving by the orchestrator).
    """
```

If a figure requires data that might be absent (e.g., `mengrowth_volumes`), it must check `data.has_volumes` and raise a clear `ValueError` or return `None` with a log message. The orchestrator handles the `None` case.

### 2.5 `orchestrator.py` — CLI entry point

```python
"""Orchestrator for the plotting suite.

Usage:
    python -m experiments.uncertainty_segmentation.plotting.orchestrator \
        /path/to/r8_M10_s42/ \
        --config experiments/uncertainty_segmentation/plotting/config.yaml \
        --output ./figures/ \
        --format pdf \
        --only fig_training_curves fig_paired_comparison   # optional: subset
"""
```

The orchestrator:
1. Loads the plot config YAML (merges with CLI overrides).
2. Calls `load_results(run_dir)` to get the `EnsembleResultsData`.
3. Calls `setup_style(config["style"])`.
4. Iterates over all figure modules in the registry.
5. For each enabled figure: calls `plot(data, config)`, saves the returned Figure, closes it.
6. Respects `--only` to generate a subset of figures.

The figure registry should be a dict mapping names to modules, discovered either via explicit import or a list:

```python
FIGURE_REGISTRY = {
    "fig_training_curves": fig_training_curves,
    "fig_performance_comparison": fig_performance_comparison,
    # ...
}
```

---

## 3. Existing figures — Refactoring notes

These 8 figures are correct. The refactoring is purely structural: extract each `plot_*` function into its own module, adapt the signature to `(data, config, ax=None)`, and replace hardcoded values with config lookups.

| Figure | Source function | Target module | Notes |
|--------|---------------|---------------|-------|
| Fig 1 | `plot_training_curves()` | `fig_training_curves.py` | Two-panel: takes `ax_loss, ax_dice` → create internally from config `figsize` |
| Fig 2 | `plot_performance_comparison()` | `fig_performance_comparison.py` | Uses `data.per_member_dice`, `data.ensemble_dice`, `data.baseline_dice`, `data.statistical_summary` |
| Fig 3 | `plot_paired_comparison()` | `fig_paired_comparison.py` | Two-panel: scatter + histogram |
| Fig 4 | `plot_forest()` | `fig_forest_plot.py` | Uses only `data.statistical_summary` |
| Fig 5 | `plot_convergence()` | `fig_convergence.py` | Three-panel: WT, TC, ET |
| Fig 6 | `plot_reliability()` | `fig_calibration.py` | Uses `data.calibration` |
| Fig 7 | `plot_best_worst()` | `fig_best_worst.py` | `n_cases` from config |
| Fig 8 | `plot_dice_by_compartment()` | `fig_dice_compartments.py` | Uses `data.statistical_summary` |

---

## 4. Missing figures — Full specifications

### 4.1 Fig 9: Inter-Member Agreement Heatmap

**File:** `fig_inter_member_agreement.py`

**Purpose:** Visualise pairwise similarity between ensemble members. Answers: "Are all members learning similar solutions, or are some specialising?"

**Data source:** `data.per_member_dice` — pivot to `[n_scans × M]` matrix on `dice_wt`.

**Implementation:**

1. Pivot `per_member_dice` to `[scan_id × member_id]` on `dice_wt`.
2. Compute the `M × M` pairwise Pearson correlation matrix across subjects.
3. Plot as a `seaborn.heatmap` or `matplotlib.imshow` with:
   - Colormap: diverging (e.g., `RdYlBu_r`), vmin=0.5, vmax=1.0
   - Annotations: correlation values in each cell (2 decimal places)
   - Diagonal: 1.0 (by definition)
4. Add a title annotation: `"Mean pairwise r = {value:.3f}"` and `"ICC(3,1) = {value:.3f}"` from `data.statistical_summary["inter_member_agreement"]`.

**Axes:** Single square axes. Member IDs on both axes. Symmetric matrix — show upper triangle only or full.

**Expected outcome:** All off-diagonal cells should be > 0.8 (high agreement). Member 5 may show slightly lower correlations (it had the lowest WT Dice in the forest plot).

---

### 4.2 Fig 10: Volume Bland-Altman (Ensemble vs Ground Truth)

**File:** `fig_volume_bland_altman.py`

**Purpose:** Assess systematic bias and proportional error in the ensemble's volume estimates compared to ground truth segmentations on the BraTS-MEN test set.

**Data source:** `data.ensemble_dice` — columns `volume_ensemble` and `volume_gt`.

**Implementation:**

This is a standard Bland-Altman plot:

1. Compute:
   - $\bar{V}_i = (V_{\text{ensemble},i} + V_{\text{GT},i}) / 2$ (mean of the two measurements)
   - $\Delta_i = V_{\text{ensemble},i} - V_{\text{GT},i}$ (difference)
2. Compute population statistics:
   - $\bar{\Delta} = \text{mean}(\Delta_i)$ (bias)
   - $s_\Delta = \text{std}(\Delta_i)$ (SD of differences)
   - Limits of agreement: $\bar{\Delta} \pm 1.96 \, s_\Delta$
3. Plot scatter: x = $\bar{V}_i$, y = $\Delta_i$.
4. Horizontal lines:
   - Solid at $y = \bar{\Delta}$ (bias), labelled with value
   - Dashed at $y = \bar{\Delta} \pm 1.96 \, s_\Delta$ (limits of agreement), labelled
   - Dashed at $y = 0$ (reference)
5. X-axis in **log scale** (volume ranges from 129 to 267K mm³).
6. Color points by sign of $\Delta$: blue if ensemble overestimates, red if underestimates.

**Axes:** Single axes. X = "Mean volume (mm³)", Y = "Difference: Ensemble − GT (mm³)".

**Annotation:** Report the bias and 95% LoA in a text box.

**Expected outcome:** The bias should be near zero if the ensemble is well-calibrated. Proportional error (larger scatter for larger volumes) is expected and justifies the log-volume transform used in the growth models.

---

### 4.3 Fig 11: Volume Trajectories with Uncertainty

**File:** `fig_volume_trajectories.py`

**Purpose:** Show the primary clinical output — per-patient longitudinal tumor volume trajectories with uncertainty bands from the ensemble. This is the figure that connects segmentation uncertainty to growth prediction.

**Data source:** `data.mengrowth_volumes` — columns: `patient_id`, `timepoint_idx`, `vol_mean`, `vol_std`, `vol_median`, `vol_mad`, `logvol_mean`, `logvol_std`, `vol_m0`..`vol_m{M-1}`.

**Guard:** If `data.mengrowth_volumes is None`, log a warning and return `None`.

**Implementation:**

1. **Patient selection:** Select `n_patients` patients (from config, default 6) to display in a 2×3 grid. Strategy `"diverse"`:
   - Compute per-patient mean `vol_std` (average uncertainty across timepoints).
   - Sort patients by this value.
   - Select patients at evenly-spaced percentiles (e.g., 10th, 30th, 50th, 70th, 90th, 95th) to show the range from low-uncertainty to high-uncertainty cases.

2. **Per-patient subplot:**
   - X-axis: `timepoint_idx` (integer, starting at 0).
   - Y-axis: volume in mm³ (or log-volume — use raw volume with log y-scale for clinical readability).
   - Plot:
     - Mean volume as solid blue line with circle markers.
     - Mean ± std as blue shaded ribbon.
     - Median as dashed teal line.
     - Median ± 1.4826·MAD as lighter teal ribbon.
     - Per-member volumes as faint grey dots (jittered slightly on x for visibility).
   - Title: `"Patient {id}"` with mean uncertainty annotation.

3. **Shared y-axis label:** "Tumor volume (mm³)". Individual y-scales per subplot (tumors vary hugely in size).

**Expected outcome:** Patients with high boundary uncertainty show wide ribbons; patients with well-defined tumors show narrow ribbons. The median/MAD ribbon should be equal to or narrower than mean/std, demonstrating robustness.

---

### 4.4 Fig 12: Volume Uncertainty vs Volume Size

**File:** `fig_volume_uncertainty.py`

**Purpose:** Check whether segmentation uncertainty scales with tumor size (heteroscedasticity). This directly justifies the heteroscedastic GP likelihood: $\sigma^2_{v,k}$ should increase with $V$, otherwise a homoscedastic model suffices.

**Data source:** `data.mengrowth_volumes` — columns: `vol_mean`, `vol_std`, `logvol_mean`, `logvol_std`, `logvol_mad_scaled`, `wt_mean_entropy`, `wt_boundary_entropy`.

**Guard:** If `data.mengrowth_volumes is None`, return `None`.

**Implementation:**

Two-panel figure:

**Panel A: Raw volume space.**
- X = `vol_mean` (mm³), Y = `vol_std` (mm³).
- Log-log axes (both volume and its uncertainty span orders of magnitude).
- Color by `wt_mean_entropy` (brighter = more uncertain).
- Fit and overlay: OLS regression line on log-log → slope gives the power law exponent. If slope ≈ 1, uncertainty is proportional to volume.
- Annotate: Pearson r and the regression equation.

**Panel B: Log-volume space.**
- X = `logvol_mean`, Y = `logvol_std`.
- Linear axes.
- Overlay `logvol_mad_scaled` as a second scatter (different marker) to compare mean/std vs median/MAD.
- If this scatter is approximately flat, then log-transforming stabilises the variance (supporting the log-volume GP formulation).

**Expected outcome:** Panel A shows heteroscedasticity (uncertainty grows with tumor size). Panel B shows that log-transforming partially stabilises it. This is the empirical justification for using $v = \log(V + 1)$ in the GP.

---

### 4.5 Fig 13: Boundary Disagreement (NIfTI-based)

**File:** `fig_boundary_disagreement.py`

**Purpose:** Qualitative illustration of where ensemble members disagree. The per-member contours should overlap in the tumor interior and diverge at the boundary, visually demonstrating what the ensemble captures.

**Data source:** NIfTI files from `data.predictions_dir`. Requires a sample scan that has full per-member predictions (e.g., `MenGrowth-0001-000`).

**Guard:** If `data.has_predictions` is False, return `None`. Also check that the selected scan has `member_{m}_mask.nii.gz` files.

**Dependencies:** `nibabel` for NIfTI loading. The MRI input volume is needed as background — load from the MenGrowth HDF5 file (T1ce channel = index 1).

**Implementation:**

1. **Scan selection:** Use the scan_id from config, or auto-select the first scan in `predictions/` that has `member_0_mask.nii.gz`.

2. **Slice selection:** Find the axial slice with the largest tumor cross-section in the ensemble mask (slice with maximum number of foreground voxels).

3. **Load data:**
   - Background: T1ce slice from MenGrowth H5 (or `ensemble_probs.nii.gz` channel 1 as proxy).
   - Per-member masks: `member_{m}_mask.nii.gz` for m = 0..M-1.
   - Ensemble mask: `ensemble_mask.nii.gz`.

4. **Plot (three panels):**
   - **Panel A: MRI with ensemble contour.** Background = T1ce (grayscale). Overlay = ensemble mask contour (blue). This is the "clinical view."
   - **Panel B: MRI with all member contours.** Background = T1ce. Overlay = M contours, each in a distinct color from a qualitative colormap (e.g., `tab10`). Where contours overlap = agreement. Where they spread = boundary uncertainty.
   - **Panel C: Agreement map.** For each voxel, count how many of the M members classified it as tumor. Plot as a heatmap (0 = no member, M = all members). The tumor core should be M; the boundary shows a gradient from M to 0. Use a sequential colormap (e.g., `YlOrRd`).

5. **Annotations:** Panel titles: "a) Ensemble prediction", "b) Per-member contours", "c) Agreement map (0–M)".

**Technical note on contours:** Use `matplotlib.pyplot.contour` with `levels=[0.5]` on each 2D binary mask slice to extract contours. This gives a clean single-pixel-wide outline.

---

### 4.6 Fig 14: Uncertainty Heatmap Overlay

**File:** `fig_uncertainty_overlay.py`

**Purpose:** Show the spatial distribution of epistemic uncertainty (mutual information) overlaid on the MRI. High MI at the tumor boundary demonstrates that the ensemble "knows what it doesn't know."

**Data source:** NIfTI files — `mutual_information.nii.gz` (or `entropy.nii.gz`) and the MRI background.

**Guard:** Same as Fig 13.

**Implementation:**

1. **Load:** MI map from `mutual_information.nii.gz` (4D: [D,H,W,C] in NIfTI convention, channel 1 = WT). Extract the WT channel. Background MRI from H5 or `ensemble_probs`.

2. **Slice:** Same axial slice as Fig 13 (or auto-select by max MI).

3. **Plot (two panels):**
   - **Panel A: Entropy map.** Background = T1ce (grayscale). Overlay = predictive entropy of WT channel as a semi-transparent heatmap (`hot` or `inferno` colormap, alpha=0.6). Colorbar labelled "Predictive entropy (nats)".
   - **Panel B: Mutual information map.** Same layout but with MI. MI should be zero everywhere except the tumor boundary (epistemic uncertainty is localised).

4. **Overlay the ensemble contour** as a thin white dashed line on both panels for spatial reference.

5. **Annotations:** Report mean MI inside the tumor core, at the boundary, and outside. These numbers are already in the volume CSV (`wt_mean_entropy`, `wt_boundary_entropy`) but here we show the spatial distribution.

---

## 5. Data availability summary

| Figure | Required data | File | Status |
|--------|--------------|------|--------|
| Fig 1–8 | Evaluation CSVs + JSONs | `evaluation/*.csv`, `evaluation/*.json` | ✅ Available |
| Fig 9 | Per-member Dice pivot | `evaluation/per_member_test_dice.csv` | ✅ Available |
| Fig 10 | Ensemble + GT volumes | `evaluation/ensemble_test_dice.csv` | ✅ Available |
| Fig 11 | MenGrowth volumes | `volumes/mengrowth_ensemble_volumes.csv` | ✅ Exists in run (not uploaded here) |
| Fig 12 | MenGrowth volumes | `volumes/mengrowth_ensemble_volumes.csv` | ✅ Same |
| Fig 13 | Per-member NIfTI masks | `predictions/MenGrowth-*/member_*_mask.nii.gz` | ✅ Exists for sample scans |
| Fig 14 | MI/entropy NIfTI | `predictions/MenGrowth-*/mutual_information.nii.gz` | ✅ Exists for sample scans |

For Figures 13–14, the agent also needs the MRI background. Two options:
- Load from MenGrowth HDF5: `data.config.paths.mengrowth_h5_file` → images array, select by scan index, extract T1ce (channel 1).
- Use `ensemble_probs.nii.gz` as a proxy (it is a probability map, not MRI, but shows tumor location). This avoids an HDF5 dependency.

The agent should prefer the HDF5 approach if the file is accessible; fall back to the probs map otherwise.

---

## 6. Implementation order

```
1. Create directory structure + __init__.py files
2. style.py (extract from monolith)
3. config.yaml (copy from Section 2.2)
4. data_loader.py (EnsembleResultsData + load_results)
5. Refactor existing 8 figures (one module each, verify visual parity)
6. orchestrator.py (CLI, registry, figure loop)
7. Verify: python -m experiments.uncertainty_segmentation.plotting.orchestrator /path/to/run --format png
   → all 8 original figures generated, visually identical to monolith output
8. fig_inter_member_agreement.py (Fig 9) — uses only existing CSVs
9. fig_volume_bland_altman.py (Fig 10) — uses only existing CSVs
10. fig_volume_trajectories.py (Fig 11) — requires mengrowth_volumes.csv
11. fig_volume_uncertainty.py (Fig 12) — requires mengrowth_volumes.csv
12. fig_boundary_disagreement.py (Fig 13) — requires NIfTI + nibabel
13. fig_uncertainty_overlay.py (Fig 14) — requires NIfTI + nibabel
```

Steps 8–9 can be implemented and tested immediately. Steps 10–12 require the mengrowth_volumes CSV (exists in the run dir, just wasn't uploaded to this conversation). Steps 12–13 require nibabel and access to the predictions directory.

---

## 7. Testing

The agent should verify each figure by:
1. Running the orchestrator on the actual `r8_M10_s42/` run directory.
2. Checking that no figure raises an exception.
3. Checking that no figure produces an empty or all-white output.
4. Checking that figures requiring missing data gracefully skip with a warning log.

For a quick smoke test without the full run directory, the agent can create a minimal test with synthetic data in `tests/growth/test_plotting.py`:

```python
def test_style_setup():
    """setup_style() doesn't crash."""
    
def test_data_loader_missing_files():
    """load_results() on empty dir returns None for optional fields."""
    
def test_bland_altman_zero_bias():
    """Bland-Altman with identical predictions has bias = 0."""
```

---

## 8. Coding standards

- All plotting code uses `matplotlib`; `seaborn` only for heatmaps where it simplifies the code.
- No `plt.show()` anywhere — figures are returned to the orchestrator which saves and closes them.
- Every module has a module-level docstring stating what the figure shows.
- Type hints on all functions.
- Config values are accessed via `config.get("key", default)` — never crash on missing config keys.
- NIfTI loading is wrapped in `try/except ImportError` for `nibabel` — if not installed, the NIfTI figures gracefully skip.
