# Inter-LoRA Rank Comparison — Plotting Requirements

> **Scope.** Produce a complete quantitative + qualitative report comparing LoRA
> adapters of varying rank for the **BSF (Brain Segmentation Foundation) →
> BraTS-MEN** transfer (GLI-pretrained encoder, frozen decoder, KQV+FC1+FC2
> projection adaptation). The scientific objective is to **identify the
> intrinsic adaptation rank** `r*` at which the GLI→MEN transfer saturates in
> Dice while remaining well calibrated and not bias-dominated.
>
> **Audience.** This file is consumed by a Claude Code agent that has access to
> the project repository, Python ≥ 3.10, the standard scientific stack
> (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `nibabel`,
> `scikit-learn`, `scipy.cluster.hierarchy`), and the project's plotting style
> module (`scienceplots`, fallback to `seaborn-v0_8-paper`).

---

## 0. Background and motivation (must be reproduced in the report intro)

The intrinsic-dimensionality hypothesis for parameter-efficient fine-tuning
states that the effective fine-tuning subspace has dimension much lower than
the full parameter count
(Aghajanyan, Zettlemoyer & Gupta, 2020, *Intrinsic Dimensionality Explains the
Effectiveness of Language Model Fine-Tuning*, arXiv:2012.13255). LoRA
(Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*,
arXiv:2106.09685) operationalises this by parameterising the update as
`ΔW = B A` with `B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}`, so `r ≪ min(d,k)`. For domain
shift in medical image segmentation (BraTS-GLI → BraTS-MEN), `r*` should
manifest as:

1. an **elbow** in `Dice(r)` per label (TC, WT, ET);
2. a transition from **bias-dominated** (`k* ≤ M` for most scans, see your
   `epistemic_taxonomy.json`) to **variance-dominated** regimes;
3. **calibration** improvement that flattens beyond `r*`;
4. **inter-member agreement (ICC)** approaching its asymptote.

The bias-dominance threshold is defined on the log-volume scale as

$$
k^{\star} \;=\; \Big\lceil \big(\sigma_{\log V} \,/\, |\mathrm{bias}_{\log V}|\big)^{2}\,\Big\rceil ,
$$

with `bias_logvol = mean_m log V_m − log V_GT`. Scans for which `k* ≤ M` (the
ensemble size) cannot be improved by drawing more members and therefore signal
*model* limitation, i.e. insufficient adapter capacity.

The Expected Calibration Error and Brier score are

$$
\mathrm{ECE} \;=\; \sum_{b=1}^{B} \frac{|S_b|}{N}\,\big|\mathrm{acc}(S_b) - \mathrm{conf}(S_b)\big|,
\qquad
\mathrm{Brier} \;=\; \tfrac{1}{N}\sum_{i=1}^{N}\big(p_i - y_i\big)^{2},
$$

with `B = 15` reliability bins (matches your `calibration.json`).

ICC(2,1) for inter-member Dice agreement across `M` members:

$$
\mathrm{ICC}(2,1) \;=\;
\frac{\mathrm{MS}_{\mathrm{R}} - \mathrm{MS}_{\mathrm{E}}}
{\mathrm{MS}_{\mathrm{R}} + (k-1)\,\mathrm{MS}_{\mathrm{E}} + \tfrac{k}{n}(\mathrm{MS}_{\mathrm{C}} - \mathrm{MS}_{\mathrm{E}})}.
$$

These quantities are already pre-computed; the agent must **only read**, never
re-compute them, except where explicitly noted.

---

## 1. Inputs — directory layout and discovery

### 1.1 Root and rank discovery

```
ROOT = ${RESULTS_ROOT}/uncertainty_segmentation/frozen_decoder/kqv_proj_fc1_fc2/stages_1234
```

The agent **must discover ranks** by globbing:

```
${ROOT}/r*_M20_s42
```

and parsing the rank from the directory name with the regex `^r(\d+)_M20_s42$`.
Expected ranks (must all be present unless flagged): `{4, 8, 16, 32, 64}`.
Hard-fail if fewer than 3 ranks are found.

The **baseline** (frozen BSF, no LoRA) lives at:

```
BASELINE_DIR = ${ROOT}/baseline_frozen_bsf   # if absent, fallback below
```

If `BASELINE_DIR` is absent, the baseline metrics for plots/tables come from
the `baseline_test_dice.csv` files inside *every* rank directory (they are
identical by construction — assert this and warn if they differ by > 1e-6).

### 1.2 Per-rank artefacts (relative to `${ROOT}/r{R}_M20_s42/`)

| Path | Used by |
| --- | --- |
| `evaluation/ensemble_test_dice.csv` | Quant1, Quant2, Qual2, Tab1, Tab2 |
| `evaluation/per_member_test_dice.csv` | Quant1, Tab1 |
| `evaluation/baseline_test_dice.csv` | All (baseline reference) |
| `evaluation/calibration.json` | Quant2, Tab1 |
| `evaluation/calibration_coverage.csv` | Quant2, Tab1 |
| `evaluation/bias_diagnostics.csv` | Quant2, Tab1 |
| `evaluation/bias_dominance_threshold.csv` | Quant2, Tab1 |
| `evaluation/epistemic_taxonomy.json` | Quant2, Tab1 |
| `evaluation/statistical_summary.json` | Quant1, Tab1, Tab2 |
| `evaluation/paired_differences.csv` | Tab2 |
| `evaluation/aggregated_training_curves.csv` | (optional) sanity check |
| `predictions/{SCAN_ID}/segmentation.nii.gz` | Qual1 (GT, BraTS-MEN only) |
| `predictions/{SCAN_ID}/ensemble_mask.nii.gz` | Qual1 |
| `predictions/{SCAN_ID}/ensemble_probs.nii.gz` | Qual1 (variance computation) |
| `predictions/{SCAN_ID}/entropy.nii.gz` | Qual1 |
| `predictions/{SCAN_ID}/mutual_information.nii.gz` | Qual1 |
| `predictions/{SCAN_ID}/member_{m}_probs.nii.gz` | Qual1 (variance map) |

Where `SCAN_ID` may live under `predictions/brats_men_test/{SCAN_ID}` (for
BraTS-MEN test) or `predictions/{MenGrowthID}/{study}` (clinical, no GT).
Probe both layouts.

### 1.3 Output layout (all artefacts produced by the agent)

```
${OUT_ROOT}/
├── figures/
│   ├── quant1_dice_vs_rank.{pdf,png,svg}
│   ├── quant2_calibration_epistemic_vs_rank.{pdf,png,svg}
│   ├── qual1_slice_grid.{pdf,png}
│   ├── qual2_clustered_heatmap.{pdf,png,svg}
│   └── _components/                # individual sub-panels at high DPI
├── tables/
│   ├── tab1_summary_per_rank.{tex,csv,md}
│   └── tab2_paired_vs_baseline.{tex,csv,md}
├── data/
│   ├── compiled_metrics.parquet    # long-form per-rank metric DataFrame
│   └── selected_slices.json        # subjects + slice indices used in Qual1
└── logs/
    └── plotting.log                 # structured logging output
```

`${OUT_ROOT}` defaults to `${ROOT}/_inter_lora_report/`.

---

## 2. Global style conventions

- **Reproducibility.** Set `numpy` and `random` seeds to 42. Embed the git SHA
  and run timestamp in every figure caption metadata (matplotlib metadata
  dict).
- **Fonts.** Serif for axis labels (`cmr10` if available, else `STIXGeneral`),
  sans-serif for tick labels. Math via `usetex=False`, `mathtext` rcParams.
- **Sizes.** Single-column 88 mm width for Quant figures; double-column 180 mm
  for Quant2 and Qual1; full-page 180×220 mm for Qual2.
- **DPI.** 600 for PNG, vector for PDF/SVG.
- **Rank colourmap.** `viridis` with `Normalize(vmin=log2(min_r),
  vmax=log2(max_r))` so ranks are evenly spaced on log axis. Baseline frozen
  BSF is **slate grey** `#3a3a3a`, dashed.
- **Label colours.**
  - TC = `#d62728` (red)
  - WT = `#2ca02c` (green)
  - ET = `#ff9f1c` (amber)
- **Confidence intervals.** Shaded ribbons at 95 %, alpha=0.20, no edge.
- **Background.**
  - Quantitative figures: white background, axes spines off top/right.
  - Qualitative slice grids: **black background**, white annotations and tick
    labels (`fig.patch.set_facecolor('black')`,
    `ax.set_facecolor('black')`).
- **Statistical annotations.** `*` p < 0.05, `**` p < 0.01, `***` p < 0.001
  (Wilcoxon signed-rank, paired, two-sided). Apply Holm–Bonferroni across
  ranks per label.
- **Caption metadata.** Each figure file embeds a one-line caption draft as
  `metadata={'Title': ..., 'Description': ...}`.

---

## 3. Quantitative figures

### 3.1 Quant 1 — Dice vs. rank (the elbow plot)

**Goal.** Visualise saturation of segmentation quality with adapter rank.

**Layout.** 1 × 3 panels (one per label TC, WT, ET); shared y-axis.

**Per panel.**

- **X-axis.** LoRA rank, log₂ scale, ticks at `[4, 8, 16, 32, 64]`.
- **Y-axis.** Dice ∈ [0.5, 1.0] (auto-tighten lower bound to
  `min(0.5, baseline_mean − 0.05)`).
- **Series 1 (ensemble).** Filled circles at `ensemble_mean` per rank, with
  vertical 95 % CI bars from `ensemble_ci95` in `statistical_summary.json`.
  Connect with a viridis-coloured line (one colour per rank, see § 2).
- **Series 2 (best member).** Hollow squares at `best_member_mean`; light
  grey vertical bars for its CI (compute from `per_member_test_dice.csv`
  using bootstrap, `n_boot=1000`, percentile method, seed=42).
- **Series 3 (per-member spread).** Strip-plot of all 20 member means at
  each rank, alpha=0.35, jittered horizontally (±0.05 in log₂ units).
- **Baseline.** Horizontal dashed line at `baseline_mean` (frozen BSF) with a
  shaded band of width 2·SE; annotate `frozen BSF` at the right edge.
- **Annotation.** Detect the elbow per label using the
  *kneedle* algorithm (Satopää et al., 2011, *Finding a "Kneedle" in a
  Haystack*, doi:10.1109/ICDCSW.2011.20) on `(log2(rank), ensemble_mean)`;
  draw a vertical dotted line at the detected `r*_label` and label it
  `r*_TC`, `r*_WT`, `r*_ET`.
- **Significance markers.** Above each rank's data point, place asterisks for
  the Wilcoxon contrast `ensemble vs. baseline` (from
  `statistical_summary.json["ensemble_vs_baseline"][label]["p_value_wilcoxon"]`,
  Holm-corrected across ranks per label).

**Caption (template).**
> Inter-LoRA Dice on the BraTS-MEN test set (n=150) for (a) TC, (b) WT,
> (c) ET. Filled circles: ensemble mean (M=20) with 95 % CI; hollow squares:
> best individual member; faint dots: per-member means. Dashed line: frozen
> BSF baseline ± 2 SE. Vertical dotted lines mark the per-label elbow `r*`
> detected with the Kneedle algorithm. Stars: Wilcoxon signed-rank vs.
> baseline (Holm-corrected). Note saturation beyond `r* = …`.

---

### 3.2 Quant 2 — Calibration and epistemic diagnostics vs. rank

**Goal.** Show that the optimal rank is the one where calibration improves
*and* bias-dominance is resolved; this discriminates `r*` from spurious Dice
plateaus.

**Layout.** 2 × 2 grid, shared x-axis (log₂ rank).

| Panel | Y-axis | Source |
| --- | --- | --- |
| (a) Calibration error | ECE (left axis) and Brier (right twin) | `calibration.json` per rank; baseline from `BASELINE_DIR` if available, else NaN with note |
| (b) Coverage deficit | nominal − empirical, for nominal ∈ {0.5, 0.8, 0.9, 0.95} | `epistemic_taxonomy.json["calibration"]` and/or `calibration_coverage.csv` |
| (c) Bias dominance | % scans with `k* ≤ M`; % degenerate ensembles | `epistemic_taxonomy.json["taxonomy"]["estimation_bias"]["bias_dominance"]` and `bias_dominance_threshold.csv` |
| (d) Inter-member agreement | ICC(2,1) for TC, WT, ET | `statistical_summary.json["inter_member_agreement"]` |

**Per panel rules.**

- (a) Two y-axes; ECE in solid line/circles, Brier in dashed/triangles.
  Annotate the rank that minimises ECE.
- (b) Four overlaid lines (one per nominal level), markers `o, s, ^, D`.
  Add horizontal dotted line at zero; annotate `r` minimising the 0.95-level
  deficit.
- (c) Stacked area or twin-axis: `pct_scans_k_star_eq_1` (red),
  `pct_scans_k_star_exceeds_M` (orange), `pct_scans_degenerate_ensemble`
  (grey). Add a horizontal reference at 0.5 (majority).
- (d) Three lines (TC red, WT green, ET amber). Annotate the rank where ICC
  flattens (first rank with `|ICC(r)−ICC(r_max)| < 0.005`).

**Cross-panel annotation.** A dashed vertical band at the **median** of the
four per-panel optimal ranks; label the band as `r*_consensus`.

**Caption (template).**
> Calibration and epistemic diagnostics across LoRA ranks. (a) ECE and Brier
> on the BraTS-MEN test set with B=15 reliability bins. (b) Coverage deficit
> at four nominal levels; positive values indicate under-coverage of the
> Gaussian-quantile interval `μ_logV ± t_{α} σ_logV`. (c) Bias-dominance
> diagnostics: fraction of scans where `k* ≤ M` (cannot be improved by adding
> ensemble members). (d) ICC(2,1) of inter-member Dice agreement.
> Vertical band: consensus optimal rank `r*` across criteria.

---

## 4. Qualitative figures

### 4.1 Qual 1 — Multi-rank slice grid (ensemble + uncertainty)

**Goal.** Visual evidence of where the rank budget is being spent: boundary
refinement, tumour-core re-segmentation, peritumoural uncertainty.

**Subjects.** Two cases:

1. **One BraTS-MEN test scan with GT.** Selected as the case nearest to the
   median `dice_mean` of the highest-rank ensemble (representative, not
   cherry-picked). Tie-break by largest tumour volume.
2. **One MenGrowth study without GT.** Selected as the case with the highest
   ensemble inter-member std on log-volume (showcases epistemic uncertainty
   in the clinical inference setting).

Persist both selections (subject ID + chosen slice index) to
`data/selected_slices.json` so the figure is deterministic.

**Slice selection.** Axial slice with the largest GT tumour area (BraTS-MEN)
or largest ensemble-mask area (MenGrowth). Use the **same slice index** across
all rows for that subject.

**Layout.** Two stacked sub-figures (one per subject), each with:

- **Rows (top → bottom).** `frozen_BSF (baseline)`, `r=4`, `r=8`, `r=16`,
  `r=32`, `r=64` — only those that exist; row labels on the left in white.
- **Columns (left → right).**
  1. **Reference image.** T1c (BraTS-MEN) or the MenGrowth equivalent;
     `vmin/vmax` from 1st/99th percentile, `cmap='gray'`.
  2. **Ground truth.** Coloured overlay (TC red, WT green, ET amber) at
     alpha=0.55. For MenGrowth: blank panel labelled "no GT" centred in
     white text.
  3. **Ensemble mask.** Same overlay scheme.
  4. **Inter-member variance map.** Voxelwise std of softmax probs across
     the 20 members, summed over foreground classes; colormap `magma`,
     `vmin=0`, `vmax=q99` (the 99th percentile of the highest-rank map for
     that subject, shared across rows).
  5. **Predictive entropy.** From `entropy.nii.gz`; colormap `inferno`,
     `vmin=0`, `vmax=ln(C)` where `C` is the number of classes.

**Visual rules.**

- **Black background** (`figure.patch`, all axes).
- White spines off; small white scale-bar (10 mm) on the bottom-left of the
  reference column, computed from NIfTI affine.
- Single shared colorbar per uncertainty column at the right of each
  sub-figure.
- Title above each sub-figure: subject ID (white).
- **Crop** all panels to a tight bounding box around the union of GT (or
  ensemble) masks across rows, padded by 15 voxels.

**Caption (template).**
> Multi-rank qualitative comparison on (top) a representative BraTS-MEN test
> scan (`SCAN_ID`) and (bottom) a MenGrowth clinical study (`STUDY_ID`).
> Columns: reference T1c, ground truth (top only), ensemble mask, inter-member
> variance, predictive entropy. Rows: frozen BSF baseline and LoRA ranks
> r ∈ {4, 8, 16, 32, 64}. Same axial slice across all rows. Variance maps
> share a colour scale per subject (99th-percentile clipped).

---

### 4.2 Qual 2 — Clustered heatmap of per-scan Dice across ranks

**Goal.** Reveal scan-level heterogeneity in the rank response, and identify
the cohort-level rank above which most scans saturate.

**Layout.** 3 stacked heatmaps (one per label TC, WT, ET), shared row order.

**Construction.**

- **Rows.** All BraTS-MEN test scans (`scan_id` from `ensemble_test_dice.csv`).
- **Columns (in order).** `frozen_BSF`, `r=4`, `r=8`, `r=16`, `r=32`, `r=64`.
- **Cell value.** Ensemble Dice for that label (`dice_{label}`). Baseline
  column from `baseline_test_dice.csv`.
- **Row order.** Hierarchical clustering on the matrix
  `[D_TC | D_WT | D_ET]` (concatenated across labels) with linkage='ward',
  metric='euclidean'. Apply the same row order to all three heatmaps. A
  dendrogram is rendered to the left of the top heatmap only.
- **Column order.** Fixed (NOT clustered) — ranks must remain in
  monotonically increasing capacity order for interpretability.
- **Colour scale.** `RdYlGn`, `vmin=0.0`, `vmax=1.0`, `center=0.7`.
- **Annotations.** Cells with Dice < 0.4 outlined in white (failure cases).
- **Right margin.** Per-scan tumour volume (log₁₀ scale) as a vertical strip
  with `cmap='cividis'` to inspect whether failures correlate with size.

**Caption (template).**
> Clustered heatmaps of per-scan ensemble Dice across LoRA ranks for (a) TC,
> (b) WT, (c) ET on the BraTS-MEN test set. Rows: scans, sorted by
> hierarchical clustering on concatenated label-Dice vectors (Ward linkage).
> Columns: frozen BSF baseline and LoRA ranks. Right strip: log₁₀ tumour
> volume. White-outlined cells: Dice < 0.4 failure cases.

---

## 5. Tables

### 5.1 Tab 1 — Summary metrics per rank

One row per rank (plus a `frozen_BSF` row). Columns:

| Column | Source / formula |
| --- | --- |
| `rank` | discovery |
| `dice_tc_mean ± ci95` | `statistical_summary.json["ensemble_vs_baseline"]["tc"]` |
| `dice_wt_mean ± ci95` | idem |
| `dice_et_mean ± ci95` | idem |
| `ece` | `calibration.json["ece"]` |
| `brier` | `calibration.json["brier_score"]` |
| `cov95_deficit` | `epistemic_taxonomy.json["calibration"]["coverage_deficit_95"]` |
| `pct_bias_dominated` | `epistemic_taxonomy.json["taxonomy"]["estimation_bias"]["bias_dominance"]["pct_scans_k_star_eq_1"] + …["pct_scans_k_star_exceeds_M"]` |
| `icc_wt` | `statistical_summary.json["inter_member_agreement"]["icc_wt"]` |

Render as Markdown, CSV, and LaTeX (`booktabs`, `siunitx`). **Bold** the best
value per column (excluding the baseline row).

### 5.2 Tab 2 — Paired contrasts vs. frozen BSF

For each rank × label, report from `statistical_summary.json` and bootstrap
of `paired_differences.csv`:

| Column |
| --- |
| `rank`, `label`, `Δdice (mean)`, `Δdice 95 % CI`, `Wilcoxon p`, `Cohen's d` |

Apply Holm–Bonferroni across all `(rank, label)` cells; report both raw and
adjusted `p`. Highlight cells with adjusted `p < 0.05` and `|d| ≥ 0.5`.

---

## 6. Compiled metrics DataFrame (`data/compiled_metrics.parquet`)

Long-form schema (one row per rank × label):

| column | dtype |
| --- | --- |
| `rank` | int64 (0 means frozen BSF) |
| `label` | str ∈ {TC, WT, ET, mean} |
| `dice_mean` | float64 |
| `dice_ci_lo` | float64 |
| `dice_ci_hi` | float64 |
| `delta_vs_baseline` | float64 |
| `delta_ci_lo` | float64 |
| `delta_ci_hi` | float64 |
| `p_wilcoxon_raw` | float64 |
| `p_wilcoxon_holm` | float64 |
| `cohens_d` | float64 |
| `ece` | float64 (broadcast across labels) |
| `brier` | float64 |
| `cov95_deficit` | float64 |
| `pct_bias_dominated` | float64 |
| `icc` | float64 |

This file is the **single source of truth** for every plot and table; build
it first, then derive figures from it.

---

## 7. Implementation skeleton (mandatory module layout)

```
inter_lora_report/
├── __init__.py
├── io_layer.py            # readers for CSV/JSON/NIfTI; rank discovery
├── compile.py             # builds compiled_metrics.parquet
├── style.py               # mpl rcParams, palettes, helpers
├── plots/
│   ├── quant1_dice_vs_rank.py
│   ├── quant2_calib_epistemic.py
│   ├── qual1_slice_grid.py
│   └── qual2_clustered_heatmap.py
├── tables/
│   ├── tab1_summary.py
│   └── tab2_paired.py
├── cli.py                 # entrypoint: `python -m inter_lora_report.cli`
└── tests/
    ├── test_io_layer.py
    ├── test_compile.py
    └── test_acceptance.py # see § 9
```

**Coding conventions** (must follow):

- Type hints everywhere (`from __future__ import annotations`).
- Atomic functions, low cyclomatic complexity (≤ 8 per function).
- `dataclasses.dataclass(slots=True, frozen=True)` for config objects
  (`PlotConfig`, `RankRun`, `LabelSpec`).
- Custom exceptions in `errors.py`: `RankDiscoveryError`,
  `MissingArtefactError`, `BaselineMismatchError`.
- Structured logging via `logging` with JSON formatter; one logger per
  module; level configurable via env var `INTER_LORA_LOGLEVEL`.
- Memory: load NIfTI volumes lazily and explicitly `del` arrays after use in
  Qual1; `gc.collect()` between subjects.
- No re-computation of metrics that already exist in CSV/JSON.
- Docstrings (NumPy style) on every public function; **no usage examples**
  in docstrings.

---

## 8. CLI contract

```
python -m inter_lora_report.cli \
    --root        /media/.../stages_1234 \
    --out-root    /media/.../stages_1234/_inter_lora_report \
    --ranks       4 8 16 32 64 \
    --baseline-dir /media/.../baseline_frozen_bsf  \
    --select-subjects auto \
    --bootstrap-n 1000 \
    --seed 42 \
    --dpi 600
```

Flags:

- `--select-subjects {auto|<id1> <id2>}` — `auto` runs the rule of § 4.1.
- `--skip {quant1,quant2,qual1,qual2,tables}` — rerun a subset.
- `--strict` — error on any missing per-rank artefact (default: warn-skip).
- `--dry-run` — discover and validate inputs, write `compiled_metrics.parquet`
  only.

Exit codes: `0` success, `2` discovery failure, `3` data integrity failure
(see § 9), `4` rendering failure.

---

## 9. Acceptance criteria

The agent must run the suite end-to-end and verify all checks below. Failure
of any **MUST** check means the task is not complete.

### 9.1 Discovery and integrity (MUST)

- [ ] At least 3 ranks discovered; all expected ranks `{4,8,16,32,64}` present
  unless explicit `--ranks` override.
- [ ] All per-rank `evaluation/` files in § 1.2 are readable; missing files
  trigger `MissingArtefactError` under `--strict`.
- [ ] `baseline_test_dice.csv` is identical (max abs diff < 1e-6) across all
  ranks, or `BASELINE_DIR` is provided.
- [ ] `compiled_metrics.parquet` exists, has no NaNs in `dice_mean`,
  `ece`, `brier`, and `pct_bias_dominated` columns, and contains exactly
  `(n_ranks + 1) × 3` rows for the per-label entries (plus a `mean` label).

### 9.2 Numerical sanity (MUST)

- [ ] For every rank, `ensemble_mean ≥ baseline_mean − 0.01` for WT
  (transfer should not catastrophically degrade); flag and continue otherwise.
- [ ] ICC values lie in `[0, 1]`.
- [ ] All Wilcoxon p-values lie in `(0, 1]`; Holm-adjusted p-values are
  monotonic.
- [ ] Detected `r*` per label is one of the discovered ranks.

### 9.3 Figure outputs (MUST)

- [ ] All 4 figures exist in PDF, PNG (and SVG where listed); PNG ≥ 600 DPI;
  PDF is vector (no rasterised axes — `set_rasterization_zorder` only on
  heatmap cells of Qual2).
- [ ] Each figure embeds the git SHA and ISO timestamp in metadata.
- [ ] Qual1: background is RGB(0,0,0); spines are off; the same slice index is
  used across rows for a given subject (assert via parsed metadata file).
- [ ] Quant1: vertical dotted lines for `r*` are annotated; legend lists
  ensemble, best member, baseline.
- [ ] Quant2: all four panels share the x-axis ticks; consensus `r*` band is
  rendered.
- [ ] Qual2: row order identical across the three heatmaps; baseline column
  is leftmost; dendrogram present on top heatmap only.

### 9.4 Tables (MUST)

- [ ] Tab 1 and Tab 2 exist as `.tex`, `.csv`, `.md`. CSVs have header row;
  LaTeX uses `booktabs`.
- [ ] Tab 1: best per-column values are bolded (excluding baseline row).
- [ ] Tab 2: Holm-corrected p-values present, monotonic non-decreasing
  within each label after sorting by raw p.

### 9.5 End-to-end smoke test (MUST)

A pytest in `tests/test_acceptance.py` runs:

```python
def test_end_to_end(tmp_path):
    # 1. Run CLI on the canonical fixture (smallest 2-rank subset).
    # 2. Assert all files in § 1.3 exist.
    # 3. Re-load compiled_metrics.parquet and reproduce the headline
    #    statistic: dice_wt_mean(r=8) within ±1e-6 of the original
    #    statistical_summary.json["ensemble_vs_baseline"]["wt"]["ensemble_mean"].
    # 4. Open each PNG with PIL and verify shape, mode, and that the
    #    Qual1 image's mean pixel value is < 30 (mostly black).
```

The agent MUST run this test and report `PASSED` before declaring completion.

### 9.6 Reproducibility (SHOULD)

- [ ] Re-running the CLI on the same inputs produces byte-identical
  `compiled_metrics.parquet` (deterministic).
- [ ] Figures differ only in metadata timestamp; pixel-level diff
  (mean abs diff < 1.0 / 255) on the PNGs across consecutive runs.

### 9.7 Scientific narrative checklist (SHOULD)

The agent emits `report_summary.md` containing:

- [ ] The detected `r*_consensus` and the three per-label `r*_label`.
- [ ] Plain-language interpretation: "At `r*=…`, Dice has saturated within X
  CI half-width of the maximum, ECE is below 1e-3, and the fraction of
  bias-dominated scans is below Y."
- [ ] Open issues (e.g., bias dominance not resolved at any tested rank,
  coverage deficit not closing).

---

## 10. Pitfalls and notes

- **`bias_dominance_threshold.csv`** uses `pct_scans_k_star_eq_1` ≈ 0.87 in
  the r=8 fixture, meaning the model is *severely bias-dominated* at r=8.
  Plot Quant2(c) is therefore the discriminative panel for `r*` selection.
- **`coverage_deficit_95` ≈ 0.64** at r=8 — Gaussian-quantile intervals on
  log-volume drastically under-cover. This is expected to *decrease* with
  rank if the bias hypothesis is correct; if it does not, report it as a
  scientific finding (intrinsic mis-specification of the parametric
  interval, not adapter capacity).
- **Bin 0 in calibration** (`bin_count = 6,194,495`) is overwhelmingly
  background voxels with `confidence ≈ 0`; ECE is dominated by the
  high-confidence bin. Note this in the caption of any reliability subplot,
  if added later.
- **MenGrowth selection** must be reproducible — persist subject + slice in
  `selected_slices.json`. Never silently rotate.
- **NIfTI orientation.** Use `nibabel.aff2axcodes` to flip to a canonical
  RAS orientation before slicing; otherwise rows of Qual1 may show
  inconsistent anatomy across ranks.
- **ET on meningiomas** is small and fragmented; thresholding the ensemble
  mask at argmax may produce empty ET. Use the existing `ensemble_mask.nii.gz`
  rather than re-thresholding.

---

## 11. References (for the report's bibliography)

1. Aghajanyan, A., Zettlemoyer, L., Gupta, S. (2020). *Intrinsic
   Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*.
   arXiv:2012.13255.
2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
   Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
   arXiv:2106.09685.
3. Satopää, V., Albrecht, J., Irwin, D., Raghavan, B. (2011). *Finding a
   "Kneedle" in a Haystack: Detecting Knee Points in System Behavior*. ICDCSW.
   doi:10.1109/ICDCSW.2011.20.
4. Naeini, M. P., Cooper, G., Hauskrecht, M. (2015). *Obtaining Well
   Calibrated Probabilities Using Bayesian Binning*. AAAI.
5. Kendall, A., Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian
   Deep Learning for Computer Vision?* NeurIPS. arXiv:1703.04977.
6. Houlsby, N., Huszár, F., Ghahramani, Z., Lengyel, M. (2011). *Bayesian
   Active Learning for Classification and Preference Learning* (BALD).
   arXiv:1112.5745.
7. Shrout, P. E., Fleiss, J. L. (1979). *Intraclass correlations: uses in
   assessing rater reliability*. Psychological Bulletin, 86(2), 420–428.

---

**End of specification.** The agent must read § 9 carefully and run the
acceptance suite before reporting completion.
