# SPEC v1.1: Follow-Up Revisions for LoRA-Ensemble Module

**Version:** 1.1 (revision of LORA_ENSEMBLE_SPEC.md v1.0)  
**Date:** 2026-04-01  
**Target Agent:** Claude Code Opus 4.6  
**Context:** The v1.0 spec was implemented. This document specifies corrections, additions, and a complete output structure design. Apply these changes to the existing code in `experiments/uncertainty_segmentation/`.

---

## 0. Summary of required changes

| ID | Category | Severity | Description |
|----|----------|----------|-------------|
| R1 | Output structure | **Critical** | Experiment directory must be parameterised by `(rank, M, seed)` — current flat `output_dir` collides across runs |
| R2 | Statistical data | **Critical** | Per-member × per-subject Dice on test set not saved — needed for bootstrap CIs, paired tests, effect sizes |
| R3 | Statistical data | **Critical** | Missing baseline comparison (frozen BrainSegFounder Dice) — required for paired statistical tests |
| R4 | Reproducibility | **High** | No experiment manifest (git commit, env, config snapshot) — use existing `save_reproducibility_artifacts()` |
| R5 | Statistical metrics | **High** | Missing inter-member agreement metrics (Fleiss' κ, ICC) — required for ensemble diversity analysis |
| R6 | Evaluation pipeline | **High** | Per-member test-set evaluation missing from pipeline — evaluate after training, not only ensemble |
| R7 | Volume CSV | **Medium** | Per-member log-volumes not stored — needed for GP heteroscedastic noise propagation |
| R8 | SLURM launch | **High** | Launch script does not correctly resolve picasso config override; evaluation job not submitted |
| R9 | Statistical summary | **High** | No automated computation of 95% CIs, p-values, Cohen's d — must be in evaluation output |
| R10 | Config | **Medium** | `out_channels` hardcoded to 3 — should come from config for maintainability |

---

## 1. Experiment output structure (R1)

### 1.1 Problem

The current code writes all output to `config.experiment.output_dir` with no parameterisation. Running the same module with different ranks or ensemble sizes overwrites previous results.

### 1.2 Specification

An experiment run is uniquely identified by the tuple `(rank, n_members, base_seed)`. The output directory must be derived automatically:

```python
def get_run_dir(config: DictConfig) -> Path:
    """Derive the experiment run directory from config parameters.
    
    Format: {output_dir}/r{rank}_M{n_members}_s{base_seed}/
    Example: results/uncertainty_segmentation/r8_M5_s42/
    """
    base = Path(config.experiment.output_dir)
    run_name = f"r{config.lora.rank}_M{config.ensemble.n_members}_s{config.ensemble.base_seed}"
    return base / run_name
```

**Every file path in every module** that currently uses `config.experiment.output_dir` directly must be replaced with `get_run_dir(config)`. Place this function in a new file `experiments/uncertainty_segmentation/engine/paths.py`.

### 1.3 Full output tree

```
{output_dir}/r{rank}_M{n_members}_s{base_seed}/
│
├── config_snapshot.yaml                          # Frozen resolved config (R4)
├── manifest.yaml                                 # Reproducibility metadata (R4)
│
├── adapters/                                     # Per-member trained checkpoints
│   ├── member_0/
│   │   ├── adapter/                              # PEFT adapter files
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── README.md
│   │   ├── decoder.pt                            # Decoder state dict
│   │   ├── training_log.csv                      # Per-epoch: epoch, train_loss, val_dice_{tc,wt,et,mean}, lr
│   │   └── training_summary.yaml                 # Best epoch, best Dice, seed, timing, param counts
│   ├── member_1/
│   ├── ...
│   └── member_{M-1}/
│
├── evaluation/                                   # Test-set evaluation (R2, R3, R5, R9)
│   ├── per_member_test_dice.csv                  # Per-member × per-subject Dice (R2)
│   ├── ensemble_test_dice.csv                    # Ensemble Dice per subject (R2)
│   ├── baseline_test_dice.csv                    # Frozen BrainSegFounder Dice per subject (R3)
│   ├── calibration.json                          # ECE, Brier, reliability bin data
│   ├── agreement.json                            # Fleiss' κ, pairwise Cohen's κ, ICC (R5)
│   └── statistical_summary.json                  # CIs, p-values, Cohen's d (R9)
│
├── volumes/                                      # Volume extraction with uncertainty
│   ├── mengrowth_volumes.csv                     # MenGrowth cohort: per-scan volumes (R7)
│   └── men_test_volumes.csv                      # BraTS-MEN test: for validation vs GT volumes
│
└── logs/                                         # SLURM logs
    ├── train_0_{jobid}.out
    ├── train_0_{jobid}.err
    ├── ...
    ├── evaluate_{jobid}.out
    └── inference_{jobid}.out
```

### 1.4 CSV schemas

These schemas are fixed contracts. Downstream analysis code depends on them.

**`per_member_test_dice.csv`** — One row per (member, scan) pair:

| Column | Type | Description |
|--------|------|-------------|
| member_id | int | Ensemble member index (0-based) |
| scan_id | str | BraTS-MEN scan identifier |
| dice_tc | float | Dice for Tumor Core |
| dice_wt | float | Dice for Whole Tumor |
| dice_et | float | Dice for Enhancing Tumor |
| dice_mean | float | Mean of the three Dice scores |
| volume_pred | float | Predicted WT volume (mm³) from this member |

This table has $M \times N_{\text{test}}$ rows. It enables:
- Bootstrap 95% CIs on per-member Dice (resample subjects).
- Paired Wilcoxon signed-rank test: ensemble vs. each member.
- Cohen's d: effect size of ensembling.
- Intraclass correlation coefficient (ICC) across members.

**`ensemble_test_dice.csv`** — One row per scan:

| Column | Type | Description |
|--------|------|-------------|
| scan_id | str | BraTS-MEN scan identifier |
| dice_tc | float | Ensemble Dice (TC) |
| dice_wt | float | Ensemble Dice (WT) |
| dice_et | float | Ensemble Dice (ET) |
| dice_mean | float | Mean Dice |
| volume_ensemble | float | Volume from ensemble-averaged mask (mm³) |
| volume_gt | float | Ground truth WT volume (mm³) from GT segmentation |

**`baseline_test_dice.csv`** — Same schema as `ensemble_test_dice.csv` but from the frozen BrainSegFounder without any LoRA adaptation. One row per scan.

**`mengrowth_volumes.csv`** — One row per MenGrowth scan:

| Column | Type | Description |
|--------|------|-------------|
| scan_id | str | MenGrowth scan identifier |
| patient_id | str | Patient identifier |
| timepoint_idx | int | Temporal index |
| vol_mean | float | Mean volume across M members (mm³) |
| vol_std | float | Std of volume across M members (mm³) |
| logvol_mean | float | Mean of log(V+1) across M members |
| logvol_std | float | Std of log(V+1) across M members |
| vol_m0, vol_m1, ... | float | Per-member volumes (mm³) |
| logvol_m0, logvol_m1, ... | float | Per-member log(V+1) values (**NEW — R7**) |
| wt_mean_entropy | float | Mean WT predictive entropy |
| wt_mean_mi | float | Mean WT mutual information |
| wt_boundary_entropy | float | Mean entropy at tumor boundary |
| wt_boundary_mi | float | Mean MI at tumor boundary |

**`statistical_summary.json`** — Computed automatically after evaluation:

```json
{
  "ensemble_vs_baseline": {
    "dice_wt": {
      "ensemble_mean": 0.872,
      "baseline_mean": 0.823,
      "delta": 0.049,
      "ci_95_lower": 0.031,
      "ci_95_upper": 0.068,
      "p_value_wilcoxon": 0.0023,
      "cohens_d": 0.61
    },
    "dice_tc": { "..." },
    "dice_et": { "..." }
  },
  "ensemble_vs_best_member": {
    "best_member_id": 2,
    "dice_wt": {
      "ensemble_mean": 0.872,
      "best_member_mean": 0.861,
      "delta": 0.011,
      "ci_95_lower": -0.003,
      "ci_95_upper": 0.025,
      "p_value_wilcoxon": 0.087,
      "cohens_d": 0.18
    }
  },
  "inter_member_agreement": {
    "icc_wt": 0.943,
    "fleiss_kappa_wt": 0.891,
    "mean_pairwise_dice_wt": 0.912
  },
  "calibration": {
    "ece": 0.032,
    "brier": 0.041
  },
  "per_member_summary": [
    {"member_id": 0, "dice_wt_mean": 0.858, "dice_wt_std": 0.092, "dice_wt_ci95": [0.831, 0.885]},
    {"member_id": 1, "...": "..."}
  ]
}
```

---

## 2. Per-member test-set evaluation (R2, R6)

### 2.1 Problem

The current `run_evaluate.py` computes only the ensemble-averaged Dice. It initialises `per_member_dice` accumulators but never fills them (the evaluation loop only calls `predictor.predict_scan()` which aggregates internally). For statistical analysis, we need per-member Dice on every test subject.

### 2.2 Specification

**Add** a new function `evaluate_per_member()` to `engine/ensemble_inference.py` (or a new file `engine/evaluate_members.py`):

```python
def evaluate_per_member(
    config: DictConfig,
    device: str = "cuda",
) -> pd.DataFrame:
    """Evaluate each ensemble member independently on test set.
    
    For each member m and each test scan:
        1. Load member m's model (LoRA adapter + decoder)
        2. Run sliding_window_segment
        3. Compute Dice (TC, WT, ET) against ground truth
        4. Compute predicted WT volume
    
    Returns:
        DataFrame with columns: member_id, scan_id, dice_tc, dice_wt, 
        dice_et, dice_mean, volume_pred
    """
```

This function must iterate **member × scan** (not use the ensemble predictor). For each member, load the model once, iterate over all test scans, compute Dice and volume. This gives $M \times N_{\text{test}}$ rows.

**Separately**, the ensemble evaluation produces one row per scan (the existing logic, but now saving per-subject).

### 2.3 Implementation note

This is computationally expensive ($M \times N_{\text{test}}$ forward passes). For the SLURM pipeline, it should run as a **separate job after training** (possibly parallelised: one GPU per member evaluating all test scans). The agent should create an `evaluate_worker.sh` SLURM script.

---

## 3. Baseline comparison (R3)

### 3.1 Problem

Without the frozen BrainSegFounder's Dice on the same test set, we cannot compute paired statistical tests or effect sizes. The ensemble may improve Dice, but we need to quantify this rigorously.

### 3.2 Specification

**Add** a function `evaluate_baseline()` that:

1. Loads the frozen BrainSegFounder (no LoRA, no decoder training) via `load_full_swinunetr()` with `freeze_encoder=True, freeze_decoder=True`.
2. Runs sliding-window inference on every BraTS-MEN test scan.
3. Computes Dice (TC, WT, ET) and WT volume per scan.
4. Saves to `evaluation/baseline_test_dice.csv`.

This only needs to be computed **once** regardless of `(rank, M, seed)`. The agent should check whether the file already exists and skip if so (or accept a `--force` flag).

---

## 4. Statistical summary computation (R5, R9)

### 4.1 Required statistical tests

After all evaluation data is collected, compute these automatically in a function `compute_statistical_summary()`:

**4.1.1. Bootstrap 95% confidence intervals on Dice**

For each method (baseline, each member, ensemble), resample subjects with replacement $B = 10{,}000$ times and compute the Dice mean on each bootstrap sample. Report the 2.5th and 97.5th percentiles (BCa method preferred, percentile method acceptable).

$$\text{CI}_{95\%} = \bigl[\hat{\theta}^*_{(0.025)},\; \hat{\theta}^*_{(0.975)}\bigr]$$

Use `scipy.stats.bootstrap()` (SciPy ≥ 1.9) or manual implementation.

**4.1.2. Paired Wilcoxon signed-rank test**

For the paired comparison ensemble vs. baseline on per-subject Dice:

$$H_0: \text{median}(\Delta_i) = 0 \quad \text{where} \quad \Delta_i = \text{Dice}^{\text{ensemble}}_i - \text{Dice}^{\text{baseline}}_i$$

Use `scipy.stats.wilcoxon()`. Report the test statistic $W$, $p$-value, and $r = Z / \sqrt{N}$ as effect size.

**4.1.3. Cohen's d (paired)**

$$d = \frac{\bar{\Delta}}{\text{SD}(\Delta)}, \quad \Delta_i = \text{Dice}^{\text{ensemble}}_i - \text{Dice}^{\text{baseline}}_i$$

Interpret: $|d| < 0.2$ negligible, $0.2$–$0.5$ small, $0.5$–$0.8$ medium, $> 0.8$ large.

**4.1.4. Intraclass Correlation Coefficient (ICC)**

For inter-member agreement on WT Dice, compute ICC(3,1) — two-way mixed, single measures, consistency:

$$\text{ICC}(3,1) = \frac{MS_R - MS_E}{MS_R + (k-1) MS_E}$$

where $MS_R$ is the mean square for rows (subjects), $MS_E$ is mean square error, and $k$ is the number of members. Use `pingouin.intraclass_corr()` or manual ANOVA computation.

**4.1.5. Fleiss' kappa (optional)**

For voxel-level inter-rater agreement. This is expensive on full volumes, so subsample (every 8th voxel) and compute on WT binary masks.

### 4.2 Implementation

Create `engine/statistical_analysis.py`:

```python
def compute_statistical_summary(
    per_member_dice: pd.DataFrame,     # per_member_test_dice.csv
    ensemble_dice: pd.DataFrame,       # ensemble_test_dice.csv
    baseline_dice: pd.DataFrame,       # baseline_test_dice.csv
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """Compute all statistical tests and return structured summary.
    
    Returns dict matching the statistical_summary.json schema.
    """
```

Dependencies: `scipy.stats` (wilcoxon, bootstrap), `numpy` (manual Cohen's d, ICC). Do **not** add `pingouin` as a required dependency — implement ICC manually or make it optional.

---

## 5. Reproducibility manifest (R4)

### 5.1 Specification

At the start of every run (training, evaluation, inference), save:

```python
from growth.utils.reproducibility import save_reproducibility_artifacts

manifest = save_reproducibility_artifacts(
    output_dir=run_dir,
    config=OmegaConf.to_container(config, resolve=True),
    extra_info={
        "experiment_type": "lora_ensemble_uncertainty",
        "run_params": {"rank": config.lora.rank, "M": config.ensemble.n_members, "seed": config.ensemble.base_seed},
    }
)
```

Additionally, save a **resolved** config snapshot:

```python
OmegaConf.save(config, run_dir / "config_snapshot.yaml", resolve=True)
```

This ensures that even if the original config files change, the exact parameters of each run are preserved.

---

## 6. Per-member log-volumes in volume CSV (R7)

### 6.1 Problem

The current `volume_extraction.py` saves `vol_m0, vol_m1, ...` (per-member raw volumes) but not per-member log-volumes. The downstream GP needs $\sigma^2_v$ in **log-space**, so we need the per-member $\log(V^{(m)} + 1)$ values to compute proper sample variance.

### 6.2 Change

In `engine/volume_extraction.py`, after the per-member volume loop, add:

```python
for m_idx, vol in enumerate(result.per_member_volumes):
    row[f"vol_m{m_idx}"] = vol
    row[f"logvol_m{m_idx}"] = math.log(vol + 1)  # <-- ADD THIS
```

This is trivial but essential for the heteroscedastic GP integration.

---

## 7. SLURM environment and launch workflow (R8)

### 7.1 Problem

The current `launch.sh` has three issues:
1. Reads `output_dir` and `n_members` from the base config only, not from the merged picasso override.
2. Does not submit the evaluation job.
3. Does not derive the run directory from `(rank, M, seed)`.

### 7.2 Complete SLURM workflow

Each experiment is launched via **a single `launch.sh` invocation** from the Picasso login node. The launch script orchestrates a chain of dependent jobs:

```
launch.sh (login node)
    │
    ├─→ [STEP 1] sbatch --array=0-{M-1}  train_worker.sh      (M GPUs, parallel)
    │       Each array element trains one LoRA member
    │
    ├─→ [STEP 2] sbatch --dependency=afterok:STEP1  evaluate_worker.sh  (1 GPU)
    │       Evaluates all M members + ensemble + baseline on BraTS-MEN test
    │       Computes statistical summary
    │
    └─→ [STEP 3] sbatch --dependency=afterok:STEP2  inference_worker.sh (1 GPU)
            Runs ensemble inference on MenGrowth cohort
            Produces volume CSV with uncertainty
```

### 7.3 Revised `launch.sh`

The launch script must:

1. **Accept** the base config path as argument (default: `experiments/uncertainty_segmentation/config.yaml`).
2. **Detect** the picasso override config and merge it (if running on Picasso).
3. **Resolve** the run directory from the **merged** config: `r{rank}_M{n_members}_s{base_seed}`.
4. **Create** the run directory and logs subdirectory.
5. **Copy** the resolved config into the run directory as `config_snapshot.yaml`.
6. **Submit** three SLURM jobs with dependency chain.
7. **Export** `RUN_DIR` to all workers so they know where to write.

```bash
# Key change: resolve params from MERGED config
MERGED_CONFIG=$(python3 -c "
from omegaconf import OmegaConf
base = OmegaConf.load('${CONFIG_PATH}')
override_path = '${PICASSO_OVERRIDE}'
import os
if os.path.exists(override_path):
    override = OmegaConf.load(override_path)
    cfg = OmegaConf.merge(base, override)
else:
    cfg = base
r = cfg.lora.rank
M = cfg.ensemble.n_members
s = cfg.ensemble.base_seed
out = cfg.experiment.output_dir
run_dir = f'{out}/r{r}_M{M}_s{s}'
print(f'{M} {run_dir}')
# Also save resolved config
import pathlib
pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, f'{run_dir}/config_snapshot.yaml', resolve=True)
print(f'{run_dir}/config_snapshot.yaml', file=__import__('sys').stderr)
")

read -r N_MEMBERS RUN_DIR <<< "${MERGED_CONFIG}"
export RUN_DIR
```

### 7.4 Worker scripts must use `RUN_DIR`

All SLURM workers must:
- Receive `RUN_DIR` via `--export`.
- Pass `--run-dir ${RUN_DIR}` to the Python CLI (add this argument to all `run_*.py` scripts).
- The Python code uses `run_dir` instead of deriving it from config (the config is already snapshot'd in `RUN_DIR/config_snapshot.yaml`).

**Add** `--run-dir` argument to `run_train.py`, `run_evaluate.py`, and `run_inference.py`. If provided, it overrides the derived run directory. If not provided, derive from config as usual (for local runs).

### 7.5 Picasso cluster specifics

For reference, here is the Picasso environment the agent should target:

| Property | Value |
|----------|-------|
| Cluster | Picasso (SCBI, Universidad de Málaga) |
| GPU nodes | DGX A100 (40GB per GPU) |
| Constraint | `--constraint=dgx` |
| Partition | default (no explicit `--partition` needed) |
| Home path | `/mnt/home/users/tic_163_uma/mpascual/` |
| Scratch | `/mnt/home/users/tic_163_uma/mpascual/fscratch/` |
| Conda env | `growth` |
| Module load | `miniconda3` (detected by existing scripts) |
| Max walltime | `0-12:00:00` for training, `0-06:00:00` for eval/inference |
| Per-job resources | 1 GPU, 16 CPUs, 64 GB RAM |

### 7.6 New SLURM file: `evaluate_worker.sh`

```bash
#SBATCH -J lora_ens_eval
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
```

This worker:
1. Runs `run_evaluate.py` with `--run-dir ${RUN_DIR}` — evaluates per-member Dice, ensemble Dice, baseline Dice, calibration, statistical tests.
2. All output goes to `${RUN_DIR}/evaluation/`.

---

## 8. Changes to existing files

### 8.1 `engine/train_member.py`

- **Change** `member_dir` derivation to use `get_run_dir(config) / "adapters" / f"member_{member_id}"`.
- **Add** `run_dir` parameter to `train_single_member()` (optional, for SLURM override).

### 8.2 `engine/ensemble_inference.py`

- **Change** `self.adapter_base_dir` to use `get_run_dir(config) / "adapters"`.
- **Add** optional `run_dir` parameter to `EnsemblePredictor.__init__()`.

### 8.3 `engine/volume_extraction.py`

- **Add** per-member log-volume columns (R7).
- **Add** ensemble-averaged WT volume column for the ensemble mask (not just per-member).

### 8.4 `run_evaluate.py`

Major rewrite:
- Call `evaluate_per_member()` → save `per_member_test_dice.csv`.
- Call `evaluate_ensemble()` → save `ensemble_test_dice.csv`.
- Call `evaluate_baseline()` → save `baseline_test_dice.csv`.
- Call `compute_statistical_summary()` → save `statistical_summary.json`.
- Save `calibration.json` and `agreement.json`.
- All files go to `run_dir / "evaluation/"`.

### 8.5 `config.yaml`

Add:
```yaml
evaluation:
  # ... existing ...
  n_bootstrap: 10000            # Bootstrap iterations for CIs
  alpha: 0.05                   # Significance level
  subsample_voxels_step: 8      # Voxel subsampling for calibration (memory)
```

### 8.6 New file: `engine/paths.py`

```python
"""Centralised path derivation for the uncertainty_segmentation module."""

from pathlib import Path
from omegaconf import DictConfig


def get_run_dir(config: DictConfig, override: Path | None = None) -> Path:
    """Derive experiment run directory.
    
    Args:
        config: Full experiment config.
        override: If provided, use this directly (from SLURM --run-dir).
    
    Returns:
        Path to the run directory.
    """
    if override is not None:
        return override
    base = Path(config.experiment.output_dir)
    run_name = (
        f"r{config.lora.rank}"
        f"_M{config.ensemble.n_members}"
        f"_s{config.ensemble.base_seed}"
    )
    return base / run_name
```

### 8.7 New file: `engine/statistical_analysis.py`

Contains `compute_statistical_summary()`, `bootstrap_ci()`, `paired_cohens_d()`, `compute_icc()`.

### 8.8 New file: `engine/evaluate_baseline.py`

Contains `evaluate_baseline()` — loads frozen BrainSegFounder, evaluates on test set, saves CSV.

### 8.9 New file: `engine/evaluate_members.py`

Contains `evaluate_per_member()` — evaluates each member independently on test set.

---

## 9. Dependency order for implementation

```
R1 (paths.py + get_run_dir)
 ├─→ Update train_member.py to use get_run_dir
 ├─→ Update ensemble_inference.py to use get_run_dir  
 ├─→ Update volume_extraction.py (R7: add logvol_m* columns)
 │
 ├─→ R4 (reproducibility: config snapshot + manifest in run_dir)
 │
 ├─→ R6 + R2 (evaluate_members.py: per-member test Dice)
 │    └─→ R3 (evaluate_baseline.py: frozen BSF Dice)
 │         └─→ R5 + R9 (statistical_analysis.py: CIs, p-values, Cohen's d, ICC)
 │              └─→ Rewrite run_evaluate.py to orchestrate all evaluation
 │
 └─→ R8 (SLURM: rewrite launch.sh, add evaluate_worker.sh, pass --run-dir)
      └─→ Update run_train.py, run_inference.py with --run-dir argument
```

**Recommended implementation order:**
1. `engine/paths.py` (R1)
2. Update all existing files to use `get_run_dir()` (R1)
3. Add `--run-dir` argument to all CLI scripts (R8)
4. `engine/volume_extraction.py` additions (R7)
5. Reproducibility artifacts (R4)
6. `engine/evaluate_members.py` (R2, R6)
7. `engine/evaluate_baseline.py` (R3)
8. `engine/statistical_analysis.py` (R5, R9)
9. Rewrite `run_evaluate.py` (orchestration)
10. SLURM scripts (R8)
11. Tests

---

## 10. Testing additions

Add to `tests/growth/test_uncertainty_segmentation.py`:

```python
class TestGetRunDir:
    def test_derives_correct_path(self):
        """r8_M5_s42 from rank=8, n_members=5, base_seed=42."""

    def test_override_takes_precedence(self):
        """--run-dir overrides derivation."""

class TestStatisticalAnalysis:
    def test_bootstrap_ci_known_distribution(self):
        """Bootstrap CI of N(0,1) samples covers 0."""
    
    def test_cohens_d_known_effect(self):
        """d ≈ 1.0 for samples shifted by 1 std."""
    
    def test_wilcoxon_detects_shift(self):
        """p < 0.05 for clearly shifted paired samples."""
    
    def test_icc_perfect_agreement(self):
        """ICC = 1.0 when all raters agree."""

class TestVolumeCSVSchema:
    def test_mengrowth_csv_has_logvol_columns(self):
        """Verify logvol_m* columns are present."""
    
    def test_per_member_columns_match_n_members(self):
        """N vol_m* columns == n_members."""
```

---

## 11. Verification checklist

After all changes are applied, verify:

- [ ] `python -c "from experiments.uncertainty_segmentation.engine.paths import get_run_dir"` works.
- [ ] Running `run_train.py --member-id 0 --run-dir /tmp/test_run` creates `/tmp/test_run/adapters/member_0/`.
- [ ] Running `run_evaluate.py --run-dir /tmp/test_run` produces all 5 CSV/JSON files in `evaluation/`.
- [ ] `statistical_summary.json` contains `ci_95_lower`, `ci_95_upper`, `p_value_wilcoxon`, `cohens_d`.
- [ ] `mengrowth_volumes.csv` contains `logvol_m0`, `logvol_m1`, ... columns.
- [ ] `config_snapshot.yaml` exists in run directory with resolved paths (no `${...}` interpolation).
- [ ] `manifest.yaml` exists with git commit hash.
- [ ] `launch.sh` correctly resolves the merged config and passes `RUN_DIR` to all workers.
- [ ] All tests in `test_uncertainty_segmentation.py` pass: `pytest -m unit -k uncertainty`.
