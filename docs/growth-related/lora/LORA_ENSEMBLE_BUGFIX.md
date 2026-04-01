# BUGFIX: LoRA-Ensemble Module — Issues Found in v1.1

**Date:** 2026-04-01  
**Severity scale:** 🔴 Crash, 🟠 Wrong results, 🟡 Missing feature, ⚪ Cosmetic

---

## 🔴 BUG-1: `run_inference.py` — `NameError` on `resolved_run_dir`

**File:** `run_inference.py`, lines ~74–75

**Problem:** `resolved_run_dir` is used before it is defined:

```python
    # Line ~74 — USED HERE (undefined)
    predictions_dir = resolved_run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # ... several lines later ...

    # Line ~85 — DEFINED HERE (too late)
    resolved_run_dir = get_run_dir(config, override=args.run_dir)
```

This raises `NameError` at runtime and makes the entire inference pipeline crash.

**Fix:** Move the `resolved_run_dir` assignment to before its first use. The corrected order:

```python
    resolved_run_dir = get_run_dir(config, override=args.run_dir)

    predictions_dir = resolved_run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # ... then use resolved_run_dir for output_dir as well ...
    output_dir = resolved_run_dir / "volumes"
```

Remove the duplicate `resolved_run_dir = get_run_dir(...)` call that appears later.

---

## 🟠 BUG-2: `evaluate_members.py` — Scan IDs extracted incorrectly

**File:** `engine/evaluate_members.py`, in both `evaluate_per_member()` and `evaluate_ensemble_per_subject()`

**Problem:** Both functions use:
```python
sid = sample.get("subject_id", f"scan_{i}")
```

`sample` is a dict returned by `BraTSDatasetH5.__getitem__()`, which returns `{"image": ..., "seg": ...}` and may or may not include `"subject_id"` depending on the H5 schema version. The v2.0 schema uses `scan_ids` at the file level, not in individual samples. The fallback `f"scan_{i}"` produces generic names like `scan_0`, `scan_1`, ... which **will not match** the scan IDs in the baseline and volume CSVs, breaking all downstream paired statistical tests (Wilcoxon, Cohen's d) that join on `scan_id`.

**Fix:** Load scan IDs from the H5 file metadata directly, the same way `volume_extraction.py` already does correctly. Both functions should:

```python
import h5py

# Before the loop:
with h5py.File(h5_path, "r") as f:
    all_scan_ids = [
        s.decode() if isinstance(s, bytes) else s for s in f["scan_ids"][:]
    ]
# Get the split indices used by the dataset
splits = BraTSDatasetH5.load_splits_from_h5(h5_path)
test_indices = splits.get(config.data.test_split, np.arange(len(all_scan_ids)))

# Inside the loop:
scan_idx = test_indices[i]
sid = all_scan_ids[scan_idx]
```

Apply this pattern to both `evaluate_per_member()` and `evaluate_ensemble_per_subject()`.

---

## 🟠 BUG-3: `out_channels=3` hardcoded — Config value ignored

**Files:** `engine/train_member.py` (lines 79, 98), `engine/ensemble_inference.py` (lines 175, 193)

**Problem:** `config.training.out_channels` was added to the config YAML but is never read. All four call sites hardcode `out_channels=3`:

```python
# train_member.py
full_model = load_full_swinunetr(..., out_channels=3, ...)
model = LoRAOriginalDecoderModel(..., out_channels=3, ...)

# ensemble_inference.py (same pattern)
full_model = load_full_swinunetr(..., out_channels=3, ...)
model = LoRAOriginalDecoderModel(..., out_channels=3, ...)
```

**Fix:** Replace with `config.training.out_channels` (or `config.training.get("out_channels", 3)` for backward compatibility). Also propagate to `C = 3` on line 209 of `ensemble_inference.py`.

---

## 🟠 BUG-4: `evaluate_baseline.py` — Misleading column name `volume_ensemble`

**File:** `engine/evaluate_baseline.py`, line ~111

**Problem:** The baseline evaluation saves the predicted volume under the column `volume_ensemble`, which is semantically wrong — this is the baseline's volume, not the ensemble's. When `statistical_analysis.py` loads both `ensemble_test_dice.csv` and `baseline_test_dice.csv`, both have a column called `volume_ensemble`, which is confusing.

**Fix:** Rename to `volume_pred` in the baseline CSV. Update any downstream code that references this column name (currently only informational — statistical tests use Dice columns, not volume).

---

## 🟡 BUG-5: `run_evaluate.py` — Calibration metrics declared but never computed

**File:** `run_evaluate.py`, line ~142

**Problem:** The calibration section is initialized as empty and never populated:

```python
stats_summary["calibration"] = {}
```

The original v1.0 `run_evaluate.py` had full calibration computation (ECE, Brier, reliability). The v1.1 rewrite created the statistical analysis pipeline but dropped the calibration computation entirely. The `compute_ece`, `compute_brier_score`, and `compute_reliability_data` functions are imported but never called.

**Fix:** Add a calibration computation step in `run_full_evaluation()`. This requires running ensemble inference on the test set with ground truth labels available. The most efficient approach: during Step 2 (ensemble evaluation), collect the ensemble `mean_probs` and the `binary_gt` per scan, subsample voxels, and compute calibration metrics. Then insert the results into `stats_summary["calibration"]`.

Alternatively, create a dedicated `compute_calibration()` function in `engine/evaluate_members.py` that takes the ensemble predictor and test dataset, and returns ECE, Brier, and reliability data. Add this as Step 4b in the evaluation pipeline.

---

## 🟡 BUG-6: Missing `volume_from_ensemble_mask` in volume CSV

**File:** `engine/volume_extraction.py`

**Problem:** The CSV contains `vol_mean` (average of per-member volumes) but not the volume derived from the ensemble consensus mask. These are mathematically different:

- `vol_mean = (1/M) Σ_m V^(m)}` — mean of per-member hard-mask volumes
- `vol_ensemble_mask = Σ_v 𝟙[p̄_{v,WT} > 0.5] · δ³` — volume from the averaged-then-thresholded mask

The ensemble mask volume is what a clinician would use (single deterministic prediction), while `vol_mean` is the statistical summary. Both should be in the CSV.

**Fix:** Add after line ~128:

```python
row["vol_ensemble_mask"] = float(result.ensemble_mask.sum().item())
row["logvol_ensemble_mask"] = math.log(float(result.ensemble_mask.sum().item()) + 1)
```

---

## 🟡 BUG-7: `convergence_analysis.py` — NaN at k=1 propagates to summary

**File:** `engine/convergence_analysis.py`, `compute_convergence_summary()`

**Problem:** At k=1, `running_se` and `running_std` are `NaN` (undefined for a single sample). When `groupby("k").agg(mean_running_se=("running_se", "mean"))` encounters the k=1 group (all NaNs), it produces NaN. The log message then prints `nan`.

**Fix:** In `compute_convergence_summary()`, filter k >= 2 before aggregation, or set k=1 values to 0 with a note. The simplest fix:

```python
# After stacked = pd.concat(...)
stacked = stacked[stacked["k"] >= 2]  # SE undefined for k=1
```

And adjust the log message to use `iloc[0]` (which is now k=2) or handle gracefully.

---

## ⚪ BUG-8: `run_evaluate.py` — Redundant `import pandas as pd`

**File:** `run_evaluate.py`, lines ~88 and ~118

**Problem:** `import pandas as pd` appears inside two conditional blocks, but `pd` is already imported at the module top level. Harmless but untidy.

**Fix:** Remove the two inner `import pandas as pd` statements.

---

## Summary of changes required

| Bug | File(s) | Severity | Lines of change |
|-----|---------|----------|-----------------|
| BUG-1 | `run_inference.py` | 🔴 Crash | ~5 |
| BUG-2 | `engine/evaluate_members.py` | 🟠 Wrong results | ~20 per function |
| BUG-3 | `engine/train_member.py`, `engine/ensemble_inference.py` | 🟠 Wrong results | ~8 |
| BUG-4 | `engine/evaluate_baseline.py` | 🟠 Misleading | ~2 |
| BUG-5 | `run_evaluate.py` + new calibration function | 🟡 Missing | ~40 |
| BUG-6 | `engine/volume_extraction.py` | 🟡 Missing | ~3 |
| BUG-7 | `engine/convergence_analysis.py` | 🟡 Cosmetic | ~3 |
| BUG-8 | `run_evaluate.py` | ⚪ Cosmetic | ~2 |

**Priority order:** BUG-1 → BUG-2 → BUG-3 → BUG-6 → BUG-5 → BUG-4 → BUG-7 → BUG-8.
