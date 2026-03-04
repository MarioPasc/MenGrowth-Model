# LoRA Experiment Config Key Audit Report

**Date:** 2026-03-04  
**Scope:** `experiments/lora/` — all 18 YAML configs vs all 10 Python modules  
**Methodology:** Exhaustive cross-reference of every `config[...]` (hard access, KeyError if missing) and `.get(...)` (safe with default) in Python code against every YAML config key

---

## Executive Summary

| Category | Count |
|----------|-------|
| **KeyError crashes** | **3 missing keys in `smoke_dual_r8.yaml`** |
| **KeyError crashes (legacy configs)** | **6 configs missing `paths.h5_file`** |
| **API mismatches (TypeError)** | **1 in `extract_features.py`** |
| **Dead config keys** | **9 distinct keys, present across all 18 configs** |

---

## PART A: Required Config Keys (Accessed With `[]`)

These keys are accessed with bare subscript — a missing key is a guaranteed runtime crash.

### A.1 All Code Access Points

**`experiment` section:**
| Key | Code File | Line(s) | Notes |
|-----|-----------|---------|-------|
| `experiment.output_dir` | `train_condition.py`, `run.py`, `extract_features.py`, `evaluate_dice.py`, `evaluate_probes.py`, `evaluate_domain_gap.py`, `evaluate_feature_quality.py`, `data_splits.py`, `generate_tables.py` | Many | Universal, all configs have it ✅ |
| `experiment.seed` | `train_condition.py`, `run.py`, `evaluate_probes.py`, `evaluate_domain_gap.py`, `evaluate_feature_quality.py`, `data_splits.py` | Many | Universal ✅ |

**`paths` section:**
| Key | Code File | Line | When Required |
|-----|-----------|------|---------------|
| `paths.checkpoint` | `train_condition.py` L1054, `extract_features.py` L180, `evaluate_dice.py` L508 | Always |
| `paths.h5_file` | `data_splits.py` L117 | Single-domain only (`run-all` → `splits`) |
| `paths.men_h5_file` | `train_condition.py` L1021 | Dual-domain only (inside `is_dual` guard) |
| `paths.gli_h5_file` | `train_condition.py` L1022 | Dual-domain only (inside `is_dual` guard) |

**`training` section:**
| Key | Code File | Line | Notes |
|-----|-----------|------|-------|
| `training.lr_encoder` | `train_condition.py` L276 | Non-baseline conditions only |
| `training.lr_decoder` | `train_condition.py` L266/268/276 | All conditions |
| `training.weight_decay` | `train_condition.py` L280 | All conditions |
| `training.batch_size` | `train_condition.py` L1029/1040 | Always |
| `training.num_workers` | `train_condition.py` L1024/1044, `evaluate_dice.py` L523 | Always |
| `training.max_epochs` | `train_condition.py` L1089 | Always |
| `training.early_stopping_patience` | `train_condition.py` L1090 | Always |

**`loss` section:**
| Key | Code File | Line | Notes |
|-----|-----------|------|-------|
| `loss.lambda_dice` | `train_condition.py` L542/1070 | Always |
| `loss.lambda_ce` | `train_condition.py` L543/1071 | Always |

**`conditions` section:**
| Key | Code File | Notes |
|-----|-----------|-------|
| `conditions` (list) | `run.py` L134, `model_factory.py` L512, all eval files | Always |
| `conditions[*].name` | `model_factory.py` L515, all iteration | Always |

**`data_splits` section (single-domain only):**
| Key | Code File | Line |
|-----|-----------|------|
| `data_splits.lora_train` | `data_splits.py` L125 |
| `data_splits.lora_val` | `data_splits.py` L126 |
| `data_splits.test` | `data_splits.py` L128 |

---

## PART B: Potential KeyError Crashes

### B.1 `picasso/smoke_dual_r8.yaml` — 3 MISSING REQUIRED KEYS

| Missing Key | Accessed At | Impact |
|-------------|-------------|--------|
| `training.weight_decay` | `train_condition.py` L280 — `AdamW(..., weight_decay=training_config["weight_decay"])` | **KeyError** during optimizer creation |
| `training.early_stopping_patience` | `train_condition.py` L1090 — `patience = training_config["early_stopping_patience"]` | **KeyError** before training loop |
| `loss` (entire section) | `train_condition.py` L1070 — `config["loss"]["lambda_dice"]` | **KeyError** during loss function creation |

**Recommended fix** — add to `smoke_dual_r8.yaml`:
```yaml
training:
  # ...existing keys...
  weight_decay: 1.0e-5
  early_stopping_patience: 10

loss:
  lambda_dice: 1.0
  lambda_ce: 1.0
  lambda_volume: 1.0
  lambda_location: 0.3
  lambda_shape: 1.0
```

### B.2 Legacy Configs Missing `paths.h5_file` — 6 FILES

These configs predate the H5 migration and only have `paths.data_root`. Running `run-all` or `splits` commands will crash with `KeyError: 'h5_file'` in `data_splits.py` L117.

| Config File | Has `data_root` | Has `h5_file` |
|-------------|:-:|:-:|
| `config/ablation.yaml` | ✅ | ❌ |
| `config/ablation_v3.yaml` | ✅ | ❌ |
| `server/LoRA_semantic_heads_icai.yaml` | ✅ | ❌ |
| `server/LoRA_no_semantic_heads_icai.yaml` | ✅ | ❌ |
| `server/DoRA_semantic_heads_icai.yaml` | ✅ | ❌ |
| `server/DoRA_no_semantic_heads_icai.yaml` | ✅ | ❌ |

All `local/` and `picasso/` configs have `h5_file` ✅.  
`dual_domain_v1.yaml` uses `men_h5_file`/`gli_h5_file` (correct for dual-domain) ✅.

**Additional impact:** Even if `data_splits` is skipped, `train_condition.py` L1038 does `config["paths"].get("h5_file")` which returns `None` for these configs, causing downstream NoneType errors in `create_dataloaders()`.

---

## PART C: API Mismatch Bug

### C.1 `extract_features.py` — `load_splits()` Wrong Call Signature

**Location:** `extract_features.py` ~L609 (single-domain feature extraction path)

```python
# ACTUAL CALL:
splits = load_splits(None, config=config)

# EXPECTED SIGNATURE (data_splits.py):
def load_splits(config_path: str) -> dict[str, list[str]]:
```

**Problem:** `load_splits()` accepts one positional arg `config_path` (a file path string). Passing `config=config` as a keyword argument raises:
```
TypeError: load_splits() got an unexpected keyword argument 'config'
```

**Impact:** Single-domain feature extraction is broken. Dual-domain path is unaffected (it uses H5 file splits directly, not `load_splits`).

**Recommended fix:**
```python
# Option A: Pass config_path instead
splits = load_splits(config_path)

# Option B: Use load_splits_h5 if H5 splits are preferred
splits = load_splits_h5(config["paths"]["h5_file"])
```

---

## PART D: Dead Config Keys (Present But Never Read)

### D.1 Universal Dead Keys (present in ALL or most configs, never read by any code)

| Dead Key | Present In | Reason Never Read |
|----------|-----------|-------------------|
| `paths.data_root` | 14 of 18 configs | Legacy NIfTI path. All loaders use `h5_file` now. No code accesses `config["paths"]["data_root"]` |
| `data.roi_size` | ALL 18 configs | Hardcoded as `DEFAULT_ROI_SIZE = (128,128,128)` in `transforms.py`. Never read from config |
| `data.feature_roi_size` | ALL 18 configs | Hardcoded as `FEATURE_ROI_SIZE = (192,192,192)` in `transforms.py`. Never read from config |
| `data.spacing` | 14 single-domain configs | Not accessed by any code |
| `training.lora_dropout` | 16 configs | Dropout is hardcoded to `0.1` in `model_factory.py` L494: `dropout=0.1` |
| `probe.normalize_features` | ALL 18 configs | Only `probe.normalize_targets` is read. `normalize_features` is passed to `GPSemanticProbes` constructor which doesn't have such a param |
| `logging.log_every_n_steps` | ALL 18 configs | Not a Lightning trainer — no code reads this |
| `logging.val_check_interval` | ALL 18 configs | Same — not a Lightning trainer |
| `model_card` | ALL 18 configs | Empty section, never read |

### D.2 Config-Specific Dead Keys

| Dead Key | Config File | Reason |
|----------|-------------|--------|
| `feature_extraction.levels` (plural `s`) | `picasso/smoke_dual_r8.yaml` | Code reads `feature_extraction.level` (singular). The misnamed key is silently ignored; code falls back to default `"encoder10"` |
| `conditions[*].use_semantic_heads` | `picasso/smoke_dual_r8.yaml` | Code reads `use_semantic_heads` exclusively from `config["training"]`, never from individual condition configs |

---

## PART E: Safe-Defaulted Keys (Present in Some Configs, Safely Absent in Others)

These keys are accessed with `.get()` and have sensible defaults. Their absence is NOT a bug, but they cause behavior differences between configs.

| Key | Default | Present In | Absent From |
|-----|---------|-----------|-------------|
| `training.val_batch_size` | `1` | `dual_domain_v1.yaml` only | All other configs |
| `training.lr_warmup_epochs` | `5` | v3 configs, dual_domain_v1 | ablation.yaml, all ICAI configs |
| `training.lr_reduce_factor` | `0.5` | v3 configs, dual_domain_v1 | ablation.yaml, all ICAI configs |
| `training.lr_reduce_patience` | `10` | v3 configs, dual_domain_v1 | ablation.yaml, all ICAI configs |
| `training.use_amp` | `False` | v3 + picasso configs | ablation.yaml, server/ configs |
| `training.grad_accum_steps` | `1` | v3 + picasso configs | ablation.yaml, server/ configs |
| `training.lambda_var_enc` | `5.0` | v3 configs | ICAI configs |
| `training.lambda_cov_enc` | `1.0` | v3 configs | ICAI configs |
| `training.vicreg_gamma` | `1.0` | v3 configs | ICAI configs |
| `training.aux_warmup_epochs` | `0` | Most configs | ablation.yaml |
| `training.aux_warmup_duration` | `10` | Most configs | ablation.yaml |
| `training.freeze_decoder` | `False` | All recent configs | ablation.yaml |
| `data.domain_ratio` | `0.5` | dual-domain configs | single-domain configs |
| `feature_extraction.pooling_mode` | `"gap"` | v3 configs | ICAI configs |
| `loss.lambda_volume` | `1.0` | All configs | Would default if absent |
| `loss.lambda_location` | `1.0` | All configs | Would default if absent |
| `loss.lambda_shape` | `0.5` | All configs | Would default if absent |

---

## PART F: Per-Config Verdict Summary

| Config File | KeyError Risk | Dead Keys | Status |
|-------------|:---:|:---:|--------|
| **picasso/smoke_dual_r8.yaml** | ❌ 3 | 4 | **BROKEN — will crash** |
| picasso/v3_rank_sweep.yaml | ✅ | 9 | OK |
| picasso/LoRA_semantic_heads_icai.yaml | ✅ | 9 | OK |
| picasso/LoRA_no_semantic_heads_icai.yaml | ✅ | 9 | OK |
| picasso/DoRA_semantic_heads_icai.yaml | ✅ | 9 | OK |
| picasso/DoRA_no_semantic_heads_icai.yaml | ✅ | 9 | OK |
| **config/ablation.yaml** | ❌ 1 | 10 | **BROKEN — missing h5_file** |
| **config/ablation_v3.yaml** | ❌ 1 | 10 | **BROKEN — missing h5_file** |
| config/dual_domain_v1.yaml | ✅ | 4 | OK |
| **server/LoRA_semantic_heads_icai.yaml** | ❌ 1 | 9 | **BROKEN — missing h5_file** |
| **server/LoRA_no_semantic_heads_icai.yaml** | ❌ 1 | 9 | **BROKEN — missing h5_file** |
| **server/DoRA_semantic_heads_icai.yaml** | ❌ 1 | 9 | **BROKEN — missing h5_file** |
| **server/DoRA_no_semantic_heads_icai.yaml** | ❌ 1 | 9 | **BROKEN — missing h5_file** |
| local/LoRA_semantic_heads_icai.yaml | ✅ | 9 | OK |
| local/LoRA_no_semantic_heads_icai.yaml | ✅ | 9 | OK |
| local/DoRA_semantic_heads_icai.yaml | ✅ | 9 | OK |
| local/DoRA_no_semantic_heads_icai.yaml | ✅ | 9 | OK |
| local/lora_v3_rank_sweep_local.yaml | ✅ | 9 | OK |

---

## PART G: Recommended Actions (Priority Order)

1. **FIX NOW:** Add `training.weight_decay`, `training.early_stopping_patience`, and `loss` section to `picasso/smoke_dual_r8.yaml`
2. **FIX NOW:** Fix `extract_features.py` `load_splits(None, config=config)` call to use proper signature
3. **FIX SOON:** Add `paths.h5_file` to all 4 `server/` configs (or archive them as deprecated)
4. **CONSIDER:** Archive `config/ablation.yaml` and `config/ablation_v3.yaml` (legacy pre-H5 configs)
5. **CLEANUP:** Fix `feature_extraction.levels` → `level` in `smoke_dual_r8.yaml`
6. **CLEANUP:** Remove condition-level `use_semantic_heads` from `smoke_dual_r8.yaml` (it's read from `training` only)
7. **LOW PRIORITY:** Remove or comment dead keys (`data.roi_size`, `data.feature_roi_size`, `data.spacing`, `paths.data_root`, `training.lora_dropout`, `probe.normalize_features`, `logging.*`, `model_card`) from all configs, or add code to actually read them for consistency
