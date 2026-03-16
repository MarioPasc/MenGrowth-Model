# MenGrowth-Model — Foundation Model Adaptation for Meningioma Growth Forecasting

## 1. What This Project Does

BSc thesis pipeline that adapts **BrainSegFounder** (a glioma-pretrained 3D SwinUNETR, 62M params, pretrained on 41K+ subjects) for **meningioma growth forecasting** from multi-modal 3D MRI. Four phases:

1. **LoRA Adaptation** (Phase 1): Inject LoRA (r=8, α=16) into SwinViT Stages 3–4 Q/K/V, train on BraTS-MEN segmentation as proxy task. Auxiliary semantic heads enrich features. Merge LoRA, discard decoder.
2. **Supervised Disentangled Projection** (Phase 2): Map frozen encoder features h ∈ ℝ^768 → structured z ∈ ℝ^128 via 2-layer MLP with spectral normalization. Partitions: z_vol(24) | z_loc(8) | z_shape(12) | z_residual(84). Losses: semantic regression + VICReg covariance + distance correlation. 4-phase curriculum.
3. **Cohort Encoding & ComBat** (Phase 3): Encode Andalusian longitudinal cohort (42 patients, 137 studies) through frozen pipeline. Assess and apply ComBat harmonization. Build per-patient temporal trajectories.
4. **GP-based Growth Prediction** (Phase 4): Three-model hierarchy — LME → Hierarchical GP (Matérn-5/2) → PA-MOGP (ICM, rank-1 coupling). LOPO-CV with 33 folds. Population linear mean from LME as GP prior.

### Pipeline Diagram

```
BraTS-MEN MRI [B,4,128³] → SwinViT (LoRA r=8, Stages 3-4) → Seg Head → L_dice + L_CE + λ·L_sem
                                    ↓ (merge LoRA, freeze)
BraTS-MEN MRI [B,4,128³] → Frozen SwinViT → GAP → h ∈ ℝ^768 → SDP MLP → z ∈ ℝ^128
                                                                              ↓ (freeze SDP)
Andalusian cohort [all t_k] → Frozen SwinViT → GAP → SDP → ComBat? → z* trajectories
                                                                              ↓
                          z*(t_obs) → GP(z_vol | t) → ẑ_vol(t*) ± σ(t*) → π_vol → V̂(t*)
                                    LME → H-GP → PA-MOGP (LOPO-CV, 33 folds)
```

### Current Status

| Phase | Title | Status | Key Output |
|-------|-------|--------|------------|
| 0 | Data Infrastructure | COMPLETE | BraTSDatasetH5, HDF5 v2.0, transforms |
| 1 | LoRA Adaptation | COMPLETE | Merged encoder, Dice WT ~0.87, experiments/lora/ |
| 2 | SDP | COMPLETE | SDP network, losses, curriculum, tests pass |
| 3 | Encoding + ComBat | COMPLETE | Infrastructure ready, pending real data run |
| 4a | LME Growth Model | COMPLETE | statsmodels REML, BLUP prediction |
| 4b | H-GP Growth Model | COMPLETE | GPy, Matérn-5/2, pooled marginal likelihood |
| 4c | PA-MOGP Model | STUBBED | Docstring only — needs implementation |
| 4d | Scalar GP | COMPLETE | 1-D wrapper for volume-only prediction |
| 5 | Evaluation | MOSTLY | GP probes, LOPO-CV, ablation A0 in progress |
| A0 | Segment-Based Baseline | IN DEV | experiments/segment_based_approach/ |

---

## 2. Critical Constants

These are verified values. Use them directly — do not re-derive.

| Constant | Value | Source |
|----------|-------|--------|
| **Channel order** | `["t2f", "t1c", "t1n", "t2w"]` = [FLAIR, T1ce, T1, T2] | `MODALITY_KEYS` in `transforms.py` |
| **ROI (training)** | 128×128×128 | BrainSegFounder fine-tuning convention |
| **ROI (features)** | 192×192×192 | 100% MEN tumor containment |
| **encoder10 output** | `[B, 768, 4, 4, 4]` (128³ input) | SwinUNETR architecture |
| **Seg output** | 3-ch sigmoid: Ch0=TC, Ch1=WT, Ch2=ET | Hierarchical overlapping |
| **Input labels** | 0=BG, 1=NCR, 2=ED, 3=ET (+4=RC for GLI) | BraTS convention |
| **LoRA rank/alpha** | r=8, α=16, target: Stages 3–4 Q/K/V | D2, D3 |
| **LoRA trainable params** | ~197K | Verified |
| **SDP partition** | vol(24) \| loc(8) \| shape(12) \| residual(84) = 128 | D10, module_3 |
| **BSF checkpoint** | `finetuned_model_fold_0.pt` (BSF-Tiny, 62M) | Cox et al. 2024 |
| **Andalusian cohort** | 42 patients, 137 studies, ~10 scanners | Private longitudinal |
| **Hardware (local)** | RTX 4060 Laptop, 8GB VRAM | CPU-only tests, analysis |
| **Hardware (Picasso)** | A100 40GB nodes | Training, evaluation |

### R1 Methodology Revision (2026-03-09)

| Finding | Action |
|---------|--------|
| Shape R² ≤ 0.11 across all conditions | Drop shape partition from SDP |
| Location temporally static for meningiomas | Move to GP covariate |
| Sub-volumes label-dependent, incompatible cross-domain | Use whole-tumor log-volume only |
| LoRA marginal: Δ R² = +0.016 (within noise) | Document; frozen baseline viable |
| Dual-domain transfer: cross-domain R² = −0.01 | Drop dual-domain LoRA |

See `docs/growth-related/domain-gli-men/METHODOLOGY_REVISION_R1.md` for full analysis.

---

## 3. Forbidden Actions

- **DO NOT** modify anything in `src/external/`. Frozen vendored BrainSegFounder code.
- **DO NOT** delete or overwrite `docs/growth-related/claude_files_BSGNeuralODE/`. Master module specs.
- **DO NOT** change channel order from `["t2f", "t1c", "t1n", "t2w"]`. Wrong order → Dice ~0.00.
- **DO NOT** use BatchNorm anywhere. SwinUNETR uses LayerNorm only.
- **DO NOT** use ROI size 96³. Must be 128³ (training) or 192³ (features).
- **DO NOT** start Phase N+1 until Phase N's CRITICAL tests pass.
- **DO NOT** run `rm -rf` or force-push to git.

---

## 4. Phase System

The project is implemented in **6 gated phases** (Phase 0 through Phase 5). **Phase N+1 cannot start until Phase N's CRITICAL tests all pass.** Use `/check-gate N` to verify.

| Phase | Module Spec | Key Output | Gate Tests |
|-------|-------------|------------|------------|
| 0 | `module_0_data.md` | BraTSDatasetH5, splits, transforms | TEST_0.1–0.4 |
| 1 | `module_1_domain_gap.md` + `module_2_lora.md` | Domain gap report, merged encoder | TEST_1.1–1.4, TEST_2.1–2.5 |
| 2 | `module_3_sdp.md` | SDP network + semantic heads, quality report | TEST_3.1–3.6 |
| 3 | `module_4_encoding.md` | Cohort latents, harmonized trajectories | TEST_4.1–4.4 |
| 4 | `module_5_growth_prediction.md` | LME/H-GP/PA-MOGP results, LOPO metrics | TEST_5.1–5.8 |
| 5 | `module_6_evaluation.md` | Final report, ablation results, 13 figures | TEST_6.1–6.4 |

**Before starting any phase**, read its module spec at `docs/growth-related/claude_files_BSGNeuralODE/module_{N}_*.md` and `DECISIONS.md`.

---

## 5. Resource Hub

All paths relative to project root `/home/mpascual/research/code/MenGrowth-Model/`. Agent environment: `~/.conda/envs/growth/` (Python 3.x, PyTorch 2.0+, MONAI 1.3+).

### 5.1 Environment & Execution

| Resource | Path / Command |
|----------|---------------|
| Conda Python | `~/.conda/envs/growth/bin/python` |
| Run all fast tests | `~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short` |
| Run phase tests | `~/.conda/envs/growth/bin/python -m pytest -m phase{N} -v --tb=short` |

### 5.2 Code

| Resource | Path |
|----------|------|
| Main package | `src/growth/` |
| Models: encoder | `src/growth/models/encoder/` (swin_loader, lora_adapter, feature_extractor) |
| Models: SDP | `src/growth/models/projection/` (sdp, partition, semantic_heads) |
| Models: growth | `src/growth/models/growth/` (lme_model, hgp_model, pamogp_model, scalar_gp) |
| Data | `src/growth/data/` (bratsmendata, transforms, semantic_features) |
| Losses | `src/growth/losses/` (segmentation, sdp_loss, vicreg, dcor, encoder_vicreg) |
| Evaluation | `src/growth/evaluation/` (gp_probes, latent_quality, lopo_evaluator) |
| Inference | `src/growth/inference/` (sliding_window, harmonization) |
| Training | `src/growth/training/` (Lightning modules, entry points) |
| Configs | `src/growth/config/` (YAML configs) |
| Tests | `tests/growth/` (37 test files, ~370 tests) |

### 5.3 Experiments

| Experiment | Path | Status |
|------------|------|--------|
| LoRA ablation (Phase 1) | `experiments/lora/` | COMPLETE |
| SDP training (Phase 2) | `experiments/sdp/` | COMPLETE |
| Segment-based baseline (A0) | `experiments/segment_based_approach/` | IN DEV |
| Domain gap analysis | `experiments/domain_gap/` | COMPLETE |

### 5.4 Data Conversion Scripts

| Script | Purpose |
|--------|---------|
| `scripts/convert_brats_men_to_h5.py` | NIfTI → HDF5 for BraTS-MEN |
| `scripts/convert_brats_gli_to_h5.py` | NIfTI → HDF5 for BraTS-GLI |
| `scripts/convert_mengrowth_to_h5.py` | NIfTI → HDF5 for MenGrowth |

### 5.5 When to Use What

| You want to... | Use | Trigger |
|----------------|-----|---------|
| Implement a pipeline phase | `/implement-phase N` | "implement phase 3", "build the SDP" |
| Run tests after code changes | `/test` or `/run-tests N` | "run tests", "check phase 2" |
| Validate before SLURM submission | `/pre-flight <config>` | "ready to train?", "check config" |
| Analyze completed results | `/analyze-run <dir>` | "analyze results", "how did training go?" |
| Scientific analysis of metrics | `/dl-scientist` | "why is R² low?", "root cause analysis" |
| Check if a phase gate is open | `/check-gate N` | "is phase 2 done?", "can I start phase 3?" |
| Explore codebase | `/explore` | "how does X work?", "where is Y?" |
| Refactor a module | `/refactor <file>` | "clean up this module", "add type hints" |

### 5.6 Slash Commands

| Command | Usage | What it does |
|---------|-------|-------------|
| `/implement-phase` | `/implement-phase 3` | Launches phase-implementer (Opus) for end-to-end phase work |
| `/run-tests` | `/run-tests 2` | Launches test-runner (Haiku) for phase verification |
| `/check-gate` | `/check-gate 1` | Reads verification report, reports OPEN/BLOCKED |
| `/pre-flight` | `/pre-flight experiments/lora/config/server/lora_semantic.yaml` | Validates config + code before SLURM |
| `/analyze-run` | `/analyze-run /path/to/results` | Post-experiment analysis vs quality targets |
| `/dl-scientist` | `/dl-scientist <results>` | Rigorous analysis with literature grounding |
| `/explore` | `/explore how does SDP loss work?` | Codebase exploration in isolated context |
| `/refactor` | `/refactor src/growth/models/growth/lme_model.py` | Production-quality refactoring |
| `/test` | `/test -m phase1` | Quick test runner |

### 5.7 Subagents

| Agent | Model | Purpose |
|-------|-------|---------|
| `phase-implementer` | Opus | Reads module spec, implements code + tests, creates verification report |
| `test-runner` | Haiku | Runs pytest, reports pass/fail, checks gate status |
| `results-analyst` | Opus | Analyzes results vs quality targets, proposes improvements |
| `pre-flight-validator` | Opus | 8-check validation before expensive GPU runs |

---

## 6. Documentation Index

> **IMPORTANT — Selective Reading:** Do NOT read all documents at once. Before starting a task, pick only the 1-3 documents directly relevant to your current work.

### 6.1 Master References (large files, read sections on demand)

| Document | Path | Contents | When to read |
|----------|------|----------|-------------|
| Master Methodology | `docs/growth-related/methodology_refined.md` | Full pipeline theory, math, data contracts, all 6 sections | When you need deep mathematical grounding beyond module specs |
| Design Decisions | `docs/growth-related/claude_files_BSGNeuralODE/DECISIONS.md` | 20 pre-resolved choices (D1–D20) | Before any design decision — check if already resolved |
| Technical Report | `docs/technical_report/main.tex` | LaTeX thesis hub → `sections/*.tex` | When writing thesis content or generating figures |

### 6.2 Module Specifications (one per phase, self-contained)

Read the spec for the phase you are working on. Each spec has: input/output contracts, code requirements, verification tests, recovery steps.

| Phase | Spec | Contents |
|-------|------|----------|
| 0 | `docs/growth-related/claude_files_BSGNeuralODE/module_0_data.md` | Dataset classes, splits, transforms, semantic features |
| 1 (gap) | `docs/growth-related/claude_files_BSGNeuralODE/module_1_domain_gap.md` | MMD, CKA, linear probes, UMAP |
| 1 (LoRA) | `docs/growth-related/claude_files_BSGNeuralODE/module_2_lora.md` | LoRA injection, training, merge, Dice |
| 2 | `docs/growth-related/claude_files_BSGNeuralODE/module_3_sdp.md` | SDP architecture, losses, curriculum, partition |
| 3 | `docs/growth-related/claude_files_BSGNeuralODE/module_4_encoding.md` | Encoding, ComBat, trajectory building |
| 4 | `docs/growth-related/claude_files_BSGNeuralODE/module_5_growth_prediction.md` | LME, H-GP, PA-MOGP, LOPO-CV, volume decoder |
| 5 | `docs/growth-related/claude_files_BSGNeuralODE/module_6_evaluation.md` | Quality targets, ablations A1-A10, 13 figures |

### 6.3 Supplementary Documents

| Topic | Path | When to read |
|-------|------|-------------|
| R1 Methodology Revision | `docs/growth-related/domain-gli-men/METHODOLOGY_REVISION_R1.md` | When working on SDP partition design or GP models post-revision |
| Neural ODE → GP Pivot | `docs/growth-related/gaussian-process/phase4_pivot_to_gp_models.md` | When implementing Phase 4 or justifying GP choice |
| SDP Justification | `docs/growth-related/BrainSegFounder-Adaptation-related/SDP_justification.md` | When working on Phase 2 theory |
| Dual-Domain Analysis | `docs/growth-related/domain-gli-men/doc1_lora_dual_domain_phase.md` | When investigating dual-domain results |
| BrainSegFounder Code | `docs/growth-related/papers/insights_from_the_code.md` | When debugging encoder/checkpoint issues |
| Segment-Based Baseline | `docs/growth-related/segment_based_approach/first_iteration.md` | When working on ablation A0 |
| LoRA Ablation Analysis | `docs/growth-related/BrainSegFounder-Adaptation-related/new-comprehensive-analysis-lora-ablation.md` | When analyzing Phase 1 results |

### 6.4 Deprecated (historical reference only)

All deprecated docs moved to `docs/deprecated/`. Includes: VAE approach (abandoned — posterior/residual collapse), early design docs, completed refactoring specs, old progress reports, miscellaneous findings.

---

## 7. Coding Standards

Full standards in `.claude/rules/coding-standards.md` (auto-loaded). The essentials:

- **Type hints** on all function signatures and return types.
- **Google-style docstrings** on all public functions/classes.
- **No magic numbers** — all hyperparameters from YAML configs via OmegaConf.
- **Prefer library functions:** MONAI transforms, `einops.rearrange`, `F.scaled_dot_product_attention`.
- **No BatchNorm** — use LayerNorm only (SwinUNETR convention).
- **Shape assertions** at tensor function boundaries.
- **Logging** via Python `logging` module with `rich` handler (no `print`).
- **Tests:** pytest with markers (`phase0`–`phase2`, `evaluation`, `experiment`, `unit`, `slow`, `real_data`).

---

## 8. Testing

- **Framework:** pytest via `~/.conda/envs/growth/bin/python -m pytest`
- **Markers:** `phase0`, `phase1`, `phase2`, `evaluation`, `experiment`, `unit`, `slow`, `real_data` (in `pyproject.toml`)
- **Gating:** A phase gate is OPEN when all its CRITICAL tests pass
- **Safe default:** `pytest -m "not slow and not real_data" -v --tb=short` (~2 min, 370 tests)
- **Full suite:** `pytest -v --tb=short` (~20 min, includes slow convergence tests)
- **By phase:** `pytest -m phase1 -v --tb=short`
- **Combine:** `pytest -m "phase1 and unit" -v --tb=short`

### What to Run After Editing

| Files changed in... | Run |
|---------------------|-----|
| `src/growth/data/` | `pytest -m phase0` |
| `src/growth/models/encoder/`, `src/growth/losses/`, `experiments/lora/` | `pytest -m phase1` |
| `src/growth/models/projection/`, `experiments/sdp/` | `pytest -m phase2` |
| `src/growth/evaluation/` | `pytest -m evaluation` |
| `experiments/lora/analysis/`, `experiments/lora/vis/` | `pytest -m experiment` |
| Unknown / broad changes | `pytest -m "not slow and not real_data"` |

---

## 9. Quick Reference: What to Read per Phase

| Phase | Must read | Useful if stuck |
|-------|-----------|----------------|
| 0 | `module_0_data.md` | `methodology_refined.md` §0 |
| 1 | `module_1_domain_gap.md`, `module_2_lora.md`, `DECISIONS.md` D1-D7 | `insights_from_the_code.md` |
| 2 | `module_3_sdp.md`, `DECISIONS.md` D5,D10-D14 | `SDP_justification.md`, `METHODOLOGY_REVISION_R1.md` |
| 3 | `module_4_encoding.md` | `methodology_refined.md` §4 |
| 4 | `module_5_growth_prediction.md`, `DECISIONS.md` D16-D20 | `phase4_pivot_to_gp_models.md` |
| 5 | `module_6_evaluation.md` | All prior module results |

---

## 10. Scientific Mindset

- Approach every task as a world-class deep learning scientist: reason step by step, justify with literature and math.
- **Do NOT please the user.** If something has theoretical flaws or is scientifically incorrect, say so.
- Every non-trivial decision must cite: a paper, a mathematical justification, or empirical data.
- Be proactive: if you identify a method or trick that could improve the approach, propose it with rationale.
- Distinguish statistical significance from practical significance.
- Prioritize correctness over speed. Every algorithm must be mathematically justified.
- When uncertain, quantify uncertainty. "This might work" → "This has ~X% chance based on [reasoning]."

### Research Workflow: Plan → Test → Analyze → Fix

1. **Plan:** Break task into checkable items. For each: objective, success metric, references.
2. **Test:** Define quantitative success criteria before running. Log all seeds and hyperparameters.
3. **Analyze:** Compute mean, std, confidence intervals. Report anomalies with evidence.
4. **Fix:** Fixes must reference what analysis revealed. Re-run to confirm. No blind patches.

---

## 11. HDF5 Dataset Schema (v2.0)

All three datasets (BraTS-MEN, BraTS-GLI, MenGrowth) use `BraTSDatasetH5` with unified schema:

```
attrs:           {n_scans, n_patients, roi_size, spacing, channel_order, version="2.0", dataset_type, domain}
images           [N_scans, 4, 192, 192, 192] float32
segs             [N_scans, 1, 192, 192, 192] int8
scan_ids         [N_scans] str
patient_ids      [N_scans] str
timepoint_idx    [N_scans] int32
semantic/        {volume [N,4], location [N,3], shape [N,3]}
longitudinal/    {patient_offsets [N_patients+1], patient_list [N_patients]}
splits/          {lora_train, lora_val, test}
metadata/        {grade, age, sex}
```

Convert NIfTI → H5: `python scripts/convert_brats_men_to_h5.py --data-root /path/to/BraTS_Men_Train --output brats_men_train.h5`

---

## 12. Key Libraries

PyTorch 2.0+, MONAI 1.3+, Lightning 2.0+, OmegaConf, peft (LoRA/DoRA), GPy>=1.13, statsmodels>=0.14, scipy, rich

## 13. Legacy Note

The VAE approach (Exp1–3, `src/vae/`) is preserved in `docs/deprecated/vae-related/` for historical reference. Abandoned due to posterior collapse, residual collapse, and KL distortion. The Neural ODE approach (original Phase 4) was also abandoned due to catastrophic overparameterization (33 patients, 112 pairs, ~3100 params). See `docs/growth-related/gaussian-process/phase4_pivot_to_gp_models.md`.
