# MenGrowth-Model — Meningioma Growth Prediction from Small Longitudinal MRI Cohorts

## 1. What This Project Does

BSc thesis framework for **meningioma growth prediction** from longitudinal 3D MRI. The central question: *at what level of model complexity does additional sophistication stop improving prediction at N=31–58 patients?*

The framework is a **3-stage complexity ladder** where each stage must earn its place by demonstrably outperforming the previous one under LOPO-CV. A **variance decomposition** quantifies the marginal contribution of each stage.

### The Three Stages

**Stage 1 — Volumetric Baseline (PRIMARY):** Segment tumor with BrainSegFounder → extract whole-tumor volume → model growth with GP/LME on scalar log-volume trajectories. Volume alone captures the dominant clinical signal for meningioma growth (Engelhardt et al. 2023). This is the strong baseline everything else must beat.

**Stage 2 — Latent Severity Model (SECONDARY):** A single latent variable $s_i \in [0,1]$ ("severity") governs each patient's growth trajectory. Formalized as a nonlinear mixed-effects model (NLME) with quantile-transformed growth. Connected to Item Response Theory and reduced Gompertz dynamics (Vaghi et al. 2020). The severity is estimated from baseline features at test time.

**Stage 3 — Representation Learning (TERTIARY):** BrainSegFounder encoder → LoRA adaptation → Supervised Disentangled Projection → PCA-compressed residual features → GP with ARD kernel. This tests whether deep features capture growth-relevant information **beyond volume**. Only justified if it outperforms Stages 1–2.

**Cross-cutting:** Variance decomposition (ΔR² per stage, paired permutation tests, bootstrap CIs) is the **central analytical contribution**.

### Pipeline Diagram

```
STAGE 1 (Volumetric Baseline):
  MenGrowth MRI → BrainSegFounder → Segmentation → WT Volume → log(V+1)
  → {ScalarGP, LME, HGP} on (t, y) trajectories → LOPO-CV → R²_baseline

STAGE 2 (Severity Model):
  Per-patient trajectories → Quantile transform → (t_quantile, q_growth)
  → Joint MLE: {θ, s_i} where g(s,t;θ) is monotonic + g(s,0)=0
  → Test-time: ŝ_new = σ(w^T [log_vol, age, sex, sphericity] + b)
  → LOPO-CV → R²_severity > R²_baseline ?

STAGE 3 (Representation Learning):
  BraTS-MEN MRI → SwinViT (LoRA r=8) → GAP → h ∈ ℝ^768
  → SDP MLP → z ∈ ℝ^128 = [z_vol(32) | z_residual(96)]
  → PCA(z_residual) → [log(V+1), z̃_res] → GP with ARD kernel
  → LOPO-CV → R²_deep > R²_severity ?

VARIANCE DECOMPOSITION:
  M₀(mean) → M₁(ScalarGP) → M₂(HGP) → M₃(Severity) → M₄(Deep+Severity)
  Report: ΔR², p-values, bootstrap CIs per transition
```

### Current Status

| Stage | Component | Status | Key Output |
|-------|-----------|--------|------------|
| 1 | Segmentation pipeline | COMPLETE | experiments/segment_based_approach/ |
| 1 | ScalarGP | COMPLETE | src/growth/models/growth/scalar_gp.py |
| 1 | LME | COMPLETE | src/growth/models/growth/lme_model.py |
| 1 | HGP | COMPLETE | src/growth/models/growth/hgp_model.py |
| 1 | Gompertz mean function | COMPLETE | hgp_model.py (mean_function="gompertz") |
| 1 | Bootstrap CIs | COMPLETE | run_stage1.py + shared/bootstrap.py |
| 1 | H5 trajectory loader | COMPLETE | stage1_volumetric/trajectory_loader.py |
| 1 | Stage 1 orchestrator | COMPLETE | experiments/stage1_volumetric/run_stage1.py |
| 1 | Stage 1 tests | COMPLETE | tests/growth/test_stage1_pipeline.py (33 tests) |
| 1 | **Stage 1 LOPO-CV** | **EVALUATED** | LME R²=0.387 (BSF-adapted best), R²=0.028 (manual). Numbers used WT volume; needs re-run with ET-only target after BraTS-MEN label fix (2026-04-18). |
| 1 | Segmentation comparison | **EVALUATED** | 4 sources × 3 models, decoder-adapted wins |
| 2 | Quantile transform | COMPLETE | stage2_severity/quantile_transform.py |
| 2 | Severity model (MLE) | COMPLETE | stage2_severity/severity_model.py |
| 2 | Severity model (Bayesian) | COMPLETE | stage2_severity/bayesian_severity_model.py |
| 2 | Severity regression head | COMPLETE | stage2_severity/severity_regression.py |
| 2 | Stage 2 tests | COMPLETE | test_stage2_severity.py + test_stage2_bayesian.py (48 tests) |
| 2 | **Stage 2 LOPO-CV** | **GATE FAILED** | R²=-3.54 (ordinal time blocker) |
| 3 | LoRA adaptation | COMPLETE | experiments/lora/ |
| 3 | SDP network | COMPLETE | src/growth/models/projection/ |
| 3 | PCA + ARD GP pipeline | BLOCKED | Stage 2 gate not passed |
| — | Variance decomposition | BLOCKED | Requires all stages evaluated |
| — | Data infrastructure | COMPLETE | BraTSDatasetH5, HDF5 v2.0 |
| — | LOPO-CV evaluator | COMPLETE | src/growth/evaluation/lopo_evaluator.py |

### Key Statistical Constraints

| Constraint | N=31 | N=58 |
|------------|------|------|
| Max predictor parameters | **2–3** | **4–5** |
| GP hyperparameters | Marginal; informative priors needed | Adequate for ML-II |
| LOPO-CV error bars | ±15–20% | ±10–15% |
| Effective N (between-patient) | 31 trajectories | 58 trajectories |

---

## 2. Critical Constants

| Constant | Value | Source |
|----------|-------|--------|
| **Channel order** | `["t2f", "t1c", "t1n", "t2w"]` = [FLAIR, T1ce, T1, T2] | `MODALITY_KEYS` in `transforms.py` |
| **ROI (training)** | 128×128×128 | BrainSegFounder fine-tuning convention |
| **ROI (features)** | 192×192×192 | 100% MEN tumor containment |
| **encoder10 output** | `[B, 768, 4, 4, 4]` (128³ input) | SwinUNETR architecture |
| **Seg output** | 3-ch sigmoid: Ch0=TC(1\|3), Ch1=WT(>0), Ch2=ET(==3) | BSF-aligned BraTS-hierarchical training; disjoint clinical regions derived downstream |
| **Input labels** | 0=BG, 1=NETC, 2=SNFH (edema), 3=ET | BraTS-MEN raw integers |
| **BSF checkpoint** | `finetuned_model_fold_0.pt` (BSF-Tiny, 62M params) | Cox et al. 2024 |
| **Longitudinal cohort** | 31–58 patients, ~3.6 obs/patient, ~10 scanners | Private Andalusian cohort |
| **Max predictors (N=31)** | 2–3 parameters | Riley et al. 2019 (pmsampsize) |
| **Hardware (local)** | RTX 4060 Laptop, 8GB VRAM | CPU-only tests, analysis |
| **Hardware (Picasso)** | A100 40GB nodes | Training, evaluation |

---

## 3. Forbidden Actions

- **DO NOT** modify anything in `src/external/`. Frozen vendored BrainSegFounder code.
- **DO NOT** change channel order from `["t2f", "t1c", "t1n", "t2w"]`. Wrong order → Dice ~0.00.
- **DO NOT** use BatchNorm anywhere. SwinUNETR uses LayerNorm only.
- **DO NOT** use ROI size 96³. Must be 128³ (training) or 192³ (features).
- **DO NOT** start Stage K+1 before demonstrating Stage K results under LOPO-CV.
- **DO NOT** fit PCA or normalize outside LOPO folds (data leakage).
- **DO NOT** use jointly-optimized severity values at test time (leakage — use regression head).
- **DO NOT** run `rm -rf` or force-push to git.

---

## 4. Stage System

The project follows a **3-stage complexity ladder**. Each stage must earn its place: Stage K+1 is only justified if it demonstrably outperforms Stage K under LOPO-CV.

| Stage | Spec | Key Question | Pass Criterion | Gate Status |
|-------|------|-------------|----------------|-------------|
| 1 | `docs/stages/stage_1_volumetric_baseline.md` | How well does volume alone predict growth? | R²_log > 0 with bootstrap CI excluding 0 | **PASS** (R²=0.387 adapted) |
| 2 | `docs/stages/stage_2_severity_model.md` | Does a latent severity improve over volume? | R²_severity > R²_baseline (p < 0.05) | **FAIL** (R²=-3.54) |
| 3 | `docs/stages/stage_3_representation_learning.md` | Do deep features add signal beyond severity? | R²_deep > R²_severity (p < 0.05) | BLOCKED |
| — | `docs/stages/variance_decomposition.md` | What fraction of variance does each stage explain? | ΔR² table with CIs | BLOCKED |

**Before starting any stage**, read its spec document and the `RESEARCH_PLAN.md` + `PLAN_OF_ACTION_v1.md` for full context.

### Stage 3 Reference Material

The old module specs describe the detailed infrastructure for Stage 3 (LoRA, SDP, encoding):

| Component | Reference Spec | Status |
|-----------|---------------|--------|
| Data infrastructure | `docs/growth-related/claude_files_BSGNeuralODE/module_0_data.md` | COMPLETE |
| Domain gap analysis | `docs/growth-related/claude_files_BSGNeuralODE/module_1_domain_gap.md` | COMPLETE |
| LoRA adaptation | `docs/growth-related/claude_files_BSGNeuralODE/module_2_lora.md` | COMPLETE |
| SDP network | `docs/growth-related/claude_files_BSGNeuralODE/module_3_sdp.md` | COMPLETE |
| Cohort encoding | `docs/growth-related/claude_files_BSGNeuralODE/module_4_encoding.md` | INFRASTRUCTURE |
| Growth prediction | `docs/growth-related/claude_files_BSGNeuralODE/module_5_growth_prediction.md` | PARTIAL |
| Evaluation | `docs/growth-related/claude_files_BSGNeuralODE/module_6_evaluation.md` | SUPERSEDED by variance decomposition |
| Design decisions | `docs/growth-related/claude_files_BSGNeuralODE/DECISIONS.md` | D1–D20 still valid for Stage 3 |

---

## 5. Resource Hub

Project root: `/home/mpascual/research/code/MenGrowth-Model/`. Agent environment: `~/.conda/envs/growth/`.

### 5.1 Environment & Execution

| Resource | Path / Command |
|----------|---------------|
| Conda Python | `~/.conda/envs/growth/bin/python` |
| Run all fast tests | `~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short` |
| Run stage tests | See §8 Testing |

### 5.2 Code

| Resource | Path |
|----------|------|
| Main package | `src/growth/` |
| **Shared infrastructure** | `src/growth/shared/` (GrowthModel ABC, LOPO, metrics, bootstrap, covariates, trajectory I/O) |
| **Stage facades** | `src/growth/stages/stage{1,2,3}_*/` (thin re-exports + stage-specific code) |
| Growth models (all stages) | `src/growth/models/growth/` (scalar_gp, lme_model, hgp_model — stage-agnostic) |
| Stage 1 specifics | `src/growth/stages/stage1_volumetric/` (Gompertz mean function) |
| Stage 2 specifics | `src/growth/stages/stage2_severity/` (severity_model, quantile_transform, growth_functions) |
| Stage 3 specifics | `src/growth/stages/stage3_latent/` (facade for encoder, SDP, projection) |
| Encoder/SDP (Stage 3) | `src/growth/models/encoder/`, `src/growth/models/projection/` |
| LOPO evaluator | `src/growth/evaluation/lopo_evaluator.py` |
| Variance decomposition | `src/growth/evaluation/variance_decomposition.py` (NEW) |
| Data | `src/growth/data/` (bratsmendata, transforms, semantic_features) |
| Losses | `src/growth/losses/` (segmentation, sdp_loss, vicreg, dcor) |
| Tests | `tests/growth/` (37+ test files) |

### 5.3 Experiments

| Experiment | Path | Stage | Status |
|------------|------|-------|--------|
| Volumetric baseline | `experiments/stage1_volumetric/` | 1 | EVALUATED (LME R²=0.387 adapted, 0.028 manual) |
| Segmentation comparison | External: see §5.6 | 1 | EVALUATED (4 sources × 3 models) |
| Severity model | `experiments/stage2_severity/` | 2 | GATE FAILED (R²=-3.54, ordinal time) |
| LoRA ablation | `experiments/stage3_latent/lora/` | 3 | COMPLETE |
| SDP training | `experiments/stage3_latent/sdp/` | 3 | COMPLETE |
| Domain gap analysis | `experiments/stage3_latent/domain_gap/` | 3 | COMPLETE |
| Variance decomposition | `experiments/variance_decomposition/` | Cross | BLOCKED |

### 5.6 External Results

Stage 1 results (4-source × 3-model comparison) are stored externally:
```
/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/
  segment_volume_prediction/BrainSegFounder_GP_predict_volume_change/
    model_comparison.json         # Full 12-config ranking
    growth_prediction/{source}/   # Per-source LOPO results
    segmentation/volume_summary.json  # Dice scores per source
```

### 5.4 When to Use What

| You want to... | Use | Trigger |
|----------------|-----|---------|
| Implement a stage | `/implement-stage N` | "implement stage 2", "build severity model" |
| Run tests after code changes | `/test` or `/run-tests` | "run tests", "check tests" |
| Validate before SLURM submission | `/pre-flight <config>` | "ready to train?", "check config" |
| Analyze completed results | `/analyze-run <dir>` | "analyze results", "compare stages" |
| Scientific analysis of metrics | `/dl-scientist` | "why is R² low?", "root cause analysis" |
| Explore codebase | `/explore` | "how does X work?", "where is Y?" |

### 5.5 Subagents

| Agent | Model | Purpose |
|-------|-------|---------|
| `stage-implementer` | Opus | Reads stage spec, implements code + tests, creates verification report |
| `test-runner` | Haiku | Runs pytest, reports pass/fail |
| `results-analyst` | Opus | Analyzes results vs quality targets, proposes improvements |
| `pre-flight-validator` | Opus | Validates config + code before SLURM submission |

---

## 6. Documentation Index

> **Selective Reading:** Do NOT read all documents at once. Pick only the 1-3 docs relevant to your current work.

### 6.1 Research Foundation (read first for any new task)

| Document | Path | Contents | When to read |
|----------|------|----------|-------------|
| Research Plan | `docs/RESEACH_PLAN.md` | Literature synthesis, 3 axes, statistical constraints, staged approach | When you need theoretical grounding or literature references |
| Plan of Action | `docs/PLAN_OF_ACTION_v1.md` | Implementation spec for all 3 stages, code paths, test criteria | When implementing any stage |
| Design Decisions | `docs/growth-related/claude_files_BSGNeuralODE/DECISIONS.md` | 20 pre-resolved choices (D1–D20) | Before any design decision |

### 6.2 Stage Specifications (self-contained, one per stage)

| Stage | Spec | Contents |
|-------|------|----------|
| 1 | `docs/stages/stage_1_volumetric_baseline.md` | ScalarGP, LME, HGP on volume; Gompertz mean; bootstrap CIs |
| 2 | `docs/stages/stage_2_severity_model.md` | NLME with severity; quantile transform; reduced Gompertz; test-time estimation |
| 3 | `docs/stages/stage_3_representation_learning.md` | LoRA selection; SDP residuals; PCA + ARD GP; severity integration |
| — | `docs/stages/variance_decomposition.md` | M₀–M₄ hierarchy; ΔR²; permutation tests; bootstrap CIs |

### 6.3 Stage 3 Reference Material (old module specs, still valid)

| Topic | Path | When to read |
|-------|------|-------------|
| Master Methodology | `docs/growth-related/methodology_refined.md` | Deep mathematical grounding for Stage 3 |
| Data Infrastructure | `docs/growth-related/claude_files_BSGNeuralODE/module_0_data.md` | Dataset classes, HDF5 schema |
| LoRA Adaptation | `docs/growth-related/claude_files_BSGNeuralODE/module_2_lora.md` | LoRA injection details |
| SDP Network | `docs/growth-related/claude_files_BSGNeuralODE/module_3_sdp.md` | SDP architecture, losses |
| Encoding | `docs/growth-related/claude_files_BSGNeuralODE/module_4_encoding.md` | ComBat, sliding window |
| R1 Revision | `docs/growth-related/domain-gli-men/METHODOLOGY_REVISION_R1.md` | Shape/location dropped, volume simplified |
| GP Pivot Rationale | `docs/growth-related/gaussian-process/phase4_pivot_to_gp_models.md` | Neural ODE → GP |

### 6.4 Technical Report

| Document | Path | Contents |
|----------|------|----------|
| LaTeX Hub | `docs/technical_report/main.tex` | `\input{sections/...}` for each chapter |
| Bibliography | `docs/technical_report/references.bib` | 42+ BibTeX entries |

### 6.5 Deprecated

All deprecated docs in `docs/deprecated/`. VAE approach (abandoned), early design docs, old progress reports.

---

## 7. Coding Standards

Full standards in `.claude/rules/coding-standards.md`. The essentials:

- **Type hints** on all function signatures and return types.
- **Google-style docstrings** on all public functions/classes.
- **No magic numbers** — all hyperparameters from YAML configs via OmegaConf.
- **Prefer library functions:** MONAI, einops, GPy, statsmodels, scipy.
- **No BatchNorm** — LayerNorm only.
- **Shape assertions** at tensor function boundaries.
- **Logging** via Python `logging` module (no `print`).

---

## 8. Testing

- **Framework:** pytest via `~/.conda/envs/growth/bin/python -m pytest`
- **Markers:** `phase0`, `phase1`, `phase2`, `evaluation`, `experiment`, `unit`, `slow`, `real_data`
- **Safe default:** `pytest -m "not slow and not real_data" -v --tb=short` (~2 min, 548 tests)
- **By area:** `pytest -m phase1`, `pytest -m evaluation`, etc.

| Files changed in... | Run |
|---------------------|-----|
| `src/growth/data/` | `pytest -m phase0` |
| `src/growth/models/encoder/`, `src/growth/losses/` | `pytest -m phase1` |
| `src/growth/models/projection/` | `pytest -m phase2` |
| `src/growth/models/growth/` | `pytest -m "not slow and not real_data"` |
| `src/growth/evaluation/` | `pytest -m evaluation` |
| Unknown / broad changes | `pytest -m "not slow and not real_data"` |

---

## 9. Quick Reference: What to Read per Stage

| Task | Must read | Useful if stuck |
|------|-----------|----------------|
| Stage 1 implementation | `stage_1_volumetric_baseline.md`, `PLAN_OF_ACTION_v1.md` §1 | `RESEACH_PLAN.md` Axis 1 |
| Stage 2 implementation | `stage_2_severity_model.md`, `PLAN_OF_ACTION_v1.md` §2 | `RESEACH_PLAN.md` Axis 3 |
| Stage 3 implementation | `stage_3_representation_learning.md`, module specs | `RESEACH_PLAN.md` Axis 2, `METHODOLOGY_REVISION_R1.md` |
| Variance decomposition | `variance_decomposition.md`, `PLAN_OF_ACTION_v1.md` §4 | `RESEACH_PLAN.md` §final |
| Data/segmentation | `module_0_data.md` | `methodology_refined.md` §0 |
| LoRA details | `module_2_lora.md`, `DECISIONS.md` D1-D7 | `insights_from_the_code.md` |
| SDP details | `module_3_sdp.md`, `DECISIONS.md` D5,D10-D14 | `SDP_justification.md` |

---

## 10. Scientific Mindset

- Approach every task as a world-class deep learning scientist: reason step by step, justify with literature and math.
- **Do NOT please the user.** If something has theoretical flaws or is scientifically incorrect, say so.
- **Parsimony is paramount at N=31–58.** The bias-variance tradeoff strongly favors simpler models. Any added complexity must demonstrably improve out-of-sample prediction.
- Every non-trivial decision must cite: a paper, a mathematical justification, or empirical data.
- Be proactive: if you identify a method that could improve the approach, propose it with rationale.
- Distinguish statistical significance from practical significance.
- When uncertain, quantify: "This has ~X% chance based on [reasoning]."

### Research Workflow: Plan → Test → Analyze → Fix

1. **Plan:** Break task into checkable items. For each: objective, success metric, references.
2. **Test:** Define quantitative success criteria before running. Log all seeds and hyperparameters.
3. **Analyze:** Compute mean, std, confidence intervals. Report anomalies with evidence.
4. **Fix:** Fixes must reference what analysis revealed. Re-run to confirm. No blind patches.

---

## 11. HDF5 Dataset Schema (v2.0)

```
attrs:           {n_scans, n_patients, roi_size, spacing, channel_order, version="2.0"}
images           [N_scans, 4, 192, 192, 192] float32
segs             [N_scans, 1, 192, 192, 192] int8
scan_ids/patient_ids/timepoint_idx
semantic/        {volume [N,4], location [N,3], shape [N,3]}
longitudinal/    {patient_offsets [N_patients+1], patient_list [N_patients]}
splits/          {lora_train, lora_val, test}
metadata/        {grade, age, sex}
```

---

## 12. Key Libraries

PyTorch 2.0+, MONAI 1.3+, Lightning 2.0+, OmegaConf, peft (LoRA/DoRA), GPy>=1.13, statsmodels>=0.14, scipy>=1.9, scikit-learn, rich, numpyro==0.15.3 + jax==0.4.30 (optional, for Bayesian severity model — pin for GPy/numpy<2.0 compat)

## 13. Legacy Note

The VAE approach (Exp1–3, `src/vae/`) and Neural ODE approach (original Phase 4) are abandoned and documented in `docs/deprecated/`. The full LoRA → SDP → GP pipeline (previously the primary approach) is now **Stage 3** — the last complexity level to test. See `docs/growth-related/gaussian-process/phase4_pivot_to_gp_models.md` for the ODE → GP pivot rationale.
