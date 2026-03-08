# MenGrowth-Model

Foundation model pipeline for **meningioma growth forecasting** from multi-modal 3D MRI.
BrainSegFounder (SwinUNETR, pretrained on 41K+ subjects) → LoRA adaptation → Supervised Disentangled Projection → GP-based growth prediction (LME → H-GP → PA-MOGP).
B.Sc. thesis project.

>[!IMPORTANT] For full context of the project (only read if asked to gather all context for a complex task), read the file: `docs/growth-related/methodology_refined.md`

## Scientific Mindset

- Approach every task as a world-class deep learning scientist: think step by step,
  reason, and justify decisions with literature references and mathematical rigor.
- Do NOT please the user. If something won't work, has theoretical flaws, or is
  scientifically incorrect -- say so. We are doing serious research.
- Be proactive and creative. If a task sparks a connection to another concept,
  report it to the user if it could enhance the research.
- When generating plans for local agents, ensure: (1) the agent has access to
  local code and will know the implementation; (2) provide theoretical background
  so the agent can validate; (3) deliver testable results from code being
  implemented; (4) respect strict folder and code organization for maintainability.
- Prioritize correctness over speed. Every algorithm must be mathematically justified.

## Environment

- Conda: `~/.conda/envs/growth/bin`
- Python: `~/.conda/envs/growth/bin/python`
- Tests: `~/.conda/envs/growth/bin/python -m pytest tests/ -v`

## Pipeline Phases

Phase order: 1 → 2 → 3 → 4. Do NOT start a phase until predecessors pass BLOCKING tests.

- **Phase 1 (LoRA Adaptation)**: COMPLETE — code in `experiments/lora/`
- **Phase 2 (SDP)**: STUBBED — needs implementation per module_3_sdp spec
- **Phase 3 (Encoding + ComBat)**: NOT STARTED — per module_4_encoding spec
- **Phase 4 (Growth Prediction)**: STUBBED — per module_5_growth_prediction spec (LME → H-GP → PA-MOGP, LOPO-CV)

## Dataset: HDF5

The pipeline uses a single HDF5 backend (`BraTSDatasetH5`, alias `BraTSMENDatasetH5`). All three H5 files use the **unified v2.0 schema**:

```
attrs:           {n_scans, n_patients, roi_size, spacing, channel_order, version="2.0", dataset_type, domain}
images           [N_scans, 4, 192, 192, 192] float32
segs             [N_scans, 1, 192, 192, 192] int8
scan_ids         [N_scans] str
patient_ids      [N_scans] str
timepoint_idx    [N_scans] int32
semantic/        {volume [N,4], location [N,3], shape [N,3]}
longitudinal/    {patient_offsets [N_patients+1], patient_list [N_patients]}
splits/          {lora_train, lora_val, test}  (patient-level indices into patient_list)
metadata/        {grade, age, sex}
```

For cross-sectional MEN data, the longitudinal structure is trivial: each patient has 1 scan at timepoint 0, so `n_scans == n_patients` and `patient_offsets = [0, 1, ..., N]`. The reader code path is identical for all datasets.

Legacy v1.0 MEN files (with `subject_ids` instead of `scan_ids`) are still auto-detected and supported.

All three datasets (BraTS-MEN, BraTS-GLI, MenGrowth) use `BraTSDatasetH5` — no NIfTI loaders remain.

Convert NIfTI → H5:
- MEN: `python scripts/convert_brats_men_to_h5.py --data-root /path/to/BraTS_Men_Train --output brats_men_train.h5`
- GLI: `python scripts/convert_brats_gli_to_h5.py --data-root /path/to/BraTS_GLI --output brats_gli.h5`
- MenGrowth: `python scripts/convert_mengrowth_to_h5.py --data-root /path/to/MenGrowth-2025 --output mengrowth.h5`

Images are spatially preprocessed but NOT intensity-normalized (normalization at runtime).

H5 transforms: `get_h5_train_transforms()` / `get_h5_val_transforms()` in `transforms.py`. Config key: `paths.h5_file`.

## Critical Conventions (bugs if violated)

- **Channel order**: `["t2f", "t1c", "t1n", "t2w"]` = [FLAIR, T1ce, T1, T2]. Wrong order → Dice ~0.00. Defined in `MODALITY_KEYS` in `src/growth/data/transforms.py`.
- **ROI size**: 128³ (matching BrainSegFounder fine-tuning), NOT 96³. encoder10 output: `[B, 768, 4, 4, 4]`.
- **Segmentation output**: 3-ch sigmoid — Ch0=TC, Ch1=WT, Ch2=ET (hierarchical overlapping, NOT individual labels).
- **Input labels**: 0=background, 1=NCR, 2=ED, 3=ET (+ 4=RC for GLI, merged into NCR for semantic features). Conversion in `segmentation.py._convert_target()`.
- **Preprocessing (H5)**: Volumes pre-preprocessed to 192³. Runtime: NormalizeIntensity → optional RandSpatialCrop(128³) for training.

## Key Libraries

PyTorch 2.0+, MONAI 1.3+, Lightning 2.0+, OmegaConf, peft (LoRA/DoRA), GPy>=1.13, statsmodels>=0.14, scipy

## Codebase Layout

```
src/growth/          # Main pipeline
  config/            # YAML configs (foundation.yaml + phase overrides)
  models/encoder/    # swin_loader, lora_adapter, feature_extractor
  models/projection/ # SDP (sdp.py, partition.py, semantic_heads.py)
  models/growth/     # GP growth models (LME, H-GP, PA-MOGP, volume decoder)
  losses/            # Dice/CE segmentation, SDP composite
  data/              # BraTS-MEN loader, transforms, semantic features
  training/          # Lightning modules + entry points (train_lora, train_sdp)
  evaluation/        # Probes, metrics, visualization
  inference/         # Sliding window, ComBat harmonization

experiments/lora/           # Phase 1 LoRA adaptation (unified single+dual domain)
  engine/                   #   train_condition, extract_features, model_factory, data_splits
  eval/                     #   evaluate_dice, evaluate_probes, evaluate_domain_gap, evaluate_feature_quality
  analysis/                 #   statistical_analysis, generate_tables, v3_cache, v3_figures
  report/                   #   HTML report generator (cli, figures, narrative, html_builder)
  vis/                      #   visualizations, dual_domain_viz
  utils/                    #   output_paths
  config/                   #   YAML configs (ablation, dual_domain, local/server/picasso)
experiments/sdp/            # Phase 2 SDP (feature extraction + training)
scripts/                    # Utilities (convert_brats_men_to_h5.py, convert_brats_gli_to_h5.py, convert_mengrowth_to_h5.py)
slurm/sdp/                  # SLURM jobs for Picasso cluster
src/vae/                    # Legacy VAE code (Exp1-3, superseded)
```

## Module Dependency Chain

```
module_0 (Data) → module_1 (Domain Gap) → module_2 (LoRA) → module_3 (SDP)
→ module_4 (Encoding) → module_5 (Growth Prediction) → module_6 (Evaluation)
```

## Scientific Development Protocol

### 1. Evidence-Grounded Changes
- Every non-trivial decision must cite: a paper, a mathematical justification, or empirical data.
- "I think this is better" is not valid. "This reduces variance because [formula/reference]" is.
- When proposing architectural or methodological changes, state the expected effect and why.
- If no evidence exists, flag it explicitly as a hypothesis and propose a way to test it.

### 2. Research Workflow: Plan → Test → Analyze → Fix
**Planning phase:**
- Break the task into checkable items in `tasks/todo.md`.
- For each item, annotate: objective, success metric, and relevant references.
- Proactively flag: "Based on [paper/method], we could also try X — want me to include it?"
- Write specs before code. Ambiguity in spec = ambiguity in results.

**Testing phase:**
- Define quantitative success criteria before running anything.
- Log all hyperparameters, seeds, and environment details (reproducibility is non-negotiable).
- Use controlled comparisons: change one variable at a time unless explicitly doing ablations.

**Analysis phase:**
- Be proactive: if results reveal an anomaly or improvement opportunity, report it with evidence.
- Propose fixes or enhancements with: (a) what you found, (b) why it matters, (c) what to do.
- Always compute and report: mean, std, confidence intervals or statistical tests where applicable.
- Distinguish between statistically significant and practically significant differences.
- If a metric degrades, investigate root cause before proposing a fix.

**Fixing phase:**
- Fixes must reference what the analysis revealed. No blind patches.
- After fixing, re-run the relevant test to confirm the fix and check for regressions.
- Update `docs/tasks/lessons.md` or task-specific file with the failure mode and the corrective pattern.

### 3. Interdisciplinary Rigor (CS × AI × Biomedicine)
- Code changes: justify with computational complexity, memory, or convergence arguments.
- Model changes: justify with loss landscape, gradient dynamics, or information-theoretic reasoning.
- Clinical/biomedical changes: justify with domain constraints (e.g., anatomical priors, acquisition physics, class imbalance in rare pathologies).
- When in doubt about clinical validity, flag it — do not assume.

### 4. Proactive Scientific Agent Behavior
- During planning and analysis: if you identify a method, paper, or trick that could improve the current approach, **propose it immediately** with a one-line rationale.
- Suggest ablations or controls the user may not have considered.
- If a result contradicts expectations, form a hypothesis and propose a diagnostic experiment.
- Never silently ignore warnings, NaNs, or unexpected distributions — investigate and report.

### 5. Code & Experiment Standards
- All functions: typed, documented (docstring, no usage examples), brief inline comments.
- Prefer libraries over custom implementations. Cite the library and version.
- Logging over print. Use `logging` module with appropriate levels.
- Atomic functions, low cyclomatic complexity, OOP with dataclasses where appropriate.
- Experiment configs: use YAML/JSON, never hardcode hyperparameters in scripts.
- Random seeds must be set and logged. Results must be reproducible.

### 6. Communication Standards
- When reporting results: tables > prose. Include units, dataset split, and N.
- When proposing changes: state the current state, the proposed change, and the expected delta.
- When uncertain: quantify uncertainty. "This might work" → "This has ~X% chance based on [reasoning]."
- Use LaTeX notation for any mathematical expression in documentation or comments.

### 7. Verification & Self-Correction
- Never mark a task done without quantitative evidence it works.
- After any correction from the user: update `docs/tasks/lessons.md` or task-specific file with the pattern.
- Challenge your own proposals before presenting them. Ask: "What could go wrong?"
- If a subagent is used, verify its output — trust but verify.
---

## Error Recovery

- **BLOCKING** test fails: follow recovery steps in module spec, stop if all fail
- **DIAGNOSTIC** test fails: log warning, continue

## Legacy Note

The VAE approach (Exp1–3, `src/vae/`) is preserved for reference. It was abandoned due to posterior collapse, residual collapse, and KL distortion. The SDP approach directly optimizes for downstream growth prediction requirements. The Neural ODE approach (original Phase 4) was also abandoned due to catastrophic overparameterization with only 33 patients / 112 forward pairs — replaced by a GP hierarchy (D16). See `docs/growth-related/phase4_pivot_to_gp_models.md`.

## Detailed Specifications (read on demand)

@docs/growth-related/claude_files_BSGNeuralODE/DECISIONS.md
@docs/growth-related/claude_files_BSGNeuralODE/module_0_data.md
@docs/growth-related/claude_files_BSGNeuralODE/module_1_domain_gap.md
@docs/growth-related/claude_files_BSGNeuralODE/module_2_lora.md
@docs/growth-related/claude_files_BSGNeuralODE/module_3_sdp.md
@docs/growth-related/claude_files_BSGNeuralODE/module_4_encoding.md
@docs/growth-related/claude_files_BSGNeuralODE/module_5_growth_prediction.md
@docs/growth-related/claude_files_BSGNeuralODE/module_6_evaluation.md
