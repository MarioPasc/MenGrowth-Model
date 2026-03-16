#!/usr/bin/env bash
# Compaction recovery hook — re-injects critical context after /compact

cat <<'CONTEXT'
=== MENGROWTH COMPACTION RECOVERY ===

PROJECT: Meningioma growth prediction from small longitudinal MRI cohorts.
FRAMEWORK: 3-stage complexity ladder. Each stage must beat the previous under LOPO-CV.

STAGE 1 (PRIMARY): Volume-only baseline
  Segment → WT volume → log(V+1) → {ScalarGP, LME, HGP} → LOPO-CV
  Spec: docs/stages/stage_1_volumetric_baseline.md

STAGE 2 (SECONDARY): Latent severity model
  Quantile transform → NLME with s∈[0,1] → monotonic g(s,t;θ) → LOPO-CV
  Spec: docs/stages/stage_2_severity_model.md

STAGE 3 (TERTIARY): Representation learning (old primary pipeline)
  BrainSegFounder → LoRA → SDP → PCA(residual) → GP+ARD → LOPO-CV
  Spec: docs/stages/stage_3_representation_learning.md

VARIANCE DECOMPOSITION: M₀→M₁→M₂→M₃→M₄, ΔR², permutation tests
  Spec: docs/stages/variance_decomposition.md

KEY CONSTRAINT: N=31-58 patients, max 2-3 predictor params at N=31.

CHANNEL ORDER (CRITICAL):
  [FLAIR, T1ce, T1, T2] = ["t2f", "t1c", "t1n", "t2w"]
  Wrong order → Dice ~0.00.

ENVIRONMENT:
  Conda: ~/.conda/envs/growth/bin
  Tests: ~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short

KEY DOCS:
  docs/RESEACH_PLAN.md — Literature synthesis, 3 axes
  docs/PLAN_OF_ACTION_v1.md — Implementation spec, all 3 stages
  docs/stages/ — Self-contained stage specs
  docs/growth-related/claude_files_BSGNeuralODE/ — Stage 3 reference specs

KEY CODE:
  src/growth/models/growth/ — GP models (scalar_gp, lme_model, hgp_model)
  src/growth/evaluation/lopo_evaluator.py — LOPO-CV framework
  experiments/segment_based_approach/ — Stage 1 pipeline

CODING RULES:
  Type hints, Google docstrings, no magic numbers, LayerNorm only,
  shape assertions, logging (no print), prefer library functions.

=== END COMPACTION RECOVERY ===
CONTEXT
