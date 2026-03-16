#!/usr/bin/env bash
# Compaction recovery hook — re-injects critical context after /compact

cat <<'CONTEXT'
=== MENGROWTH COMPACTION RECOVERY ===

PROJECT: Meningioma growth prediction from small longitudinal MRI cohorts.
FRAMEWORK: 3-stage complexity ladder. Each stage must beat the previous under LOPO-CV.

CODEBASE STRUCTURE:
  src/growth/shared/           — GrowthModel ABC, LOPOEvaluator, metrics, bootstrap, covariates
  src/growth/stages/stage1_*/  — Gompertz mean function (Stage 1 specific)
  src/growth/stages/stage2_*/  — SeverityModel, QuantileTransform, growth functions
  src/growth/stages/stage3_*/  — Facade for encoder, SDP, projection
  src/growth/models/growth/    — Stage-agnostic models (ScalarGP, LME, HGP)
  src/growth/evaluation/       — GP probes, latent quality, variance_decomposition
  experiments/stage1_volumetric/ — Stage 1 experiments (symlink: segment_based_approach)
  experiments/stage2_severity/   — Stage 2 experiments
  experiments/stage3_latent/     — Stage 3 (lora/, sdp/, domain_gap/)
  experiments/variance_decomposition/ — Cross-stage ΔR² analysis

STAGES:
  Stage 1 (PRIMARY): Volume-only baseline (ScalarGP, LME, HGP on log-volume)
  Stage 2 (SECONDARY): Latent severity NLME (quantile transform, reduced Gompertz)
  Stage 3 (TERTIARY): Representation learning (LoRA→SDP→PCA→GP+ARD)
  Variance Decomposition: M₀→M₁→M₂→M₃→M₄, ΔR², permutation tests

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

CODING RULES:
  Type hints, Google docstrings, no magic numbers, LayerNorm only,
  shape assertions, logging (no print), prefer library functions.

=== END COMPACTION RECOVERY ===
CONTEXT
