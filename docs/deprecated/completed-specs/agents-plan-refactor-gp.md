 GP Probe Refactoring Plan                              

 Context

 The LoRA ablation currently evaluates encoder feature quality using Ridge regression
 (linear) and MLP (nonlinear) probes. These produce point-estimate R^2 with no uncertainty,
 and the MLP results are confounded by hyperparameter sensitivity. The refactoring replaces
 both with Gaussian Process probes (Linear kernel + RBF kernel), providing:
 - Identical point-estimate R^2 (GP-linear = Ridge mathematically)
 - Posterior predictive variance (uncertainty per test point)
 - Log-marginal likelihood for principled kernel comparison (Bayesian Occam's razor)
 - R^2 credible intervals for rigorous condition comparison
 - Sausage plots for visualization

 The full LoRA phase will be re-executed on Picasso after this refactoring.

 Spec: docs/growth-related/gaussian-process/gp_probe_refactoring_spec.md

 ---
 Phase A: Create src/growth/evaluation/gp_probes.py

 New core module with:

 - GPProbeResults dataclass: r2, r2_per_dim, r2_ci_lo, r2_ci_hi, mse, predictions,
 predictive_std, log_marginal_likelihood, kernel_type, optimized_params
 - GPSemanticResults dataclass: linear dict, rbf dict, nonlinearity_evidence dict
 - GPProbe class: kernel_type ("linear"/"rbf"), normalize_features, normalize_targets,
 n_restarts, r2_ci_samples. Fits independent GPy.models.GPRegression per target dim.
 ARD=False. Noise floor constraint. CI via posterior sampling.
 - GPSemanticProbes class: Paired linear+RBF probes for volume/location/shape. fit(),
 evaluate() -> GPSemanticResults, get_summary() -> flat dict
 - extract_sausage_data() function for visualization data

 Key implementation details:
 - Feature/target standardization via StandardScaler before GP fitting
 - GPy.kern.Linear(D, ARD=False) / GPy.kern.RBF(D, ARD=False)
 - model.Gaussian_noise.variance.constrain_bounded(1e-6, 10.0)
 - Linear kernel: model.optimize(messages=False, max_iters=500) (convex)
 - RBF kernel: model.optimize_restarts(num_restarts=n_restarts, verbose=False, max_iters=500)
 - Handle r2_ci_samples=0 gracefully (skip CI, set ci_lo=ci_hi=r2)

 ---
 Phase B: Create tests/growth/test_gp_probes.py

 9 tests from spec section 6.1:

 1. test_gp_linear_matches_ridge — R^2 within 0.05 of sklearn Ridge
 2. test_predictive_variance_positive — std > 0 and finite for both kernels
 3. test_r2_ci_brackets_point_estimate — CI contains point R^2
 4. test_rbf_better_than_linear_for_nonlinear_data — RBF R^2 > linear R^2 + 0.1
 5. test_lml_favors_correct_kernel — delta LML > 0 for nonlinear data
 6. test_semantic_probes_fit_evaluate — correct GPSemanticResults structure
 7. test_high_r2_for_linear_data — R^2 > 0.90 on clean linear data
 8. test_missing_target_raises — ValueError for missing targets
 9. test_get_summary_keys — all expected keys present

 Run: ~/.conda/envs/growth/bin/python -m pytest tests/growth/test_gp_probes.py -v

 ---
 Phase C: Refactor Core Modules

 C1: src/growth/evaluation/latent_quality.py

 - Remove: ProbeResults, LinearProbe, SemanticProbes (lines 30-280)
 - Keep: ALL domain-shift metrics (compute_cka, compute_mmd, distance_correlation,
 compute_dci, etc.)
 - Update: compute_r2_scores() — use GPProbe(kernel_type="linear") instead of LinearProbe
 - Update: evaluate_latent_quality() — use GPSemanticProbes instead of SemanticProbes

 C2: src/growth/evaluation/__init__.py

 - Remove imports of: EnhancedLinearProbe, EnhancedProbeResults, EnhancedSemanticProbes,
 MLPProbe, analyze_feature_quality, compute_multi_scale_features, LinearProbe, ProbeResults,
 SemanticProbes
 - Add imports: GPProbe, GPProbeResults, GPSemanticProbes, GPSemanticResults,
 extract_sausage_data

 C3: Delete src/growth/evaluation/enhanced_probes.py

 C4: experiments/lora_ablation/pipeline/evaluate_probes.py

 - Replace EnhancedSemanticProbes with GPSemanticProbes
 - Config: remove alpha_mlp, hidden_sizes; add n_restarts, r2_ci_samples
 - Output keys: _mlp -> _rbf, add _ci_lo, _ci_hi, lml_*, nonlinearity_evidence_*
 - Predictions: mlp -> rbf_mean/rbf_std, linear -> linear_mean/linear_std
 - Pickle: probes_enhanced.pkl -> probes_gp.pkl
 - Logs: "MLP" -> "GP-RBF", "Nonlinearity Gap" -> "Nonlinearity Evidence (delta LML)"

 C5: scripts/sdp_diagnostics.py

 - Replace compute_linear_probe_ceiling() (inline Ridge with alpha sweep) with GPProbe-based
 version using both linear and RBF kernels

 C6: tests/growth/test_latent_quality.py

 - Remove TestLinearProbe (7 tests) and TestSemanticProbes (3 tests)
 - Keep all other test classes unchanged

 ---
 Phase D: Downstream File Updates

 Mechanical _mlp -> _rbf renames + nonlinearity_gap -> nonlinearity_evidence across:

 ┌──────────────────────────────────────────────────────────────┬─────────────────────────┐
 │                             File                             │         Changes         │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/analysis/analyze_results.py        │ _mlp->_rbf, table       │
 │                                                              │ headers                 │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/analysis/generate_tables.py        │ _mlp->_rbf, CSV/LaTeX   │
 │                                                              │ columns                 │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/analysis/enhanced_diagnostics.py   │ _mlp->_rbf,             │
 │                                                              │ gap->evidence           │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/analysis/visualizations.py         │ _mlp->_rbf, rename plot │
 │                                                              │  function               │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/analysis/v3_figures.py             │ _mlp->_rbf              │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/analysis/v3_cache.py               │ _mlp->_rbf in fallback  │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/analysis/regenerate_analysis.py    │ _mlp->_rbf              │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/report/narrative.py                │ _mlp->_rbf, text        │
 │                                                              │ updates                 │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/report/figures.py                  │ _mlp->_rbf, rename gap  │
 │                                                              │ figure                  │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/report/cli.py                      │ _mlp->_rbf              │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │ experiments/lora_ablation/scripts/post_hoc_analysis.py       │ _mlp->_rbf              │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │                                                              │ reads                   │
 │ experiments/lora_ablation/analysis/compute_domain_metrics.py │ metrics_enhanced.json   │
 │                                                              │ (compatible)            │
 ├──────────────────────────────────────────────────────────────┼─────────────────────────┤
 │                                                              │ Replace                 │
 │ experiments/sdp/evaluate_sdp.py                              │ LinearProbe/Enhanced    │
 │                                                              │ imports with GPProbe    │
 └──────────────────────────────────────────────────────────────┴─────────────────────────┘

 Config files (probe section update — remove use_mlp_probes, alpha_mlp, hidden_sizes; add
 n_restarts, r2_ci_samples):
 - experiments/lora_ablation/config/picasso/v3_rank_sweep.yaml
 - All other configs under config/local/, config/server/, config/picasso/

 NOT modified: experiments/lora_ablation/pipeline/train_condition.py (inline Ridge for
 checkpoint selection stays as-is — it's a fast training-time metric, not the formal probe
 evaluation)

 ---
 Phase E: SLURM Smoke Test

 Create slurm/lora_adaptation/smoke_test_v4.sh:
 - CPU partition, 30min walltime, 16GB RAM, 4 CPUs
 - Step 1: Run GP probe unit tests
 - Step 2: Run latent quality tests
 - Step 3: Import validation
 - Step 4: Quick end-to-end on synthetic 768-dim data

 Also create slurm/lora_adaptation/launch_smoke_v4.sh for a GPU smoke test:
 - 1 GPU, 2h walltime
 - Runs baseline + lora_r8 for 3 epochs on a tiny subset
 - Extracts features + runs GP probes
 - Verifies full pipeline works end-to-end

 ---
 Phase F: Verification

 1. ~/.conda/envs/growth/bin/python -m pytest tests/growth/test_gp_probes.py -v — all 9 pass
 2. ~/.conda/envs/growth/bin/python -m pytest tests/growth/test_latent_quality.py -v —
 remaining tests pass
 3. ~/.conda/envs/growth/bin/python -m pytest tests/ -v --tb=short — full suite passes
 4. grep -r "enhanced_probes\|EnhancedLinearProbe\|EnhancedSemanticProbes\|MLPProbe" src/
 experiments/ tests/ scripts/ --include="*.py" — zero results
 5. Verify GPy available: python -c "import GPy; print(GPy.__version__)"

 ---
 Key Risks

 1. GPy numerical stability on 768-dim: Mitigated by StandardScaler + noise floor constraint
 2. RBF optimization time: ~10-30s per target dim with N=800, 3 restarts. Total ~5min —
 acceptable
 3. Nonlinearity evidence interpretation: Old gap was R^2 units (0-0.3), new evidence is LML
 nats (can be large). Threshold-based logic in enhanced_diagnostics.py needs adjustment
 (delta LML > 10 = strong evidence)
 4. R^2 values may shift slightly: GP hyperparameter optimization vs fixed Ridge alpha=1.0.
 Existing tests check structure not exact values, so should pass
