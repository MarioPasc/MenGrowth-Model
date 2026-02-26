# DECISIONS.md — Resolved Design Choices

All binary/multi-option decisions pre-resolved. Do not revisit these unless explicitly instructed.

---

## D1. Adapter Type
**Decision:** LoRA (standard)
**Rationale:** DoRA evaluated only as ablation condition A2. The primary pipeline uses LoRA. DoRA adds magnitude-direction decomposition complexity with uncertain benefit for this architecture.

## D2. LoRA Rank
**Decision:** r=8, α=16 (effective scale α/r = 2.0)
**Rationale:** Confirmed as optimal by rank ablation across r ∈ {2, 4, 8, 16, 32}. Balances capacity and parameter efficiency (~197K trainable LoRA params).

## D3. LoRA Target Modules
**Decision:** Q, K, V projection matrices in Stages 3–4 only
**Rationale:** Stages 0–2 frozen to preserve low-level anatomical features from UK Biobank SSL pretraining. Stages 3–4 contain highest-level features most in need of domain adaptation.

## D4. Auxiliary Semantic Heads (Phase 1)
**Decision:** Enabled (λ_aux=0.1, warmup starts epoch 5, ramps over 10 epochs)
**Rationale:** Multi-task learning enriches encoder features for downstream SDP. Heads are discarded after Phase 1.

## D5. SDP Curriculum Schedule
**Decision:** Enabled (4-phase curriculum)
**Schedule:**
| Phase | Epochs | Active Losses |
|-------|--------|---------------|
| Warm-up | 0–9 | L_var only |
| Semantic | 10–39 | + L_vol, L_loc, L_shape |
| Independence | 40–59 | + L_cov, L_dCor |
| Full | 60–100 | All losses at full strength |

**Rationale:** Prevents early optimization from collapsing dimensions before semantic structure is established.

## D6. Channel Order
**Decision:** `[t2f, t1c, t1n, t2w]` = [FLAIR, T1ce, T1, T2]
**Rationale:** Matches BrainSegFounder convention. Verified in existing codebase (`swin_loader.py` line 32, `transforms.py` line 39). Wrong order causes near-zero Dice.

## D7. Decoder (Phase 1)
**Decision:** Full SwinUNETR pretrained decoder
**Rationale:** Provides stronger gradient signal to encoder during LoRA training. Discarded after Phase 1.

## D8. Temporal Pairs (Phase 4) — SUPERSEDED by D16
**Decision:** ~~Forward-only (t_i < t_j)~~
**Status:** Superseded. GP models operate on observation-level data, not transition pairs. See D16.

## D9. Gompertz Approach — SUPERSEDED by D16
**Decision:** ~~Decode-then-model (4 physical dimensions)~~
**Status:** Superseded. Gompertz dynamics are not part of the GP models. Gompertz may appear as an optional GP mean function in ablation A8. See D16.

## D10. Residual Partition — REINTERPRETED
**Decision:** Frozen (carried forward from t₀)
**Rationale:** The residual partition z_res⁸⁴ is still frozen, but the mechanism is now trivial: Models A and B (LME, H-GP) ignore it entirely (operate on z_vol only), Model C (PA-MOGP) excludes it from the active subspace z_active ∈ ℝ⁴⁴. No longer ODE-specific.
**Original rationale preserved:** 84 unsupervised residual dims would cause overfitting. Exclusion reduces effective model dimension from 128 to 44 (for PA-MOGP) or 24 (for LME/H-GP).

## D11. SDP Batch Size
**Decision:** Full-batch (800 subjects)
**Rationale:** SDP operates on precomputed h ∈ ℝ^768 feature vectors. Full-batch is computationally trivial and provides exact covariance/dCor estimates. Eliminates mini-batch estimation noise.

## D12. Shape Features
**Decision:** 3 features (sphericity, surface_area_log, solidity) via `compute_shape_array()`
**Rationale:** Aspect ratios excluded due to poor linear probe R² from noisy bounding box estimation. Revisit as SDP ablation when implementing Phase 2.

## D13. Spectral Normalization
**Decision:** On ALL SDP linear layers (not just final layer)
**Rationale:** SN on only the final layer does not guarantee global Lipschitz continuity. SN on all layers gives bound L_g ≤ L_LN · 1 · L_GELU · 1 ≈ 1.13.

## D14. Normalization Scope
**Decision:** Compute μ,σ on train_pool (800) only
**Rationale:** Same normalization parameters applied without recomputation to val (100), test (100), and Andalusian cohort. Prevents information leakage.

## D15. Data Splits
**Decision:** 525 lora_train + 100 lora_val + 225 sdp_train + 150 test = 1000
**Rationale:** Phase 2 SDP uses sdp_train (encoder frozen, no gradient contamination). sdp_train subset enables strict ablation of encoder-familiar vs. encoder-unfamiliar subjects. Note: actual server configs may use different splits than canonical values (see `experiments/lora_ablation/config/server/`).

---

## D16. Growth Prediction Framework
**Decision:** Three-model GP hierarchy (LME → H-GP → PA-MOGP) instead of Neural ODE.
**Rationale:** With 33 patients (112 forward pairs, 57.6% having only 2 studies), the Neural ODE's ~3,100+ parameters are catastrophically overparameterized (parameter-to-observation ratio ≈ 27.7). The GP hierarchy provides models with 6–95 parameters, closed-form inference, calibrated uncertainty, and principled handling of heterogeneous observation counts. See `phase4_pivot_to_gp_models.md` Section 1.

## D17. LOPO-CV Protocol
**Decision:** Leave-One-Patient-Out Cross-Validation (33 folds).
**Rationale:** With only 33 patients, k-fold CV (e.g., 5-fold) leaves too few patients per fold for stable GP hyperparameter estimation. LOPO uses maximum training data per fold (32 patients, ~97 observations) while providing unbiased per-patient prediction error.

## D18. GP Mean Function
**Decision:** Population linear mean from LME (m(t) = β̂₀ + β̂₁t).
**Rationale:** The GP learns deviations from the population trend. This creates natural nesting (GP with linear mean ⊃ LME) and ensures that for n_i = 2 patients, predictions revert to the well-estimated population trend rather than the zero function. Gompertz mean is evaluated as ablation A8.

## D19. Hierarchical Hyperparameter Sharing
**Decision:** Shared kernel hyperparameters across patients (empirical Bayes), patient-specific posteriors.
**Rationale:** With n_i ∈ [2, 6], per-patient kernel fitting is ill-conditioned. Pooled marginal likelihood across 33 patients provides stable estimates of the 3 shared hyperparameters (per dimension in H-GP) or 95 total parameters (PA-MOGP).

## D20. PA-MOGP Coupling Structure
**Decision:** Rank-1 cross-partition coupling (B_cross = ww^T) using volume temporal kernel.
**Rationale:** Encodes the mechanistic hypothesis that volume growth is the primary driver and shape/location changes are secondary consequences. Rank-1 keeps the parameter count at 44 (vs. 990 for a full 44×44 coregionalization matrix). The coupling uses k_vol as its temporal kernel because cross-partition effects are hypothesized to operate on the same timescale as volume growth.
