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

## D8. Temporal Pairs (Phase 4)
**Decision:** Forward-only (t_i < t_j)
**Rationale:** Tumor growth is biologically irreversible. Reverse pairs contradict the Gompertz prior (dV/dt > 0) and would force the neural correction to overpower the physics prior. Yields ~155 forward pairs from 42 patients.

## D9. Gompertz Approach
**Decision:** Decode-then-model (4 physical dimensions)
**Rationale:** Gompertz dynamics operate on decoded physical volumes V̂ = π_vol(z_vol) ∈ ℝ⁴, not on raw latent dims. This avoids sign problems (latent dims can be negative), reduces Gompertz-governed dims from 24 to 4, and maintains biological interpretability.

## D10. Residual ODE Partition
**Decision:** Frozen (dz_res/dt = 0)
**Rationale:** 84 unsupervised residual dims would cause massive overfitting with only ~155 training pairs. Freezing reduces effective ODE dimension from 128 to 44. Learned residual dynamics (η=0.001) evaluated as ablation A8.

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
