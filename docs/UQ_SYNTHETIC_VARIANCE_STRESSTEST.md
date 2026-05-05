# Synthetic σ²_v Stress Test for LME vs LMEHetero

## 0. Why this experiment

The conditional-calibration finding from
`UQ_HETERO_CALIBRATION_ANSWER.md` says:

> Hetero re-allocates sharpness to where the data justifies it.
> Homo is stuck at the *average* and is therefore systematically
> miscalibrated in opposite directions on clean vs noisy scans.

This is an *observational* finding on the empirical σ²_v
distribution from the M=20 LoRA ensemble. The empirical distribution
is fixed by the ensemble, so we cannot vary the dispersion of σ²_v
without retraining. To test the claim **causally**, we replace the
empirical σ²_v with synthetic σ²_v profiles whose first two moments
(mean, dispersion, skew, etc.) we control, and re-run the full LOPO
loop. If the claim is correct, the calibration phenomena
(homo cov collapses on the high tertile, hetero cov holds, IS@95
halves) should track the synthetic profile in a predictable way.

The y-target `logvol_mean` and the trajectory geometry stay fixed —
only the **per-scan σ²_v values** are replaced. Both LME and
LMEHetero are then re-fit per fold with the same data.

## 1. What we hold fixed and what we vary

**Fixed across all profiles**:

- N = 56 patients, ≈ 179 scans, last_from_rest LOPO protocol.
- y = `logvol_mean` from the empirical M=20 ensemble.
- Time = ordinal index (or `years_from_baseline` once §3.1 of
  `UQ_THESIS_GAP_ANALYSIS.md` is resolved).
- Random seed for fold splits, REML restarts, etc.
- Floor variance for LMEHetero: 1e-6 (test below changes this too).

**Varied (the synthetic profile)**: a vector
$\boldsymbol\sigma^2_v = (\sigma^2_{v,1}, \dots, \sigma^2_{v,N})$ of
length N=179, replacing the empirical column
`uncertainty/logvol_std**2`. We sweep through five families:

| Profile | Definition | Probes |
|---|---|---|
| **A — constant** | $\sigma^2_{v,k} \equiv c$, with $c \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$ | When per-scan σ²_v is degenerate, hetero ≡ homo (up to the fixed-effect cov term). Verifies the implementation has no offset bias. |
| **B — bimodal (matched empirical)** | A fraction $p$ of scans drawn from $\mathcal{LN}(\mu_\text{hi}, \tau_\text{hi})$, the rest from $\mathcal{LN}(\mu_\text{lo}, \tau_\text{lo})$. Tune so mean+median+top-decile match `MenGrowth.h5`. | Sanity check: should reproduce the empirical conditional-calibration table within sampling noise. |
| **C — bimodal sweep** | Same as B but vary $p \in \{0, 0.05, 0.10, 0.20, 0.40\}$ keeping mean fixed. | Causal test of the variance-redistribution claim. *Predicts:* homo cov@95 on the high tertile collapses monotonically with $p$; hetero cov@95 stays near nominal. |
| **D — log-normal continuous** | $\log\sigma^2_v \sim \mathcal{N}(\mu, \tau^2)$, with $\tau \in \{0, 0.25, 0.5, 1.0, 2.0\}$ (overall mean fixed). | Smooth dispersion sweep: asks "how much does dispersion alone, with zero mass at extremes, drive the gap?" |
| **E — adversarial** | Concentrate σ²_v on the *training* scans only (target σ²_{v,*} = 0); or vice versa. | Stress test: hetero collapses to homo when target σ²_v vanishes; large widening when target σ²_v dominates. |

For each profile we hold **the marginal mean** $\overline{\sigma^2_v}$
fixed (equal to the empirical 0.42) so we are isolating *dispersion*,
not magnitude. A separate magnitude sweep (profile A scaled) is
reported as an appendix.

## 2. Procedure (per profile)

```
for profile in {A, B, C, D, E}:
    for sample_seed in {1..R}:                  # R = 50 by default
        sigma_v2 = sample_profile(profile, sample_seed, N=179)
        write tmp_uncertainty group with sigma_v2 into a copy of
            MenGrowth.h5
        for model in {LME, LMEHetero}:
            run LOPO last_from_rest with that file
            store {fold predictions, sigma_n_sq_REML, CI, cov, IS, CRPS}
        compute:
            - marginal table (LME vs LMEHetero)
            - conditional table by σ²_{v,*} tertile
            - paired bootstrap of ΔIS@95 (10000 resamples) per
              tertile
    aggregate over sample_seed: mean ± 95% CI of every metric.
```

**Implementation notes**:

- Re-use `LMEHeteroGrowthModel` and the LOPO evaluator from
  `src/growth/shared/lopo.py`. Inject the synthetic σ²_v through a
  fresh `uncertainty_loader.UncertaintySource` that overrides only
  `logvol_std`.
- Do NOT regenerate `logvol_mean`. The y target must remain the
  empirical mean so the only change between profiles is the
  predictive variance budget.
- Save per-fold per-target σ²_{v,*} alongside metrics — required
  for the per-tertile bootstrap.

## 3. Falsifiable predictions

If the variance-redistribution claim is correct, the following must
hold across the synthetic sweep:

1. **Profile A (constant σ²_v)**: LMEHetero and LME marginal
   metrics agree to within sampling noise at every $c$. Any
   systematic gap reveals an implementation bug (e.g. floor
   variance acting asymmetrically).
2. **Profile C ($p$ sweep)**: define
   $\Delta\text{cov}^{95}_\text{high}(p) =
   \text{cov}^{95}_\text{LME, high}(p) -
   \text{cov}^{95}_\text{LMEHetero, high}(p)$.
   *Predict:* monotonically decreasing, going from ≈0 at $p=0$ to
   ≤ −0.10 at $p=0.4$.
3. **Profile D ($\tau$ sweep)**: $\hat\sigma^2_{n,\text{homo}}$ rises
   with $\tau$ (REML absorbs more dispersion); $\hat\sigma^2_{n,\text{het}}$
   stays approximately at the biological residual; the ratio
   $\hat\sigma^2_{n,\text{homo}} / \hat\sigma^2_{n,\text{het}}$ tracks
   $1 + \overline{\sigma^2_v}/\hat\sigma^2_{n,\text{het}}$. *Predict:*
   ratio rises smoothly with $\tau$.
4. **Profile B (matched empirical)**: reproduces the observed
   conditional table to within $\pm 1\sigma$ of the bootstrap CI.
5. **Profile E (adversarial)**: when σ²_{v,*} = 0 everywhere on the
   target set, LMEHetero coverage matches LME (trivial limit). When
   σ²_{v,*} dominates, LMEHetero coverage stays at nominal and LME
   collapses to ≪ 0.5.

Any prediction that fails falsifies the claim or reveals an
implementation problem.

## 4. Metrics and statistics

For each (profile, sample_seed) and each model, compute:

- Marginal: cov@{50,80,90,95}, IS@95, CRPS, NLL, mean CI width,
  $\hat\sigma^2_{n,\text{REML}}$.
- Conditional by σ²_{v,*} tertile: same metrics, n per tertile.
- PIT histogram + KS-vs-Uniform p-value.
- Per-fold residual $r_k = (y_k - \hat y_k)$ and standardised
  residual $z_k = r_k / s^*_k$ for distribution diagnostics.

Across `sample_seed`, report mean ± 95 % bootstrap CI. Paired
comparisons within profile use paired bootstrap of $\Delta$IS@95
on the high tertile (10 000 resamples).

## 5. Outputs

```
experiments/stage1_volumetric/synthetic_uq/
├── configs/
│   └── synthetic_profiles.yaml         # profile definitions, seeds
├── runs/
│   └── {profile}/{sample_seed}/{model}/{lopo_results.json,
│                                         hyperparameters.json,
│                                         conditional_calibration.json}
├── aggregated/
│   ├── marginal_table.csv
│   ├── conditional_table.csv
│   ├── reml_budget_identity.csv        # for prediction 3
│   └── delta_cov_high_vs_p.csv         # for prediction 2
└── figures/
    ├── fig_a_constant_sigma.pdf
    ├── fig_b_matched_empirical.pdf
    ├── fig_c_p_sweep.pdf
    ├── fig_d_tau_sweep.pdf
    └── fig_e_adversarial.pdf
```

## 6. What this experiment answers (and what it does not)

**Answers**:

- Is the empirical conditional-calibration result a property of σ²_v
  *dispersion* (not magnitude, not correlated nuisance variables)?
- Does the homo model's coverage collapse on the high-σ²_v tertile
  scale predictably with σ²_v dispersion, as the variance-budget
  argument requires?
- At what dispersion threshold does the propagation effect become
  detectable at N=56?

**Does not answer**:

- Whether the empirical σ²_v from the M=20 LoRA ensemble is
  *correctly* estimated (calibrated). That requires test–retest
  data on the segmentation, not synthetic profiles.
- Whether continuous-time (`years_from_baseline`) changes the
  conclusion. Re-run separately under that time encoding.

## 7. Implementation checklist

- [ ] `experiments/stage1_volumetric/synthetic_uq/sample_profiles.py` —
      pure NumPy/SciPy sampling for profiles A–E with deterministic
      seeds.
- [ ] `experiments/stage1_volumetric/synthetic_uq/run_one.py` —
      single (profile, sample_seed, model) LOPO run; reuses
      `LOPOEvaluator`.
- [ ] `experiments/stage1_volumetric/synthetic_uq/aggregate.py` —
      aggregates `runs/` into `aggregated/`; produces figures.
- [ ] CI / parity test: profile A at any $c$ → LME and LMEHetero
      metrics agree to within MC error. Use this as a regression test.
- [ ] Cost budget: 5 profiles × 5 levels × 50 seeds × 2 models ×
      ≈ 5 s/fold × 56 folds ≈ 50 CPU-hours. Embarrassingly parallel,
      one process per (profile, seed). Run on Picasso CPU partition.
