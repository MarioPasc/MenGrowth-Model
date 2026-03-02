# Phase 4 Pivot: From Neural ODE to GP-Based Growth Prediction

## Executive Summary

This document specifies a complete redesign of **Module 5 (Phase 4 — Growth Forecasting)** and consequent updates to **Module 6 (Evaluation)**, **DECISIONS.md**, and **prposed_CLAUDE.md** (the project's `CLAUDE.md`). The Neural ODE approach is replaced by three Gaussian-process-centric models of increasing complexity, evaluated via Leave-One-Patient-Out Cross-Validation (LOPO-CV).

**What changes:**
- Module 5 (`module_5_neural_ode.md`) → replaced entirely by `module_5_growth_prediction.md`
- Module 6 (`module_6_evaluation.md`) → updated: Phase 4 quality targets, ablation matrix, figure list
- DECISIONS.md → decisions D8, D9, D10 become obsolete; new decisions D16–D20 added
- prposed_CLAUDE.md → Phase 4 description, dependency chain, environment, directory structure updated
- `methodology_refined.md` Section 5 → replaced

**What does NOT change:**
- Modules 0–4 are unaffected. The upstream pipeline (LoRA → SDP → Encoding → Trajectories) remains identical.
- The input contract for Module 5 (`trajectories.json`, `phase2_sdp.pt`) remains the same.
- The SDP partition structure ($z_{\text{vol}}^{24}$, $z_{\text{loc}}^{8}$, $z_{\text{shape}}^{12}$, $z_{\text{res}}^{84}$) is preserved and exploited.

---

## 1. Motivation: Why Abandon the Neural ODE

### 1.1 Statistical Insufficiency

The Andalusian longitudinal cohort, after quality control, contains:

| Statistic | Value |
|---|---|
| Total patients ($N$) | 33 |
| Total studies ($S$) | 100 |
| Mean studies/patient ($\bar{n}$) | 3.03 |
| Std studies/patient ($\sigma_n$) | 1.17 |
| Range | [2, 6] |
| Patients with exactly 2 studies | 19 (57.6%) |

The number of forward temporal pairs (the effective training set for any transition model) is:

$$P = \sum_{i=1}^{33} \binom{n_i}{2} = 19\binom{2}{2} + 4\binom{3}{2} + 6\binom{4}{2} + 3\binom{5}{2} + 1\binom{6}{2} = 19 + 12 + 36 + 30 + 15 = 112$$

The original Neural ODE design (methodology_refined.md Section 5) specified a `PartitionODE` with:
- `GompertzDynamics`: learnable $\alpha$ (1), $K$ (4), correction MLP (4+44→32→4 = 1,568 params)
- `loc_mlp`: 12→32→8 = 640 params
- `shape_mlp`: 16→32→12 = 896 params
- Total ODE parameters: **~3,100+**

With 112 training pairs (and LOPO-CV leaving 97–111 per fold), the parameter-to-observation ratio is $3100/112 \approx 27.7$, which is catastrophically overparameterized. Even with the frozen residual partition (D10), jerk regularization, and Gaussian perturbation augmentation, this regime is fundamentally hostile to generalization.

**The Bayesian Information Criterion (BIC) quantifies this:**

$$\text{BIC} = k \ln(n) - 2 \ln(\hat{L})$$

For $k = 3100$ and $n = 112$: the penalty term alone is $3100 \times \ln(112) \approx 14{,}600$, dwarfing any achievable log-likelihood improvement.

The original methodology estimated ~155 pairs from 42 patients (methodology_refined.md Section 5.2). The actual cohort after QC has only 33 patients and 112 pairs — **28% fewer pairs than planned**, exacerbating the problem.

### 1.2 Stochastic Extension (SDE) Is Even More Demanding

The planned stochastic extension (Neural SDE) adds a diffusion network $\sigma_\theta(z, t)$ with comparable parameter count to the drift $f_\theta(z, t)$, approximately doubling the free parameters to ~6,000+. This is definitively infeasible with 112 pairs.

### 1.3 The 2-Study Patient Problem

57.6% of patients (19/33) have exactly 2 longitudinal studies. A 2-observation patient contributes exactly 1 forward pair. For a Neural ODE, this single pair must simultaneously constrain:
- The initial condition sensitivity of the ODE dynamics
- The time-horizon extrapolation behavior
- The partition-specific dynamics rates

This is fundamentally underdetermined per patient. While the Neural ODE trains globally across all patients, the per-patient adaptation capacity is nil — there are no per-patient parameters to adjust at inference time.

### 1.4 The Alternative: Models with Principled Small-$n$ Guarantees

The replacement models share three properties that make them suitable for this regime:
1. **Population-level parameter sharing** with **per-patient adaptation** via random effects (LME) or posterior conditioning (GP)
2. **Closed-form inference** — no iterative optimization at test time, no ODE solver instabilities
3. **Calibrated uncertainty** that grows honestly with temporal extrapolation distance

---

## 2. The Three Models

The models form a complexity hierarchy: each is a strict generalization of the previous. This provides a clean scientific narrative and enables principled model comparison.

### 2.1 Model A — Baseline: Linear Mixed-Effects Model (LME) on $z_{\text{vol}}$

#### Scientific Rationale

The LME tests the simplest viable hypothesis: *latent volume trajectories are approximately linear over the observed time horizons*. If the SDP has successfully structured the latent space, the volume partition $z_{\text{vol}} \in \mathbb{R}^{24}$ should evolve smoothly, and a linear model may capture the dominant dynamics. The LME is the standard baseline in longitudinal clinical studies (Laird & Ware, "Random-Effects Models for Longitudinal Data," *Biometrics*, 1982) and provides automatic shrinkage estimation that is optimal for mixed-observation-count designs.

#### Mathematical Formulation

For patient $i \in \{1, \ldots, N\}$, latent dimension $d \in \{1, \ldots, 24\}$ (volume partition), at time $t_{ij}$ (months from first scan):

$$z_{i}^{(d)}(t_{ij}) = \underbrace{(\beta_0^{(d)} + b_{0i}^{(d)})}_{\text{patient-specific intercept}} + \underbrace{(\beta_1^{(d)} + b_{1i}^{(d)})}_{\text{patient-specific slope}} \cdot t_{ij} + \epsilon_{ij}^{(d)}$$

**Fixed effects** (population-level):
- $\beta_0^{(d)} \in \mathbb{R}$: mean intercept for dimension $d$
- $\beta_1^{(d)} \in \mathbb{R}$: mean temporal slope for dimension $d$

**Random effects** (patient-specific deviations):

$$\begin{pmatrix} b_{0i}^{(d)} \\ b_{1i}^{(d)} \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \; \Omega^{(d)} = \begin{pmatrix} \tau_0^{(d)2} & \rho^{(d)} \tau_0^{(d)} \tau_1^{(d)} \\ \rho^{(d)} \tau_0^{(d)} \tau_1^{(d)} & \tau_1^{(d)2} \end{pmatrix}\right)$$

**Residual noise:**

$$\epsilon_{ij}^{(d)} \sim \mathcal{N}(0, \sigma^{(d)2})$$

**In compact matrix form for the full volume partition:**

$$\mathbf{z}_{\text{vol},i}(t) = (\boldsymbol{\beta}_0 + \mathbf{b}_{0i}) + (\boldsymbol{\beta}_1 + \mathbf{b}_{1i}) \cdot t + \boldsymbol{\epsilon}_i(t) \in \mathbb{R}^{24}$$

#### Parameter Count

Per dimension: 2 fixed effects + 3 variance components ($\tau_0^2, \tau_1^2, \rho$) + 1 residual variance = **6 parameters**, estimated from all 100 observations (not 112 pairs — LME uses observation-level data directly).

Each model per dimension is estimated independently. With 24 volume dimensions: $6 \times 24 = 144$ total population-level parameters, but each individual model uses all 100 observations. Effective ratio: $100/6 \approx 16.7$ observations per parameter per model — adequate for REML.

#### Estimation

Restricted Maximum Likelihood (REML), which provides unbiased variance component estimates:

$$\ell_{\text{REML}}(\Omega, \sigma^2) = -\frac{1}{2} \left[ (N_{\text{obs}} - p) \ln(2\pi) + \ln |V| + \ln |X^\top V^{-1} X| + \mathbf{r}^\top V^{-1} \mathbf{r} \right]$$

where $V = Z \Omega Z^\top + \sigma^2 I$ is the marginal covariance, $X$ is the fixed-effects design matrix, $Z$ is the random-effects design matrix.

#### Single-Patient Inference (BLUP)

Given a new patient $i^*$ with observed timepoints $\{(t_j, z_{\text{vol}}^{(d)}(t_j))\}_{j=1}^{n_{i^*}}$, the Best Linear Unbiased Predictor of the random effects is:

$$\hat{\mathbf{b}}_{i^*}^{(d)} = \Omega^{(d)} Z_{i^*}^\top \left(Z_{i^*} \Omega^{(d)} Z_{i^*}^\top + \sigma^{(d)2} I\right)^{-1} \left(\mathbf{z}_{i^*}^{(d),\text{obs}} - X_{i^*} \hat{\boldsymbol{\beta}}^{(d)}\right)$$

The prediction at future time $t^*$:

$$\hat{z}_{\text{vol},i^*}^{(d)}(t^*) = (\hat{\beta}_0^{(d)} + \hat{b}_{0,i^*}^{(d)}) + (\hat{\beta}_1^{(d)} + \hat{b}_{1,i^*}^{(d)}) \cdot t^*$$

**Shrinkage property for $n_i = 2$:** With only 2 observations, the BLUP automatically shrinks the patient's individual estimates toward the population mean $(\beta_0, \beta_1)$. The shrinkage factor is:

$$\text{shrinkage} = \frac{\sigma^2}{\sigma^2 + n_i \tau^2}$$

More observations → less shrinkage → more individualized prediction. This is exactly the correct behavior for our heterogeneous observation-count design.

#### Volume Prediction via Semantic Head

$$\hat{V}_{i^*}(t^*) = \pi_{\text{vol}}\left(\hat{\mathbf{z}}_{\text{vol},i^*}(t^*)\right) \in \mathbb{R}^4$$

Since $\pi_{\text{vol}}$ is a linear head ($W \in \mathbb{R}^{4 \times 24}$, $b \in \mathbb{R}^4$), prediction intervals propagate exactly:

$$\text{Var}(\hat{V}(t^*)) = W \, \text{diag}\left(\text{Var}(\hat{z}_1(t^*)), \ldots, \text{Var}(\hat{z}_{24}(t^*))\right) W^\top$$

#### References

- Laird, N. M. & Ware, J. H. "Random-Effects Models for Longitudinal Data," *Biometrics*, 1982.
- Robinson, G. K. "That BLUP Is a Good Thing: The Estimation of Random Effects," *Statistical Science*, 1991.
- Verbeke, G. & Molenberghs, G. *Linear Mixed Models for Longitudinal Data*, Springer, 2000.

---

### 2.2 Model B — Literature-Backed: Hierarchical Gaussian Process (H-GP) on $z_{\text{vol}}$

#### Scientific Rationale

The GP generalizes the LME by allowing **nonlinear** temporal trajectories without specifying the functional form, while providing **calibrated posterior uncertainty**. GPs are the Bayesian non-parametric standard for small-sample time series. The hierarchical variant shares kernel hyperparameters across patients (empirical Bayes), addressing the limited observations per patient.

This approach is directly backed by Schulam & Saria ("A Framework for Individualizing Predictions of Disease Trajectories by Exploiting Multi-Resolution Structure," *NeurIPS*, 2015), who applied hierarchical GPs to clinical trajectories in a comparable sample-size regime.

#### Mathematical Formulation

For patient $i$, dimension $d$ of the volume partition:

$$z_i^{(d)}(t) \sim \mathcal{GP}\left(m^{(d)}(t), \; k^{(d)}(t, t')\right)$$

**Population-informed mean function** (uses LME population estimates as the GP prior mean):

$$m^{(d)}(t) = \hat{\beta}_0^{(d)} + \hat{\beta}_1^{(d)} \cdot t$$

The GP learns **deviations from the population linear trend**. This creates a natural nesting: if the true dynamics are linear, the GP posterior reduces to the LME.

**Temporal kernel — Matérn-$\frac{5}{2}$** (twice-differentiable sample paths, physically appropriate for smooth tumor growth):

$$k_{\text{M52}}(t, t') = \sigma_f^2 \left(1 + \frac{\sqrt{5} \, |\Delta t|}{\ell} + \frac{5 \, \Delta t^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5} \, |\Delta t|}{\ell}\right)$$

with $\Delta t = |t - t'|$. Full kernel with observation noise:

$$k(t, t') = k_{\text{M52}}(t, t') + \sigma_n^2 \, \delta(t, t')$$

#### Hierarchical Hyperparameter Sharing (Empirical Bayes)

Per-patient fitting of $(\sigma_f, \ell, \sigma_n)$ is ill-conditioned with $n_i \in [2, 6]$. Instead, share hyperparameters across all patients by maximizing the pooled marginal log-likelihood:

$$\hat{\theta}_{\text{GP}}^{(d)} = \arg\max_{\theta} \sum_{i=1}^{N} \log p(\mathbf{z}_i^{(d)} | \theta)$$

where for each patient:

$$\log p(\mathbf{z}_i^{(d)} | \theta) = -\frac{1}{2} \left[\mathbf{y}_i^\top K_{y,i}^{-1} \mathbf{y}_i + \log |K_{y,i}| + n_i \log(2\pi)\right]$$

with $\mathbf{y}_i = \mathbf{z}_i^{(d)} - \mathbf{m}_i^{(d)}$ and $K_{y,i} \in \mathbb{R}^{n_i \times n_i}$.

This uses all 100 observations across all 33 patients to estimate just **3 hyperparameters per latent dimension**. Total: $3 \times 24 = 72$ hyperparameters for the full volume partition.

#### Single-Patient Inference (Posterior Conditioning)

Given patient $i^*$ with observations $\mathcal{D}_{i^*} = \{(t_j, z_{i^*}^{(d)}(t_j))\}_{j=1}^{n_{i^*}}$:

$$p\left(z_{i^*}^{(d)}(t^*) \mid \mathcal{D}_{i^*}, \hat{\theta}\right) = \mathcal{N}\left(\mu_{i^*}^{(d)}(t^*), \; \sigma_{i^*}^{(d)2}(t^*)\right)$$

**Predictive mean:**

$$\mu_{i^*}^{(d)}(t^*) = m^{(d)}(t^*) + \mathbf{k}_*^\top K_{y,i^*}^{-1} \left(\mathbf{z}_{i^*}^{(d)} - \mathbf{m}_{i^*}^{(d)}\right)$$

**Predictive variance:**

$$\sigma_{i^*}^{(d)2}(t^*) = k(t^*, t^*) - \mathbf{k}_*^\top K_{y,i^*}^{-1} \mathbf{k}_*$$

where $\mathbf{k}_* = [k(t^*, t_j)]_{j=1}^{n_{i^*}}$ and $K_{y,i^*} \in \mathbb{R}^{n_{i^*} \times n_{i^*}}$.

**Computational cost:** $\mathcal{O}(n_{i^*}^3)$ per patient — for $n_{i^*} \leq 6$, this is a $6 \times 6$ matrix inversion. Instantaneous.

#### Behavior with $n_i = 2$ (Critical Property)

For a patient with exactly 2 observations:
- The posterior mean **interpolates** through the two points and **reverts to the population mean** $m(t)$ as $|t - t_j| \to \infty$, at a rate controlled by the length-scale $\ell$.
- The posterior variance is minimal at observed times and **increases monotonically** with extrapolation distance.
- The reversion rate $\ell$ is estimated from all 33 patients, so the extrapolation behavior is informed by the full population.

This is precisely the correct behavior: uncertain predictions honestly reflect the lack of patient-specific data.

#### Volume Prediction and Uncertainty

Since $\pi_{\text{vol}}$ is a linear head:

$$\hat{V}_{i^*}(t^*) = W \, \hat{\boldsymbol{\mu}}_{\text{vol},i^*}(t^*) + b$$

$$\text{Cov}(\hat{V}_{i^*}(t^*)) = W \, \text{diag}\left(\sigma_{i^*,1}^2(t^*), \ldots, \sigma_{i^*,24}^2(t^*)\right) W^\top$$

#### References

- Rasmussen, C. E. & Williams, C. K. I. *Gaussian Processes for Machine Learning*, MIT Press, 2006. (Chapters 2, 5.)
- Schulam, P. & Saria, S. "A Framework for Individualizing Predictions of Disease Trajectories by Exploiting Multi-Resolution Structure," *NeurIPS*, 2015.
- Liu, H. et al. "When Gaussian Process Meets Big Data: A Review of Scalable GPs," *IEEE TNNLS*, 2020.

---

### 2.3 Model C — Novel Contribution: Partition-Aware Multi-Output GP (PA-MOGP)

#### Scientific Rationale and Novelty Claim

This is the thesis's novel contribution. The core insight: **the SDP creates a structured latent space with semantically distinct partitions that should exhibit different temporal dynamics, and a multi-output GP can exploit this structure through a composite kernel design.**

The novelty is threefold:
1. **Partition-specific temporal kernels** reflecting distinct biological dynamics (volume vs. location vs. shape).
2. **Cross-partition covariance** encoding the mechanistic hypothesis that volume growth drives secondary changes in shape and location.
3. **Operation in a foundation model's disentangled latent space** — combining representation learning with principled temporal modeling.

No prior work has combined foundation model disentangled representations with structured multi-output Gaussian processes for tumor growth prediction.

#### Mathematical Formulation

**Active latent subspace:** $\tilde{z} = [z_{\text{vol}}^{24} | z_{\text{loc}}^{8} | z_{\text{shape}}^{12}] \in \mathbb{R}^{44}$

The residual partition $z_{\text{res}}^{84}$ is frozen at $t_0$ values (carried forward unchanged from the first observation), consistent with the original Neural ODE design rationale (D10): the residual encodes scanner, texture, and contextual information that does not evolve temporally.

**Joint process:**

$$\tilde{z}_i(t) \sim \mathcal{GP}\left(\mathbf{m}(t), \; \mathbf{K}(t, t')\right)$$

where $\mathbf{K}(t, t') \in \mathbb{R}^{44 \times 44}$.

**Partition-specific temporal kernels:**

Each partition receives a kernel reflecting its expected biological dynamics:

| Partition | Kernel | Rationale |
|---|---|---|
| $z_{\text{vol}}$ (24 dims) | Matérn-5/2 | Smooth, twice-differentiable growth; captures inflection near carrying capacity |
| $z_{\text{loc}}$ (8 dims) | Squared Exponential | Very slow, infinitely smooth centroid drift (meningiomas are largely stationary) |
| $z_{\text{shape}}$ (12 dims) | Matérn-3/2 | Once-differentiable; shape can change more abruptly than volume |

**Volume kernel (Matérn-5/2):**

$$k_{\text{vol}}(\Delta t) = \sigma_{\text{vol}}^2 \left(1 + \frac{\sqrt{5}\Delta t}{\ell_{\text{vol}}} + \frac{5\Delta t^2}{3\ell_{\text{vol}}^2}\right) \exp\left(-\frac{\sqrt{5}\Delta t}{\ell_{\text{vol}}}\right)$$

**Location kernel (SE):**

$$k_{\text{loc}}(\Delta t) = \sigma_{\text{loc}}^2 \exp\left(-\frac{\Delta t^2}{2\ell_{\text{loc}}^2}\right)$$

**Shape kernel (Matérn-3/2):**

$$k_{\text{shape}}(\Delta t) = \sigma_{\text{shape}}^2 \left(1 + \frac{\sqrt{3}\Delta t}{\ell_{\text{shape}}}\right) \exp\left(-\frac{\sqrt{3}\Delta t}{\ell_{\text{shape}}}\right)$$

**Composite kernel with cross-partition coupling (Intrinsic Coregionalization Model):**

$$\mathbf{K}(t, t') = \underbrace{\mathbf{B}_{\text{vol}}^{\text{diag}} \otimes k_{\text{vol}}(\Delta t)}_{\text{within-volume}} + \underbrace{\mathbf{B}_{\text{loc}}^{\text{diag}} \otimes k_{\text{loc}}(\Delta t)}_{\text{within-location}} + \underbrace{\mathbf{B}_{\text{shape}}^{\text{diag}} \otimes k_{\text{shape}}(\Delta t)}_{\text{within-shape}} + \underbrace{\mathbf{B}_{\text{cross}} \otimes k_{\text{vol}}(\Delta t)}_{\text{volume-driven coupling}} + \sigma_n^2 I_{44} \, \delta(t, t')$$

where:
- $\mathbf{B}_{p}^{\text{diag}} \in \mathbb{R}^{44 \times 44}$ are diagonal matrices (per-dimension signal variances within each partition, zero outside).
- $\mathbf{B}_{\text{cross}} = \mathbf{w}\mathbf{w}^\top$ with $\mathbf{w} \in \mathbb{R}^{44}$ is a rank-1 matrix encoding **how volume dynamics drive changes in location and shape**. The coupling term uses $k_{\text{vol}}$ as its temporal kernel — the mechanistic hypothesis is that cross-partition correlations operate on the same timescale as volume growth.

#### Parameter Count

| Component | Parameters |
|---|---|
| $k_{\text{vol}}$: $\sigma_{\text{vol}}, \ell_{\text{vol}}$ | 2 |
| $k_{\text{loc}}$: $\sigma_{\text{loc}}, \ell_{\text{loc}}$ | 2 |
| $k_{\text{shape}}$: $\sigma_{\text{shape}}, \ell_{\text{shape}}$ | 2 |
| $\mathbf{B}_{\text{vol}}^{\text{diag}}$: 24 variances | 24 |
| $\mathbf{B}_{\text{loc}}^{\text{diag}}$: 8 variances | 8 |
| $\mathbf{B}_{\text{shape}}^{\text{diag}}$: 12 variances | 12 |
| $\mathbf{B}_{\text{cross}}$: rank-1, $\mathbf{w} \in \mathbb{R}^{44}$ | 44 |
| $\sigma_n^2$ (observation noise) | 1 |
| **Total** | **95** |

All 95 hyperparameters estimated from the pooled marginal likelihood across 33 patients × 44 dimensions = 4,400 observation-dimension pairs. Effective ratio: $4400/95 \approx 46$.

#### Hierarchical Estimation

Same empirical Bayes approach as Model B, but now maximizing the **multivariate** marginal likelihood:

$$\hat{\theta} = \arg\max_\theta \sum_{i=1}^{N} \log p\left(\text{vec}(\tilde{\mathbf{Z}}_i) \mid \theta\right)$$

where $\tilde{\mathbf{Z}}_i \in \mathbb{R}^{n_i \times 44}$ is the observed active latent matrix for patient $i$, and the likelihood is:

$$\log p\left(\text{vec}(\tilde{\mathbf{Z}}_i) \mid \theta\right) = -\frac{1}{2}\left[\mathbf{y}_i^\top \mathbf{K}_{y,i}^{-1} \mathbf{y}_i + \log|\mathbf{K}_{y,i}| + 44 n_i \log(2\pi)\right]$$

with $\mathbf{K}_{y,i} \in \mathbb{R}^{44 n_i \times 44 n_i}$.

#### Single-Patient Inference

Given patient $i^*$ with $\tilde{\mathbf{Z}}_{i^*} \in \mathbb{R}^{n_{i^*} \times 44}$:

$$p\left(\tilde{z}_{i^*}(t^*) \mid \tilde{\mathbf{Z}}_{i^*}\right) = \mathcal{N}\left(\boldsymbol{\mu}_{i^*}(t^*), \; \Sigma_{i^*}(t^*)\right)$$

$$\boldsymbol{\mu}_{i^*}(t^*) = \mathbf{m}(t^*) + \mathbf{K}_{*,\text{obs}}^\top \mathbf{K}_{\text{obs,obs}}^{-1} \left(\text{vec}(\tilde{\mathbf{Z}}_{i^*}) - \text{vec}(\mathbf{M}_{i^*})\right) \in \mathbb{R}^{44}$$

$$\Sigma_{i^*}(t^*) = \mathbf{K}_{**} - \mathbf{K}_{*,\text{obs}}^\top \mathbf{K}_{\text{obs,obs}}^{-1} \mathbf{K}_{*,\text{obs}} \in \mathbb{R}^{44 \times 44}$$

Maximum matrix size: $264 \times 264$ (for $n_{i^*} = 6, d = 44$). Instantaneous inversion.

#### Volume Prediction with Full Covariance

$$\hat{V}_{i^*}(t^*) = W_{\text{vol}} \, \boldsymbol{\mu}_{i^*,\text{vol}}(t^*) + b_{\text{vol}}$$

$$\text{Cov}(\hat{V}_{i^*}(t^*)) = W_{\text{vol}} \, \Sigma_{i^*,\text{vol-vol}}(t^*) \, W_{\text{vol}}^\top$$

where $\Sigma_{i^*,\text{vol-vol}}(t^*)$ is the $24 \times 24$ upper-left block of $\Sigma_{i^*}(t^*)$.

The key advantage over Model B: the predictive covariance of the volume partition is informed by **location and shape observations** through the cross-partition coupling — additional observations of centroid or shape constrain the volume prediction.

#### Novelty Statement (for Thesis Text)

*"We propose a Partition-Aware Multi-Output Gaussian Process (PA-MOGP) that leverages the structured disentanglement of the Supervised Disentangled Projection (SDP) latent space to assign biologically informed temporal priors to distinct semantic partitions. The PA-MOGP enables principled growth prediction with full multivariate uncertainty quantification from as few as two longitudinal observations per patient, and encodes the mechanistic hypothesis that volume growth drives secondary geometric changes through a learned cross-partition coupling term."*

#### References

- Álvarez, M. A. et al. "Kernels for Vector-Valued Functions: A Review," *Foundations and Trends in Machine Learning*, 2012.
- Bonilla, E. V. et al. "Multi-task Gaussian Process Prediction," *NeurIPS*, 2007.
- Schulam, P. & Saria, S. "A Framework for Individualizing Predictions of Disease Trajectories by Exploiting Multi-Resolution Structure," *NeurIPS*, 2015.

---

## 3. Comparative Summary

| Property | **LME (Baseline)** | **H-GP (Literature)** | **PA-MOGP (Novel)** |
|---|---|---|---|
| Operates on | $z_{\text{vol}} \in \mathbb{R}^{24}$ | $z_{\text{vol}} \in \mathbb{R}^{24}$ | $\tilde{z} \in \mathbb{R}^{44}$ (vol+loc+shape) |
| Temporal model | Linear | Nonlinear (kernel) | Nonlinear, heterogeneous per partition |
| Cross-dimension | Independent per dim | Independent per dim | Coupled via ICM |
| Hyperparameters | 6/dim × 24 = 144 | 3 shared + 24 diag = 27 per-dim family | 95 total (shared) |
| Uncertainty | Prediction intervals (analytic) | Full posterior (analytic) | Full multivariate posterior (analytic) |
| Handles $n_i = 2$ | Yes (BLUP shrinkage) | Yes (reversion to prior mean) | Yes (multivariate reversion) |
| Irregular $\Delta t$ | Yes (native to LME) | Yes (native to GP) | Yes (native to GP) |
| Inference cost | $\mathcal{O}(1)$ (linear formula) | $\mathcal{O}(n_i^3)$ per dim (trivial) | $\mathcal{O}((44 n_i)^3)$ (trivial for $n_i \leq 6$) |
| Implementation | `statsmodels` / `lme4` | `GPy` / `sklearn` | `GPy` ICM / `gpytorch` MultitaskGP |
| Hypothesis tested | Linear dynamics suffice? | Nonlinear per-dim dynamics help? | Cross-partition coupling exists? |

**Scientific narrative:** LME establishes whether linear dynamics suffice → H-GP reveals nonlinear structure → PA-MOGP tests whether the SDP partition structure provides additional predictive power.

---

## 4. Evaluation Framework (Module 6 Updates)

### 4.1 Metrics

All metrics computed under Leave-One-Patient-Out Cross-Validation (LOPO-CV): for each fold $k \in \{1, \ldots, 33\}$, train on 32 patients, predict the held-out patient's future timepoints from their earliest observation.

**Primary metric — Volume Prediction $R^2$ (LOPO):**

$$R^2 = 1 - \frac{\sum_{i,j>1} \| V_{ij}^{\text{true}} - \hat{V}_{ij} \|^2}{\sum_{i,j>1} \| V_{ij}^{\text{true}} - \bar{V} \|^2}$$

where the sum is over all patients $i$ and timepoints $j > 1$ (predicting from $t_1$ forward), and $\bar{V}$ is the global mean volume.

**Secondary metrics:**
- **Latent MSE (LOPO):** $\frac{1}{|P_{\text{test}}|} \sum \| z^*_{i,t_j} - \hat{z}_{i,t_j} \|^2$ on volume partition.
- **Calibration:** Fraction of true values falling within 95% prediction intervals (target: 0.95 ± 0.05).
- **Mean Absolute Error (MAE) of decoded volume change:** $\frac{1}{|P_{\text{test}}|} \sum | \Delta V^{\text{true}} - \Delta \hat{V} |$ in physical units ($\text{mm}^3$ or $\text{cm}^3$).
- **Per-patient trajectory correlation:** Pearson $r$ between predicted and actual volume trajectories per patient (for patients with $n_i \geq 3$).

### 4.2 Quality Targets (replaces Phase 4 targets in Module 6)

| Metric | Target | Minimum |
|---|---|---|
| Volume prediction $R^2$ (LOPO) — best model | ≥ 0.70 | ≥ 0.50 |
| Calibration (95% CI coverage) — GP models | 0.90–0.98 | ≥ 0.80 |
| Per-patient trajectory $r$ (patients $n_i \geq 3$) | ≥ 0.80 | ≥ 0.60 |
| PA-MOGP $R^2$ > H-GP $R^2$ (improvement from coupling) | > 0 | — |

### 4.3 Ablation Matrix (replaces A6, A7, A8; adds new ablations)

| Experiment | Variable | Conditions | Primary Metric |
|---|---|---|---|
| A1–A5 | *Unchanged from current Module 6* | | |
| A6: Growth model comparison | Model | {LME, H-GP, PA-MOGP} | Vol $R^2$ (LOPO) |
| A7: ComBat effect on prediction | Harmonization | {with, without} | Vol $R^2$ (LOPO) |
| A8: GP mean function | Mean function | {zero, linear (from LME), Gompertz fit} | Vol $R^2$ (LOPO) |
| A9: GP kernel selection | Kernel (H-GP) | {Matérn-3/2, Matérn-5/2, SE} | Vol $R^2$ (LOPO) |
| A10: Cross-partition coupling | PA-MOGP structure | {with coupling, without coupling} | Vol $R^2$ (LOPO), calibration |

### 4.4 Figure List (replaces figures 8–11; figures 1–7 unchanged)

| # | Figure | Source |
|---|---|---|
| 1–7 | *Unchanged* | Modules 1–4 |
| 8 | **Patient trajectories in latent space** — 2D UMAP with temporal arrows for 5–10 patients with $n_i \geq 3$ | Module 4 |
| 9 | **Volume prediction scatter** — Predicted vs actual $\Delta V$ for all LOPO-CV test pairs, colored by model (LME/H-GP/PA-MOGP) | Module 5 |
| 10 | **Trajectory prediction with uncertainty** — 3–4 example patients showing predicted trajectory (mean ± 95% CI) overlaid on actual observations, for all three models | Module 5 |
| 11 | **Model comparison bar chart** — $R^2$, MAE, calibration for each model | Module 5 |
| 12 | **Learned kernel hyperparameters** — Length-scales $\ell$ per partition (PA-MOGP), revealing characteristic timescales of volume/location/shape dynamics | Module 5 |
| 13 | **Cross-partition coupling weights** — Heatmap of $\mathbf{w}\mathbf{w}^\top$ showing which volume dimensions most strongly drive location/shape changes | Module 5 |

---

## 5. New and Updated Decisions

### Obsolete Decisions (to be marked as superseded)

- **D8 (Temporal Pairs: Forward-Only):** No longer applicable. GP models operate on observation-level data, not transition pairs.
- **D9 (Gompertz Decode-Then-Model):** No longer applicable. Gompertz dynamics are not part of the GP models (though Gompertz may appear as an optional GP mean function in ablation A8).
- **D10 (Residual ODE Partition: Frozen):** Reinterpreted. The residual partition is still frozen (carried forward from $t_0$), but the mechanism is now trivial: Models A and B ignore it entirely, Model C excludes it from the active subspace.

### New Decisions

#### D16. Growth Prediction Framework
**Decision:** Three-model GP hierarchy (LME → H-GP → PA-MOGP) instead of Neural ODE.
**Rationale:** With 33 patients (112 forward pairs, 57.6% having only 2 studies), the Neural ODE's ~3,100+ parameters are catastrophically overparameterized. The GP hierarchy provides models with 6–95 parameters, closed-form inference, calibrated uncertainty, and principled handling of heterogeneous observation counts. See Section 1 of this document.

#### D17. LOPO-CV Protocol
**Decision:** Leave-One-Patient-Out Cross-Validation (33 folds).
**Rationale:** With only 33 patients, k-fold CV (e.g., 5-fold) leaves too few patients per fold for stable GP hyperparameter estimation. LOPO uses maximum training data per fold (32 patients, ~97 observations) while providing unbiased per-patient prediction error.

#### D18. GP Mean Function
**Decision:** Population linear mean from LME ($m(t) = \hat{\beta}_0 + \hat{\beta}_1 t$).
**Rationale:** The GP learns deviations from the population trend. This creates natural nesting (GP with linear mean ⊃ LME) and ensures that for $n_i = 2$ patients, predictions revert to the well-estimated population trend rather than the zero function. Gompertz mean is evaluated as ablation A8.

#### D19. Hierarchical Hyperparameter Sharing
**Decision:** Shared kernel hyperparameters across patients (empirical Bayes), patient-specific posteriors.
**Rationale:** With $n_i \in [2, 6]$, per-patient kernel fitting is ill-conditioned. Pooled marginal likelihood across 33 patients provides stable estimates of the 3 shared hyperparameters (per dimension in H-GP) or 95 total parameters (PA-MOGP).

#### D20. PA-MOGP Coupling Structure
**Decision:** Rank-1 cross-partition coupling ($\mathbf{B}_{\text{cross}} = \mathbf{w}\mathbf{w}^\top$) using volume temporal kernel.
**Rationale:** Encodes the mechanistic hypothesis that volume growth is the primary driver and shape/location changes are secondary consequences. Rank-1 keeps the parameter count at 44 (vs. 990 for a full $44 \times 44$ coregionalization matrix). The coupling uses $k_{\text{vol}}$ as its temporal kernel because cross-partition effects are hypothesized to operate on the same timescale as volume growth.

---

## 6. Updated Module 5 Specification

### Module 5: Growth Prediction (Phase 4) — Replaces Neural ODE

#### Overview
Train and evaluate three growth prediction models of increasing complexity on the SDP latent trajectories from the Andalusian longitudinal cohort. All models produce volume predictions via decoding through the frozen semantic head $\pi_{\text{vol}}$.

#### Input
- `trajectories.json` from Module 4 (per-patient latent trajectories, 33 patients, 100 studies)
- `phase2_sdp.pt` from Module 3 (for frozen semantic head $\pi_{\text{vol}}$: $\mathbb{R}^{24} \to \mathbb{R}^4$)

#### Input Contract
```python
# Trajectories from Module 4
trajectories: List[dict]  # 33 patients, each with ≥2 timepoints
# Each trajectory:
# {"patient_id": str, "timepoints": [{"z": [128 floats], "t": float, "date": str}, ...]}

# Semantic head (for volume decoding)
pi_vol: nn.Linear  # shape [24, 4], from trained SDP semantic heads

# Normalization parameters (from SDP training, D14)
vol_mean: np.ndarray  # shape [4], semantic target means
vol_std: np.ndarray   # shape [4], semantic target stds
```

#### Output
- `lme_results.json` — LME fixed/random effects, per-patient predictions, LOPO metrics
- `hgp_results.json` — H-GP hyperparameters, per-patient predictions with uncertainty, LOPO metrics
- `pamogp_results.json` — PA-MOGP hyperparameters, cross-partition coupling, predictions with uncertainty, LOPO metrics
- `model_comparison.json` — Head-to-head metrics ($R^2$, MAE, calibration) for all three models
- `growth_figures/` — Figures 9–13

#### Output Contract
```python
# Per-model results (structure shared across all three)
model_result: dict = {
    "model_name": str,                            # "LME" | "H-GP" | "PA-MOGP"
    "hyperparameters": dict,                      # model-specific
    "lopo_predictions": List[dict],               # per-patient predictions
    # Each prediction:
    # {
    #   "patient_id": str,
    #   "observed_times": List[float],
    #   "predicted_times": List[float],
    #   "z_vol_predicted": List[List[float]],     # shape [n_pred, 24]
    #   "z_vol_predicted_std": List[List[float]], # shape [n_pred, 24] (GP models only)
    #   "v_predicted": List[List[float]],         # shape [n_pred, 4] (decoded volumes)
    #   "v_actual": List[List[float]],            # shape [n_pred, 4]
    # }
    "metrics": {
        "vol_r2": float,
        "vol_mae": float,
        "latent_mse": float,
        "calibration_95": float,                  # GP models only
        "per_patient_r": List[float],             # for patients with n_i >= 3
    },
}

# Model comparison
comparison: dict = {
    "models": ["LME", "H-GP", "PA-MOGP"],
    "vol_r2": [float, float, float],
    "vol_mae": [float, float, float],
    "calibration": [None, float, float],
    "best_model": str,
    "coupling_improvement": float,                # PA-MOGP R² - H-GP R²
}
```

#### Code Requirements

1. **`TrajectoryDataset`** — Loads and organizes latent trajectories.
   ```python
   @dataclass
   class PatientTrajectory:
       """Single patient's longitudinal latent data."""
       patient_id: str
       times: np.ndarray          # shape [n_i], months from first scan
       z_vol: np.ndarray          # shape [n_i, 24], volume partition
       z_active: np.ndarray       # shape [n_i, 44], vol+loc+shape (for PA-MOGP)
       z_full: np.ndarray         # shape [n_i, 128], full latent (for storage)

   class TrajectoryDataset:
       """Loads trajectories.json and organizes into PatientTrajectory objects."""
       def __init__(self, trajectories_path: str):
           ...
       def get_patient(self, patient_id: str) -> PatientTrajectory:
           ...
       def get_all(self) -> List[PatientTrajectory]:
           ...
       def lopo_split(self, held_out_id: str) -> Tuple[List[PatientTrajectory], PatientTrajectory]:
           """Returns (train_patients, test_patient)."""
   ```

2. **`LMEGrowthModel`** — Linear Mixed-Effects baseline.
   ```python
   class LMEGrowthModel:
       """Per-dimension LME: z_d(t) = (β₀ + b₀ᵢ) + (β₁ + b₁ᵢ)·t + ε"""
       def fit(self, patients: List[PatientTrajectory]) -> None:
           """Fit 24 independent LME models via REML."""
       def predict(self, patient: PatientTrajectory, t_pred: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
           """Returns (z_vol_mean [n_pred, 24], z_vol_std [n_pred, 24])."""
   ```

3. **`HierarchicalGPModel`** — Per-dimension GP with shared hyperparameters.
   ```python
   class HierarchicalGPModel:
       """Per-dimension GP with population linear mean, Matérn-5/2 kernel,
       hierarchical hyperparameter sharing across patients."""
       def fit(self, patients: List[PatientTrajectory],
               lme_model: LMEGrowthModel) -> None:
           """Fit shared hyperparameters via pooled marginal likelihood."""
       def predict(self, patient: PatientTrajectory, t_pred: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
           """Returns (z_vol_mean [n_pred, 24], z_vol_std [n_pred, 24])."""
   ```

4. **`PAMOGPModel`** — Partition-Aware Multi-Output GP.
   ```python
   class PAMOGPModel:
       """Multi-output GP with partition-specific kernels and
       rank-1 cross-partition coupling."""
       def fit(self, patients: List[PatientTrajectory],
               lme_model: LMEGrowthModel) -> None:
           """Fit all hyperparameters via pooled multivariate marginal likelihood."""
       def predict(self, patient: PatientTrajectory, t_pred: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
           """Returns (z_active_mean [n_pred, 44], z_active_cov [n_pred, 44, 44])."""
       def get_coupling_weights(self) -> np.ndarray:
           """Returns w ∈ ℝ^44 from the rank-1 coupling term."""
   ```

5. **`VolumeDecoder`** — Decodes latent predictions to physical volumes.
   ```python
   class VolumeDecoder:
       """Decodes z_vol predictions through frozen π_vol, with uncertainty propagation."""
       def __init__(self, pi_vol: nn.Linear, vol_mean: np.ndarray, vol_std: np.ndarray):
           ...
       def decode(self, z_vol_mean: np.ndarray, z_vol_std: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
           """Returns (V_mean [n, 4], V_std [n, 4]) in original scale."""
   ```

6. **`LOPOEvaluator`** — Leave-One-Patient-Out evaluation loop.
   ```python
   class LOPOEvaluator:
       """Runs LOPO-CV for a given model, computes all metrics."""
       def evaluate(self, model_class: type, dataset: TrajectoryDataset,
                    decoder: VolumeDecoder, **model_kwargs) -> dict:
           """Returns full results dict matching output contract."""
   ```

#### Training Configuration

| Parameter | Value |
|---|---|
| Cross-validation | Leave-One-Patient-Out (33 folds) |
| LME optimizer | REML (via `statsmodels.MixedLM`) |
| GP hyperparameter optimizer | L-BFGS-B (via `scipy.optimize.minimize`) |
| GP kernel (H-GP) | Matérn-5/2 (default; SE and Matérn-3/2 as ablation A9) |
| GP mean function | Population linear from LME (D18) |
| Time unit | Months from first scan |
| Random seed | 42 |

#### Verification Tests

```
TEST_5.1: LME fitting [BLOCKING]
  - Fit LME on all 33 patients for 1 latent dimension
  - Assert β₀, β₁ are finite
  - Assert Ω is positive semi-definite
  - Assert σ² > 0
  Recovery: Check for singular covariance (reduce to random intercept only)

TEST_5.2: LME prediction [BLOCKING]
  - For a held-out patient with n_i = 2: predict at t₂ from t₁
  - Assert prediction is finite and within 3σ of the population mean
  - Assert prediction with n_i = 4 has smaller error than n_i = 2 (shrinkage)
  Recovery: Check BLUP computation, verify random effects extraction

TEST_5.3: GP hyperparameter fitting [BLOCKING]
  - Fit shared hyperparameters for 1 dimension
  - Assert ℓ > 0, σ_f > 0, σ_n > 0
  - Assert ℓ is in plausible range (1–120 months)
  - Assert log-marginal-likelihood is finite
  Recovery: Initialize hyperparameters from data range; add jitter to kernel diagonal

TEST_5.4: GP predictive distribution [BLOCKING]
  - Condition GP on 3 observations, predict at intermediate and extrapolation points
  - Assert predictive mean interpolates through observations (up to noise)
  - Assert predictive variance at observed times < variance at extrapolation times
  - Assert 95% CI contains observations
  Recovery: Check kernel matrix conditioning; increase σ_n lower bound

TEST_5.5: PA-MOGP cross-partition coupling [BLOCKING]
  - Fit PA-MOGP on all patients
  - Assert coupling weights w ∈ ℝ^44 are finite
  - Assert B_cross = ww^T is positive semi-definite
  - Assert full kernel matrix K ∈ ℝ^{44n × 44n} is positive definite for all patients
  Recovery: Add diagonal jitter; reduce rank or disable coupling

TEST_5.6: LOPO-CV completeness [BLOCKING]
  - Run LOPO-CV for LME model (fastest)
  - Assert 33 folds completed
  - Assert per-fold predictions are finite
  - Assert aggregated R² is finite
  Recovery: Check for patients causing singular fits; exclude and document

TEST_5.7: Volume decoding [BLOCKING]
  - Decode predicted z_vol through π_vol
  - Assert decoded volumes have correct shape [n, 4]
  - Assert decoded volumes are in plausible physical range
  Recovery: Check normalization parameters (vol_mean, vol_std from D14)

TEST_5.8: Uncertainty calibration [DIAGNOSTIC]
  - For GP models: compute fraction of true values within 95% CI
  - Report coverage (target: 0.90–0.98)
  Note: DIAGNOSTIC — calibration is an evaluation metric, not a correctness check
```

---

## 7. Updated Environment Requirements

**Remove:**
- `torchdiffeq` (Neural ODE solvers — no longer needed)

**Add:**
- `GPy>=1.13` (Gaussian Process library with multi-output GP support via ICM)
- `statsmodels>=0.14` (for `MixedLM` — LME fitting via REML)

**Unchanged:**
- `scipy` (already in environment — used for L-BFGS-B optimization of GP hyperparameters)
- `scikit-learn` (already in environment — provides `GaussianProcessRegressor` as fallback)
- `numpy`, `torch`, `matplotlib` (already present)

---

## 8. Updated Directory Structure (Module 5 portion)

```
src/growth/
├── models/
│   ├── growth/                          # NEW — replaces ode/
│   │   ├── __init__.py
│   │   ├── base.py                      # Abstract base class for growth models
│   │   ├── lme_model.py                 # LMEGrowthModel
│   │   ├── hgp_model.py                 # HierarchicalGPModel
│   │   ├── pamogp_model.py              # PAMOGPModel
│   │   └── volume_decoder.py            # VolumeDecoder (π_vol + uncertainty)
│   ├── encoder/                         # UNCHANGED
│   ├── projection/                      # UNCHANGED
│   └── segmentation/                    # UNCHANGED
├── evaluation/
│   ├── lopo_evaluator.py                # LOPOEvaluator
│   ├── growth_metrics.py                # R², MAE, calibration, per-patient r
│   └── growth_figures.py                # Figures 9–13
├── data/
│   ├── trajectory_dataset.py            # TrajectoryDataset + PatientTrajectory dataclass
│   └── ...                              # UNCHANGED
└── ...                                  # UNCHANGED
```

The `src/growth/models/ode/` directory is **removed entirely**.

---

## 9. Updated CLAUDE.md Changes

### Project Overview (line 5)
Replace: *"(4) Neural ODE growth forecasting with Gompertz-informed dynamics in the disentangled latent space"*
With: *"(4) Growth prediction via a three-model GP hierarchy (LME → Hierarchical GP → Partition-Aware Multi-Output GP) operating on the disentangled latent space, evaluated under LOPO-CV"*

### Module Dependency Chain
No change — Module 5 still depends on Module 4 outputs.

### Environment
Remove `torchdiffeq`. Add `GPy>=1.13`, `statsmodels>=0.14`.

### Directory Structure
Replace `ode/` subtree with `growth/` subtree as specified in Section 8.

### Config Files
Replace `phase4_ode.yaml` with `phase4_growth.yaml`:
```yaml
# configs/phase4_growth.yaml
growth:
  time_unit: months
  seed: 42

lme:
  optimizer: reml
  # No additional hyperparameters — REML estimates everything

hgp:
  kernel: matern52              # ablation A9: {matern32, matern52, se}
  mean_function: linear_from_lme  # ablation A8: {zero, linear, gompertz}
  optimizer: lbfgsb
  max_iter: 1000
  n_restarts: 5                 # random restarts for hyperparameter optimization

pamogp:
  vol_kernel: matern52
  loc_kernel: se
  shape_kernel: matern32
  coupling_rank: 1              # rank of B_cross
  optimizer: lbfgsb
  max_iter: 2000
  n_restarts: 5

evaluation:
  cv: lopo                      # leave-one-patient-out
  metrics: [vol_r2, vol_mae, latent_mse, calibration_95, per_patient_r]
  prediction_horizon: all       # predict all future timepoints from t_1
```

---

## 10. Affected Files — Summary for Agent

| File | Action | What Changes |
|---|---|---|
| `module_5_neural_ode.md` | **Replace entirely** | New `module_5_growth_prediction.md` per Section 6 of this document |
| `module_6_evaluation.md` | **Update** | Phase 4 quality targets (§4.2), ablation matrix (§4.3), figure list (§4.4) |
| `DECISIONS.md` | **Update** | Mark D8, D9, D10 as superseded; add D16–D20 (§5) |
| `prposed_CLAUDE.md` | **Update** | Overview, directory structure, environment, config files (§9) |
| `methodology_refined.md` | **Update** | Section 5 replaces Neural ODE with GP hierarchy; Section 6 evaluation updates; references updated |
| `src/growth/models/ode/` | **Delete** | Entire directory removed |
| `src/growth/models/growth/` | **Create** | New directory with files per Section 8 |
| `configs/phase4_ode.yaml` | **Replace** | New `phase4_growth.yaml` per Section 9 |

---

## 11. References (Complete)

### Growth Prediction Models
1. Laird, N. M. & Ware, J. H. "Random-Effects Models for Longitudinal Data," *Biometrics*, 1982.
2. Robinson, G. K. "That BLUP Is a Good Thing: The Estimation of Random Effects," *Statistical Science*, 1991.
3. Verbeke, G. & Molenberghs, G. *Linear Mixed Models for Longitudinal Data*, Springer, 2000.
4. Rasmussen, C. E. & Williams, C. K. I. *Gaussian Processes for Machine Learning*, MIT Press, 2006.
5. Bonilla, E. V. et al. "Multi-task Gaussian Process Prediction," *NeurIPS*, 2007.
6. Álvarez, M. A. et al. "Kernels for Vector-Valued Functions: A Review," *Foundations and Trends in Machine Learning*, 2012.
7. Schulam, P. & Saria, S. "A Framework for Individualizing Predictions of Disease Trajectories by Exploiting Multi-Resolution Structure," *NeurIPS*, 2015.
8. Liu, H. et al. "When Gaussian Process Meets Big Data: A Review of Scalable GPs," *IEEE TNNLS*, 2020.

### Tumor Growth (Context)
9. Benzekry, S. et al. "Classical Mathematical Models for Description and Prediction of Experimental Tumor Growth," *PLOS Computational Biology*, 2014.
10. Ribba, B. et al. "A Tumor Growth Inhibition Model for Low-Grade Glioma Treated with Chemotherapy or Radiotherapy," *Clinical Cancer Research*, 2012.

### Replaced Approach (Historical Reference)
11. Chen, R. T. Q. et al. "Neural Ordinary Differential Equations," *NeurIPS*, 2018.
12. Rubanova, Y. et al. "Latent ODEs for Irregularly-Sampled Time Series," *NeurIPS*, 2019.
