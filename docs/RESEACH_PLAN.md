# Meningioma growth prediction from small longitudinal MRI cohorts

**Volume alone is a remarkably strong baseline for meningioma growth prediction, and any approach using learned representations or latent variables must demonstrably exceed it to justify added complexity at N=31–58.** The Gompertz growth model best describes meningioma trajectories, hierarchical Gaussian Processes with Matérn 5/2 kernels are well-suited for temporal modeling at this scale, and the proposed latent severity framework is mathematically equivalent to a nonlinear mixed-effects model with a single random effect — a well-studied formulation with strong connections to Item Response Theory. The critical constraint is the parameter-to-patient ratio: with 31 patients, only **2–3 predictor parameters** can be reliably estimated, expanding to 4–5 at N=58. This hard statistical boundary governs every architectural decision across all three axes.

---

## Axis 1: Gompertz growth and the volumetric baseline

### Meningioma growth is heterogeneous but mathematically well-characterized

The definitive study on meningioma growth modeling is **Engelhardt et al. (2023, eBioMedicine)**, which compared linear, exponential, power-law, and Gompertz models across 294 patients with ≥3 imaging scans using a mixed-effects framework. The **Gompertz model provided the best fit**, and hierarchical clustering revealed at least three distinct growth subgroups: pseudo-exponential (aggressive), linear, and decelerating. Approximately 21% of tumors transitioned to lower growth clusters over ~56 months, and younger patients with smaller tumors were overrepresented in the pseudo-exponential cluster.

These findings are corroborated by Fountain et al. (2017), whose systematic review found that benign meningiomas follow **logistic/Gompertzian sigmoid curves** with growth deceleration, while atypical meningiomas grow quasi-exponentially. Benjamin et al. (2021) documented in 37 patients that volumetric perimeter methods detect significantly higher growth rates than diameter methods (15.2% vs. 5.6% annual growth, p<0.01), establishing that 3D segmentation-based volumetry is the measurement standard. Across cohorts, approximately **67% of meningiomas show measurable growth** at 4–5 year follow-up using volumetric criteria, though fewer than 25% grow exponentially.

The practical implication is clear: a volume-only GP or LME baseline using Gompertz dynamics captures the dominant clinical signal. No published out-of-sample prediction benchmarks (R², RMSE) exist specifically for meningioma growth — this represents a genuine literature gap that the project can fill.

### GP kernel selection for longitudinal tumor volume

For modeling tumor volume trajectories with **~3.6 observations per patient**, the kernel choice is critical. The Gaussian Process Panel Model framework (Karch et al., 2020) demonstrates that LME models are a special case of GP models — a linear kernel with white noise is mathematically equivalent to random intercept/slope LME. This means any GP formulation subsumes the LME baseline.

The recommended kernels, in order of priority:

- **Matérn 5/2**: Twice differentiable (smooth but not infinitely so), appropriate for biological growth processes. Avoids the overly smooth behavior of RBF kernels that can produce boundary artifacts with short time series.
- **Linear + Matérn 5/2 composite**: Combines a parametric growth trend (linear kernel → random slopes) with nonparametric flexibility for deviations. This directly encodes the prior belief that growth is approximately monotonic with patient-specific rates, while allowing nonlinear deceleration.
- **Periodic kernels**: Not recommended. Tumor growth is not cyclic, and periodic components would waste parameters on clinically implausible patterns.

**Hierarchical structure is essential.** With only ~3.6 observations per patient, individual-patient GPs are poorly constrained. Szczesniak et al. (2016/2018) demonstrated Joint Hierarchical GPs (JHGP) with a population-level GP capturing the shared trajectory and individual-level GPs capturing patient-specific deviations. Applied to cystic fibrosis (N=38), this achieved Pearson correlation ~0.90+ for predicted vs. observed trajectories. The key architectural insight: patients with sparse observations get "pulled" toward the population mean, borrowing strength across the cohort. The Intrinsic Coregionalization Model (ICM) and Linear Model of Coregionalization (LMC) formulations from Álvarez et al. (2012) provide the mathematical machinery.

### Radiomic features beyond volume are high-risk at this sample size

Radiomic features beyond volume have shown promise for meningioma characterization but not specifically for growth prediction. Oi et al. (2025) identified 10 significant PyRadiomics features (GLCM, shape, first-order) from T2 MRI in 49 patients, but could not build multivariate models due to overfitting risk. Laukamp et al. (2019) achieved AUC=0.91 for grade differentiation using 4 rigorously selected features from 71 patients. Zhang et al. (2023) found relative ADC was an independent growth predictor (AUC=0.88) in 64 patients.

The pattern is consistent: **at sample sizes below ~70, only 3–4 radiomic features can enter a model simultaneously** without severe overfitting. With N=31, the project should use at most **1–2 features beyond volume**, selected via domain knowledge (e.g., sphericity, ADC) rather than data-driven feature selection. Volume itself is typically the strongest single predictor and should always be included as the baseline feature.

### ComBat harmonization: necessary for intensity, potentially harmful for volume

Longitudinal ComBat (Beer et al., 2020) is the gold standard for multi-scanner harmonization of longitudinal data. However, a critical caveat emerged from ISMRM 2022 validation work: **harmonizing volumetric data can introduce scanner differences that did not exist in the unharmonized data**, especially with LongComBat. Volume derived from segmentation is a geometric property relatively robust to scanner differences — the segmentation operates on each scan independently, and the volume is computed from the resulting binary mask.

The recommendation: **test empirically whether scanner effects are present in volumetric features** (via KS test or mixed model with scanner as covariate) before applying ComBat. For intensity-based radiomic features, LongComBat is strongly recommended if multi-scanner data is used. An alternative is to include scanner as a covariate in the GP/LME model directly, which is simpler and avoids the risk of harmonization artifacts.

---

## Axis 2: When do foundation model representations beat volume?

### BrainSegFounder architecture and feature extraction

BrainSegFounder is built on the Swin Transformer encoder (SwinUNETR variant) with hierarchical features across 4 stages. The bottleneck embedding space dimensions are **48→96→192→384→768** at progressively coarser resolutions, with the deepest features being **768-dimensional**. Its novel two-stage pretraining — first on 41,400 UK Biobank brain MRIs for normal anatomy, then on BraTS/ATLAS for pathological features — gives it strong inductive bias for brain tumor analysis.

The most directly relevant result comes from a 2025 npj Precision Oncology study: a deep learning representation framework on baseline MRI predicted meningioma growth at 3 and 5 years with **AUCs of 0.756 and 0.727**, respectively, in N=1,239 patients. Cox regression confirmed deep learning predictions as independent predictors (HR=1.996, 95% CI: 1.117–3.568). This provides strong evidence that learned representations capture growth-relevant information, but the sample size was **20× larger** than the current project.

### The critical question: do deep features add signal beyond volume?

Evidence is mixed but informative. For meningioma grading (a related but distinct task), volumetric features alone achieve **AUC 82.4%** while hybrid radiomic + deep learning models reach **AUC 94.5%**. For liver tumor grading at small N, simple radiomics with 8 selected features outperformed DenseNet — the "intrinsic manifold was much easier to model than the original volumes." For oropharyngeal cancer, self-supervised features showed best internal performance while radiomics had better external generalizability, suggesting deep features may overfit to training distributions.

The theoretical case for learned representations when volume dominates the target rests on **residual variation beyond volume**: texture heterogeneity, morphological irregularity, peritumoral interface features, and enhancement patterns. The "blessing of compositionality" (Physical Review X, 2024) shows deep networks overcome the curse of dimensionality through hierarchical composition, but this advantage materializes only with sufficient training data. **At N=31–58, the bias-variance tradeoff strongly favors parsimonious models.**

### Dimensionality reduction is non-negotiable

Raw 768-dimensional features fed into a GP with N=31 patients is a recipe for failure. The recommended pipeline:

1. Freeze the BrainSegFounder encoder and extract bottleneck features (768-dim)
2. Apply PCA retaining ~90% variance (typically reduces to 15–30 components)
3. Use supervised feature selection (LASSO, elastic net) or supervised PCA to identify **3–6 predictive components**
4. Feed these into a GP with ARD (Automatic Relevance Determination) kernel

ARD assigns per-dimension lengthscales, performing soft feature selection. However, ARD with 768 dimensions and N=60 creates a highly non-convex optimization landscape — it must be applied to the **reduced** feature set, not raw features. Active subspace methods (Tripathy et al., 2016) and Randomly Projected Additive GPs (Delbridge et al., ICML 2020) offer principled alternatives that build dimensionality reduction directly into the GP.

For LoRA-based domain adaptation, **rank 4–8 is recommended** at this sample size. Parameter-efficient fine-tuning outperforms full fine-tuning at very small N by constraining the hypothesis space, but "with tiny or noisy datasets, LoRA adapters can overfit quickly" — monitoring is essential.

### A structured decomposition: volume + residual features

The most principled approach is to explicitly decompose the prediction into:
```
ŷ = f(volume_trajectory) + g(residual_deep_features)
```
This can be implemented via supervised disentanglement (Attri-VAE style), where designated latent dimensions encode volume while residual dimensions capture texture, shape, and peritumoral features. The challenge: if volume explains most of the prediction variance, the residual dimensions may capture noise rather than signal at small N. **Quantifying this variance decomposition** — what fraction does volume alone explain? — should be a primary analytical objective.

---

## Axis 3: The latent severity model as a nonlinear mixed-effects framework

### Formal equivalence to NLME with a single random effect

The advisor's proposed model — q(s, t) where s ∈ [0,1] is latent severity and t ∈ [0,1] is normalized time — is formally a **nonlinear mixed-effects model**:

$$y_{ij} = g(s_i, t_{ij}; \theta) + \varepsilon_{ij}$$

where $s_i \sim F(\cdot)$ is the patient-specific random effect, $g(\cdot)$ is the constrained monotonic function, and $\theta$ are shared parameters. This formulation has deep roots in pharmacometric tumor modeling. **Vaghi et al. (2020, PLOS Computational Biology)** demonstrated a "reduced Gompertz" model where strong inter-parameter correlation (R² > 0.92) motivated collapsing all individual variation into a **single patient-specific parameter** — directly analogous to the severity variable s. With Bayesian inference from limited data, this reduced model achieved mean prediction error of 12.2% versus 78% for MLE.

### Connections to Item Response Theory are structural, not superficial

The analogy to IRT is precise. In the two-parameter logistic (2PL) IRT model, $P(\text{correct} | \theta, \beta) = \sigma(a(\theta - \beta))$, where $\theta$ is latent ability and $\beta$ is item difficulty. In the proposed model, $s$ plays the role of latent ability (severity), $t$ functions like item difficulty (time), and $q$ is the response probability (growth quantile). Both enforce monotonicity via positive discrimination parameters and assume a unidimensional latent trait with local independence given the latent variable.

The constraint $q(s, 0) = 0$ (no growth at time zero) has no direct IRT analog but is a natural boundary condition that IRT-inspired implementations can accommodate. Proust-Lima et al. (2014, 2023) developed joint latent class models in the `lcmm` R package that formalize the choice between **discrete latent classes** (fast/slow growers) and **continuous latent traits** (severity on a spectrum). With N=31–58, the continuous trait is more parsimonious and avoids the difficulty of selecting the number of classes.

### Monotonic implementation: constrained neural networks vs. monotonic GPs

Two competitive implementations exist for enforcing monotonicity in both s and t:

**Constrained Monotonic Neural Networks** (Runje & Shankaranarayana, 2023, ICML): The most practical implementation. Their key insight is that constraining weights to be non-negative with standard activations only approximates convex functions — they fix this by constructing additional activation functions from point reflections. The `MonoDense` layer allows specifying a `monotonicity_indicator` per input dimension, directly enforcing $\partial q/\partial s \geq 0$ and $\partial q/\partial t \geq 0$. The boundary constraint $q(s, 0) = 0$ can be enforced structurally by multiplying the output by a function $h(t)$ where $h(0) = 0$.

**Monotonic Gaussian Processes** (Riihimäki & Vehtari, 2010): The principled Bayesian alternative. Monotonicity is enforced via **virtual derivative observations**: since the derivative of a GP is also a GP, introducing constraints $\partial f/\partial x \geq 0$ at selected grid points creates a joint GP with augmented covariance blocks. Expectation Propagation handles the non-Gaussian likelihood. López-Lopera et al. (2018, 2022) extended this to guarantee constraint satisfaction **everywhere** in the domain (not just at virtual points) using finite-dimensional basis function representations. Their R package `lineqGPR` implements this. For 2D monotonicity (in both s and t), Deronzier et al. (2026) recently published block-additive GPs under monotonicity constraints that handle interactions between input variables.

**At N=31–58, monotonic GPs are arguably superior** due to principled uncertainty quantification, natural regularization via the prior, and exact posterior inference being computationally trivial at this scale (O(n³) with n ≈ 100 is instantaneous).

### Estimating severity from a single baseline MRI

This is the hardest subproblem. With only one baseline observation, the posterior over $s_i$ will be heavily dominated by the prior. Three approaches exist in order of sophistication:

**Empirical Bayes / BLUP**: In the LME framework, the Best Linear Unbiased Prediction of the random effect is $\hat{u}_j = DZ'_j V_j^{-1}(y_j - X_j\hat{\beta})$. With a single observation (n_j = 1), the reliability is $\sigma^2_s / (\sigma^2_s + \sigma^2_\varepsilon)$, which produces heavy shrinkage toward the population mean. This is honest but uninformative — the model essentially predicts the average trajectory for new patients.

**Amortized variational inference**: An encoder network maps baseline imaging features to an approximate posterior $q(s_i | \text{MRI}_i) = \mathcal{N}(\mu_\phi(\text{MRI}_i), \sigma^2_\phi(\text{MRI}_i))$. Margossian & Blei (2024, UAI) proved this is optimal for simple hierarchical models with conditionally independent latent variables — which matches the severity model structure. In practice, this means training a lightweight CNN or radiomics-to-severity regression head jointly with the growth model.

**Domain-informed proxy features**: Tumor volume at baseline, sphericity, T2 signal intensity, presence of calcification, peritumoral edema, and patient age are all correlated with growth aggressiveness (Oya et al., 2011; AIMSS scoring system). These can serve as informative covariates for severity estimation. The AIMSS score specifically combines volume, calcification, edema, and T2 signal for meningioma risk stratification, and could be used as an initial severity prior.

The practical recommendation: combine the amortized inference approach with domain-informed features. Use a simple linear model mapping [volume, age, sphericity, T2 signal] → $\hat{s}$ as the initial severity estimate, and refine with additional observations as they become available.

---

## The parameter-to-patient ratio governs everything

### Hard constraints from sample size theory

Riley et al. (2019, Statistics in Medicine) established that minimum sample sizes for prediction models must satisfy four criteria, implemented in the `pmsampsize` R/Stata package. For continuous outcomes with moderate anticipated R² ≈ 0.3 and 5 predictor parameters, the minimum n is approximately **60–100**. Van der Ploeg et al. (2014) found that modern ML methods need **>200 events per variable** for stable performance, while logistic regression stabilizes at 20–50. Harrell's guideline suggests **15–20 subjects per candidate predictor** for continuous regression.

The practical bounds for this project:

| Metric | N=31 | N=58 |
|--------|------|------|
| Maximum predictor parameters | **2–3** | **4–5** |
| LME fixed effects | Unbiased | Unbiased |
| LME random effects SE | May be biased | Acceptable (≥50 groups) |
| GP hyperparameters | Marginal; requires informative priors | Adequate for ML-II |
| LOPO-CV error bars | ±15–20% | ±10–15% |
| Effective N (ICC=0.5) | ~49 observations | ~90 observations |

The distinction between **N_patients and N_observations** is crucial. Snijders & Bosker's rule: the effective sample size for patient-level (between-patient) effects approximates N_patients, not N_observations. With ICC=0.5 and 3.6 observations per patient, the effective N_total is approximately 49, but for between-patient generalization — which is exactly what LOPO-CV measures — the relevant number is **31 independent patient trajectories**.

### LOPO-CV is appropriate but comes with wide confidence intervals

Varoquaux (2018, NeuroImage) demonstrated that cross-validation error bars in neuroimaging are **±10% even at N=100**. LOPO-CV is nearly unbiased but has the **highest variance** among CV estimators because test folds contain only one patient and training sets overlap by N-2 patients. Supplementing with **.632+ bootstrap** (Efron & Tibshirani, 1997) provides lower-variance performance estimates. Reporting the **full distribution of per-patient errors**, not just the mean, is essential for clinical credibility.

LOPO-CV is nonetheless the correct choice: splitting within patients would create temporal data leakage, and with only 3–4 observations per patient, within-patient folds are meaningless. The clustered structure of the data demands patient-level holdout.

### The N=31 → N=58 expansion is pivotal

Expanding from 31 to 58 patients moves the project from **marginal to adequate** across multiple statistical thresholds. It crosses the Maas & Hox 50-group threshold for unbiased LME standard errors, reduces LOPO-CV error bars by approximately 30–40%, and allows 1–2 additional predictor parameters. This expansion should be **the highest operational priority**.

---

## Connecting the three axes into a unified framework

### The staged complexity approach

The three axes are not competing alternatives but nested levels of complexity:

**Stage 1 — Volume-only hierarchical GP (Axis 1 baseline):** Fit a population + individual hierarchical GP with Matérn 5/2 kernel on log-transformed volume trajectories. This captures Gompertzian dynamics nonparametrically while borrowing strength across patients. Expected to explain the majority of predictable variance. This is the strong baseline against which all other approaches must demonstrate improvement.

**Stage 2 — Volume + severity (Axis 3 integration):** The severity variable can be incorporated as a latent input to the GP: $y_{ij} = \text{GP}(s_i, t_{ij}) + \varepsilon_{ij}$, where $s_i$ is inferred from baseline features. This is a GP-LVM with partial observation (time is known, severity is latent). If the population-level GP already captures the dominant growth pattern, severity modulates individual deviations — equivalent to explaining the random effects in the LME formulation.

**Stage 3 — Volume + severity + deep features (Axis 2 integration):** Foundation model features, reduced to 3–5 dimensions via PCA + supervised selection, provide additional inputs to the severity estimation or directly to the GP. The key test: does the marginal likelihood improve? If the additional complexity does not increase the marginal likelihood, Bayesian Occam's razor dictates preferring the simpler model.

### Quantifying the value of each component

The project should report a **variance decomposition** at each stage:

- What fraction of LOPO-CV prediction variance does volume alone explain?
- How much additional variance does the severity parameterization capture?
- Do deep features add statistically significant predictive signal beyond volume + severity?

At N=31, the "flat maximum" effect is likely: different approaches may converge to similar performance because the data cannot discriminate between models. If this occurs, the simpler model wins by default for clinical deployment.

### Clinical interpretability favors the proposed framework

Volumetric models map directly to clinical practice (EANO guidelines use tumor size and growth rate for surveillance decisions). The GP framework provides natural uncertainty intervals that communicate prediction confidence — wider intervals mean less certainty, which clinicians intuitively understand. The severity variable, if validated, translates to "this patient's tumor is in the Xth percentile of aggressiveness," which is clinically actionable for surgery timing and surveillance interval decisions.

Deep features from foundation models are the least interpretable component but can be explained post-hoc via Grad-CAM spatial attention maps and SHAP value decomposition. For FDA regulatory purposes (SaMD Class II, 510(k) pathway), documenting model transparency per GMLP principles is essential regardless of architecture choice.

---

## Practical recommendations and open opportunities

The most important near-term actions, in priority order:

1. **Establish the volumetric baseline rigorously.** Fit a hierarchical GP (Matérn 5/2 kernel) and an LME (random intercept + slope on log-volume) as parallel baselines. Report LOPO-CV RMSE, MAE, calibration, and prediction interval coverage with bootstrap confidence intervals.

2. **Expand to N≥50 as fast as possible.** The jump from 31 to 58 crosses multiple statistical adequacy thresholds and is the single highest-impact improvement to model reliability.

3. **Implement the severity model as an NLME.** Start with the reduced Gompertz parameterization (Vaghi et al., 2020), which naturally produces a single patient-specific parameter. Use Bayesian estimation via Stan/brms with informative priors on severity distribution.

4. **Test deep features incrementally.** Extract BrainSegFounder features, reduce to 3–5 dimensions, and test whether they improve severity estimation from single baseline scans. Report marginal likelihood comparisons.

5. **Consider biophysics-informed data augmentation.** TumorFlow (2026) and HybrSyn (2025) demonstrate synthetic longitudinal MRI generation guided by tumor growth models. This could expand training data while respecting biological growth dynamics. The ST-ConvLSTM success at N=33 for pancreatic neuroendocrine tumors provides direct precedent.

6. **Position the dataset as a community resource.** No public longitudinal meningioma dataset exists. Publishing it would be a significant contribution independent of modeling results.

## Conclusion

This project sits at an unusual intersection where the statistical constraints of small-cohort longitudinal data meet the richness of modern foundation model representations. The literature converges on a clear message: **parsimony is paramount at N=31–58, and volume trajectories modeled via Gompertzian dynamics are the foundation everything else must build upon.** The advisor's latent severity model is mathematically well-grounded — it is a nonlinear mixed-effects model with a single random effect, with direct analogs in pharmacometric tumor modeling (reduced Gompertz) and psychometric measurement (IRT). Its main challenge is test-time severity estimation from a single scan, where amortized inference with domain-informed features offers the most promising path. Foundation model representations provide a principled way to estimate severity and capture beyond-volume features, but only if dimensionality is aggressively reduced (768→3–6 dimensions) and the added complexity demonstrably improves marginal likelihood. The staged approach — volume baseline → severity parameterization → deep feature augmentation — provides both a rigorous experimental design and a natural complexity ladder where each step must earn its place through improved out-of-sample prediction.