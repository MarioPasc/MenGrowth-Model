# Supervised Disentangled Projection for meningioma growth prediction: a comprehensive research foundation

**The Supervised Disentangled Projection (SDP) module — mapping frozen SwinUNETR features h ∈ ℝ^768 to a structured z ∈ ℝ^128 with semantic partitions — rests on solid theoretical and empirical ground.** Locatello et al.'s impossibility theorem (ICML 2019) proves that unsupervised disentanglement cannot succeed without inductive biases, making the SDP's explicit supervision a theoretical necessity rather than a mere design convenience. With 800 labeled BraTS-MEN samples, the module has roughly eight times more labels than the ~100 that Locatello et al. (ICLR 2020) showed sufficient for disentanglement. The combination of VICReg within-partition regularization, distance correlation cross-partition independence, and spectral normalization creates a well-constrained optimization landscape. Downstream, the partition structure naturally maps onto block-diagonal multi-output GP kernels, dramatically reducing parameter counts for the 42-patient Andalusian cohort.

---

## 1. The impossibility theorem demands supervision

The most important theoretical justification for the SDP comes from Locatello et al., "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations," *ICML 2019* (Best Paper, PMLR 97:4114–4124). Their **Theorem 1** proves that for any unsupervised generative model producing disentangled representations, one can construct another model with identical marginal distribution but entangled latents. Training over 12,000 models across six methods (β-VAE, FactorVAE, DIP-VAE I/II, β-TCVAE, AnnealedVAE) on seven datasets, they found that hyperparameter choice explained only **37%** of disentanglement score variance, with random seeds dominating. No unsupervised model selection criterion reliably identifies disentangled representations.

The follow-up, Locatello et al., "Disentangling Factors of Variations Using Few Labels," *ICLR 2020*, showed that supervision resolves this impossibility. As few as **100 labeled examples** (0.01–0.5% of training data) sufficed for both model selection and semi-supervised training. Their semi-supervised β-TCVAE with 1,000 labels produced near-diagonal mutual information matrices between latent dimensions and ground-truth factors. The SDP's 800 fully-supervised meningioma samples, where volume, location, and shape are directly computable from segmentation masks, provide supervision that is orders of magnitude richer than these minimal requirements.

Higgins et al., "Towards a Definition of Disentangled Representations," *arXiv:1812.02230*, 2018, formalized disentanglement via group theory: a representation is disentangled w.r.t. a group decomposition G = G₁ × G₂ × ... × Gₙ if there exists a corresponding decomposition Z = Z₁ ⊕ Z₂ ⊕ ... ⊕ Zₙ where each Zᵢ is affected only by Gᵢ and invariant to all Gⱼ (j ≠ i). The SDP's partition structure (volume 24d, location 8d, shape 12d, residual 84d) maps directly onto this formalism, with each semantic factor corresponding to a subgroup of transformations acting on the tumor state space. This provides the SDP with **principled group-theoretic grounding** rather than ad hoc design.

For quantitative evaluation, Eastwood and Williams, "A Framework for the Quantitative Evaluation of Disentangled Representations," *ICLR 2018*, defined the DCI framework. A feature importance matrix R is constructed by training regressors (LASSO or Random Forest) from each latent dimension to each ground-truth factor. **Disentanglement** measures whether each code dimension encodes at most one factor (via column-wise entropy of normalized R). **Completeness** measures whether each factor is captured by at most one code dimension (row-wise entropy). **Informativeness** measures prediction accuracy. For the SDP, a well-disentangled projection should produce a block-diagonal R matrix aligned with the partition structure.

### Recent medical imaging work validates the approach

Liu et al., "Learning Disentangled Representations in the Imaging Domain," *Medical Image Analysis* 80:102516, 2022, provide a comprehensive tutorial-survey confirming that the dominant paradigm in medical imaging disentanglement is content-style separation. Several recent papers demonstrate partition-style disentanglement specifically for brain tumors:

- Zhou, "Multi-modal brain tumor segmentation via disentangled representation learning and region-aware contrastive learning," *Pattern Recognition* 149:110252, 2024 — decomposes features into tumor-region-specific subspaces (enhancing tumor, tumor core, whole tumor) with BraTS 2018/2019 evaluation.
- Yang et al., "D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities," *IEEE TMI* 41(10):2953–2964, 2022 — dual disentanglement of modality-specific and tumor-region features.
- Ouyang et al., "Representation Disentanglement for Multi-modal Brain MRI Analysis," *IPMI 2021*, LNCS 12729, pp. 321–333 — explicitly separates anatomical (shape) from modality (appearance) information in brain MRI.

These works demonstrate that the SDP's strategy of explicit supervised partition-based disentanglement for brain MRI is well-aligned with current methodological trends in the field.

---

## 2. VICReg within partitions, distance correlation between them

### VICReg formulation and its application to structured latent spaces

Bardes, Ponce, and LeCun, "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning," *ICLR 2022* (arXiv:2105.04906), define three loss components operating on batch embeddings Z ∈ ℝ^{N×d}:

**Variance term** (hinge loss on per-dimension standard deviation): v(Z) = (1/d) Σⱼ max(0, γ − √(Var(zʲ) + ε)), where γ = 1. This prevents dimensional collapse by ensuring each dimension maintains unit-scale variance. **Covariance term** (off-diagonal penalty): c(Z) = (1/d) Σᵢ≠ⱼ [C(Z)]²ᵢⱼ, where C(Z) is the batch covariance matrix. This decorrelates dimensions, preventing redundancy. The default weights are λ = μ = 25 for variance/invariance and ν = 1 for covariance.

For the SDP, these terms apply **within each partition independently**: v(Z_vol), c(Z_vol), v(Z_loc), c(Z_loc), etc. This ensures non-collapse and decorrelation within each semantic subspace. Ben-Shaul et al., "An Information-Theoretic Perspective on VICReg," *NeurIPS 2023*, showed that VICReg's combined variance + covariance terms approximate maximization of log-det of the covariance matrix — a Gaussian entropy upper bound — providing information-theoretic justification. Shwartz-Ziv et al., "Variance-Covariance Regularization Improves Representation Learning," *arXiv:2306.13292*, 2023, demonstrated that VICReg's variance and covariance terms improve supervised learning by addressing gradient starvation and neural collapse.

A critical insight: VICReg's covariance term captures only **linear** decorrelation. For cross-partition independence — the stronger requirement that volume, location, shape, and residual subspaces are statistically independent — a nonlinear measure is needed.

### Distance correlation captures all nonlinear dependencies

Székely, Rizzo, and Bakirov, "Measuring and Testing Dependence by Correlation of Distances," *Annals of Statistics* 35(6):2769–2794, 2007, introduced distance correlation (dCor), defined through pairwise distance matrices and their double-centering. The critical property: **dCor(X, Y) = 0 if and only if X and Y are statistically independent** (for random vectors with finite first moments). Unlike Pearson correlation, dCor detects arbitrary nonlinear dependencies. It works for vectors of different dimensions — ideal for comparing the 24d volume partition against the 8d location partition.

The empirical computation involves O(n²) pairwise distance matrices, which for n = 800 produces 640,000 entries per partition pair. Across all C(4,2) = 6 cross-partition pairs, this requires approximately **30 MB of GPU memory** — computationally trivial. The implementation is straightforward in PyTorch:

```python
def dcov_sq(X, Y):
    n = X.shape[0]
    A = torch.cdist(X, X)
    B = torch.cdist(Y, Y)
    A = A - A.mean(0, keepdim=True) - A.mean(1, keepdim=True) + A.mean()
    B = B - B.mean(0, keepdim=True) - B.mean(1, keepdim=True) + B.mean()
    return (A * B).sum() / (n * n)
```

Kasieczka and Shih, "DisCo Fever: Robust Networks Through Distance Correlation," *Physical Review Letters* 125(12):122001, 2020, established dCor as a neural network regularizer, showing it maintains a **convex objective** when added to classification losses — far simpler and more stable than adversarial approaches. Zhen et al., "On the Versatile Uses of Partial Distance Correlation in Deep Learning," *ECCV 2022*, LNCS 13686:327–346, extended this to disentangled representation learning and confounding removal.

### dCor outperforms alternatives for this regime

For n = 800 with multivariate partitions, three independence measures were evaluated:

- **dCor**: Parameter-free, O(n²), trivially differentiable, proven as a regularizer. Captures all nonlinear dependencies.
- **HSIC** (Gretton et al., "Measuring Statistical Dependence with Hilbert-Schmidt Norms," *ALT 2005*, LNAI 3734:63–77): Requires kernel bandwidth selection (additional hyperparameter), O(n²), same theoretical guarantees with universal kernels. A strong alternative if kernel flexibility is desired.
- **MINE** (Belghazi et al., "MINE: Mutual Information Neural Estimation," *ICML 2018*): Requires training an auxiliary network, known to have variance growing exponentially with true MI (Song and Ermon, 2020). **Not recommended** for n = 800 due to high variance and instability.

The recommended composite loss is:

**L_total = L_regression + α·Σ_k v(Z_k) + β·Σ_k c(Z_k) + γ·Σ_{k<l} dCor²(Z_k, Z_l)**

where v and c are within-partition VICReg terms, dCor² enforces cross-partition independence, and L_regression provides supervised semantic anchoring. Starting weights: α ≈ 10–25, β ≈ 1, γ ≈ 1–10, with tuning guided by monitoring dCor values during training.

---

## 3. Spectral normalization constrains the projection to a 1-Lipschitz map

Miyato et al., "Spectral Normalization for Generative Adversarial Networks," *ICLR 2018* (arXiv:1802.05957), replace each weight matrix W with W̄ = W/σ(W), where σ(W) is the largest singular value, approximated via a single power iteration per training step (negligible overhead). For the 2-layer SDP MLP (768→512→128) with ReLU activations (1-Lipschitz), the overall Lipschitz bound is **Lip(f) ≤ 1 × 1 = 1**, meaning ‖z₁ − z₂‖ ≤ ‖h₁ − h₂‖ for all input pairs. The mapping is a contraction: it cannot amplify distances between encoder features.

This has three concrete benefits. First, generalization bounds from Bartlett et al., "Spectrally-normalized margin bounds for neural networks," *NeurIPS 2017*, scale with the product of spectral norms — fixing all to 1 yields tighter bounds. Gouk et al., "Regularisation of neural networks by enforcing Lipschitz continuity," *Machine Learning* 110:393–416, 2021, showed Lipschitz constraints are **particularly beneficial with limited training data**. Second, for downstream Neural ODE use, by the Picard-Lindelöf theorem, solution sensitivity satisfies ‖z₁(t) − z₂(t)‖ ≤ e^{Lt}·‖z₁(0) − z₂(0)‖. A 1-Lipschitz projection ensures well-conditioned initial conditions z(0), reducing ODE solver instability and the number of function evaluations (Finlay et al., "How to Train Your Neural ODE," *ICML 2020*). Third, the contraction property prevents the MLP from creating artificial high-frequency structure in the latent space that could confuse downstream GP models.

Among alternatives — gradient penalty (Gulrajani et al., "Improved Training of Wasserstein GANs," *NeurIPS 2017*), weight clipping (Arjovsky et al., "Wasserstein GANs," *ICML 2017*), and orthogonal regularization (Brock et al., "Large Scale GAN Training," *ICLR 2019*) — spectral normalization is the clear winner for a projection MLP: it provides a hard constraint, adds ~1% computational overhead, and preserves the relative spectral structure of weights unlike orthogonal regularization which forces all singular values to 1.

---

## 4. The 800-sample regime is manageable with proper regularization

The SDP MLP has approximately **459,000 parameters** (768×512 + 512 + 512×128 + 128) for 800 training samples — a parameter-to-sample ratio of ~574:1. While classical statistics would predict catastrophic overfitting, several factors mitigate this.

**Frozen encoder features dramatically reduce effective complexity.** The h ∈ ℝ^768 vectors from a pre-trained BrainSegFounder SwinUNETR are already structured, semantically rich representations — not raw pixel values. The task is supervised dimensionality reduction of pre-extracted features, analogous to transfer learning where even ~100 samples suffice for fine-tuning. The effective intrinsic dimensionality of h is likely far below 768.

**Spectral normalization constrains the function class.** With Lip(f) ≤ 1, the space of learnable functions is severely restricted regardless of parameter count. Combined with weight decay (recommended λ = **1e-3 to 1e-2** for this regime), the effective capacity is well below the nominal parameter count. The double descent phenomenon (Belkin et al., "Reconciling modern machine-learning practice and the classical bias-variance trade-off," *PNAS* 116(32):15849–15854, 2019; Nakkiran et al., "Deep Double Descent," *ICLR 2020*) also suggests that heavily overparameterized models can generalize well with proper regularization.

**The supervised partition structure decomposes the problem.** Each partition is trained with its own regression target, effectively reducing the problem to several smaller supervised dimensionality reductions: 768→24 for volume (~18K params), 768→8 for location (~6K params), 768→12 for shape (~9K params). Each sub-problem has a more favorable sample-to-effective-parameter ratio.

Practical recommendations for training in this regime, synthesized across multiple sources:

- **Batch size 32–64** (not full-batch): Keskar et al., "On Large-Batch Training for Deep Learning," *ICLR 2017*, showed large batches converge to sharp minima that generalize poorly. With n = 800, batch size 64 yields 12.5 iterations/epoch with beneficial gradient noise. The noise scale g ≈ ε(N/B − 1) drops to zero for full-batch training, eliminating this implicit regularization.
- **Dropout p = 0.3–0.5** on the hidden layer (512d), but not the output layer (128d). Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *JMLR* 15:1929–1958, 2014.
- **Feature-space Mixup** (Zhang et al., "mixup: Beyond Empirical Risk Minimization," *ICLR 2018*; Verma et al., "Manifold Mixup," *ICML 2019*): Creating convex combinations h̃ = λh_i + (1−λ)h_j with labels ỹ = λy_i + (1−λ)y_j effectively multiplies training data by O(N²) virtual pairs. Use α = 0.2–0.4 for the Beta distribution. Apply to h-space (before the MLP), so the MLP must produce structured z even from interpolated inputs.
- **Gaussian noise injection**: h̃ = h + ε, ε ~ N(0, σ²I) with σ = 1–5% of per-feature standard deviation. Equivalent to Tikhonov regularization (Bishop, 1995).
- **Early stopping** with patience 15–30 epochs, using 10–20% of data for validation.
- **Architecture**: Consider reducing hidden width from 512 to **256** (228K params, ratio ~285:1) as a more conservative option. Alternatively, a single-layer linear projection (768→128, ~98K params) provides the strongest sample-efficiency baseline.

---

## 5. Tracing latent partitions back to input MRI volumes

Verifying that partition semantics align with intention requires tracing latent activations back to input voxels. The most effective approach combines Jacobian analysis of the projection layer with gradient-based attribution through the full encoder chain.

### Integrated Gradients is the best primary method

Sundararajan, Taly, and Yan, "Axiomatic Attribution for Deep Networks," *ICML 2017*, vol. 70, pp. 3319–3328, defined Integrated Gradients (IG) as the path integral of gradients from a baseline to the input. IG satisfies **sensitivity** (changed features get non-zero attribution) and **implementation invariance** (functionally equivalent networks produce identical attributions). For 3D MRI volumes, the baseline is typically a zero-valued volume, with 50–300 interpolation steps. Critically, IG works for **any scalar target**, not just classification logits. For partition-specific attribution:

```python
class PartitionModel(nn.Module):
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return z[..., start:end].sum(dim=-1)  # scalar for IG
```

The **Captum library** (Kokhlikyan et al., "Captum: A Unified and Generic Model Interpretability Library for PyTorch," *arXiv:2009.07896*, 2020) supports 3D input volumes natively — all attribution methods operate on arbitrary-dimensional PyTorch tensors. IntegratedGradients, Saliency, GradientShap, and LayerIntegratedGradients are the most applicable for the SDP. The Occlusion method accepts 3D `sliding_window_shapes` tuples but is very expensive for 3D (e.g., ~1,728 forward passes for a 96³ volume with stride 8 and 16³ patches).

### Jacobian analysis reveals projection structure directly

For the projection layer, the Jacobian ∂z/∂h equals the weight matrix W. SVD of partition-specific submatrices (e.g., W_vol ∈ ℝ^{24×768}) reveals which encoder feature directions most influence each partition. The singular values indicate information bandwidth — how much encoder variation each partition captures. This analysis is computationally free and highly interpretable.

The full chain Jacobian ∂z/∂x through SwinUNETR is expensive (128 × ~884K entries for 96³ input) but can be computed via **vector-Jacobian products**: 128 backward passes yield all rows, which is feasible. Wei et al., "Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation," *ICCV 2021*, pp. 6721–6730, showed that orthogonal Jacobian vectors across dimensions indicate good disentanglement — a post-hoc diagnostic for the SDP.

### Linear probing is the most direct semantic verification

Alain and Bengio, "Understanding Intermediate Layers Using Linear Classifier Probes," *ICLR Workshop 2017* (arXiv:1610.01644), established linear probing as the standard approach. For the SDP:

- Train linear regressors from z_vol → known tumor volume: high R² confirms volume partition captures volumetric information.
- Train linear regressors from z_loc → center-of-mass coordinates: confirms spatial encoding.
- Train linear regressors from z_shape → shape descriptors (e.g., surface-area-to-volume ratio, sphericity): confirms morphological encoding.
- **Cross-probing** (z_vol → location features) should yield low R², confirming disentanglement.

Cetin et al., "Attri-VAE: Attribute-based Interpretable Representations of Medical Images with Variational Autoencoders," *Computerized Medical Imaging and Graphics* 104:102158, 2023, demonstrated this exact approach for cardiac imaging, using modularity, MIG, and SAP metrics to verify attribute-specific latent dimensions.

---

## 6. Ablation plan grounded in the disentanglement literature

### Standard methodology from foundational papers

The disentanglement literature provides clear precedent for ablation design. Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework," *ICLR 2017*, varied β ∈ {1, 2, 4, 8, 16} to establish the disentanglement-reconstruction tradeoff. Kim and Mnih, "Disentangling by Factorising," *ICML 2018* (arXiv:1802.05983), ablated γ ∈ {6, 10, 20, 40} for the total correlation penalty, finding γ = 40 optimal. Chen et al., "Isolating Sources of Disentanglement in VAEs," *NeurIPS 2018*, decomposed the ELBO and showed only the TC term (β) matters — α and γ have minimal effect. Locatello et al. (2019) found that random seed variation explains more variance than method or hyperparameter choice, making **multiple seeds mandatory** (minimum 3, ideally 5).

### Recommended three-tier ablation plan

**Tier 1 — Essential (6 configurations, each with 3 seeds = 18 runs):**

1. Full model (all losses active) — baseline
2. No supervised regression loss — tests necessity of semantic anchoring
3. No VICReg (variance + covariance) — tests necessity of within-partition regularization
4. No cross-partition dCor — tests necessity of independence enforcement
5. Supervised regression only — lower bound without any regularization
6. VICReg only (no supervision, no dCor) — tests whether regularization alone creates structure

**Tier 2 — Important (4 configurations):**

7. Latent dimension d ∈ {64, 128, 256} with proportional partition scaling (volume: 18.75%, location: 6.25%, shape: 9.375%, residual: remainder)
8. VICReg variance-only vs. covariance-only (dissecting the two components)
9. dCor weight sensitivity: γ ∈ {0.01, 0.1, 1.0}
10. Curriculum scheduling vs. all-at-once training

**Tier 3 — Nice-to-have (3 configurations):**

11. Partition reallocation (e.g., volume 32d, location 12d, shape 20d, residual 64d)
12. Architecture depth (1-layer linear vs. 2-layer MLP)
13. Cyclical annealing (Fu et al., "Cyclical Annealing Schedule," *NAACL 2019*, pp. 240–250) vs. monotonic loss scheduling

### Curriculum learning schedule

Based on Burgess et al., "Understanding disentangling in β-VAE," *arXiv:1804.03599*, 2018, and the logic that supervised signals should establish partition semantics before independence regularization is imposed:

- **Phase 1 (epochs 1–20):** Supervised regression only — anchors partition semantics
- **Phase 2 (epochs 20–50):** Linearly ramp VICReg variance and covariance from 0→full weight — prevents collapse as partitions consolidate
- **Phase 3 (epochs 50–100):** Linearly ramp cross-partition dCor from 0→full weight — enforces independence after partitions have learned their primary semantics
- **Phase 4 (epochs 100+):** All losses at full weight — joint optimization to convergence

### Evaluation metrics adapted for partition-based disentanglement

For partition-based (vector-wise) rather than dimension-wise disentanglement, standard metrics require adaptation:

- **DCI** (Eastwood and Williams, 2018) is the most directly applicable. Train LASSO/RF regressors from all 128 latent dimensions to each factor — the importance matrix should exhibit block structure. Both D and C scores capture partition alignment.
- **MIG** (Chen et al., 2018): Adapt to partition level by computing MI between each partition (as a whole) and each factor, then measuring the gap between the best and second-best partition per factor. Note: with 800 samples and 128 dimensions, MI estimation via binning can be noisy — use non-parametric KSG estimators.
- **SAP** (Kumar et al., "Variational Inference of Disentangled Latent Concepts from Unlabeled Observations," *ICLR 2018*): Compute at partition level, using R² of linear regression from each partition to each factor.
- **Cross-partition dCor matrix**: Report the 4×4 matrix of pairwise distance correlations between partitions after training — a direct independence diagnostic.

Carbonneau et al., "Measuring Disentanglement: A Review of Metrics," *IEEE TNNLS*, 2022 (arXiv:2012.09276), recommend using multiple complementary metrics, as predictor-based metrics (DCI, SAP) are limited by predictor linearity while information-based metrics (MIG, Modularity) struggle with nonlinear factor-code relations.

---

## 7. JEPA provides the conceptual framework for latent-space prediction

LeCun, "A Path Towards Autonomous Machine Intelligence," *OpenReview Preprint*, 2022, proposed the Joint Embedding Predictive Architecture (JEPA) as the foundation for learning world models. The core idea: two variables x and y are encoded into representations s_x and s_y, and a predictor estimates s_y from s_x. Predictions occur in **representation space, not pixel space**, allowing encoders to eliminate irrelevant details. LeCun argued that generative models waste capacity predicting unpredictable details — "the texture of carpet, leaves moving in wind" — while JEPA focuses on semantically meaningful prediction.

Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," *CVPR 2023*, pp. 15619–15629 (I-JEPA), demonstrated this for images: a ViT context encoder processes visible patches, and a narrow ViT predictor predicts masked target representations (computed by an EMA target encoder) using L1 loss. V-JEPA (Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video," *TMLR 2024*) extended this to video, predicting masked spatiotemporal regions in latent space, demonstrating that **temporal dynamics are naturally captured by latent-space prediction**.

The SDP pipeline — SwinUNETR encoder → SDP projection → structured latent → GP/LME temporal prediction — is conceptually aligned with JEPA's "predict in latent space" principle. Both systems avoid pixel-level prediction. The key differences are instructive: JEPA uses neural predictors trained end-to-end, while the SDP uses GPs/LMEs that provide closed-form uncertainty quantification and handle the irregular time intervals inherent in clinical longitudinal data (42 patients, months-to-years between scans). The SDP's explicit semantic partitioning is absent in JEPA, but LeCun's hierarchical JEPA concept — different abstraction levels for different prediction horizons — resonates with the partition-specific dynamics idea (volume evolving slowly, shape changing more rapidly).

The energy-based perspective on the composite loss is also valuable: the combination of regression, variance, covariance, and independence terms defines an energy landscape where **low energy corresponds to representations that are both faithful and well-structured**, paralleling JEPA's non-contrastive training criteria that jointly maximize informativeness and predictability while preventing collapse.

---

## 8. Structured latent spaces enable partition-aware longitudinal dynamics

### Direct precedent from medical imaging

Several recent works directly validate the SDP's approach of predicting disease progression in structured latent spaces:

**Zhao et al., "Longitudinal Self-Supervised Learning," *Medical Image Analysis* 71:102051, 2021**, is the most directly relevant precedent. LSSL disentangles brain-age factors from latent MRI representations using self-supervised learning on 811 ADNI subjects with up to 8 scans each. Changes in a single factor induce change along one direction in representation space, revealing accelerated aging effects of Alzheimer's. This demonstrates that disentangled latent spaces enable meaningful longitudinal analysis of brain MRI.

**Puglisi et al., "Brain Latent Progression," *Medical Image Analysis*, 2025**, uses a **linear mixed-effects model as the temporal predictor** in a small latent space for 3D brain MRI disease progression — architecturally identical to the SDP's downstream LME approach. Trained on 11,730 MRIs from 2,805 subjects, it demonstrates that LME in well-structured latent spaces produces clinically meaningful progression predictions.

Recent tumor growth prediction work further supports latent-space approaches: Chen et al., "Vestibular Schwannoma Growth Prediction from Longitudinal MRI by Time-Conditioned Neural Fields," *MICCAI 2024* (arXiv:2404.02614), encodes tumors into low-dimensional latent codes and predicts future codes using time-conditioned ConvLSTM, finding that **latent-space prediction outperforms direct image-space prediction**. Zhang et al., "Spatio-Temporal Convolutional LSTMs for Tumor Growth Prediction," *IEEE TMI* 39(4):1114–1126, 2020, achieved 83.2% Dice on 33 patients, while Elazab et al., "GP-GAN: Brain Tumor Growth Prediction Using Stacked 3D GANs from Longitudinal MR Images," *Neural Networks* 132:321–332, 2020, combined GPs with GANs for growth prediction.

### Partition structure naturally maps to MOGP kernel design

The SDP's partition structure enables a principled mapping to multi-output GP kernels. Following Álvarez et al., "Kernels for Vector-Valued Functions: A Review," *Foundations and Trends in Machine Learning* 4(3):195–266, 2012, the **Linear Model of Coregionalization (LMC)** expresses cross-output covariances as K = Σ_q B_q ⊗ k_q, where B_q are coregionalization matrices and k_q are base kernels. The SDP partition structure naturally yields a sum-of-separable-kernels architecture:

**k_total(z, z', t, t') = k_vol(z_vol)·k_t_vol(t, t') + k_loc(z_loc)·k_t_loc(t, t') + k_shape(z_shape)·k_t_shape(t, t') + k_res(z_res)·k_t_res(t, t')**

Each partition gets a **partition-specific temporal kernel** with independently optimized hyperparameters. Volume dynamics might use a Matérn kernel with long lengthscale (slow growth), location dynamics a linear + noise kernel (drift), shape dynamics a squared-exponential with shorter lengthscale (morphological changes), and residual a pure noise kernel. The block-diagonal coregionalization matrix B = diag(B_vol, B_loc, B_shape, B_res) reduces parameters from O(128²) = 16,384 to O(24² + 8² + 12² + 84²) ≈ **7,876** — a dramatic reduction critical for the 42-patient downstream cohort.

**Disentanglement provably helps GP performance.** Independent latent dimensions enable Automatic Relevance Determination (ARD) kernel decomposition k(z, z') = Π_i k_i(z_i, z'_i; ℓ_i), where each lengthscale ℓ_i can be independently optimized and irrelevant dimensions automatically "turned off." With only 42 patients and sparse timepoints, this reduction in effective hyperparameters is essential for identifiability (Rasmussen and Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006; Bonilla et al., "Multi-task Gaussian Process Prediction," *NeurIPS 2007*).

The Neural ODE connection also reinforces the latent-space temporal approach. Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series," *NeurIPS 2019* (arXiv:1907.03907), showed that Latent ODEs naturally handle arbitrary time gaps — precisely the irregular MRI intervals in the Andalusian cohort. If Neural ODEs are explored as alternatives to GPs, the spectrally normalized SDP ensures well-conditioned initial conditions z(0), reducing trajectory divergence per Grönwall's inequality.

---

## 9. Tools and libraries for implementation

**GPyTorch** (Gardner et al., "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration," *NeurIPS 2018*) is recommended over GPy for downstream GP modeling. It provides GPU acceleration via the PyTorch backend, scalable inference (KISS-GP, LOVE), and native integration with the PyTorch ecosystem including autograd and optimizers — enabling potential end-to-end differentiable pipelines. **GPy** (Sheffield ML group, http://github.com/SheffieldML/GPy) remains useful for multi-output GP models via the coregionalization API and has a more mature GPLVM implementation, but is CPU-only with O(n³) scaling.

For distance correlation, the **dcor library** (Ramos-Carreño and Torrecilla, "dcor: Distance correlation and energy statistics in Python," *SoftwareX* 22:101326, 2023) implements both biased/unbiased estimators and fast O(n log n) algorithms for univariate cases. For differentiable training loss, implement dCor directly in PyTorch using `torch.cdist` and double-centering (as shown in Section 2), since the dcor library operates on NumPy arrays and does not support autograd.

**Captum** (Kokhlikyan et al., 2020) provides all necessary attribution methods for 3D volumes. **disentanglement_lib** (Google, associated with Locatello et al. 2019) implements DCI, MIG, SAP, and Modularity metrics in TensorFlow/NumPy — extract the metric computation code and adapt to PyTorch. The **disentangling-vae** repository by YannDubs (GitHub) provides a cleaner PyTorch implementation of 5 VAE losses with all disentanglement metrics, more modern than disentanglement_lib.

For experiment tracking, Weights & Biases or TensorBoard should log all ablation experiments systematically, tracking loss components, dCor matrices, DCI scores, and partition-level R² across training.

---

## Conclusion

The SDP module's design is not just pragmatically motivated but theoretically grounded at every level. The impossibility theorem makes supervision a formal requirement, not a convenience. Distance correlation provides the strongest independence guarantee available for the 800-sample regime without hyperparameter overhead. Spectral normalization's 1-Lipschitz bound flows directly into ODE stability theory and generalization bounds. The partition structure maps naturally onto MOGP kernel decompositions that are essential for the 42-patient downstream task. Three less obvious insights emerge from this research synthesis: first, the curriculum schedule (supervised-first, then regularization) has strong theoretical support from the capacity-annealing literature and should be treated as a core design element rather than an optional tweak; second, the 84-dimensional residual partition, lacking supervision, poses the greatest overfitting risk and should receive KL or information-bottleneck regularization; third, the SDP's connection to JEPA's "predict in latent space" principle — combined with the Puglisi et al. demonstration that LME models work well in structured brain MRI latent spaces — provides a strong narrative frame for positioning this work within the broader representation learning literature.