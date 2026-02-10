# Methodology: Foundation Model Adaptation for Meningioma Growth Forecasting

**Project:** MenGrowth-Model — BSc Thesis  
**Version:** 3.0 (Modular Agent-Ready)  
**Date:** February 2026  
**Objective:** Adapt BrainSegFounder (a glioma-pretrained 3D brain MRI foundation model) into a meningioma-domain encoder with smooth, disentangled latent representations suitable for downstream Neural ODE growth forecasting.

---

## Document Structure

This document is designed as both a standalone scientific methods specification and a modular task decomposition for agentic coding. Each section follows this template:

1. **Background** — theoretical reasoning, literature support, mathematical formulations.
2. **Data** — which datasets and splits are used, input/output tensor contracts.
3. **Outputs** — deliverables (checkpoints, metrics, figures).
4. **Code Requirements** — what to implement (not the code itself).
5. **Verification Tests** — automated checks the coding agent runs to confirm correctness.
6. **Analysis** — statistical and visual assessments to evaluate this module's contribution.

Sections are designed to be executed sequentially. Each section's outputs feed into the next section's inputs.

---

## Table of Contents

- [Section 0: Data Infrastructure and Preprocessing](#section-0-data-infrastructure-and-preprocessing)
- [Section 1: Domain Gap Analysis — Glioma to Meningioma](#section-1-domain-gap-analysis--glioma-to-meningioma)
- [Section 2: Phase 1 — LoRA Encoder Adaptation](#section-2-phase-1--lora-encoder-adaptation)
- [Section 3: Phase 2 — Supervised Disentangled Projection](#section-3-phase-2--supervised-disentangled-projection)
- [Section 4: Phase 3 — Cohort Encoding and Harmonization](#section-4-phase-3--cohort-encoding-and-harmonization)
- [Section 5: Phase 4 — Neural ODE Growth Forecasting](#section-5-phase-4--neural-ode-growth-forecasting)
- [Section 6: End-to-End Evaluation Framework](#section-6-end-to-end-evaluation-framework)
- [Appendix A: Common Configuration Contract](#appendix-a-common-configuration-contract)
- [Appendix B: References](#appendix-b-references)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE                                    │
│                                                                             │
│  Section 1: Domain Gap Analysis (BraTS-GLI vs BraTS-MEN, frozen encoder)   │
│  ┌──────────────┐    ┌────────────┐    ┌──────────────┐                    │
│  │ BraTS-GLI    │───→│ Frozen BSF │───→│ Feature      │───→ UMAP, MMD,    │
│  │ BraTS-MEN    │    │ Encoder    │    │ Extraction   │    CKA metrics    │
│  └──────────────┘    └────────────┘    └──────────────┘                    │
│                                                                             │
│  Section 2: Phase 1 — Encoder Adaptation (BraTS-MEN, 500+100 subjects)     │
│  ┌──────────────┐    ┌────────────┐    ┌────────────┐                      │
│  │ BraTS-MEN MRI│───→│ SwinViT    │───→│ Seg Head   │───→ L_dice + L_CE   │
│  │ [B,4,128³]   │    │ (LoRA r=8) │    │ (discard)  │    + λ_aux·L_sem   │
│  └──────────────┘    └────────────┘    └────────────┘                      │
│                                                                             │
│  Section 3: Phase 2 — Disentangled Projection (BraTS-MEN, 800 subjects)    │
│  ┌──────────────┐    ┌────────────┐    ┌─────────┐                         │
│  │ BraTS-MEN MRI│───→│ SwinViT    │───→│ SDP MLP │───→ z ∈ ℝ^128          │
│  │ [B,4,128³]   │    │ (frozen)   │    │ (train) │    ↓                   │
│  └──────────────┘    └────────────┘    └─────────┘    L_sem+L_cov+L_var   │
│                                                        +L_dCor             │
│                                                                             │
│  Section 4: Phase 3 — Encoding + Harmonization (Private Cohort)            │
│  ┌──────────────┐    ┌────────────┐    ┌─────────┐    ┌───────┐           │
│  │ Andalusian   │───→│ SwinViT    │───→│ SDP MLP │───→│ComBat │→ z*      │
│  │ [all t_k]    │    │ (frozen)   │    │(frozen) │    │       │           │
│  └──────────────┘    └────────────┘    └─────────┘    └───────┘           │
│                                                                             │
│  Section 5: Phase 4 — Neural ODE (Private Cohort trajectories)             │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │ z*(t₀) ──→ ODESolve(f_θ, z*(t₀), t₀, t₁) ──→ ẑ(t₁)  │              │
│  │              ↓                                           │              │
│  │       Gompertz-informed partition-aware dynamics         │              │
│  └──────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 0: Data Infrastructure and Preprocessing

### 0.1 Background

The pipeline requires two datasets: the public BraTS-MEN 2024 challenge dataset (cross-sectional, $N = 1000$ subjects) and a private longitudinal Andalusian cohort ($N = 42$ patients, $N_{\text{studies}} = 137$, mean studies/patient $= 3.26 \pm 1.29$). Both datasets must conform to a common tensor contract to ensure that the same encoder processes both identically.

BrainSegFounder (Cox et al., "BrainSegFounder: Towards Foundation Models for Neuroimage Segmentation," Medical Image Analysis, 2024) was pretrained on UK Biobank data registered to MNI152 space and fine-tuned on BraTS 2021 data in SRI24 space. BraTS-MEN follows the same SRI24 preprocessing pipeline as BraTS 2021. The Andalusian cohort has been preprocessed following the BraTS challenge convention (co-registered, SRI24 template, skull-stripped, 1mm isotropic resampling), making it directly compatible.

**BraTS-MEN Label Convention.** BraTS-MEN 2024 (LaBella et al., "The ASNR-MICCAI BraTS Meningioma Challenge," arXiv, 2024) uses the same label encoding as BraTS 2021 glioma: $\{0 = \text{background}, 1 = \text{NCR}, 2 = \text{ED}, 3 = \text{ET}\}$. The hierarchical overlapping output convention is: Ch0 = TC (Tumor Core = NCR ∪ ET), Ch1 = WT (Whole Tumor = NCR ∪ ED ∪ ET), Ch2 = ET (Enhancing Tumor). This is identical to BrainSegFounder's training convention, enabling direct checkpoint reuse.

> **Meningioma sub-region semantics.** Although the label encoding is identical, the biological interpretation differs. Meningiomas typically exhibit homogeneous enhancement (ET dominates) with less necrosis (NCR) and peritumoral edema (ED) than gliomas. This implies: (i) sub-region prevalence distributions will differ from glioma training data, (ii) volume features $V_{\text{NCR}}$ and $V_{\text{ED}}$ may have higher zero-prevalence rates, and (iii) the semantic feature normalization (Section 3) must account for this. These differences are characterized in Section 0.6 (Analysis).

**Andalusian Cohort.** 42 patients from Hospital Regional Universitario de Málaga and associated institutions in the Andalusian Healthcare System. Multi-institutional acquisition with approximately 10 distinct MRI scanners. Each study includes all 4 modalities (T1n, T1c, T2w, T2-FLAIR) and expert-annotated segmentation labels. All studies are preprocessed following the University of Pennsylvania BraTS preprocessing pipeline: rigid intra-subject co-registration, SRI24 atlas alignment, skull-stripping, and resampling to 1 mm isotropic resolution. This ensures direct compatibility with BraTS-MEN data and the BrainSegFounder input convention.

### 0.2 Data

| Dataset | $N_{\text{subjects}}$ | $N_{\text{studies}}$ | Modalities | Labels | Use |
|---------|----------------------|---------------------|------------|--------|-----|
| BraTS-MEN 2024 | 1000 | 1000 | T1n, T1c, T2w, T2-FLAIR | Seg (NCR/ED/ET) | Phases 1–2 |
| Andalusian Cohort | 42 | 137 | T1n, T1c, T2w, T2-FLAIR | Seg (NCR/ED/ET) | Phases 3–4 |
| BraTS 2021 (GLI) | 1251 | 1251 | T1n, T1c, T2w, T2-FLAIR | Seg (NCR/ED/ET) | Domain gap analysis only |

**Common Tensor Contract:**

```
Input per subject:  x ∈ ℝ^{B × 4 × 128 × 128 × 128}
Channel order:      [t2f, t1c, t1n, t2w]  (matching BrainSegFounder)
Intensity:          Z-score normalized per subject (channel-wise, nonzero voxels)
Spatial:            Isotropic 1mm, RAS orientation, CropForeground + 128³ ROI
Labels (input):     {0=BG, 1=NCR, 2=ED, 3=ET}
Labels (output):    3-channel sigmoid: [TC, WT, ET] (hierarchical overlapping)
Training crops:     CropForeground + RandSpatialCrop(128³)
Validation:         CropForeground + CenterSpatialCrop(128³)
Inference:          Sliding window 128³ patches, stride 64, Gaussian blending
```

**BraTS-MEN Data Splits (fixed across all experiments):**

| Split | $N$ | Use |
|-------|-----|-----|
| `train_pool` | 800 | Phases 1–2 training (subdivided below) |
| ↳ `lora_train` | 500 | Phase 1 LoRA encoder training |
| ↳ `lora_val` | 100 | Phase 1 early stopping |
| ↳ `sdp_only` | 200 | Never seen during LoRA (strict eval subset) |
| `val` | 100 | Model selection, probe evaluation |
| `test` | 100 | Final held-out evaluation |

Phase 2 SDP training uses all 800 `train_pool` subjects (the encoder is frozen, so there is no gradient contamination from LoRA training). The `val` set is used for SDP model selection and probe $R^2$ evaluation. The `test` set is used only for final reporting. The `sdp_only` subset (200 subjects never seen during LoRA) enables a strict ablation evaluating whether SDP benefits from encoder-familiar vs. encoder-unfamiliar subjects.

### 0.3 Outputs

- `data_splits.json`: Fixed data split assignment for reproducibility.
- `semantic_features_cache/`: Precomputed volume, location, and shape features per subject.
- Preprocessing validation report: distribution summaries, missing data checks.

### 0.4 Code Requirements

1. **`BraTSMENDataset`** — MONAI-based dataset class loading 4-channel MRI + segmentation + semantic features. Must support both train and val transforms. Caching of semantic features to disk.
2. **`LongitudinalDataset`** — Dataset class for the Andalusian cohort handling multi-timepoint patient data with temporal metadata (acquisition dates, $\Delta t$ computation).
3. **`SemanticFeatureExtractor`** — Computes volume (log-transformed mm³ per sub-region), location (center of mass in physical coordinates), and shape (sphericity, surface area, solidity, 3 aspect ratios) from segmentation masks using `scipy.ndimage`.

> Note: The existing codebase (`semantic_features.py`) computes all 6 shape features via `compute_shape_features()` but the convenience function `compute_shape_array()` returns only 3 (excluding aspect ratios due to poor linear probe $R^2$). For SDP training, use all 6 features from `compute_shape_features()`, as the nonlinear SDP projection may capture aspect ratio information that linear probes cannot.
4. **`DataSplitter`** — Deterministic split generation with fixed seed. Saves and loads splits.
5. **`get_train_transforms()` / `get_val_transforms()`** — MONAI transform pipelines matching BrainSegFounder convention.

### 0.5 Verification Tests

```
TEST_0.1: Dataset loading
  - Load 5 random BraTS-MEN subjects
  - Assert output shape == [4, 128, 128, 128] (float32)
  - Assert label shape == [128, 128, 128] (int)
  - Assert unique label values ⊆ {0, 1, 2, 3}
  - Assert all 4 channels have nonzero variance

TEST_0.2: Semantic features
  - For each subject, compute semantic features
  - Assert V_total > 0 (tumor exists)
  - Assert centroid within volume bounds
  - Assert 0 < sphericity ≤ 1.0
  - Assert surface_area > 0

TEST_0.3: Data splits
  - Load splits, assert no overlap between any pair of sets
  - Assert |lora_train| + |lora_val| + |sdp_only| + |val| + |test| == 1000
  - Assert |train_pool| == |lora_train| + |lora_val| + |sdp_only| == 800
  - Assert splits are deterministic (reload and compare)

TEST_0.4: Transform consistency
  - Apply train transform 10 times to same subject
  - Assert output shapes are always [4, 128, 128, 128]
  - Apply val transform, assert deterministic output

TEST_0.5: Longitudinal dataset
  - Load 3 patients from Andalusian cohort
  - Assert each has ≥ 2 timepoints
  - Assert temporal metadata (dates) are monotonically ordered
  - Assert all timepoints have shape [4, 128, 128, 128]
```

### 0.6 Analysis

- **Semantic feature distributions**: Histogram of log-volumes, centroid coordinates, and shape metrics across BraTS-MEN. Report mean, std, range for each feature. These distributions define normalization parameters for Phase 2.
- **Label statistics**: Fraction of subjects with each sub-region present. Meningiomas may have different sub-region prevalence than gliomas (e.g., less necrosis).
- **Andalusian cohort temporal statistics**: Distribution of inter-study intervals ($\Delta t$ in months), number of timepoints per patient, total temporal span per patient.

---

## Section 1: Domain Gap Analysis — Glioma to Meningioma

### 1.1 Background

BrainSegFounder was pretrained on BraTS 2021 gliomas — intra-axial, infiltrative tumors with heterogeneous enhancement patterns and significant surrounding edema. Meningiomas are biologically distinct: extra-axial, well-circumscribed, homogeneously enhancing, and arising from the meninges. This domain shift is not merely distributional — it reflects fundamental differences in tumor morphology and imaging appearance.

The hypothesis is that the frozen BrainSegFounder encoder captures general brain anatomy features (from UK Biobank SSL pretraining) that are domain-invariant, but the glioma-specific fine-tuning (BraTS 2021 Stage 3) has shaped the high-level features toward glioma morphology. Formally, let $\mathcal{M}_{\text{GLI}}$ denote the feature manifold learned from glioma fine-tuning and $\mathcal{M}_{\text{MEN}}$ the ideal meningioma manifold. The domain gap is:

$$d(\mathcal{M}_{\text{GLI}}, \mathcal{M}_{\text{MEN}}) > 0$$

This section quantifies this gap using three complementary metrics, establishing the empirical necessity of domain adaptation (Section 2).

**Maximum Mean Discrepancy (MMD).** Given feature sets $\{h_i^{\text{GLI}}\}_{i=1}^{n}$ and $\{h_j^{\text{MEN}}\}_{j=1}^{m}$ extracted from the frozen encoder, the MMD with Gaussian kernel is (Gretton et al., "A Kernel Two-Sample Test," JMLR, 2012):

$$\text{MMD}^2(\mathcal{F}_{\text{GLI}}, \mathcal{F}_{\text{MEN}}) = \frac{1}{n^2}\sum_{i,i'} k(h_i, h_{i'}) - \frac{2}{nm}\sum_{i,j} k(h_i, h_j) + \frac{1}{m^2}\sum_{j,j'} k(h_j, h_{j'})$$

where $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$ with $\sigma$ set to the median pairwise distance. MMD = 0 if and only if the two distributions are identical.

**Centered Kernel Alignment (CKA).** CKA measures representational similarity between two feature matrices (Kornblith et al., "Similarity of Neural Network Representations Revisited," ICML, 2019):

$$\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

where $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^{m \times d}$ are centered feature matrices. CKA = 1 indicates identical representations.

**Linear Probe $R^2$.** A linear regression from frozen features to semantic targets (volume, location, shape) measures how accessible meningioma semantics are without adaptation. Negative $R^2$ indicates that the features encode meningioma semantics worse than a constant prediction, empirically demonstrating the domain gap.

### 1.2 Data

- **BraTS 2021 (GLI)**: Random subset of 200 subjects (features extracted from frozen BSF encoder).
- **BraTS-MEN 2024**: `val` split (100 subjects) + 100 from `sdp_only` (features extracted from frozen BSF encoder).
- Feature extraction level: `encoder10` output → GAP → $h \in \mathbb{R}^{768}$.

### 1.3 Outputs

- `domain_gap_report.json`: MMD value with permutation test $p$-value, CKA score, per-feature linear probe $R^2$.
- `domain_umap.png`: 2D UMAP visualization with GLI and MEN colored separately.
- `domain_feature_variance.png`: Per-dimension variance comparison between GLI and MEN features.

### 1.4 Code Requirements

1. **`DomainGapAnalyzer`** — Class that takes two sets of features and computes MMD (with permutation test, $n_{\text{perm}} = 1000$), CKA, and linear probe metrics.
2. **`FeatureExtractor`** — Extracts GAP-pooled features from the frozen BSF encoder for a given dataset. Caches to disk.
3. **`DomainVisualizer`** — UMAP plots, variance comparison plots, t-SNE alternative.

### 1.5 Verification Tests

```
TEST_1.1: Feature extraction
  - Extract features from 10 GLI and 10 MEN subjects
  - Assert shape == [N, 768] for each
  - Assert no NaN or Inf values
  - Assert feature variance > 0 for all dimensions

TEST_1.2: MMD computation
  - Compute MMD between two identical sets → assert MMD ≈ 0
  - Compute MMD between GLI and random noise → assert MMD >> 0
  - Compute MMD between GLI and MEN → assert 0 < MMD < MMD(GLI, noise)

TEST_1.3: Linear probe
  - Fit Ridge regression from frozen features → semantic targets
  - Assert returned R² values are finite
  - For frozen GLI encoder on MEN data: expect R² < 0.3 for volume
    (based on prior ablation showing negative R² for some features)

TEST_1.4: Permutation test
  - MMD permutation test produces p-value ∈ [0, 1]
  - With n_perm=100 (fast check), p-value for GLI vs MEN should be < 0.05
```

### 1.6 Analysis

This section produces the core empirical argument for domain adaptation:

1. **MMD with significance**: Report $\text{MMD}^2$ and permutation-test $p$-value. If $p < 0.01$, the domain gap is statistically significant.
2. **CKA**: Report CKA between GLI and MEN feature matrices. Values below 0.8 suggest substantial representational divergence.
3. **UMAP visualization**: Qualitative assessment of cluster separation. If GLI and MEN form distinct clusters, LoRA adaptation is clearly needed.
4. **Linear probe $R^2$ table**: Volume, location, and shape $R^2$ for frozen encoder on MEN data. Negative or near-zero values (as observed in the LoRA ablation v1) confirm that meningioma semantics are not linearly accessible from glioma-tuned features.
5. **Per-dimension variance ratio**: $\text{Var}(h_j^{\text{MEN}}) / \text{Var}(h_j^{\text{GLI}})$ across all 768 dimensions. Dimensions where the ratio deviates substantially from 1.0 indicate domain-specific feature usage.

**Expected outcome**: The frozen encoder's features will show statistically significant distributional shift (MMD $p < 0.01$), moderate CKA (0.5–0.8 range), and poor linear probe performance on meningioma semantics ($R^2_{\text{vol}} < 0.3$). This establishes the empirical motivation for LoRA adaptation in Section 2.

---

## Section 2: Phase 1 — LoRA Encoder Adaptation

### 2.1 Background

Given the domain gap established in Section 1, the encoder must be adapted from glioma to meningioma morphology. The adaptation must be parameter-efficient (to avoid catastrophic forgetting of the pretrained anatomy knowledge) and must preserve the encoder's capacity for rich feature extraction.

**BrainSegFounder Architecture.** The checkpoint is BrainSegFounder-Tiny (BSF-T, 62M parameters total, 9.0M in `swinViT`, `depths=(2,2,2,2)`, `feature_size=48`, bottleneck dimension 768). This variant is confirmed by the loaded parameter count (62.19M) and uniform depth configuration; BSF-Small (64M) uses `depths=(2,2,6,2)` with 6 Swin Blocks at Stage 2. The checkpoint (`finetuned_model_fold_0.pt`) achieves Dice 0.9027 on BraTS 2021 fold 0 (Cox et al., 2024, Table 8). BSF-T was pretrained in three stages: (Stage 1) SSL on 41,400 UK Biobank subjects, (Stage 2) SSL on 1,251 BraTS 2021 subjects, (Stage 3) supervised segmentation fine-tuning on BraTS 2021.

> **Note on BSF variant selection.** BSF-Small achieves marginally higher average Dice (0.9115 vs 0.9110) on BraTS 2021 gliomas. If the BSF-Small checkpoint is available, it may yield better downstream performance. However, the architectural difference (4 additional Swin Blocks in Stage 2) also increases the trainable parameter count under LoRA if Stage 2 is included.

The encoder produces hierarchical features (with 128³ input). The SwinUNETR architecture combines the SwinViT backbone (`swinViT`, 9.0M params) with UnetrBasicBlock skip connections (`encoder1`–`encoder4`) and a bottleneck block (`encoder10`, 31.85M params — two $3^3$ Conv-Norm-Act blocks at 768 channels):

```
Input: [B, 4, 128, 128, 128]
SwinViT Stage 0: [B, 48,  64, 64, 64]   ← Patch embed (4→48) + 2× Swin Blocks
SwinViT Stage 1: [B, 96,  32, 32, 32]   ← Patch merge + 2× Swin Blocks
SwinViT Stage 2: [B, 192, 16, 16, 16]   ← Patch merge + 2× Swin Blocks
SwinViT Stage 3: [B, 384,  8,  8,  8]   ← Patch merge + 2× Swin Blocks
SwinViT Stage 4: [B, 768,  4,  4,  4]   ← 2× Swin Blocks (deepest)
encoder10:        [B, 768,  4,  4,  4]   ← UnetrBasicBlock(768→768, 2× Conv3d)
```

Feature extraction: $h = \text{GAP}(\text{encoder10}(\text{SwinViT}_4(x))) \in \mathbb{R}^{768}$.

**Note on input resolution**: BrainSegFounder was pretrained with 96³ crops. We use 128³ to match the BrainSegFounder supervised fine-tuning configuration and BraTS-MEN conventions. The Swin Transformer uses relative position bias (not absolute positional embeddings), making it resolution-agnostic in principle. MONAI's SwinUNETR handles the different spatial dimensions by computing relative position indices on-the-fly. The Stage 4 output changes from $[768, 3, 3, 3]$ at 96³ to $[768, 4, 4, 4]$ at 128³.

**Low-Rank Adaptation (LoRA).** LoRA (Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR, 2022) injects trainable low-rank matrices into frozen weight matrices:

$$W' = W_{\text{pretrained}} + \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}$$

where $A$ is initialized from $\mathcal{N}(0, \sigma^2)$ and $B$ is initialized to zero, ensuring $W' = W_{\text{pretrained}}$ at initialization. The rank $r$ controls the capacity-efficiency tradeoff. Our ablation study indicates $r = 8$ provides optimal balance for this architecture.

**LoRA target modules**: Q, K, V projection matrices in all self-attention layers of Stages 3 and 4. Stages 0–2 are frozen to preserve low-level anatomical features learned from 41,400 UK Biobank subjects. LoRA scaling: $\alpha = 16$, effective scale $\alpha / r = 2.0$.

Targeted layers: Stages 3 and 4 only, specifically the attn.qkv projections                 
                
The function _find_lora_targets() (lora_adapter.py:66) walks the model and matches modules  
where the name contains both layers{stage} and attn.qkv. The concrete targets are:

```
swinViT.layers3.0.blocks.0.attn.qkv   # Stage 3, Block 0 — dim 192→576
swinViT.layers3.0.blocks.1.attn.qkv   # Stage 3, Block 1
swinViT.layers4.0.blocks.0.attn.qkv   # Stage 4, Block 0 — dim 384→1152
swinViT.layers4.0.blocks.1.attn.qkv   # Stage 4, Block 1
```

That's 4 QKV linear layers total, each getting a low-rank A/B pair.

```
What's frozen
Component: Stages 1–2 (layers1, layers2)
Trainable?: Frozen — preserves low-level anatomical features
────────────────────────────────────────
Component: Stages 3–4 QKV (layers3, layers4)
Trainable?: LoRA adapters only (base weights frozen)
────────────────────────────────────────
Component: Bottleneck (encoder10)
Trainable?: Frozen
────────────────────────────────────────
Component: Decoder (decoder1-5, out)
Trainable?: Trainable but not LoRA-adapted — uses pretrained original decoder weights
────────────────────────────────────────
Component: Semantic heads (aux)
Trainable?: Trainable (if enabled)
```

**DoRA** (Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation of Pre-Trained Models," ICML, 2024) decomposes weights into magnitude and direction components and applies LoRA only to direction. Both LoRA and DoRA are evaluated as ablation conditions (A2). The primary pipeline uses standard LoRA; DoRA comparison is deferred to the ablation study.

**Auxiliary Semantic Heads.** During Phase 1, optional auxiliary regression heads predict volume, location, and shape features from the encoder bottleneck. These provide multi-task learning signal that enriches the feature space for downstream SDP. The auxiliary loss is ramped in after a warmup period:

$$\mathcal{L}_{\text{Phase 1}} = \mathcal{L}_{\text{Dice}} + \lambda_{\text{CE}} \mathcal{L}_{\text{CE}} + \lambda_{\text{aux}} \cdot \rho(e) \cdot \sum_{p \in \mathcal{P}} \lambda_p^{\text{aux}} \mathcal{L}_p^{\text{sem}}$$

where $\rho(e)$ is a linear warmup from 0 to 1 over epochs $[e_{\text{start}}, e_{\text{start}} + e_{\text{dur}}]$.

**Segmentation as proxy task.** Segmentation is the optimal proxy for encoder adaptation because it implicitly encodes all three downstream semantic factors: volume (tumor vs. non-tumor voxel counts), location (spatial distribution of predicted labels), and shape (boundary geometry of the segmentation mask). BraTS-MEN provides segmentation labels for all 1,000 subjects, requiring no additional annotation.

### 2.2 Data

- **Training**: `lora_train` split (500 subjects from BraTS-MEN).
- **Validation**: `lora_val` split (100 subjects from BraTS-MEN).
- **Input**: $x \in \mathbb{R}^{B \times 4 \times 128^3}$, labels $y \in \{0,1,2,3\}^{128^3}$.
- **Augmentation**: `RandFlip(prob=0.5)`, `RandRotate90(prob=0.5)`, `RandScaleIntensity(factors=0.1)`, `RandShiftIntensity(offsets=0.1)`.

### 2.3 Outputs

- `phase1_encoder_merged.pt`: Merged encoder checkpoint ($W_{\text{merged}} = W_{\text{pretrained}} + \frac{\alpha}{r} BA$). All LoRA weights absorbed into base weights.
- `phase1_training_log.csv`: Per-epoch Dice, CE, semantic losses (if aux heads enabled).
- `phase1_best_dice.json`: Best validation Dice per sub-region.

### 2.4 Code Requirements

1. **`LoRASwinViT`** — Wrapper that injects LoRA adapters into specified SwinViT stages. Must support both LoRA and DoRA modes. Provides `merge_lora()` method to absorb adapters into base weights.
2. **`LoRALitModule`** (PyTorch Lightning) — Training module with:
   - Separate parameter groups: LoRA params (lr=1e-4), decoder params (lr=5e-4), aux heads (lr=1e-3).
   - DiceCELoss from MONAI with label conversion (integer labels → overlapping TC/WT/ET).
   - Optional auxiliary semantic heads with warmup scheduling.
   - Validation step computing per-region Dice scores.
3. **`swin_loader.py`** — Checkpoint loading with key mapping. Must handle the BSF-Tiny checkpoint structure: `swinViT.*`, `encoder1–10.*`, `decoder1–5.*`, `out.*`.
4. **`SemanticHeads`** — Auxiliary regression heads from GAP-pooled encoder10 features to volume/location/shape targets.

**Training Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 100 | Sufficient for LoRA convergence from strong init |
| LR (LoRA) | 1e-4 | Standard for LoRA |
| LR (decoder) | 5e-4 | Larger for randomly-initialized decoder head |
| Optimizer | AdamW | Decoupled weight decay |
| Weight decay | 0.01 | Standard regularization |
| Batch size | 2/GPU × $N_{\text{GPU}}$ | Memory-constrained by 128³ input |
| Scheduler | Cosine decay, 10-epoch warmup | Smooth convergence |
| Gradient clipping | `max_norm=1.0` | Stability |
| Precision | bf16-mixed | Memory efficiency |
| Decoder | Full SwinUNETR decoder (pretrained) | Stronger gradient signal to encoder |
| Aux semantic heads | Enabled ($\lambda_{\text{aux}} = 0.1$, warmup at epoch 5 over 10 epochs) | Multi-task enrichment |

**Post-Phase 1 Protocol:**

1. Merge LoRA weights: $W_{\text{merged}} = W_{\text{pretrained}} + \frac{\alpha}{r} BA$.
2. Discard segmentation decoder and auxiliary heads.
3. Freeze all encoder parameters.
4. Save merged checkpoint.

### 2.5 Verification Tests

```
TEST_2.1: LoRA injection
  - Load BSF checkpoint, inject LoRA with r=8 into stages 3-4
  - Assert trainable parameters ≈ 197K (LoRA only)
  - Assert stages 0-2 have zero gradients
  - Forward pass produces output shape [B, 3, 128, 128, 128]

TEST_2.2: LoRA initialization preserves output
  - Forward pass through base model: y_base = model(x)
  - Forward pass through LoRA model (B=0): y_lora = model_lora(x)
  - Assert ||y_base - y_lora|| < 1e-5 (LoRA is identity at init)

TEST_2.3: Training step
  - Run 1 training step, assert loss is finite and > 0
  - Assert LoRA parameters have nonzero gradients
  - Assert frozen parameters have zero gradients

TEST_2.4: LoRA merge
  - Merge LoRA weights, assert no LoRA params remain
  - Forward pass through merged model: y_merged = model_merged(x)
  - Forward pass through LoRA model: y_lora = model_lora(x)
  - Assert ||y_merged - y_lora|| < 1e-5

TEST_2.5: Segmentation quality
  - After training, validation Dice (WT) > 0.80
  - Dice improvement over frozen baseline > 0.05

TEST_2.6: Semantic head predictions (if aux enabled)
  - Aux vol R² on val set > 0.0 (above constant baseline)
  - Aux loc R² on val set > 0.0
```

### 2.6 Analysis

1. **Segmentation performance**: Report per-region Dice scores (TC, WT, ET) on `lora_val`. Compare with frozen BSF baseline.
2. **LoRA rank ablation** (already completed): Report Dice and linear probe $R^2$ across $r \in \{2, 4, 8, 16, 32\}$. Confirm $r = 8$ as optimal.
3. **LoRA vs DoRA ablation**: Compare Dice, probe $R^2$, and training stability for LoRA and DoRA at $r = 8$.
4. **Feature quality post-adaptation**: Repeat domain gap analysis (Section 1 metrics) with the LoRA-adapted encoder. Expect reduced MMD, improved linear probe $R^2$, and overlapping UMAP distributions between adapted GLI features and MEN features.
5. **Training curves**: Loss convergence, learning rate schedule, and Dice progression.
6. **Gradient analysis**: Monitor gradient norms at LoRA layers vs. decoder layers. Ensure LoRA gradients are not vanishing.

---

## Section 3: Phase 2 — Supervised Disentangled Projection

### 3.1 Background

After Phase 1, the encoder produces meningioma-adapted features $h \in \mathbb{R}^{768}$. Phase 2 trains a lightweight projection network that maps these features to a structured, disentangled latent space $z \in \mathbb{R}^{128}$ suitable for Neural ODE dynamics.

**Why not jointly train encoder + projection?** Gradient isolation. Joint training would cause semantic regression gradients to flow into the encoder via the projection, shaping encoder features toward the 13 semantic targets at the expense of general meningioma understanding. By decoupling the phases, the encoder learns the richest possible features via dense per-voxel segmentation loss, and the projection independently learns the best disentangled mapping from those features.

**Disentanglement without a VAE.** A common misconception is that disentanglement requires a generative model. This is false (Locatello et al., "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations," ICML, 2019). For supervised disentanglement — where the generative factors are known — discriminative approaches are strictly superior. Formally, disentanglement requires (Higgins et al., "Towards a Definition of Disentangled Representations," arXiv:1812.02230, 2018):

1. **Informativeness**: Each latent partition $z_p$ encodes information about its target factor $y_p$. Formally, $I(z_p; y_p)$ is maximized.
2. **Independence**: Latent partitions are statistically independent. Formally, $I(z_i; z_j) \approx 0$ for $i \neq j$.

The Supervised Disentangled Projection (SDP) achieves both through direct optimization via four complementary mechanisms.

**Theoretical context.** Under the framework of Locatello et al. ("Disentangling Factors of Variation Using Few Labels," ICLR, 2020), supervised disentanglement with $O(d \log d)$ labels achieves provably correct disentanglement for nonlinear ICA generative models. With $N = 800$ non-test BraTS-MEN subjects used for SDP training and $d = 128$ latent dimensions, the sample size approaches the $O(d \log d) \approx 896$ regime identified by Locatello et al. (2020). While the formal sample complexity bound is not strictly satisfied, we operate within the same order of magnitude, and the SDP draws principled inspiration from this framework. We note that the theorem assumes a nonlinear ICA generative model that may not hold exactly for our data.

**Spectral normalization** on **all** linear layers of the projection network controls the Lipschitz constant of each layer to $\sigma_{\max}(W) \leq 1$. For the 2-layer MLP $g_\phi = \text{SN}(\text{Linear}_2) \circ \text{GELU} \circ \text{SN}(\text{Linear}_1) \circ \text{LayerNorm}$, the global Lipschitz bound is:

$$L_g \leq L_{\text{LN}} \cdot 1 \cdot L_{\text{GELU}} \cdot 1$$

where $L_{\text{GELU}} \approx 1.13$. This provides a practical (though not tight) bound on latent space sensitivity to input perturbations, supporting ODE solver stability (Miyato et al., "Spectral Normalization for Generative Adversarial Networks," ICLR, 2018).

### 3.2 Architecture

```
Frozen SwinViT encoder10: [B, 768, 4, 4, 4]
       ↓
AdaptiveAvgPool3d(1): [B, 768]
       ↓
LayerNorm(768)
       ↓
SpectralNorm(Linear(768, 512)) → GELU → Dropout(0.1)
       ↓
SpectralNorm(Linear(512, 128))
       ↓
z ∈ ℝ^128 = [z_vol(24) | z_loc(8) | z_shape(12) | z_residual(84)]
       ↓                ↓               ↓
   π_vol(24→4)     π_loc(8→3)     π_shape(12→6)
       ↓                ↓               ↓
   ŷ_vol ∈ ℝ^4    ŷ_loc ∈ ℝ^3    ŷ_shape ∈ ℝ^6
```

Total trainable parameters: ~500K (projection MLP) + ~3K (semantic heads) ≈ 503K.

**Latent Space Partitioning:**

| Partition | Dims | Indices | Target Features | Purpose |
|-----------|------|---------|-----------------|---------|
| $z_{\text{vol}}$ | 24 | 0–23 | $\log(V_{\text{total}}+1), \log(V_{\text{NCR}}+1), \log(V_{\text{ED}}+1), \log(V_{\text{ET}}+1)$ | Volume encoding |
| $z_{\text{loc}}$ | 8 | 24–31 | $c_x, c_y, c_z$ (physical mm, normalized) | Centroid location |
| $z_{\text{shape}}$ | 12 | 32–43 | sphericity, surface area, solidity, 3 aspect ratios | Shape encoding |
| $z_{\text{residual}}$ | 84 | 44–127 | — (unsupervised) | Texture, context, scanner |

### 3.3 Loss Function

$$\boxed{\mathcal{L}_{\text{SDP}} = \underbrace{\sum_{p \in \mathcal{P}} \lambda_p \mathcal{L}_p^{\text{sem}}}_{\text{Informativeness}} + \underbrace{\lambda_{\text{cov}} \mathcal{L}_{\text{cov}}}_{\text{Linear independence}} + \underbrace{\lambda_{\text{var}} \mathcal{L}_{\text{var}}}_{\text{Collapse prevention}} + \underbrace{\lambda_{\text{dCor}} \sum_{\substack{(i,j) \in \mathcal{P}^2 \\ i < j}} \text{dCor}(z_i, z_j)}_{\text{Nonlinear independence}}}$$

**Term 1 — Semantic Regression (Informativeness):**

For each partition $p \in \mathcal{P} = \{\text{vol, loc, shape}\}$:

$$\mathcal{L}_p^{\text{sem}} = \frac{1}{k_p} \| \pi_p(z_p) - y_p \|_2^2$$

where $\pi_p$ is a lightweight linear projection head and $y_p$ is the ground truth semantic target (normalized to $\mu = 0, \sigma = 1$ using training set statistics).

**Term 2 — Cross-Partition Covariance Regularization (VICReg-adapted):**

Given batch covariance matrix $C \in \mathbb{R}^{128 \times 128}$ and cross-partition mask $M$ where $M_{ij} = 1$ iff $\text{part}(i) \neq \text{part}(j)$ (Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning," ICLR, 2022):

$$\mathcal{L}_{\text{cov}} = \frac{1}{|\{(i,j): M_{ij}=1\}|} \sum_{i,j} M_{ij} \cdot C_{ij}^2$$

Within-partition correlations are unconstrained.

**Term 3 — Variance Preservation (Collapse Prevention):**

$$\mathcal{L}_{\text{var}} = \frac{1}{d} \sum_{j=1}^{d} \max\left(0, \gamma - \sqrt{\text{Var}(z_j) + \epsilon}\right)$$

with $\gamma = 1.0$ and $\epsilon = 10^{-4}$.

**Term 4 — Distance Correlation (Nonlinear Independence):**

$$\text{dCor}(z_i, z_j) = \frac{\text{dCov}(z_i, z_j)}{\sqrt{\text{dVar}(z_i) \cdot \text{dVar}(z_j) + \epsilon}}$$

where $\text{dCor}(z_i, z_j) = 0 \iff z_i \perp\!\!\!\perp z_j$ — a strictly stronger condition than zero covariance (Székely et al., "Measuring and Testing Dependence by Correlation of Distances," Annals of Statistics, 2007).

### 3.4 Data

- **Training**: `train_pool` (800 subjects) — encoder is frozen, so no gradient contamination from LoRA training.
- **Evaluation**: `val` split (100 subjects).
- **Input**: Frozen encoder features $h \in \mathbb{R}^{768}$ (precomputed and cached from Phase 1 merged encoder).
- **Targets**: Semantic features computed from segmentation masks, normalized to $\mu = 0, \sigma = 1$.

> **Normalization scope:** Semantic feature normalization parameters ($\mu_p, \sigma_p$) for each target are computed on the SDP training set only (800 non-test subjects). These parameters are applied without recomputation to the validation set (100), test set (100), and the Andalusian cohort. This prevents information leakage from evaluation subjects into the normalization statistics.

### 3.5 Outputs

- `phase2_sdp.pt`: Trained SDP network + semantic heads.
- `phase2_training_log.csv`: Per-epoch losses (each term individually).
- `phase2_quality_report.json`: $R^2$ per semantic target, dCor between partitions, per-dimension variance.
- `latent_umap.png`: UMAP of 128-d latent space colored by volume, location, shape.

### 3.6 Code Requirements

1. **`SDP`** — 2-layer MLP with LayerNorm, GELU, Dropout, spectral normalization on all linear layers.
2. **`LatentPartition`** — Splits $z$ into named partitions with configurable dimensions.
3. **`SemanticHeads`** — Per-partition linear projection heads.
4. **`SDPLoss`** — Composite loss with configurable weights. Each term computed independently and logged.
5. **`dCorLoss`** — Differentiable distance correlation between partition tensors.
6. **`VICRegLoss`** — Cross-partition covariance + variance hinge loss.
7. **`SDPLitModule`** (PyTorch Lightning) — Training module. Encoder is frozen; only SDP + heads are trained.

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $\lambda_{\text{vol}}$ | 20.0 | Highest priority (Gompertz proxy) |
| $\lambda_{\text{loc}}$ | 12.0 | Moderate priority |
| $\lambda_{\text{shape}}$ | 15.0 | High priority |
| $\lambda_{\text{cov}}$ | 5.0 | VICReg default scale |
| $\lambda_{\text{var}}$ | 5.0 | VICReg default scale |
| $\lambda_{\text{dCor}}$ | 2.0 | Moderate nonlinear penalty |
| Learning rate | 1e-3 | Small network, fast convergence |
| Optimizer | AdamW | Standard |
| Weight decay | 0.01 | Mild regularization |
| Epochs | 100 | Single-phase, converges fast |
| Batch size | Full-batch (800) | Precomputed features; eliminates covariance estimation noise |
| Scheduler | Cosine decay, 5-epoch warmup | Standard |

> Since SDP training operates on precomputed feature vectors $h \in \mathbb{R}^{768}$ (not 3D volumes), full-batch optimization over all 800 training subjects is computationally trivial and provides exact covariance/dCor estimates. Mini-batch training is unnecessary.

**Curriculum** (4-phase):

| Phase | Epochs | Active Losses |
|-------|--------|---------------|
| Warm-up | 0–9 | $\mathcal{L}_{\text{var}}$ only |
| Semantic | 10–39 | + $\mathcal{L}_{\text{vol}}, \mathcal{L}_{\text{loc}}, \mathcal{L}_{\text{shape}}$ |
| Independence | 40–59 | + $\mathcal{L}_{\text{cov}}, \mathcal{L}_{\text{dCor}}$ |
| Full | 60–100 | All losses at full strength |

### 3.7 Verification Tests

```
TEST_3.1: SDP forward pass
  - Input: h ∈ ℝ^{64×768} (batch of 64 features)
  - Assert output z shape == [64, 128]
  - Assert z partitions sum to 128 dimensions
  - Assert semantic head outputs: vol [64,4], loc [64,3], shape [64,6]

TEST_3.2: Spectral normalization
  - Assert spectral norm of final layer weight ≈ 1.0
  - After 10 training steps, assert spectral norm still ≈ 1.0

TEST_3.3: Loss computation
  - Compute each loss term individually, assert all finite and ≥ 0
  - Assert dCor ∈ [0, 1] for each partition pair
  - Assert L_var = 0 when all dimensions have std > γ

TEST_3.4: Training convergence
  - After 100 epochs, assert total loss decreased by ≥ 50%
  - Assert no NaN losses during training

TEST_3.5: Semantic quality
  - On val set: Vol R² ≥ 0.80 (target: 0.90)
  - On val set: Loc R² ≥ 0.85 (target: 0.95)
  - On val set: Shape R² ≥ 0.30 (target: 0.40)

TEST_3.6: Disentanglement quality
  - Max cross-partition Pearson correlation < 0.30 (target: 0.20)
  - dCor(vol, loc) < 0.20 (target: 0.10)
  - Per-dimension variance > 0.3 for ≥ 90% of dimensions

TEST_3.7: Latent space smoothness (Lipschitz check)
  - For 100 pairs of similar inputs (||h1 - h2|| < ε):
    Assert ||z1 - z2|| / ||h1 - h2|| < L for some bounded L
```

### 3.8 Analysis

1. **Semantic regression quality**: $R^2$ per feature on `val` set. Compare with: (a) frozen encoder + linear probe (Section 1 baseline), (b) LoRA-adapted encoder + linear probe (Section 2).
2. **Disentanglement metrics**:
   - Cross-partition Pearson correlation matrix (4×4 block structure).
   - dCor matrix between all partition pairs.
   - DCI disentanglement score (Eastwood & Williams, "A Framework for the Quantitative Evaluation of Disentangled Representations," ICLR, 2018) — optional but recommended.
3. **Dimensional analysis**: Per-dimension variance histogram. Identify any collapsed dimensions (std < 0.1).
4. **UMAP visualizations**: 128-d latent space colored by (a) log-volume, (b) centroid z-coordinate, (c) sphericity. Smooth color gradients indicate successful semantic encoding.
5. **Ablation: latent dimension $d$**: If time permits, compare $d \in \{64, 128, 256\}$ on $R^2$ and dCor metrics.
6. **Ablation: with vs. without VICReg/dCor losses**: Measure entanglement when regularization is removed.

---

## Section 4: Phase 3 — Cohort Encoding and Harmonization

### 4.1 Background

After Phases 1–2, the encoder + SDP form a frozen deterministic mapping:

$$z_{i,t} = g_\phi(\text{GAP}(\text{SwinViT}(x_{i,t}))) \in \mathbb{R}^{128}$$

This section encodes all Andalusian cohort MRI volumes into the 128-d latent space and assesses whether scanner/site harmonization is necessary.

**ComBat Harmonization.** The Andalusian cohort involves approximately 10 distinct MRI scanners. Scanner-induced batch effects can introduce systematic shifts in the latent space that confound temporal dynamics. ComBat (Johnson et al., "Adjusting batch effects in microarray expression data using empirical Bayes methods," Biostatistics, 2007) corrects for additive and multiplicative site effects:

$$z^*_{i,t,j} = \frac{z_{i,t,j} - \hat{\alpha}_j - X_i \hat{\beta}_j - \hat{\gamma}_{s(i),j}}{\hat{\delta}_{s(i),j}} + \hat{\alpha}_j + X_i \hat{\beta}_j$$

where $j$ indexes latent dimensions, $s(i)$ is the scanner/site of patient $i$, $\hat{\gamma}$ and $\hat{\delta}$ are additive and multiplicative site effects estimated via empirical Bayes.

**Temporal preservation.** Site correction parameters $(\hat{\gamma}_{s}, \hat{\delta}_{s})$ are constant across all timepoints for a given patient (same scanner across visits). Therefore, intra-patient temporal dynamics are preserved up to a constant rescaling:

$$\Delta z^* = z^*_{i,t_2} - z^*_{i,t_1} = \frac{z_{i,t_2} - z_{i,t_1}}{\hat{\delta}_{s(i)}} = \frac{\Delta z}{\hat{\delta}_{s(i)}}$$

**Assumption: same scanner across visits.** The temporal preservation guarantee above assumes each patient is scanned on the same scanner at all timepoints. This must be verified from the scanner metadata. If scanner changes exist between visits for the same patient, LongComBat (Beer et al., NeuroImage, 2020) should be used instead of standard ComBat.

**ComBat necessity assessment.** The foundation model was pretrained on 41,400 UK Biobank subjects from multiple scanners, conferring some degree of scanner invariance. We assess whether additional harmonization is needed by comparing BraTS-MEN and Andalusian cohort latent distributions before applying ComBat.

### 4.2 Data

- **BraTS-MEN** (full 1000 subjects): Encoded through frozen encoder + SDP → reference distribution.
- **Andalusian cohort** (42 patients, 137 studies): Encoded through frozen encoder + SDP → target distribution.
- **Scanner metadata**: Scanner ID per study in the Andalusian cohort.

### 4.3 Outputs

- `latent_bratsmen.pt`: $z \in \mathbb{R}^{1000 \times 128}$ — BraTS-MEN latent vectors.
- `latent_andalusian.pt`: $z \in \mathbb{R}^{137 \times 128}$ — Andalusian cohort latent vectors with patient/timepoint metadata.
- `latent_andalusian_harmonized.pt`: $z^* \in \mathbb{R}^{137 \times 128}$ — After ComBat (if applied).
- `combat_assessment.json`: MMD before/after ComBat, UMAP plots, decision rationale.
- `trajectories.json`: Per-patient latent trajectories $\{(z^*_{i,t_k}, t_k)\}_{k=1}^{n_i}$.

### 4.4 Code Requirements

1. **`CohortEncoder`** — Batch encoding pipeline: loads volumes, applies frozen encoder + SDP, saves latent vectors with metadata.
2. **`SlidingWindowEncoder`** — For volumes larger than 128³: overlapping patches with tumor-weighted averaging.

   $$h = \frac{\sum_{p} w_p \cdot \text{GAP}(\text{SwinViT}(x_p))}{\sum_{p} w_p}, \quad w_p = \frac{|\text{mask}_p \cap \text{tumor}|}{|\text{patch}_p|}$$

3. **`LatentComBat`** — Wrapper around `neuroCombat` or custom implementation. Inputs: latent matrix, site labels, optional biological covariates.
4. **`HarmonizationAssessor`** — Computes MMD, UMAP, and per-dimension KS tests between BraTS-MEN and Andalusian distributions. Outputs a decision: harmonize or skip.
5. **`TrajectoryBuilder`** — Constructs per-patient temporal trajectories from encoded timepoints.

### 4.5 Verification Tests

```
TEST_4.1: Encoding determinism
  - Encode same volume twice through frozen pipeline
  - Assert ||z1 - z2|| < 1e-6 (deterministic, no dropout)

TEST_4.2: Encoding shape
  - Encode 5 Andalusian volumes
  - Assert output shape [5, 128]
  - Assert no NaN values

TEST_4.3: ComBat temporal preservation
  - For a patient with 3 timepoints:
    Δz = z(t2) - z(t1), Δz* = z*(t2) - z*(t1)
  - Assert Δz* = Δz / δ_site (up to numerical precision)
  - Assert ratio is constant across all dimension pairs

TEST_4.4: Trajectory construction
  - For each patient, assert timepoints are temporally ordered
  - Assert Δt > 0 for all consecutive pairs
  - Assert trajectory length matches patient metadata

TEST_4.5: Distribution assessment
  - Compute MMD between BraTS-MEN and Andalusian latents
  - Assert computation completes without error
  - Report result (no pass/fail — this is diagnostic)

TEST_4.6: Scanner consistency verification
  - For each patient, check scanner IDs across all timepoints
  - Report fraction of patients with consistent scanner
  - If any patient has scanner changes, flag for LongComBat
  - DIAGNOSTIC (does not block pipeline)
```

### 4.6 Analysis

1. **Pre-harmonization assessment**:
   - UMAP of BraTS-MEN (blue) vs Andalusian (red) latent vectors. If substantial overlap exists, ComBat may be unnecessary.
   - MMD with permutation test. If $p > 0.05$, distributions are not significantly different.
   - Per-dimension Kolmogorov-Smirnov tests. Report fraction of dimensions with significant shift ($p < 0.05$ after Bonferroni correction).

2. **Post-harmonization assessment** (if ComBat applied):
   - Repeat UMAP and MMD. Confirm reduced distributional shift.
   - Verify that intra-patient temporal dynamics are preserved (correlation between $\Delta z$ and $\Delta z^*$ trajectories).

3. **Scanner effect analysis**:
   - UMAP colored by scanner ID. If scanner clusters are visible in latent space, ComBat is justified.
   - Variance explained by scanner vs. patient identity (ANOVA on latent dimensions).

4. **Trajectory visualization**:
   - 2D UMAP trajectories for 5–10 patients with $\geq 3$ timepoints. Arrows indicating temporal direction. Smooth, directional trajectories suggest the latent space captures growth dynamics.
   - Volume partition ($z_{\text{vol}}$) trajectory vs. ground truth volume change. Correlation should be positive and significant.

---

## Section 5: Phase 4 — Neural ODE Growth Forecasting

### 5.1 Background

The Neural ODE (Chen et al., "Neural Ordinary Differential Equations," NeurIPS, 2018) models continuous-time evolution in the disentangled latent space:

$$\frac{dz(t)}{dt} = f_\theta(z(t), t)$$

Given an initial state $z(t_0)$ and a time horizon $\Delta t$, the ODE solver predicts the future state:

$$\hat{z}(t_1) = z(t_0) + \int_{t_0}^{t_1} f_\theta(z(\tau)) \, d\tau = \text{ODESolve}(f_\theta, z(t_0), t_0, t_1)$$

**Physics-informed architecture.** The latent space partitioning enables a Gompertz-informed Neural ODE where the volume partition follows tumor growth kinetics (Benzekry et al., "Classical Mathematical Models for Description and Prediction of Experimental Tumor Growth," PLOS Computational Biology, 2014):

**Volume dynamics (decode-then-model):** The Gompertz growth model is applied to decoded physical volumes, not raw latent dimensions. The semantic head $\pi_{\text{vol}}$ decodes $z_{\text{vol}} \to \hat{V} \in \mathbb{R}^4$ (the 4 log-volume predictions). Gompertz dynamics operate on these physical quantities:

$$\frac{d\hat{V}}{dt} = \underbrace{\alpha \cdot \hat{V} \odot \ln\left(\frac{K}{\hat{V} + \epsilon}\right)}_{\text{Gompertz growth}} + \underbrace{\eta_{\text{vol}} \cdot h_\theta(\hat{V}, z_{\text{other}})}_{\text{Neural correction}}$$

where $\hat{V} = \pi_{\text{vol}}(z_{\text{vol}}) \in \mathbb{R}^4$ (total, NCR, ED, ET log-volumes), $\alpha \in \mathbb{R}^+$ (softplus) is the growth rate, $K \in \mathbb{R}^4_+$ (softplus) is the carrying capacity vector, and $\eta_{\text{vol}} = 0.01$ scales the neural residual. The Neural ODE integrates in decoded volume space, then the volume partition $z_{\text{vol}}$ is updated via the pseudo-inverse of $\pi_{\text{vol}}$ or a learned re-encoder. This reduces Gompertz-governed dimensions from 24 to 4 and maintains biological interpretability.

**Location partition** (dims 24–31):

$$\frac{dz_{\text{loc}}}{dt} = \eta_{\text{loc}} \cdot \text{MLP}_{\text{loc}}(z_{\text{vol}}, z_{\text{loc}})$$

Location changes are slow and modulated by volume (mass effect). $\eta_{\text{loc}} = 0.01$.

**Shape partition** (dims 32–43):

$$\frac{dz_{\text{shape}}}{dt} = \eta_{\text{shape}} \cdot \text{MLP}_{\text{shape}}(z_{\text{vol}}, z_{\text{shape}})$$

Shape changes driven by volume growth. $\eta_{\text{shape}} = 0.01$.

**Residual partition** (dims 44–127):

$$\frac{dz_{\text{res}}}{dt} = 0 \quad \text{(frozen)}$$

The residual partition is carried forward unchanged from the initial state: $z_{\text{res}}(t_1) = z_{\text{res}}(t_0)$. This reduces the effective ODE dimension from 128 to 44 ($z_{\text{vol}}$: 24 + $z_{\text{loc}}$: 8 + $z_{\text{shape}}$: 12), dramatically reducing overfitting risk given the modest training set (~155 forward pairs from 42 patients). The residual state preserves texture, context, and scanner information without requiring the ODE to model their dynamics.

**Training objective:**

$$\mathcal{L}_{\text{ODE}} = \underbrace{\sum_{(i,t_0,t_1)} \|z^*_{i,t_1} - \hat{z}_{i,t_1}\|_2^2}_{\text{Trajectory MSE}} + \underbrace{\lambda_{\text{reg}} \|\theta_{\text{ODE}}\|_2^2}_{\text{Weight decay}} + \underbrace{\lambda_{\text{smooth}} \int_{t_0}^{t_1} \left\|\frac{d^2z}{dt^2}\right\|^2 dt}_{\text{Jerk regularization}}$$

where $\hat{z}_{i,t_1} = \text{ODESolve}(f_\theta, z^*_{i,t_0}, t_0, t_1)$.

**Data augmentation via temporal pairing.** For a patient with $n$ timepoints, all $\binom{n}{2} = n(n-1)/2$ **forward** ordered pairs $(t_i, t_j)$ where $t_i < t_j$ are used. Reverse pairs are excluded because tumor growth is biologically irreversible — the Gompertz prior enforces $dV/dt > 0$, and reverse pairs would force the neural correction to overpower the physics prior. With 42 patients averaging 3.26 timepoints: approximately $42 \times \binom{3.26}{2} \approx 155$ forward training pairs. To compensate for the reduced data, small Gaussian perturbations ($\sigma = 0.01$) are applied to $z(t_0)$ as data augmentation.

### 5.2 Data

- **Training pairs**: All $\binom{n}{2}$ forward temporal pairs from 42 patients (≈155 pairs). Leave-one-patient-out cross-validation.
- **Input**: $(z^*_{i,t_0}, z^*_{i,t_1}, \Delta t = t_1 - t_0)$ from Section 4.
- **Time normalization**: $\Delta t$ in years (or months — specify consistently).

### 5.3 Outputs

- `ode_model.pt`: Trained Neural ODE parameters.
- `ode_trajectories.pt`: Predicted vs. actual trajectories for all patients.
- `gompertz_params.json`: Per-patient $(\hat{\alpha}_i, \hat{K}_i)$ extracted from the trained model.
- `risk_stratification.json`: Per-patient risk scores.

### 5.4 Code Requirements

1. **`GompertzDynamics`** — Decoded volume ODE function operating on $\hat{V} \in \mathbb{R}^4$ with learnable $\alpha$, $K$, and neural correction.
2. **`PartitionODE`** — Full partition-aware ODE function composing Gompertz (4 decoded volume dims), MLP (location 8, shape 12), frozen residual. Effective ODE dimension: 44.
3. **`ODEFunc`** — `torchdiffeq`-compatible wrapper for adjoint method.
4. **`ODELitModule`** (PyTorch Lightning) — Training with `torchdiffeq.odeint_adjoint`. Leave-one-patient-out cross-validation.
5. **`TrajectoryDataset`** — Generates all temporal pairs from patient trajectories.
6. **`RiskStratifier`** — Extracts Gompertz parameters, computes risk scores.

### 5.5 Verification Tests

```
TEST_5.1: ODE forward pass
  - Given z0 ∈ ℝ^128 and Δt = 1.0 (year)
  - z1 = ODESolve(f_θ, z0, 0, 1.0)
  - Assert z1.shape == [128]
  - Assert ||z1 - z0|| > 0 (dynamics are non-trivial)
  - Assert no NaN values

TEST_5.2: Gompertz stability
  - Initialize z_vol > 0 (all positive)
  - Integrate for 10 years
  - Assert z_vol remains bounded (carrying capacity constraint)
  - Assert z_vol > 0 (no negative volumes)

TEST_5.3: Reversibility
  - z1 = ODESolve(f_θ, z0, 0, Δt)
  - z0_hat = ODESolve(f_θ, z1, Δt, 0)
  - Assert ||z0 - z0_hat|| < 1e-3 (ODE reversibility)

TEST_5.4: Training step
  - Run 1 training step on a batch of pairs
  - Assert loss is finite and > 0
  - Assert gradients are nonzero for ODE parameters

TEST_5.5: Partition-aware dynamics
  - Assert ||dz_vol/dt|| >> ||dz_res/dt|| (volume changes faster than residual)
  - This follows from η_vol >> η_res
```

### 5.6 Analysis

1. **Trajectory prediction quality**:
   - Per-patient trajectory MSE in latent space (leave-one-patient-out).
   - Volume partition prediction $R^2$: decode $z_{\text{vol}}$ → actual volume via semantic head, compare predicted vs. actual volume change.
   - Location/shape partition prediction quality (analogous).

2. **Gompertz parameter analysis**:
   - Distribution of $\hat{\alpha}_i$ (growth rates) and $\hat{K}_i$ (carrying capacities).
   - Correlation of $\hat{\alpha}_i$ with clinical variables (if available): WHO grade, Ki-67 index, etc.
   - Biological plausibility: $\hat{\alpha}_i > 0$ (growth, not shrinkage), $\hat{K}_i$ in clinically reasonable range.

3. **Risk stratification**:
   - Standardized risk score: $\text{Risk}_i = (\hat{\alpha}_i - \bar{\alpha}) / \sigma_\alpha$.
   - Kaplan-Meier-style visualization of high vs. low risk groups (if clinical outcome data available).

4. **Ablation: Gompertz vs. pure neural ODE**: Compare trajectory MSE and volume $R^2$ between:
   - Gompertz-informed (physics prior + neural correction).
   - Pure MLP ODE (no physics prior).
   - Gompertz only (no neural correction).

5. **Trajectory visualization**: 2D PCA/UMAP projection of predicted trajectories overlaid with actual trajectories. Smooth, parallel-transport-like evolution indicates good dynamics modeling.

---

## Section 6: End-to-End Evaluation Framework

### 6.1 Background

This section defines the evaluation protocol that validates the entire pipeline from raw MRI to growth prediction. It consolidates per-section metrics into a unified framework.

### 6.2 Quality Targets

**Phase 1 (Encoder Adaptation):**

| Metric | Target | Minimum |
|--------|--------|---------|
| Dice (WT) on `lora_val` | ≥ 0.85 | ≥ 0.80 |
| Dice improvement over frozen BSF | ≥ 0.05 | ≥ 0.02 |
| Linear probe Vol $R^2$ (adapted) | ≥ 0.50 | ≥ 0.30 |

**Phase 2 (SDP):**

| Metric | Target | Minimum |
|--------|--------|---------|
| Vol $R^2$ on `val` | ≥ 0.90 | ≥ 0.80 |
| Loc $R^2$ on `val` | ≥ 0.95 | ≥ 0.85 |
| Shape $R^2$ on `val` | ≥ 0.40 | ≥ 0.25 |
| Max cross-partition correlation | < 0.20 | < 0.30 |
| Per-dimension variance > 0.5 | ≥ 95% | ≥ 85% |
| dCor(vol, loc) | < 0.10 | < 0.20 |

**Phase 4 (Neural ODE):**

| Metric | Target | Minimum |
|--------|--------|---------|
| Volume prediction $R^2$ (LOPO-CV) | ≥ 0.70 | ≥ 0.50 |
| Trajectory MSE | Monotonically decreasing during training | — |
| Gompertz $\hat{\alpha}_i > 0$ for all patients | 100% | ≥ 90% |

### 6.3 Ablation Study Matrix

| Experiment | Variable | Conditions | Primary Metric |
|------------|----------|------------|----------------|
| A1: LoRA rank | $r$ | {2, 4, 8, 16, 32} | Dice, probe $R^2$ |
| A2: LoRA vs DoRA | Adapter type | {LoRA, DoRA} at $r=8$ | Dice, probe $R^2$ |
| A3: Aux semantic heads | Phase 1 aux | {with, without} | Phase 2 $R^2$ |
| A4: SDP dimension | $d$ | {64, 128, 256} | $R^2$, dCor |
| A5: VICReg + dCor | Regularization | {full, no cov, no dCor, no both} | Cross-partition corr |
| A6: Gompertz prior | ODE architecture | {Gompertz+MLP, MLP only, Gompertz only} | Trajectory MSE, Vol $R^2$ |
| A7: ComBat | Harmonization | {with, without} | Phase 4 trajectory MSE |
| A8: Residual dynamics | ODE residual | {frozen (default), learned with $\eta$=0.001} | Trajectory MSE, overfitting |

### 6.4 Figure List

1. **Pipeline overview diagram** (already in methodology).
2. **Domain gap UMAP**: GLI vs MEN features, frozen encoder.
3. **LoRA ablation**: Dice and probe $R^2$ vs. rank.
4. **Phase 2 training curves**: Individual loss terms over epochs.
5. **Disentanglement matrix**: Cross-partition correlation heatmap.
6. **Latent UMAP colored by semantics**: Volume, location, shape.
7. **Cohort distribution comparison**: BraTS-MEN vs Andalusian UMAP.
8. **Patient trajectories**: 2D latent space with temporal arrows.
9. **Volume prediction**: Predicted vs actual volume change scatter plot.
10. **Gompertz parameter distribution**: Histogram of growth rates.
11. **Risk stratification**: Ranked patients by growth rate.

---

## Appendix A: Common Configuration Contract

All phases share a common YAML configuration schema. Phase-specific overrides extend the base config.

```yaml
# foundation.yaml (base)
paths:
  bratsmen_root: /path/to/BraTS_Men_Train
  andalusian_root: /path/to/andalusian_cohort
  bsf_checkpoint: /path/to/finetuned_model_fold_0.pt
  output_root: /path/to/outputs

encoder:
  feature_size: 48
  in_channels: 4
  out_channels: 3
  depths: [2, 2, 2, 2]
  num_heads: [3, 6, 12, 24]
  roi_size: [128, 128, 128]

data:
  channel_order: [t2f, t1c, t1n, t2w]  # [FLAIR, T1ce, T1, T2] — matches BrainSegFounder
  label_mapping: {0: background, 1: NCR, 2: ED, 3: ET}
  seed: 42

latent:
  total_dim: 128
  vol_dim: 24
  loc_dim: 8
  shape_dim: 12
  residual_dim: 84

logging:
  framework: wandb  # or tensorboard
  project: mengrowth-model
```

**Libraries:**

| Library | Role |
|---------|------|
| PyTorch 2.0+ | Models, losses, AMP |
| MONAI 1.3+ | NIfTI loading, transforms, SwinUNETR, DiceCELoss |
| PyTorch Lightning 2.0+ | Training loop, checkpointing, logging |
| OmegaConf 2.3+ | Hierarchical configuration |
| torchdiffeq | Neural ODE solvers (adjoint method) |
| scipy | Convex hull, center of mass, shape features |
| neuroCombat | Scanner harmonization |
| scikit-learn | Ridge regression probes, metrics |
| umap-learn | Visualization |

**Reproducibility:** `seed_everything(42, workers=True)`. Save resolved config in every run directory.

---

## Appendix B: References

### Foundation Model & Architecture

1. Cox, J. et al. "BrainSegFounder: Towards Foundation Models for Neuroimage Segmentation." *Medical Image Analysis*, 2024. [arXiv:2406.10395v3]
2. Hatamizadeh, A. et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." *BrainLes, MICCAI*, 2022.

### Parameter-Efficient Adaptation

3. Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.
4. Liu, S.-Y. et al. "DoRA: Weight-Decomposed Low-Rank Adaptation of Pre-Trained Models." *ICML*, 2024.
5. Dutt, R. et al. "Parameter-Efficient Fine-Tuning for Medical Image Analysis." *TMLR*, 2024.

### Disentanglement Theory

6. Locatello, F. et al. "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations." *ICML*, 2019.
7. Locatello, F. et al. "Disentangling Factors of Variation Using Few Labels." *ICLR*, 2020.
8. Higgins, I. et al. "Towards a Definition of Disentangled Representations." *arXiv:1812.02230*, 2018.
9. Eastwood, C. & Williams, C. K. I. "A Framework for the Quantitative Evaluation of Disentangled Representations." *ICLR*, 2018.

### Regularization & Independence

10. Bardes, A. et al. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *ICLR*, 2022.
11. Székely, G. J. et al. "Measuring and Testing Dependence by Correlation of Distances." *Annals of Statistics*, 2007.
12. Miyato, T. et al. "Spectral Normalization for Generative Adversarial Networks." *ICLR*, 2018.

### Neural ODEs & Tumor Growth

13. Chen, R. T. Q. et al. "Neural Ordinary Differential Equations." *NeurIPS*, 2018.
14. Rubanova, Y. et al. "Latent ODEs for Irregularly-Sampled Time Series." *NeurIPS*, 2019.
15. Benzekry, S. et al. "Classical Mathematical Models for Description and Prediction of Experimental Tumor Growth." *PLOS Computational Biology*, 2014.

### Data & Harmonization

16. LaBella, D. et al. "The ASNR-MICCAI BraTS Meningioma Challenge." *arXiv*, 2024.
17. Johnson, W. E. et al. "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics*, 2007.

### Domain Gap & Representation Analysis

18. Gretton, A. et al. "A Kernel Two-Sample Test." *JMLR*, 2012.
19. Kornblith, S. et al. "Similarity of Neural Network Representations Revisited." *ICML*, 2019.

### Other

20. Pope, P. et al. "The Intrinsic Dimension of Images and Its Relevance to Learning." *ICLR*, 2021.
21. Hoffman, M. D. & Johnson, M. J. "ELBO Surgery: Yet Another Way to Carve Up the Variational Evidence Lower Bound." *NeurIPS Workshop*, 2016.
