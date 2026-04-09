# SPEC: Hierarchical Feature Analysis of BrainSegFounder Encoder

**Date:** 2026-04-04  
**Target Agent:** Claude Code Opus 4.6  
**Location:** `experiments/uncertainty_segmentation/feature_analysis/`  
**Deliverables:** One figure, one table, one self-contained Python module

---

## 0. Scientific objective

Provide empirical evidence that LoRA adaptation of SwinViT Stages 3–4 targets the tumor-specific feature subspace while preserving the general anatomical representations in Stages 0–2. The experiment produces exactly **one figure** and **one table** for inclusion in the thesis.

The claim has two parts:
1. **The frozen encoder already segregates anatomy vs. pathology across stages.** Stages 0–2 encode spatially diffuse, anatomically general features. Stages 3–4 encode features preferentially activated by tumor tissue.
2. **LoRA adaptation amplifies this segregation.** After training with the unfrozen decoder providing supervisory gradients, the LoRA-adapted Stages 3–4 become more tumor-selective than their frozen counterparts, while Stages 0–2 remain unchanged (they receive no LoRA parameters and no gradient updates).

---

## 1. Background and references

- **Network Dissection** (Bau et al., CVPR 2017): quantifies unit interpretability by computing IoU between thresholded activation maps and concept segmentation masks. Our TSI is the continuous, medical-domain analogue.
- **Grad-CAM** (Selvaraju et al., ICCV 2017): gradient-based class-activation mapping. We deliberately avoid Grad-CAM because (a) it requires a scalar output (classification), not a spatial segmentation, (b) it conflates feature content with gradient magnitude, and (c) it depends on the decoder's gradient flow. Our TSI is gradient-free and measures what the encoder *represents*, not what the loss *optimises*.
- **TransXAI** (Scientific Reports, 2024): applied Grad-CAM to hybrid ViT/CNN segmentation networks, showing that internal layers follow a "human-like hierarchical approach for localising brain tumor parts." Our experiment formalises and quantifies this observation for SwinUNETR.
- **Hierarchical feature learning in vision transformers** (Liu et al., ICCV 2021, Swin Transformer): established that Swin stages progressively increase receptive field and semantic abstraction level. Our analysis verifies this for the specific case of BrainSegFounder on meningioma MRI.

---

## 2. Method: Tumor Selectivity Index (TSI)

### 2.1 Definition

For a given input volume $\mathbf{x} \in \mathbb{R}^{4 \times D \times H \times W}$ with ground-truth whole-tumor binary mask $\mathbf{m} \in \{0,1\}^{D \times H \times W}$, let the SwinViT encoder produce hidden states at each stage:

$$\mathbf{h}^{(s)} = \text{SwinViT}_s(\mathbf{x}) \in \mathbb{R}^{C_s \times D_s \times H_s \times W_s}, \quad s \in \{0, 1, 2, 3, 4\}$$

Downsample the GT mask to stage $s$'s spatial resolution via nearest-neighbour interpolation:

$$\mathbf{m}^{(s)} = \text{NNInterp}(\mathbf{m},\; [D_s, H_s, W_s])$$

For channel $c$ at stage $s$, compute the mean absolute activation inside and outside the tumor:

$$\mu^{\text{tumor}}_{s,c} = \frac{\sum_v |h^{(s)}_{c,v}| \cdot m^{(s)}_v}{\sum_v m^{(s)}_v + \epsilon}, \quad \mu^{\text{non-tumor}}_{s,c} = \frac{\sum_v |h^{(s)}_{c,v}| \cdot (1 - m^{(s)}_v)}{\sum_v (1 - m^{(s)}_v) + \epsilon}$$

The Tumor Selectivity Index is:

$$\text{TSI}_{s,c} = \frac{\mu^{\text{tumor}}_{s,c}}{\mu^{\text{non-tumor}}_{s,c} + \epsilon}, \quad \epsilon = 10^{-8}$$

- TSI $\gg 1$: channel is tumor-selective (activates preferentially inside tumor).
- TSI $\approx 1$: channel is anatomically general (activates uniformly).
- TSI $< 1$: channel is tumor-suppressed (encodes "absence of normal tissue").

### 2.2 Per-stage summary statistics

For each stage $s$, report:
- $\overline{\text{TSI}}_s = \text{mean}_c(\text{TSI}_{s,c})$ and $\text{SD}_s$
- $\text{Frac}_s(\tau) = \frac{|\{c : \text{TSI}_{s,c} > \tau\}|}{C_s}$ for $\tau \in \{1.5, 2.0\}$
- One-sample Wilcoxon signed-rank test: $H_0: \text{median}(\text{TSI}_{s,\cdot}) = 1$

### 2.3 The two-condition comparison

| Condition | Description | Stages with LoRA | Decoder |
|-----------|-------------|-----------------|---------|
| **Frozen** | Original BrainSegFounder checkpoint, no adaptation | None | Frozen (pretrained) |
| **LoRA-adapted** | One trained ensemble member (e.g., member 0 from the r8_M10_s42 run) | 3, 4 | Trained (loaded from `decoder.pt`) |

For the **frozen** condition, use `load_full_swinunetr(checkpoint, freeze_encoder=True, freeze_decoder=True)`.

For the **LoRA-adapted** condition, load the full model including LoRA adapter and trained decoder: this is the same loading path as `EnsemblePredictor._load_member_model()`.

The comparison isolates the effect of LoRA+decoder training:
- Stages 0–2 should have **identical TSI** between conditions (no LoRA was applied, and no gradient flows into frozen stages).
- Stages 3–4 should have **higher TSI** in the adapted condition if LoRA + decoder training improved tumor feature selectivity.

### 2.4 Multi-scan robustness

Compute TSI on $N \geq 20$ BraTS-MEN test scans (where GT masks are available). Report mean ± std across scans for every metric. This ensures the results are not driven by a single easy/hard case.

---

## 3. SwinViT architecture details (for the agent)

The hidden states are obtained via `model.swinViT(x, model.normalize)` which returns a list of 5 tensors. The agent should inspect the actual shapes at runtime, but the expected architecture is:

| Stage | Output key | Channels ($C_s$) | Spatial resolution | Feature dim |
|-------|-----------|------|-----|-----|
| 0 | `hidden_states[0]` | 48 | $48^3$ | Low-level (edges, intensity gradients) |
| 1 | `hidden_states[1]` | 96 | $24^3$ | Tissue boundaries, contrast patterns |
| 2 | `hidden_states[2]` | 192 | $12^3$ | Regional anatomy, ventricle/sulcus geometry |
| 3 | `hidden_states[3]` | 384 | $6^3$ | Pathology-related semantic features |
| 4 | `hidden_states[4]` | 768 | $3^3$ | Bottleneck: maximally abstract |

**Important:** The BrainSegFounder-Tiny model has 2 Swin blocks per stage for all stages. Stages 3–4 have the QKV projections targeted by LoRA. The hidden state is extracted *after* the stage's blocks and patch merging (for stages 0–3) or after the final block (stage 4).

The agent should verify the shapes by running:
```python
model = load_full_swinunetr(checkpoint, freeze_encoder=True, freeze_decoder=True, out_channels=3, device="cuda")
x = torch.randn(1, 4, 192, 192, 192, device="cuda")
hidden_states = model.swinViT(x, model.normalize)
for i, h in enumerate(hidden_states):
    print(f"Stage {i}: {h.shape}")
```

---

## 4. The figure

### 4.1 Layout

A **5-column × 3-row panel** figure, `figsize=(12, 7)`:

```
Columns:    Stage 0        Stage 1        Stage 2        Stage 3        Stage 4

Row A:    [Mean activation heatmap overlaid on MRI axial slice]
          (upsampled to input resolution, normalised per-panel)

Row B:    [Top-3 tumor-selective channels overlaid on MRI]
          (the 3 channels with highest TSI, averaged, contoured)

Row C:    [TSI histogram per stage]
          (x = TSI value, y = channel count, vertical line at τ=1.5)
```

**Column header annotations:** Stage name, channel count, resolution, and condition label.

### 4.2 Row A: Mean activation heatmap

For each stage $s$:
1. Compute $\bar{a}^{(s)} = \frac{1}{C_s} \sum_c |h^{(s)}_c| \in \mathbb{R}^{D_s \times H_s \times W_s}$.
2. Trilinearly upsample to input resolution: `F.interpolate(a, size=(D, H, W), mode='trilinear')`.
3. Extract the axial slice with maximum tumor cross-section (same slice index for all panels).
4. Overlay as a heatmap on the T1ce axial slice (MRI as grayscale background, activation as `inferno` colormap with alpha=0.5).
5. Draw the GT tumor contour in white dashed line.
6. Normalise each panel's colormap independently (each stage has different activation magnitudes).

**Expected visual progression:** Stage 0 shows activation across the entire brain (edges, sulci). Stage 4 shows a hot blob concentrated on the tumor.

### 4.3 Row B: Top tumor-selective channels

For each stage:
1. Identify the 3 channels with highest TSI.
2. Compute their mean activation (same upsampling procedure).
3. Extract the same axial slice.
4. Overlay on MRI as a semi-transparent heatmap (distinct colormap, e.g., `plasma`).
5. Draw contours of the top-channel activation at the 75th percentile level to show where these specific channels "fire."

**Expected visual:** For Stages 0–1, even the most tumor-selective channels are relatively diffuse. For Stages 3–4, the top channels tightly delineate the tumor region.

### 4.4 Row C: TSI distribution histograms

For each stage:
1. Plot a histogram of TSI values across all $C_s$ channels (20 bins, range [0, 4] with overflow bin).
2. Vertical dashed red line at TSI = 1.5 (selectivity threshold $\tau$).
3. Vertical solid grey line at TSI = 1.0 (null hypothesis: no selectivity).
4. Annotate: `"Frac(TSI > 1.5) = X%"`.
5. Fill the region TSI > 1.5 in a distinct colour to make the fraction visually immediate.

**Expected visual:** Stage 0 histogram is centred near 1.0 with almost no mass above 1.5. Stage 4 histogram is right-shifted with substantial mass above 1.5.

### 4.5 Frozen vs. adapted comparison

The figure should show **both conditions** for Row C (the histograms). Two approaches:

**Option A (recommended for clarity):** Two separate histogram figures — one for frozen, one for adapted. Same layout. The visual comparison is between the two figures.

**Option B (compact):** Overlay the two histograms in Row C using different colours (frozen = grey, adapted = blue) with alpha blending. This fits in one figure but may be cluttered.

The agent should implement **Option A**: two 5×3 panel figures. Call them `fig_tsi_frozen.pdf` and `fig_tsi_adapted.pdf`. For the thesis, they will appear side-by-side or one after another.

**Additionally,** produce a **difference summary panel** (a small 1×5 figure): for each stage, show the Δ(mean TSI) between adapted and frozen, with error bars (95% CI across scans). Stages 0–2 should have Δ ≈ 0 (no LoRA applied); Stages 3–4 should have Δ > 0 if LoRA increased selectivity.

---

## 5. The table

**Table 1: Tumor Selectivity Index across SwinViT encoder stages (N=20 BraTS-MEN test scans).**

| Stage | $C_s$ | Resolution | Condition | Mean TSI ± SD | Frac(TSI>1.5) | Frac(TSI>2.0) | Wilcoxon p ($H_0$: median=1) |
|-------|--------|-----------|-----------|--------------|---------------|---------------|---------------------------|
| 0 | 48 | 48³ | Frozen | ? ± ? | ?% | ?% | ? |
| 0 | 48 | 48³ | LoRA-adapted | ? ± ? | ?% | ?% | ? |
| 1 | 96 | 24³ | Frozen | ? ± ? | ?% | ?% | ? |
| 1 | 96 | 24³ | LoRA-adapted | ? ± ? | ?% | ?% | ? |
| 2 | 192 | 12³ | Frozen | ? ± ? | ?% | ?% | ? |
| 2 | 192 | 12³ | LoRA-adapted | ? ± ? | ?% | ?% | ? |
| 3 | 384 | 6³ | Frozen | ? ± ? | ?% | ?% | ? |
| 3 | 384 | 6³ | LoRA-adapted | ? ± ? | ?% | ?% | ? |
| 4 | 768 | 3³ | Frozen | ? ± ? | ?% | ?% | ? |
| 4 | 768 | 3³ | LoRA-adapted | ? ± ? | ?% | ?% | ? |

Additional row at bottom: **Paired Wilcoxon test (Adapted vs. Frozen per stage)** — for each stage, test whether the mean TSI across channels is higher in the adapted condition. For Stages 0–2, expect $p > 0.05$ (no difference). For Stages 3–4, expect $p < 0.05$ (LoRA increased selectivity).

**Multi-scan aggregation:** The TSI values in the table are first computed per scan (giving one TSI distribution per stage per scan), then the summary statistics (mean TSI, Frac>τ) are averaged across $N$ scans. The ± values are standard deviations across scans. The Wilcoxon test operates on the per-scan mean TSI values: for each scan, the mean TSI at stage $s$ is one data point, giving $N$ paired observations for the frozen-vs-adapted test.

---

## 6. Module structure

```
experiments/uncertainty_segmentation/feature_analysis/
├── __init__.py
├── tsi_analysis.py          # Core: compute_tsi(), compute_tsi_multi_scan()
├── model_loader.py          # Load frozen and LoRA-adapted models
├── figure_tsi.py            # Generate the figure panels
├── table_tsi.py             # Generate the table (as CSV + LaTeX)
├── run_analysis.py          # CLI entry point
└── config.yaml              # Analysis-specific config
```

### 6.1 `config.yaml`

```yaml
# Feature analysis configuration

paths:
  # Will be read from the parent uncertainty_segmentation config if not set
  checkpoint_dir: null
  checkpoint_filename: finetuned_model_fold_0.pt
  men_h5_file: null
  
  # LoRA-adapted model (one member from a completed run)
  run_dir: null          # e.g., results/uncertainty_segmentation/r8_M10_s42
  member_id: 0           # Which ensemble member to use for the adapted condition

analysis:
  n_scans: 20            # Number of test scans to analyse
  scan_selection: "random"  # "random" | "first" | explicit list
  seed: 42               # For reproducible scan sampling
  tsi_thresholds: [1.5, 2.0]
  epsilon: 1.0e-8

figure:
  figsize_main: [12, 7]     # 5×3 panel figure
  figsize_delta: [7, 2.5]   # 1×5 delta summary
  colormap_activation: "inferno"
  colormap_selective: "plasma"
  slice_selection: "max_tumor"  # "max_tumor" | "center" | explicit index
  save_format: "pdf"
  save_dpi: 300
```

### 6.2 `tsi_analysis.py` — Core computation

```python
@dataclasses.dataclass
class TSIResult:
    """TSI computation result for one scan at one stage."""
    stage: int
    n_channels: int
    resolution: tuple[int, int, int]
    tsi_per_channel: np.ndarray      # [C_s]
    mean_tsi: float
    std_tsi: float
    frac_above: dict[float, float]   # {threshold: fraction}
    wilcoxon_p: float                # H0: median TSI = 1
    top_k_channels: list[int]        # Indices of top-K TSI channels
    
    # For visualisation (optional, set via flag)
    mean_activation_map: np.ndarray | None   # [D_s, H_s, W_s]
    top_channels_map: np.ndarray | None      # [D_s, H_s, W_s] (mean of top-K)


@dataclasses.dataclass 
class ScanTSIResult:
    """TSI results for all stages of one scan, under one model condition."""
    scan_id: str
    condition: str                    # "frozen" or "adapted"
    stages: list[TSIResult]           # len = 5 (one per stage)


def compute_tsi_single_scan(
    model: torch.nn.Module,
    images: torch.Tensor,
    gt_mask: torch.Tensor,
    thresholds: list[float] = [1.5, 2.0],
    top_k: int = 3,
    return_maps: bool = False,
    epsilon: float = 1e-8,
) -> list[TSIResult]:
    """Compute TSI for all 5 stages on one scan.
    
    Args:
        model: SwinUNETR model (frozen or LoRA-adapted).
        images: Input [1, 4, D, H, W].
        gt_mask: Binary WT mask [D, H, W] or [1, D, H, W].
        thresholds: TSI thresholds for Frac computation.
        top_k: Number of top tumor-selective channels to track.
        return_maps: If True, store activation maps for visualisation.
        epsilon: Numerical stability constant.
    
    Returns:
        List of 5 TSIResult, one per stage.
    """
```

Implementation notes:
- Use `model.swinViT(images, model.normalize)` to get hidden states.
- For the LoRA-adapted model, the hidden states from Stages 0–2 should be identical to frozen (since LoRA is only on 3–4 and stages 0–2 have no trainable parameters). **However**, this is only true if the LoRA forward pass does not modify Stages 0–2 at all. Verify this empirically — if there is any numerical difference, it should be below float32 precision.
- The Wilcoxon test: `scipy.stats.wilcoxon(tsi_per_channel - 1.0, alternative='greater')`. This tests whether the channels have median TSI > 1. Use `alternative='greater'` (one-sided) because the scientific hypothesis is directional.
- For `return_maps=True`: compute `mean_activation_map = abs(hidden_states[s]).mean(dim=1).squeeze(0)` and `top_channels_map = abs(hidden_states[s][:, top_indices]).mean(dim=1).squeeze(0)`.

### 6.3 `model_loader.py` — Load both conditions

```python
def load_frozen_model(config: DictConfig, device: str = "cuda") -> nn.Module:
    """Load original BrainSegFounder without any adaptation."""
    
def load_adapted_model(config: DictConfig, device: str = "cuda") -> nn.Module:
    """Load LoRA-adapted model (one ensemble member).
    
    Uses the same loading path as EnsemblePredictor._load_member_model().
    Loads: base checkpoint → LoRA adapter → trained decoder.
    """
```

For the adapted model, reuse `EnsemblePredictor._load_member_model()` or replicate its logic:
1. `load_full_swinunetr(checkpoint_path, freeze_encoder=True, freeze_decoder=True, out_channels=3)`
2. `LoRASwinViT.load_lora(base_encoder=full_model, adapter_path=..., trainable=False)`
3. `LoRAOriginalDecoderModel(lora_encoder=..., freeze_decoder=True)`
4. `model.decoder.load_state_dict(torch.load(decoder.pt))`

**Important:** The hidden states come from the encoder (SwinViT), not the decoder. But the decoder weights are loaded because the model wrapper expects them to be present. For TSI computation, only the encoder output matters.

**For the adapted model, the agent must extract hidden states from the LoRA-wrapped encoder.** The method `model.swinViT(x, model.normalize)` should work if the model is a `LoRAOriginalDecoderModel` that wraps a `LoRASwinViT` whose inner `model` is the PEFT-wrapped SwinUNETR. The agent should trace through the code to find the correct call path:
- `LoRAOriginalDecoderModel.lora_encoder.model.swinViT(x, normalize)` — this goes through the PEFT wrapper and applies LoRA during the forward pass.

Test by verifying that `hidden_states[0]` from the frozen and adapted models are numerically identical (since Stage 0 has no LoRA) while `hidden_states[3]` and `hidden_states[4]` differ.

### 6.4 `figure_tsi.py` — Visualisation

The main function:

```python
def generate_tsi_figure(
    scan_result_frozen: ScanTSIResult,
    scan_result_adapted: ScanTSIResult,
    mri_slice: np.ndarray,          # 2D T1ce axial slice for background
    gt_mask_slice: np.ndarray,      # 2D GT mask axial slice
    slice_idx: int,
    config: dict,
    output_dir: Path,
) -> tuple[Figure, Figure, Figure]:
    """Generate the three figures: frozen panel, adapted panel, delta summary.
    
    Returns (fig_frozen, fig_adapted, fig_delta).
    """
```

For the MRI background slice: load from the BraTS-MEN HDF5 file. Channel index 1 = T1ce (in the [FLAIR, T1ce, T1, T2] ordering). Extract the axial slice at the selected index.

### 6.5 `table_tsi.py` — Table generation

```python
def generate_tsi_table(
    all_results_frozen: list[ScanTSIResult],
    all_results_adapted: list[ScanTSIResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Aggregate TSI across N scans and produce the table.
    
    Saves:
        - tsi_table.csv (machine-readable)
        - tsi_table.tex (LaTeX-formatted for thesis)
    
    Returns the DataFrame.
    """
```

For the cross-condition Wilcoxon test: for each stage $s$, collect the per-scan mean TSI under both conditions → paired Wilcoxon. Report in an additional panel of the table or as a footnote.

### 6.6 `run_analysis.py` — CLI entry point

```bash
python -m experiments.uncertainty_segmentation.feature_analysis.run_analysis \
    --config experiments/uncertainty_segmentation/feature_analysis/config.yaml \
    --run-dir results/uncertainty_segmentation/r8_M10_s42 \
    --output results/uncertainty_segmentation/r8_M10_s42/feature_analysis/ \
    --device cuda:0
```

The script:
1. Loads both models (frozen + adapted member 0).
2. Loads $N$ test scans from BraTS-MEN H5.
3. For each scan, computes TSI under both conditions (first scan with `return_maps=True` for visualisation; remaining scans without maps to save memory).
4. Generates the figure (using the first scan's maps) and the table (using all $N$ scans).
5. Saves everything to the output directory.

---

## 7. Compute cost

- 2 models × 20 scans × 1 forward pass = 40 forward passes.
- Each forward pass on 192³ input takes ~3 seconds on A100.
- Total: ~2 minutes. No backpropagation, no training.
- This runs comfortably on a single GPU. No SLURM array needed.

---

## 8. Output structure

```
{run_dir}/feature_analysis/
├── figures/
│   ├── tsi_frozen.pdf           # 5×3 panel: frozen BrainSegFounder
│   ├── tsi_adapted.pdf          # 5×3 panel: LoRA-adapted model
│   └── tsi_delta.pdf            # 1×5 summary: Δ(mean TSI) per stage
├── tables/
│   ├── tsi_table.csv            # Machine-readable aggregate table
│   └── tsi_table.tex            # LaTeX-formatted for thesis
├── data/
│   ├── tsi_frozen_per_scan.csv  # Per-scan, per-stage TSI summaries (frozen)
│   ├── tsi_adapted_per_scan.csv # Per-scan, per-stage TSI summaries (adapted)
│   └── tsi_all_channels.npz    # Raw per-channel TSI arrays (for reproducibility)
└── config_snapshot.yaml         # Resolved config used for this analysis
```

---

## 9. Verification checklist

- [ ] Frozen model hidden states shape: `[1, 48, 48, 48, 48]`, ..., `[1, 768, 3, 3, 3]` (verify at runtime).
- [ ] `hidden_states[0]` from frozen and adapted models are numerically identical (Stages 0–2 have no LoRA).
- [ ] `hidden_states[3]` from frozen and adapted models differ (Stage 3 has LoRA).
- [ ] TSI of a constant activation (all ones) is 1.0 regardless of mask shape.
- [ ] Wilcoxon test on a vector of all 1.0s returns $p = 1.0$ (no evidence against $H_0$).
- [ ] Figure panels render without error; each stage column has correct resolution annotation.
- [ ] Table LaTeX compiles without error in the thesis build.
- [ ] Total runtime < 5 minutes on a single A100.

---

## 10. Coding standards

Same as the parent `uncertainty_segmentation` module:
- Type hints, Google docstrings, structured logging, no magic numbers.
- `torch.no_grad()` everywhere (no training, no gradients needed).
- Memory management: delete hidden states after TSI computation, `torch.cuda.empty_cache()` between scans.
- All paths derived from config; no hardcoded paths.
