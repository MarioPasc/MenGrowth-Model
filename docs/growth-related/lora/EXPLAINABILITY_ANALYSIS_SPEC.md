# SPEC: Encoder Explainability Analysis — TSI + ASI + DAD

**Date:** 2026-04-14  
**Target Agent:** Claude Code Opus 4.6  
**Location:** `experiments/uncertainty_segmentation/explainability/`  
**Replaces:** The previous `feature_analysis/` module (TSI-only)  
**Compute budget:** < 15 min on a single A100 (no training, no backprop)  
**Deliverables:** Three metrics, three figures, two tables, all raw data for regeneration

---

## 0. Context and objective

This module produces the empirical evidence for **which SwinViT encoder stages should receive LoRA adaptation** for meningioma domain transfer. The previous TSI-only analysis had two flaws: (1) background voxels (air) inflated the selectivity of shallow stages, and (2) stage-output activations are dominated by the residual connection, making them insensitive to what the attention mechanism actually does.

This module implements three complementary metrics that address these flaws:

| Metric | What it measures | Where it hooks | Confound addressed |
|--------|-----------------|----------------|-------------------|
| **TSI** (corrected) | Spatial feature selectivity (brain-masked) | Stage output activations | Background dilution |
| **ASI** | Attention pattern tumor-selectivity | Attention weight matrices inside Swin blocks | Residual connection bypass |
| **DAD** | Cross-domain attention divergence (GLI vs MEN) | Attention weight matrices | Justifies *why* adaptation is needed |

Together, TSI answers "what does this stage represent?", ASI answers "does the attention mechanism focus on tumor?", and DAD answers "does the attention pattern need to change for meningioma?"

---

## 1. Architecture reference

### 1.1 SwinViT encoder hierarchy (BrainSegFounder-Tiny, 62M)

For an input of shape `[1, 4, 192, 192, 192]` (4 MRI modalities, 192³ isotropic):

| Stage | MONAI path | Blocks | Heads | Channels | Token resolution | Window size |
|-------|-----------|--------|-------|----------|-----------------|-------------|
| 0 | `swinViT.layers1[0].blocks[0,1]` | 2 | 3 | 48 | 96³ | 7³ (or clamped) |
| 1 | `swinViT.layers2[0].blocks[0,1]` | 2 | 6 | 96 | 48³ | 7³ |
| 2 | `swinViT.layers3[0].blocks[0,1]` | 2 | 12 | 192 | 24³ | 7³ |
| 3 | `swinViT.layers4[0].blocks[0,1]` | 2 | 24 | 384 | 12³ | 7³ (clamped to 12) |
| 4 | `swinViT.layers4[0]` → bottleneck | 2 | 24 | 768 | 6³ | 6 (clamped) |

**Important:** The agent must verify these paths and shapes at runtime. Use:
```python
for name, module in model.swinViT.named_modules():
    if "WindowAttention" in type(module).__name__:
        print(name, module.num_heads, module.dim)
```

**Note on stage 4:** In MONAI's SwinUNETR, stage 4 is the output of `swinViT.layers4` (same as stage 3's deeper blocks in some implementations). The bottleneck is processed by `encoder10` which is a convolutional block (no attention). The agent must trace the exact model structure to determine which `WindowAttention` modules belong to which stage.

### 1.2 WindowAttention forward pass

The MONAI `WindowAttention` class computes:

```python
class WindowAttention(nn.Module):
    def forward(self, x, mask=None):
        b, n, c = x.shape                          # b = n_windows × batch, n = window_size³, c = dim
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # [3, b, heads, n, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)              # [b, heads, n, n]
        attn = attn + self.relative_position_bias   # learned relative position bias
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float(-100.0))
        attn = self.softmax(attn)                    # [b, heads, n, n] — THIS IS WHAT WE NEED
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

The tensor `attn` after softmax is the attention weight matrix $\mathbf{A} \in [0,1]^{B \times H \times N \times N}$ where $B$ = number of windows × batch, $H$ = number of heads, $N$ = tokens per window.

### 1.3 Hook strategy

We register a forward hook on each `WindowAttention` module that captures the attention weights *after softmax but before dropout*. The cleanest approach: temporarily monkey-patch the `forward` method to store `attn` in a buffer, or use a hook on the `softmax` output.

**Recommended implementation:** Subclass or patch `WindowAttention.forward` to store `self._last_attn = attn` before the dropout line. Then access `module._last_attn` after the forward pass. This avoids the complexity of registering intermediate hooks on tensor operations.

Alternative (cleaner): Use `torch.nn.Module.register_forward_hook` but note that the hook receives `(module, input, output)` where `output` is the projected `x`, not the attention weights. We need the attention weights specifically, so the monkey-patch approach is more reliable.

**The agent should implement this as a context manager** that patches all `WindowAttention` modules in the model, runs inference, collects attention maps, and then restores the original forward methods.

---

## 2. Datasets and their roles

| Dataset | H5 file | Has GT masks | Role in this analysis |
|---------|---------|-------------|----------------------|
| BraTS-MEN | `BraTS_MEN.h5` | ✅ Yes (WT, TC, ET) | Primary TSI/ASI computation (meningioma features) |
| BraTS-GLI | `BraTS_GLI.h5` | ✅ Yes | DAD cross-domain comparison (glioma vs meningioma attention) |
| MenGrowth | `MenGrowth.h5` | ❌ No labels | *Not used* in this analysis (no GT masks for TSI/ASI) |

**Scan selection:** Use the BraTS-MEN test split (N≈150 scans). For TSI/ASI, randomly sample $N_{\text{TSI}} = 20$ scans (configurable). For DAD, sample $N_{\text{DAD}} = 20$ MEN scans and 20 GLI scans (matched by approximate tumor volume to control for size effects).

**Brain mask derivation:** For each scan, compute $\mathbf{b} = \mathbb{1}[\text{T1ce} > \tau]$ where T1ce is channel index 1 (in [FLAIR, T1ce, T1, T2] ordering) and $\tau$ is a small intensity threshold. The BraTS data is skull-stripped and z-score normalised, so $\tau = 0.01$ after normalisation reliably separates brain tissue from background. Verify visually on 2–3 scans.

---

## 3. Metric definitions

### 3.1 TSI (Tumor Selectivity Index) — corrected

**Change from previous version:** The non-tumor denominator now excludes background voxels.

For stage $s$, channel $c$, with hidden state $\mathbf{h}^{(s)} \in \mathbb{R}^{C_s \times D_s \times H_s \times W_s}$ upsampled to input resolution $[D, H, W]$:

Define three masks at input resolution:
- Tumor mask: $\mathcal{T} = \{v : m_v = 1\}$
- Brain-non-tumor mask: $\mathcal{B} = \{v : b_v = 1 \wedge m_v = 0\}$  
- Background: $\bar{\mathcal{B}} = \{v : b_v = 0\}$ (excluded)

$$\text{TSI}_{s,c} = \frac{\mu^{\mathcal{T}}_{s,c}}{\mu^{\mathcal{B}}_{s,c} + \epsilon}, \quad \mu^{\mathcal{T}}_{s,c} = \frac{\sum_{v \in \mathcal{T}} |h^{(s)}_{c,v}|}{|\mathcal{T}|}, \quad \mu^{\mathcal{B}}_{s,c} = \frac{\sum_{v \in \mathcal{B}} |h^{(s)}_{c,v}|}{|\mathcal{B}|}$$

**Implementation:** Upsample the hidden state to 192³ via trilinear interpolation before computing TSI. This eliminates the mask-disappearance confound at coarse stages.

```python
h_up = F.interpolate(
    hidden_states[s].float(), size=(D, H, W), mode='trilinear', align_corners=False
)
```

**Per-stage summary:**
- $\overline{\text{TSI}}_s = \text{mean}_c(\text{TSI}_{s,c})$, $\text{SD}_s$
- $\text{Frac}_s(\tau)$ for $\tau \in \{1.5, 2.0\}$
- One-sample Wilcoxon signed-rank test: $H_0: \text{median}(\text{TSI}_{s,\cdot}) = 1$ (one-sided, alternative='greater')

### 3.2 ASI (Attention Selectivity Index)

**Purpose:** Measure whether the attention mechanism at each stage routes information preferentially between tumor tokens.

**Definition.** For head $h$ in block $b$ of stage $s$, consider a window $w$ that contains both tumor and non-tumor tokens. Let the attention matrix for this window be $\mathbf{A}^{h} \in [0,1]^{N \times N}$ where $N$ = tokens per window. Partition the $N$ tokens into tumor set $\mathcal{T}_w$ and non-tumor set $\mathcal{N}_w$ using the GT mask downsampled to the token resolution of stage $s$.

$$\text{ASI}^{(s,b)}_{h,w} = \frac{\bar{A}^h_{\mathcal{T} \to \mathcal{T}}}{\bar{A}^h_{\mathcal{T} \to \mathcal{N}} + \epsilon}$$

where:

$$\bar{A}^h_{\mathcal{T} \to \mathcal{T}} = \frac{1}{|\mathcal{T}_w|} \sum_{i \in \mathcal{T}_w} \frac{\sum_{j \in \mathcal{T}_w} A^h_{ij}}{|\mathcal{T}_w|}, \quad \bar{A}^h_{\mathcal{T} \to \mathcal{N}} = \frac{1}{|\mathcal{T}_w|} \sum_{i \in \mathcal{T}_w} \frac{\sum_{j \in \mathcal{N}_w} A^h_{ij}}{|\mathcal{N}_w|}$$

In words: for each tumor query token $i$, compute the average attention it pays to other tumor tokens vs. non-tumor tokens, normalised by group size. Then average over all tumor query tokens in the window. The ratio gives ASI.

**Interpretation:**
- ASI $\gg 1$: tumor tokens preferentially attend to each other → the attention mechanism is tumor-selective.
- ASI $\approx 1$: attention is spatially indiscriminate → no tumor-specific routing.
- ASI $< 1$: tumor tokens attend more to non-tumor context → cross-boundary attention.

**Window selection:** ASI is only defined for *boundary windows* — windows that contain at least $\min_{\mathcal{T}} = 5$ tumor tokens and $\min_{\mathcal{N}} = 5$ non-tumor tokens. Interior and exterior windows are excluded. This is by design: boundary windows are where segmentation uncertainty lives and where LoRA adaptation matters most.

**Token-to-mask mapping:** Each token at stage $s$ covers a spatial region of $(2 \times 2^s)^3$ input voxels (patch size 2, then halving at each stage via patch merging). To assign a token to tumor/non-tumor, downsample the GT mask to the token grid of stage $s$ and threshold: a token is "tumor" if > 50% of its receptive field overlaps with the GT mask. Use `F.avg_pool3d(mask, kernel_size=2**(s+1))` and threshold at 0.5.

**Aggregation:**
- Per-stage ASI: average over all boundary windows, all heads, all blocks in the stage.
- Per-head ASI: for visualisation, keep head-level resolution.
- Per-scan ASI: compute per scan, then report mean ± std across $N$ scans.

### 3.3 DAD (Domain Attention Divergence)

**Purpose:** Quantify how much the attention pattern at each stage differs between GLI and MEN inputs. High DAD → LoRA needed to adapt the attention for meningioma.

**Definition.** For stage $s$, block $b$, head $h$, and a matched pair of scans (one GLI, one MEN), extract the attention matrices from corresponding boundary windows. The DAD is the symmetric KL divergence between the attention distributions:

$$\text{DAD}^{(s,b)}_{h} = \frac{1}{2}\left[D_{\text{KL}}(\bar{\mathbf{a}}^{\text{GLI}}_h \| \bar{\mathbf{a}}^{\text{MEN}}_h) + D_{\text{KL}}(\bar{\mathbf{a}}^{\text{MEN}}_h \| \bar{\mathbf{a}}^{\text{GLI}}_h)\right]$$

where $\bar{\mathbf{a}}^{\text{GLI}}_h$ is the average attention distribution (averaged over all query tokens and boundary windows) for head $h$ under GLI input, and similarly for MEN.

**Practical computation:** Because GLI and MEN scans have different tumors at different locations, we cannot align window-to-window. Instead:

1. For each domain (GLI and MEN), collect all boundary-window attention matrices.
2. For each head $h$, compute the average row-wise attention distribution: $\bar{a}^{\text{domain}}_{h,j} = \mathbb{E}_{i,w}[A^h_{i,j}]$ where $j$ is the position within the window. This gives a distribution over the $N$ window positions.
3. Compute symmetric KL between the GLI and MEN average distributions.

This measures whether the *spatial attention pattern* within windows differs between domains, averaged over many windows.

**Permutation null:** To assess significance, permute the domain labels (shuffle GLI and MEN scans), recompute DAD 1000 times, and report the $p$-value as the fraction of permuted DADs exceeding the observed DAD.

---

## 4. Module structure

```
experiments/uncertainty_segmentation/explainability/
├── __init__.py
├── config.yaml                     # Analysis configuration
│
├── engine/
│   ├── __init__.py
│   ├── hooks.py                    # AttentionCaptureContext: patches WindowAttention to store attn
│   ├── brain_mask.py               # Brain mask derivation from MRI intensity
│   ├── tsi.py                      # Brain-masked TSI computation
│   ├── asi.py                      # Attention Selectivity Index computation
│   ├── dad.py                      # Domain Attention Divergence computation
│   ├── model_loader.py             # Load frozen / LoRA-adapted models
│   └── data_loader.py              # Load scans + GT masks from H5 files
│
├── figures/
│   ├── __init__.py
│   ├── fig_tsi_panel.py            # 5×3 panel figure (corrected TSI)
│   ├── fig_asi_panel.py            # ASI per stage with head-level detail
│   ├── fig_dad_bar.py              # DAD per stage with significance
│   └── fig_summary.py              # Combined summary figure for thesis
│
├── tables/
│   ├── __init__.py
│   └── generate_tables.py          # CSV + LaTeX table generation
│
├── run_analysis.py                 # CLI: full pipeline (compute + figures + tables)
├── run_figures_only.py             # CLI: regenerate figures from saved data (no GPU)
└── run_single_metric.py            # CLI: compute one metric only (for debugging)
```

---

## 5. Core implementation: `hooks.py`

This is the most architecturally sensitive file. It must correctly identify and patch every `WindowAttention` module in the model.

```python
"""Context manager for capturing attention weights from SwinViT WindowAttention modules.

Usage:
    with AttentionCapture(model) as capture:
        hidden_states = model.swinViT(x, model.normalize)
        attn_maps = capture.get_attention_maps()
        # attn_maps: dict[str, list[Tensor]]
        # keys: "stage_0_block_0", "stage_0_block_1", "stage_1_block_0", ...
        # values: list of attention tensors [n_windows*batch, heads, N, N]
"""

@dataclasses.dataclass
class CapturedAttention:
    """Attention data from one WindowAttention module."""
    stage: int
    block: int
    num_heads: int
    window_size: tuple[int, ...]
    attn_weights: torch.Tensor    # [n_windows, heads, N, N] after softmax
    
class AttentionCapture:
    """Context manager that patches WindowAttention modules to capture attention weights."""
    
    def __init__(self, model: nn.Module) -> None:
        ...
    
    def __enter__(self) -> "AttentionCapture":
        """Discover all WindowAttention modules, patch their forward methods."""
        ...
    
    def __exit__(self, *args) -> None:
        """Restore original forward methods, release captured tensors."""
        ...
    
    def get_attention_maps(self) -> dict[str, CapturedAttention]:
        """Return captured attention data, keyed by 'stage_{s}_block_{b}'."""
        ...
```

**Implementation strategy for patching:**

```python
def _create_patched_forward(self, original_forward, key):
    """Create a new forward that stores attn weights."""
    captured = self._captured  # reference to the dict
    
    def patched_forward(x, mask=None):
        b, n, c = x.shape
        qkv = module.qkv(x).reshape(b, n, 3, module.num_heads, c // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * module.scale
        attn = q @ k.transpose(-2, -1)
        
        # Add relative position bias
        relative_position_bias = module.relative_position_bias_table[
            module.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, module.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, module.num_heads, n, n)
        
        attn = module.softmax(attn)
        
        # ---- CAPTURE HERE ----
        captured[key] = attn.detach().cpu()
        # ----------------------
        
        attn = module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = module.proj(x)
        x = module.proj_drop(x)
        return x
    
    return patched_forward
```

**CRITICAL:** The agent must verify this matches the actual MONAI `WindowAttention.forward()` in the installed version. The implementation above is based on MONAI 1.0–1.3. If the version differs (e.g., uses flash attention), adapt accordingly. The key constraint is: capture `attn` after softmax, before dropout.

**Memory note:** Attention matrices can be large. At stage 0 (96³ tokens, window size 7³ = 343 tokens), there are $(96/7)^3 \approx 2744$ windows per volume. Each attention matrix is $[343 \times 343]$ floats = 470K values. Total for stage 0: $2744 \times 3\ \text{heads} \times 343^2 \times 4\ \text{bytes} \approx 3.8\ \text{GB}$. This is too large.

**Solution:** Do NOT capture full attention matrices at stage 0. Capture only at stages $\geq 1$ where the window count is manageable. For TSI, stage 0 uses the existing (non-attention) activation approach which is sufficient. For ASI and DAD, start from stage 1.

Alternatively, compute ASI/DAD statistics *within the patched forward* and store only the summary, not the full attention tensor. This is the recommended approach for memory efficiency:

```python
# Inside the patched forward, after softmax:
# Compute ASI summary statistics on-the-fly
if self._compute_asi and key in self._target_stages:
    asi_stats = compute_window_asi(attn, token_mask_for_this_stage)
    self._asi_buffer[key].append(asi_stats)
# Store only the summary, not the full [n_windows, heads, N, N] tensor
```

---

## 6. Config

```yaml
# experiments/uncertainty_segmentation/explainability/config.yaml

paths:
  checkpoint_dir: null        # Override from parent config or CLI
  checkpoint_filename: finetuned_model_fold_0.pt
  men_h5_file: null
  gli_h5_file: null           # For DAD computation
  run_dir: null               # For loading LoRA-adapted model
  member_id: 0                # Which ensemble member for the adapted condition

analysis:
  # Scan selection
  n_scans_tsi: 20
  n_scans_asi: 20
  n_scans_dad: 20             # Per domain (20 GLI + 20 MEN)
  scan_selection: "random"
  seed: 42
  
  # TSI
  tsi_thresholds: [1.5, 2.0]
  brain_mask_threshold: 0.01   # Intensity threshold for brain vs background
  tsi_top_k: 3                 # Top-K most selective channels for visualisation
  
  # ASI
  asi_min_tumor_tokens: 5      # Minimum tumor tokens for a boundary window
  asi_min_nontumor_tokens: 5
  asi_stages: [1, 2, 3, 4]    # Stages to compute ASI (skip stage 0 for memory)
  token_tumor_threshold: 0.5   # Token is "tumor" if > 50% mask overlap
  
  # DAD
  dad_permutations: 1000       # Permutation test iterations
  dad_stages: [1, 2, 3, 4]

  # Conditions
  conditions:
    - name: "frozen"
      description: "Original BrainSegFounder, no adaptation"
      model_type: "frozen"
    - name: "adapted_r8"
      description: "LoRA r=8 ensemble member 0"
      model_type: "lora_adapted"
      rank: 8

figure:
  save_format: "pdf"
  save_dpi: 300
  tsi_panel_figsize: [12, 7]
  asi_panel_figsize: [10, 5]
  dad_bar_figsize: [6, 4]
  summary_figsize: [14, 5]
  colormap_activation: "inferno"
  slice_selection: "max_tumor"
```

---

## 7. Figures

### 7.1 Figure: Brain-masked TSI panel (corrected)

Same 5×3 layout as the previous TSI figure, but with the brain mask correction applied. The figure title should explicitly state "Brain-masked" to distinguish from the uncorrected version.

Row A: Mean activation heatmap (brain-masked: set background voxels to NaN before overlaying).
Row B: Top-3 TSI channels (same correction).
Row C: TSI histograms with corrected values.

Produce TWO versions: `tsi_frozen_brainmasked.pdf` and `tsi_adapted_brainmasked.pdf`.

### 7.2 Figure: ASI per stage

**Layout:** A 1×4 panel (stages 1–4), each showing:

- **Box plot** of per-window ASI values (aggregated across scans), one box per head within the stage. X-axis = head index. Y-axis = ASI. Horizontal line at ASI = 1.0 (null).
- **Annotation:** Per-stage median ASI and Wilcoxon $p$ (testing median > 1).
- **Color:** Heads with significant ASI (> 1.5) highlighted.

Produce for both conditions (frozen, adapted). If comparing, overlay as paired box plots.

### 7.3 Figure: DAD per stage

**Layout:** Single bar chart. X-axis = stage (1, 2, 3, 4). Y-axis = DAD (symmetric KL). Error bars = std across heads. Stars for permutation test significance.

**Annotation:** p-values from permutation test above each bar.

### 7.4 Figure: Thesis summary (combined)

**Layout:** A single 3-panel figure for the thesis:

- **Panel (a):** Bar chart of brain-masked mean TSI per stage (with 95% CI across scans). Shows feature selectivity gradient.
- **Panel (b):** Bar chart of mean ASI per stage (with 95% CI). Shows attention selectivity gradient.
- **Panel (c):** Bar chart of DAD per stage (with permutation p-values). Shows where domain adaptation is needed.

Each panel has a horizontal dashed line at the "null" value (TSI=1.0, ASI=1.0, DAD=0.0). Stages selected for LoRA are highlighted with a coloured background band.

This is the single figure that justifies the stage selection. It should be self-contained.

---

## 8. Tables

### 8.1 Table 1: TSI + ASI combined

| Stage | $C_s$ | Condition | Mean TSI ± SD | Frac(TSI>1.5) | Wilcoxon $p$ (TSI>1) | Mean ASI ± SD | Frac(ASI>1.5) | Wilcoxon $p$ (ASI>1) |
|-------|--------|-----------|---------------|---------------|-----------------------|---------------|---------------|-----------------------|
| 0 | 48 | Frozen | ? | ? | ? | — | — | — |
| 1 | 96 | Frozen | ? | ? | ? | ? | ? | ? |
| 1 | 96 | Adapted | ? | ? | ? | ? | ? | ? |
| ... | | | | | | | | |

ASI is reported starting from stage 1 (stage 0 skipped for memory).

### 8.2 Table 2: DAD with permutation significance

| Stage | Heads | Mean DAD | SD (across heads) | Permutation $p$ | Interpretation |
|-------|-------|---------|-------------------|-----------------|----------------|
| 1 | 6 | ? | ? | ? | ? |
| 2 | 12 | ? | ? | ? | ? |
| 3 | 24 | ? | ? | ? | ? |
| 4 | 24 | ? | ? | ? | ? |

---

## 9. Output structure (for regeneration without re-running)

```
{output_dir}/
├── config_snapshot.yaml                 # Frozen config
│
├── raw/                                 # Raw data (GPU-computed, expensive)
│   ├── tsi_frozen_per_scan.csv          # Per-scan, per-stage, per-channel TSI
│   ├── tsi_adapted_per_scan.csv         # Same for adapted model
│   ├── tsi_frozen_channels.npz          # Full channel-level TSI arrays
│   ├── tsi_adapted_channels.npz
│   ├── asi_frozen_per_scan.csv          # Per-scan, per-stage, per-head ASI
│   ├── asi_adapted_per_scan.csv
│   ├── asi_window_stats.npz             # Per-window ASI for distribution plots
│   ├── dad_per_head.csv                 # Per-stage, per-head DAD values
│   ├── dad_permutation_null.npz         # Permutation null distribution
│   └── activation_maps/                 # Saved activation maps for figure generation
│       ├── scan_{id}_hidden_states.npz  # Upsampled stage outputs (1 sample scan)
│       ├── scan_{id}_brain_mask.npy
│       └── scan_{id}_gt_mask.npy
│
├── tables/
│   ├── tsi_asi_table.csv
│   ├── tsi_asi_table.tex
│   ├── dad_table.csv
│   └── dad_table.tex
│
└── figures/
    ├── tsi_frozen_brainmasked.pdf
    ├── tsi_adapted_brainmasked.pdf
    ├── asi_frozen.pdf
    ├── asi_adapted.pdf
    ├── dad_stages.pdf
    └── summary_combined.pdf
```

The `run_figures_only.py` script reads from `raw/` and `tables/` to regenerate all figures without GPU access. This supports iterative figure refinement on a laptop.

---

## 10. CLI interface

### Full pipeline (requires GPU)

```bash
python -m experiments.uncertainty_segmentation.explainability.run_analysis \
    --config experiments/uncertainty_segmentation/explainability/config.yaml \
    --run-dir results/uncertainty_segmentation/r8_M10_s42 \
    --output results/uncertainty_segmentation/r8_M10_s42/explainability/ \
    --device cuda:0
```

Execution order:
1. Load frozen model → compute TSI (20 scans) + ASI (20 scans) → save to `raw/`.
2. Load adapted model → compute TSI (20 scans) + ASI (20 scans) → save to `raw/`.
3. Load both models → compute DAD (20 GLI + 20 MEN scans) → save to `raw/`.
4. Generate tables from `raw/` → save to `tables/`.
5. Generate figures from `raw/` + `tables/` → save to `figures/`.

### Figures only (no GPU needed)

```bash
python -m experiments.uncertainty_segmentation.explainability.run_figures_only \
    --data-dir results/uncertainty_segmentation/r8_M10_s42/explainability/raw/ \
    --output results/uncertainty_segmentation/r8_M10_s42/explainability/figures/ \
    --format pdf --dpi 300
```

---

## 11. Verification checklist

**Architecture:**
- [ ] Print all `WindowAttention` module paths and confirm stage assignment.
- [ ] Verify attention matrix shape: `[n_windows * batch, heads, N, N]` where N = window_size³.
- [ ] Confirm that patched forward produces identical output to original (compare Dice on 1 scan).

**TSI correctness:**
- [ ] Brain mask covers ~40–60% of voxels (not 99% — that would mean the threshold is wrong).
- [ ] TSI of a constant-activation channel with brain mask = 1.0 regardless of tumor location.
- [ ] Stage 0 TSI should remain near 1.0 after brain mask correction.
- [ ] Compare uncorrected vs corrected TSI for stage 1: expect a drop.

**ASI correctness:**
- [ ] ASI of a uniform attention matrix (all entries equal) = 1.0.
- [ ] ASI of a block-diagonal attention (tumor attends only to tumor) → ASI ≫ 1.
- [ ] Only boundary windows are included (verify: 0 < tumor_fraction < 1 for all selected windows).

**DAD correctness:**
- [ ] DAD of two identical attention distributions = 0.0.
- [ ] Permutation null distribution is centered near the observed DAD (sanity: if GLI ≡ MEN, DAD should be n.s.).

**Memory:**
- [ ] Stage 0 attention is NOT captured (memory guard).
- [ ] Peak GPU memory during analysis < 20GB (fits A100 40GB with margin).
- [ ] All tensors moved to CPU after extraction, `torch.cuda.empty_cache()` between scans.

---

## 12. Expected outcomes and interpretation guide

| Stage | Expected TSI (corrected) | Expected ASI | Expected DAD | Interpretation |
|-------|------------------------|-------------|-------------|----------------|
| 0 | ~1.0 (n.s.) | — (not computed) | — | General anatomy (edges, gradients). Domain-invariant. |
| 1 | ~1.1–1.3 (reduced from 1.71 after brain mask) | ~1.0–1.2 | Low | Tissue boundaries. Some tumor response but mostly anatomical. No LoRA needed. |
| 2 | ~1.5–2.0 (may remain high) | ~1.3–1.8 | Moderate | Highest spatial tumor selectivity. Attention begins to route tumor-specifically. LoRA candidate. |
| 3 | ~1.3–1.6 | ~1.5–2.0 (higher ASI than TSI) | High | Abstract context. ASI may exceed TSI because attention is tumor-selective even though output activations are diluted by residual. Strong LoRA candidate. |
| 4 | ~1.0–1.3 | ~1.0–1.5 | Moderate–High | Bottleneck. Fewer tokens, less spatial structure. Include for cascade safety. |

The key prediction: **ASI should show a different hierarchy than TSI.** TSI peaks at stage 2 (spatial selectivity of output activations). ASI may peak at stage 3 (attention mechanism is most tumor-discriminative at the semantic level). This would resolve the apparent contradiction between TSI results and the LoRA targeting rationale.

If both TSI and ASI peak at stage 2, then the justification for LoRA on stages 2-3-4 is: "Stage 2 shows the highest feature selectivity (TSI) and attention selectivity (ASI), while stages 3-4 are included to prevent cascade distribution mismatch and because DAD shows the attention patterns at these stages diverge most between glioma and meningioma domains."

---

## 13. References

1. Bau, D. et al. (2017). Network Dissection: Quantifying Interpretability of Deep Visual Representations. CVPR. — Methodological ancestor of TSI.
2. Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV. — Why we chose TSI/ASI over gradient-based methods.
3. Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. ACL. — Attention rollout and analysis methods for transformers.
4. Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV. — Architecture reference.
5. Hu, E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. — LoRA targeting QKV projections.
6. Cox, J. et al. (2024). BrainSegFounder: Towards Brain Segmentation Foundation Models. — The model under analysis.
7. Hatamizadeh, A. et al. (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. MICCAI BrainLes. — SwinUNETR architecture.
