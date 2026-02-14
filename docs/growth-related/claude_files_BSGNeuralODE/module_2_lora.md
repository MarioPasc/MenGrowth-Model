# Module 2: LoRA Encoder Adaptation (Phase 1)

## Overview
Adapt the BrainSegFounder encoder from glioma to meningioma domain via LoRA on Stages 3–4, using segmentation as proxy task with optional auxiliary semantic heads.

## Input
- BSF checkpoint: `finetuned_model_fold_0.pt` (62M params, BSF-Tiny)
- BraTS-MEN data with `data_splits.json` from Module 0
- Semantic features cache from Module 0

## Input Contract
```python
# BSF checkpoint
checkpoint_path: str  # path to finetuned_model_fold_0.pt

# Data
x: torch.Tensor       # shape [B, 4, 128, 128, 128], dtype float32
y: torch.Tensor       # shape [B, 128, 128, 128], dtype int64, values {0,1,2,3}
splits: dict           # from data_splits.json
```

## Output
- `phase1_encoder_merged.pt` — Merged encoder (LoRA absorbed into base weights, decoder/heads discarded)
- `phase1_training_log.csv` — Per-epoch Dice, CE, semantic losses
- `phase1_best_dice.json` — Best validation Dice per sub-region

## Output Contract
```python
# Merged encoder checkpoint (decoder and aux heads discarded)
state_dict: dict  # keys: swinViT.*, encoder1-10.* (no decoder*, out.*, aux_*)

# Dice metrics
dice: dict = {
    "TC": float,  # Tumor Core
    "WT": float,  # Whole Tumor
    "ET": float,  # Enhancing Tumor
}
```

## Reuse Directives

| Existing File | What to Import | Path |
|---------------|----------------|------|
| `swin_loader.py` | `load_swin_encoder()` | `src/growth/models/encoder/swin_loader.py` |
| `model_factory.py` | LoRA injection patterns | `experiments/lora_ablation/model_factory.py` |
| `train_condition.py` | Training loop patterns | `experiments/lora_ablation/train_condition.py` |
| `transforms.py` | Transform pipelines | `src/growth/data/transforms.py` |
| `semantic_features.py` | `compute_shape_array()` | `src/growth/data/semantic_features.py` |

**Extensive reuse from `experiments/lora_ablation/` (~9.4K lines).** Adapt existing code rather than rewriting.

**Note:** Phase 1 training is fully implemented in `experiments/lora_ablation/train_condition.py` as a custom training loop (not a Lightning LitModule). `LoRASwinViT` and `AuxiliarySemanticHeads` are already implemented at their canonical locations (see directory structure below).

## Code Requirements

1. **`LoRASwinViT`** — Already implemented at `src/growth/models/encoder/lora_adapter.py`. Wrapper that injects LoRA adapters into specified SwinViT stages.
   ```python
   class LoRASwinViT(nn.Module):
       def __init__(self, base_model: SwinUNETR, lora_config: dict):
           """Inject LoRA into stages 3-4 Q/K/V projections."""
       def merge_lora(self) -> None:
           """Absorb LoRA weights: W_merged = W + (α/r) * B @ A"""
       def get_trainable_params(self) -> List[nn.Parameter]:
           """Returns only LoRA parameters."""
   ```

2. **`LoRALitModule`** (PyTorch Lightning) — **Note:** The LoRA ablation experiment uses a custom training loop in `experiments/lora_ablation/train_condition.py` instead of a LitModule. A LitModule may be created for the main pipeline if desired:
   - Separate parameter groups: LoRA (lr=1e-4), decoder (lr=5e-4), aux heads (lr=1e-3)
   - `DiceCELoss` from MONAI with label conversion (integer → overlapping TC/WT/ET)
   - Auxiliary semantic heads with warmup scheduling (start epoch 5, ramp 10 epochs)
   - Validation: per-region Dice scores

3. **`SemanticHeads`** — Already implemented as `AuxiliarySemanticHeads` at `src/growth/models/segmentation/semantic_heads.py`. Auxiliary regression heads from GAP-pooled encoder10 → volume/location/shape.
   ```python
   class SemanticHeads(nn.Module):
       def __init__(self, in_dim: int = 768):
           self.vol_head = nn.Linear(768, 4)   # 4 log-volumes
           self.loc_head = nn.Linear(768, 3)   # 3 centroid coords
           self.shape_head = nn.Linear(768, 3) # 3 shape features (sphericity, surface_area_log, solidity)
   ```

## Checkpoint Key Mapping

BSF checkpoint keys follow this structure:
```python
# LoRA TARGETS (inject LoRA here):
"swinViT.layers3.0.blocks.0.attn.qkv.weight"  # Stage 3, Block 0, QKV
"swinViT.layers3.0.blocks.1.attn.qkv.weight"  # Stage 3, Block 1, QKV
"swinViT.layers4.0.blocks.0.attn.qkv.weight"  # Stage 4, Block 0, QKV
"swinViT.layers4.0.blocks.1.attn.qkv.weight"  # Stage 4, Block 1, QKV

# FREEZE (do NOT adapt):
"swinViT.layers1.0.blocks.0.attn.qkv.weight"  # Stage 1 — frozen
"swinViT.layers2.0.blocks.0.attn.qkv.weight"  # Stage 2 — frozen
"encoder10.layer.0.conv1.conv.weight"           # Bottleneck — frozen
```

Use `growth.utils.checkpoint.extract_encoder_weights()` for key filtering.

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 100 | Sufficient for LoRA convergence from strong init |
| LR (LoRA) | 1e-4 | Standard for LoRA |
| LR (decoder) | 5e-4 | Larger for pretrained decoder head |
| LR (aux heads) | 1e-3 | Small heads, fast convergence |
| Optimizer | AdamW | Decoupled weight decay |
| Weight decay | 0.01 | Standard regularization |
| Batch size | 2/GPU | Memory-constrained by 128³ input |
| Scheduler | Cosine decay, 10-epoch warmup | Smooth convergence |
| Gradient clipping | max_norm=1.0 | Stability |
| Precision | bf16-mixed | Memory efficiency |
| Aux heads | Enabled (λ_aux=0.1, warmup epoch 5 over 10 epochs) | See D4 |

## Configuration Snippet
```yaml
# configs/phase1_lora.yaml
lora:
  rank: 8
  alpha: 16
  target_stages: [3, 4]
  target_modules: [qkv]
  lora_dropout: 0.1
  mode: lora  # or dora for ablation A2

training:
  epochs: 100
  lr_lora: 1e-4
  lr_decoder: 5e-4
  lr_aux: 1e-3
  optimizer: adamw
  weight_decay: 0.01
  batch_size: 2
  scheduler: cosine
  warmup_epochs: 10
  grad_clip: 1.0
  precision: bf16-mixed

aux_heads:
  enabled: true
  lambda_aux: 0.1
  warmup_start_epoch: 5
  warmup_duration: 10
```

## Post-Phase 1 Protocol

1. Merge LoRA weights: `W_merged = W_pretrained + (α/r) * B @ A`
2. Discard segmentation decoder (`decoder1-5.*`, `out.*`)
3. Discard auxiliary heads (`aux_*`)
4. Freeze all encoder parameters
5. Save merged checkpoint as `phase1_encoder_merged.pt`

## Smoke Test
```python
import torch
from monai.networks.nets import SwinUNETR

# Create model and verify LoRA injection
model = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, feature_size=48)
x = torch.randn(1, 4, 128, 128, 128)
with torch.no_grad():
    y_base = model(x)
# Inject LoRA, verify output unchanged (B=0 initialization)
# lora_model = LoRASwinViT(model, config)
# y_lora = lora_model(x)
# assert torch.allclose(y_base, y_lora, atol=1e-5)
```

## Verification Tests

```
TEST_2.1: LoRA injection [BLOCKING]
  - Load BSF checkpoint, inject LoRA with r=8 into stages 3-4
  - Assert trainable parameters ≈ 197K (LoRA only)
  - Assert stages 0-2 have zero gradients
  - Forward pass produces output shape [B, 3, 128, 128, 128]
  Recovery: Verify checkpoint key names match expected pattern

TEST_2.2: LoRA initialization preserves output [BLOCKING]
  - Forward pass through base model: y_base = model(x)
  - Forward pass through LoRA model (B=0): y_lora = model_lora(x)
  - Assert ||y_base - y_lora|| < 1e-5
  Recovery: Check that B matrix is initialized to zeros

TEST_2.3: Training step [BLOCKING]
  - Run 1 training step, assert loss is finite and > 0
  - Assert LoRA parameters have nonzero gradients
  - Assert frozen parameters have zero gradients
  Recovery: Check parameter groups and requires_grad settings

TEST_2.4: LoRA merge [BLOCKING]
  - Merge LoRA weights, assert no LoRA params remain
  - Forward pass through merged model ≈ forward pass through LoRA model
  - Assert ||y_merged - y_lora|| < 1e-5
  Recovery: Verify merge formula W + (alpha/r) * B @ A

TEST_2.5: Segmentation quality [BLOCKING]
  - After training, validation Dice (WT) > 0.80
  - Dice improvement over frozen baseline > 0.05
  Recovery steps (in order):
    1. Increase epochs to 150
    2. Reduce LR to 5e-5
    3. Add LoRA to Stage 2
    4. If still failing, report diagnostics and stop

TEST_2.6: Semantic head predictions [DIAGNOSTIC]
  - Aux vol R² on val set > 0.0 (above constant baseline)
  - Aux loc R² on val set > 0.0
  Note: DIAGNOSTIC — poor aux R² does not block; it only means
  the encoder features may be suboptimal for Phase 2
```
