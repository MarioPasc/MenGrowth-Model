# BrainSegFounder: Insights from the Code

Comprehensive analysis of the BrainSegFounder vendored code (`src/external/BrainSegFounder/`) and its paper (`2406.10395v3.pdf`), documenting all implementation details relevant to our pipeline.

## Architecture

SwinUNETR with the following constructor parameters (from `launch.sh` and `main_FinetuningSwinUNETR_4Channels.py`):

| Parameter | Value | Source |
|-----------|-------|--------|
| `feature_size` | 48 | launch.sh, main.py line 66 |
| `in_channels` | 4 | launch.sh (`--in_channels=4`) |
| `out_channels` | 3 | TC, WT, ET (hierarchical overlapping) |
| `depths` | (2, 2, 2, 2) | launch.sh |
| `num_heads` | (3, 6, 12, 24) | launch.sh |
| `norm_name` | "instance" | InstanceNorm in CNN blocks, LayerNorm in transformer |

### Hidden States (128^3 input, MONAI 1.5+)

| Stage | Output Shape | Channels | Spatial |
|-------|-------------|----------|---------|
| patch_embed | [B, 48, 64, 64, 64] | 48 | input/2 |
| layers1 | [B, 96, 32, 32, 32] | 96 | input/4 |
| layers2 | [B, 192, 16, 16, 16] | 192 | input/8 |
| layers3 | [B, 384, 8, 8, 8] | 384 | input/16 |
| layers4 | [B, 768, 4, 4, 4] | 768 | input/32 |
| encoder10 | [B, 768, 4, 4, 4] | 768 | input/32 |

**Note**: encoder10 preserves channels (768 -> 768), it does **not** double them. The channel doubling happens within the swinViT stages.

## 3-Stage Training Pipeline

1. **Self-Supervised Learning (SSL)**: Pretraining on 41K+ subjects with masked image modeling. Uses 96^3 ROI size.
2. **BraTS Pretraining**: Supervised training on BraTS 2021 with 128^3 ROI size.
3. **Fine-tuning**: Per-fold fine-tuning on BraTS 2021 with 128^3 ROI size.

**Key insight**: The paper mentions 96^3 for SSL, but the actual fine-tuning code uses 128^3. Our pipeline correctly uses 128^3.

## Data Preprocessing

### Training Pipeline (from `data_utils.py` lines 118-136)

```
LoadImaged → ConvertToMultiChannelBasedOnBratsClassesd →
CropForegroundd(k_divisible=[128,128,128]) →
RandSpatialCropd(roi_size=[128,128,128]) →
RandFlipd(prob=0.5, axis=0) → RandFlipd(prob=0.5, axis=1) → RandFlipd(prob=0.5, axis=2) →
NormalizeIntensityd(nonzero=True, channel_wise=True) →
RandScaleIntensityd(factors=0.1, prob=1.0) →
RandShiftIntensityd(offsets=0.1, prob=1.0) →
ToTensord
```

**Key parameters:**
- `CropForegroundd` uses `k_divisible=[roi_x, roi_y, roi_z]` — ensures cropped region is at least roi_size in each dimension, reducing zero-padding artifacts
- `RandScaleIntensityd` and `RandShiftIntensityd` use `prob=1.0` (always applied), NOT 0.5
- No `SpatialPadd` between CropForeground and RandSpatialCrop (k_divisible handles this)
- No `RandRotate90d` — only random flips

### Validation Pipeline (from `data_utils.py` lines 137-144)

```
LoadImaged → ConvertToMultiChannelBasedOnBratsClassesd →
NormalizeIntensityd(nonzero=True, channel_wise=True) →
ToTensord
```

No spatial cropping at all for validation — full volumes are passed to sliding window inference.

## Label Conventions

### BraTS 2021 Labels
- 0: Background
- 1: NCR (Necrotic tumor core)
- 2: ED (Peritumoral edematous/invaded tissue)
- 4: ET (GD-enhancing tumor) — **note: label 4, not 3**

### Our Data (BraTS-MEN)
- Uses labels 0, 1, 2, 3 (label 3 = ET instead of 4)
- GLI remap: label 4 → 3 is handled in our data pipeline

### 3-Channel Sigmoid Output (TC/WT/ET)
- Ch0: TC = (label==1) | (label==4) [BraTS 2021] or (label==1) | (label==3) [our data]
- Ch1: WT = (label==1) | (label==2) | (label==4) [BraTS 2021]
- Ch2: ET = (label==4) [BraTS 2021] or (label==3) [our data]

Conversion done by `ConvertToMultiChannelBasedOnBratsClassesd` in BrainSegFounder, and `_convert_target()` in our `segmentation.py`.

## Checkpoint Structure

### Fine-tuned Checkpoint
- Key: `checkpoint['state_dict']` containing all model weights
- Clean keys (no `module.` DDP prefix) in fine-tuned checkpoints
- Optional metadata: `checkpoint.get('epoch')`, `checkpoint.get('best_acc')`
- Direct loading: `model.load_state_dict(model_dict)` (test.py line 83)

### SSL Pretrained Checkpoint
- Has `module.swinViT.` prefix — needs stripping when loading into SwinUNETR
- Handled in `main_FinetuningSwinUNETR_4Channels.py` lines 155-194

## Inference Pipeline

### Sliding Window Parameters
- Window size: 128^3 (matching training ROI)
- Overlap: 0.5 (50%) for training-time validation, 0.6 (60%) for test.py
- Batch size: 4 (sw_batch_size) for validation, 1 for test
- Post-processing: `torch.sigmoid()` + threshold at 0.5
- Output mapping: TC→1, WT→2, ET→4 (hierarchical to individual labels)

### Post-processing (from `test.py` lines 102-109)
```python
prob = torch.sigmoid(model_inferer(image))
seg = (prob[0] > 0.5).astype(np.int8)
seg_out = np.zeros(...)
seg_out[seg[1] == 1] = 2  # WT (channel 1)
seg_out[seg[0] == 1] = 1  # TC (channel 0) — overwrites WT where TC exists
seg_out[seg[2] == 1] = 4  # ET (channel 2) — overwrites TC where ET exists
```

## Discrepancies Between Paper and Code

| Aspect | Paper | Code | Resolution |
|--------|-------|------|------------|
| ROI size | 96^3 (SSL) | 128^3 (fine-tuning) | Use 128^3 — matches fine-tuned checkpoint |
| img_size param | Mentioned | Passed in older MONAI; not accepted in newer | Skip — not needed in current MONAI |
| k_divisible | Not mentioned | Used in CropForegroundd | Added to our pipeline |
| Intensity aug prob | Not specified | 1.0 (always applied) | Fixed in our pipeline |
| RandRotate90d | Not used | Not used | We added it for domain adaptation (documented) |

## Channel Order

**CRITICAL**: BrainSegFounder expects `[FLAIR, T1ce, T1, T2]` = `["t2f", "t1c", "t1n", "t2w"]`.

This was empirically verified by testing all 24 channel permutations — only this order produces meaningful Dice scores (~0.77 WT on GLI). Wrong order gives Dice ~0.00 even on the training domain.

Defined in `MODALITY_KEYS` in `src/growth/data/transforms.py`.
