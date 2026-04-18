# Critical Conventions (violating these causes silent failures)

## Channel Order
BrainSegFounder expects [FLAIR, T1ce, T1, T2] = `["t2f", "t1c", "t1n", "t2w"]`.
Wrong order causes Dice ~0.00 even on the training domain (GLI).
Defined in `MODALITY_KEYS` in `src/growth/data/transforms.py`.

## ROI Size
128×128×128 (matching BrainSegFounder fine-tuning). NOT 96³.
encoder10 output: `[B, 768, 4, 4, 4]` with 128³ input.

## Segmentation Convention (TC/WT/ET)
BrainSegFounder uses 3-channel sigmoid output (hierarchical overlapping). The
mapping from raw integer labels to TC/WT/ET is **domain-dependent** because
BraTS-MEN and BraTS-GLI assign different semantics to the same integers.

| Raw label  | BraTS-GLI            | BraTS-MEN                          |
|-----------:|----------------------|------------------------------------|
| 0          | Background           | Background                         |
| 1          | NETC (necrotic core) | NETC (rare in meningioma)          |
| 2          | SNFH (edema)         | SNFH (peritumoral edema)           |
| 3          | ET (enhancing tumor) | ET (enhancing meningioma — main mass) |
| 4          | RC (resection cavity)| —                                  |

3-channel target conversion (must match what BSF was pretrained to output):

- **GLI**: `TC = (1|3|4)`, `WT = (seg>0)`, `ET = (3)`
- **MEN**: `TC = empty (zeros)`, `WT = (1|2|3)`, `ET = (1|3)`

For MEN, BSF effectively only models 2 tissues — SNFH (edema) and ET (enhancing tumor) —
so we translate BraTS-MEN → BSF native by **merging NETC (label 1) into ET**. NETC is
part of the solid meningioma mass; BSF was never trained to distinguish it from ET, so
folding it in gives the model a single coherent "meningioma mass" target. The TC sigmoid
channel has no analogue in this 2-label space and is held empty (target = zeros, model
learns to suppress it). Downstream growth uses ET volume = the merged mass.

Single source of truth: `_convert_single_domain` in `src/growth/losses/segmentation.py`.
**Never** redefine this conversion locally — import the helper.

### Stage 1 volume target (meningioma growth)
For meningioma growth modeling (Stage 1) the tracked endpoint is the
**enhancing-tumor (ET) volume = label 3**, not the whole-tumor (WT) volume.
SNFH/edema (label 2) is non-neoplastic and would inject noise into the
trajectory. `semantic/volume[:, 3]` is `log(V_ET + 1)`.

## Preprocessing Pipeline
- Training: `CropForegroundd` → `SpatialPadd(128³)` → `RandSpatialCropd(128³)`
- Validation: `CropForegroundd` → `SpatialPadd(128³)` → `ResizeWithPadOrCropd(128³)`
- Inference: `sliding_window_inference` with 128³ patches, 50% overlap, Gaussian blending
