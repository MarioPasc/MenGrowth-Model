# Critical Conventions (violating these causes silent failures)

## Channel Order
BrainSegFounder expects [FLAIR, T1ce, T1, T2] = `["t2f", "t1c", "t1n", "t2w"]`.
Wrong order causes Dice ~0.00 even on the training domain (GLI).
Defined in `MODALITY_KEYS` in `src/growth/data/transforms.py`.

## ROI Size
128×128×128 (matching BrainSegFounder fine-tuning). NOT 96³.
encoder10 output: `[B, 768, 4, 4, 4]` with 128³ input.

## Segmentation Convention (TC/WT/ET)
BrainSegFounder uses 3-channel sigmoid output (hierarchical overlapping):
- Ch0: TC (Tumor Core) = `(label==1) | (label==3)`
- Ch1: WT (Whole Tumor) = `(label==1) | (label==2) | (label==3)`
- Ch2: ET (Enhancing Tumor) = `(label==3)`
These are NOT individual labels. Conversion in `segmentation.py._convert_target()`.

## Preprocessing Pipeline
- Training: `CropForegroundd` → `SpatialPadd(128³)` → `RandSpatialCropd(128³)`
- Validation: `CropForegroundd` → `SpatialPadd(128³)` → `ResizeWithPadOrCropd(128³)`
- Inference: `sliding_window_inference` with 128³ patches, 50% overlap, Gaussian blending
