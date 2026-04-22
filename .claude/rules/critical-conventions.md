# Critical Conventions (violating these causes silent failures)

## Channel Order
BrainSegFounder expects [FLAIR, T1ce, T1, T2] = `["t2f", "t1c", "t1n", "t2w"]`.
Wrong order causes Dice ~0.00 even on the training domain (GLI).
Defined in `MODALITY_KEYS` in `src/growth/data/transforms.py`.

## ROI Size
128×128×128 (matching BrainSegFounder fine-tuning). NOT 96³.
encoder10 output: `[B, 768, 4, 4, 4]` with 128³ input.

## Segmentation Convention
Three-channel sigmoid output. Raw BraTS-MEN integer labels are
1=NETC, 2=SNFH (edema), 3=ET; BraTS-GLI adds 4=RC.

| Raw label | BraTS-GLI            | BraTS-MEN                 |
|----------:|----------------------|---------------------------|
| 0         | Background           | Background                |
| 1         | NETC (necrotic core) | NETC (rare in meningioma) |
| 2         | SNFH (edema)         | SNFH (peritumoral edema)  |
| 3         | ET (enhancing tumor) | ET (enhancing meningioma) |
| 4         | RC (resection cavity)| —                         |

### Training target: BraTS-hierarchical (BSF-aligned)
The model is trained against the standard BraTS hierarchical regions
that the finetuned BrainSegFounder decoder already emits on
meningioma. This keeps the pretrained output head aligned and
minimises what LoRA has to learn.

- **MEN**: `TC = (==1) | (==3)`, `WT = (seg > 0)`, `ET = (==3)`
- **GLI**: `TC = (==1) | (==3) | (==4)`, `WT = (seg > 0)`, `ET = (==3)`

Hierarchy: `ET ⊂ TC ⊂ WT`. Single source of truth:
`_convert_single_domain` in `src/growth/losses/segmentation.py`.
**Never** redefine this conversion locally — import the helper.

### Downstream label convention: disjoint clinical regions
For volumes, plots, and statistics we derive **disjoint** per-voxel
regions from the three hierarchical channels. These satisfy
`TC_necrotic ⊂ WT_meningioma`, `WT_meningioma ⊥ ED_edema`,
`TC_necrotic ⊥ ED_edema`:

- `WT_meningioma (labels 1|3) = ch0 ≥ τ`          (= BraTS-TC)
- `ET_enhancing  (label == 3) = ch2 ≥ τ`          (= BraTS-ET)
- `TC_necrotic   (label == 1) = WT_men ∧ ¬ET_enh`
- `ED_edema      (label == 2) = (ch1 ≥ τ) ∧ ¬WT_men`

Helper: `growth.inference.postprocess.derive_disjoint_regions`.
Do not recompute this inline — import the helper.

### Volume target for MenGrowth / BraTS-MEN
Meningioma volume = `WT_meningioma` (BSF ch0 thresholded + CC-cleaned).
ED is tracked separately; BraTS-WT (ch1, includes edema) is NOT the
volume label. Prior H5 semantic features may still store
`semantic/volume[:, 3]` as `log(V_ET + 1)` — Stage 1 code uses this
directly and is isolated from the LoRA-uncertainty pipeline.

### Postprocessing
Binary masks pass through
`growth.inference.postprocess.remove_small_components(mask, min_voxels=64)`
— drops 3D connected components below a voxel-size threshold (26-
connectivity). Size-threshold, **not** keep-largest: edema can be
legitimately bilateral. Configurable via `inference.min_component_voxels`
in `experiments/uncertainty_segmentation/config.yaml`.

## Preprocessing Pipeline
- Training: `CropForegroundd` → `SpatialPadd(128³)` → `RandSpatialCropd(128³)`
- Validation: `CropForegroundd` → `SpatialPadd(128³)` → `ResizeWithPadOrCropd(128³)`
- Inference: `sliding_window_inference` with 128³ patches, 50% overlap, Gaussian blending
