# Why the Benchmark Predictions Are Comparable Despite Living in Different Frames

## Context

The BraTS-MEN benchmark compares two families of segmentation outputs on the same 150-scan test split:

1. **External challenge containers** (`BraTS25_1`, `BraTS25_2`, …): receive the
   raw BraTS-MEN canonical NIfTIs (240 × 240 × 155, 1×1×1 mm, LPS-oriented),
   and emit predictions on the same 240 × 240 × 155 grid. This is the
   subject-space frame in which the BraTS challenge defines its targets.
2. **Our BSF + LoRA ensemble**: reads the project H5 file, whose volumes have
   already been pushed through the canonical pipeline
   `Orient(RAS) → Resample(1 mm) → CropForeground → SpatialPad(192³) → CenterCrop(192³)`.
   Predictions are therefore written in the 192³ ROI frame.

The frames differ in (a) world axes (LPS vs. RAS) and (b) FOV (240 × 240 × 155
vs. 192³ after a content-aware foreground crop). The H5 transform is
deterministic per case but its crop offsets are not stored alongside the H5,
so we cannot invert the H5 → subject mapping with the data we have on disk.

## Why per-case metrics are still directly comparable

The key invariant is that the H5 → 192³ pipeline is a **lossless rearrangement
of the tumour-bearing voxels**. The two component transforms — orientation
relabelling and `CropForeground + SpatialPad + CenterCrop(192³)` — together do
not modify any voxel value, only its spatial position. We confirmed this
empirically on `BraTS-MEN-00040-000`:

| Quantity                         | Subject 240×240×155 | H5 192³ |
|----------------------------------|--------------------:|--------:|
| TC voxel count (`==1` ∪ `==3`)   |              45 185 |  45 185 |
| WT voxel count (`>0`)            |              45 187 |  45 187 |
| Voxel size (mm)                  |             1×1×1   |  1×1×1  |

For the three reported metrics this means:

- **Dice** is shape-invariant. It depends only on cardinalities of
  `|GT|`, `|Pred|`, and `|GT ∩ Pred|`, none of which changes under a
  bijective rearrangement of voxels. Computing Dice in either frame returns
  the same number.
- **HD95 (mm)** is computed in physical units and uses surface voxels of GT
  and Pred. Because (i) the rearrangement is rigid (no resampling distorts
  distances) and (ii) the voxel spacing is identical (1×1×1 mm in both
  frames), the 95-th percentile bidirectional surface distance is the same
  whether we evaluate in the subject grid or in the H5 grid.
- **Lesion recall (any-overlap)** is defined per connected component of GT.
  The number and adjacency of components are preserved under
  `CropForeground + Pad + CenterCrop` (no foreground voxel is dropped, no
  component is split or merged), so each component's overlap with the
  prediction is the same in either frame.

In short: every per-case metric is invariant under the H5 transform, so the
**numbers reported for "Ours" in the H5 192³ frame are directly comparable to
the numbers reported for the external models in subject space**. Paired
Wilcoxon tests and Cohen's d in `aggregate.py` use the same `case_id` for both
sides of every pair, so the comparison is patient-aligned, not just
patient-marginal.

## What this assumption requires of the launcher

The argument above depends on the H5 192³ ROI being **large enough to contain
the full tumour for every test case**, so that no foreground voxel is lost
during cropping. This is the explicit design choice documented in
`src/growth/data/transforms.py:41-42` ("192³ for feature extraction —
guarantees 100 % tumor containment; 128³ center crop only captures
38.8 % MEN / 30.0 % GLI tumors fully"). If a future test set violates that
assumption, the invariance breaks for the affected cases, and Dice/HD95 from
"Ours" will be optimistic by a small amount proportional to the voxels lost.
The benchmark code does not currently audit this; if the H5 pipeline is ever
changed to use a tighter ROI, that audit must be added.

## Why we do not remap to a common grid

The naive remap "centre-crop xy + centre-pad z" is wrong because the H5 uses
`CropForegroundd`, which is content-aware. We verified on
`BraTS-MEN-00040-000` that a forward centre-crop of the subject GT yields
zero overlap with the H5 GT despite identical voxel counts: the centroids
of TC are at `(112.6, 134.6, 98.6)` for the centre-crop attempt versus
`(77.5, 67.1, 103.8)` in the actual H5, off by 35-67 voxels in xy. Inverting
the H5 transform per case would require persisting the foreground bbox
alongside each H5 entry, which the current schema does not do.

The qualitative figure handles this differently: for each selected
best/median/worst case, the subject-space external NIfTI is pushed forward
through the same MONAI pipeline used to build the H5 (`Orientationd("RAS") →
CropForegroundd(source=t1n) → SpatialPadd(192³) → ResizeWithPadOrCropd(192³)`),
so all columns are visualised on the same 192³ canvas. This is feasible only
because the qualitative figure operates on three cases per metric, not the
full 150-scan grid.

## Take-away

Per-case Dice, HD95 and lesion recall are coordinate-invariant under the
H5 → 192³ transform whenever the ROI fully contains the foreground, which
holds by construction for this dataset. The benchmark therefore reports
numbers that can be directly compared across rows of the boxplot, used in
paired statistical tests, and quoted side-by-side in tables, with the single
caveat that all per-frame computations assume isotropic 1 mm voxels.
