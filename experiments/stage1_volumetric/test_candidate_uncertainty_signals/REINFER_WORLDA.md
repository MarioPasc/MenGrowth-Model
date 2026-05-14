# World-A Re-inference — per-member voxel masks for the variance-map figures

**Date:** 2026-05-14
**Why:** `MenGrowth.h5` is a Frankenstein — its `segs` / `per_member_volumes` /
`logvol_*` come from the correct original `r32_M20_s42` inference ("World A"),
but its entropy/MI fields and the `per_member_segmentations_deep_ensemble/per_scan/`
voxel masks come from a **broken re-inference** ("World B"). See
`memory/project_h5_uncertainty_two_inferences.md`. World B over-segments because
`reinfer_h5_uncertainty.py` fed the ensemble **un-normalised** images
(`f["images"]` read raw) instead of running them through
`get_h5_val_transforms` (z-score intensity normalisation) as the original
`engine/volume_extraction.py` does.

This job re-runs the ensemble **correctly** and saves per-member voxel masks
for all 179 scans — the inputs the segmentation-variance-map thesis figures need.

## The fix

`reinfer_h5_uncertainty.py` now builds a `BraTSDatasetH5` with
`get_h5_val_transforms(roi_size=inference_roi_size)` and reads each scan through
it (`_build_val_dataset`), exactly as `volume_extraction.py` did. The only
substantive transform is `NormalizeIntensityd(nonzero=True, channel_wise=True)`;
the spatial `ResizeWithPadOrCrop` is a no-op at 192³.

## Submit on Picasso

```bash
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model

bash experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/reinfer_launcher.sh \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/configs/r32_inference.yaml \
    /mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/LoRA_finetuned_BSF/r32_M20_s42 \
    /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/h5_growth_datasets/MenGrowth.h5 \
    /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/per_member_segmentations_r32_worldA \
    8
```

Args: `<LORA_CONFIG> <LORA_RUN_DIR> <MENGROWTH_H5> <OUTPUT_DIR> [N_SHARDS]`.
Add `--dry-run` to print the `sbatch` commands without submitting.

- **Array job** (`reinfer_worldA_array`): 8 GPU shards, `--constraint=dgx`,
  ~12 min/shard on an A100, `--save-member-masks --save-ensemble-mask`.
- **Merge job** (`reinfer_worldA_merge`): CPU, `afterany` on the array.
  Concatenates `shards/shard_*.csv` → `recomputed_uncertainty.csv` and
  **verifies** the recomputed `logvol_mean` against the H5's `logvol_mean`.

## Verify before trusting the output

The merge log must print **`VERIFY PASSED`** — that means the re-inference
reproduces World A (≥95 % of scans within 0.05 log-volume of the H5). If it
prints `VERIFY FAILED`, the preprocessing fix is still incomplete and the
per-member masks must not be used.

## Prerequisites on Picasso

- LoRA adapters at `<LORA_RUN_DIR>/adapters/member_{0..19}/{adapter,decoder.pt}`.
- BSF backbone at the `paths.checkpoint_dir` in `configs/r32_inference.yaml`
  (`.../fscratch/checkpoints/BrainSegFounder_finetuned_BraTS/finetuned_model_fold_0.pt`)
  — adjust the config if it has moved.
- `MenGrowth.h5` synced to the path passed as `<MENGROWTH_H5>`.

## Output

```
<OUTPUT_DIR>/
  per_scan/<scan_id>/member_{0..19}_mask.nii.gz   # World-A per-member voxel masks
  per_scan/<scan_id>/ensemble_mask.nii.gz
  per_scan/<scan_id>/metrics.json
  shards/shard_*.csv
  recomputed_uncertainty.csv
```

The figure scripts read `per_scan/<scan_id>/member_*_mask.nii.gz` to compute
the voxel-wise variance map `p(1-p)`. Point their `PER_MEMBER_DIR` at this
`<OUTPUT_DIR>/per_scan` once the job completes and `VERIFY PASSED`.

## Optional: patch the H5

`PATCH_H5=true bash reinfer_launcher.sh ...` additionally overwrites the H5
`/uncertainty/` entropy-MI scalars from this (now World-A-consistent)
re-inference, with a timestamped backup. Default is off — the verification
runs regardless; patching the experiment H5 is a separate, deliberate choice.
