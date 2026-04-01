# SPEC: LoRA-Ensemble Uncertainty-Aware Segmentation

**Version:** 1.0  
**Date:** 2026-04-01  
**Target Agent:** Claude Code Opus 4.6  
**Target Location:** `experiments/uncertainty_segmentation/`  
**Repository:** MenGrowth-Model

---

## 0. Objective

Implement a **LoRA-Ensemble** module for uncertainty-aware meningioma segmentation. The module trains $M$ independent LoRA adapters on a shared frozen BrainSegFounder backbone, each with a different random seed, then aggregates their predictions to produce:

1. A mean segmentation map (voxel-wise class probabilities).
2. Per-voxel epistemic uncertainty (predictive entropy or variance).
3. Per-ensemble-member volume estimates, yielding a **distribution over tumor volumes** with mean and standard deviation.
4. Uncertainty-propagated volume estimates to feed into the downstream GP/LME growth models.

### Scientific rationale

The LoRA-Ensemble (Mühlematter et al., 2024, arXiv:2405.14438) exploits the observation that LoRA adapters trained from different random initializations converge to functionally diverse solutions because the low-rank constraint restricts each adapter to a different subspace of the weight perturbation manifold. This provides calibrated epistemic uncertainty at a fraction of the cost of a full deep ensemble (parameter overhead: $M \times 197\text{K}$ vs. $M \times 62\text{M}$).

### Why this matters

The current pipeline treats segmentation output as a deterministic oracle. A voxel-count-derived volume $V = \sum_v \mathbb{1}[\hat{s}_v > 0] \cdot \delta^3$ is a point estimate with zero uncertainty from the segmentation model. This specification adds a principled uncertainty channel that propagates through the growth prediction pipeline via a heteroscedastic GP/LME likelihood.

---

## 1. Scope

### 1.1 In scope

- New experiment submodule at `experiments/uncertainty_segmentation/`.
- Training $M$ LoRA adapters on BraTS-MEN 2024 HDF5 data (labelled, cross-sectional, ~800 subjects).
- Inference (ensemble prediction) on the **MenGrowth** longitudinal cohort (unlabelled, ~54 patients).
- Dice evaluation on BraTS-MEN held-out test set (per-member and ensemble-averaged).
- Volume extraction with uncertainty (mean ± std across ensemble members).
- Calibration metrics: Expected Calibration Error (ECE), Brier score, reliability diagrams.
- SLURM array job infrastructure for Picasso (1 GPU per adapter).

### 1.2 Out of scope (for now)

- Downstream GP/LME integration (separate task; this module produces volume CSVs with uncertainty).
- MC Dropout comparison (future ablation).
- Test-Time Augmentation (future ablation).
- Evidential Deep Learning (rejected approach).
- Output directory structure fine-tuning (will iterate).

---

## 2. Existing infrastructure to reuse

### 2.1 Key files (read these to understand the codebase)

| Purpose | Location | Notes |
|---------|----------|-------|
| LoRA adapter wrapper | `src/growth/models/encoder/lora_adapter.py` | `LoRASwinViT` class, uses HuggingFace PEFT. Has `save_lora()`, `load_lora()`, `merge_lora()`. |
| SwinUNETR loader | `src/growth/models/encoder/swin_loader.py` | `load_swin_encoder()`, `load_full_swinunetr()`, `create_swinunetr()`. |
| Model factory | `experiments/stage3_latent/lora/engine/model_factory.py` | `create_ablation_model()` — creates LoRA or frozen models from config. |
| Training engine | `experiments/stage3_latent/lora/engine/train_condition.py` | Training loop with Dice+CE loss, early stopping, validation. |
| HDF5 data loader | `src/growth/data/bratsmendata.py` | `BraTSDatasetH5` — reads unified H5 v2.0 schema. `create_dataloaders()` helper. |
| Transforms | `src/growth/data/transforms.py` | `get_h5_train_transforms()`, `get_h5_val_transforms()`. |
| Sliding window inference | `src/growth/inference/sliding_window.py` | `sliding_window_segment()` — MONAI-based, 128³ patches, Gaussian weighting. |
| Segmentation loss | `src/growth/losses/` | `SegmentationLoss3Ch` (Dice + BCE). |
| Dice metric | `src/growth/losses/` | `DiceMetric3Ch`. |
| Volume extraction | `src/growth/data/semantic_features.py` | `compute_volumes()`, `compute_log_volumes()`. |
| Path management | `src/growth/utils/paths.py` | `OutputPathManager`, `ComponentPaths`. |
| Foundation config | `src/growth/config/foundation.yaml` | Base paths, data config. |
| Server config | `src/growth/config/server/foundation_icai.yaml` | Picasso-specific paths. |
| Existing LoRA SLURM | `slurm/lora/train_worker.sh` | Template for SLURM worker script. |
| Existing SLURM launcher | `slurm/lora/launch.sh` | Array job submission pattern. |
| Segmentation model (full) | `src/growth/models/segmentation/original_decoder.py` | `LoRAOriginalDecoderModel` — LoRA encoder + original SwinUNETR decoder. |

### 2.2 Key patterns to follow

- **Config-driven:** All paths and hyperparameters in a central `config.yaml`. Use `omegaconf` for loading/interpolation.
- **Structured logging:** Use `logging` module, never `print()`.
- **Type hints:** All function signatures annotated.
- **Dataclasses:** For structured results.
- **MONAI for medical imaging:** Transforms, inferers, metrics.
- **HuggingFace PEFT for LoRA:** `LoraConfig`, `get_peft_model`, `PeftModel.from_pretrained()`.

### 2.3 Critical data facts

- **Channel order (CRITICAL):** `[FLAIR, T1ce, T1, T2]` = `["t2f", "t1c", "t1n", "t2w"]`. Wrong order → Dice ≈ 0.00.
- **BraTS labels:** 0 (background), 1 (ET), 2 (NET/NCR), 3 (cystic). Whole tumor (WT) = labels {1, 2, 3}.
- **ROI sizes:** Training = `[128, 128, 128]`. Feature extraction / validation = `[192, 192, 192]`.
- **Spacing:** 1.0 mm isotropic after BraTS preprocessing. Voxel volume = 1.0 mm³.
- **HDF5 schema:** Images `[N, 4, 192, 192, 192]` float32. Segs `[N, 1, 192, 192, 192]` int8. Splits: `lora_train`, `lora_val`, `test`.
- **BrainSegFounder-Tiny:** 62M parameters. Bottleneck dim = 768. Swin Transformer encoder + UNet decoder.
- **LoRA targets:** QKV projections in Stages 3–4: `swinViT.layers{3,4}.0.blocks.{0,1}.attn.qkv`. 4 adapted layers total.
- **MenGrowth HDF5:** Same schema but with `longitudinal/` group (multi-timepoint per patient). **No segmentation labels** — this is the clinical deployment cohort.

---

## 3. Architecture

### 3.1 Training phase (on BraTS-MEN)

```
For m = 1, ..., M:
    seed_m = base_seed + m
    
    1. Load BrainSegFounder checkpoint (frozen)
    2. Inject LoRA adapters (rank r, alpha 2r, dropout 0.1) into Stages 3-4 QKV
       with random init seeded by seed_m
    3. Unfreeze decoder (or attach fresh decoder — match existing lora training pattern)
    4. Train on BraTS-MEN lora_train split with Dice + CE loss
    5. Early stopping on lora_val Dice
    6. Save adapter checkpoint to output_dir/adapters/member_{m}/
    7. Evaluate Dice on test split
```

Each member is an independent SLURM job (1 GPU). The SLURM array index maps to member ID.

### 3.2 Inference phase (on MenGrowth)

```
1. Load frozen BrainSegFounder backbone (once)
2. For each MenGrowth scan:
    a. Load volume from MenGrowth HDF5
    b. For m = 1, ..., M:
        i.  Load adapter m onto backbone (via PEFT)
        ii. Run sliding_window_segment(model, volume, roi=128³, overlap=0.5)
        iii. Softmax → class probabilities p^(m)_{v,c} ∈ [0,1]^C for every voxel v
        iv. Argmax → hard mask → V^(m) = sum(WT voxels) × 1.0 mm³
    c. Compute ensemble statistics:
        - Mean probability:  p̄_{v,c} = (1/M) Σ_m p^(m)_{v,c}
        - Predictive entropy: H[v] = -Σ_c p̄_{v,c} log(p̄_{v,c})
        - Mutual information: MI[v] = H[v] - (1/M) Σ_m H[p^(m)_v]  (epistemic)
        - Ensemble mask:     ŝ_v = argmax_c p̄_{v,c}
        - Volume stats:      V̄ = mean(V^(m)),  σ_V = std(V^(m))
        - Log-volume stats:  v̄ = mean(log(V^(m)+1)),  σ_v = std(log(V^(m)+1))
    d. Save per-scan results
3. Aggregate into per-patient volume trajectory CSV with uncertainty columns
```

**Inference can run on a single GPU** because only one adapter is loaded at a time. The backbone stays in GPU memory; only the LoRA matrices (~197K params) are swapped per member.

### 3.3 Mathematical formulation of volume uncertainty propagation

Each ensemble member $m$ produces a hard segmentation mask $\hat{\mathbf{s}}^{(m)}$, from which a volume is derived:

$$V^{(m)} = \sum_{v \in \Omega} \mathbb{1}\bigl[\hat{s}^{(m)}_v \in \{1,2,3\}\bigr] \cdot \delta^3$$

The ensemble volume distribution in log-space:

$$\bar{v} = \frac{1}{M}\sum_{m=1}^{M} \log(V^{(m)} + 1), \quad \sigma^2_v = \frac{1}{M-1}\sum_{m=1}^{M}\bigl(\log(V^{(m)} + 1) - \bar{v}\bigr)^2$$

This $\sigma^2_v$ becomes the **per-observation heteroscedastic noise** in the downstream GP:

$$y_k \sim \mathcal{N}\bigl(f(t_k),\; \sigma^2_{v,k} + \sigma^2_n\bigr)$$

where $\sigma^2_n$ is the GP's intrinsic noise variance and $\sigma^2_{v,k}$ is the segmentation-derived volume uncertainty at timepoint $k$. This is the key contribution: segmentation boundary uncertainty propagates into growth prediction uncertainty.

---

## 4. Configuration schema

Create `experiments/uncertainty_segmentation/config.yaml` as the single source of truth. The agent should also create a `config/picasso/` subdirectory for server-specific path overrides following the existing pattern in `src/growth/config/server/`.

```yaml
# experiments/uncertainty_segmentation/config.yaml
# LoRA-Ensemble for Uncertainty-Aware Meningioma Segmentation

# =============================================================================
# Experiment metadata
# =============================================================================
experiment:
  name: "lora_ensemble_uncertainty"
  description: "LoRA-Ensemble (M adapters, shared backbone) for uncertainty-aware segmentation"
  output_dir: ./results/uncertainty_segmentation

# =============================================================================
# Paths (override in config/picasso/ for server)
# =============================================================================
paths:
  # BrainSegFounder checkpoint directory (contains finetuned_model_fold_0.pt)
  checkpoint_dir: /path/to/BrainSegFounder_finetuned_BraTS
  
  # BraTS-MEN 2024 HDF5 (training data for LoRA adapters)
  men_h5_file: /path/to/BraTS_MEN.h5
  
  # MenGrowth longitudinal HDF5 (inference-only, no labels)
  mengrowth_h5_file: /path/to/MenGrowth.h5

# =============================================================================
# Ensemble configuration
# =============================================================================
ensemble:
  n_members: 5              # M: number of LoRA adapters (tunable: 3, 5, 7, 10)
  base_seed: 42             # seed_m = base_seed + m for m in 1..M

# =============================================================================
# LoRA configuration (fixed rank per run)
# =============================================================================
lora:
  rank: 8                   # r (fixed for all members in this run)
  alpha: 16                 # α (scaling = α/r = 2.0)
  dropout: 0.1              # LoRA dropout
  target_stages: [3, 4]     # SwinViT stages to adapt
  use_dora: false           # DoRA variant (Weight-Decomposed LoRA)

# =============================================================================
# Training configuration (each member trained independently)
# =============================================================================
training:
  epochs: 100
  batch_size: 2             # Per-GPU batch size (A100 40GB)
  val_batch_size: 1
  learning_rate:
    encoder: 1.0e-4         # LoRA adapter LR
    decoder: 1.0e-4         # Decoder LR
  weight_decay: 1.0e-5
  early_stopping:
    patience: 15
    min_delta: 0.001
    monitor: "val_dice_mean"
  gradient_clip: 1.0
  num_workers: 8
  pin_memory: true
  
  # Decoder configuration
  decoder_type: "original"  # Use full SwinUNETR decoder (not lightweight head)
  freeze_decoder: false     # Train decoder alongside LoRA adapters

# =============================================================================
# Loss configuration
# =============================================================================
loss:
  lambda_dice: 1.0
  lambda_ce: 1.0

# =============================================================================
# Data configuration
# =============================================================================
data:
  modalities: ["t2f", "t1c", "t1n", "t2w"]   # CRITICAL: FLAIR, T1ce, T1, T2
  roi_size: [128, 128, 128]                   # Training ROI
  val_roi_size: [192, 192, 192]               # Validation / inference ROI
  spacing: [1.0, 1.0, 1.0]
  train_split: "lora_train"
  val_split: "lora_val"
  test_split: "test"
  augment_train: true

# =============================================================================
# Inference configuration (ensemble prediction on MenGrowth)
# =============================================================================
inference:
  sw_roi_size: [128, 128, 128]    # Sliding window patch size
  sw_batch_size: 4                # Patches per forward pass
  sw_overlap: 0.5                 # Overlap fraction
  sw_mode: "gaussian"             # Blending mode
  save_probability_maps: false    # Save full 4D probability arrays (large!)
  save_uncertainty_maps: true     # Save voxel-wise entropy/MI maps
  save_ensemble_masks: false      # Save per-member hard masks (debugging)

# =============================================================================
# Evaluation configuration
# =============================================================================
evaluation:
  compute_ece: true               # Expected Calibration Error
  ece_n_bins: 15                  # Number of bins for ECE
  compute_brier: true             # Brier score
  compute_reliability: true       # Reliability diagram data
```

---

## 5. Module structure

```
experiments/uncertainty_segmentation/
├── __init__.py
├── config.yaml                          # Default config (local paths)
├── config/
│   └── picasso/
│       └── config_picasso.yaml          # Server path overrides
├── README.md                            # Module documentation
│
├── engine/
│   ├── __init__.py
│   ├── train_member.py                  # Train a single LoRA adapter (core training loop)
│   ├── ensemble_inference.py            # Load M adapters, run ensemble prediction
│   ├── uncertainty_metrics.py           # Entropy, MI, ECE, Brier, reliability
│   └── volume_extraction.py            # Extract volumes per member, compute stats
│
├── run_train.py                         # CLI entry point: train member m (called by SLURM)
├── run_inference.py                     # CLI entry point: ensemble inference on MenGrowth
├── run_evaluate.py                      # CLI entry point: evaluate ensemble on BraTS-MEN test
│
└── slurm/
    ├── launch.sh                        # Orchestrator: submit array + analysis jobs
    ├── train_worker.sh                  # SLURM worker: trains one member (1 GPU)
    ├── inference_worker.sh              # SLURM worker: ensemble inference (1 GPU)
    └── setup.sh                         # Pre-flight validation
```

---

## 6. Implementation goals (dependency-ordered)

### Goal 1: Configuration and scaffolding

**Create** the directory structure above with `__init__.py` files, the `config.yaml`, and the `README.md`. The README should briefly describe the LoRA-Ensemble approach, the module's purpose, and how to run it.

**Depends on:** Nothing.  
**Verifiable:** `python -c "from experiments.uncertainty_segmentation import *"` succeeds.

### Goal 2: Training engine (`engine/train_member.py`)

**Implement** a function `train_single_member(config, member_id, device)` that:

1. Sets the global seed to `config.ensemble.base_seed + member_id` (torch, numpy, python random, CUDA).
2. Loads the BrainSegFounder checkpoint from `config.paths.checkpoint_dir`. Use existing `load_swin_encoder()` or `load_full_swinunetr()` from `src/growth/models/encoder/swin_loader.py`.
3. Creates a `LoRASwinViT` adapter with the config-specified rank, alpha, dropout, target stages. Use existing class from `src/growth/models/encoder/lora_adapter.py`.
4. Wraps it in `LoRAOriginalDecoderModel` from `src/growth/models/segmentation/original_decoder.py` (LoRA encoder + original SwinUNETR decoder, unfrozen decoder).
5. Creates train/val dataloaders from BraTS-MEN HDF5 using `create_dataloaders()` from `src/growth/data/bratsmendata.py`.
6. Trains with `SegmentationLoss3Ch` (Dice + CE) and `DiceMetric3Ch` for validation.
7. Implements early stopping on validation Dice.
8. Saves the LoRA adapter via `lora_encoder.save_lora(output_dir / "adapters" / f"member_{member_id}" / "adapter")`.
9. Also saves the decoder state dict separately: `torch.save(decoder.state_dict(), ..., "decoder.pt")`.
10. Saves a training log (CSV with epoch, train_loss, val_dice columns) and a summary YAML.

**Key design decision:** Each member trains the *same architecture* (same LoRA rank, same decoder type) but with *different random initialization* and *different data augmentation ordering*. The seed controls both PyTorch weight init and dataloader shuffling.

**Reuse extensively** from `experiments/stage3_latent/lora/engine/train_condition.py`. That file has the training loop with Dice+CE loss, DiceMetric3Ch, early stopping, and model creation. Adapt it — do not rewrite from scratch.

**Depends on:** Goal 1.  
**Verifiable:** Train 1 member on BraTS-MEN with `--member-id 0` (locally, small epoch count). Check that adapter directory is created with PEFT files. Check that decoder state dict is saved. Check that training CSV has expected columns.

### Goal 3: Training CLI (`run_train.py`)

**Implement** a CLI entry point:

```bash
python -m experiments.uncertainty_segmentation.run_train \
    --config experiments/uncertainty_segmentation/config.yaml \
    --member-id 0 \
    --device cuda:0
```

This script:
1. Loads config with OmegaConf.
2. Validates paths (checkpoint exists, H5 exists).
3. Calls `train_single_member(config, member_id, device)`.
4. Returns exit code 0 on success.

**Depends on:** Goal 2.  
**Verifiable:** `python -m experiments.uncertainty_segmentation.run_train --help` shows expected arguments.

### Goal 4: SLURM infrastructure

**Create** `slurm/` scripts following the existing pattern in `slurm/lora/`.

**`launch.sh`:**
- Reads `n_members` from config (or accepts as argument).
- Submits a SLURM array job `--array=0-{M-1}` where each task trains one member.
- Submits a dependent inference job after all training completes.

**`train_worker.sh`:**
```bash
#SBATCH -J lora_ens_train
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
```
- `MEMBER_ID=$SLURM_ARRAY_TASK_ID`
- Activates conda, cd to repo, runs `run_train.py --member-id $MEMBER_ID`.

Follow the exact patterns in `slurm/lora/train_worker_dual_domain.sh` for environment setup (module loading, conda activation, pre-flight checks).

**Depends on:** Goal 3.  
**Verifiable:** `bash experiments/uncertainty_segmentation/slurm/setup.sh` passes all checks (paths exist, imports work).

### Goal 5: Ensemble inference engine (`engine/ensemble_inference.py`)

**Implement** a class `EnsemblePredictor`:

```python
class EnsemblePredictor:
    """Loads M LoRA adapters on a shared backbone and produces ensemble predictions."""
    
    def __init__(self, config: OmegaConf, device: str = "cuda"):
        """Load backbone once, discover adapter checkpoints."""
        ...
    
    def predict_scan(self, images: torch.Tensor) -> EnsemblePrediction:
        """Run M forward passes, aggregate into ensemble statistics.
        
        Args:
            images: [1, 4, D, H, W] input volume.
            
        Returns:
            EnsemblePrediction dataclass with mean_probs, entropy, MI, 
            per_member_masks, per_member_volumes, ensemble_mask, etc.
        """
        ...
```

**Adapter loading strategy:** The backbone stays in GPU memory throughout. For each member, load the LoRA adapter via `LoRASwinViT.load_lora()`, run inference, then discard the adapter. This avoids having M copies in memory. Alternatively, if memory allows, merge all LoRA weights into M copies of the decoder-only model (since the encoder is shared). The first approach (sequential loading) is safer for GPU memory.

**Key implementation details:**

1. For each adapter, load it, create the full segmentation model (LoRA encoder + decoder), run `sliding_window_segment()` from `src/growth/inference/sliding_window.py`, collect softmax probabilities.
2. Accumulate softmax outputs in a running mean/variance to avoid storing M full-resolution probability maps (which would be M × 4 × 192 × 192 × 192 × float32 = huge).
3. Compute per-member hard masks and volumes immediately, storing only scalars.
4. At the end of M passes, compute final entropy and mutual information maps.

**Memory-efficient aggregation (Welford's online algorithm):**

```python
# Running mean and M2 for online variance
mean_probs = torch.zeros(C, D, H, W)  # running mean
M2_probs = torch.zeros(C, D, H, W)    # sum of squared deviations
member_volumes = []

for m in range(M):
    probs_m = softmax(sliding_window_segment(model_m, images))  # [1, C, D, H, W]
    
    # Welford update
    delta = probs_m - mean_probs
    mean_probs += delta / (m + 1)
    delta2 = probs_m - mean_probs
    M2_probs += delta * delta2
    
    # Volume from hard mask
    mask_m = (probs_m[0, 1:].sum(0) > 0.5)  # WT = labels 1+2+3
    vol_m = mask_m.sum().item() * voxel_volume
    member_volumes.append(vol_m)
    
    # For MI: accumulate per-member entropies
    h_m = -(probs_m * torch.log(probs_m + 1e-8)).sum(dim=1)  # [D, H, W]
    mean_member_entropy += h_m / M

var_probs = M2_probs / (M - 1)
predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=0)
mutual_information = predictive_entropy - mean_member_entropy
```

**Depends on:** Goal 2 (trained adapters exist).  
**Verifiable:** Given M=2 adapters (even randomly initialized), the inference produces mean_probs of shape `[C, D, H, W]`, a scalar volume ± std, and non-zero entropy maps.

### Goal 6: Volume extraction with uncertainty (`engine/volume_extraction.py`)

**Implement** functions:

```python
def extract_ensemble_volumes(
    predictor: EnsemblePredictor,
    h5_path: Path,
    config: OmegaConf,
) -> pd.DataFrame:
    """Extract volumes for all scans in an HDF5 file.
    
    Returns DataFrame with columns:
        scan_id, patient_id, timepoint_idx,
        volume_mean_mm3, volume_std_mm3,
        log_volume_mean, log_volume_std,
        volume_member_0, ..., volume_member_{M-1},
        mean_entropy, mean_boundary_entropy
    """
```

This iterates over all scans in the MenGrowth HDF5, calls `predictor.predict_scan()`, and collects results into a structured DataFrame. The DataFrame should also include the per-member volumes as separate columns (for downstream analysis).

**Output CSV format:**

| scan_id | patient_id | timepoint_idx | vol_mean | vol_std | logvol_mean | logvol_std | vol_m0 | vol_m1 | ... | mean_entropy |
|---------|------------|---------------|----------|---------|-------------|------------|--------|--------|-----|--------------|
| MEN_001_00 | MEN_001 | 0 | 15432.0 | 312.5 | 9.644 | 0.020 | 15120 | 15744 | ... | 0.045 |

**Depends on:** Goal 5.  
**Verifiable:** CSV is generated with correct columns. Volume values are positive. Std is non-zero (ensemble members disagree on boundary voxels).

### Goal 7: Uncertainty metrics (`engine/uncertainty_metrics.py`)

**Implement** calibration evaluation functions:

```python
def compute_ece(
    probs: np.ndarray,        # [N_voxels, C] predicted probabilities
    labels: np.ndarray,       # [N_voxels] ground truth class labels
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (multiclass)."""

def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Multiclass Brier score."""

def compute_reliability_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Return per-bin accuracy and confidence for reliability diagrams."""
```

These operate on the BraTS-MEN **test split** where ground truth labels are available. The metrics quantify whether the ensemble's softmax probabilities are calibrated (i.e., when the model says 80% confident, it is correct 80% of the time).

**Depends on:** Goal 5.  
**Verifiable:** ECE is a float in [0, 1]. Brier score is a float in [0, 2]. Reliability data has n_bins entries.

### Goal 8: Evaluation CLI (`run_evaluate.py`)

**Implement** a CLI that:

1. Loads M trained adapters.
2. Runs ensemble inference on BraTS-MEN test split (where labels exist).
3. Computes per-member Dice, ensemble-averaged Dice.
4. Computes ECE, Brier score, reliability data.
5. Saves results to `output_dir/evaluation/`.

```bash
python -m experiments.uncertainty_segmentation.run_evaluate \
    --config experiments/uncertainty_segmentation/config.yaml \
    --device cuda:0
```

**Depends on:** Goal 5, Goal 7.  
**Verifiable:** Evaluation produces JSON with Dice per member, ensemble Dice, ECE, Brier. Ensemble Dice ≥ best single-member Dice (expected for well-calibrated ensembles).

### Goal 9: Inference CLI (`run_inference.py`)

**Implement** a CLI that runs ensemble inference on MenGrowth:

```bash
python -m experiments.uncertainty_segmentation.run_inference \
    --config experiments/uncertainty_segmentation/config.yaml \
    --device cuda:0
```

This calls `extract_ensemble_volumes()` and saves the volume CSV plus optional uncertainty maps.

**Depends on:** Goal 5, Goal 6.  
**Verifiable:** CSV file exists with one row per MenGrowth scan, all volume columns populated.

---

## 7. Testing strategy

Create `tests/growth/test_uncertainty_segmentation.py`:

```python
pytestmark = [pytest.mark.unit]

class TestEnsemblePredictor:
    """Tests with synthetic data (no real checkpoints)."""
    
    def test_welford_aggregation():
        """Verify online mean/variance matches offline computation."""
    
    def test_entropy_computation():
        """Verify predictive entropy on known distributions."""
    
    def test_volume_extraction():
        """Verify volume = sum(WT voxels) × voxel_spacing³."""

class TestUncertaintyMetrics:
    def test_ece_perfect_calibration():
        """ECE = 0 when probs match empirical accuracy."""
    
    def test_brier_score_perfect():
        """Brier = 0 when probs are one-hot and correct."""
```

These tests should **not** require real data or GPUs. Use synthetic tensors.

---

## 8. Discovery tasks for the agent

The agent needs to investigate these before writing code:

### 8.1 BrainSegFounder checkpoint structure

Run `python -c "import torch; sd = torch.load('path/to/finetuned_model_fold_0.pt', map_location='cpu'); print(list(sd.keys())[:20])"` to understand the state dict key structure. The existing code in `swin_loader.py` handles this, but the agent should understand the key prefix patterns.

### 8.2 Model forward pass signature

The `LoRAOriginalDecoderModel.forward(x)` takes `[B, 4, D, H, W]` and returns `[B, 4, D, H, W]` logits (4 classes: background, ET, NET, cystic). Verify by reading `src/growth/models/segmentation/original_decoder.py`.

### 8.3 Decoder saving/loading

The current `LoRASwinViT.save_lora()` saves only the LoRA adapter weights (via PEFT). The decoder is part of `LoRAOriginalDecoderModel` and is **not** saved by PEFT. The agent must save the decoder separately (`torch.save(model.decoder.state_dict(), ...)`). On reload, load the base model, apply LoRA adapter, then load decoder weights.

### 8.4 MenGrowth HDF5 inspection

The agent should inspect the MenGrowth H5 file to verify:
- The `longitudinal/` group structure.
- Whether `segs` contains zeros (no labels) or actual segmentations.
- The `patient_offsets` array for mapping scans to patients.

---

## 9. Coding standards

Follow the project-wide standards documented in `.claude/rules/testing.md` and `.claude/hooks/compact-context.sh`:

- **Type hints** on all functions.
- **Google-style docstrings** (description, Args, Returns, Raises).
- **`logging` module** (never `print`).
- **Atomic function design**: each function does one thing.
- **OOP**: Use dataclasses for structured data, classes for stateful components.
- **Custom exceptions**: Create `EnsembleTrainingError`, `EnsembleInferenceError` as needed.
- **Explicit memory management**: Call `torch.cuda.empty_cache()` between adapter loads during inference. Delete unused tensors.
- **Low cyclomatic complexity**: Prefer early returns and guard clauses.
- **Use established libraries**: MONAI for medical imaging, PEFT for LoRA, omegaconf for config.
- **No magic numbers**: All constants from config or named module-level variables.

---

## 10. Summary of critical decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Uncertainty method | LoRA-Ensemble | Parameter-efficient, calibrated, reuses existing LoRA infra |
| Number of members | 5 (default, tunable) | Mühlematter et al. show diminishing returns beyond M=5–8 |
| LoRA rank | Fixed per run (default 8) | Matches existing ablation; rank variation is an orthogonal study |
| Diversity source | Random seed (init + data order) | Sufficient per Mühlematter et al.; no hyperparameter variation |
| Checkpoint resume | BrainSegFounder Stage 3 (finetuned) | SSL-only checkpoint not publicly available; LoRA on Stages 3–4 restructures high-level features regardless |
| Decoder strategy | Original SwinUNETR decoder, unfrozen | Matches existing `LoRAOriginalDecoderModel` pattern |
| Volume uncertainty | Ensemble disagreement (not soft probability) | Direct, no approximations, interpretable |
| Memory strategy | Sequential adapter loading (one at a time) | Safe for GPU memory; ~197K param swap is near-instant |
| Aggregation | Welford online algorithm | Avoids storing M full-resolution probability maps |

---

## 11. References

1. Mühlematter, D.J. et al. (2024). *LoRA-Ensemble: Efficient Uncertainty Modelling for Self-Attention Networks.* arXiv:2405.14438.
2. Lakshminarayanan, B. et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.
3. Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016.
4. Cox, J. et al. (2024). *BrainSegFounder: Towards Brain Segmentation Foundation Models.* arXiv.
5. Hu, E.J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
6. Lambert, B. et al. (2024). *Uncertainty quantification in medical image analysis.* Artificial Intelligence in Medicine, 150, 102830.
7. Scalco, E. et al. (2024). *Uncertainty quantification in multi-class segmentation: Bayesian vs non-Bayesian.* Medical Physics.
8. Nair, T. et al. (2018). *Exploring uncertainty measures in deep networks for MS lesion detection.* MICCAI 2018.
