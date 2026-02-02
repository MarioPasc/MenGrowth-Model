# LoRA Ablation Experiment

This experiment evaluates the effectiveness of Low-Rank Adaptation (LoRA) for adapting a pretrained BrainSegFounder encoder to meningioma segmentation, measuring both segmentation quality and semantic feature decodability from learned representations.

## Objective

Compare different LoRA ranks (2, 4, 8, 16, 32) against a frozen encoder baseline to determine:

1. **Optimal LoRA rank** for meningioma domain adaptation
2. **Semantic decodability** of learned representations via linear and MLP probes
3. **Segmentation quality** measured by Dice coefficient

## Architecture

### Encoder

The experiment uses the **BrainSegFounder** Swin-UNETR encoder pretrained on BraTS data. The encoder is frozen for all conditions, with LoRA adapters injected at stages 3-4 for the LoRA conditions.

### Decoder Types

The experiment supports two decoder architectures controlled by `training.decoder_type`:

| Decoder Type | Parameters | Description |
|--------------|------------|-------------|
| `"original"` | ~30M | Full SwinUNETR decoder with pretrained weights (recommended) |
| `"lightweight"` | ~2M | Custom SegmentationHead CNN decoder |

**Recommendation**: Use `"original"` decoder for best performance (~0.85 Dice on BraTS). The pretrained decoder weights are essential for good segmentation quality.

### Semantic Heads (Optional)

When `training.use_semantic_heads: true`, auxiliary prediction heads are added to predict semantic features from the bottleneck representation:

- **Volume head**: Predicts log-transformed tumor volumes (total, NCR, ED, ET)
- **Location head**: Predicts tumor centroid coordinates (x, y, z)
- **Shape head**: Predicts morphological features (sphericity, surface area, solidity, aspect ratios)

The auxiliary semantic loss is controlled by `training.lambda_aux` and individual lambdas in the `loss` section.

## Experimental Conditions

| Condition | LoRA Rank | LoRA Alpha | Trainable Encoder Params |
|-----------|-----------|------------|--------------------------|
| `baseline` | - | - | 0 (frozen) |
| `lora_r2` | 2 | 4 | ~49K |
| `lora_r4` | 4 | 8 | ~98K |
| `lora_r8` | 8 | 16 | ~197K |
| `lora_r16` | 16 | 32 | ~393K |
| `lora_r32` | 32 | 64 | ~786K |

All conditions train the decoder (~30M params for original decoder).

## Evaluation Metrics

### Primary Metrics

**Linear Probe R²**: Measures how linearly separable semantic features are in the learned representation. This is the key metric for evaluating representation quality.

- `r2_volume_linear`: Volume prediction R² with Ridge regression
- `r2_location_linear`: Location prediction R² with Ridge regression
- `r2_shape_linear`: Shape prediction R² with Ridge regression

### Secondary Metrics

**MLP Probe R²**: Measures how much semantic information is encoded (linearly or nonlinearly).

- `r2_volume_mlp`: Volume prediction R² with 2-layer MLP
- `r2_location_mlp`: Location prediction R² with 2-layer MLP
- `r2_shape_mlp`: Shape prediction R² with 2-layer MLP

**Nonlinearity Gap**: `MLP R² - Linear R²` indicates how much information requires nonlinear decoding. A small gap suggests features are linearly organized.

**Segmentation Dice**:
- `dice_mean`: Mean Dice across tumor subregions
- `dice_0`, `dice_1`, `dice_2`: Per-class Dice (NCR, ED, ET)

## Configuration

### Key Configuration Options

```yaml
experiment:
  name: lora_ablation
  seed: 42
  output_dir: /path/to/results

training:
  decoder_type: "original"    # "original" | "lightweight"
  use_semantic_heads: false   # Enable auxiliary semantic loss during training
  freeze_decoder: false       # Whether to freeze decoder weights
  lambda_aux: 0.1            # Weight for auxiliary semantic loss
  max_epochs: 100
  batch_size: 2
  lr_encoder: 1e-4           # Learning rate for LoRA parameters
  lr_decoder: 1e-4           # Learning rate for decoder parameters

probe:
  use_mlp_probes: true       # Enable MLP probes (in addition to linear)
  alpha_linear: 1.0          # Ridge regression regularization
  alpha_mlp: 1e-4            # MLP weight decay
  mlp_hidden_dim: 256
  mlp_epochs: 100

feature_extraction:
  level: multi_scale         # "multi_scale" | "encoder10"
  # multi_scale: concatenate layers 2+3+4 (1344-dim)
  # encoder10: use layer 4 only (768-dim)
```

### Ablation Configurations

**Ablation 1: No Semantic Heads**
```yaml
training:
  use_semantic_heads: false
```
Evaluates pure segmentation-driven representation learning.

**Ablation 2: Semantic Heads for All Conditions**
```yaml
training:
  use_semantic_heads: true
  lambda_aux: 0.1
```
Evaluates whether auxiliary semantic loss improves representation quality for both baseline and LoRA conditions.

## Usage

### Full Pipeline

```bash
# Run all conditions
growth-exp-lora-ablation --config experiments/lora_ablation/config/ablation.yaml run-all

# Or with Python module
python -m experiments.lora_ablation.run_ablation \
    --config experiments/lora_ablation/config/ablation.yaml \
    run-all
```

### Individual Steps

```bash
# Train a single condition
python -m experiments.lora_ablation.run_ablation \
    --config experiments/lora_ablation/config/ablation.yaml \
    train --condition lora_r8

# Extract features
python -m experiments.lora_ablation.run_ablation \
    --config experiments/lora_ablation/config/ablation.yaml \
    extract --condition lora_r8

# Evaluate probes
python -m experiments.lora_ablation.run_ablation \
    --config experiments/lora_ablation/config/ablation.yaml \
    evaluate --condition lora_r8

# Generate visualizations
python -m experiments.lora_ablation.run_ablation \
    --config experiments/lora_ablation/config/ablation.yaml \
    visualize
```

### Quick Test

```bash
# Test training with 2 epochs
python -m experiments.lora_ablation.train_condition \
    --config experiments/lora_ablation/config/ablation.yaml \
    --condition baseline \
    --max-epochs 2
```

## Output Structure

```
output_dir/
├── conditions/
│   ├── baseline/
│   │   ├── checkpoint.pt           # Best model checkpoint
│   │   ├── best_model.pt           # Full model state dict
│   │   ├── training_log.csv        # Per-epoch training metrics
│   │   ├── training_summary.yaml   # Training configuration and results
│   │   ├── features_test_*.pt      # Extracted features
│   │   ├── targets_test.pt         # Semantic targets
│   │   ├── metrics.json            # Probe evaluation metrics
│   │   └── predictions.json        # Probe predictions
│   ├── lora_r2/
│   │   ├── adapter/                # LoRA adapter weights
│   │   │   ├── adapter_model.safetensors
│   │   │   └── adapter_config.json
│   │   └── ...
│   └── ...
├── figures/
│   ├── r2_comparison_enhanced.pdf  # Linear vs MLP R² comparison
│   ├── variance_per_dim.pdf        # Feature variance analysis
│   ├── nonlinearity_gap.pdf        # MLP-Linear R² gap
│   ├── umap_semantic.pdf           # UMAP colored by semantics
│   └── scatter_*.pdf               # Prediction scatter plots
└── summary.json                    # Cross-condition summary
```

## Expected Results

Based on preliminary experiments with the original decoder:

| Condition | Dice Mean | Volume R² (Linear) | Location R² (Linear) |
|-----------|-----------|--------------------|--------------------|
| baseline | ~0.82 | ~0.55 | ~0.75 |
| lora_r4 | ~0.83 | ~0.62 | ~0.78 |
| lora_r8 | ~0.84 | ~0.68 | ~0.82 |
| lora_r16 | ~0.84 | ~0.70 | ~0.83 |
| lora_r32 | ~0.84 | ~0.71 | ~0.84 |

**Key Observations**:
- LoRA adaptation improves semantic decodability more than segmentation Dice
- Ranks 8-16 provide a good trade-off between adaptation capacity and efficiency
- Location features are more linearly decodable than volume features
- Shape features are hardest to predict (R² typically < 0.4)

## Implementation Details

### LoRA Injection Points

LoRA adapters are injected at attention layers in stages 3-4 of the Swin Transformer encoder:
- `swinViT.layers3.blocks.*.attn.{qkv,proj}`
- `swinViT.layers4.blocks.*.attn.{qkv,proj}`

This targets the deepest encoder stages where domain-specific adaptations are most effective.

### Feature Extraction

For probe evaluation, features are extracted from:
- **Multi-scale** (`level: multi_scale`): Concatenation of encoder layers 2, 3, 4 after global average pooling (1344-dim)
- **Single-scale** (`level: encoder10`): Layer 4 only after global average pooling (768-dim)

### Probe Training

Linear and MLP probes are trained on the **probe_train** split and evaluated on the **probe_test** split. This ensures probes evaluate representation quality, not overfitting to training data.

## Files

| File | Description |
|------|-------------|
| `run_ablation.py` | Main orchestrator with subcommands |
| `train_condition.py` | Training script for single condition |
| `extract_features.py` | Feature extraction from trained models |
| `evaluate_probes.py` | Linear and MLP probe training/evaluation |
| `visualizations.py` | Publication-quality figure generation |
| `model_factory.py` | Unified model creation factory |
| `data_splits.py` | Data split management |
| `analyze_results.py` | Cross-condition analysis |
| `statistical_analysis.py` | Statistical significance testing |

## References

- **LoRA**: Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
- **BrainSegFounder**: [Foundation model for brain tumor segmentation]
- **SwinUNETR**: Hatamizadeh et al. (2022). "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images."
