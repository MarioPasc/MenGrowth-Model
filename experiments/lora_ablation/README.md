# LoRA Ablation Experiment

Systematic ablation of **LoRA/DoRA rank** on BrainSegFounder (SwinUNETR) for
meningioma encoder adaptation. Measures downstream semantic linear probe R²,
segmentation Dice, and domain gap (GLI vs MEN) across ranks 2-64.

## Directory Structure

```
experiments/lora_ablation/
├── run_ablation.py              # CLI orchestrator (entry point)
├── generate_report.py           # HTML report entry point
├── __init__.py                  # Compat shims for old flat imports
├── README.md
│
├── pipeline/                    # Core training / extraction / evaluation
│   ├── train_condition.py       #   Train one ablation condition
│   ├── extract_features.py      #   Extract encoder10 features
│   ├── extract_domain_features.py # Extract GLI + MEN features for domain gap
│   ├── evaluate_dice.py         #   Sliding-window Dice on test set
│   ├── evaluate_probes.py       #   Linear + MLP probe R² evaluation
│   ├── evaluate_feature_quality.py # PCA rank, DCI, variance spectrum
│   ├── model_factory.py         #   Build SwinUNETR + LoRA for a condition
│   ├── data_splits.py           #   Deterministic train/val/test splits
│   └── output_paths.py          #   Canonical output path helpers
│
├── analysis/                    # Post-training analysis & visualization
│   ├── analyze_results.py       #   Statistical analysis + report generation
│   ├── statistical_analysis.py  #   Friedman, Wilcoxon, effect sizes
│   ├── enhanced_diagnostics.py  #   Gradient norms, feature quality diagnostics
│   ├── visualizations.py        #   UMAP, variance, R² bar charts
│   ├── domain_visualizations.py #   Domain shift figures (KDE, UMAP, retention)
│   ├── compute_domain_metrics.py#   MMD, CKA, proxy A-distance
│   ├── generate_tables.py       #   CSV + LaTeX summary tables
│   ├── regenerate_analysis.py   #   Regenerate all outputs from cached data
│   ├── v3_cache.py              #   Precompute figure data for v3 conditions
│   └── v3_figures.py            #   Eight thesis-quality v3 figures
│
├── scripts/                     # Standalone diagnostics (not imported by pipeline)
│   ├── diagnose_frozen_gli.py   #   Frozen encoder Dice on GLI vs MEN
│   ├── post_hoc_analysis.py     #   Post-hoc gradient / checkpoint analysis
│   └── merge_lora_checkpoint.py #   Merge LoRA weights into base checkpoint
│
├── report/                      # HTML report generator
│   ├── cli.py                   #   Argparse entry point
│   ├── data_loader.py           #   Load metrics from result directories
│   ├── figures.py               #   Matplotlib figure builders
│   ├── html_builder.py          #   Jinja2 HTML assembly
│   ├── narrative.py             #   Auto-generated prose sections
│   ├── style.py                 #   Re-exports from experiments/utils/settings.py
│   └── templates/               #   Jinja2 HTML templates
│
└── config/                      # YAML configurations
    ├── ablation.yaml            #   v2 default (6 conditions, ranks 2-32)
    ├── ablation_v3.yaml         #   v3 (10 conditions, VICReg, revised shape)
    ├── local/                   #   Local machine overrides
    ├── server/                  #   ICAI server overrides
    └── picasso/                 #   Picasso HPC overrides
```

## Subcommands

All commands go through `run_ablation.py`:

```bash
python -m experiments.lora_ablation.run_ablation --config <yaml> <command>
```

| Command              | Description                                      |
|----------------------|--------------------------------------------------|
| `run-all`            | Full pipeline: splits, train, extract, probes, Dice, viz, tables |
| `analyze-only`       | Re-run analysis on existing checkpoints          |
| `splits`             | Generate deterministic data splits               |
| `train --condition X`| Train a single condition                         |
| `train-all`          | Train all conditions sequentially                |
| `extract --condition X` | Extract features for one condition            |
| `extract-all`        | Extract features for all conditions              |
| `domain --condition X`  | Extract GLI + MEN features for domain gap     |
| `domain-all`         | Domain features for all conditions               |
| `probes --condition X`  | Evaluate probes for one condition              |
| `probes-all`         | Evaluate probes for all conditions               |
| `test-dice --condition X` | Dice evaluation for one condition           |
| `test-dice-all`      | Dice for all conditions (MEN + GLI)              |
| `feature-quality`    | PCA rank, DCI, variance spectrum                 |
| `visualize`          | Generate all figures                             |
| `generate-tables`    | CSV + LaTeX summary tables                       |
| `analyze`            | Statistical analysis + markdown report           |
| `enhanced-diagnostics` | Gradient norms, feature quality checks         |
| `regenerate`         | Regenerate figures/tables from cached data       |

## Pipeline Stages

```
1. splits        ->  Deterministic train/val/test assignment
2. train-all     ->  LoRA fine-tuning per condition (Dice + aux semantic loss)
3. extract-all   ->  encoder10 features [N, 768, 4, 4, 4] -> [N, 768]
4. domain-all    ->  Same extraction for GLI subjects (optional)
5. probes-all    ->  Linear + MLP probe R² for volume, location, shape
6. test-dice-all ->  Sliding-window Dice on held-out MEN + GLI
7. visualize     ->  UMAP, variance spectrum, R² bars, domain shift
8. tables        ->  CSV + LaTeX with all metrics
9. analyze       ->  Friedman test, Wilcoxon pairs, effect sizes, report
```

## Configuration

Configs use OmegaConf YAML. Key sections:

- `experiment`: name, seed, output_dir
- `paths`: checkpoint, data_root, glioma_root
- `data`: roi_size, feature_roi_size, spacing
- `data_splits`: lora_train, lora_val, sdp_train, test counts
- `conditions[]`: name, lora_rank, lora_alpha, target_stages, ...
- `training`: max_epochs, lr, decoder_type, num_workers, ...
- `loss`: lambda_dice, lambda_ce, lambda_volume, lambda_location, lambda_shape
- `probe`: use_mlp_probes, alpha_linear, mlp_hidden_dim, mlp_epochs

## Architecture

### Encoder

BrainSegFounder Swin-UNETR encoder pretrained on 41K+ subjects. LoRA adapters
injected at attention layers in stages 3-4:
- `swinViT.layers3.blocks.*.attn.{qkv,proj}`
- `swinViT.layers4.blocks.*.attn.{qkv,proj}`

### Decoder Types

| Type | Params | Description |
|------|--------|-------------|
| `"original"` | ~30M | Full SwinUNETR decoder with pretrained weights (recommended) |
| `"lightweight"` | ~2M | Custom SegmentationHead CNN decoder |

### Semantic Heads (Optional)

Auxiliary prediction heads from the bottleneck:
- **Volume**: log-transformed tumor volumes (total, NCR, ED, ET)
- **Location**: tumor centroid coordinates (x, y, z)
- **Shape**: sphericity, enhancement ratio, infiltration index (v3)

### Experimental Conditions

| Condition | LoRA Rank | Trainable Encoder Params |
|-----------|-----------|--------------------------|
| `baseline` / `baseline_frozen` | - | 0 (frozen) |
| `lora_r2` | 2 | ~49K |
| `lora_r4` | 4 | ~98K |
| `lora_r8` | 8 | ~197K |
| `lora_r16` | 16 | ~393K |
| `lora_r32` | 32 | ~786K |
| `lora_r64_full` (v3) | 64 | ~1.6M |

## Style Constants

All figure colors, labels, markers, and plot settings live in
`experiments/utils/settings.py` (single source of truth).
`report/style.py` is a thin re-export layer.

## Output Structure

```
<output_dir>/
├── conditions/<name>/
│   ├── checkpoints/             # Model checkpoints
│   ├── features.pt              # Extracted features
│   ├── features_glioma.pt       # GLI features (if --domain-features)
│   ├── probe_results.json       # Linear + MLP R² scores
│   ├── test_dice_men.json       # MEN Dice per class
│   └── test_dice_gli.json       # GLI Dice per class
├── figures/                     # Publication-quality figures (PDF + PNG)
├── figure_cache/                # Precomputed data for fast regeneration
├── comprehensive_results.csv
├── comprehensive_table.tex
├── test_dice_summary.csv
├── analysis_report.md
└── meta/run_manifest.json       # Reproducibility artifacts
```

## Picasso HPC Deployment

SLURM job scripts in `slurm/lora_adaptation/`:

```bash
# v3 training (one condition per GPU)
sbatch slurm/lora_adaptation/train_worker_v3.sh <condition_name>

# v3 analysis (CPU only, after all training)
sbatch slurm/lora_adaptation/analysis_worker_v3.sh

# v2 full pipeline
sbatch slurm/lora_adaptation/launch.sh
```

## Backward Compatibility

After the subpackage reorganization, old flat import paths still work via
`sys.meta_path` shims in `__init__.py`:

```python
# Both work:
from experiments.lora_ablation.model_factory import create_ablation_model  # compat
from experiments.lora_ablation.pipeline.model_factory import create_ablation_model  # canonical
```

## References

- **LoRA**: Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
- **DoRA**: Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." arXiv:2402.09353
- **BrainSegFounder**: Foundation model for brain tumor segmentation (41K+ subjects)
- **SwinUNETR**: Hatamizadeh et al. (2022). "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images."
