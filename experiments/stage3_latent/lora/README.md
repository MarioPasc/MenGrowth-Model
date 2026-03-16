# experiments/lora/ — Unified LoRA Adaptation Module

Unified package for Phase 1 (LoRA/DoRA encoder adaptation) of the MenGrowth pipeline.
Handles both **single-domain** (MEN-only rank ablation) and **dual-domain** (MEN+GLI mixed-batch) experiments via YAML config.

## Quick Start

```bash
# Single-domain ablation (MEN only)
python -m experiments.lora.run --config experiments/lora/config/ablation_v3.yaml run-all

# Dual-domain experiment (MEN + GLI)
python -m experiments.lora.run --config experiments/lora/config/dual_domain_v1.yaml run-all

# Individual steps
python -m experiments.lora.run --config <config.yaml> splits
python -m experiments.lora.run --config <config.yaml> train --condition lora_r8
python -m experiments.lora.run --config <config.yaml> extract --condition lora_r8
python -m experiments.lora.run --config <config.yaml> probes --condition lora_r8
python -m experiments.lora.run --config <config.yaml> dice --condition lora_r8
```

## Directory Structure

```
experiments/lora/
├── run.py                    # Unified CLI orchestrator (all subcommands)
├── generate_report.py        # HTML report entry point
├── README.md                 # This file
│
├── engine/                   # Core training + extraction
│   ├── train_condition.py    # Training loop (single + dual domain)
│   ├── extract_features.py   # Feature extraction (encoder10, multi-scale, TAP)
│   ├── model_factory.py      # Model creation (frozen, baseline, LoRA, DoRA)
│   └── data_splits.py        # Patient-level train/val/test splits
│
├── eval/                     # Evaluation modules
│   ├── evaluate_dice.py      # Per-domain Dice (single + dual)
│   ├── evaluate_probes.py    # GP probes (per-domain + cross-domain)
│   ├── evaluate_feature_quality.py  # PCA rank, DCI, variance diagnostics
│   └── evaluate_domain_gap.py      # MMD², CKA, PAD metrics
│
├── analysis/                 # Post-training analysis
│   ├── analyze_results.py    # Comprehensive analysis + recommendations
│   ├── statistical_analysis.py  # Bootstrap CI, Wilcoxon, effect sizes
│   ├── generate_tables.py    # CSV + LaTeX tables
│   ├── enhanced_diagnostics.py  # Gradient + feature diagnostics
│   ├── regenerate_analysis.py   # Regenerate from cached data
│   ├── v3_cache.py           # Figure data precomputation
│   └── v3_figures.py         # Thesis-quality figures
│
├── report/                   # HTML report generator
│   ├── cli.py                # Report CLI
│   ├── data_loader.py        # Load experiment results
│   ├── figures.py            # Publication figures
│   ├── html_builder.py       # Jinja2 HTML builder
│   ├── narrative.py          # Auto-generated narrative
│   ├── style.py              # Style constants
│   └── templates/            # Jinja2 templates
│
├── vis/                      # Visualization
│   ├── visualizations.py     # UMAP, scatter, R² comparison
│   └── dual_domain_viz.py    # Dual-domain UMAP, sausage plots
│
├── utils/                    # Shared utilities
│   └── output_paths.py       # Output directory helpers
│
├── scripts/                  # Standalone utilities
│   ├── merge_lora_checkpoint.py  # Merge LoRA into base encoder
│   └── post_hoc_analysis.py     # Post-hoc statistical analysis
│
└── config/                   # YAML configurations
    ├── ablation.yaml          # v1 ablation config
    ├── ablation_v3.yaml       # v3 ablation (GP probes, encoder10)
    ├── dual_domain_v1.yaml    # Dual-domain MEN+GLI config
    ├── local/                 # Local development configs
    ├── server/                # Server configs (4 variants)
    └── picasso/               # Picasso HPC configs
```

## Single-Domain vs Dual-Domain

The module auto-detects the experiment type from config:

- **Single-domain**: Config has `paths.h5_file` → MEN-only training + evaluation
- **Dual-domain**: Config has `paths.men_h5_file` + `paths.gli_h5_file` → Mixed-batch training with per-domain evaluation

Dual-domain adds:
- Per-domain feature extraction and Dice evaluation
- Cross-domain GP probes (train GLI → test MEN, and vice versa)
- Domain gap metrics (MMD², CKA, PAD)
- VICReg encoder regularization

## CLI Subcommands

| Command | Description |
|---------|-------------|
| `splits` | Generate patient-level data splits |
| `train` | Train one condition |
| `train-all` | Train all conditions |
| `extract` | Extract features for one condition |
| `extract-all` | Extract all features |
| `probes` | GP probes for one condition |
| `probes-all` | GP probes for all |
| `dice` | Dice evaluation (per-domain for dual) |
| `dice-all` | Dice for all conditions |
| `domain-gap` | MMD², CKA, PAD (dual-domain only) |
| `feature-quality` | PCA rank, DCI, variance |
| `visualize` | Generate all figures |
| `generate-tables` | CSV + LaTeX tables |
| `analyze` | Statistical analysis |
| `enhanced-diagnostics` | Gradient + feature diagnostics |
| `regenerate` | Regenerate from cached data |
| `run-all` | Complete pipeline |
| `analyze-only` | Re-run analysis only |

## SLURM (Picasso)

```bash
# Full rank sweep (7 conditions × 1 A100 GPU each)
bash slurm/lora_adaptation/launch_v3.sh

# Smoke test (dual-domain, 2 epochs)
bash slurm/lora_adaptation/tests/smoke_loginexa.sh
```
