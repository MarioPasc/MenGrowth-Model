#!/usr/bin/env python
"""Generate a scientific report for the LoRA ablation experiments.

Thin entry point — delegates to experiments.lora_ablation.report.cli.

Usage:
    python -m experiments.lora_ablation.generate_report \
        --results-dir /path/to/LoRA_Adaptation \
        --output-dir ./report_output \
        --mode both \
        --compare-semantic \
        --skip-umap
"""

from experiments.lora_ablation.report.cli import main

if __name__ == "__main__":
    main()
