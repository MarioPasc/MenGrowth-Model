#!/usr/bin/env python
"""Generate a scientific report for the LoRA ablation experiments.

Thin entry point — delegates to experiments.lora.report.cli.

Usage:
    python -m experiments.lora.generate_report \
        --results-dir /media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/LORA_Module/lora_dual_domain_v1 \
        --output-dir ./report_output \
        --mode both \
        --compare-semantic \
        --skip-umap
"""

from experiments.lora.report.cli import main

if __name__ == "__main__":
    main()
