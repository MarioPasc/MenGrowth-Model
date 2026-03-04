#!/usr/bin/env python
"""Generate a scientific report for the LoRA ablation experiments.

Thin entry point — delegates to experiments.lora.report.cli.

Usage:
    python -m experiments.lora.generate_report \
        --results-dir /path/to/LoRA_Adaptation \
        --output-dir ./report_output \
        --mode both \
        --compare-semantic \
        --skip-umap
"""

from experiments.lora.report.cli import main

if __name__ == "__main__":
    main()
