"""Thin wrapper so the analysis can be run as ``python -m …benchmark.run_analysis``."""

from __future__ import annotations

import sys

from experiments.uncertainty_segmentation.benchmark.analysis.cli import main

if __name__ == "__main__":
    sys.exit(main())
