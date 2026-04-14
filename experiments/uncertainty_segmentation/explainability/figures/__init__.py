"""Figure generation modules for the explainability analysis.

Each submodule consumes the per-scan CSV / NPZ artefacts in
``{output_dir}/raw/`` and writes a PDF in ``{output_dir}/figures/``.
The split between ``run_analysis.py`` (compute) and
``run_figures_only.py`` (render) lets users iterate on figure
formatting without rerunning the GPU pipeline.
"""
