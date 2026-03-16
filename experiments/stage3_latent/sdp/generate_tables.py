#!/usr/bin/env python
# experiments/sdp/generate_tables.py
"""Generate CSV and LaTeX tables from SDP evaluation results.

Produces:
    tables/quality_summary.csv  — Key metrics in tabular form
    tables/quality_summary.tex  — LaTeX table for thesis

Usage:
    python -m experiments.sdp.generate_tables --run-dir outputs/sdp/my_run/
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from experiments.sdp.output_paths import load_run_paths

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_json_safe(path: Path) -> dict[str, Any]:
    """Load JSON file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _collect_metrics(paths: "SDPRunPaths") -> list[dict[str, str]]:
    """Collect key metrics from evaluation JSONs into a flat table.

    Args:
        paths: Run paths.

    Returns:
        List of {metric, value, threshold, status} dicts.
    """
    rows = []

    # Quality report (BLOCKING thresholds)
    quality = _load_json_safe(paths.quality_report_path)
    if quality:
        thresholds = {
            "r2_vol": (">=", 0.80),
            "r2_loc": (">=", 0.85),
            "r2_shape": (">=", 0.30),
            "max_cross_partition_corr": ("<=", 0.30),
        }
        for key, value in quality.items():
            if isinstance(value, (int, float)):
                direction, thresh = thresholds.get(key, ("", None))
                if thresh is not None:
                    if direction == ">=" and value >= thresh:
                        status = "PASS"
                    elif direction == "<=" and value <= thresh:
                        status = "PASS"
                    else:
                        status = "FAIL"
                    thresh_str = f"{direction} {thresh}"
                else:
                    status = ""
                    thresh_str = ""

                rows.append(
                    {
                        "metric": key,
                        "value": f"{value:.4f}",
                        "threshold": thresh_str,
                        "status": status,
                    }
                )

    # DCI scores
    dci = _load_json_safe(paths.dci_scores_path)
    if dci:
        for key in ["disentanglement", "completeness", "informativeness"]:
            if key in dci:
                rows.append(
                    {
                        "metric": f"dci_{key}",
                        "value": f"{dci[key]:.4f}",
                        "threshold": "",
                        "status": "",
                    }
                )

    # Variance analysis
    var = _load_json_safe(paths.variance_analysis_path)
    if var:
        for key in [
            "effective_rank",
            "mean_dim_std",
            "min_dim_std",
            "collapsed_dims_03",
            "pct_dims_std_gt_03",
        ]:
            if key in var:
                val = var[key]
                fmt = ".4f" if isinstance(val, float) else "d"
                rows.append(
                    {
                        "metric": key,
                        "value": f"{val:{fmt}}",
                        "threshold": "",
                        "status": "",
                    }
                )

    # Full metrics summary
    full = _load_json_safe(paths.full_metrics_path)
    if full and "summary" in full:
        for key, val in full["summary"].items():
            rows.append(
                {
                    "metric": key,
                    "value": f"{val:.4f}",
                    "threshold": "",
                    "status": "",
                }
            )

    return rows


def generate_csv(paths: "SDPRunPaths") -> Path:
    """Generate CSV quality summary table.

    Args:
        paths: Run paths.

    Returns:
        Path to generated CSV file.
    """
    rows = _collect_metrics(paths)
    csv_path = paths.tables / "quality_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "threshold", "status"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved CSV table: {csv_path}")
    return csv_path


def generate_latex(paths: "SDPRunPaths") -> Path:
    """Generate LaTeX quality summary table for thesis.

    Args:
        paths: Run paths.

    Returns:
        Path to generated .tex file.
    """
    rows = _collect_metrics(paths)
    tex_path = paths.tables / "quality_summary.tex"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{SDP Quality Summary}",
        r"\label{tab:sdp_quality}",
        r"\begin{tabular}{lrll}",
        r"\toprule",
        r"Metric & Value & Threshold & Status \\",
        r"\midrule",
    ]

    for row in rows:
        metric = row["metric"].replace("_", r"\_")
        value = row["value"]
        threshold = row["threshold"]
        status = row["status"]
        if status == "PASS":
            status = r"\textcolor{green!60!black}{PASS}"
        elif status == "FAIL":
            status = r"\textcolor{red}{FAIL}"
        lines.append(f"  {metric} & {value} & {threshold} & {status} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Saved LaTeX table: {tex_path}")
    return tex_path


def main(run_dir: str) -> None:
    """Generate all tables for an SDP run.

    Args:
        run_dir: Path to completed SDP run directory.
    """
    paths = load_run_paths(run_dir)
    paths.tables.mkdir(parents=True, exist_ok=True)

    generate_csv(paths)
    generate_latex(paths)

    logger.info("Table generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDP tables")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.run_dir)
