"""Output writing utilities for analysis pipeline.

Handles writing CSVs, JSONs, and organizing output directories.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def setup_output_directory(
    run_dir: str,
    output_subdir: str = "analysis",
) -> Path:
    """Create output directory structure for analysis results.

    Args:
        run_dir: Path to experiment run directory
        output_subdir: Subdirectory name for analysis outputs

    Returns:
        Path to analysis output directory
    """
    run_path = Path(run_dir)
    output_path = run_path / output_subdir

    # Create subdirectories
    (output_path / "plots").mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_path}")
    return output_path


def write_summary_json(
    summary: Any,  # AnalysisSummary
    output_dir: Union[str, Path],
    filename: str = "summary.json",
) -> str:
    """Write analysis summary to JSON file.

    Args:
        summary: AnalysisSummary dataclass
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to written file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename

    # Convert to dict
    data = summary.to_dict()

    # Add metadata
    data["_generated_at"] = datetime.now().isoformat()
    data["_generator"] = "vae.tools.analysis"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Summary written to {filepath}")
    return str(filepath)


def write_metrics_csv(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    filename: str,
) -> str:
    """Write metrics DataFrame to CSV.

    Args:
        df: DataFrame to write
        output_dir: Output directory
        filename: Output filename (should end with .csv)

    Returns:
        Path to written file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    df.to_csv(filepath, index=False)

    logger.info(f"Metrics written to {filepath}")
    return str(filepath)


def write_comparison_json(
    comparison: Any,  # ComparisonSummary
    output_dir: Union[str, Path],
    filename: str = "comparison_summary.json",
) -> str:
    """Write comparison summary to JSON file.

    Args:
        comparison: ComparisonSummary dataclass
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to written file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename

    data = comparison.to_dict()
    data["_generated_at"] = datetime.now().isoformat()
    data["_generator"] = "vae.tools.analysis"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Comparison written to {filepath}")
    return str(filepath)


def collect_output_manifest(
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Collect manifest of all output files.

    Args:
        output_dir: Analysis output directory

    Returns:
        Dictionary with file manifest
    """
    output_path = Path(output_dir)

    manifest = {
        "output_dir": str(output_path.absolute()),
        "generated_at": datetime.now().isoformat(),
        "files": {
            "stage1": [],
            "stage2": [],
        },
    }

    # Stage 1 files (CSVs and JSONs in root)
    for ext in ["*.csv", "*.json"]:
        for f in output_path.glob(ext):
            manifest["files"]["stage1"].append({
                "name": f.name,
                "path": str(f.relative_to(output_path)),
                "size_bytes": f.stat().st_size,
            })

    # Stage 2 files (plots)
    plots_dir = output_path / "plots"
    if plots_dir.exists():
        for ext in ["*.png", "*.pdf"]:
            for f in plots_dir.glob(ext):
                manifest["files"]["stage2"].append({
                    "name": f.name,
                    "path": str(f.relative_to(output_path)),
                    "size_bytes": f.stat().st_size,
                })

    return manifest


def write_manifest(
    output_dir: Union[str, Path],
    manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """Write output manifest to JSON.

    Args:
        output_dir: Analysis output directory
        manifest: Pre-computed manifest (if None, will collect)

    Returns:
        Path to manifest file
    """
    if manifest is None:
        manifest = collect_output_manifest(output_dir)

    output_path = Path(output_dir)
    filepath = output_path / "manifest.json"

    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2)

    return str(filepath)
