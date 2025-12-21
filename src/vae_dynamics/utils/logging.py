"""Logging utilities.

This module provides functions for configuring Python's logging system
to output to both console and file.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    run_dir: Optional[str] = None,
    log_filename: str = "train.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logging to console and optionally to file.

    Args:
        run_dir: Path to run directory. If provided, logs will also be
                 written to <run_dir>/<log_filename>.
        log_filename: Name of log file within run_dir.
        level: Logging level (default: INFO).

    Returns:
        Root logger instance.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if run_dir provided)
    if run_dir is not None:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / log_filename

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to {log_file}")

    return root_logger
