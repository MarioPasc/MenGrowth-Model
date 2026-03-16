# src/growth/data/trajectory_dataset.py
"""Backward-compatible re-exports.

Canonical location: ``growth.shared.trajectory_io``.
"""

from growth.shared.trajectory_io import (
    load_trajectories,
    save_trajectories,
    trajectories_to_dataframe,
)

__all__ = ["load_trajectories", "save_trajectories", "trajectories_to_dataframe"]
