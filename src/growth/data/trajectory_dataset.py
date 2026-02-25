# src/growth/data/trajectory_dataset.py
"""
Trajectory dataset for GP-based growth prediction.

Loads trajectories.json and organizes into PatientTrajectory dataclass objects.
Supports LOPO-CV splits via lopo_split(held_out_id).

Unlike the former Neural ODE TrajectoryDataset (which generated pairwise
temporal combinations), this operates on observation-level data directly —
GP models do not require transition pairs.
"""
