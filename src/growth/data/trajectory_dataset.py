# src/growth/data/trajectory_dataset.py
"""
Trajectory dataset for GP-based growth prediction.

Loads trajectories.json and organizes into PatientTrajectory dataclass objects.
Trajectories consist of 32-dim volume latent vectors (z_vol) per timepoint,
with location centroid as static covariate.

Supports LOPO-CV splits via lopo_split(held_out_id).
GP models operate on observation-level data directly (no transition pairs).
"""
