# src/growth/models/growth/base.py
"""
Abstract base class for growth prediction models.

All growth models (LME, H-GP, PA-MOGP) share the same interface:
- fit(patients) → train on a list of PatientTrajectory objects
- predict(patient, t_pred) → return (mean, std) predictions at given times
"""
