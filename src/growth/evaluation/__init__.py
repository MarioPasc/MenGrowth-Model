# src/growth/evaluation/__init__.py
"""
Evaluation module for the growth forecasting pipeline.

Provides metrics for latent quality, ODE predictions, and clinical risk assessment.

Components:
- latent_quality: Linear probes, R² metrics, correlation analysis
- enhanced_probes: MLP probes, target normalization, nonlinearity analysis
"""

from .latent_quality import (
    ProbeResults,
    LinearProbe,
    SemanticProbes,
    compute_r2_scores,
    compute_cross_correlation,
    compute_partition_correlation,
    distance_correlation,
    compute_dcor_matrix,
    compute_variance_per_dim,
    evaluate_latent_quality,
)

from .enhanced_probes import (
    EnhancedProbeResults,
    EnhancedLinearProbe,
    MLPProbe,
    EnhancedSemanticProbes,
    analyze_feature_quality,
    compute_multi_scale_features,
)

__all__ = [
    # Basic probes
    "ProbeResults",
    "LinearProbe",
    "SemanticProbes",
    "compute_r2_scores",
    "compute_cross_correlation",
    "compute_partition_correlation",
    "distance_correlation",
    "compute_dcor_matrix",
    "compute_variance_per_dim",
    "evaluate_latent_quality",
    # Enhanced probes
    "EnhancedProbeResults",
    "EnhancedLinearProbe",
    "MLPProbe",
    "EnhancedSemanticProbes",
    "analyze_feature_quality",
    "compute_multi_scale_features",
]
