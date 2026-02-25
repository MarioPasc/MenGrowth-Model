# src/growth/evaluation/__init__.py
"""
Evaluation module for the growth forecasting pipeline.

Provides metrics for latent quality, ODE predictions, and clinical risk assessment.

Components:
- latent_quality: Linear probes, R² metrics, correlation analysis
- enhanced_probes: MLP probes, target normalization, nonlinearity analysis
- statistics: Bootstrap CIs, effect sizes, statistical tests
- segmentation_metrics: Dice evaluation utilities
- visualization: Publication-quality plots
"""

from .enhanced_probes import (
    EnhancedLinearProbe,
    EnhancedProbeResults,
    EnhancedSemanticProbes,
    MLPProbe,
    analyze_feature_quality,
    compute_multi_scale_features,
)
from .latent_quality import (
    DCIResults,
    DomainShiftMetrics,
    LinearProbe,
    ProbeResults,
    SemanticProbes,
    compute_cka,
    compute_cross_correlation,
    compute_dci,
    compute_dcor_matrix,
    compute_domain_classifier_accuracy,
    compute_domain_shift_metrics,
    # Domain shift metrics
    compute_effective_rank,
    compute_mmd,
    compute_partition_correlation,
    compute_proxy_a_distance,
    compute_r2_scores,
    compute_variance_per_dim,
    distance_correlation,
    evaluate_latent_quality,
    mmd_permutation_test,
)
from .segmentation_metrics import (
    SegmentationEvaluator,
    compute_dice_coefficient,
    compute_per_class_dice,
)
from .statistics import (
    BootstrapCI,
    PairedTestResult,
    StatisticalTest,
    bonferroni_correction,
    bootstrap_ci,
    bootstrap_delta_ci,
    cohens_d,
    holm_bonferroni_correction,
    interpret_cohens_d,
    paired_statistical_test,
)
from .visualization import (
    HAS_MATPLOTLIB,
    HAS_SEABORN,
    HAS_UMAP,
    plot_correlation_matrix,
    plot_prediction_scatter,
    plot_r2_comparison,
    plot_umap,
    plot_variance_spectrum,
    save_figure,
    set_publication_style,
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
    # DCI disentanglement
    "DCIResults",
    "compute_dci",
    # Domain shift metrics
    "compute_effective_rank",
    "compute_mmd",
    "compute_cka",
    "mmd_permutation_test",
    "compute_domain_classifier_accuracy",
    "compute_proxy_a_distance",
    "DomainShiftMetrics",
    "compute_domain_shift_metrics",
    # Enhanced probes
    "EnhancedProbeResults",
    "EnhancedLinearProbe",
    "MLPProbe",
    "EnhancedSemanticProbes",
    "analyze_feature_quality",
    "compute_multi_scale_features",
    # Statistics
    "BootstrapCI",
    "PairedTestResult",
    "StatisticalTest",
    "bootstrap_ci",
    "bootstrap_delta_ci",
    "cohens_d",
    "interpret_cohens_d",
    "paired_statistical_test",
    "holm_bonferroni_correction",
    "bonferroni_correction",
    # Segmentation metrics
    "SegmentationEvaluator",
    "compute_dice_coefficient",
    "compute_per_class_dice",
    # Visualization
    "set_publication_style",
    "save_figure",
    "plot_umap",
    "plot_variance_spectrum",
    "plot_prediction_scatter",
    "plot_r2_comparison",
    "plot_correlation_matrix",
    "HAS_MATPLOTLIB",
    "HAS_SEABORN",
    "HAS_UMAP",
]
