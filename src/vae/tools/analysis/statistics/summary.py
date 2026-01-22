"""Summary generation for experiment analysis.

Aggregates metrics from all analysis modules into a comprehensive
summary.json file with grades, recommendations, and warnings.
"""

import logging
from typing import Dict, Any, List

from ..schemas import (
    AnalysisSummary,
    PerformanceMetrics,
    CollapseMetrics,
    ODEUtilityMetrics,
    TrendMetrics,
    ODEReadinessGrade,
)
from ..loaders import extract_experiment_metadata
from .performance import compute_performance_metrics
from .collapse import compute_collapse_metrics_with_trajectory
from .ode_utility import compute_ode_utility_metrics
from .trends import compute_trend_metrics

logger = logging.getLogger(__name__)


def generate_summary(
    data: Dict[str, Any],
    epoch: int = None,
) -> AnalysisSummary:
    """Generate comprehensive analysis summary.

    Args:
        data: Dictionary from load_experiment_data()
        epoch: Specific epoch to analyze (None = best/final)

    Returns:
        AnalysisSummary dataclass with all metrics and recommendations
    """
    summary = AnalysisSummary()

    # Extract metadata
    summary.metadata = extract_experiment_metadata(data)

    # Compute all metrics
    summary.performance = compute_performance_metrics(data, epoch=epoch)
    summary.collapse = compute_collapse_metrics_with_trajectory(data, epoch=epoch)
    summary.ode_utility = compute_ode_utility_metrics(
        data,
        epoch=epoch,
        vol_r2=summary.performance.vol_r2,
        loc_r2=summary.performance.loc_r2,
        shape_r2=summary.performance.shape_r2,
    )
    summary.trends = compute_trend_metrics(data)

    # Generate recommendations and warnings
    summary.recommendations = _generate_recommendations(summary)
    summary.warnings = _generate_warnings(summary)

    # Overall assessment
    summary.overall_grade = summary.ode_utility.grade
    summary.ready_for_ode = summary.overall_grade in [
        ODEReadinessGrade.A,
        ODEReadinessGrade.B,
    ]

    return summary


def _generate_recommendations(summary: AnalysisSummary) -> List[str]:
    """Generate actionable recommendations based on metrics.

    Args:
        summary: AnalysisSummary with computed metrics

    Returns:
        List of recommendation strings
    """
    recommendations = []
    perf = summary.performance
    collapse = summary.collapse
    ode = summary.ode_utility

    # Volume R² recommendations
    if perf.vol_r2 < 0.70:
        recommendations.append(
            f"Volume R² is low ({perf.vol_r2:.2f}). Consider: "
            "increasing lambda_vol, adding more training epochs, "
            "or checking volume feature normalization."
        )
    elif perf.vol_r2 < 0.85:
        recommendations.append(
            f"Volume R² ({perf.vol_r2:.2f}) is below target 0.85. "
            "Fine-tuning lambda_vol or extending training may help."
        )

    # Location R² recommendations
    if perf.loc_r2 < 0.80:
        recommendations.append(
            f"Location R² is low ({perf.loc_r2:.2f}). Consider: "
            "increasing lambda_loc or checking centroid computation."
        )

    # Shape R² recommendations
    if perf.shape_r2 < 0.25:
        recommendations.append(
            f"Shape R² is very low ({perf.shape_r2:.2f}). Consider: "
            "reducing number of shape features, using more robust shape metrics, "
            "or increasing z_shape dimensions."
        )

    # Factor independence
    if ode.max_cross_corr >= 0.50:
        recommendations.append(
            f"High cross-partition correlation ({ode.max_cross_corr:.2f}). "
            "Increase lambda_cross_partition (try 10.0+) or start cross-partition "
            "penalty earlier in training."
        )
    elif ode.max_cross_corr >= 0.30:
        recommendations.append(
            f"Cross-partition correlation ({ode.max_cross_corr:.2f}) above target 0.30. "
            "Moderate increase in lambda_cross_partition recommended."
        )

    # Residual collapse
    if collapse.residual_collapsed:
        recommendations.append(
            "Residual dimensions collapsed. Consider: "
            "reducing semantic loss weights, adding auxiliary reconstruction "
            "from residual only, or increasing kl_free_bits."
        )

    # Decoder bypass
    if collapse.decoder_bypassed:
        recommendations.append(
            "Decoder may be ignoring latent codes (z=0 test shows similar reconstruction). "
            "Consider: disabling SBD for initial training, increasing KL weight, "
            "or adding z-dependent auxiliary losses."
        )

    # Residual deflation (not collapsed but declining)
    if collapse.residual_deflating and not collapse.residual_collapsed:
        recommendations.append(
            f"Residual variance declined {collapse.residual_var_decline_pct:.0%} from peak. Consider: "
            "reducing semantic loss weights in late training, adding auxiliary z_residual-only "
            "reconstruction, or reducing supervised partition capacity to force overflow into residual."
        )

    # Training stability
    if summary.trends.grad_explosions > 10:
        recommendations.append(
            f"Detected {summary.trends.grad_explosions} gradient explosions. "
            "Consider: reducing learning rate, increasing gradient_clip_val, "
            "or checking for NaN in semantic targets."
        )

    return recommendations


def _generate_warnings(summary: AnalysisSummary) -> List[str]:
    """Generate warning messages for critical issues.

    Args:
        summary: AnalysisSummary with computed metrics

    Returns:
        List of warning strings
    """
    warnings = []
    perf = summary.performance
    collapse = summary.collapse
    ode = summary.ode_utility

    # Critical warnings
    if collapse.residual_collapsed and collapse.au_frac_residual < 0.05:
        warnings.append(
            "CRITICAL: Residual partition has near-complete collapse "
            f"({collapse.au_frac_residual:.1%} active). Model may be unusable for ODE."
        )

    if collapse.decoder_bypassed and collapse.decoder_bypass_ratio < 0.10:
        warnings.append(
            "CRITICAL: Decoder appears to completely ignore latent codes. "
            "Reconstructions may be template-based rather than sample-specific."
        )

    if perf.vol_r2 < 0.50:
        warnings.append(
            f"WARNING: Volume R² ({perf.vol_r2:.2f}) is very low. "
            "Volume dynamics will not be captured by Neural ODE."
        )

    if ode.max_cross_corr >= 0.70:
        warnings.append(
            f"WARNING: Very high factor entanglement ({ode.max_cross_corr:.2f}). "
            "Factors may not evolve independently in ODE integration."
        )

    # Residual deflation warning
    if collapse.residual_deflating:
        warnings.append(
            f"WARNING: Residual variance deflating ({collapse.residual_var_decline_pct:.0%} decline from peak). "
            f"Peak variance {collapse.residual_var_peak:.4f} at epoch {collapse.residual_var_peak_epoch}, "
            f"now {collapse.z_residual_var:.4f}. Semantic losses may be dominating information flow."
        )
    elif collapse.z_residual_var < 0.02 and collapse.residual_var_peak > 0.03:
        warnings.append(
            f"WARNING: Low residual variance ({collapse.z_residual_var:.4f}) despite earlier peak "
            f"({collapse.residual_var_peak:.4f}). Partition may be underutilized."
        )

    # Convergence warnings
    if not summary.trends.loss_converged:
        warnings.append(
            "WARNING: Training may not have converged. "
            "Consider extending training or checking for oscillations."
        )

    if summary.trends.loss_stability > 0.10:
        warnings.append(
            f"WARNING: Training is unstable (CV={summary.trends.loss_stability:.2%} in last 50 epochs). "
            "Consider reducing learning rate or adjusting schedule timing."
        )

    return warnings


def summarize_for_comparison(summaries: Dict[str, AnalysisSummary]) -> Dict[str, Any]:
    """Create a comparison-friendly summary of multiple runs.

    Args:
        summaries: Dictionary mapping run_id to AnalysisSummary

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "run_ids": list(summaries.keys()),
        "metrics": {},
        "rankings": {},
    }

    # Extract key metrics for each run
    for run_id, summary in summaries.items():
        comparison["metrics"][run_id] = {
            "vol_r2": summary.performance.vol_r2,
            "loc_r2": summary.performance.loc_r2,
            "shape_r2": summary.performance.shape_r2,
            "max_cross_corr": summary.ode_utility.max_cross_corr,
            "ode_readiness": summary.ode_utility.ode_readiness,
            "grade": summary.overall_grade.value,
            "residual_collapsed": summary.collapse.residual_collapsed,
            "decoder_bypassed": summary.collapse.decoder_bypassed,
        }

    # Compute rankings
    if summaries:
        sorted_by_vol = sorted(
            summaries.keys(),
            key=lambda k: summaries[k].performance.vol_r2,
            reverse=True,
        )
        sorted_by_ode = sorted(
            summaries.keys(),
            key=lambda k: summaries[k].ode_utility.ode_readiness,
            reverse=True,
        )
        comparison["rankings"]["by_vol_r2"] = sorted_by_vol
        comparison["rankings"]["by_ode_readiness"] = sorted_by_ode
        comparison["best_run"] = sorted_by_ode[0] if sorted_by_ode else None

    return comparison
