"""Pydantic schemas for experiment analysis validation.

These schemas define the structure of analysis outputs and provide
validation for experiment data consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ODEReadinessGrade(str, Enum):
    """ODE readiness grade classification."""
    A = "A"  # All targets met (vol R² >= 0.85, loc R² >= 0.90, cross_corr < 0.30)
    B = "B"  # Vol + Loc pass
    C = "C"  # Vol passes only
    D = "D"  # Vol close (>= 0.70)
    F = "F"  # Vol fails (< 0.70)


@dataclass
class ExperimentMetadata:
    """Metadata about the experiment run."""
    run_id: str = ""
    run_dir: str = ""
    experiment_type: str = "unknown"  # "semivae", "dipvae", etc.
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "unknown"
    max_epochs: int = 0
    completed_epochs: int = 0

    # Hardware info
    gpu_model: Optional[str] = None
    gpu_count: int = 1

    # Key config parameters
    z_dim: int = 128
    batch_size: int = 2
    learning_rate: float = 1e-4

    # Partition dimensions
    z_vol_dim: int = 0
    z_loc_dim: int = 0
    z_shape_dim: int = 0
    z_residual_dim: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics at a specific epoch or aggregate."""
    epoch: int = -1  # -1 for aggregate/final

    # Reconstruction quality
    recon_mse: float = 0.0
    recon_mse_per_modality: Dict[str, float] = field(default_factory=dict)
    ssim_mean: float = 0.0
    ssim_per_modality: Dict[str, float] = field(default_factory=dict)
    psnr_mean: float = 0.0
    psnr_per_modality: Dict[str, float] = field(default_factory=dict)

    # Semantic encoding quality (R²)
    vol_r2: float = 0.0
    loc_r2: float = 0.0
    shape_r2: float = 0.0

    # Semantic MSE (raw)
    vol_mse: float = 0.0
    loc_mse: float = 0.0
    shape_mse: float = 0.0


@dataclass
class CollapseMetrics:
    """Metrics for detecting posterior/decoder collapse."""
    epoch: int = -1

    # Active Units
    au_count_total: int = 0
    au_frac_total: float = 0.0
    au_count_residual: int = 0
    au_frac_residual: float = 0.0

    # Per-partition variance
    z_vol_var: float = 0.0
    z_loc_var: float = 0.0
    z_shape_var: float = 0.0
    z_residual_var: float = 0.0

    # Decoder bypass test
    # If (recon_z0_mse - recon_mu_mse) / recon_mu_mse < 0.20, decoder ignores z
    recon_mu_mse: float = 0.0
    recon_z0_mse: float = 0.0
    decoder_bypass_ratio: float = 0.0
    decoder_bypassed: bool = False

    # KL diagnostics
    kl_raw_per_dim: float = 0.0
    kl_below_floor: bool = False

    # Variance trajectory analysis (z_residual)
    residual_var_peak: float = 0.0       # Maximum variance reached
    residual_var_peak_epoch: int = 0     # Epoch of peak variance
    residual_var_decline_pct: float = 0.0  # (peak - final) / peak
    residual_deflating: bool = False     # decline > 50%

    # Collapse flags
    residual_collapsed: bool = False  # AU_frac_residual < 0.10


@dataclass
class ODEUtilityMetrics:
    """Metrics for Neural ODE readiness assessment."""
    epoch: int = -1

    # Cross-partition correlations (absolute values)
    corr_vol_loc: float = 0.0
    corr_vol_shape: float = 0.0
    corr_loc_shape: float = 0.0
    max_cross_corr: float = 0.0

    # Factor independence score: 1 - max_cross_corr
    independence_score: float = 1.0

    # Semantic R² values (for ODE readiness calculation)
    vol_r2: float = 0.0
    loc_r2: float = 0.0
    shape_r2: float = 0.0

    # ODE readiness composite score (original)
    # Formula: 0.50 * vol_r2 + 0.25 * loc_r2 + 0.25 * independence_score
    ode_readiness: float = 0.0

    # Expanded ODE readiness composite score
    # Formula: 0.40*vol_r2 + 0.20*loc_r2 + 0.15*shape_r2 + 0.15*independence + 0.10*residual_health
    ode_readiness_expanded: float = 0.0
    residual_health: float = 0.0  # min(au_frac_residual / 0.10, 1.0)

    # Individual readiness flags
    vol_ready: bool = False   # vol_r2 >= 0.85
    loc_ready: bool = False   # loc_r2 >= 0.90
    shape_ready: bool = False # shape_r2 >= 0.35
    factors_independent: bool = False  # max_cross_corr < 0.30

    # Overall grade
    grade: ODEReadinessGrade = ODEReadinessGrade.F


@dataclass
class TrendMetrics:
    """Training dynamics and convergence metrics."""
    # Loss convergence
    loss_converged: bool = False
    loss_final: float = 0.0
    loss_best: float = 0.0
    loss_best_epoch: int = 0

    # Convergence rate (epochs to reach 90% of final performance)
    epochs_to_90pct: int = -1

    # Stability (variance in last N epochs)
    loss_stability: float = 0.0  # std/mean in last 50 epochs

    # Gradient health
    grad_norm_mean: float = 0.0
    grad_norm_max: float = 0.0
    grad_explosions: int = 0  # count of grad_norm > 10

    # Learning rate at end
    final_lr: float = 0.0


@dataclass
class StatisticalTestResults:
    """Results from statistical tests."""
    test_name: str = ""
    statistic: float = 0.0
    p_value: float = 0.0
    p_adjusted: float = 0.0  # FDR-corrected p-value
    significant: bool = False  # p_adjusted < 0.05
    effect_size: Optional[float] = None
    effect_interpretation: str = ""  # "negligible", "small", "medium", "large"
    confidence_interval: Optional[tuple] = None
    notes: str = ""


@dataclass
class AnalysisSummary:
    """Complete analysis summary for an experiment."""
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    # Final metrics (at best/last epoch)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    collapse: CollapseMetrics = field(default_factory=CollapseMetrics)
    ode_utility: ODEUtilityMetrics = field(default_factory=ODEUtilityMetrics)
    trends: TrendMetrics = field(default_factory=TrendMetrics)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Overall assessment
    overall_grade: ODEReadinessGrade = ODEReadinessGrade.F
    ready_for_ode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        import dataclasses

        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        return convert(self)


@dataclass
class ComparisonSummary:
    """Summary for multi-run comparison."""
    run_ids: List[str] = field(default_factory=list)

    # Run name mapping (folder_name -> display_name)
    name_map: Dict[str, str] = field(default_factory=dict)

    # Per-run summaries
    summaries: Dict[str, AnalysisSummary] = field(default_factory=dict)

    # Comparison metrics
    best_run_by_vol_r2: str = ""
    best_run_by_ode_readiness: str = ""
    best_run_overall: str = ""

    # Statistical test results
    test_results: List[StatisticalTestResults] = field(default_factory=list)

    # Stability metrics per run: {run_id: {metric_name: cv_value}}
    stability_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Bootstrap confidence intervals: {run_id: {metric_name: (lower, upper)}}
    confidence_intervals: Dict[str, Dict[str, tuple]] = field(default_factory=dict)

    # Convergence epochs: {run_id: {threshold_name: epoch}}
    convergence_epochs: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        import dataclasses

        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        return convert(self)
