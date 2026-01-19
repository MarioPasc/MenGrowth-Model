"""VAE training callbacks module.

Exports all callback classes for VAE training:
- Core callbacks: Reconstruction visualization and console logging
- Metrics callbacks: Unified CSV logging and run metadata
- Diagnostic callbacks: Latent diagnostics and Active Units
- SemiVAE callbacks: Partition diagnostics and semantic tracking
"""

from .core_callbacks import (
    ReconstructionCallback,
    TrainingLoggingCallback,
)
from .metrics_csv_callback import (
    UnifiedCSVCallback,
    RunMetadataCallback,
)
from .diagnostics_callbacks import (
    LatentDiagnosticsCallback,
    ActiveUnitsCallback,
    GradientStatsCallback,
)
from .semivae_callbacks import (
    SemiVAEDiagnosticsCallback,
    SemiVAELatentVisualizationCallback,
    SemiVAESemanticTrackingCallback,
)

__all__ = [
    # Core
    "ReconstructionCallback",
    "TrainingLoggingCallback",
    # Metrics/CSV
    "UnifiedCSVCallback",
    "RunMetadataCallback",
    # Diagnostics
    "LatentDiagnosticsCallback",
    "ActiveUnitsCallback",
    "GradientStatsCallback",
    # SemiVAE
    "SemiVAEDiagnosticsCallback",
    "SemiVAELatentVisualizationCallback",
    "SemiVAESemanticTrackingCallback",
]
