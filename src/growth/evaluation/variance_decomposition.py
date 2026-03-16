# src/growth/evaluation/variance_decomposition.py
"""Variance decomposition across the 3-stage complexity ladder.

Quantifies the marginal ΔR² contributed by each model/stage transition,
with paired permutation tests for statistical significance and BCa
bootstrap CIs. This is the central analytical contribution of the thesis.

Model hierarchy:
    M₀: Population mean only (2 params)
    M₁: ScalarGP (5 params)
    M₂: HGP (6-8 params)
    M₃: Severity model (3+N params)
    M₄: Deep features + severity (3+N+k params)

Spec: ``docs/stages/variance_decomposition.md``
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from growth.shared.bootstrap import BootstrapResult, bootstrap_metric, paired_permutation_test
from growth.shared.metrics import compute_mae, compute_r2

logger = logging.getLogger(__name__)


@dataclass
class DeltaR2Result:
    """Result of a single ΔR² comparison between two models.

    Args:
        model_simple: Name of the simpler model.
        model_complex: Name of the more complex model.
        r2_simple: R² of the simpler model.
        r2_complex: R² of the more complex model.
        delta_r2: R²_complex - R²_simple.
        p_value: p-value from paired permutation test.
        ci: Bootstrap CI for ΔR².
        significant: Whether ΔR² is significant at α=0.05.
    """

    model_simple: str
    model_complex: str
    r2_simple: float
    r2_complex: float
    delta_r2: float
    p_value: float
    ci: BootstrapResult | None = None
    significant: bool = False


@dataclass
class VarianceDecompositionResult:
    """Full variance decomposition across all models.

    Args:
        models: Ordered list of model names (from simplest to most complex).
        r2_values: R² for each model.
        mae_values: MAE for each model.
        transitions: List of DeltaR2Result for each consecutive pair.
        per_patient_errors: Dict mapping model_name → per-patient error array.
    """

    models: list[str] = field(default_factory=list)
    r2_values: list[float] = field(default_factory=list)
    mae_values: list[float] = field(default_factory=list)
    transitions: list[DeltaR2Result] = field(default_factory=list)
    per_patient_errors: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "models": self.models,
            "r2_values": self.r2_values,
            "mae_values": self.mae_values,
            "transitions": [
                {
                    "from": t.model_simple,
                    "to": t.model_complex,
                    "delta_r2": t.delta_r2,
                    "p_value": t.p_value,
                    "significant": t.significant,
                    "r2_simple": t.r2_simple,
                    "r2_complex": t.r2_complex,
                }
                for t in self.transitions
            ],
        }


class VarianceDecomposition:
    """Compute variance decomposition across nested growth models.

    Given per-patient LOPO-CV predictions from multiple models (ordered
    by complexity), computes ΔR² for each transition with statistical tests.

    Args:
        n_permutations: Number of permutations for significance testing.
        n_bootstrap: Number of bootstrap samples for CI estimation.
        alpha: Significance level.
        seed: Random seed.
    """

    def __init__(
        self,
        n_permutations: int = 10000,
        n_bootstrap: int = 2000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.seed = seed

    def decompose(
        self,
        y_true: np.ndarray,
        model_predictions: dict[str, np.ndarray],
        model_order: list[str] | None = None,
    ) -> VarianceDecompositionResult:
        """Run variance decomposition on aligned predictions.

        Args:
            y_true: Ground truth values, shape ``[N_patients]``.
            model_predictions: Dict mapping model name → predicted values
                ``[N_patients]``. All must have the same length as y_true.
            model_order: Order of models from simplest to most complex.
                If None, uses insertion order of model_predictions.

        Returns:
            VarianceDecompositionResult with ΔR² per transition.
        """
        if model_order is None:
            model_order = list(model_predictions.keys())

        y_true = np.asarray(y_true)

        # Compute per-model metrics
        r2_values: list[float] = []
        mae_values: list[float] = []
        per_patient_errors: dict[str, np.ndarray] = {}

        for model_name in model_order:
            y_pred = np.asarray(model_predictions[model_name])
            assert len(y_pred) == len(y_true), (
                f"Model {model_name}: prediction length {len(y_pred)} != "
                f"y_true length {len(y_true)}"
            )
            r2_values.append(compute_r2(y_true, y_pred))
            mae_values.append(compute_mae(y_true, y_pred))
            per_patient_errors[model_name] = np.abs(y_true - y_pred)

        # Compute transitions
        transitions: list[DeltaR2Result] = []
        for i in range(len(model_order) - 1):
            name_simple = model_order[i]
            name_complex = model_order[i + 1]

            errors_simple = per_patient_errors[name_simple]
            errors_complex = per_patient_errors[name_complex]

            delta_r2 = r2_values[i + 1] - r2_values[i]

            # Permutation test
            ptest = paired_permutation_test(
                errors_simple,
                errors_complex,
                n_permutations=self.n_permutations,
                seed=self.seed + i,
            )

            # Bootstrap CI for ΔR²
            def delta_r2_fn(y_t: np.ndarray, idx: np.ndarray) -> float:
                """Compute ΔR² for a bootstrap sample."""
                y_s = np.asarray(model_predictions[name_simple])[idx.astype(int)]
                y_c = np.asarray(model_predictions[name_complex])[idx.astype(int)]
                return compute_r2(y_t, y_c) - compute_r2(y_t, y_s)

            ci = bootstrap_metric(
                y_true,
                np.arange(len(y_true), dtype=np.float64),
                delta_r2_fn,
                n_bootstrap=self.n_bootstrap,
                seed=self.seed + i + 100,
            )

            transitions.append(
                DeltaR2Result(
                    model_simple=name_simple,
                    model_complex=name_complex,
                    r2_simple=r2_values[i],
                    r2_complex=r2_values[i + 1],
                    delta_r2=delta_r2,
                    p_value=ptest.p_value,
                    ci=ci,
                    significant=ptest.p_value < self.alpha,
                )
            )

            logger.info(
                f"Transition {name_simple} → {name_complex}: "
                f"ΔR²={delta_r2:+.4f}, p={ptest.p_value:.4f}, "
                f"{'SIGNIFICANT' if ptest.p_value < self.alpha else 'not significant'}"
            )

        return VarianceDecompositionResult(
            models=model_order,
            r2_values=r2_values,
            mae_values=mae_values,
            transitions=transitions,
            per_patient_errors=per_patient_errors,
        )
