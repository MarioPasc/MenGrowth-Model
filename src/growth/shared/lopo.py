# src/growth/shared/lopo.py
"""Leave-One-Patient-Out (LOPO) cross-validation evaluator.

Runs LOPO-CV for a given growth model class, computes prediction metrics
(R^2, MAE, RMSE, calibration, per-patient correlation) under configurable
prediction protocols, and returns structured results for serialization.

This is the canonical location. Backward-compatible re-exports exist at
``growth.evaluation.lopo_evaluator``.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

from growth.shared.growth_models import (
    FitResult,
    GrowthModel,
    PatientTrajectory,
)

logger = logging.getLogger(__name__)


@dataclass
class LOPOFoldResult:
    """Results from a single LOPO fold (one held-out patient).

    Args:
        patient_id: Held-out patient identifier.
        n_timepoints: Number of timepoints for this patient.
        n_train_patients: Number of training patients in this fold.
        n_train_observations: Total training observations in this fold.
        fit_result: Model fitting diagnostics for this fold.
        predictions: Per-protocol prediction results. Maps protocol name to
            list of dicts with keys ``time``, ``pred_mean``, ``pred_var``,
            ``actual``, ``lower_95``, ``upper_95``.
        fit_time_s: Wall-clock time for model fitting (seconds).
    """

    patient_id: str
    n_timepoints: int
    n_train_patients: int
    n_train_observations: int
    fit_result: FitResult
    predictions: dict[str, list[dict]] = field(default_factory=dict)
    fit_time_s: float = 0.0


@dataclass
class LOPOResults:
    """Aggregated results from full LOPO-CV.

    Args:
        model_name: Name of the evaluated model.
        fold_results: Per-fold results.
        aggregate_metrics: Dict of aggregated metric names to values.
        failed_folds: Patient IDs of folds that failed.
    """

    model_name: str
    fold_results: list[LOPOFoldResult]
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    failed_folds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "model_name": self.model_name,
            "n_folds": len(self.fold_results),
            "n_failed": len(self.failed_folds),
            "failed_folds": self.failed_folds,
            "aggregate_metrics": self.aggregate_metrics,
            "fold_results": [
                {
                    "patient_id": fr.patient_id,
                    "n_timepoints": fr.n_timepoints,
                    "n_train_patients": fr.n_train_patients,
                    "n_train_observations": fr.n_train_observations,
                    "fit_result": {
                        "log_marginal_likelihood": fr.fit_result.log_marginal_likelihood,
                        "hyperparameters": fr.fit_result.hyperparameters,
                        "condition_number": fr.fit_result.condition_number,
                    },
                    "predictions": fr.predictions,
                    "fit_time_s": fr.fit_time_s,
                }
                for fr in self.fold_results
            ],
        }


class LOPOEvaluator:
    """Leave-One-Patient-Out cross-validation evaluator.

    Evaluates a growth model by holding out each patient in turn, fitting
    on the remaining patients, and predicting the held-out patient's
    observations.

    Args:
        prediction_protocols: List of protocol names to evaluate.
            Supported: ``"last_from_rest"`` (predict last from first N-1),
            ``"all_from_first"`` (predict all remaining from first 1).
            If None, defaults to both.
    """

    def __init__(
        self,
        prediction_protocols: list[str] | None = None,
    ) -> None:
        self.prediction_protocols = prediction_protocols or [
            "last_from_rest",
            "all_from_first",
        ]

    def evaluate(
        self,
        model_class: type[GrowthModel],
        patients: list[PatientTrajectory],
        **model_kwargs,
    ) -> LOPOResults:
        """Run full LOPO-CV.

        Args:
            model_class: GrowthModel subclass to instantiate per fold.
            patients: All patient trajectories.
            **model_kwargs: Keyword arguments passed to ``model_class()``.

        Returns:
            LOPOResults with per-fold and aggregated metrics.
        """
        n_patients = len(patients)
        logger.info(
            f"LOPO-CV: {n_patients} folds, model={model_class.__name__}, "
            f"protocols={self.prediction_protocols}"
        )

        fold_results: list[LOPOFoldResult] = []
        failed_folds: list[str] = []

        for i, held_out in enumerate(patients):
            train_patients = [p for j, p in enumerate(patients) if j != i]

            logger.debug(
                f"Fold {i + 1}/{n_patients}: held_out={held_out.patient_id} "
                f"(n={held_out.n_timepoints})"
            )

            try:
                fold_result = self._run_fold(model_class, train_patients, held_out, **model_kwargs)
                fold_results.append(fold_result)
            except Exception as e:
                logger.warning(f"Fold {i + 1} failed for {held_out.patient_id}: {e}")
                failed_folds.append(held_out.patient_id)

        # Instantiate a model just to get its name
        tmp_model = model_class(**model_kwargs)
        model_name = tmp_model.name()

        aggregate_metrics = self._compute_aggregate_metrics(fold_results)

        results = LOPOResults(
            model_name=model_name,
            fold_results=fold_results,
            aggregate_metrics=aggregate_metrics,
            failed_folds=failed_folds,
        )

        logger.info(
            f"LOPO-CV complete: {len(fold_results)}/{n_patients} folds succeeded. "
            f"Metrics: {aggregate_metrics}"
        )

        return results

    def _run_fold(
        self,
        model_class: type[GrowthModel],
        train_patients: list[PatientTrajectory],
        held_out: PatientTrajectory,
        **model_kwargs,
    ) -> LOPOFoldResult:
        """Run a single LOPO fold."""
        model = model_class(**model_kwargs)

        t0 = time.monotonic()
        fit_result = model.fit(train_patients)
        fit_time = time.monotonic() - t0

        n_train_obs = sum(p.n_timepoints for p in train_patients)

        predictions: dict[str, list[dict]] = {}

        for protocol in self.prediction_protocols:
            preds = self._apply_protocol(model, held_out, protocol)
            if preds is not None:
                predictions[protocol] = preds

        return LOPOFoldResult(
            patient_id=held_out.patient_id,
            n_timepoints=held_out.n_timepoints,
            n_train_patients=len(train_patients),
            n_train_observations=n_train_obs,
            fit_result=fit_result,
            predictions=predictions,
            fit_time_s=fit_time,
        )

    def _apply_protocol(
        self,
        model: GrowthModel,
        patient: PatientTrajectory,
        protocol: str,
    ) -> list[dict] | None:
        """Apply a prediction protocol to a held-out patient."""
        if protocol == "last_from_rest":
            return self._protocol_last_from_rest(model, patient)
        elif protocol == "all_from_first":
            return self._protocol_all_from_first(model, patient)
        else:
            logger.warning(f"Unknown protocol: {protocol}")
            return None

    def _protocol_last_from_rest(
        self, model: GrowthModel, patient: PatientTrajectory
    ) -> list[dict]:
        """Predict last timepoint from all preceding observations."""
        n = patient.n_timepoints
        assert n >= 2, f"last_from_rest needs >=2 timepoints, got {n}"

        t_pred = np.array([patient.times[-1]])
        pred = model.predict(patient, t_pred, n_condition=n - 1)

        return [
            {
                "time": float(patient.times[-1]),
                "pred_mean": float(pred.mean[0, 0]),
                "pred_var": float(pred.variance[0, 0]),
                "actual": float(patient.observations[-1, 0]),
                "lower_95": float(pred.lower_95[0, 0]),
                "upper_95": float(pred.upper_95[0, 0]),
                "n_conditioning": n - 1,
            }
        ]

    def _protocol_all_from_first(
        self, model: GrowthModel, patient: PatientTrajectory
    ) -> list[dict] | None:
        """Predict all remaining timepoints from first observation only."""
        if patient.n_timepoints < 3:
            return None

        t_pred = patient.times[1:]
        pred = model.predict(patient, t_pred, n_condition=1)

        results: list[dict] = []
        for j in range(len(t_pred)):
            results.append(
                {
                    "time": float(t_pred[j]),
                    "pred_mean": float(pred.mean[j, 0]),
                    "pred_var": float(pred.variance[j, 0]),
                    "actual": float(patient.observations[j + 1, 0]),
                    "lower_95": float(pred.lower_95[j, 0]),
                    "upper_95": float(pred.upper_95[j, 0]),
                    "n_conditioning": 1,
                }
            )
        return results

    def _compute_aggregate_metrics(self, fold_results: list[LOPOFoldResult]) -> dict[str, float]:
        """Compute aggregated metrics across all folds."""
        from growth.shared.metrics import compute_calibration, compute_mae, compute_r2, compute_rmse

        metrics: dict[str, float] = {}

        for protocol in self.prediction_protocols:
            all_pred: list[float] = []
            all_actual: list[float] = []
            all_lower: list[float] = []
            all_upper: list[float] = []
            all_ci_width: list[float] = []

            for fr in fold_results:
                if protocol not in fr.predictions:
                    continue
                for pred_dict in fr.predictions[protocol]:
                    all_pred.append(pred_dict["pred_mean"])
                    all_actual.append(pred_dict["actual"])
                    all_lower.append(pred_dict["lower_95"])
                    all_upper.append(pred_dict["upper_95"])
                    all_ci_width.append(pred_dict["upper_95"] - pred_dict["lower_95"])

            if len(all_pred) == 0:
                continue

            pred_arr = np.array(all_pred)
            actual_arr = np.array(all_actual)
            lower_arr = np.array(all_lower)
            upper_arr = np.array(all_upper)

            prefix = protocol

            # Log-space metrics (native space for log1p volumes)
            metrics[f"{prefix}/r2_log"] = compute_r2(actual_arr, pred_arr)
            metrics[f"{prefix}/mae_log"] = compute_mae(actual_arr, pred_arr)
            metrics[f"{prefix}/rmse_log"] = compute_rmse(actual_arr, pred_arr)

            # Original-space metrics (expm1 to undo log1p)
            pred_orig = np.expm1(np.maximum(pred_arr, 0.0))
            actual_orig = np.expm1(np.maximum(actual_arr, 0.0))
            metrics[f"{prefix}/r2_original"] = compute_r2(actual_orig, pred_orig)
            metrics[f"{prefix}/mae_original"] = compute_mae(actual_orig, pred_orig)
            metrics[f"{prefix}/rmse_original"] = compute_rmse(actual_orig, pred_orig)

            # Calibration
            metrics[f"{prefix}/calibration_95"] = compute_calibration(
                actual_arr, lower_arr, upper_arr
            )

            # Mean CI width
            metrics[f"{prefix}/mean_ci_width_log"] = float(np.mean(all_ci_width))

        # Per-patient Pearson r
        if "all_from_first" in self.prediction_protocols:
            patient_rs: list[float] = []
            for fr in fold_results:
                if "all_from_first" not in fr.predictions:
                    continue
                preds_list = fr.predictions["all_from_first"]
                if len(preds_list) < 2:
                    continue
                p_pred = np.array([d["pred_mean"] for d in preds_list])
                p_actual = np.array([d["actual"] for d in preds_list])
                if np.std(p_actual) < 1e-10 or np.std(p_pred) < 1e-10:
                    continue
                r, _ = scipy_stats.pearsonr(p_pred, p_actual)
                if np.isfinite(r):
                    patient_rs.append(float(r))

            if patient_rs:
                metrics["per_patient_r_mean"] = float(np.mean(patient_rs))
                metrics["per_patient_r_std"] = float(np.std(patient_rs))
                metrics["per_patient_r_n"] = float(len(patient_rs))

        return metrics

    @staticmethod
    def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Compute R^2 (coefficient of determination).

        Delegates to ``growth.shared.metrics.compute_r2``. Kept for backward
        compatibility with tests that call ``LOPOEvaluator._r2()`` directly.
        """
        from growth.shared.metrics import compute_r2

        return compute_r2(actual, predicted)
