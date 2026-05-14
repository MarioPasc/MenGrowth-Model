# src/growth/evaluation/conformal_lopo.py
"""Nested Leave-One-Patient-Out evaluation of conformal calibration layers.

The :class:`~growth.shared.lopo.LOPOEvaluator` evaluates a growth model's
*native* (parametric) intervals. Conformal calibration layers, however, must
be *calibrated* on data and then evaluated on units that were not used for
calibration. :class:`ConformalLOPOEvaluator` does this honestly with a nested
loop:

- **Outer loop** holds out one patient ``*`` at a time (the test unit).
- **Inner loop** runs a jackknife over the remaining ``N - 1`` training
  patients to produce leave-one-out residuals and predictions, which calibrate
  the conformal layers for fold ``*``.

For each held-out patient it produces, under the ``last_from_rest`` protocol
(predict the last timepoint from all preceding ones — one prediction per
patient, which keeps patient-level exchangeability clean):

- ``parametric``      — the base model's native Gaussian interval;
- ``jackknife_plus``  — Barber et al. (2021), from ``{mu_{-i}(x*) + r_i}``;
- ``cqr_norm``        — normalised conformity reusing the base model's
                        predictive ``sigma_hat``;
- ``cqr_proper``      — conformalised quantile regression on patient features.

Cost is ``O(N^2)`` base-model fits per call; for an LME at ``N ~ 54`` this is
a few minutes, and for the ensemble model it scales with the ensemble size M.

See ``docs/CONFORMAL_PATH_ANALYSIS.md`` and
``experiments/stage1_volumetric/conformal_calibration/DESIGN.md``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from growth.shared.conformal import (
    CQRCalibrator,
    NormalizedConformalCalibrator,
    beta_binomial_coverage_ci,
    jackknife_plus_interval,
)
from growth.shared.growth_models import GrowthModel, PatientTrajectory
from growth.shared.metrics import (
    compute_calibration,
    compute_interval_score,
    compute_r2,
)

logger = logging.getLogger(__name__)

ALL_LAYERS: tuple[str, ...] = ("parametric", "jackknife_plus", "cqr_norm", "cqr_proper")

CQRFeatureFn = Callable[[PatientTrajectory], np.ndarray]


class ConformalLOPOError(Exception):
    """Raised when the conformal LOPO evaluator is misconfigured or misused."""


@dataclass
class ConformalLOPOFoldResult:
    """Per-patient result of the nested conformal LOPO evaluation.

    Attributes
    ----------
    patient_id : str
        Held-out patient identifier.
    time : float
        Query time of the ``last_from_rest`` target.
    actual : float
        Observed value at the query time.
    n_condition : int
        Number of observations conditioned on.
    sigma_v_sq_target : float
        Per-target measurement variance, for downstream tertile stratification
        (``nan`` if the trajectory carries no ``observation_variance``).
    parametric_mean : float
        Base-model point prediction (shared by every calibration layer).
    parametric_var : float
        Base-model predictive variance.
    intervals : dict[str, tuple[float, float]]
        ``layer -> (lower, upper)`` for every evaluated calibration layer.
    """

    patient_id: str
    time: float
    actual: float
    n_condition: int
    sigma_v_sq_target: float
    parametric_mean: float
    parametric_var: float
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class ConformalLOPOResults:
    """Aggregated results of a nested conformal LOPO evaluation."""

    model_name: str
    alpha: float
    layers: list[str]
    fold_results: list[ConformalLOPOFoldResult]
    failed_folds: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def aggregate_metrics(self) -> dict[str, float]:
        """Compute per-layer IS, coverage, mean width plus the shared R^2_log.

        Returns
        -------
        dict[str, float]
            Keys ``{layer}/is_95``, ``{layer}/coverage_95``,
            ``{layer}/coverage_95_ci_low``, ``{layer}/coverage_95_ci_high``,
            ``{layer}/mean_width``, and the layer-independent ``r2_log``.
        """
        if not self.fold_results:
            return {}
        actual = np.array([fr.actual for fr in self.fold_results])
        pred = np.array([fr.parametric_mean for fr in self.fold_results])
        metrics: dict[str, float] = {"r2_log": compute_r2(actual, pred)}

        for layer in self.layers:
            lower = np.array([fr.intervals[layer][0] for fr in self.fold_results])
            upper = np.array([fr.intervals[layer][1] for fr in self.fold_results])
            n = len(actual)
            n_covered = int(np.sum((actual >= lower) & (actual <= upper)))
            cov = compute_calibration(actual, lower, upper)
            ci_low, ci_high = beta_binomial_coverage_ci(n_covered, n, confidence=0.95)
            metrics[f"{layer}/is_95"] = compute_interval_score(
                actual, lower, upper, alpha=self.alpha
            )
            metrics[f"{layer}/coverage_95"] = cov
            metrics[f"{layer}/coverage_95_ci_low"] = ci_low
            metrics[f"{layer}/coverage_95_ci_high"] = ci_high
            metrics[f"{layer}/mean_width"] = float(np.mean(upper - lower))
        return metrics

    def per_patient_table(self) -> list[dict]:
        """Return a long-form per-patient/per-layer record list (one row per fold/layer)."""
        rows: list[dict] = []
        for fr in self.fold_results:
            for layer in self.layers:
                lo, hi = fr.intervals[layer]
                rows.append(
                    {
                        "patient_id": fr.patient_id,
                        "layer": layer,
                        "time": fr.time,
                        "actual": fr.actual,
                        "pred_mean": fr.parametric_mean,
                        "pred_var": fr.parametric_var,
                        "lower": lo,
                        "upper": hi,
                        "width": hi - lo,
                        "covered": bool(fr.actual >= lo and fr.actual <= hi),
                        "interval_score": compute_interval_score(
                            np.array([fr.actual]),
                            np.array([lo]),
                            np.array([hi]),
                            alpha=self.alpha,
                        ),
                        "sigma_v_sq_target": fr.sigma_v_sq_target,
                    }
                )
        return rows

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "model_name": self.model_name,
            "alpha": self.alpha,
            "layers": list(self.layers),
            "n_folds": len(self.fold_results),
            "failed_folds": self.failed_folds,
            "metadata": self.metadata,
            "aggregate_metrics": self.aggregate_metrics(),
            "fold_results": [
                {
                    "patient_id": fr.patient_id,
                    "time": fr.time,
                    "actual": fr.actual,
                    "n_condition": fr.n_condition,
                    "sigma_v_sq_target": fr.sigma_v_sq_target,
                    "parametric_mean": fr.parametric_mean,
                    "parametric_var": fr.parametric_var,
                    "intervals": {k: list(v) for k, v in fr.intervals.items()},
                }
                for fr in self.fold_results
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConformalLOPOResults:
        """Deserialise from a dict produced by :meth:`to_dict`."""
        folds = [
            ConformalLOPOFoldResult(
                patient_id=fd["patient_id"],
                time=fd["time"],
                actual=fd["actual"],
                n_condition=fd["n_condition"],
                sigma_v_sq_target=fd["sigma_v_sq_target"],
                parametric_mean=fd["parametric_mean"],
                parametric_var=fd["parametric_var"],
                intervals={k: tuple(v) for k, v in fd["intervals"].items()},
            )
            for fd in data.get("fold_results", [])
        ]
        return cls(
            model_name=data["model_name"],
            alpha=data["alpha"],
            layers=list(data["layers"]),
            fold_results=folds,
            failed_folds=data.get("failed_folds", []),
            metadata=data.get("metadata", {}),
        )


def default_cqr_features(patient: PatientTrajectory) -> np.ndarray:
    """Default CQR feature vector for the ``last_from_rest`` target.

    Uses ``[t_target, y_last_observed, t_target - t_last_observed]`` — the
    query time, the most recent observation before the target, and the gap to
    it. This captures the dominant autoregressive signal in log-volume
    trajectories without a random-effects structure (the quantile regressors
    are deliberately simple; CQR here is a calibration sensitivity check).

    Parameters
    ----------
    patient : PatientTrajectory
        Trajectory with at least two timepoints.

    Returns
    -------
    np.ndarray
        Feature vector of shape ``[3]``.
    """
    if patient.n_timepoints < 2:
        raise ConformalLOPOError(
            f"default_cqr_features needs >= 2 timepoints, patient "
            f"{patient.patient_id} has {patient.n_timepoints}"
        )
    t_target = float(patient.times[-1])
    t_last = float(patient.times[-2])
    y_last = float(patient.observations[-2, 0])
    return np.array([t_target, y_last, t_target - t_last], dtype=np.float64)


class ConformalLOPOEvaluator:
    """Nested-LOPO evaluator for conformal calibration layers.

    Parameters
    ----------
    alpha : float
        Target miscoverage; intervals target ``1 - alpha`` coverage.
    layers : tuple[str, ...]
        Calibration layers to evaluate; subset of :data:`ALL_LAYERS`.
    jackknife_score : {"signed", "symmetric"}
        Non-conformity score passed to :func:`jackknife_plus_interval`.
    cqr_calib_fraction : float
        Fraction of the training patients used to conformalise CQR.
    cqr_feature_fn : CQRFeatureFn or None
        Maps a :class:`PatientTrajectory` to a CQR feature vector. Defaults to
        :func:`default_cqr_features`.
    seed : int
        Seed for the CQR proper-train / calibration split.
    y_min, y_max : float
        Plausible-range bounds for the target. Passed to the crepes-backed
        ``cqr_norm`` layer so that, at small calibration sizes where crepes
        would otherwise return a maximum-size (``inf``) interval, the interval
        is clamped to a finite, meaningful range (log-volume is bounded by
        anatomy). Defaults to unbounded.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        layers: tuple[str, ...] = ALL_LAYERS,
        jackknife_score: str = "signed",
        cqr_calib_fraction: float = 0.33,
        cqr_feature_fn: CQRFeatureFn | None = None,
        seed: int = 42,
        y_min: float = -np.inf,
        y_max: float = np.inf,
    ) -> None:
        unknown = set(layers) - set(ALL_LAYERS)
        if unknown:
            raise ConformalLOPOError(f"unknown calibration layers: {sorted(unknown)}")
        self.alpha = alpha
        self.layers = tuple(layers)
        self.jackknife_score = jackknife_score
        self.cqr_calib_fraction = cqr_calib_fraction
        self.cqr_feature_fn = cqr_feature_fn or default_cqr_features
        self.seed = seed
        self.y_min = y_min
        self.y_max = y_max

    @staticmethod
    def _predict_last_from_rest(model: GrowthModel, patient: PatientTrajectory) -> dict[str, float]:
        """Run the ``last_from_rest`` protocol for a single patient.

        Returns a dict with ``mean``, ``var``, ``lower_95``, ``upper_95``,
        ``actual``, ``time`` and ``n_condition``.
        """
        n = patient.n_timepoints
        if n < 2:
            raise ConformalLOPOError(
                f"last_from_rest needs >= 2 timepoints, {patient.patient_id} has {n}"
            )
        t_pred = np.array([patient.times[-1]], dtype=np.float64)
        pred = model.predict(patient, t_pred, n_condition=n - 1)
        return {
            "mean": float(pred.mean[0, 0]),
            "var": float(pred.variance[0, 0]),
            "lower_95": float(pred.lower_95[0, 0]),
            "upper_95": float(pred.upper_95[0, 0]),
            "actual": float(patient.observations[-1, 0]),
            "time": float(patient.times[-1]),
            "n_condition": n - 1,
        }

    def evaluate(
        self,
        model_class: type[GrowthModel],
        patients: list[PatientTrajectory],
        **model_kwargs,
    ) -> ConformalLOPOResults:
        """Run the nested conformal LOPO evaluation.

        Parameters
        ----------
        model_class : type[GrowthModel]
            Base growth model class; instantiated fresh for every fit.
        patients : list[PatientTrajectory]
            All patient trajectories (each needs >= 2 timepoints).
        **model_kwargs
            Keyword arguments forwarded to ``model_class(...)``.

        Returns
        -------
        ConformalLOPOResults
            Per-patient intervals for every requested calibration layer.
        """
        n_patients = len(patients)
        if n_patients < 3:
            raise ConformalLOPOError(f"nested conformal LOPO needs >= 3 patients, got {n_patients}")
        model_name = model_class(**model_kwargs).name()
        logger.info(
            "Conformal LOPO: %d folds, model=%s, layers=%s",
            n_patients,
            model_name,
            list(self.layers),
        )

        fold_results: list[ConformalLOPOFoldResult] = []
        failed_folds: list[str] = []
        t_start = time.monotonic()

        for held_out_idx, held_out in enumerate(patients):
            train = [p for j, p in enumerate(patients) if j != held_out_idx]
            try:
                fold_result = self._run_fold(model_class, train, held_out, **model_kwargs)
                fold_results.append(fold_result)
            except Exception as exc:  # noqa: BLE001 — record and continue
                logger.warning("Conformal LOPO fold failed for %s: %s", held_out.patient_id, exc)
                failed_folds.append(held_out.patient_id)

        elapsed = time.monotonic() - t_start
        logger.info(
            "Conformal LOPO complete: %d/%d folds in %.1fs",
            len(fold_results),
            n_patients,
            elapsed,
        )
        return ConformalLOPOResults(
            model_name=model_name,
            alpha=self.alpha,
            layers=list(self.layers),
            fold_results=fold_results,
            failed_folds=failed_folds,
            metadata={
                "n_patients": n_patients,
                "protocol": "last_from_rest",
                "jackknife_score": self.jackknife_score,
                "cqr_calib_fraction": self.cqr_calib_fraction,
                "seed": self.seed,
                "wall_time_s": elapsed,
            },
        )

    def _run_fold(
        self,
        model_class: type[GrowthModel],
        train: list[PatientTrajectory],
        held_out: PatientTrajectory,
        **model_kwargs,
    ) -> ConformalLOPOFoldResult:
        """Evaluate every calibration layer for one held-out patient."""
        n_train = len(train)

        # --- base model on all N-1 training patients (parametric layer) ---
        base = model_class(**model_kwargs)
        base.fit(train)
        base_pred = self._predict_last_from_rest(base, held_out)
        base_sigma = float(np.sqrt(max(base_pred["var"], 1e-12)))

        intervals: dict[str, tuple[float, float]] = {}
        if "parametric" in self.layers:
            intervals["parametric"] = (base_pred["lower_95"], base_pred["upper_95"])

        need_jackknife = "jackknife_plus" in self.layers
        need_norm = "cqr_norm" in self.layers

        # --- inner jackknife over the N-1 training patients ---
        if need_jackknife or need_norm:
            loo_residuals: list[float] = []
            loo_sigmas: list[float] = []
            loo_pred_at_test: list[float] = []
            for i, inner_patient in enumerate(train):
                inner_train = [p for j, p in enumerate(train) if j != i]
                try:
                    inner_model = model_class(**model_kwargs)
                    inner_model.fit(inner_train)
                    inner_pred = self._predict_last_from_rest(inner_model, inner_patient)
                    loo_residuals.append(inner_pred["actual"] - inner_pred["mean"])
                    loo_sigmas.append(float(np.sqrt(max(inner_pred["var"], 1e-12))))
                    if need_jackknife:
                        test_pred = self._predict_last_from_rest(inner_model, held_out)
                        loo_pred_at_test.append(test_pred["mean"])
                except Exception as exc:  # noqa: BLE001 — drop this calibration unit
                    logger.debug(
                        "inner LOO unit %s failed in fold %s: %s",
                        inner_patient.patient_id,
                        held_out.patient_id,
                        exc,
                    )
            if len(loo_residuals) < 2:
                raise ConformalLOPOError(
                    f"fewer than 2 usable inner LOO units for fold {held_out.patient_id}"
                )
            loo_residuals_arr = np.asarray(loo_residuals, dtype=np.float64)
            loo_sigmas_arr = np.asarray(loo_sigmas, dtype=np.float64)

            if need_jackknife:
                lo, hi = jackknife_plus_interval(
                    np.asarray(loo_pred_at_test, dtype=np.float64),
                    loo_residuals_arr,
                    alpha=self.alpha,
                    score=self.jackknife_score,
                )
                intervals["jackknife_plus"] = (lo, hi)

            if need_norm:
                norm_cal = NormalizedConformalCalibrator(confidence=1.0 - self.alpha)
                norm_cal.fit(loo_residuals_arr, loo_sigmas_arr)
                norm_iv = norm_cal.predict_interval(
                    np.array([base_pred["mean"]]),
                    np.array([base_sigma]),
                    y_min=self.y_min,
                    y_max=self.y_max,
                )
                intervals["cqr_norm"] = (float(norm_iv[0, 0]), float(norm_iv[0, 1]))

        # --- conformalised quantile regression ---
        if "cqr_proper" in self.layers:
            x_train = np.vstack([self.cqr_feature_fn(p) for p in train])
            y_train = np.array([float(p.observations[-1, 0]) for p in train], dtype=np.float64)
            cqr = CQRCalibrator(
                alpha=self.alpha,
                calib_fraction=self.cqr_calib_fraction,
                seed=self.seed,
            )
            cqr.fit(x_train, y_train)
            cqr_iv = cqr.predict_interval(self.cqr_feature_fn(held_out).reshape(1, -1))
            intervals["cqr_proper"] = (float(cqr_iv[0, 0]), float(cqr_iv[0, 1]))

        sigma_v_sq_target = (
            float(held_out.observation_variance[-1])
            if held_out.observation_variance is not None
            else float("nan")
        )
        return ConformalLOPOFoldResult(
            patient_id=held_out.patient_id,
            time=base_pred["time"],
            actual=base_pred["actual"],
            n_condition=base_pred["n_condition"],
            sigma_v_sq_target=sigma_v_sq_target,
            parametric_mean=base_pred["mean"],
            parametric_var=base_pred["var"],
            intervals=intervals,
        )
