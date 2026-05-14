# src/growth/shared/conformal.py
"""Distribution-free interval calibration (conformal prediction).

Under the single assumption of exchangeability between calibration and test
units, conformal prediction turns any point (or quantile) predictor into one
with a finite-sample marginal-coverage guarantee, with no distributional
assumption on the residuals. This module provides the calibration layers used
to re-calibrate the Stage 1 growth models:

- :func:`jackknife_plus_interval` — Jackknife+ (Barber et al. 2021). Uses every
  leave-one-out residual; no data-splitting waste. Worst-case coverage
  ``>= 1 - 2*alpha``, realised coverage typically ``~ 1 - alpha``.
- :class:`SplitConformalCalibrator` — textbook split conformal (Lei et al.
  2018), backed by ``crepes``. Included as a reference point; wastes data.
- :class:`NormalizedConformalCalibrator` — locally-adaptive normalised
  conformity (Papadopoulos et al. 2008), backed by ``crepes``. Reuses an
  external heteroscedastic ``sigma_hat`` to set interval *shape* while the
  conformal quantile fixes the *scale*.
- :class:`CQRCalibrator` — Conformalised Quantile Regression (Romano et al.
  2019). Conformalises a pair of trained quantile regressors.
- :class:`ConformalPredictiveSystemCalibrator` — full conformal predictive
  distribution (Vovk et al. 2020), backed by ``crepes``; yields a
  distribution-free CRPS in addition to intervals.
- :func:`beta_binomial_coverage_ci` — exact (Clopper-Pearson) interval for an
  observed coverage proportion, for honest small-N reporting.

References
----------
Barber, Candès, Ramdas, Tibshirani. "Predictive inference with the
    jackknife+." Annals of Statistics 49(1):486-507, 2021.
Romano, Patterson, Candès. "Conformalized quantile regression." NeurIPS 2019.
Papadopoulos, Vovk, Gammerman. "Normalized nonconformity measures for
    regression conformal prediction." AIAI 2008.
Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. "Distribution-free predictive
    inference for regression." JASA 113(523):1094-1111, 2018.
Vovk, Petej, Nouretdinov, et al. "Conformal calibrators." COPA 2020.
"""

from __future__ import annotations

import logging

import numpy as np
from crepes import ConformalPredictiveSystem, ConformalRegressor
from scipy import stats
from sklearn.linear_model import QuantileRegressor

logger = logging.getLogger(__name__)


class ConformalError(Exception):
    """Raised when a conformal calibrator is misconfigured or misused."""


# --------------------------------------------------------------------------
# Jackknife+
# --------------------------------------------------------------------------
def _jackknife_plus_ranks(n: int, alpha: float) -> tuple[int, int]:
    """1-indexed lower/upper order-statistic ranks for Jackknife+.

    Barber et al. (2021): the lower edge is the ``floor(alpha * (n + 1))``-th
    smallest completed value and the upper edge is the
    ``ceil((1 - alpha) * (n + 1))``-th smallest. Ranks are clamped to
    ``[1, n]``; clamping (which only triggers at very small ``n``) widens the
    interval to ``[min, max]`` and is therefore conservative.

    Parameters
    ----------
    n : int
        Number of leave-one-out calibration units.
    alpha : float
        Target miscoverage (e.g. 0.05 for a 95% interval).

    Returns
    -------
    tuple[int, int]
        ``(lower_rank, upper_rank)``, both 1-indexed and in ``[1, n]``.
    """
    lo = int(np.floor(alpha * (n + 1)))
    hi = int(np.ceil((1.0 - alpha) * (n + 1)))
    lo_clamped = min(max(lo, 1), n)
    hi_clamped = min(max(hi, 1), n)
    if lo != lo_clamped or hi != hi_clamped:
        logger.debug(
            "Jackknife+ ranks clamped at n=%d, alpha=%.3f: (%d,%d)->(%d,%d)",
            n,
            alpha,
            lo,
            hi,
            lo_clamped,
            hi_clamped,
        )
    return lo_clamped, hi_clamped


def jackknife_plus_interval(
    loo_predictions_at_test: np.ndarray,
    loo_residuals: np.ndarray,
    alpha: float = 0.05,
    score: str = "signed",
) -> tuple[float, float]:
    """Jackknife+ prediction interval for a single test point.

    Given, for a held-out test point ``x*``, the leave-one-out predictions
    ``mu_hat_{-i}(x*)`` and the leave-one-out calibration residuals
    ``r_i = y_i - mu_hat_{-i}(x_i)``, the Jackknife+ interval is built from
    the *completed* values ``mu_hat_{-i}(x*) + score(r_i)`` (Barber et al.
    2021).

    Parameters
    ----------
    loo_predictions_at_test : np.ndarray
        ``mu_hat_{-i}(x*)`` for each calibration unit ``i``, shape ``[n]``.
    loo_residuals : np.ndarray
        Signed leave-one-out residuals ``r_i = y_i - mu_hat_{-i}(x_i)``,
        shape ``[n]``.
    alpha : float
        Target miscoverage; the interval targets ``1 - alpha`` coverage
        (worst-case ``>= 1 - 2*alpha``).
    score : {"signed", "symmetric"}
        ``"signed"`` builds both edges from ``mu_hat_{-i}(x*) + r_i`` and so
        permits an asymmetric interval (recommended when over- and
        under-prediction are not equally likely). ``"symmetric"`` builds the
        lower edge from ``mu_hat_{-i}(x*) - |r_i|`` and the upper edge from
        ``mu_hat_{-i}(x*) + |r_i|`` (Barber's original absolute-score form).

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` interval bounds.
    """
    pred = np.asarray(loo_predictions_at_test, dtype=np.float64).ravel()
    res = np.asarray(loo_residuals, dtype=np.float64).ravel()
    if pred.shape != res.shape:
        raise ConformalError(
            f"loo_predictions_at_test {pred.shape} and loo_residuals "
            f"{res.shape} must have the same shape"
        )
    n = pred.shape[0]
    if n == 0:
        raise ConformalError("jackknife_plus_interval requires >= 1 calibration unit")
    lo_rank, hi_rank = _jackknife_plus_ranks(n, alpha)

    if score == "signed":
        completed = pred + res
        lower = float(np.sort(completed)[lo_rank - 1])
        upper = float(np.sort(completed)[hi_rank - 1])
    elif score == "symmetric":
        abs_res = np.abs(res)
        lower = float(np.sort(pred - abs_res)[lo_rank - 1])
        upper = float(np.sort(pred + abs_res)[hi_rank - 1])
    else:
        raise ConformalError(f"unknown score '{score}'; expected 'signed' or 'symmetric'")

    if lower > upper:  # can happen under heavy clamping at tiny n
        lower, upper = upper, lower
    return lower, upper


# --------------------------------------------------------------------------
# Split / normalised conformal (crepes-backed)
# --------------------------------------------------------------------------
class SplitConformalCalibrator:
    """Textbook split-conformal regressor (Lei et al. 2018), via ``crepes``.

    Fits on signed calibration residuals ``y_i - mu_hat(x_i)`` and produces
    intervals ``mu_hat(x*) + [q_lo, q_hi]`` where the conformal quantiles are
    order statistics of the calibration residuals. Included as a reference
    calibration point; it wastes data relative to Jackknife+.

    Parameters
    ----------
    confidence : float
        Target coverage (e.g. 0.95).
    """

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence
        self._cr: ConformalRegressor | None = None

    def fit(self, calibration_residuals: np.ndarray) -> SplitConformalCalibrator:
        """Fit on signed calibration residuals ``y_i - mu_hat(x_i)``."""
        res = np.asarray(calibration_residuals, dtype=np.float64).ravel()
        if res.size == 0:
            raise ConformalError("SplitConformalCalibrator needs >= 1 residual")
        self._cr = ConformalRegressor()
        self._cr.fit(residuals=res)
        return self

    def predict_interval(
        self,
        y_hat: np.ndarray,
        y_min: float = -np.inf,
        y_max: float = np.inf,
    ) -> np.ndarray:
        """Return ``[n, 2]`` lower/upper bounds for point predictions ``y_hat``.

        ``y_min``/``y_max`` clamp the interval to a plausible range; at very
        small calibration sizes ``crepes`` returns a maximum-size interval,
        and finite bounds keep it meaningful (log-volume is bounded by anatomy).
        """
        if self._cr is None:
            raise ConformalError("SplitConformalCalibrator not fitted")
        y_hat = np.atleast_1d(np.asarray(y_hat, dtype=np.float64))
        return np.asarray(
            self._cr.predict_int(y_hat=y_hat, confidence=self.confidence, y_min=y_min, y_max=y_max)
        )


class NormalizedConformalCalibrator:
    """Locally-adaptive normalised-conformity calibrator (Papadopoulos 2008).

    The non-conformity score is ``(y_i - mu_hat(x_i)) / sigma_hat(x_i)`` for an
    externally supplied heteroscedastic ``sigma_hat`` (e.g. the predictive std
    of a heteroscedastic LME or an ensemble mixture). The conformal quantile is
    taken on the *normalised* residuals and re-multiplied by ``sigma_hat(x*)``
    at prediction time, so ``sigma_hat`` sets interval *shape* while the
    conformal step fixes the *scale*. If ``sigma_hat`` carries no shape
    information the interval collapses to a constant width and Jackknife+
    behaviour is recovered.

    Parameters
    ----------
    confidence : float
        Target coverage (e.g. 0.95).
    sigma_floor : float
        Lower bound applied to every ``sigma_hat`` value to avoid division by
        (near-)zero.
    """

    def __init__(self, confidence: float = 0.95, sigma_floor: float = 1e-6) -> None:
        self.confidence = confidence
        self.sigma_floor = sigma_floor
        self._cr: ConformalRegressor | None = None

    def fit(
        self,
        calibration_residuals: np.ndarray,
        calibration_sigmas: np.ndarray,
    ) -> NormalizedConformalCalibrator:
        """Fit on signed residuals and their matched ``sigma_hat`` values.

        Parameters
        ----------
        calibration_residuals : np.ndarray
            Signed residuals ``y_i - mu_hat(x_i)``, shape ``[n]``.
        calibration_sigmas : np.ndarray
            Heteroscedastic ``sigma_hat(x_i)``, shape ``[n]``, strictly
            positive after flooring.
        """
        res = np.asarray(calibration_residuals, dtype=np.float64).ravel()
        sig = np.maximum(np.asarray(calibration_sigmas, dtype=np.float64).ravel(), self.sigma_floor)
        if res.shape != sig.shape:
            raise ConformalError(f"residuals {res.shape} and sigmas {sig.shape} must match")
        if res.size == 0:
            raise ConformalError("NormalizedConformalCalibrator needs >= 1 residual")
        self._cr = ConformalRegressor()
        self._cr.fit(residuals=res, sigmas=sig)
        return self

    def predict_interval(
        self,
        y_hat: np.ndarray,
        sigma_hat: np.ndarray,
        y_min: float = -np.inf,
        y_max: float = np.inf,
    ) -> np.ndarray:
        """Return ``[n, 2]`` bounds for predictions ``y_hat`` with shape ``sigma_hat``.

        ``y_min``/``y_max`` clamp the interval to a plausible range; at very
        small calibration sizes ``crepes`` returns a maximum-size interval,
        and finite bounds keep it meaningful (log-volume is bounded by anatomy).
        """
        if self._cr is None:
            raise ConformalError("NormalizedConformalCalibrator not fitted")
        y_hat = np.atleast_1d(np.asarray(y_hat, dtype=np.float64))
        sig = np.maximum(np.atleast_1d(np.asarray(sigma_hat, dtype=np.float64)), self.sigma_floor)
        return np.asarray(
            self._cr.predict_int(
                y_hat=y_hat, sigmas=sig, confidence=self.confidence, y_min=y_min, y_max=y_max
            )
        )


class ConformalPredictiveSystemCalibrator:
    """Full conformal predictive distribution (Vovk et al. 2020), via ``crepes``.

    Produces a distribution-free predictive distribution per test point, from
    which both ``(1 - alpha)`` intervals and a proper CRPS can be read off
    without a Gaussian assumption. Optionally normalised by an external
    ``sigma_hat``.

    Parameters
    ----------
    confidence : float
        Target coverage for :meth:`predict_interval`.
    sigma_floor : float
        Lower bound applied to ``sigma_hat`` when normalisation is used.
    """

    def __init__(self, confidence: float = 0.95, sigma_floor: float = 1e-6) -> None:
        self.confidence = confidence
        self.sigma_floor = sigma_floor
        self._cps: ConformalPredictiveSystem | None = None
        self._normalised = False

    def fit(
        self,
        calibration_residuals: np.ndarray,
        calibration_sigmas: np.ndarray | None = None,
    ) -> ConformalPredictiveSystemCalibrator:
        """Fit on signed calibration residuals, optionally normalised."""
        res = np.asarray(calibration_residuals, dtype=np.float64).ravel()
        if res.size == 0:
            raise ConformalError("ConformalPredictiveSystemCalibrator needs >= 1 residual")
        self._cps = ConformalPredictiveSystem()
        if calibration_sigmas is not None:
            sig = np.maximum(
                np.asarray(calibration_sigmas, dtype=np.float64).ravel(), self.sigma_floor
            )
            self._cps.fit(residuals=res, sigmas=sig)
            self._normalised = True
        else:
            self._cps.fit(residuals=res)
            self._normalised = False
        return self

    def predict_interval(
        self,
        y_hat: np.ndarray,
        sigma_hat: np.ndarray | None = None,
        y_min: float = -np.inf,
        y_max: float = np.inf,
    ) -> np.ndarray:
        """Return ``[n, 2]`` conformal predictive intervals.

        ``y_min``/``y_max`` clamp the interval to a plausible range.
        """
        if self._cps is None:
            raise ConformalError("ConformalPredictiveSystemCalibrator not fitted")
        y_hat = np.atleast_1d(np.asarray(y_hat, dtype=np.float64))
        kwargs: dict = {
            "y_hat": y_hat,
            "confidence": self.confidence,
            "y_min": y_min,
            "y_max": y_max,
        }
        if self._normalised:
            if sigma_hat is None:
                raise ConformalError("calibrator was fitted normalised; sigma_hat required")
            kwargs["sigmas"] = np.maximum(
                np.atleast_1d(np.asarray(sigma_hat, dtype=np.float64)), self.sigma_floor
            )
        return np.asarray(self._cps.predict_int(**kwargs))


# --------------------------------------------------------------------------
# Conformalised Quantile Regression
# --------------------------------------------------------------------------
class CQRCalibrator:
    """Conformalised Quantile Regression (Romano, Patterson, Candès 2019).

    Splits the supplied data into a proper-training part and a calibration
    part, trains an ``alpha/2`` and a ``1 - alpha/2`` linear quantile
    regressor on the proper-training part, then conformalises with the
    calibration-set score ``E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))``.
    The conformalised interval is ``[q_lo(x*) - Q, q_hi(x*) + Q]`` where ``Q``
    is the ``ceil((1 - alpha)(n_cal + 1))``-th smallest of ``{E_i}``.

    At ``N ~ 54`` this is a *sensitivity check* on the locally-adaptive
    family; :class:`NormalizedConformalCalibrator` is the recommended default
    because it adds no extra estimation burden.

    Parameters
    ----------
    alpha : float
        Target miscoverage (e.g. 0.05).
    calib_fraction : float
        Fraction of the supplied data held out for conformalisation.
    quantile_reg_alpha : float
        L1 regularisation strength of the underlying
        :class:`sklearn.linear_model.QuantileRegressor`. ``0.0`` is
        unregularised; a small positive value stabilises tiny-N fits.
    solver : str
        Linear-programming solver for the quantile regressors.
    seed : int
        Seed for the proper-train / calibration split.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        calib_fraction: float = 0.33,
        quantile_reg_alpha: float = 0.0,
        solver: str = "highs",
        seed: int = 42,
    ) -> None:
        if not 0.0 < calib_fraction < 1.0:
            raise ConformalError("calib_fraction must lie in (0, 1)")
        self.alpha = alpha
        self.calib_fraction = calib_fraction
        self.quantile_reg_alpha = quantile_reg_alpha
        self.solver = solver
        self.seed = seed
        self._q_lo: QuantileRegressor | None = None
        self._q_hi: QuantileRegressor | None = None
        self._conformity: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> CQRCalibrator:
        """Train the quantile regressors and compute the conformity correction.

        Parameters
        ----------
        x : np.ndarray
            Calibration feature matrix, shape ``[n, p]``.
        y : np.ndarray
            Calibration targets, shape ``[n]``.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.shape[0] != y.shape[0]:
            raise ConformalError(f"x has {x.shape[0]} rows but y has {y.shape[0]}")
        n = x.shape[0]
        if n < 4:
            raise ConformalError(f"CQR needs >= 4 units to split; got {n}")

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_cal = max(1, int(round(self.calib_fraction * n)))
        n_cal = min(n_cal, n - 2)  # leave >= 2 for proper training
        cal_idx, train_idx = perm[:n_cal], perm[n_cal:]

        x_tr, y_tr = x[train_idx], y[train_idx]
        x_cal, y_cal = x[cal_idx], y[cal_idx]

        self._q_lo = QuantileRegressor(
            quantile=self.alpha / 2.0, alpha=self.quantile_reg_alpha, solver=self.solver
        ).fit(x_tr, y_tr)
        self._q_hi = QuantileRegressor(
            quantile=1.0 - self.alpha / 2.0, alpha=self.quantile_reg_alpha, solver=self.solver
        ).fit(x_tr, y_tr)

        # Independently fitted quantile regressors can cross at small N; the
        # rearrangement of Chernozhukov, Fernandez-Val & Galichon (2010)
        # restores monotonicity without changing the marginal quantile levels.
        lo_cal_raw = self._q_lo.predict(x_cal)
        hi_cal_raw = self._q_hi.predict(x_cal)
        lo_cal = np.minimum(lo_cal_raw, hi_cal_raw)
        hi_cal = np.maximum(lo_cal_raw, hi_cal_raw)
        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)

        rank = int(np.ceil((1.0 - self.alpha) * (n_cal + 1)))
        if rank > n_cal:
            # finite-sample correction exceeds the sample: widen to the max score
            self._conformity = float(np.max(scores))
            logger.debug("CQR conformity rank %d > n_cal %d; using max score", rank, n_cal)
        else:
            self._conformity = float(np.sort(scores)[rank - 1])
        return self

    def predict_interval(self, x: np.ndarray) -> np.ndarray:
        """Return ``[n, 2]`` conformalised quantile-regression intervals.

        The two quantile predictions are rearranged to be non-crossing before
        the conformity correction is applied, so the returned interval always
        satisfies ``lower <= upper``.
        """
        if self._q_lo is None or self._q_hi is None:
            raise ConformalError("CQRCalibrator not fitted")
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        lo_raw = self._q_lo.predict(x)
        hi_raw = self._q_hi.predict(x)
        lower = np.minimum(lo_raw, hi_raw) - self._conformity
        upper = np.maximum(lo_raw, hi_raw) + self._conformity
        return np.column_stack([lower, upper])


# --------------------------------------------------------------------------
# Honest small-N coverage reporting
# --------------------------------------------------------------------------
def beta_binomial_coverage_ci(
    n_covered: int,
    n_total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) confidence interval for an observed coverage.

    At ``N ~ 54`` a single held-out coverage estimate carries substantial
    Beta-binomial uncertainty; an observed coverage of e.g. 0.95 is consistent
    with a wide band. Reporting this interval keeps small-N coverage claims
    honest (``docs/CONFORMAL_PATH_ANALYSIS.md`` Sec. 4.1).

    Parameters
    ----------
    n_covered : int
        Number of test units whose interval covered the truth.
    n_total : int
        Total number of test units.
    confidence : float
        Confidence level of the returned interval (e.g. 0.95).

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` bounds on the true coverage probability.
    """
    if n_total <= 0:
        raise ConformalError("n_total must be positive")
    if not 0 <= n_covered <= n_total:
        raise ConformalError("n_covered must lie in [0, n_total]")
    a = (1.0 - confidence) / 2.0
    lower = 0.0 if n_covered == 0 else float(stats.beta.ppf(a, n_covered, n_total - n_covered + 1))
    upper = (
        1.0
        if n_covered == n_total
        else float(stats.beta.ppf(1.0 - a, n_covered + 1, n_total - n_covered))
    )
    return lower, upper
