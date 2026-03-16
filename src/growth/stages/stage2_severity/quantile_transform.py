# src/growth/stages/stage2_severity/quantile_transform.py
"""Quantile-based time and growth normalization for the severity model.

Maps raw observation times and growth values to [0, 1] quantile space using
empirical CDFs. This is a rank-based nonlinear transformation that destroys
magnitude information — by design (see D23).

Also provides inverse quantile transform for mapping predicted growth
quantiles back to delta-log-volume space.

Usage::

    qt = QuantileTransform()
    qt.fit(all_times, all_growths)
    result = qt.transform(patient_times, patient_growths)

    # Inverse: predicted quantile -> delta log-volume
    delta_log_vol = qt.inverse_growth(q_predicted)
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


@dataclass
class QuantileTransformResult:
    """Result of quantile-transforming a patient's data.

    Args:
        t_quantile: Time quantiles, shape ``[n_i]``, values in (0, 1).
        q_growth: Growth quantiles, shape ``[n_i]``, values in [0, 1].
    """

    t_quantile: np.ndarray
    q_growth: np.ndarray


class QuantileTransform:
    """Quantile transform for time and growth values.

    Fits empirical CDFs on pooled training data, then transforms individual
    patients' observations to quantile space. The transform ensures:
    - t_quantile in (0, 1) for all observations
    - q_growth in [0, 1] with q(baseline) = 0 by convention

    Warning:
        This is a rank-based transform that destroys magnitude information.
        A 1 cm^3 growth and a 100 cm^3 growth may be adjacent in quantile space.
        This is intentional per the advisor's proposal (D23).
    """

    def __init__(self) -> None:
        self._fitted = False
        self._all_times: np.ndarray | None = None
        self._all_growths: np.ndarray | None = None
        self._sorted_growths: np.ndarray | None = None
        self._growth_ecdf_values: np.ndarray | None = None

    def fit(
        self,
        all_times: np.ndarray,
        all_growths: np.ndarray,
    ) -> "QuantileTransform":
        """Fit the empirical CDFs on pooled training data.

        Should receive only **non-baseline** observations (elapsed > 0).
        Baseline observations (growth=0, elapsed=0) are handled separately
        by the severity model.

        Args:
            all_times: Elapsed times from baseline for ALL training observations,
                shape ``[N_total]``.
            all_growths: Growth values (delta log-volume) for ALL training
                observations, shape ``[N_total]``.

        Returns:
            self (for chaining).
        """
        self._all_times = np.asarray(all_times, dtype=np.float64)
        self._all_growths = np.asarray(all_growths, dtype=np.float64)

        # Pre-compute sorted growths and their ECDF values for inverse_growth()
        self._sorted_growths = np.sort(self._all_growths)
        n = len(self._sorted_growths)
        self._growth_ecdf_values = np.arange(1, n + 1) / (n + 1)

        self._fitted = True
        logger.info(
            f"QuantileTransform fitted on {len(self._all_times)} observations. "
            f"Time range: [{self._all_times.min():.1f}, {self._all_times.max():.1f}], "
            f"Growth range: [{self._all_growths.min():.3f}, {self._all_growths.max():.3f}]"
        )
        return self

    def transform(
        self,
        times: np.ndarray,
        growths: np.ndarray,
    ) -> QuantileTransformResult:
        """Transform observation times and growths to quantile space.

        Uses the fitted empirical CDF. New values outside the training range
        are clipped to (0, 1).

        Args:
            times: Elapsed times for one patient, shape ``[n_i]``.
            growths: Growth values for one patient, shape ``[n_i]``.

        Returns:
            QuantileTransformResult with t_quantile and q_growth in [0, 1].
        """
        assert self._fitted, "Call fit() before transform()"

        times = np.asarray(times, dtype=np.float64)
        growths = np.asarray(growths, dtype=np.float64)

        t_q = self._ecdf(times, self._all_times)
        g_q = self._ecdf(growths, self._all_growths)

        return QuantileTransformResult(t_quantile=t_q, q_growth=g_q)

    def inverse_growth(self, q: np.ndarray) -> np.ndarray:
        """Inverse quantile transform: growth quantile -> delta log-volume.

        Uses linear interpolation on the sorted empirical CDF of the
        training growth values.

        Args:
            q: Growth quantiles in [0, 1], shape ``[n]``.

        Returns:
            Estimated delta log-volume values, shape ``[n]``.
        """
        assert self._fitted, "Call fit() before inverse_growth()"
        q = np.asarray(q, dtype=np.float64)

        return np.interp(
            q,
            self._growth_ecdf_values,
            self._sorted_growths,
            left=self._sorted_growths[0],
            right=self._sorted_growths[-1],
        )

    @property
    def growth_std(self) -> float:
        """Standard deviation of growth values in the reference distribution."""
        if self._all_growths is None or len(self._all_growths) == 0:
            return 0.01
        return float(np.std(self._all_growths))

    @property
    def n_reference(self) -> int:
        """Number of reference observations used for fitting."""
        if self._all_growths is None:
            return 0
        return len(self._all_growths)

    def fit_transform(
        self,
        all_times: np.ndarray,
        all_growths: np.ndarray,
    ) -> QuantileTransformResult:
        """Fit and transform in one step (for training data).

        Args:
            all_times: All training observation times.
            all_growths: All training growth values.

        Returns:
            QuantileTransformResult for the training data.
        """
        self.fit(all_times, all_growths)
        return self.transform(all_times, all_growths)

    @staticmethod
    def _ecdf(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute empirical CDF quantiles of values w.r.t. reference distribution.

        Uses rankdata with method='average' and the (rank) / (n+1) convention
        to produce values in (0, 1).

        Args:
            values: Values to transform, shape ``[m]``.
            reference: Reference distribution, shape ``[n]``.

        Returns:
            Quantiles in (0, 1), shape ``[m]``.
        """
        combined = np.concatenate([reference, values])
        ranks = rankdata(combined, method="average")
        n = len(reference)
        query_ranks = ranks[n:]
        quantiles = query_ranks / (len(combined) + 1)
        return np.clip(quantiles, 0.001, 0.999)
