# src/growth/stages/stage2_severity/quantile_transform.py
"""Quantile-based time and growth normalization for the severity model.

Maps raw observation times and growth values to [0, 1] quantile space using
empirical CDFs. This is a rank-based nonlinear transformation that destroys
magnitude information — by design (see D23).

Usage::

    qt = QuantileTransform()
    qt.fit(all_times, all_growths)
    t_q, g_q = qt.transform(patient_times, patient_growths)
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
    - t_quantile ∈ (0, 1) for all observations
    - q_growth ∈ [0, 1] with q(baseline) = 0 by convention

    Warning:
        This is a rank-based transform that destroys magnitude information.
        A 1 cm³ growth and a 100 cm³ growth may be adjacent in quantile space.
        This is intentional per the advisor's proposal (D23).

    Args:
        time_key: Whether to use ``"elapsed"`` (days/months from baseline) or
            ``"ordinal"`` (timepoint index) for the time quantile.
    """

    def __init__(self, time_key: str = "elapsed") -> None:
        self.time_key = time_key
        self._fitted = False
        self._all_times: np.ndarray | None = None
        self._all_growths: np.ndarray | None = None

    def fit(
        self,
        all_times: np.ndarray,
        all_growths: np.ndarray,
    ) -> "QuantileTransform":
        """Fit the empirical CDFs on pooled training data.

        Args:
            all_times: Elapsed times from baseline for ALL training observations,
                shape ``[N_total]``.
            all_growths: Growth values (ΔV or log-ratio) for ALL training
                observations, shape ``[N_total]``.

        Returns:
            self (for chaining).
        """
        self._all_times = np.asarray(all_times, dtype=np.float64)
        self._all_growths = np.asarray(all_growths, dtype=np.float64)
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

        # Compute quantiles by ranking within the fitted distribution
        t_q = self._ecdf(times, self._all_times)
        g_q = self._ecdf(growths, self._all_growths)

        return QuantileTransformResult(t_quantile=t_q, q_growth=g_q)

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
        # Extract ranks of the query values (last m entries)
        query_ranks = ranks[n:]
        quantiles = query_ranks / (len(combined) + 1)
        return np.clip(quantiles, 0.001, 0.999)
