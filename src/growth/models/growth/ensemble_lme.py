# src/growth/models/growth/ensemble_lme.py
"""Ensemble-of-trajectories Bayesian model averaging over LME fits (Path B).

Instead of collapsing an M-member segmentation ensemble to a single scalar
measurement variance ``sigma_v^2`` per scan (the heteroscedastic-LME route),
this model keeps the full ensemble: it fits one homoscedastic LME to each of
the M trajectory realisations and forms the equal-weight Gaussian mixture

    p(y* | t*, D)  =  (1/M) * sum_{m=1}^{M} N(y*; mu_m(t*), sigma_m^2(t*))

where ``(mu_m, sigma_m^2)`` is the predictive of the LME fitted to the m-th
ensemble member's trajectory data. This is a proper Bayesian model average
over the unknown measurement realisation (CASUS framing, Judge et al. 2025).

The law of total variance decomposes the mixture variance into

    Var(y*)  =  E_m[sigma_m^2]  +  Var_m[mu_m]
                within-model        measurement (member-disagreement)

The heteroscedastic-LME route captures only an approximation of the second
term; the mixture captures both. See ``docs/CONFORMAL_PATH_ANALYSIS.md`` Sec. 5.

References:
    Hoeting, Madigan, Raftery, Volinsky. "Bayesian Model Averaging: A
        Tutorial." Statistical Science 14(4):382-417, 1999.
    Judge et al. "CASUS: Contour Sampling for Uncertainty in Segmentation."
        2025.
"""

import logging

import numpy as np
from scipy import optimize, stats

from .base import FitResult, GrowthModel, PatientTrajectory, PredictionResult
from .lme_model import LMEGrowthModel

logger = logging.getLogger(__name__)


class EnsembleLMEError(Exception):
    """Raised when the ensemble-LME model is misconfigured or misused."""


def _gaussian_mixture_quantile(
    means: np.ndarray,
    sigmas: np.ndarray,
    q: float,
    weights: np.ndarray | None = None,
) -> float:
    """Exact quantile of an equal- or arbitrary-weight 1-D Gaussian mixture.

    Solves ``F(y) = q`` where ``F(y) = sum_m w_m * Phi((y - mu_m) / sigma_m)``
    by Brent root-finding on a bracket wide enough to contain every component.

    Parameters
    ----------
    means : np.ndarray
        Component means, shape ``[M]``.
    sigmas : np.ndarray
        Component standard deviations, shape ``[M]``, strictly positive.
    q : float
        Target cumulative probability in ``(0, 1)``.
    weights : np.ndarray or None
        Mixture weights, shape ``[M]``. ``None`` means uniform ``1/M``.

    Returns
    -------
    float
        The value ``y`` such that the mixture CDF equals ``q``.
    """
    means = np.asarray(means, dtype=np.float64)
    sigmas = np.maximum(np.asarray(sigmas, dtype=np.float64), 1e-12)
    m = means.shape[0]
    w = np.full(m, 1.0 / m) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    def cdf(y: float) -> float:
        return float(np.sum(w * stats.norm.cdf((y - means) / sigmas)))

    lo = float(np.min(means - 8.0 * sigmas))
    hi = float(np.max(means + 8.0 * sigmas))
    # Expand the bracket until it strictly contains the target probability.
    for _ in range(64):
        if cdf(lo) <= q <= cdf(hi):
            break
        span = hi - lo
        lo -= span
        hi += span
    return float(optimize.brentq(lambda y: cdf(y) - q, lo, hi, xtol=1e-8, rtol=1e-10))


class EnsembleLMEGrowthModel(GrowthModel):
    """Equal-weight Bayesian model average over per-member LME fits.

    Requires every training and test trajectory to carry an
    ``observation_ensemble`` array of shape ``[n_i, M]`` (column ``m`` is the
    m-th measurement realisation of the scalar observed value). One
    :class:`LMEGrowthModel` is fitted per member; predictions are combined
    into a Gaussian mixture.

    Parameters
    ----------
    method : str
        REML (``"reml"``) or ML (``"ml"``) estimation for each member LME.
    n_members : int or None
        Number of ensemble members to use. ``None`` uses every column of the
        ``observation_ensemble`` array seen at fit time. A smaller value uses
        the leading ``n_members`` columns (useful for smoke tests).
    use_covariates : bool
        Whether the member LMEs use static covariates as fixed effects.
    covariate_names : list[str] or None
        Covariate names passed through to each member LME.
    missing_strategy : str
        Covariate-missingness strategy passed through to each member LME.
    """

    def __init__(
        self,
        method: str = "reml",
        n_members: int | None = None,
        use_covariates: bool = False,
        covariate_names: list[str] | None = None,
        missing_strategy: str = "skip",
    ) -> None:
        self.method = method
        self.n_members = n_members
        self.use_covariates = use_covariates
        self.covariate_names = covariate_names or []
        self.missing_strategy = missing_strategy
        self._member_models: list[LMEGrowthModel] = []
        self._n_members_fitted: int = 0
        self._fitted: bool = False

    @staticmethod
    def _member_view(patient: PatientTrajectory, m: int) -> PatientTrajectory:
        """Return a scalar :class:`PatientTrajectory` for ensemble member ``m``.

        The observations are replaced by column ``m`` of
        ``observation_ensemble``; covariates are carried through; the
        ensemble and heteroscedastic-variance fields are dropped (each member
        model is an ordinary homoscedastic LME).
        """
        if patient.observation_ensemble is None:
            raise EnsembleLMEError(
                f"patient {patient.patient_id} has no observation_ensemble; "
                f"EnsembleLMEGrowthModel requires the per-member array"
            )
        return PatientTrajectory(
            patient_id=patient.patient_id,
            times=patient.times,
            observations=patient.observation_ensemble[:, m],
            covariates=patient.covariates,
        )

    def _resolve_n_members(self, patients: list[PatientTrajectory]) -> int:
        """Determine M from the config and the trajectories, validating consistency."""
        sizes = {p.ensemble_size for p in patients}
        if sizes == {0}:
            raise EnsembleLMEError(
                "no trajectory carries an observation_ensemble; cannot fit EnsembleLMEGrowthModel"
            )
        sizes.discard(0)
        if len(sizes) > 1:
            raise EnsembleLMEError(
                f"inconsistent ensemble sizes across trajectories: {sorted(sizes)}"
            )
        available = sizes.pop()
        if self.n_members is None:
            return available
        if self.n_members > available:
            raise EnsembleLMEError(
                f"n_members={self.n_members} exceeds the available ensemble size {available}"
            )
        return self.n_members

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit one homoscedastic LME per ensemble member.

        Parameters
        ----------
        patients : list[PatientTrajectory]
            Training trajectories; each must carry ``observation_ensemble``.

        Returns
        -------
        FitResult
            Aggregated diagnostics. ``log_marginal_likelihood`` is the mean
            REML criterion across members; ``metadata`` holds the per-member
            criteria and the resolved ensemble size.
        """
        if len(patients) == 0:
            raise EnsembleLMEError("cannot fit with zero patients")

        m_count = self._resolve_n_members(patients)
        self._member_models = []
        member_lml: list[float] = []
        max_cond = 0.0

        for m in range(m_count):
            member_patients = [self._member_view(p, m) for p in patients]
            model = LMEGrowthModel(
                method=self.method,
                use_covariates=self.use_covariates,
                covariate_names=self.covariate_names,
                missing_strategy=self.missing_strategy,
            )
            fit_result = model.fit(member_patients)
            self._member_models.append(model)
            member_lml.append(fit_result.log_marginal_likelihood)
            max_cond = max(max_cond, fit_result.condition_number)

        self._n_members_fitted = m_count
        self._fitted = True
        n_total = sum(p.n_timepoints for p in patients)
        logger.info(
            "EnsembleLME fit: M=%d members, %d patients, %d observations, mean member LML=%.2f",
            m_count,
            len(patients),
            n_total,
            float(np.mean(member_lml)),
        )

        return FitResult(
            log_marginal_likelihood=float(np.mean(member_lml)),
            hyperparameters={"n_members": float(m_count)},
            condition_number=max_cond,
            n_train_patients=len(patients),
            n_train_observations=n_total,
            metadata={"member_log_marginal_likelihoods": member_lml},
        )

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict the equal-weight Gaussian mixture over the M member LMEs.

        Each member LME conditions on the test patient's *own* member-``m``
        realisation of the preceding observations. The mixture mean and
        variance follow the law of total variance; the 95% interval bounds
        are exact mixture-CDF quantiles (not a Gaussian approximation).

        Parameters
        ----------
        patient : PatientTrajectory
            Test trajectory; must carry ``observation_ensemble``.
        t_pred : np.ndarray
            Query times, shape ``[n_pred]``.
        n_condition : int or None
            Condition each member LME on its first ``n_condition`` observations.

        Returns
        -------
        PredictionResult
            ``mean``/``variance`` are the mixture moments; ``lower_95``/
            ``upper_95`` are exact mixture quantiles. ``metadata`` carries the
            per-component means/variances and the within/between variance
            decomposition.
        """
        if not self._fitted:
            raise EnsembleLMEError("model not fitted; call fit() first")
        if patient.observation_ensemble is None:
            raise EnsembleLMEError(f"test patient {patient.patient_id} has no observation_ensemble")
        if patient.ensemble_size < self._n_members_fitted:
            raise EnsembleLMEError(
                f"test patient {patient.patient_id} has ensemble size "
                f"{patient.ensemble_size} < {self._n_members_fitted} fitted members"
            )

        t_pred = np.atleast_1d(np.asarray(t_pred, dtype=np.float64))
        n_pred = t_pred.shape[0]
        m_count = self._n_members_fitted

        comp_means = np.empty((n_pred, m_count), dtype=np.float64)
        comp_vars = np.empty((n_pred, m_count), dtype=np.float64)
        for m in range(m_count):
            member_patient = self._member_view(patient, m)
            pred = self._member_models[m].predict(member_patient, t_pred, n_condition)
            comp_means[:, m] = pred.mean[:, 0]
            comp_vars[:, m] = pred.variance[:, 0]

        # Law of total variance: E_m[sigma_m^2] + Var_m[mu_m] (population var, ddof=0).
        mixture_mean = comp_means.mean(axis=1)
        within_var = comp_vars.mean(axis=1)
        between_var = comp_means.var(axis=1)
        mixture_var = within_var + between_var

        comp_sigmas = np.sqrt(np.maximum(comp_vars, 1e-12))
        lower = np.array(
            [
                _gaussian_mixture_quantile(comp_means[i], comp_sigmas[i], 0.025)
                for i in range(n_pred)
            ]
        )
        upper = np.array(
            [
                _gaussian_mixture_quantile(comp_means[i], comp_sigmas[i], 0.975)
                for i in range(n_pred)
            ]
        )

        return PredictionResult(
            mean=mixture_mean,
            variance=mixture_var,
            lower_95=lower,
            upper_95=upper,
            metadata={
                "component_means": comp_means,
                "component_variances": comp_vars,
                "within_var": within_var,
                "between_var": between_var,
                "n_members": m_count,
            },
        )

    def name(self) -> str:
        """Human-readable model name."""
        m_str = self._n_members_fitted if self._fitted else (self.n_members or "auto")
        return f"EnsembleLME(M={m_str}, method={self.method})"
