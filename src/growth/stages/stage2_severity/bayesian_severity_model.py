# src/growth/stages/stage2_severity/bayesian_severity_model.py
"""Latent severity NLME growth model — Option B: Bayesian via numpyro MCMC.

Same generative model as Option A (MLE), but estimates the full posterior
distribution p(theta, {s_i} | data) via the NUTS sampler. This provides:
- Full posterior distributions on population parameters
- Per-patient severity credible intervals
- Posterior predictive distributions for calibrated uncertainty

Priors (weakly informative):
- s_i ~ Beta(2, 2): mild regularization toward 0.5
- alpha_0, alpha_1, beta ~ HalfNormal(1.0): positive, weakly constrained
- sigma ~ HalfNormal(0.2): observation noise

References:
    - Vaghi et al. (2020) PLOS Computational Biology
    - Hoffman & Gelman (2014) JMLR — NUTS sampler
    - Betancourt (2017) — practical HMC

Spec: ``docs/stages/stage_2_severity_model.md``
"""

import logging

import numpy as np

from growth.shared.growth_models import (
    FitResult,
    GrowthModel,
    PatientTrajectory,
    PredictionResult,
)
from growth.stages.stage2_severity.quantile_transform import QuantileTransform
from growth.stages.stage2_severity.severity_regression import SeverityRegressionHead

logger = logging.getLogger(__name__)

# Lazy imports for numpyro/jax to avoid import errors when not installed
_NUMPYRO_AVAILABLE = False
try:
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive

    _NUMPYRO_AVAILABLE = True
except ImportError:
    pass


def _check_numpyro() -> None:
    """Raise ImportError if numpyro is not available."""
    if not _NUMPYRO_AVAILABLE:
        raise ImportError(
            "numpyro and jax are required for BayesianSeverityModel. "
            "Install with: pip install numpyro 'jax[cpu]' arviz"
        )


def _gompertz_jax(
    s: "jnp.ndarray",
    t: "jnp.ndarray",
    alpha_0: float,
    alpha_1: float,
    beta: float,
) -> "jnp.ndarray":
    """Reduced Gompertz growth function in JAX.

    g(s, t) = clip(exp(alpha(s)/beta * (1 - exp(-beta*t))) - 1, 0, 1)
    where alpha(s) = alpha_0 + alpha_1 * s
    """
    alpha = alpha_0 + alpha_1 * s
    growth = jnp.exp(alpha / (beta + 1e-8) * (1.0 - jnp.exp(-beta * t))) - 1.0
    return jnp.clip(growth, 0.0, 1.0)


def _sigmoid_jax(
    s: "jnp.ndarray",
    t: "jnp.ndarray",
    w1: float,
    w2: float,
    b: float,
) -> "jnp.ndarray":
    """Weighted sigmoid growth function in JAX.

    g(s, t) = clip(t * sigmoid(w1*s + w2*t + b), 0, 1)
    """
    sigmoid = 1.0 / (1.0 + jnp.exp(-(w1 * s + w2 * t + b)))
    return jnp.clip(t * sigmoid, 0.0, 1.0)


class BayesianSeverityModel(GrowthModel):
    """Bayesian severity NLME model using MCMC (numpyro NUTS).

    Same growth functions and quantile transform as SeverityModel (MLE),
    but estimates the full posterior via Hamiltonian Monte Carlo.

    Args:
        growth_function: ``"gompertz_reduced"`` or ``"weighted_sigmoid"``.
        n_warmup: NUTS warmup (adaptation) iterations.
        n_samples: NUTS sampling iterations after warmup.
        n_chains: Number of parallel MCMC chains.
        severity_features: Feature names for test-time severity regression.
        seed: Random seed.
    """

    def __init__(
        self,
        growth_function: str = "gompertz_reduced",
        n_warmup: int = 500,
        n_samples: int = 1000,
        n_chains: int = 2,
        severity_features: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        _check_numpyro()

        self._growth_function_name = growth_function
        self._n_warmup = n_warmup
        self._n_samples = n_samples
        self._n_chains = n_chains
        self._severity_features = severity_features or ["log_volume", "sphericity"]
        self._seed = seed

        # Fitted state
        self._qt: QuantileTransform | None = None
        self._posterior_samples: dict[str, np.ndarray] | None = None
        self._regression_head: SeverityRegressionHead | None = None
        self._train_patient_ids: set[str] = set()
        self._severity_means: dict[str, float] = {}

    def _numpyro_model(
        self,
        t_q: "jnp.ndarray",
        q_obs: "jnp.ndarray",
        patient_idx: "jnp.ndarray",
        n_patients: int,
    ) -> None:
        """Numpyro probabilistic model definition.

        Args:
            t_q: Time quantiles, shape [N_obs].
            q_obs: Observed growth quantiles, shape [N_obs].
            patient_idx: Patient index per observation, shape [N_obs].
            n_patients: Number of patients.
        """
        # Per-patient severity: Beta(2, 2) prior
        s = numpyro.sample("s", dist.Beta(2.0, 2.0).expand([n_patients]))

        # Observation noise
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.2))

        if self._growth_function_name == "gompertz_reduced":
            alpha_0 = numpyro.sample("alpha_0", dist.HalfNormal(1.0))
            alpha_1 = numpyro.sample("alpha_1", dist.HalfNormal(1.0))
            beta = numpyro.sample("beta", dist.HalfNormal(1.0))
            q_pred = _gompertz_jax(s[patient_idx], t_q, alpha_0, alpha_1, beta)
        elif self._growth_function_name == "weighted_sigmoid":
            w1 = numpyro.sample("w1", dist.HalfNormal(2.0))
            w2 = numpyro.sample("w2", dist.HalfNormal(2.0))
            b = numpyro.sample("b", dist.Normal(0.0, 2.0))
            q_pred = _sigmoid_jax(s[patient_idx], t_q, w1, w2, b)
        else:
            raise ValueError(f"Unknown growth function: {self._growth_function_name}")

        numpyro.sample("obs", dist.Normal(q_pred, sigma), obs=q_obs)

    def fit(self, patients: list[PatientTrajectory]) -> FitResult:
        """Fit via MCMC (NUTS sampler).

        Steps:
        1. Compute growth values and quantile transform (same as MLE)
        2. Run NUTS sampler
        3. Extract posterior means for severity
        4. Fit severity regression head

        Args:
            patients: Training patient trajectories.

        Returns:
            FitResult with posterior summary statistics.
        """
        self._train_patient_ids = {p.patient_id for p in patients}

        # 1. Compute growth values and quantile transform
        all_t_nonbaseline: list[float] = []
        all_g_nonbaseline: list[float] = []
        patient_data: list[tuple[np.ndarray, np.ndarray]] = []

        for p in patients:
            baseline = float(p.observations[0, 0])
            growth = p.observations[:, 0] - baseline
            elapsed = p.times - p.times[0]
            patient_data.append((elapsed, growth))

            for j in range(len(elapsed)):
                if elapsed[j] > 0:
                    all_t_nonbaseline.append(elapsed[j])
                    all_g_nonbaseline.append(growth[j])

        self._qt = QuantileTransform()
        if all_t_nonbaseline:
            self._qt.fit(np.array(all_t_nonbaseline), np.array(all_g_nonbaseline))
        else:
            self._qt.fit(np.array([1.0]), np.array([0.0]))

        # Transform to quantile space (excluding baselines)
        all_t_q: list[float] = []
        all_q_g: list[float] = []
        all_pidx: list[int] = []

        for i, (elapsed, growth) in enumerate(patient_data):
            mask = elapsed > 0
            if not np.any(mask):
                continue
            result = self._qt.transform(elapsed[mask], growth[mask])
            all_t_q.extend(result.t_quantile.tolist())
            all_q_g.extend(result.q_growth.tolist())
            all_pidx.extend([i] * int(mask.sum()))

        t_q_arr = jnp.array(all_t_q)
        q_g_arr = jnp.array(all_q_g)
        pidx_arr = jnp.array(all_pidx, dtype=jnp.int32)
        n_patients = len(patients)

        # 2. Run NUTS sampler
        rng_key = jrandom.PRNGKey(self._seed)
        kernel = NUTS(self._numpyro_model)
        mcmc = MCMC(
            kernel,
            num_warmup=self._n_warmup,
            num_samples=self._n_samples,
            num_chains=self._n_chains,
            progress_bar=False,
        )
        mcmc.run(rng_key, t_q_arr, q_g_arr, pidx_arr, n_patients)

        self._posterior_samples = {k: np.array(v) for k, v in mcmc.get_samples().items()}

        # 3. Extract posterior severity means
        s_samples = self._posterior_samples["s"]  # [n_samples, n_patients]
        s_means = np.mean(s_samples, axis=0)

        self._severity_means = {p.patient_id: float(s_means[i]) for i, p in enumerate(patients)}

        logger.info(
            f"Bayesian fit: {self._n_samples} samples x {self._n_chains} chains, "
            f"severity range=[{s_means.min():.3f}, {s_means.max():.3f}], "
            f"severity std={s_means.std():.3f}"
        )

        # 4. Fit severity regression head (same as MLE)
        self._regression_head = SeverityRegressionHead(feature_names=self._severity_features)
        self._regression_head.fit(patients, s_means)

        # Build FitResult with posterior summaries
        hypers: dict[str, float] = {
            "severity_mean": float(s_means.mean()),
            "severity_std": float(s_means.std()),
        }
        for param_name in ["alpha_0", "alpha_1", "beta", "w1", "w2", "b", "sigma"]:
            if param_name in self._posterior_samples:
                samples = self._posterior_samples[param_name]
                hypers[f"{param_name}_mean"] = float(np.mean(samples))
                hypers[f"{param_name}_std"] = float(np.std(samples))

        return FitResult(
            log_marginal_likelihood=0.0,  # Not directly available from MCMC
            hyperparameters=hypers,
            n_train_patients=n_patients,
            n_train_observations=len(all_t_q),
        )

    def predict(
        self,
        patient: PatientTrajectory,
        t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult:
        """Predict using posterior predictive distribution.

        For each posterior sample, evaluates the growth function and
        inverse-transforms to log-volume space. Returns mean, variance,
        and 95% CIs from the posterior predictive ensemble.

        Args:
            patient: Patient trajectory.
            t_pred: Query times, shape [n_pred].
            n_condition: Unused (severity model doesn't condition on data).

        Returns:
            PredictionResult with posterior predictive statistics.
        """
        assert self._posterior_samples is not None, "Call fit() first"
        assert self._qt is not None, "Call fit() first"

        t_pred = np.asarray(t_pred, dtype=np.float64)
        if t_pred.ndim == 0:
            t_pred = t_pred[np.newaxis]

        baseline = float(patient.observations[0, 0])
        elapsed = t_pred - patient.times[0]

        # Transform prediction times to quantile space
        t_q = np.zeros_like(elapsed)
        mask = elapsed > 0
        if np.any(mask):
            qt_result = self._qt.transform(elapsed[mask], np.zeros(int(mask.sum())))
            t_q[mask] = qt_result.t_quantile

        # Get severity: fitted (training) or regression (held-out)
        if patient.patient_id in self._severity_means:
            # Use posterior samples for training patients
            s_samples = self._posterior_samples["s"]
            pid_idx = list(self._severity_means.keys()).index(patient.patient_id)
            s_values = s_samples[:, pid_idx]  # [n_samples]
        else:
            # Held-out: point estimate from regression
            s_point = self._regression_head.predict(patient) if self._regression_head else 0.5
            s_values = np.full(len(next(iter(self._posterior_samples.values()))), s_point)

        # Posterior predictive: for each sample, evaluate growth function
        n_samples = len(s_values)
        n_pred = len(t_pred)
        pred_ensemble = np.zeros((n_samples, n_pred))

        for k in range(n_samples):
            s_k = s_values[k]

            if self._growth_function_name == "gompertz_reduced":
                a0 = float(self._posterior_samples["alpha_0"][k])
                a1 = float(self._posterior_samples["alpha_1"][k])
                b = float(self._posterior_samples["beta"][k])
                alpha = a0 + a1 * s_k
                q_pred = np.clip(np.exp(alpha / (b + 1e-8) * (1 - np.exp(-b * t_q))) - 1, 0, 1)
            elif self._growth_function_name == "weighted_sigmoid":
                w1 = float(self._posterior_samples["w1"][k])
                w2 = float(self._posterior_samples["w2"][k])
                bias = float(self._posterior_samples["b"][k])
                sigmoid = 1.0 / (1.0 + np.exp(-(w1 * s_k + w2 * t_q + bias)))
                q_pred = np.clip(t_q * sigmoid, 0, 1)
            else:
                q_pred = np.zeros(n_pred)

            pred_growth = self._qt.inverse_growth(q_pred)
            pred_ensemble[k, :] = baseline + pred_growth

        # Summarize posterior predictive
        mean = np.mean(pred_ensemble, axis=0)
        variance = np.var(pred_ensemble, axis=0)
        lower = np.percentile(pred_ensemble, 2.5, axis=0)
        upper = np.percentile(pred_ensemble, 97.5, axis=0)

        return PredictionResult(
            mean=mean,
            variance=variance,
            lower_95=lower,
            upper_95=upper,
        )

    def name(self) -> str:
        """Human-readable model name."""
        return f"BayesianSeverity({self._growth_function_name})"

    @property
    def posterior_samples(self) -> dict[str, np.ndarray] | None:
        """Access raw MCMC posterior samples after fitting."""
        return self._posterior_samples

    @property
    def severity_summary(self) -> dict[str, dict[str, float]] | None:
        """Per-patient severity posterior summary (mean, std, 95% CI)."""
        if self._posterior_samples is None:
            return None

        s_samples = self._posterior_samples["s"]  # [n_samples, n_patients]
        result: dict[str, dict[str, float]] = {}

        for i, pid in enumerate(self._severity_means.keys()):
            s_i = s_samples[:, i]
            result[pid] = {
                "mean": float(np.mean(s_i)),
                "std": float(np.std(s_i)),
                "ci_lower": float(np.percentile(s_i, 2.5)),
                "ci_upper": float(np.percentile(s_i, 97.5)),
            }

        return result

    @property
    def fitted_severities(self) -> dict[str, float] | None:
        """Posterior mean severities (for compatibility with MLE interface)."""
        return self._severity_means if self._severity_means else None
