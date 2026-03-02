# src/growth/evaluation/gp_probes.py
"""Gaussian Process probes for encoder feature quality evaluation.

Replaces both Ridge regression (linear) and MLP (nonlinear) probes with
a unified Bayesian framework using GP regression under two kernels:

- **GP-Linear**: Mathematically equivalent to Ridge regression, but
  additionally provides posterior predictive variance and log-marginal
  likelihood. The GP posterior mean with linear kernel k(x,x') = σ_f² x'x
  and noise σ_n² recovers Ridge with α = σ_n² / σ_f².

- **GP-RBF**: Replaces MLP probes. The RBF kernel is a universal
  approximator with automatic complexity control via marginal likelihood
  optimization (no manual hyperparameter tuning).

Key advantages over Ridge/MLP:
- Identical point-estimate R² (GP-linear ≈ Ridge)
- Posterior predictive variance (uncertainty per test point)
- Log-marginal likelihood for principled kernel comparison
- R² credible intervals for rigorous condition comparison
- Sausage plots for visualization

References:
    Rasmussen & Williams, "Gaussian Processes for Machine Learning", 2006.
    Alain & Bengio, "Understanding intermediate layers using linear
    classifier probes", ICLR 2017 Workshop.
"""

import logging
from dataclasses import dataclass, field

import GPy
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class GPProbeResults:
    """Results from a single GP probe evaluation.

    Attributes:
        r2: Point-estimate R² (from posterior mean predictions).
        r2_per_dim: Per-target-dimension R².
        r2_ci_lo: Lower bound of 95% credible interval on R².
        r2_ci_hi: Upper bound of 95% credible interval on R².
        mse: Mean squared error (from posterior mean).
        predictions: Posterior mean predictions [N_test, output_dim].
        predictive_std: Posterior std per test point [N_test, output_dim].
        log_marginal_likelihood: Model evidence log p(y|X,θ*).
        kernel_type: 'linear' or 'rbf'.
        optimized_params: Dict of optimized kernel hyperparameters.
    """

    r2: float
    r2_per_dim: np.ndarray
    r2_ci_lo: float
    r2_ci_hi: float
    mse: float
    predictions: np.ndarray
    predictive_std: np.ndarray
    log_marginal_likelihood: float
    kernel_type: str
    optimized_params: dict = field(default_factory=dict)


@dataclass
class GPSemanticResults:
    """Aggregated results across all semantic targets and both kernels.

    Attributes:
        linear: Dict mapping target name -> GPProbeResults for linear kernel.
        rbf: Dict mapping target name -> GPProbeResults for RBF kernel.
        nonlinearity_evidence: Dict mapping target name -> delta LML (RBF - linear).
    """

    linear: dict  # {name: GPProbeResults}
    rbf: dict  # {name: GPProbeResults}
    nonlinearity_evidence: dict  # {name: float}


class GPProbe:
    """Gaussian Process regression probe for feature quality evaluation.

    Fits an independent GP per target dimension. Supports linear and RBF
    kernels with automatic hyperparameter optimization via log-marginal
    likelihood maximization.

    Mathematical equivalence: GP with linear kernel and noise variance
    sigma_n^2 recovers Ridge regression with alpha = sigma_n^2 / sigma_f^2.
    The GP additionally provides predictive variance for uncertainty
    quantification.

    Args:
        kernel_type: 'linear' or 'rbf'.
        normalize_features: Whether to standardize input features.
        normalize_targets: Whether to standardize target variables.
        n_restarts: Number of random restarts for RBF hyperparameter optimization.
        r2_ci_samples: Number of posterior samples for R² credible intervals.
            Set to 0 to skip CI computation.
    """

    def __init__(
        self,
        kernel_type: str = "linear",
        normalize_features: bool = True,
        normalize_targets: bool = True,
        n_restarts: int = 3,
        r2_ci_samples: int = 500,
    ):
        if kernel_type not in ("linear", "rbf"):
            raise ValueError(f"kernel_type must be 'linear' or 'rbf', got '{kernel_type}'")

        self.kernel_type = kernel_type
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        self.n_restarts = n_restarts
        self.r2_ci_samples = r2_ci_samples

        self._feature_scaler: StandardScaler | None = None
        self._target_scalers: list[StandardScaler] = []
        self._models: list[GPy.models.GPRegression] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPProbe":
        """Fit independent GPs per target dimension.

        Args:
            X: Encoder features [N, D].
            y: Target semantic features [N, K].

        Returns:
            Self for method chaining.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        assert X.shape[0] == y.shape[0], (
            f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}"
        )

        n_samples, input_dim = X.shape
        output_dim = y.shape[1]

        # Standardize features
        if self.normalize_features:
            self._feature_scaler = StandardScaler()
            X_scaled = self._feature_scaler.fit_transform(X)
        else:
            self._feature_scaler = None
            X_scaled = X.copy()

        # Fit independent GP per target dimension
        self._models = []
        self._target_scalers = []

        for k in range(output_dim):
            y_k = y[:, k : k + 1]  # Keep 2D for GPy

            # Standardize target
            if self.normalize_targets:
                t_scaler = StandardScaler()
                y_k_scaled = t_scaler.fit_transform(y_k)
                self._target_scalers.append(t_scaler)
            else:
                y_k_scaled = y_k.copy()
                self._target_scalers.append(None)

            # Build kernel
            kernel = self._build_kernel(input_dim)

            # Create GP model
            model = GPy.models.GPRegression(X_scaled, y_k_scaled, kernel)
            model.Gaussian_noise.variance.constrain_bounded(1e-6, 10.0)

            # Optimize hyperparameters
            self._optimize_model(model)

            self._models.append(model)

            logger.debug(
                f"GP({self.kernel_type}) dim {k}: "
                f"LML={model.log_likelihood():.2f}, "
                f"noise_var={float(model.Gaussian_noise.variance):.4f}"
            )

        self._fitted = True
        logger.debug(
            f"GPProbe({self.kernel_type}) fitted: {n_samples} samples, "
            f"{input_dim}D -> {output_dim}D"
        )
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> GPProbeResults:
        """Evaluate GP probe on test data.

        Args:
            X: Encoder features [N_test, D].
            y: Target semantic features [N_test, K].

        Returns:
            GPProbeResults with R², uncertainty, and predictions.
        """
        if not self._fitted:
            raise RuntimeError("GPProbe must be fitted before evaluation")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_test = X.shape[0]
        output_dim = y.shape[1]

        assert len(self._models) == output_dim, (
            f"Model output dim ({len(self._models)}) != test target dim ({output_dim})"
        )

        # Scale features
        if self._feature_scaler is not None:
            X_scaled = self._feature_scaler.transform(X)
        else:
            X_scaled = X

        # Collect predictions and variances
        predictions = np.zeros((n_test, output_dim))
        predictive_std = np.zeros((n_test, output_dim))
        total_lml = 0.0
        optimized_params = {}

        for k, model in enumerate(self._models):
            mu_k, var_k = model.predict(X_scaled)
            std_k = np.sqrt(np.clip(var_k, 1e-12, None))

            # De-standardize predictions
            t_scaler = self._target_scalers[k] if self._target_scalers else None
            if t_scaler is not None:
                mu_k = t_scaler.inverse_transform(mu_k)
                std_k = std_k * t_scaler.scale_

            predictions[:, k] = mu_k.ravel()
            predictive_std[:, k] = std_k.ravel()
            total_lml += model.log_likelihood()

            # Collect optimized params
            optimized_params[f"dim_{k}"] = {
                "noise_var": float(model.Gaussian_noise.variance),
                "lml": float(model.log_likelihood()),
            }

        # Compute R²
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

        # Per-dimension R²
        r2_per_dim = np.array(
            [
                float(
                    1.0
                    - np.sum((y[:, k] - predictions[:, k]) ** 2)
                    / (np.sum((y[:, k] - y[:, k].mean()) ** 2) + 1e-12)
                )
                for k in range(output_dim)
            ]
        )

        # MSE
        mse = float(np.mean((y - predictions) ** 2))

        # R² credible intervals
        r2_ci_lo, r2_ci_hi = self._compute_r2_ci(predictions, predictive_std, y)

        return GPProbeResults(
            r2=r2,
            r2_per_dim=r2_per_dim,
            r2_ci_lo=r2_ci_lo,
            r2_ci_hi=r2_ci_hi,
            mse=mse,
            predictions=predictions,
            predictive_std=predictive_std,
            log_marginal_likelihood=float(total_lml),
            kernel_type=self.kernel_type,
            optimized_params=optimized_params,
        )

    def _build_kernel(self, input_dim: int) -> GPy.kern.Kern:
        """Build GPy kernel.

        Args:
            input_dim: Feature dimensionality.

        Returns:
            GPy kernel instance.
        """
        if self.kernel_type == "linear":
            return GPy.kern.Linear(input_dim, ARD=False)
        else:
            return GPy.kern.RBF(input_dim, ARD=False)

    def _optimize_model(self, model: GPy.models.GPRegression) -> None:
        """Optimize GP hyperparameters via log-marginal likelihood.

        Args:
            model: GPy regression model to optimize.
        """
        if self.kernel_type == "linear":
            # Linear kernel is convex — single optimization suffices
            model.optimize(messages=False, max_iters=500)
        else:
            # RBF kernel may have local optima — use restarts
            if self.n_restarts > 0:
                model.optimize_restarts(
                    num_restarts=self.n_restarts,
                    verbose=False,
                    max_iters=500,
                    robust=True,
                )
            else:
                model.optimize(messages=False, max_iters=500)

    def _compute_r2_ci(
        self,
        predictions: np.ndarray,
        predictive_std: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[float, float]:
        """Compute R² credible intervals via posterior sampling.

        Args:
            predictions: Posterior mean [N, K].
            predictive_std: Posterior std [N, K].
            y_test: Ground truth [N, K].

        Returns:
            Tuple of (ci_lo, ci_hi) for 95% credible interval.
        """
        if self.r2_ci_samples <= 0:
            # Skip CI computation
            r2 = float(
                1.0
                - np.sum((y_test - predictions) ** 2)
                / (np.sum((y_test - y_test.mean(axis=0)) ** 2) + 1e-12)
            )
            return r2, r2

        ss_tot = np.sum((y_test - y_test.mean(axis=0)) ** 2)
        rng = np.random.RandomState(42)
        r2_samples = np.zeros(self.r2_ci_samples)

        for i in range(self.r2_ci_samples):
            y_sample = predictions + predictive_std * rng.randn(*predictions.shape)
            ss_res = np.sum((y_test - y_sample) ** 2)
            r2_samples[i] = 1.0 - ss_res / (ss_tot + 1e-12)

        ci_lo = float(np.percentile(r2_samples, 2.5))
        ci_hi = float(np.percentile(r2_samples, 97.5))
        return ci_lo, ci_hi


class GPSemanticProbes:
    """GP-based semantic probes for all feature types.

    Manages paired GP probes (linear + RBF) for volume, location, and shape.
    Computes nonlinearity evidence as delta log-marginal likelihood.

    Args:
        input_dim: Feature dimensionality (768 for encoder10).
        normalize_features: Standardize features before GP fitting.
        normalize_targets: Standardize targets before GP fitting.
        n_restarts: Random restarts for RBF optimization.
        r2_ci_samples: Posterior samples for R² credible intervals.
    """

    FEATURE_DIMS = {
        "volume": 4,
        "location": 3,
        "shape": 3,
    }

    def __init__(
        self,
        input_dim: int = 768,
        normalize_features: bool = True,
        normalize_targets: bool = True,
        n_restarts: int = 3,
        r2_ci_samples: int = 500,
    ):
        self.input_dim = input_dim
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        self.n_restarts = n_restarts
        self.r2_ci_samples = r2_ci_samples

        self._linear_probes: dict[str, GPProbe] = {}
        self._rbf_probes: dict[str, GPProbe] = {}

    def fit(
        self,
        X: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> "GPSemanticProbes":
        """Fit all probes on training data.

        Args:
            X: Encoder features [N, input_dim].
            targets: Dict mapping feature names to target arrays.
                Must contain 'volume', 'location', and 'shape'.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If any required target is missing.
        """
        for name in self.FEATURE_DIMS:
            if name not in targets:
                raise ValueError(f"Missing target: {name}")

        for name in self.FEATURE_DIMS:
            logger.info(f"Fitting GP probes for {name}...")

            self._linear_probes[name] = GPProbe(
                kernel_type="linear",
                normalize_features=self.normalize_features,
                normalize_targets=self.normalize_targets,
                r2_ci_samples=self.r2_ci_samples,
            )
            self._linear_probes[name].fit(X, targets[name])

            self._rbf_probes[name] = GPProbe(
                kernel_type="rbf",
                normalize_features=self.normalize_features,
                normalize_targets=self.normalize_targets,
                n_restarts=self.n_restarts,
                r2_ci_samples=self.r2_ci_samples,
            )
            self._rbf_probes[name].fit(X, targets[name])

        logger.info(
            f"Fitted {len(self.FEATURE_DIMS)} semantic GP probe pairs on {X.shape[0]} samples"
        )
        return self

    def evaluate(
        self,
        X: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> GPSemanticResults:
        """Evaluate all probes on test data.

        Args:
            X: Encoder features [N, input_dim].
            targets: Dict mapping feature names to target arrays.

        Returns:
            GPSemanticResults with linear, rbf, and nonlinearity evidence.
        """
        linear_results = {}
        rbf_results = {}
        nonlinearity_evidence = {}

        for name in self.FEATURE_DIMS:
            if name not in targets:
                raise ValueError(f"Missing target: {name}")

            linear_results[name] = self._linear_probes[name].evaluate(X, targets[name])
            rbf_results[name] = self._rbf_probes[name].evaluate(X, targets[name])

            # Nonlinearity evidence: delta LML (RBF - linear)
            delta_lml = (
                rbf_results[name].log_marginal_likelihood
                - linear_results[name].log_marginal_likelihood
            )
            nonlinearity_evidence[name] = float(delta_lml)

        return GPSemanticResults(
            linear=linear_results,
            rbf=rbf_results,
            nonlinearity_evidence=nonlinearity_evidence,
        )

    def get_summary(self, results: GPSemanticResults) -> dict[str, float]:
        """Get flat summary dict compatible with JSON logging.

        Args:
            results: Results from evaluate().

        Returns:
            Dict with all metric keys.
        """
        summary: dict[str, float] = {}

        for name in results.linear:
            lin = results.linear[name]
            rbf = results.rbf[name]

            summary[f"r2_{name}_linear"] = lin.r2
            summary[f"r2_{name}_rbf"] = rbf.r2
            summary[f"r2_{name}_linear_ci_lo"] = lin.r2_ci_lo
            summary[f"r2_{name}_linear_ci_hi"] = lin.r2_ci_hi
            summary[f"r2_{name}_rbf_ci_lo"] = rbf.r2_ci_lo
            summary[f"r2_{name}_rbf_ci_hi"] = rbf.r2_ci_hi
            summary[f"mse_{name}_linear"] = lin.mse
            summary[f"mse_{name}_rbf"] = rbf.mse
            summary[f"lml_{name}_linear"] = lin.log_marginal_likelihood
            summary[f"lml_{name}_rbf"] = rbf.log_marginal_likelihood
            summary[f"nonlinearity_evidence_{name}"] = results.nonlinearity_evidence[name]

        # Averages
        summary["r2_mean_linear"] = float(np.mean([r.r2 for r in results.linear.values()]))
        summary["r2_mean_rbf"] = float(np.mean([r.r2 for r in results.rbf.values()]))

        return summary


def extract_sausage_data(
    results: GPProbeResults,
    y_test: np.ndarray,
    target_dim: int = 0,
) -> dict[str, np.ndarray]:
    """Extract data for sausage plots.

    Sorts test subjects by ground-truth value for the given target dimension
    and returns predictions with 95% prediction bands.

    Args:
        results: GPProbeResults from evaluate().
        y_test: Ground-truth target values [N_test, K].
        target_dim: Which target dimension to extract.

    Returns:
        Dict with:
            'y_true': Ground-truth values sorted.
            'y_pred_mean': GP posterior mean, same sort order.
            'y_pred_lo': Lower 95% prediction band.
            'y_pred_hi': Upper 95% prediction band.
            'sort_idx': Sort indices for reconstructing original order.
    """
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    y_true = y_test[:, target_dim]
    y_pred = results.predictions[:, target_dim]
    y_std = results.predictive_std[:, target_dim]

    sort_idx = np.argsort(y_true)

    return {
        "y_true": y_true[sort_idx],
        "y_pred_mean": y_pred[sort_idx],
        "y_pred_lo": y_pred[sort_idx] - 2.0 * y_std[sort_idx],
        "y_pred_hi": y_pred[sort_idx] + 2.0 * y_std[sort_idx],
        "sort_idx": sort_idx,
    }
