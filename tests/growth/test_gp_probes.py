# tests/growth/test_gp_probes.py
"""Tests for GP-based probe evaluation.

Tests cover:
- GP-linear / Ridge equivalence (TEST_GP.1)
- Predictive variance positivity (TEST_GP.2)
- R² credible intervals (TEST_GP.3)
- RBF captures nonlinearity (TEST_GP.4)
- LML favors correct kernel (TEST_GP.5)
- SemanticProbes integration (TEST_GP.6)
- High R² on linear data (TEST_GP.7)
- Missing target validation (TEST_GP.8)
- get_summary keys (TEST_GP.9)
"""

import numpy as np
import pytest

from growth.evaluation.gp_probes import (
    GPProbe,
    GPProbeResults,
    GPSemanticProbes,
    GPSemanticResults,
)


def test_gp_linear_matches_ridge():
    """GP with linear kernel produces R² within 0.05 of Ridge regression.

    Mathematical basis: GP-linear posterior mean = Ridge solution when
    alpha = sigma_n^2 / sigma_f^2. Small deviations arise from GP
    hyperparameter optimization vs. fixed alpha=1.0.
    """
    np.random.seed(42)
    n, d, k = 500, 64, 4
    X = np.random.randn(n, d)
    W = np.random.randn(d, k) * 0.1
    y = X @ W + np.random.randn(n, k) * 0.01

    # Ridge baseline
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:400])
    X_test_s = scaler.transform(X[400:])
    ridge = Ridge(alpha=1.0).fit(X_train_s, y[:400])
    r2_ridge = r2_score(y[400:], ridge.predict(X_test_s))

    # GP-linear
    gp = GPProbe(kernel_type="linear", r2_ci_samples=0)
    gp.fit(X[:400], y[:400])
    results = gp.evaluate(X[400:], y[400:])

    assert abs(results.r2 - r2_ridge) < 0.05, (
        f"GP-linear R²={results.r2:.4f} differs from Ridge R²={r2_ridge:.4f} by > 0.05"
    )


def test_predictive_variance_positive():
    """GP predictive std is strictly positive and finite."""
    np.random.seed(42)
    X = np.random.randn(200, 32)
    y = np.random.randn(200, 3)

    for kernel in ["linear", "rbf"]:
        gp = GPProbe(kernel_type=kernel, r2_ci_samples=0, n_restarts=1)
        gp.fit(X[:160], y[:160])
        results = gp.evaluate(X[160:], y[160:])

        assert results.predictive_std.shape == (40, 3)
        assert np.all(results.predictive_std > 0), f"{kernel}: found non-positive predictive std"
        assert np.all(np.isfinite(results.predictive_std)), (
            f"{kernel}: found non-finite predictive std"
        )


def test_r2_ci_brackets_point_estimate():
    """R² credible interval is non-degenerate and point R² >= CI lower bound.

    The CI samples from the predictive distribution (mean + noise), so
    sampled R² values are generally lower than the point R² from the mean.
    The point R² should be at or above ci_lo.
    """
    np.random.seed(42)
    n, d = 300, 32
    X = np.random.randn(n, d)
    y = X[:, :4] + np.random.randn(n, 4) * 0.5

    gp = GPProbe(kernel_type="linear", r2_ci_samples=1000)
    gp.fit(X[:240], y[:240])
    results = gp.evaluate(X[240:], y[240:])

    assert results.r2_ci_lo < results.r2_ci_hi, "CI is degenerate"
    assert np.isfinite(results.r2_ci_lo) and np.isfinite(results.r2_ci_hi), "CI bounds not finite"
    assert results.r2 >= results.r2_ci_lo, (
        f"Point R²={results.r2:.4f} below CI lower bound {results.r2_ci_lo:.4f}"
    )


def test_rbf_better_than_linear_for_nonlinear_data():
    """GP-RBF achieves higher R² than GP-linear on nonlinear data."""
    np.random.seed(42)
    n, d = 500, 16
    X = np.random.randn(n, d)
    y = np.sin(X[:, 0:1] * 2) + np.cos(X[:, 1:2]) + np.random.randn(n, 1) * 0.1

    gp_lin = GPProbe(kernel_type="linear", r2_ci_samples=0)
    gp_rbf = GPProbe(kernel_type="rbf", n_restarts=3, r2_ci_samples=0)
    gp_lin.fit(X[:400], y[:400])
    gp_rbf.fit(X[:400], y[:400])
    res_lin = gp_lin.evaluate(X[400:], y[400:])
    res_rbf = gp_rbf.evaluate(X[400:], y[400:])

    assert res_rbf.r2 > res_lin.r2 + 0.1, (
        f"RBF R²={res_rbf.r2:.4f} not substantially better than linear R²={res_lin.r2:.4f}"
    )


def test_lml_favors_correct_kernel():
    """LML(RBF) > LML(linear) for nonlinear data."""
    np.random.seed(42)
    n, d = 400, 16
    X = np.random.randn(n, d)

    # Nonlinear data
    y_nonlinear = np.sin(X[:, 0:1] * 3) + np.random.randn(n, 1) * 0.1

    gp_lin = GPProbe(kernel_type="linear", r2_ci_samples=0)
    gp_rbf = GPProbe(kernel_type="rbf", n_restarts=3, r2_ci_samples=0)
    gp_lin.fit(X, y_nonlinear)
    gp_rbf.fit(X, y_nonlinear)
    res_lin = gp_lin.evaluate(X, y_nonlinear)
    res_rbf = gp_rbf.evaluate(X, y_nonlinear)

    delta_lml = res_rbf.log_marginal_likelihood - res_lin.log_marginal_likelihood
    assert delta_lml > 0, (
        f"RBF LML ({res_rbf.log_marginal_likelihood:.2f}) should exceed "
        f"linear LML ({res_lin.log_marginal_likelihood:.2f}) for nonlinear data"
    )


def test_semantic_probes_fit_evaluate():
    """GPSemanticProbes fits and evaluates with correct output structure."""
    np.random.seed(42)
    X = np.random.randn(200, 768)
    targets = {
        "volume": np.random.randn(200, 4),
        "location": np.random.randn(200, 3),
        "shape": np.random.randn(200, 3),
    }

    probes = GPSemanticProbes(input_dim=768, n_restarts=1, r2_ci_samples=0)
    probes.fit(X[:160], {k: v[:160] for k, v in targets.items()})
    results = probes.evaluate(X[160:], {k: v[160:] for k, v in targets.items()})

    assert isinstance(results, GPSemanticResults)
    for name in ["volume", "location", "shape"]:
        assert name in results.linear
        assert name in results.rbf
        assert name in results.nonlinearity_evidence
        assert isinstance(results.linear[name], GPProbeResults)
        assert results.linear[name].kernel_type == "linear"
        assert results.rbf[name].kernel_type == "rbf"


def test_high_r2_for_linear_data():
    """GP probes achieve R² > 0.90 on linearly generated data."""
    np.random.seed(42)
    n, d, k = 500, 64, 4
    X = np.random.randn(n, d)
    W = np.random.randn(d, k) * 0.1
    y = X @ W + np.random.randn(n, k) * 0.01

    gp = GPProbe(kernel_type="linear", r2_ci_samples=0)
    gp.fit(X[:400], y[:400])
    results = gp.evaluate(X[400:], y[400:])

    assert results.r2 > 0.90, f"R²={results.r2:.4f} < 0.90 on clean linear data"


def test_missing_target_raises():
    """GPSemanticProbes raises ValueError for missing targets."""
    probes = GPSemanticProbes(input_dim=768)
    X = np.random.randn(100, 768)
    targets = {"volume": np.random.randn(100, 4)}

    with pytest.raises(ValueError, match="Missing target"):
        probes.fit(X, targets)


def test_get_summary_keys():
    """get_summary returns all expected metric keys."""
    np.random.seed(42)
    X = np.random.randn(200, 64)
    targets = {
        "volume": np.random.randn(200, 4),
        "location": np.random.randn(200, 3),
        "shape": np.random.randn(200, 3),
    }
    probes = GPSemanticProbes(input_dim=64, n_restarts=1, r2_ci_samples=50)
    probes.fit(X[:160], {k: v[:160] for k, v in targets.items()})
    results = probes.evaluate(X[160:], {k: v[160:] for k, v in targets.items()})
    summary = probes.get_summary(results)

    for name in ["volume", "location", "shape"]:
        assert f"r2_{name}_linear" in summary
        assert f"r2_{name}_rbf" in summary
        assert f"r2_{name}_linear_ci_lo" in summary
        assert f"r2_{name}_linear_ci_hi" in summary
        assert f"r2_{name}_rbf_ci_lo" in summary
        assert f"r2_{name}_rbf_ci_hi" in summary
        assert f"lml_{name}_linear" in summary
        assert f"lml_{name}_rbf" in summary
        assert f"nonlinearity_evidence_{name}" in summary
    assert "r2_mean_linear" in summary
    assert "r2_mean_rbf" in summary
