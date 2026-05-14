# tests/growth/test_ensemble_lme.py
"""Unit tests for the ensemble-of-trajectories Bayesian model average (Path B).

Synthetic-data tests with analytically known expected behaviour for
``growth.models.growth.ensemble_lme.EnsembleLMEGrowthModel``:

- The mixture variance equals the law-of-total-variance decomposition
  (within-model + between-model) exactly.
- A degenerate ensemble whose M members are identical reproduces a single
  homoscedastic LME (zero between-model variance).
- The exact mixture quantiles bracket the mixture mean and, for M=1, reduce
  to the Gaussian quantile.
- Misuse (no ensemble, inconsistent sizes, predict-before-fit) raises
  ``EnsembleLMEError``.
"""

import numpy as np
import pytest

from growth.models.growth.ensemble_lme import (
    EnsembleLMEError,
    EnsembleLMEGrowthModel,
    _gaussian_mixture_quantile,
)
from growth.models.growth.lme_model import LMEGrowthModel
from growth.shared.growth_models import PatientTrajectory

pytestmark = [pytest.mark.phase4, pytest.mark.unit]


def _make_patients(
    n_patients: int = 16,
    n_members: int = 8,
    member_noise: float = 0.15,
    seed: int = 0,
    identical_members: bool = False,
) -> list[PatientTrajectory]:
    """Build synthetic log-linear trajectories with an M-member observation ensemble.

    Each patient has a random intercept/slope; ``observation_ensemble`` column
    ``m`` is the truth plus i.i.d. member noise (unless ``identical_members``).
    ``observations`` is the per-scan ensemble mean (the shared y target).
    """
    rng = np.random.default_rng(seed)
    patients: list[PatientTrajectory] = []
    for pid in range(n_patients):
        n = int(rng.integers(3, 5))
        times = np.sort(rng.choice(np.arange(0, 7), size=n, replace=False)).astype(float)
        b0 = 2.0 + rng.normal(0, 0.4)
        b1 = 0.3 + rng.normal(0, 0.08)
        truth = b0 + b1 * times
        if identical_members:
            ensemble = np.repeat(truth[:, None], n_members, axis=1)
        else:
            ensemble = truth[:, None] + rng.normal(0, member_noise, (n, n_members))
        observations = ensemble.mean(axis=1)
        patients.append(
            PatientTrajectory(
                patient_id=f"P{pid:02d}",
                times=times,
                observations=observations,
                observation_ensemble=ensemble,
            )
        )
    return patients


def test_fit_predict_basic_shapes() -> None:
    """fit/predict succeed and return finite, ordered intervals."""
    patients = _make_patients()
    model = EnsembleLMEGrowthModel(n_members=8)
    fit_result = model.fit(patients[:-1])
    assert fit_result.hyperparameters["n_members"] == 8.0
    held_out = patients[-1]
    pred = model.predict(
        held_out, np.array([held_out.times[-1]]), n_condition=held_out.n_timepoints - 1
    )
    assert pred.mean.shape == (1, 1)
    assert np.all(np.isfinite(pred.variance))
    assert pred.lower_95[0, 0] < pred.mean[0, 0] < pred.upper_95[0, 0]


def test_law_of_total_variance_holds_exactly() -> None:
    """Mixture variance == E_m[sigma_m^2] + Var_m[mu_m] to numerical precision."""
    patients = _make_patients()
    model = EnsembleLMEGrowthModel(n_members=8)
    model.fit(patients[:-1])
    held_out = patients[-1]
    pred = model.predict(held_out, held_out.times[1:], n_condition=1)
    md = pred.metadata
    reconstructed = md["within_var"] + md["between_var"]
    np.testing.assert_allclose(pred.variance[:, 0], reconstructed, rtol=1e-9, atol=1e-12)
    # within-model variance is the mean of the component variances
    np.testing.assert_allclose(md["within_var"], md["component_variances"].mean(axis=1), rtol=1e-9)


def test_identical_members_reproduce_single_lme() -> None:
    """If every member is identical, the mixture collapses to one homoscedastic LME."""
    patients = _make_patients(identical_members=True, n_members=6)
    ens_model = EnsembleLMEGrowthModel(n_members=6)
    ens_model.fit(patients[:-1])
    held_out = patients[-1]
    t_pred = np.array([held_out.times[-1]])
    ens_pred = ens_model.predict(held_out, t_pred, n_condition=held_out.n_timepoints - 1)

    single = LMEGrowthModel()
    single.fit(patients[:-1])
    single_pred = single.predict(held_out, t_pred, n_condition=held_out.n_timepoints - 1)

    # Between-model variance must vanish; mean and variance match the single LME.
    np.testing.assert_allclose(ens_pred.metadata["between_var"], 0.0, atol=1e-9)
    np.testing.assert_allclose(ens_pred.mean, single_pred.mean, rtol=1e-6)
    np.testing.assert_allclose(ens_pred.variance, single_pred.variance, rtol=1e-6)


def test_between_variance_grows_with_member_disagreement() -> None:
    """Noisier ensemble members produce a larger between-model variance term."""
    quiet = _make_patients(member_noise=0.05, seed=1)
    loud = _make_patients(member_noise=0.6, seed=1)
    between_quiet = []
    between_loud = []
    for patients, store in ((quiet, between_quiet), (loud, between_loud)):
        model = EnsembleLMEGrowthModel(n_members=8)
        model.fit(patients[:-1])
        held_out = patients[-1]
        pred = model.predict(
            held_out, np.array([held_out.times[-1]]), n_condition=held_out.n_timepoints - 1
        )
        store.append(float(pred.metadata["between_var"][0]))
    assert between_loud[0] > between_quiet[0]


def test_n_members_subset_uses_leading_columns() -> None:
    """A smaller n_members uses the leading ensemble columns and reports it."""
    patients = _make_patients(n_members=8)
    model = EnsembleLMEGrowthModel(n_members=4)
    fit_result = model.fit(patients[:-1])
    assert fit_result.hyperparameters["n_members"] == 4.0
    assert "M=4" in model.name()


def test_missing_ensemble_raises() -> None:
    """Trajectories without observation_ensemble cannot be fitted."""
    plain = [
        PatientTrajectory(patient_id="A", times=np.array([0.0, 1.0, 2.0]), observations=np.zeros(3))
    ]
    model = EnsembleLMEGrowthModel()
    with pytest.raises(EnsembleLMEError):
        model.fit(plain)


def test_predict_before_fit_raises() -> None:
    patients = _make_patients()
    with pytest.raises(EnsembleLMEError):
        EnsembleLMEGrowthModel().predict(patients[0], np.array([1.0]))


def test_n_members_exceeding_available_raises() -> None:
    patients = _make_patients(n_members=5)
    model = EnsembleLMEGrowthModel(n_members=10)
    with pytest.raises(EnsembleLMEError):
        model.fit(patients)


def test_predict_requires_test_ensemble() -> None:
    """A test patient lacking observation_ensemble cannot be predicted."""
    patients = _make_patients(n_members=6)
    model = EnsembleLMEGrowthModel(n_members=6)
    model.fit(patients[:-1])
    plain = PatientTrajectory(
        patient_id="bad", times=np.array([0.0, 1.0, 2.0]), observations=np.zeros(3)
    )
    with pytest.raises(EnsembleLMEError):
        model.predict(plain, np.array([2.0]))


# --------------------------------------------------------------------------
# Gaussian mixture quantile
# --------------------------------------------------------------------------
def test_mixture_quantile_single_component_matches_gaussian() -> None:
    """For M=1 the mixture quantile reduces to the Gaussian quantile."""
    from scipy import stats

    for q in (0.025, 0.5, 0.975):
        got = _gaussian_mixture_quantile(np.array([1.7]), np.array([0.4]), q)
        expected = stats.norm.ppf(q, loc=1.7, scale=0.4)
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_mixture_quantile_is_monotone_in_q() -> None:
    """The mixture quantile is monotone increasing in the target probability."""
    means = np.array([0.0, 1.0, 2.5])
    sigmas = np.array([0.3, 0.5, 0.2])
    q_lo = _gaussian_mixture_quantile(means, sigmas, 0.025)
    q_mid = _gaussian_mixture_quantile(means, sigmas, 0.5)
    q_hi = _gaussian_mixture_quantile(means, sigmas, 0.975)
    assert q_lo < q_mid < q_hi


def test_mixture_quantile_cdf_inversion_is_exact() -> None:
    """Evaluating the mixture CDF at the returned quantile recovers the target q."""
    from scipy import stats

    means = np.array([-1.0, 0.5, 3.0])
    sigmas = np.array([0.4, 0.6, 0.3])
    for q in (0.05, 0.25, 0.9):
        y = _gaussian_mixture_quantile(means, sigmas, q)
        cdf = np.mean(stats.norm.cdf((y - means) / sigmas))
        np.testing.assert_allclose(cdf, q, atol=1e-6)
