# Module 5: Growth Prediction (Phase 4)

## Overview
Train and evaluate three growth prediction models of increasing complexity on the SDP latent trajectories from the Andalusian longitudinal cohort. All models produce volume predictions via decoding through the frozen semantic head $\pi_{\text{vol}}$.

## Key Design Decisions

1. **Three-model GP hierarchy:** LME → H-GP → PA-MOGP (see D16)
2. **LOPO-CV:** Leave-One-Patient-Out with 33 folds (see D17)
3. **Population linear mean:** GP prior mean from LME fixed effects (see D18)
4. **Hierarchical hyperparameter sharing:** Shared kernel hyperparameters, patient-specific posteriors (see D19)
5. **Rank-1 cross-partition coupling:** PA-MOGP coupling via $\mathbf{B}_{\text{cross}} = \mathbf{w}\mathbf{w}^\top$ (see D20)
6. **Frozen residual:** $z_{\text{res}}^{84}$ carried forward from $t_0$ (unchanged from original rationale)

## Input
- `trajectories.json` from Module 4 (per-patient latent trajectories, 33 patients, 100 studies)
- `phase2_sdp.pt` from Module 3 (for frozen semantic head $\pi_{\text{vol}}$: $\mathbb{R}^{24} \to \mathbb{R}^4$)

## Input Contract
```python
# Trajectories from Module 4
trajectories: List[dict]  # 33 patients, each with ≥2 timepoints
# Each trajectory:
# {"patient_id": str, "timepoints": [{"z": [128 floats], "t": float, "date": str}, ...]}

# Semantic head (for volume decoding)
pi_vol: nn.Linear  # shape [24, 4], from trained SDP semantic heads

# Normalization parameters (from SDP training, D14)
vol_mean: np.ndarray  # shape [4], semantic target means
vol_std: np.ndarray   # shape [4], semantic target stds
```

## Output
- `lme_results.json` — LME fixed/random effects, per-patient predictions, LOPO metrics
- `hgp_results.json` — H-GP hyperparameters, per-patient predictions with uncertainty, LOPO metrics
- `pamogp_results.json` — PA-MOGP hyperparameters, cross-partition coupling, predictions with uncertainty, LOPO metrics
- `model_comparison.json` — Head-to-head metrics ($R^2$, MAE, calibration) for all three models
- `growth_figures/` — Figures 9–13

## Output Contract
```python
# Per-model results (structure shared across all three)
model_result: dict = {
    "model_name": str,                            # "LME" | "H-GP" | "PA-MOGP"
    "hyperparameters": dict,                      # model-specific
    "lopo_predictions": List[dict],               # per-patient predictions
    # Each prediction:
    # {
    #   "patient_id": str,
    #   "observed_times": List[float],
    #   "predicted_times": List[float],
    #   "z_vol_predicted": List[List[float]],     # shape [n_pred, 24]
    #   "z_vol_predicted_std": List[List[float]], # shape [n_pred, 24] (GP models only)
    #   "v_predicted": List[List[float]],         # shape [n_pred, 4] (decoded volumes)
    #   "v_actual": List[List[float]],            # shape [n_pred, 4]
    # }
    "metrics": {
        "vol_r2": float,
        "vol_mae": float,
        "latent_mse": float,
        "calibration_95": float,                  # GP models only
        "per_patient_r": List[float],             # for patients with n_i >= 3
    },
}

# Model comparison
comparison: dict = {
    "models": ["LME", "H-GP", "PA-MOGP"],
    "vol_r2": [float, float, float],
    "vol_mae": [float, float, float],
    "calibration": [None, float, float],
    "best_model": str,
    "coupling_improvement": float,                # PA-MOGP R² - H-GP R²
}
```

## The Three Models

### Model A — Baseline: Linear Mixed-Effects Model (LME) on $z_{\text{vol}}$

For patient $i$, latent dimension $d \in \{1, \ldots, 24\}$, at time $t_{ij}$ (months from first scan):

$$z_{i}^{(d)}(t_{ij}) = (\beta_0^{(d)} + b_{0i}^{(d)}) + (\beta_1^{(d)} + b_{1i}^{(d)}) \cdot t_{ij} + \epsilon_{ij}^{(d)}$$

Random effects: $(b_{0i}, b_{1i}) \sim \mathcal{N}(0, \Omega^{(d)})$, residual: $\epsilon \sim \mathcal{N}(0, \sigma^{(d)2})$.

**Parameter count:** 6 per dimension × 24 = 144 total. Estimated from all 100 observations via REML.

**Single-patient inference:** BLUP with automatic shrinkage for patients with few observations.

### Model B — Literature-Backed: Hierarchical Gaussian Process (H-GP) on $z_{\text{vol}}$

Per-dimension GP with population linear mean from LME and Matérn-5/2 temporal kernel:

$$z_i^{(d)}(t) \sim \mathcal{GP}(m^{(d)}(t), k^{(d)}(t, t'))$$

$$m^{(d)}(t) = \hat{\beta}_0^{(d)} + \hat{\beta}_1^{(d)} \cdot t$$

$$k_{\text{M52}}(\Delta t) = \sigma_f^2 \left(1 + \frac{\sqrt{5}\Delta t}{\ell} + \frac{5\Delta t^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}\Delta t}{\ell}\right)$$

**Hyperparameters:** 3 shared per dimension ($\sigma_f, \ell, \sigma_n$) × 24 = 72 total. Estimated via pooled marginal likelihood across all 33 patients.

**Single-patient inference:** Posterior conditioning provides calibrated uncertainty that reverts to population mean with extrapolation distance.

### Model C — Novel Contribution: Partition-Aware Multi-Output GP (PA-MOGP)

Active latent subspace: $\tilde{z} = [z_{\text{vol}}^{24} | z_{\text{loc}}^{8} | z_{\text{shape}}^{12}] \in \mathbb{R}^{44}$

Partition-specific temporal kernels:

| Partition | Kernel | Rationale |
|---|---|---|
| $z_{\text{vol}}$ (24 dims) | Matérn-5/2 | Smooth, twice-differentiable growth |
| $z_{\text{loc}}$ (8 dims) | Squared Exponential | Very slow, infinitely smooth centroid drift |
| $z_{\text{shape}}$ (12 dims) | Matérn-3/2 | Once-differentiable; shape can change abruptly |

Composite kernel with cross-partition coupling (Intrinsic Coregionalization Model):

$$\mathbf{K}(t, t') = \mathbf{B}_{\text{vol}}^{\text{diag}} \otimes k_{\text{vol}} + \mathbf{B}_{\text{loc}}^{\text{diag}} \otimes k_{\text{loc}} + \mathbf{B}_{\text{shape}}^{\text{diag}} \otimes k_{\text{shape}} + \mathbf{B}_{\text{cross}} \otimes k_{\text{vol}} + \sigma_n^2 I_{44} \delta(t, t')$$

where $\mathbf{B}_{\text{cross}} = \mathbf{w}\mathbf{w}^\top$ with $\mathbf{w} \in \mathbb{R}^{44}$ (rank-1 coupling, D20).

**Parameter count:** 95 total. Estimated from pooled multivariate marginal likelihood (4,400 observation-dimension pairs).

## Code Requirements

1. **`TrajectoryDataset`** — Loads and organizes latent trajectories.
   ```python
   @dataclass
   class PatientTrajectory:
       """Single patient's longitudinal latent data."""
       patient_id: str
       times: np.ndarray          # shape [n_i], months from first scan
       z_vol: np.ndarray          # shape [n_i, 24], volume partition
       z_active: np.ndarray       # shape [n_i, 44], vol+loc+shape (for PA-MOGP)
       z_full: np.ndarray         # shape [n_i, 128], full latent (for storage)

   class TrajectoryDataset:
       """Loads trajectories.json and organizes into PatientTrajectory objects."""
       def __init__(self, trajectories_path: str):
           ...
       def get_patient(self, patient_id: str) -> PatientTrajectory:
           ...
       def get_all(self) -> List[PatientTrajectory]:
           ...
       def lopo_split(self, held_out_id: str) -> Tuple[List[PatientTrajectory], PatientTrajectory]:
           """Returns (train_patients, test_patient)."""
   ```

2. **`LMEGrowthModel`** — Linear Mixed-Effects baseline.
   ```python
   class LMEGrowthModel:
       """Per-dimension LME: z_d(t) = (β₀ + b₀ᵢ) + (β₁ + b₁ᵢ)·t + ε"""
       def fit(self, patients: List[PatientTrajectory]) -> None:
           """Fit 24 independent LME models via REML."""
       def predict(self, patient: PatientTrajectory, t_pred: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
           """Returns (z_vol_mean [n_pred, 24], z_vol_std [n_pred, 24])."""
   ```

3. **`HierarchicalGPModel`** — Per-dimension GP with shared hyperparameters.
   ```python
   class HierarchicalGPModel:
       """Per-dimension GP with population linear mean, Matérn-5/2 kernel,
       hierarchical hyperparameter sharing across patients."""
       def fit(self, patients: List[PatientTrajectory],
               lme_model: LMEGrowthModel) -> None:
           """Fit shared hyperparameters via pooled marginal likelihood."""
       def predict(self, patient: PatientTrajectory, t_pred: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
           """Returns (z_vol_mean [n_pred, 24], z_vol_std [n_pred, 24])."""
   ```

4. **`PAMOGPModel`** — Partition-Aware Multi-Output GP.
   ```python
   class PAMOGPModel:
       """Multi-output GP with partition-specific kernels and
       rank-1 cross-partition coupling."""
       def fit(self, patients: List[PatientTrajectory],
               lme_model: LMEGrowthModel) -> None:
           """Fit all hyperparameters via pooled multivariate marginal likelihood."""
       def predict(self, patient: PatientTrajectory, t_pred: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
           """Returns (z_active_mean [n_pred, 44], z_active_cov [n_pred, 44, 44])."""
       def get_coupling_weights(self) -> np.ndarray:
           """Returns w ∈ ℝ^44 from the rank-1 coupling term."""
   ```

5. **`VolumeDecoder`** — Decodes latent predictions to physical volumes.
   ```python
   class VolumeDecoder:
       """Decodes z_vol predictions through frozen π_vol, with uncertainty propagation."""
       def __init__(self, pi_vol: nn.Linear, vol_mean: np.ndarray, vol_std: np.ndarray):
           ...
       def decode(self, z_vol_mean: np.ndarray, z_vol_std: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
           """Returns (V_mean [n, 4], V_std [n, 4]) in original scale."""
   ```

6. **`LOPOEvaluator`** — Leave-One-Patient-Out evaluation loop.
   ```python
   class LOPOEvaluator:
       """Runs LOPO-CV for a given model, computes all metrics."""
       def evaluate(self, model_class: type, dataset: TrajectoryDataset,
                    decoder: VolumeDecoder, **model_kwargs) -> dict:
           """Returns full results dict matching output contract."""
   ```

## Training Configuration

| Parameter | Value |
|---|---|
| Cross-validation | Leave-One-Patient-Out (33 folds) |
| LME optimizer | REML (via `statsmodels.MixedLM`) |
| GP hyperparameter optimizer | L-BFGS-B (via `scipy.optimize.minimize`) |
| GP kernel (H-GP) | Matérn-5/2 (default; SE and Matérn-3/2 as ablation A9) |
| GP mean function | Population linear from LME (D18) |
| Time unit | Months from first scan |
| Random seed | 42 |

## Configuration Snippet
```yaml
# configs/phase4_growth.yaml
growth:
  time_unit: months
  seed: 42

lme:
  optimizer: reml

hgp:
  kernel: matern52
  mean_function: linear_from_lme
  optimizer: lbfgsb
  max_iter: 1000
  n_restarts: 5

pamogp:
  vol_kernel: matern52
  loc_kernel: se
  shape_kernel: matern32
  coupling_rank: 1
  optimizer: lbfgsb
  max_iter: 2000
  n_restarts: 5

evaluation:
  cv: lopo
  metrics: [vol_r2, vol_mae, latent_mse, calibration_95, per_patient_r]
  prediction_horizon: all
```

## Smoke Test
```python
import numpy as np

# Synthetic trajectories
trajectories = [
    {"patient_id": f"P{i:03d}",
     "timepoints": [
         {"z": np.random.randn(128).tolist(), "t": 0.0, "date": "2020-01-01"},
         {"z": np.random.randn(128).tolist(), "t": 12.0, "date": "2021-01-01"},
     ]}
    for i in range(33)
]

# Check trajectory structure
assert len(trajectories) == 33
assert all(len(t["timepoints"]) >= 2 for t in trajectories)
assert all(t["timepoints"][1]["t"] > t["timepoints"][0]["t"] for t in trajectories)

# Check volume partition extraction
z_vol = np.array(trajectories[0]["timepoints"][0]["z"][:24])
assert z_vol.shape == (24,)
```

## Verification Tests

```
TEST_5.1: LME fitting [BLOCKING]
  - Fit LME on all 33 patients for 1 latent dimension
  - Assert β₀, β₁ are finite
  - Assert Ω is positive semi-definite
  - Assert σ² > 0
  Recovery: Check for singular covariance (reduce to random intercept only)

TEST_5.2: LME prediction [BLOCKING]
  - For a held-out patient with n_i = 2: predict at t₂ from t₁
  - Assert prediction is finite and within 3σ of the population mean
  - Assert prediction with n_i = 4 has smaller error than n_i = 2 (shrinkage)
  Recovery: Check BLUP computation, verify random effects extraction

TEST_5.3: GP hyperparameter fitting [BLOCKING]
  - Fit shared hyperparameters for 1 dimension
  - Assert ℓ > 0, σ_f > 0, σ_n > 0
  - Assert ℓ is in plausible range (1–120 months)
  - Assert log-marginal-likelihood is finite
  Recovery: Initialize hyperparameters from data range; add jitter to kernel diagonal

TEST_5.4: GP predictive distribution [BLOCKING]
  - Condition GP on 3 observations, predict at intermediate and extrapolation points
  - Assert predictive mean interpolates through observations (up to noise)
  - Assert predictive variance at observed times < variance at extrapolation times
  - Assert 95% CI contains observations
  Recovery: Check kernel matrix conditioning; increase σ_n lower bound

TEST_5.5: PA-MOGP cross-partition coupling [BLOCKING]
  - Fit PA-MOGP on all patients
  - Assert coupling weights w ∈ ℝ^44 are finite
  - Assert B_cross = ww^T is positive semi-definite
  - Assert full kernel matrix K ∈ ℝ^{44n × 44n} is positive definite for all patients
  Recovery: Add diagonal jitter; reduce rank or disable coupling

TEST_5.6: LOPO-CV completeness [BLOCKING]
  - Run LOPO-CV for LME model (fastest)
  - Assert 33 folds completed
  - Assert per-fold predictions are finite
  - Assert aggregated R² is finite
  Recovery: Check for patients causing singular fits; exclude and document

TEST_5.7: Volume decoding [BLOCKING]
  - Decode predicted z_vol through π_vol
  - Assert decoded volumes have correct shape [n, 4]
  - Assert decoded volumes are in plausible physical range
  Recovery: Check normalization parameters (vol_mean, vol_std from D14)

TEST_5.8: Uncertainty calibration [DIAGNOSTIC]
  - For GP models: compute fraction of true values within 95% CI
  - Report coverage (target: 0.90–0.98)
  Note: DIAGNOSTIC — calibration is an evaluation metric, not a correctness check
```

## Failure Recovery (GP fitting issues)
If GP hyperparameter fitting diverges or produces degenerate results:
1. Add jitter to kernel diagonal: `K += 1e-6 * I`
2. Constrain length-scale bounds: `ℓ ∈ [1, 120]` months
3. Increase number of random restarts for L-BFGS-B
4. Fall back to simpler kernel (Matérn-5/2 → SE)
5. For PA-MOGP: disable cross-partition coupling (reduces to independent per-partition GPs)
6. If all fail, report diagnostics with per-dimension fits and stop

## References

- Laird, N. M. & Ware, J. H. "Random-Effects Models for Longitudinal Data," *Biometrics*, 1982.
- Robinson, G. K. "That BLUP Is a Good Thing," *Statistical Science*, 1991.
- Rasmussen, C. E. & Williams, C. K. I. *Gaussian Processes for Machine Learning*, MIT Press, 2006.
- Bonilla, E. V. et al. "Multi-task Gaussian Process Prediction," *NeurIPS*, 2007.
- Álvarez, M. A. et al. "Kernels for Vector-Valued Functions: A Review," *FnTML*, 2012.
- Schulam, P. & Saria, S. "Individualizing Predictions of Disease Trajectories," *NeurIPS*, 2015.
