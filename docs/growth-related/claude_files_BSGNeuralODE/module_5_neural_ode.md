# Module 5: Neural ODE Growth Forecasting (Phase 4)

## Overview
Train a partition-aware Neural ODE with Gompertz-informed volume dynamics on decoded physical volumes, operating on forward-only temporal pairs from the Andalusian longitudinal cohort.

## Key Design Decisions (from review fixes)

1. **Decode-then-model:** Gompertz operates on decoded physical volumes V̂ ∈ ℝ⁴, NOT on raw latent dims (see D9)
2. **Forward-only pairs:** ~155 pairs with Gaussian perturbation augmentation (see D8)
3. **Frozen residual:** dz_res/dt = 0, effective ODE is 44-dim (see D10)

## Input
- `trajectories.json` from Module 4
- `phase2_sdp.pt` from Module 3 (for semantic head π_vol)

## Input Contract
```python
# Trajectories from Module 4
trajectories: List[dict]  # 42 patients, each with ≥2 timepoints
# Each trajectory:
# {"patient_id": str, "timepoints": [{"z": [128 floats], "t": float, "date": str}, ...]}

# Semantic head (for decode-then-model)
pi_vol: nn.Linear  # shape [24, 4], from trained SDP semantic heads

# Total forward pairs
n_pairs: int  # ≈ 155
```

## Output
- `ode_model.pt` — Trained Neural ODE parameters
- `ode_trajectories.pt` — Predicted vs actual trajectories for all patients
- `gompertz_params.json` — Per-patient (α, K) extracted from trained model
- `risk_stratification.json` — Per-patient risk scores

## Output Contract
```python
# ODE model outputs
z_pred: torch.Tensor     # shape [N_pairs, 128], predicted z(t1)
z_actual: torch.Tensor   # shape [N_pairs, 128], actual z(t1)

# Gompertz parameters
gompertz: dict = {
    "alpha": float,       # global growth rate (softplus)
    "K": List[float],     # carrying capacities, length 4 (per sub-region)
}

# Risk scores
risk: List[dict] = [
    {"patient_id": str, "risk_score": float, "alpha_hat": float, "K_hat": List[float]},
    ...
]
```

## Code Requirements

1. **`GompertzDynamics`** — Volume ODE function operating on DECODED volumes.
   ```python
   class GompertzDynamics(nn.Module):
       def __init__(self, n_vol: int = 4, n_correction: int = 32):
           self.log_alpha = nn.Parameter(torch.zeros(1))    # softplus → α > 0
           self.log_K = nn.Parameter(torch.zeros(n_vol))    # softplus → K > 0
           self.correction = nn.Sequential(                  # neural correction
               nn.Linear(n_vol + 44, n_correction),          # V̂ + z_other
               nn.Tanh(),
               nn.Linear(n_correction, n_vol),
           )
           self.eta_vol = 0.01

       def forward(self, V_hat: torch.Tensor, z_other: torch.Tensor) -> torch.Tensor:
           """
           V_hat: [B, 4] — decoded physical volumes
           z_other: [B, 20] — loc(8) + shape(12) partition
           Returns: dV/dt ∈ ℝ^{B×4}

           dV/dt = α · V̂ ⊙ ln(K / (V̂ + ε)) + η · h_θ(V̂, z_other)
           """
           alpha = F.softplus(self.log_alpha)
           K = F.softplus(self.log_K)
           gompertz = alpha * V_hat * torch.log(K / (V_hat + 1e-6))
           correction = self.eta_vol * self.correction(torch.cat([V_hat, z_other], dim=-1))
           return gompertz + correction
   ```

2. **`PartitionODE`** — Full partition-aware ODE function. **44 active dimensions.**
   ```python
   class PartitionODE(nn.Module):
       """
       Partition-aware dynamics:
         - Volume (decoded, 4 dims): Gompertz + neural correction
         - Location (8 dims): MLP modulated by volume
         - Shape (12 dims): MLP modulated by volume
         - Residual (84 dims): FROZEN (dz_res/dt = 0)

       Effective ODE dimension: 24 + 8 + 12 = 44 (residual frozen)
       """
       def __init__(self, pi_vol: nn.Linear):
           self.pi_vol = pi_vol           # frozen semantic head
           self.gompertz = GompertzDynamics()
           self.loc_mlp = nn.Sequential(
               nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 8))  # vol_decoded(4)+loc(8) → 32 → 8
           self.shape_mlp = nn.Sequential(
               nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 12)) # vol_decoded(4)+shape(12) → 32 → 12
           self.eta_loc = 0.01
           self.eta_shape = 0.01

       def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
           """z: [B, 128] → dz/dt: [B, 128]"""
           z_vol = z[:, :24]
           z_loc = z[:, 24:32]
           z_shape = z[:, 32:44]
           # z_res = z[:, 44:]  # frozen, dz_res/dt = 0

           # Decode volume to physical space
           V_hat = self.pi_vol(z_vol)  # [B, 4]

           # Volume dynamics (in decoded space)
           dV = self.gompertz(V_hat, torch.cat([z_loc, z_shape], dim=-1))
           # Map back to latent volume partition via pseudo-inverse
           dz_vol = self._volume_to_latent(dV)  # [B, 24]

           # Location dynamics
           dz_loc = self.eta_loc * self.loc_mlp(torch.cat([V_hat, z_loc], dim=-1))

           # Shape dynamics
           dz_shape = self.eta_shape * self.shape_mlp(torch.cat([V_hat, z_shape], dim=-1))

           # Residual: frozen
           dz_res = torch.zeros_like(z[:, 44:])

           return torch.cat([dz_vol, dz_loc, dz_shape, dz_res], dim=-1)
   ```

3. **`ODEFunc`** — `torchdiffeq`-compatible wrapper for adjoint method.
   ```python
   class ODEFunc(nn.Module):
       """Wrapper for torchdiffeq.odeint_adjoint."""
       def forward(self, t, z):
           return self.partition_ode(t, z)
   ```

4. **`ODELitModule`** (PyTorch Lightning) — Training with LOPO-CV.
   ```python
   class ODELitModule(LightningModule):
       def training_step(self, batch, batch_idx):
           z0, z1_actual, dt = batch
           # Add Gaussian perturbation augmentation
           z0_perturbed = z0 + torch.randn_like(z0) * 0.01
           z1_pred = odeint_adjoint(self.ode_func, z0_perturbed, torch.tensor([0, dt]))
           loss = F.mse_loss(z1_pred[-1], z1_actual) + self.jerk_reg()
           return loss
   ```

5. **`TrajectoryDataset`** — Generates forward temporal pairs.
   ```python
   class TrajectoryDataset(Dataset):
       def __init__(self, trajectories: List[dict]):
           """
           From each patient with n timepoints, generate C(n,2) forward pairs.
           Forward only: t_i < t_j. No reverse pairs.
           Total ≈ 155 pairs from 42 patients.
           """
           self.pairs = []
           for traj in trajectories:
               tps = traj["timepoints"]
               for i in range(len(tps)):
                   for j in range(i+1, len(tps)):
                       self.pairs.append((tps[i]["z"], tps[j]["z"],
                                          tps[j]["t"] - tps[i]["t"]))
   ```

6. **`RiskStratifier`** — Extracts Gompertz parameters and computes risk scores.
   ```python
   class RiskStratifier:
       def stratify(self, ode_model: PartitionODE, trajectories: List[dict]) -> List[dict]:
           """
           Risk_i = (α̂_i - ᾱ) / σ_α
           """
   ```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Cross-validation | Leave-one-patient-out (LOPO) |
| ODE solver | `torchdiffeq.odeint_adjoint` |
| Solver method | `dopri5` |
| rtol / atol | 1e-3 / 1e-3 |
| Epochs | 200 |
| Learning rate | 1e-3 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| λ_smooth (jerk reg) | 0.01 |
| z₀ perturbation σ | 0.01 |
| Time unit | months |

## Loss Function

$$\mathcal{L}_{ODE} = \sum_{(i,t_0,t_1)} ||z^*_{i,t_1} - \hat{z}_{i,t_1}||_2^2 + \lambda_{reg} ||\theta_{ODE}||_2^2 + \lambda_{smooth} \int_{t_0}^{t_1} ||d^2z/dt^2||^2 dt$$

## Configuration Snippet
```yaml
# configs/phase4_ode.yaml
ode:
  solver: dopri5
  rtol: 1e-3
  atol: 1e-3
  adjoint: true
  effective_dim: 44      # residual frozen

dynamics:
  eta_vol: 0.01
  eta_loc: 0.01
  eta_shape: 0.01
  correction_hidden: 32

training:
  epochs: 200
  lr: 1e-3
  optimizer: adamw
  weight_decay: 0.01
  lambda_smooth: 0.01
  cv: lopo              # leave-one-patient-out
  augmentation:
    z0_perturbation_sigma: 0.01
  pairs: forward_only   # NO reverse pairs

risk:
  score_type: standardized_alpha  # (α - μ) / σ
```

## Smoke Test
```python
import torch
from torchdiffeq import odeint

# Synthetic trajectory
z0 = torch.randn(1, 128)
dt = torch.tensor([0.0, 6.0])  # 6 months

# Simple ODE function
def f(t, z):
    dz = torch.zeros_like(z)
    dz[:, :44] = 0.01 * z[:, :44]  # small dynamics for active dims
    return dz

z1 = odeint(f, z0, dt)[-1]
assert z1.shape == (1, 128)
assert not torch.isnan(z1).any()
# Residual should be unchanged
assert torch.allclose(z1[:, 44:], z0[:, 44:], atol=1e-5)
```

## Verification Tests

```
TEST_5.1: ODE forward pass [BLOCKING]
  - Given z0 ∈ ℝ^128 and Δt = 1.0 (year = 12 months)
  - z1 = ODESolve(f_θ, z0, 0, 12.0)
  - Assert z1.shape == [128]
  - Assert ||z1 - z0|| > 0 (dynamics are non-trivial for active dims)
  - Assert no NaN values
  Recovery: Reduce rtol/atol to 1e-4, check for division by zero in Gompertz

TEST_5.2: Gompertz stability [BLOCKING]
  - Initialize V̂ > 0 (all positive decoded volumes)
  - Integrate for 10 years (120 months)
  - Assert V̂ remains bounded (carrying capacity constraint)
  - Assert V̂ > 0 (no negative volumes)
  Recovery steps (in order):
    1. Reduce integration time step (rtol/atol from 1e-3 to 1e-4)
    2. Clamp V̂ to [ε, K_max] after each step
    3. Simplify to Gompertz-only (set η_vol = 0, no neural correction)

TEST_5.3: Forward-only pairs [BLOCKING]
  - Assert all training pairs have Δt > 0
  - Assert no reverse pairs exist in TrajectoryDataset
  - Assert total pairs ≈ 155 (± 10%)
  Recovery: Check TrajectoryDataset pair generation logic

TEST_5.4: Training step [BLOCKING]
  - Run 1 training step on a batch of pairs
  - Assert loss is finite and > 0
  - Assert gradients are nonzero for ODE parameters
  Recovery: Check adjoint method setup, verify torchdiffeq installation

TEST_5.5: Partition dynamics [DIAGNOSTIC]
  - Assert ||dz_vol/dt|| >> ||dz_res/dt|| ≈ 0
  - Assert dz_res/dt ≈ 0 for all residual dimensions (frozen)
  - Assert z_res(t1) ≈ z_res(t0) after integration
  Note: DIAGNOSTIC — confirms frozen residual is working correctly
```

## Failure Recovery (ODE divergence)
If the ODE diverges during training:
1. Reduce integration tolerances: `rtol=1e-4, atol=1e-4`
2. Reduce learning rate to 1e-4
3. Increase jerk regularization: `λ_smooth=0.1`
4. Simplify to Gompertz-only (disable neural correction terms)
5. If all fail, report diagnostics with trajectory plots and stop
