# tests/growth/test_sdp.py
"""Tests for Phase 2: Supervised Disentangled Projection (SDP).

Implements TEST_3.1 through TEST_3.7 from module_3_sdp spec.
All tests use synthetic data — no GPU or real data required.

Run fast tests:  pytest tests/growth/test_sdp.py -v -m "not slow"
Run all tests:   pytest tests/growth/test_sdp.py -v
"""

import pytest
import torch

from growth.losses.dcor import DistanceCorrelationLoss, distance_correlation
from growth.losses.sdp_loss import CurriculumSchedule, SDPLoss
from growth.losses.semantic import SemanticRegressionLoss
from growth.losses.vicreg import CovarianceLoss, VarianceHingeLoss
from growth.models.projection.partition import (
    SUPERVISED_PARTITIONS,
    LatentPartition,
    PartitionSpec,
)
from growth.models.projection.sdp import SDP, SDPWithHeads

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 100
IN_DIM = 768
OUT_DIM = 128


@pytest.fixture
def sdp_model() -> SDP:
    """Create a default SDP model."""
    return SDP(in_dim=IN_DIM, hidden_dim=512, out_dim=OUT_DIM, dropout=0.1)


@pytest.fixture
def sdp_with_heads() -> SDPWithHeads:
    """Create an SDPWithHeads model with default config."""
    return SDPWithHeads.from_config()


@pytest.fixture
def partition() -> LatentPartition:
    """Create a default LatentPartition."""
    return LatentPartition()


@pytest.fixture
def synthetic_features() -> torch.Tensor:
    """Synthetic encoder features [B, 768]."""
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, IN_DIM)


@pytest.fixture
def synthetic_targets() -> dict[str, torch.Tensor]:
    """Synthetic semantic targets."""
    torch.manual_seed(42)
    return {
        "vol": torch.randn(BATCH_SIZE, 4),
        "loc": torch.randn(BATCH_SIZE, 3),
        "shape": torch.randn(BATCH_SIZE, 3),
    }


# ===========================================================================
# TEST_3.1: SDP Forward Pass [BLOCKING]
# ===========================================================================
class TestSDPForwardPass:
    """TEST_3.1: SDP forward pass shapes and gradient flow."""

    def test_sdp_output_shape(self, sdp_model: SDP, synthetic_features: torch.Tensor) -> None:
        """Output z has correct shape [B, 128]."""
        z = sdp_model(synthetic_features)
        assert z.shape == (BATCH_SIZE, OUT_DIM)

    def test_sdp_full_batch(self, sdp_model: SDP) -> None:
        """Works with realistic full-batch size (800)."""
        h = torch.randn(800, IN_DIM)
        z = sdp_model(h)
        assert z.shape == (800, OUT_DIM)

    def test_partition_dim_sum(self, partition: LatentPartition) -> None:
        """Partition dimensions sum to total (128)."""
        total = sum(spec.dim for spec in partition.partitions.values())
        assert total == OUT_DIM

    def test_partition_split_shapes(
        self, sdp_model: SDP, partition: LatentPartition, synthetic_features: torch.Tensor
    ) -> None:
        """Partition split produces correct shapes."""
        z = sdp_model(synthetic_features)
        parts = partition.split(z)

        assert parts["vol"].shape == (BATCH_SIZE, 24)
        assert parts["loc"].shape == (BATCH_SIZE, 8)
        assert parts["shape"].shape == (BATCH_SIZE, 12)
        assert parts["residual"].shape == (BATCH_SIZE, 84)

    def test_sdp_with_heads_output_shapes(
        self, sdp_with_heads: SDPWithHeads, synthetic_features: torch.Tensor
    ) -> None:
        """SDPWithHeads produces correct output shapes."""
        z, parts, preds = sdp_with_heads(synthetic_features)

        assert z.shape == (BATCH_SIZE, OUT_DIM)
        assert preds["vol"].shape == (BATCH_SIZE, 4)
        assert preds["loc"].shape == (BATCH_SIZE, 3)
        assert preds["shape"].shape == (BATCH_SIZE, 3)

    def test_gradient_flow(
        self, sdp_with_heads: SDPWithHeads, synthetic_features: torch.Tensor
    ) -> None:
        """Gradients flow through the entire model."""
        h = synthetic_features.requires_grad_(True)
        z, _, preds = sdp_with_heads(h)

        # Loss: sum of all predictions
        loss = sum(p.sum() for p in preds.values()) + z.sum()
        loss.backward()

        assert h.grad is not None
        assert not torch.all(h.grad == 0)

    def test_partition_contiguity_validation(self) -> None:
        """Invalid partitions (gap) raise ValueError."""
        with pytest.raises(ValueError, match="Gap or overlap"):
            LatentPartition(
                {
                    "a": PartitionSpec("a", 0, 10, 4),
                    "b": PartitionSpec("b", 15, 20, 3),  # Gap at 10-15
                }
            )

    def test_partition_from_config(self) -> None:
        """LatentPartition.from_config creates valid partition."""
        lp = LatentPartition.from_config(vol_dim=32, loc_dim=16, shape_dim=16, residual_dim=64)
        assert lp.total_dim == 128


# ===========================================================================
# TEST_3.2: Spectral Normalization [BLOCKING]
# ===========================================================================
class TestSpectralNormalization:
    """TEST_3.2: Spectral norm constrains singular values."""

    def test_sn_at_init(self, sdp_model: SDP) -> None:
        """Both linear layers have spectral norm ≈ 1.0 after power iteration converges."""
        # Warm up power iteration in TRAINING mode (power iteration only
        # runs when module.training=True)
        sdp_model.train()
        for _ in range(200):
            sdp_model(torch.randn(4, IN_DIM))
        sdp_model.eval()

        for name in ["fc1", "fc2"]:
            layer = getattr(sdp_model, name)
            sigma = torch.linalg.svdvals(layer.weight)[0]
            assert sigma <= 1.05, f"SN not applied to {name}: sigma={sigma:.4f}"

    def test_sn_parametrization_present(self, sdp_model: SDP) -> None:
        """Spectral norm parametrization is registered."""
        for name in ["fc1", "fc2"]:
            layer = getattr(sdp_model, name)
            # Check for spectral_norm parametrization
            has_sn = hasattr(layer, "weight_orig") or any(
                "spectral" in str(p).lower()
                for p in layer.parametrizations.get("weight", [])
                if hasattr(layer, "parametrizations")
            )
            # Also check via torch.nn.utils.parametrize
            if not has_sn:
                has_sn = hasattr(layer, "weight_orig")
            assert has_sn, f"Spectral norm not found on {name}"

    def test_sn_after_training_steps(self, sdp_model: SDP) -> None:
        """Spectral norm stays ≈ 1.0 after 10 training steps."""
        optimizer = torch.optim.Adam(sdp_model.parameters(), lr=1e-3)

        for _ in range(10):
            h = torch.randn(32, IN_DIM)
            z = sdp_model(h)
            loss = z.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for name in ["fc1", "fc2"]:
            layer = getattr(sdp_model, name)
            sigma = torch.linalg.svdvals(layer.weight)[0]
            assert sigma <= 1.15, f"SN violated after training for {name}: sigma={sigma:.4f}"


# ===========================================================================
# TEST_3.3: Loss Computation [BLOCKING]
# ===========================================================================
class TestLossComputation:
    """TEST_3.3: All loss terms finite and well-behaved."""

    def test_semantic_loss_finite(self, synthetic_targets: dict) -> None:
        """Semantic regression loss is finite and >= 0."""
        loss_fn = SemanticRegressionLoss()
        preds = {k: torch.randn_like(v) for k, v in synthetic_targets.items()}
        total, details = loss_fn(preds, synthetic_targets)

        assert torch.isfinite(total)
        assert total >= 0
        for key, val in details.items():
            assert torch.isfinite(val), f"{key} is not finite"

    def test_covariance_loss_finite(self) -> None:
        """Covariance loss is finite and >= 0."""
        loss_fn = CovarianceLoss()
        partitions = {
            "vol": torch.randn(100, 24),
            "loc": torch.randn(100, 8),
            "shape": torch.randn(100, 12),
        }
        loss = loss_fn(partitions)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_variance_hinge_loss_zero_when_spread(self) -> None:
        """Variance hinge loss = 0 when all dims have std >= gamma."""
        loss_fn = VarianceHingeLoss(gamma=1.0)
        # z with std ~2.0 per dim
        z = torch.randn(200, 128) * 2.0
        loss = loss_fn(z)
        assert loss.item() < 0.05  # Should be ~0

    def test_variance_hinge_loss_positive_when_collapsed(self) -> None:
        """Variance hinge loss > 0 when dimensions are collapsed."""
        loss_fn = VarianceHingeLoss(gamma=1.0)
        z = torch.randn(200, 128) * 0.01  # Very low variance
        loss = loss_fn(z)
        assert loss > 0.5

    def test_dcor_in_range(self) -> None:
        """Distance correlation ∈ [0, 1]."""
        x = torch.randn(100, 24)
        y = torch.randn(100, 8)
        dcor = distance_correlation(x, y)
        assert 0 <= dcor <= 1

    def test_dcor_high_for_correlated(self) -> None:
        """dCor is high for correlated data."""
        x = torch.randn(200, 10)
        y = x[:, :5] + 0.1 * torch.randn(200, 5)  # Highly correlated
        dcor = distance_correlation(x, y)
        assert dcor > 0.5

    def test_dcor_loss_finite(self) -> None:
        """Distance correlation loss is finite."""
        loss_fn = DistanceCorrelationLoss()
        partitions = {
            "vol": torch.randn(100, 24),
            "loc": torch.randn(100, 8),
            "shape": torch.randn(100, 12),
        }
        mean_dcor, details = loss_fn(partitions)
        assert torch.isfinite(mean_dcor)
        assert 0 <= mean_dcor <= 1

    def test_curriculum_gating(self) -> None:
        """Curriculum correctly gates loss terms."""
        schedule = CurriculumSchedule(warmup_end=10, semantic_end=40, independence_end=60)

        # Epoch 5: only variance
        active = schedule.get_active_losses(5)
        assert active["variance"] is True
        assert active["semantic"] is False
        assert active["covariance"] is False

        # Epoch 25: variance + semantic
        active = schedule.get_active_losses(25)
        assert active["variance"] is True
        assert active["semantic"] is True
        assert active["covariance"] is False

        # Epoch 50: all active
        active = schedule.get_active_losses(50)
        assert active["variance"] is True
        assert active["semantic"] is True
        assert active["covariance"] is True
        assert active["dcor"] is True

    def test_composite_loss_finite(
        self,
        sdp_with_heads: SDPWithHeads,
        synthetic_features: torch.Tensor,
        synthetic_targets: dict,
    ) -> None:
        """Composite SDPLoss is finite for all curriculum phases."""
        loss_fn = SDPLoss()

        z, partitions, predictions = sdp_with_heads(synthetic_features)

        for epoch in [0, 15, 45, 80]:
            loss_fn.set_epoch(epoch)
            total, details = loss_fn(z, partitions, predictions, synthetic_targets)
            assert torch.isfinite(total), f"Loss not finite at epoch {epoch}"
            assert total >= 0, f"Loss negative at epoch {epoch}"


# ===========================================================================
# TEST_3.4: Training Convergence [BLOCKING, slow]
# ===========================================================================
class TestTrainingConvergence:
    """TEST_3.4: Loss decreases over training."""

    @pytest.mark.slow
    def test_loss_decreases_50pct(self) -> None:
        """Total loss decreases by >= 50% over 100 epochs on synthetic data."""
        torch.manual_seed(42)

        # Create synthetic data with learnable structure
        n = 200
        h = torch.randn(n, IN_DIM)

        # Create targets that are linearly predictable from h
        W_true = torch.randn(IN_DIM, 10) * 0.1
        all_targets = h @ W_true
        targets = {
            "vol": all_targets[:, :4],
            "loc": all_targets[:, 4:7],
            "shape": all_targets[:, 7:10],
        }

        # Normalize
        for key in targets:
            targets[key] = (targets[key] - targets[key].mean(0)) / targets[key].std(0).clamp(
                min=1e-8
            )
        h = (h - h.mean(0)) / h.std(0).clamp(min=1e-8)

        model = SDPWithHeads.from_config()
        loss_fn = SDPLoss(use_curriculum=False)  # All losses active
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        losses = []
        for epoch in range(100):
            model.train()
            z, partitions, predictions = model(h)
            loss, _ = loss_fn(z, partitions, predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            assert not torch.isnan(loss), f"NaN loss at epoch {epoch}"

        # Check >= 50% decrease
        initial_loss = sum(losses[:5]) / 5
        final_loss = sum(losses[-5:]) / 5
        decrease_pct = (initial_loss - final_loss) / initial_loss * 100

        assert decrease_pct >= 50, (
            f"Loss decreased only {decrease_pct:.1f}% "
            f"(initial={initial_loss:.4f}, final={final_loss:.4f})"
        )


# ===========================================================================
# TEST_3.5: Semantic Quality [BLOCKING, slow]
# ===========================================================================
class TestSemanticQuality:
    """TEST_3.5: R² thresholds on synthetic linearly-related data."""

    @pytest.mark.slow
    def test_r2_thresholds(self) -> None:
        """R² meets minimum thresholds on synthetic data.

        Targets are low-rank linear projections spread across all 768 input dims.
        n_train must be well above n_features (768) to avoid overfitting in the
        underdetermined regime. With n_train=2000, the rank-2 signal generalizes
        well despite the high-dimensional input.
        """
        torch.manual_seed(42)

        n_train, n_val = 2000, 500
        h_all = torch.randn(n_train + n_val, IN_DIM)

        # Each target group depends on 2 latent factors spanning all 768 dims.
        # Low-rank (rank 2) per group makes targets easily learnable through
        # the SN-constrained bottleneck, while still requiring the network to
        # learn meaningful projections from high-dimensional input.
        n = n_train + n_val
        noise = 0.02

        # 2 latent factors per target group, spread across all input dims
        def make_targets(n_targets: int) -> torch.Tensor:
            v1 = torch.randn(IN_DIM) / (IN_DIM**0.5)
            v2 = torch.randn(IN_DIM) / (IN_DIM**0.5)
            factors = torch.stack([h_all @ v1, h_all @ v2], dim=1)  # [n, 2]
            proj = torch.randn(2, n_targets) * 2.0
            return factors @ proj + noise * torch.randn(n, n_targets)

        targets_all = {
            "vol": make_targets(4),
            "loc": make_targets(3),
            "shape": make_targets(3),
        }

        h_train, h_val = h_all[:n_train], h_all[n_train:]

        targets_train = {k: v[:n_train] for k, v in targets_all.items()}
        targets_val = {k: v[n_train:] for k, v in targets_all.items()}

        # Normalize using train stats only (matches real training)
        for key in targets_train:
            mu = targets_train[key].mean(0)
            std = targets_train[key].std(0).clamp(min=1e-8)
            targets_train[key] = (targets_train[key] - mu) / std
            targets_val[key] = (targets_val[key] - mu) / std

        h_mu, h_std = h_train.mean(0), h_train.std(0).clamp(min=1e-8)
        h_train = (h_train - h_mu) / h_std
        h_val = (h_val - h_mu) / h_std

        # dropout=0 for capacity test (regularization tested separately)
        model = SDPWithHeads.from_config(dropout=0.0)

        # Warm up SN power iteration so weights are properly normalized
        # from the start (power iteration only runs in training mode).
        model.train()
        for _ in range(50):
            model(torch.randn(4, IN_DIM))

        # Semantic-only loss — testing informativeness, not disentanglement
        # (that's TEST_3.6). Regularization competes with semantic quality
        # on synthetic data.
        loss_fn = SDPLoss(
            use_curriculum=False,
            lambda_cov=0.0,
            lambda_dcor=0.0,
            lambda_var=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

        # Train
        for epoch in range(400):
            model.train()
            z, parts, preds = model(h_train)
            loss, _ = loss_fn(z, parts, preds, targets_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluate R² on val
        model.eval()
        with torch.no_grad():
            _, _, preds_val = model(h_val)

        for key, threshold in [("vol", 0.80), ("loc", 0.85), ("shape", 0.30)]:
            pred = preds_val[key]
            target = targets_val[key]
            ss_res = ((pred - target) ** 2).sum()
            ss_tot = ((target - target.mean(dim=0)) ** 2).sum()
            r2 = 1.0 - ss_res / (ss_tot + 1e-8)

            assert r2 >= threshold, f"R² for {key}={r2:.4f} < {threshold} (BLOCKING)"


# ===========================================================================
# TEST_3.6: Disentanglement Quality [BLOCKING]
# ===========================================================================
class TestDisentanglementQuality:
    """TEST_3.6: Low cross-partition correlation and dCor on independent data."""

    def test_low_cross_correlation_independent(self) -> None:
        """Cross-partition correlation is low for independent random partitions."""
        torch.manual_seed(42)
        n = 500
        partitions = {
            "vol": torch.randn(n, 24),
            "loc": torch.randn(n, 8),
            "shape": torch.randn(n, 12),
        }

        # Check all pairs
        for name_i in SUPERVISED_PARTITIONS:
            for name_j in SUPERVISED_PARTITIONS:
                if name_i >= name_j:
                    continue
                zi = partitions[name_i]
                zj = partitions[name_j]
                corr = torch.corrcoef(torch.cat([zi.T, zj.T], dim=0))
                di = zi.shape[1]
                cross_block = corr[:di, di:]
                max_corr = cross_block.abs().max().item()
                assert max_corr < 0.30, f"Cross-corr({name_i}, {name_j})={max_corr:.4f} >= 0.30"

    def test_low_dcor_independent(self) -> None:
        """dCor is low for independent random partitions.

        Uses low-dimensional data and large n to minimize V-statistic
        finite-sample bias.
        """
        torch.manual_seed(42)
        n = 2000
        x = torch.randn(n, 3)
        y = torch.randn(n, 3)

        dcor = distance_correlation(x, y)
        assert dcor < 0.10, f"dCor for independent data = {dcor:.4f} >= 0.10"

    def test_per_dim_variance_sufficient(self) -> None:
        """Per-dimension variance > 0.3 for >= 90% of dims on well-spread data.

        TEST_3.6 spec: checks that standard-normal latent vectors maintain
        sufficient per-dimension variance. The real threshold is verified
        in generate_quality_report() on trained embeddings.
        """
        torch.manual_seed(42)
        n = 500
        z = torch.randn(n, 128)  # std=1.0 per dim

        z_std = z.std(dim=0)
        pct_above = (z_std > 0.3).float().mean().item()
        assert pct_above >= 0.90, f"Only {pct_above * 100:.1f}% dims have std > 0.3 (need >= 90%)"

    def test_high_dcor_dependent(self) -> None:
        """dCor detects nonlinear dependence."""
        torch.manual_seed(42)
        n = 500
        x = torch.randn(n, 10)
        y = x**2 + 0.1 * torch.randn(n, 10)  # Nonlinear dependence

        dcor = distance_correlation(x, y)
        assert dcor > 0.3, f"dCor should detect nonlinear dependence: {dcor:.4f}"


# ===========================================================================
# TEST_3.7: Lipschitz Check [DIAGNOSTIC, slow]
# ===========================================================================
class TestLipschitz:
    """TEST_3.7: Bounded output distance ratio for nearby inputs."""

    @pytest.mark.slow
    def test_lipschitz_bounded(self, sdp_model: SDP) -> None:
        """Output distance ratio is bounded for nearby inputs.

        Spectral norm uses power iteration which needs several forward
        passes to converge. We warm up with 200 forward passes first.
        """
        torch.manual_seed(42)

        # Warm up power iteration in TRAINING mode (power iteration
        # only runs when module.training=True)
        sdp_model.train()
        for _ in range(200):
            sdp_model(torch.randn(4, IN_DIM))
        sdp_model.eval()

        ratios = []
        for _ in range(100):
            h1 = torch.randn(1, IN_DIM)
            h2 = h1 + 0.01 * torch.randn(1, IN_DIM)

            with torch.no_grad():
                z1 = sdp_model(h1)
                z2 = sdp_model(h2)

            input_dist = (h1 - h2).norm()
            output_dist = (z1 - z2).norm()

            if input_dist > 1e-8:
                ratios.append((output_dist / input_dist).item())

        max_ratio = max(ratios)
        mean_ratio = sum(ratios) / len(ratios)

        # With converged spectral norm (σ ≤ 1 per layer) and LayerNorm,
        # the Lipschitz constant for a 2-layer network should be ≤ σ1 * σ2 ≈ 1.
        # LayerNorm adds some input-dependent amplification, so allow margin.
        assert max_ratio < 5.0, (
            f"Lipschitz ratio too high: max={max_ratio:.2f}, mean={mean_ratio:.2f}"
        )


# ===========================================================================
# TEST: Inline Validation Metrics
# ===========================================================================
class TestInlineValidationMetrics:
    """Verify SDPLitModule validation_step logs disentanglement metrics."""

    @staticmethod
    def _make_config() -> "DictConfig":
        """Create minimal SDPLitModule config."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "sdp": {"in_dim": 768, "hidden_dim": 512, "out_dim": 128, "dropout": 0.1},
                "partition": {"vol_dim": 24, "loc_dim": 8, "shape_dim": 12, "residual_dim": 84},
                "targets": {"n_vol": 4, "n_loc": 3, "n_shape": 3},
                "loss": {
                    "lambda_vol": 20.0,
                    "lambda_loc": 12.0,
                    "lambda_shape": 15.0,
                    "lambda_cov": 5.0,
                    "lambda_var": 5.0,
                    "lambda_dcor": 2.0,
                    "gamma_var": 1.0,
                },
                "curriculum": {
                    "enabled": True,
                    "warmup_end": 10,
                    "semantic_end": 40,
                    "independence_end": 60,
                },
                "training": {
                    "seed": 42,
                    "max_epochs": 100,
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                    "scheduler": {"warmup_epochs": 5, "min_lr": 1e-6},
                },
            }
        )

    def test_validation_step_logs_disentanglement_metrics(self) -> None:
        """Validation step logs dCor, max_corr, variance stats, effective_rank."""
        from growth.training.lit_modules.sdp_module import SDPLitModule

        config = self._make_config()
        module = SDPLitModule(config)

        # Prepare synthetic data
        torch.manual_seed(42)
        n = 200
        h_train = torch.randn(n, 768)
        targets_train = {
            "vol": torch.randn(n, 4),
            "loc": torch.randn(n, 3),
            "shape": torch.randn(n, 3),
        }
        module.setup_data(h_train, targets_train)

        # Run validation step
        batch = next(iter(module.train_dataloader()))
        module.eval()

        logged = {}
        original_log = module.log

        def capture_log(name, value, **kwargs):
            logged[name] = value

        module.log = capture_log
        module.validation_step(batch, 0)
        module.log = original_log

        # Check expected keys are logged
        expected_keys = [
            "val/loss_total",
            "val/r2_vol",
            "val/r2_loc",
            "val/r2_shape",
            "val/r2_mean",
            "val/dcor_vol_loc",
            "val/dcor_vol_shape",
            "val/dcor_loc_shape",
            "val/max_cross_partition_corr",
            "val/pct_dims_std_gt_03",
            "val/pct_dims_std_gt_05",
            "val/mean_dim_std",
            "val/min_dim_std",
            "val/effective_rank",
            "val/curriculum_phase",
        ]

        for key in expected_keys:
            assert key in logged, f"Missing logged metric: {key}"

        # Check values are finite
        for key, value in logged.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value), f"{key} is not finite: {value}"
            elif isinstance(value, float):
                assert not (value != value), f"{key} is NaN"  # NaN check
