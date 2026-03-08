# tests/growth/test_gp_probes_real_data.py
"""Integration tests for GP probes on real BraTS-MEN data.

These tests run a short training loop (2-3 epochs) on real H5 data,
extract features, and verify GP probe evaluation produces valid results.

Requires:
  - BraTS-MEN H5 file (set via MENGROWTH_H5_PATH env var or Picasso default)
  - BrainSegFounder checkpoint (set via MENGROWTH_CKPT_PATH env var or Picasso default)
  - GPU with >= 16GB VRAM

Run via SLURM:
  sbatch slurm/lora_adaptation/run_tests_gp_real.sh

Or locally:
  MENGROWTH_H5_PATH=/path/to/BraTS_MEN.h5 \
  MENGROWTH_CKPT_PATH=/path/to/finetuned_model_fold_0.pt \
  python -m pytest tests/growth/test_gp_probes_real_data.py -v -s
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# ---- Picasso defaults (overridable via env vars) ----
_PICASSO_H5 = (
    "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/meningiomas/brats_men/BraTS_MEN.h5"
)
_PICASSO_CKPT = (
    "/mnt/home/users/tic_163_uma/mpascual/fscratch/"
    "checkpoints/BrainSegFounder_finetuned_BraTS/finetuned_model_fold_0.pt"
)
_LOCAL_H5 = "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/BraTS_MEN.h5"
_LOCAL_CKPT = (
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/"
    "checkpoints/BrainSegFounder_finetuned_BraTS/finetuned_model_fold_0.pt"
)

pytestmark = [pytest.mark.evaluation, pytest.mark.real_data, pytest.mark.slow]


def _resolve_path(env_var: str, picasso_default: str, local_default: str) -> Path | None:
    """Resolve a data path from env var, Picasso default, or local default."""
    explicit = os.environ.get(env_var)
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for candidate in [picasso_default, local_default]:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def _get_h5_path() -> Path | None:
    return _resolve_path("MENGROWTH_H5_PATH", _PICASSO_H5, _LOCAL_H5)


def _get_ckpt_path() -> Path | None:
    return _resolve_path("MENGROWTH_CKPT_PATH", _PICASSO_CKPT, _LOCAL_CKPT)


# Skip entire module if data unavailable
pytestmark = pytest.mark.skipif(
    _get_h5_path() is None or _get_ckpt_path() is None,
    reason="Real H5 data or checkpoint not available",
)


# ---- Fixtures ----


@pytest.fixture(scope="module")
def h5_path() -> Path:
    """Resolved H5 file path."""
    return _get_h5_path()


@pytest.fixture(scope="module")
def ckpt_path() -> Path:
    """Resolved checkpoint path."""
    return _get_ckpt_path()


@pytest.fixture(scope="module")
def device() -> str:
    """Device for compute — GPU if available."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def smoke_config(h5_path: Path, ckpt_path: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build a minimal config for 3-epoch smoke training."""
    output_dir = tmp_path_factory.mktemp("gp_smoke")
    cfg = {
        "experiment": {
            "name": "gp_smoke_test",
            "seed": 42,
            "output_dir": str(output_dir),
        },
        "paths": {
            "checkpoint": str(ckpt_path),
            "data_root": str(h5_path.parent),
            "h5_file": str(h5_path),
        },
        "data": {
            "roi_size": [128, 128, 128],
            "feature_roi_size": [192, 192, 192],
            "spacing": [1.0, 1.0, 1.0],
        },
        "data_splits": {
            "lora_train": 50,
            "lora_val": 20,
            "test": 20,
        },
        "conditions": [
            {
                "name": "baseline_frozen",
                "lora_rank": None,
                "skip_training": True,
                "description": "Frozen BrainSegFounder",
            },
            {
                "name": "lora_r8_full",
                "lora_rank": 8,
                "lora_alpha": 16,
                "use_vicreg": True,
                "lambda_aux_override": 0.3,
                "description": "LoRA r=8 + VICReg",
            },
        ],
        "training": {
            "max_epochs": 3,
            "early_stopping_patience": 3,
            "batch_size": 2,
            "lr_encoder": 1.0e-4,
            "lr_decoder": 5.0e-4,
            "weight_decay": 1.0e-5,
            "num_workers": 4,
            "lora_dropout": 0.1,
            "gradient_clip": 1.0,
            "decoder_type": "original",
            "freeze_decoder": False,
            "use_semantic_heads": True,
            "lambda_aux": 0.1,
            "aux_warmup_epochs": 1,
            "aux_warmup_duration": 2,
            "lambda_var_enc": 5.0,
            "lambda_cov_enc": 1.0,
            "vicreg_gamma": 1.0,
            "lr_warmup_epochs": 1,
            "lr_reduce_factor": 0.5,
            "lr_reduce_patience": 2,
            "use_amp": True,
            "grad_accum_steps": 1,
            "enable_gradient_monitoring": False,
        },
        "loss": {
            "lambda_dice": 1.0,
            "lambda_ce": 1.0,
            "lambda_volume": 1.0,
            "lambda_location": 0.3,
            "lambda_shape": 1.0,
        },
        "probe": {
            "n_restarts": 1,
            "r2_ci_samples": 100,
            "normalize_features": True,
            "normalize_targets": True,
        },
        "feature_extraction": {
            "level": "encoder10",
            "batch_size": 1,
            "pooling_mode": "both",
        },
        "logging": {
            "log_every_n_steps": 10,
            "val_check_interval": 1.0,
        },
    }

    # Write config to disk (needed by pipeline functions)
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    cfg["_config_path"] = str(config_path)

    return cfg


@pytest.fixture(scope="module")
def splits(smoke_config: dict) -> dict:
    """Generate deterministic data splits."""
    from growth.utils.seed import set_seed

    set_seed(smoke_config["experiment"]["seed"])

    from experiments.lora.engine.data_splits import main as generate_splits

    config_path = smoke_config["_config_path"]
    generate_splits(config_path)

    from experiments.lora.engine.data_splits import load_splits

    return load_splits(config_path)


@pytest.fixture(scope="module")
def baseline_frozen_features(
    smoke_config: dict, splits: dict, ckpt_path: Path, device: str
) -> dict:
    """Extract features for baseline_frozen condition."""
    from growth.utils.seed import set_seed

    set_seed(42)

    from experiments.lora.engine.extract_features import extract_features

    extract_features(
        condition_name="baseline_frozen",
        config=smoke_config,
        splits=splits,
        device=device,
    )

    condition_dir = (
        Path(smoke_config["experiment"]["output_dir"]) / "conditions" / "baseline_frozen"
    )
    return {
        "features_probe": torch.load(condition_dir / "features_probe.pt", weights_only=True),
        "targets_probe": torch.load(condition_dir / "targets_probe.pt", weights_only=True),
        "features_test": torch.load(condition_dir / "features_test.pt", weights_only=True),
        "targets_test": torch.load(condition_dir / "targets_test.pt", weights_only=True),
    }


@pytest.fixture(scope="module")
def lora_r8_trained(smoke_config: dict, splits: dict, device: str) -> Path:
    """Train lora_r8_full for 3 epochs and return condition dir."""
    from growth.utils.seed import set_seed

    set_seed(42)

    from experiments.lora.engine.train_condition import train_condition

    train_condition(
        condition_name="lora_r8_full",
        config=smoke_config,
        splits=splits,
        max_epochs=3,
        device=device,
    )

    return Path(smoke_config["experiment"]["output_dir"]) / "conditions" / "lora_r8_full"


@pytest.fixture(scope="module")
def lora_r8_features(smoke_config: dict, splits: dict, lora_r8_trained: Path, device: str) -> dict:
    """Extract features for lora_r8_full condition."""
    from growth.utils.seed import set_seed

    set_seed(42)

    from experiments.lora.engine.extract_features import extract_features

    extract_features(
        condition_name="lora_r8_full",
        config=smoke_config,
        splits=splits,
        device=device,
    )

    condition_dir = lora_r8_trained
    return {
        "features_probe": torch.load(condition_dir / "features_probe.pt", weights_only=True),
        "targets_probe": torch.load(condition_dir / "targets_probe.pt", weights_only=True),
        "features_test": torch.load(condition_dir / "features_test.pt", weights_only=True),
        "targets_test": torch.load(condition_dir / "targets_test.pt", weights_only=True),
    }


# =============================================================================
# Tests
# =============================================================================


class TestBaselineFrozenGPProbes:
    """GP probe evaluation on frozen BrainSegFounder features."""

    def test_feature_shapes(self, baseline_frozen_features: dict) -> None:
        """Extracted features have correct dimensionality."""
        feats_probe = baseline_frozen_features["features_probe"]
        feats_test = baseline_frozen_features["features_test"]

        assert feats_probe.ndim == 2, f"Expected 2D, got {feats_probe.ndim}D"
        assert feats_test.ndim == 2, f"Expected 2D, got {feats_test.ndim}D"
        assert feats_probe.shape[1] == 768, f"Expected 768-dim, got {feats_probe.shape[1]}"
        assert feats_test.shape[1] == 768, f"Expected 768-dim, got {feats_test.shape[1]}"
        assert not torch.any(torch.isnan(feats_probe)), "NaN in probe features"
        assert not torch.any(torch.isnan(feats_test)), "NaN in test features"

    def test_target_shapes(self, baseline_frozen_features: dict) -> None:
        """Semantic targets have expected dimensions."""
        targets = baseline_frozen_features["targets_probe"]

        assert "volume" in targets
        assert "location" in targets
        assert "shape" in targets
        assert targets["volume"].shape[1] == 4
        assert targets["location"].shape[1] == 3
        assert targets["shape"].shape[1] == 3

    def test_gp_linear_probes_run(self, baseline_frozen_features: dict) -> None:
        """GP-linear probes produce valid R² on real frozen features."""
        from growth.evaluation.gp_probes import GPSemanticProbes

        X_probe = baseline_frozen_features["features_probe"].numpy()
        X_test = baseline_frozen_features["features_test"].numpy()
        targets_probe = {
            k: v.numpy() for k, v in baseline_frozen_features["targets_probe"].items() if k != "all"
        }
        targets_test = {
            k: v.numpy() for k, v in baseline_frozen_features["targets_test"].items() if k != "all"
        }

        probes = GPSemanticProbes(input_dim=768, n_restarts=1, r2_ci_samples=100)
        probes.fit(X_probe, targets_probe)
        results = probes.evaluate(X_test, targets_test)
        summary = probes.get_summary(results)

        # Structural checks
        for name in ["volume", "location", "shape"]:
            r2_lin = summary[f"r2_{name}_linear"]
            r2_rbf = summary[f"r2_{name}_rbf"]
            assert np.isfinite(r2_lin), f"r2_{name}_linear is not finite: {r2_lin}"
            assert np.isfinite(r2_rbf), f"r2_{name}_rbf is not finite: {r2_rbf}"
            # R² can be negative for bad fits, but should not be absurdly low
            assert r2_lin > -5.0, f"r2_{name}_linear absurdly low: {r2_lin}"
            assert r2_rbf > -5.0, f"r2_{name}_rbf absurdly low: {r2_rbf}"

        # CIs must be finite
        for name in ["volume", "location", "shape"]:
            ci_lo = summary[f"r2_{name}_linear_ci_lo"]
            ci_hi = summary[f"r2_{name}_linear_ci_hi"]
            assert np.isfinite(ci_lo) and np.isfinite(ci_hi), (
                f"CI bounds not finite for {name}: [{ci_lo}, {ci_hi}]"
            )
            assert ci_lo <= ci_hi, f"CI inverted for {name}: {ci_lo} > {ci_hi}"

        # LML values must be finite
        for name in ["volume", "location", "shape"]:
            lml_lin = summary[f"lml_{name}_linear"]
            lml_rbf = summary[f"lml_{name}_rbf"]
            assert np.isfinite(lml_lin), f"LML linear not finite for {name}"
            assert np.isfinite(lml_rbf), f"LML RBF not finite for {name}"

    def test_predictive_uncertainty_positive(self, baseline_frozen_features: dict) -> None:
        """GP predictive std is positive for all test points on real data."""
        from growth.evaluation.gp_probes import GPProbe

        X_probe = baseline_frozen_features["features_probe"].numpy()
        X_test = baseline_frozen_features["features_test"].numpy()
        y_probe = baseline_frozen_features["targets_probe"]["volume"].numpy()
        y_test = baseline_frozen_features["targets_test"]["volume"].numpy()

        for kernel in ["linear", "rbf"]:
            gp = GPProbe(kernel_type=kernel, n_restarts=1, r2_ci_samples=0)
            gp.fit(X_probe, y_probe)
            results = gp.evaluate(X_test, y_test)

            assert np.all(results.predictive_std > 0), (
                f"{kernel}: non-positive predictive std on real data"
            )
            assert np.all(np.isfinite(results.predictive_std)), (
                f"{kernel}: non-finite predictive std on real data"
            )


class TestLoRAR8GPProbes:
    """GP probe evaluation on LoRA r=8 adapted features (3 epochs)."""

    def test_feature_shapes(self, lora_r8_features: dict) -> None:
        """LoRA-adapted features have correct dimensionality."""
        feats = lora_r8_features["features_probe"]

        assert feats.ndim == 2
        assert feats.shape[1] == 768
        assert not torch.any(torch.isnan(feats))
        assert not torch.any(torch.isinf(feats))

    def test_gp_probes_run(self, lora_r8_features: dict) -> None:
        """GP probes produce valid results on LoRA-adapted features."""
        from growth.evaluation.gp_probes import GPSemanticProbes

        X_probe = lora_r8_features["features_probe"].numpy()
        X_test = lora_r8_features["features_test"].numpy()
        targets_probe = {
            k: v.numpy() for k, v in lora_r8_features["targets_probe"].items() if k != "all"
        }
        targets_test = {
            k: v.numpy() for k, v in lora_r8_features["targets_test"].items() if k != "all"
        }

        probes = GPSemanticProbes(input_dim=768, n_restarts=1, r2_ci_samples=100)
        probes.fit(X_probe, targets_probe)
        results = probes.evaluate(X_test, targets_test)
        summary = probes.get_summary(results)

        # All metrics must be finite
        for key, val in summary.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

        # Nonlinearity evidence must be finite
        for name in ["volume", "location", "shape"]:
            ne = summary[f"nonlinearity_evidence_{name}"]
            assert np.isfinite(ne), f"nonlinearity_evidence_{name} not finite"

    def test_lora_features_differ_from_frozen(
        self, baseline_frozen_features: dict, lora_r8_features: dict
    ) -> None:
        """LoRA adaptation should change the feature distribution."""
        frozen = baseline_frozen_features["features_probe"].numpy()
        lora = lora_r8_features["features_probe"].numpy()

        # Features should not be identical (LoRA changed the encoder)
        assert not np.allclose(frozen, lora, atol=1e-4), (
            "LoRA-adapted features are identical to frozen — adaptation had no effect"
        )

        # Mean absolute difference should be measurable
        mad = np.mean(np.abs(frozen - lora))
        assert mad > 1e-4, f"Mean abs difference too small: {mad}"


class TestFullPipelineProbeEvaluation:
    """Test the evaluate_probes pipeline function on real data."""

    def test_evaluate_probes_pipeline(
        self, smoke_config: dict, baseline_frozen_features: dict
    ) -> None:
        """evaluate_probes_enhanced produces valid JSON output."""
        from experiments.lora.eval.evaluate_probes import evaluate_probes_enhanced

        summary = evaluate_probes_enhanced(
            condition_name="baseline_frozen",
            config=smoke_config,
            device="cpu",
        )

        # Check essential keys exist
        assert "r2_mean_linear" in summary
        assert "r2_mean_rbf" in summary
        assert "variance_mean" in summary

        # Verify saved files
        cond_dir = Path(smoke_config["experiment"]["output_dir"]) / "conditions" / "baseline_frozen"
        assert (cond_dir / "metrics_enhanced.json").exists()
        assert (cond_dir / "metrics.json").exists()
        assert (cond_dir / "probes_gp.pkl").exists()
        assert (cond_dir / "predictions_enhanced.json").exists()

        # Verify metrics.json is valid JSON with expected keys
        with open(cond_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "r2_volume" in metrics
        assert "r2_mean_rbf" in metrics
        assert "variance_mean" in metrics
