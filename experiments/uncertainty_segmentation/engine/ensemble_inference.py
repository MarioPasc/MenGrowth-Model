# experiments/uncertainty_segmentation/engine/ensemble_inference.py
"""Ensemble inference engine for LoRA-Ensemble uncertainty segmentation.

Loads M independently-trained LoRA adapters on a shared BrainSegFounder backbone
and produces ensemble predictions with Welford online aggregation for memory
efficiency.

Key design decisions:
    - Fresh base model reload per member (PeftModel.from_pretrained mutates base)
    - Welford algorithm for running mean/variance (avoids storing M prob maps)
    - Binary entropy per sigmoid channel (not categorical softmax)
    - Volume from WT channel (ch1) > 0.5 threshold

References:
    Welford, B.P. (1962). Note on a Method for Calculating Corrected Sums
        of Squares and Products. Technometrics, 4(3), 419-420.
    Mühlematter et al. (2024). LoRA-Ensemble. arXiv:2405.14438.
"""

import dataclasses
import logging
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from growth.inference.sliding_window import sliding_window_segment
from growth.models.encoder.lora_adapter import LoRASwinViT
from growth.models.encoder.swin_loader import load_full_swinunetr
from growth.models.segmentation.original_decoder import LoRAOriginalDecoderModel

from .paths import get_run_dir
from .uncertainty_metrics import compute_binary_entropy, compute_mutual_information

logger = logging.getLogger(__name__)


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclasses.dataclass
class EnsemblePrediction:
    """Result of ensemble prediction for a single scan.

    All spatial tensors are on CPU to avoid GPU memory accumulation.

    Attributes:
        mean_probs: Mean sigmoid probabilities [C, D, H, W].
        var_probs: Per-voxel variance across members [C, D, H, W].
        predictive_entropy: Binary entropy of mean probs [C, D, H, W].
        mutual_information: Epistemic uncertainty [C, D, H, W].
        ensemble_mask: WT binary mask from mean probs [D, H, W].
        per_member_volumes: Volume (mm³) per ensemble member.
        volume_mean: Mean volume across members (mm³).
        volume_std: Std of volume across members (mm³).
        log_volume_mean: Mean of log(V+1) across members.
        log_volume_std: Std of log(V+1) across members.
        volume_median: Median volume (mm³) — robust to outlier members.
        volume_mad: Median absolute deviation of volume (mm³).
        log_volume_median: Median of log(V+1).
        log_volume_mad: MAD of log(V+1).
        n_members: Number of ensemble members used.
        per_member_probs: Per-member probability maps (None unless requested).
        per_member_masks: Per-member WT masks (None unless requested).
    """

    mean_probs: torch.Tensor
    var_probs: torch.Tensor
    predictive_entropy: torch.Tensor
    mutual_information: torch.Tensor
    ensemble_mask: torch.Tensor
    per_member_volumes: list[float]
    volume_mean: float
    volume_std: float
    log_volume_mean: float
    log_volume_std: float
    volume_median: float
    volume_mad: float
    log_volume_median: float
    log_volume_mad: float
    n_members: int
    per_member_probs: list[torch.Tensor] | None = None
    per_member_masks: list[torch.Tensor] | None = None


# =============================================================================
# Ensemble Predictor
# =============================================================================


class EnsemblePredictor:
    """Loads M LoRA adapters and produces uncertainty-aware ensemble predictions.

    Usage:
        predictor = EnsemblePredictor(config, device="cuda")
        result = predictor.predict_scan(images)  # images: [1, 4, D, H, W]

    Args:
        config: Full experiment configuration.
        device: Device for inference.
    """

    def __init__(
        self,
        config: DictConfig,
        device: str = "cuda",
        run_dir: Path | str | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.checkpoint_path = str(
            Path(config.paths.checkpoint_dir) / config.paths.checkpoint_filename
        )
        resolved_run_dir = get_run_dir(config, override=run_dir)
        self.adapter_base_dir = resolved_run_dir / "adapters"
        self.n_members = config.ensemble.n_members

        # Sliding window config
        self.sw_roi_size = tuple(config.inference.sw_roi_size)
        self.sw_batch_size = config.inference.sw_batch_size
        self.sw_overlap = config.inference.sw_overlap
        self.sw_mode = config.inference.sw_mode

        # Discover trained members
        self.available_members = self._discover_members()
        logger.info(
            f"EnsemblePredictor: {len(self.available_members)}/{self.n_members} "
            f"members available"
        )

    def _discover_members(self) -> list[int]:
        """Find all member directories with completed training.

        A member is considered complete if both adapter/adapter_config.json
        and decoder.pt exist.

        Returns:
            Sorted list of available member IDs.
        """
        available = []
        for m in range(self.n_members):
            member_dir = self.adapter_base_dir / f"member_{m}"
            adapter_config = member_dir / "adapter" / "adapter_config.json"
            decoder_pt = member_dir / "decoder.pt"
            if adapter_config.exists() and decoder_pt.exists():
                available.append(m)
            else:
                logger.warning(f"Member {m} incomplete: {member_dir}")
        return sorted(available)

    def _load_member_model(self, member_id: int) -> nn.Module:
        """Load a complete model for one ensemble member.

        Reloads the base SwinUNETR from checkpoint each time because
        PeftModel.from_pretrained() mutates the base model in-place.

        Args:
            member_id: Ensemble member index.

        Returns:
            LoRAOriginalDecoderModel in eval mode on self.device.
        """
        member_dir = self.adapter_base_dir / f"member_{member_id}"

        # 1. Load fresh base model with pretrained weights
        full_model = load_full_swinunetr(
            self.checkpoint_path,
            freeze_encoder=True,
            freeze_decoder=True,
            out_channels=self.config.training.get("out_channels", 3),
            device="cpu",  # Load on CPU first
        )

        # 2. Load LoRA adapter onto base model
        adapter_path = str(member_dir / "adapter")
        lora_encoder = LoRASwinViT.load_lora(
            base_encoder=full_model,
            adapter_path=adapter_path,
            device="cpu",
            trainable=False,
        )

        # 3. Create segmentation model
        model = LoRAOriginalDecoderModel(
            lora_encoder=lora_encoder,
            freeze_decoder=True,
            out_channels=self.config.training.get("out_channels", 3),
            use_semantic_heads=False,
        )

        # 4. Load decoder weights
        decoder_state = torch.load(
            member_dir / "decoder.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.decoder.load_state_dict(decoder_state)

        # Move to device and set eval
        model = model.to(self.device)
        model.eval()

        logger.debug(f"Loaded member {member_id} from {member_dir}")
        return model

    def predict_scan(
        self,
        images: torch.Tensor,
        save_per_member: bool = False,
    ) -> EnsemblePrediction:
        """Run ensemble prediction on a single scan volume.

        Uses Welford's online algorithm for running mean and variance,
        avoiding storage of M full-resolution probability maps.

        Args:
            images: Input volume [1, 4, D, H, W] on any device.
                Will be moved to self.device internally.
            save_per_member: If True, retain per-member probability maps and
                masks on CPU. Increases memory (~85MB per member for 192³).
                Used for sample scans that need detailed thesis figures.

        Returns:
            EnsemblePrediction with all statistics on CPU.
        """
        images = images.to(self.device)
        M = len(self.available_members)
        assert M >= 1, "No trained ensemble members available"

        _, C_in, D, H, W = images.shape
        C = self.config.training.get("out_channels", 3)  # TC, WT, ET

        # Welford accumulators (on device for speed)
        mean_probs = torch.zeros(C, D, H, W, device=self.device)
        M2_probs = torch.zeros(C, D, H, W, device=self.device)
        mean_member_entropy = torch.zeros(C, D, H, W, device=self.device)

        per_member_volumes: list[float] = []
        collected_probs: list[torch.Tensor] = []
        collected_masks: list[torch.Tensor] = []

        for idx, member_id in enumerate(self.available_members):
            logger.info(f"  Running member {member_id} ({idx + 1}/{M})...")

            # Load model
            model = self._load_member_model(member_id)

            # Sliding window inference → logits
            with torch.no_grad():
                logits = sliding_window_segment(
                    model=model,
                    images=images,
                    roi_size=self.sw_roi_size,
                    sw_batch_size=self.sw_batch_size,
                    overlap=self.sw_overlap,
                    mode=self.sw_mode,
                )  # [1, C, D, H, W]

            # Sigmoid → probabilities (explicitly float32 for Welford stability)
            probs_m = torch.sigmoid(logits).float().squeeze(0)  # [C, D, H, W]

            # Welford online update for mean and variance
            n = idx + 1
            delta = probs_m - mean_probs
            mean_probs += delta / n
            delta2 = probs_m - mean_probs
            M2_probs += delta * delta2

            # Per-member binary entropy (for MI computation)
            h_m = compute_binary_entropy(probs_m)  # [C, D, H, W]
            mean_member_entropy += h_m / M

            # Volume from WT channel (ch1) hard mask
            wt_mask = probs_m[1] > 0.5
            vol_m = float(wt_mask.sum().item())  # 1mm³ isotropic → mm³
            per_member_volumes.append(vol_m)

            # Optionally save per-member spatial data on CPU
            if save_per_member:
                collected_probs.append(probs_m.cpu())
                collected_masks.append(wt_mask.cpu())

            # Cleanup GPU memory
            del model, logits, probs_m, h_m, delta, delta2
            torch.cuda.empty_cache()

        # Finalize statistics
        var_probs = M2_probs / max(1, M - 1) if M > 1 else torch.zeros_like(M2_probs)

        # Predictive entropy and mutual information
        predictive_entropy = compute_binary_entropy(mean_probs)
        mutual_info = compute_mutual_information(predictive_entropy, mean_member_entropy)

        # Ensemble mask from mean probabilities
        ensemble_mask = mean_probs[1] > 0.5  # WT channel

        # Volume statistics (mean/std)
        log_volumes = [math.log(v + 1) for v in per_member_volumes]

        if M > 1:
            vol_mean = sum(per_member_volumes) / M
            vol_std = (
                sum((v - vol_mean) ** 2 for v in per_member_volumes) / (M - 1)
            ) ** 0.5
            logvol_mean = sum(log_volumes) / M
            logvol_std = (
                sum((lv - logvol_mean) ** 2 for lv in log_volumes) / (M - 1)
            ) ** 0.5
        else:
            vol_mean = per_member_volumes[0]
            vol_std = 0.0
            logvol_mean = log_volumes[0]
            logvol_std = 0.0

        # Robust volume statistics (median/MAD)
        vol_median = float(statistics.median(per_member_volumes))
        vol_mad = float(
            statistics.median([abs(v - vol_median) for v in per_member_volumes])
        )
        logvol_median = float(statistics.median(log_volumes))
        logvol_mad = float(
            statistics.median([abs(lv - logvol_median) for lv in log_volumes])
        )

        return EnsemblePrediction(
            mean_probs=mean_probs.cpu(),
            var_probs=var_probs.cpu(),
            predictive_entropy=predictive_entropy.cpu(),
            mutual_information=mutual_info.cpu(),
            ensemble_mask=ensemble_mask.cpu(),
            per_member_volumes=per_member_volumes,
            volume_mean=vol_mean,
            volume_std=vol_std,
            log_volume_mean=logvol_mean,
            log_volume_std=logvol_std,
            volume_median=vol_median,
            volume_mad=vol_mad,
            log_volume_median=logvol_median,
            log_volume_mad=logvol_mad,
            n_members=M,
            per_member_probs=collected_probs if save_per_member else None,
            per_member_masks=collected_masks if save_per_member else None,
        )
