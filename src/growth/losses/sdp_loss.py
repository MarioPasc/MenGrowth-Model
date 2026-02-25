# src/growth/losses/sdp_loss.py
"""Combined SDP loss for Phase 2 training.

Aggregates: semantic regression + covariance + variance + dCor losses.
Provides the complete objective for disentangled projection learning.

4-phase curriculum schedule (D5):
  Phase 0-9 (Warm-up):     L_var only
  Phase 10-39 (Semantic):   + L_vol, L_loc, L_shape
  Phase 40-59 (Independence): + L_cov, L_dCor
  Phase 60+ (Full):         All at full strength
"""

import logging

import torch
from torch import nn

from .dcor import DistanceCorrelationLoss
from .semantic import SemanticRegressionLoss
from .vicreg import CovarianceLoss, VarianceHingeLoss

logger = logging.getLogger(__name__)


class CurriculumSchedule:
    """4-phase curriculum schedule for loss term activation.

    Args:
        warmup_end: Epoch when warm-up ends (semantic losses start).
        semantic_end: Epoch when independence losses start.
        independence_end: Epoch when full training begins.

    Example:
        >>> sched = CurriculumSchedule()
        >>> sched.get_active_losses(5)   # {'variance': True, 'semantic': False, ...}
        >>> sched.get_active_losses(25)  # {'variance': True, 'semantic': True, ...}
    """

    def __init__(
        self,
        warmup_end: int = 10,
        semantic_end: int = 40,
        independence_end: int = 60,
    ) -> None:
        self.warmup_end = warmup_end
        self.semantic_end = semantic_end
        self.independence_end = independence_end

    def get_active_losses(self, epoch: int) -> dict[str, bool]:
        """Determine which loss terms are active at given epoch.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Dict with boolean flags for each loss group.
        """
        return {
            "variance": True,  # Always active
            "semantic": epoch >= self.warmup_end,
            "covariance": epoch >= self.semantic_end,
            "dcor": epoch >= self.semantic_end,
        }


class SDPLoss(nn.Module):
    """Composite loss for SDP training with curriculum scheduling.

    Combines:
        - SemanticRegressionLoss (informativeness)
        - CovarianceLoss (linear decorrelation)
        - VarianceHingeLoss (collapse prevention)
        - DistanceCorrelationLoss (nonlinear independence)

    Args:
        lambda_vol: Semantic weight for volume.
        lambda_loc: Semantic weight for location.
        lambda_shape: Semantic weight for shape.
        lambda_cov: Weight for covariance loss.
        lambda_var: Weight for variance hinge loss.
        lambda_dcor: Weight for distance correlation loss.
        gamma_var: Target minimum std for variance hinge.
        cov_partition_names: Partition names to include in covariance penalty.
            Default: supervised partitions only. Set to
            ["vol", "loc", "shape", "residual"] to include residual.
        use_curriculum: Whether to use curriculum scheduling.
        warmup_end: Epoch when warm-up ends.
        semantic_end: Epoch when independence losses start.
        independence_end: Epoch when full training begins.

    Example:
        >>> loss_fn = SDPLoss()
        >>> loss_fn.set_epoch(50)
        >>> total, details = loss_fn(z, partitions, predictions, targets)
    """

    def __init__(
        self,
        lambda_vol: float = 20.0,
        lambda_loc: float = 12.0,
        lambda_shape: float = 15.0,
        lambda_cov: float = 5.0,
        lambda_var: float = 5.0,
        lambda_dcor: float = 2.0,
        gamma_var: float = 1.0,
        cov_partition_names: tuple[str, ...] = ("vol", "loc", "shape"),
        use_curriculum: bool = True,
        warmup_end: int = 10,
        semantic_end: int = 40,
        independence_end: int = 60,
    ) -> None:
        super().__init__()

        self.lambda_cov = lambda_cov
        self.lambda_var = lambda_var
        self.lambda_dcor = lambda_dcor
        self.cov_partition_names = list(cov_partition_names)
        self.use_curriculum = use_curriculum

        # Sub-losses
        self.semantic_loss = SemanticRegressionLoss(
            lambda_vol=lambda_vol,
            lambda_loc=lambda_loc,
            lambda_shape=lambda_shape,
        )
        self.cov_loss = CovarianceLoss()
        self.var_loss = VarianceHingeLoss(gamma=gamma_var)
        self.dcor_loss = DistanceCorrelationLoss()

        # Curriculum
        self.schedule = CurriculumSchedule(
            warmup_end=warmup_end,
            semantic_end=semantic_end,
            independence_end=independence_end,
        )
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for curriculum scheduling.

        Args:
            epoch: Current training epoch.
        """
        self._current_epoch = epoch

    def forward(
        self,
        z: torch.Tensor,
        partitions: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute composite SDP loss.

        Args:
            z: Full latent vector [B, 128].
            partitions: Dict of partition tensors.
            predictions: Dict of semantic predictions from heads.
            targets: Dict of semantic ground truth targets.

        Returns:
            Tuple of (total loss, dict with all individual loss terms).
        """
        details: dict[str, torch.Tensor] = {}

        # Determine active losses
        if self.use_curriculum:
            active = self.schedule.get_active_losses(self._current_epoch)
        else:
            active = {
                "variance": True,
                "semantic": True,
                "covariance": True,
                "dcor": True,
            }

        total_loss = torch.tensor(0.0, device=z.device)

        # Variance hinge (always active)
        l_var = self.var_loss(z)
        details["loss_var"] = l_var.detach()
        total_loss = total_loss + self.lambda_var * l_var

        # Semantic regression
        if active["semantic"]:
            l_sem, sem_details = self.semantic_loss(predictions, targets)
            details["loss_semantic"] = l_sem.detach()
            details.update(sem_details)
            total_loss = total_loss + l_sem
        else:
            details["loss_semantic"] = torch.tensor(0.0, device=z.device)

        # Covariance (configurable partition scope)
        if active["covariance"]:
            l_cov = self.cov_loss(partitions, partition_names=self.cov_partition_names)
            details["loss_cov"] = l_cov.detach()
            total_loss = total_loss + self.lambda_cov * l_cov
        else:
            details["loss_cov"] = torch.tensor(0.0, device=z.device)

        # Distance correlation
        if active["dcor"]:
            l_dcor, dcor_details = self.dcor_loss(partitions)
            details["loss_dcor"] = l_dcor.detach()
            details.update(dcor_details)
            total_loss = total_loss + self.lambda_dcor * l_dcor
        else:
            details["loss_dcor"] = torch.tensor(0.0, device=z.device)

        details["loss_total"] = total_loss.detach()
        details["epoch"] = torch.tensor(float(self._current_epoch), device=z.device)

        return total_loss, details
