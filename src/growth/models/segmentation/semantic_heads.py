# src/growth/models/segmentation/semantic_heads.py
"""Auxiliary semantic prediction heads for Phase 1 LoRA training.

These heads provide additional supervision during encoder adaptation by
predicting semantic features (volume, location, shape) from bottleneck
features. This forces the encoder to learn representations that are
linearly predictive of clinically meaningful attributes.

The auxiliary loss is:
    L_aux = λ_vol * MSE(pred_vol, gt_vol) + λ_loc * MSE(pred_loc, gt_loc)
            + λ_shape * MSE(pred_shape, gt_shape)

This approach is inspired by:
    - Multi-task learning (Caruana, 1997)
    - Auxiliary tasks for representation learning (Liebel & Koerner, 2018)
    - Supervised disentanglement (Locatello et al., 2020)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SemanticHead(nn.Module):
    """Single semantic prediction head.

    A lightweight MLP that predicts semantic features from bottleneck.

    Args:
        input_dim: Dimension of input features.
        output_dim: Dimension of output predictions.
        hidden_dim: Optional hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            # Simple linear projection
            self.net = nn.Linear(input_dim, output_dim)
        else:
            # Two-layer MLP
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict semantic features.

        Args:
            x: Input features [B, input_dim].

        Returns:
            Predictions [B, output_dim].
        """
        return self.net(x)


class AuxiliarySemanticHeads(nn.Module):
    """Collection of auxiliary semantic prediction heads.

    Predicts volume, location, and shape from bottleneck features.
    Used during Phase 1 LoRA training to provide additional supervision.

    Args:
        input_dim: Dimension of bottleneck features (768).
        volume_dim: Number of volume features (4: total, NCR, ED, ET).
        location_dim: Number of location features (3: x, y, z).
        shape_dim: Number of shape features (3: sphericity, enhancement_ratio, infiltration_index).
        hidden_dim: Hidden dimension for MLPs.
        dropout: Dropout rate.

    Example:
        >>> heads = AuxiliarySemanticHeads(input_dim=768)
        >>> features = torch.randn(4, 768)
        >>> preds = heads(features)
        >>> preds['pred_volume'].shape
        torch.Size([4, 4])
    """

    def __init__(
        self,
        input_dim: int = 768,
        volume_dim: int = 4,
        location_dim: int = 3,
        shape_dim: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.volume_head = SemanticHead(input_dim, volume_dim, hidden_dim, dropout)
        self.location_head = SemanticHead(input_dim, location_dim, hidden_dim, dropout)
        self.shape_head = SemanticHead(input_dim, shape_dim, hidden_dim, dropout)

        # Store dimensions
        self.dims = {
            'volume': volume_dim,
            'location': location_dim,
            'shape': shape_dim,
        }

        logger.info(
            f"AuxiliarySemanticHeads: vol({volume_dim}), loc({location_dim}), "
            f"shape({shape_dim}), hidden={hidden_dim}"
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict all semantic features.

        Args:
            features: Bottleneck features [B, 768].

        Returns:
            Dict with 'pred_volume', 'pred_location', 'pred_shape'.
        """
        return {
            'pred_volume': self.volume_head(features),
            'pred_location': self.location_head(features),
            'pred_shape': self.shape_head(features),
        }

    def get_param_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class AuxiliarySemanticLoss(nn.Module):
    """Loss function for auxiliary semantic prediction.

    Computes weighted MSE loss for volume, location, and shape predictions.

    Args:
        lambda_volume: Weight for volume loss.
        lambda_location: Weight for location loss.
        lambda_shape: Weight for shape loss.
        normalize_targets: If True, normalize targets before computing loss.

    Example:
        >>> loss_fn = AuxiliarySemanticLoss(lambda_volume=1.0)
        >>> preds = {'pred_volume': torch.randn(4, 4), ...}
        >>> targets = {'volume': torch.randn(4, 4), ...}
        >>> loss, components = loss_fn(preds, targets)
    """

    def __init__(
        self,
        lambda_volume: float = 1.0,
        lambda_location: float = 1.0,
        lambda_shape: float = 1.0,
        normalize_targets: bool = True,
    ):
        super().__init__()

        self.lambda_volume = lambda_volume
        self.lambda_location = lambda_location
        self.lambda_shape = lambda_shape
        self.normalize_targets = normalize_targets

        # Running statistics for normalization
        self.register_buffer('volume_mean', torch.zeros(4))
        self.register_buffer('volume_std', torch.ones(4))
        self.register_buffer('location_mean', torch.zeros(3))
        self.register_buffer('location_std', torch.ones(3))
        self.register_buffer('shape_mean', torch.zeros(3))
        self.register_buffer('shape_std', torch.ones(3))

        self._stats_initialized = False

    def update_statistics(
        self,
        volume: torch.Tensor,
        location: torch.Tensor,
        shape: torch.Tensor,
    ):
        """Update running statistics for target normalization.

        Should be called once with full training set statistics.

        Args:
            volume: Volume targets [N, 4].
            location: Location targets [N, 3].
            shape: Shape targets [N, 3].
        """
        self.volume_mean = volume.mean(dim=0)
        self.volume_std = volume.std(dim=0).clamp(min=1e-6)
        self.location_mean = location.mean(dim=0)
        self.location_std = location.std(dim=0).clamp(min=1e-6)
        self.shape_mean = shape.mean(dim=0)
        self.shape_std = shape.std(dim=0).clamp(min=1e-6)
        self._stats_initialized = True

        logger.info("Target statistics updated for normalization")

    def _normalize(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize tensor."""
        return (x - mean.to(x.device)) / std.to(x.device)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute auxiliary semantic loss.

        Args:
            predictions: Dict with 'pred_volume', 'pred_location', 'pred_shape'.
            targets: Dict with 'volume', 'location', 'shape'.

        Returns:
            Tuple of (total_loss, component_dict).
        """
        components = {}

        # Volume loss
        pred_vol = predictions['pred_volume']
        tgt_vol = targets['volume']
        if self.normalize_targets and self._stats_initialized:
            tgt_vol = self._normalize(tgt_vol, self.volume_mean, self.volume_std)
        vol_loss = F.mse_loss(pred_vol, tgt_vol)
        components['vol_loss'] = vol_loss.item()

        # Location loss
        pred_loc = predictions['pred_location']
        tgt_loc = targets['location']
        if self.normalize_targets and self._stats_initialized:
            tgt_loc = self._normalize(tgt_loc, self.location_mean, self.location_std)
        loc_loss = F.mse_loss(pred_loc, tgt_loc)
        components['loc_loss'] = loc_loss.item()

        # Shape loss
        pred_shape = predictions['pred_shape']
        tgt_shape = targets['shape']
        if self.normalize_targets and self._stats_initialized:
            tgt_shape = self._normalize(tgt_shape, self.shape_mean, self.shape_std)
        shape_loss = F.mse_loss(pred_shape, tgt_shape)
        components['shape_loss'] = shape_loss.item()

        # Total weighted loss
        total_loss = (
            self.lambda_volume * vol_loss +
            self.lambda_location * loc_loss +
            self.lambda_shape * shape_loss
        )
        components['total'] = total_loss.item()

        return total_loss, components


class MultiScaleSemanticHeads(nn.Module):
    """Semantic heads with multi-scale feature input.

    Takes features from multiple encoder stages for richer predictions.

    Args:
        stage_dims: Tuple of channel dimensions for each stage.
        volume_dim: Number of volume outputs.
        location_dim: Number of location outputs.
        shape_dim: Number of shape outputs.
        hidden_dim: Hidden dimension.
    """

    def __init__(
        self,
        stage_dims: Tuple[int, ...] = (192, 384, 768),  # layers2, 3, 4
        volume_dim: int = 4,
        location_dim: int = 3,
        shape_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        total_dim = sum(stage_dims)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Prediction heads
        self.volume_head = nn.Linear(hidden_dim, volume_dim)
        self.location_head = nn.Linear(hidden_dim, location_dim)
        self.shape_head = nn.Linear(hidden_dim, shape_dim)

    def forward(
        self,
        multi_scale_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict from multi-scale features.

        Args:
            multi_scale_features: Concatenated features [B, sum(stage_dims)].

        Returns:
            Dict with 'pred_volume', 'pred_location', 'pred_shape'.
        """
        fused = self.fusion(multi_scale_features)

        return {
            'pred_volume': self.volume_head(fused),
            'pred_location': self.location_head(fused),
            'pred_shape': self.shape_head(fused),
        }
