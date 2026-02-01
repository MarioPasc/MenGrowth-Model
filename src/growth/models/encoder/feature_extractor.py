# src/growth/models/encoder/feature_extractor.py
"""Feature extraction from SwinUNETR encoder.

Applies global average pooling to encoder features to produce
fixed-size feature vectors for downstream tasks.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# Valid feature extraction levels
FeatureLevel = Literal["encoder10", "layers4", "multi_scale"]

# Output dimensions for each level (with feature_size=48)
# Note: MONAI 1.5+ SwinUNETR hidden states have doubled channel counts at each stage
LEVEL_DIMS: Dict[str, int] = {
    "encoder10": 768,      # After encoder10 (same as layers4 output)
    "layers4": 768,        # Raw swinViT.layers4 output (hidden_states[4])
    "multi_scale": 1344,   # layers2(192) + layers3(384) + layers4(768)
}


class FeatureExtractor(nn.Module):
    """Extract features from SwinUNETR encoder.

    Wraps a SwinUNETR model and provides methods to extract
    feature vectors at different levels of the hierarchy.

    Features are extracted via global average pooling (GAP)
    over spatial dimensions.

    Args:
        encoder: SwinUNETR model (with or without decoder).
        level: Feature extraction level:
            - "encoder10": 768-dim from encoder10 bottleneck (default)
            - "layers4": 384-dim from swinViT.layers4 output
            - "multi_scale": 672-dim concatenation of layers2+3+4
        normalize: Whether to apply layer normalization after GAP.

    Example:
        >>> from growth.models.encoder.swin_loader import load_swin_encoder
        >>> encoder = load_swin_encoder("checkpoint.pt")
        >>> extractor = FeatureExtractor(encoder, level="layers4")
        >>> x = torch.randn(2, 4, 96, 96, 96)
        >>> features = extractor(x)  # [2, 384]
    """

    def __init__(
        self,
        encoder: nn.Module,
        level: FeatureLevel = "encoder10",
        normalize: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.level = level
        self._normalize = normalize

        # Validate level
        if level not in LEVEL_DIMS:
            raise ValueError(
                f"Unknown feature level: {level}. "
                f"Choose from: {list(LEVEL_DIMS.keys())}"
            )

        self._output_dim = LEVEL_DIMS[level]

        # Optional layer normalization
        if normalize:
            self.layer_norm = nn.LayerNorm(self._output_dim)
        else:
            self.layer_norm = None

        logger.info(
            f"FeatureExtractor initialized: level={level}, "
            f"output_dim={self._output_dim}, normalize={normalize}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input.

        Args:
            x: Input tensor of shape [B, C, D, H, W].
                Expected: [B, 4, 96, 96, 96] for BraTS data.

        Returns:
            Feature tensor of shape [B, output_dim].
        """
        if self.level == "encoder10":
            features = self._extract_encoder10(x)
        elif self.level == "layers4":
            features = self._extract_layers4(x)
        elif self.level == "multi_scale":
            features = self._extract_multi_scale(x)
        else:
            raise ValueError(f"Unknown feature level: {self.level}")

        # Apply optional normalization
        if self.layer_norm is not None:
            features = self.layer_norm(features)

        return features

    def _get_swin_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get hidden states from all swinViT stages.

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            List of hidden states from each stage:
            - hidden_states[0]: After patch_embed [B, 48, 48, 48, 48]
            - hidden_states[1]: After layers1 [B, 48, 24, 24, 24]
            - hidden_states[2]: After layers2 [B, 96, 12, 12, 12]
            - hidden_states[3]: After layers3 [B, 192, 6, 6, 6]
            - hidden_states[4]: After layers4 [B, 384, 3, 3, 3]
        """
        return self.encoder.swinViT(x, self.encoder.normalize)

    def _extract_encoder10(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from encoder10 (768-dim).

        Uses encoder10 to process the deepest swinViT features,
        then applies GAP to get a 768-dim vector.

        For 96^3 input: layers4 output is [B, 384, 3, 3, 3].
        After encoder10: [B, 768, 3, 3, 3].
        After GAP: [B, 768].
        """
        # Get swinViT hidden states
        hidden_states = self._get_swin_hidden_states(x)

        # Get deepest features (stage 4)
        x4 = hidden_states[4]  # [B, 384, 3, 3, 3]

        # Process through encoder10 (bottleneck processor)
        # encoder10 doubles channels: 384 -> 768
        enc10 = self.encoder.encoder10(x4)  # [B, 768, 3, 3, 3]

        # Global Average Pooling
        features = F.adaptive_avg_pool3d(enc10, 1).flatten(1)  # [B, 768]

        return features

    def _extract_layers4(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from layers4 (768-dim).

        Uses swinViT.layers4 output directly with GAP.
        Same as encoder10 output since encoder10 preserves dimensions.
        """
        # Get swinViT hidden states
        hidden_states = self._get_swin_hidden_states(x)

        # Stage 4 output: [B, 768, 3, 3, 3]
        x4 = hidden_states[4]

        # Global Average Pooling
        features = F.adaptive_avg_pool3d(x4, 1).flatten(1)  # [B, 768]

        return features

    def _extract_multi_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale features (1344-dim).

        Concatenates GAP features from layers2, layers3, and layers4
        for a richer multi-scale representation.

        Total: 192 + 384 + 768 = 1344 dimensions.
        """
        # Get swinViT hidden states
        hidden_states = self._get_swin_hidden_states(x)

        # Extract and pool features from stages 2, 3, 4
        features = []
        for i in [2, 3, 4]:  # layers2, layers3, layers4
            xi = hidden_states[i]
            fi = F.adaptive_avg_pool3d(xi, 1).flatten(1)
            features.append(fi)

        # Concatenate: [B, 192] + [B, 384] + [B, 768] = [B, 1344]
        features = torch.cat(features, dim=1)

        return features

    @property
    def output_dim(self) -> int:
        """Return output feature dimension."""
        return self._output_dim

    @property
    def feature_dim(self) -> int:
        """Alias for output_dim (backwards compatibility)."""
        return self._output_dim


def create_feature_extractor(
    encoder: nn.Module,
    level: str = "encoder10",
    normalize: bool = False,
) -> FeatureExtractor:
    """Factory function to create FeatureExtractor.

    Args:
        encoder: SwinUNETR encoder model.
        level: Feature level ("encoder10", "layers4", "multi_scale").
        normalize: Whether to apply LayerNorm after GAP.

    Returns:
        Configured FeatureExtractor instance.

    Example:
        >>> encoder = load_swin_encoder("checkpoint.pt")
        >>> extractor = create_feature_extractor(encoder, level="layers4")
    """
    return FeatureExtractor(encoder=encoder, level=level, normalize=normalize)


def get_feature_dim(level: str) -> int:
    """Get output dimension for a feature level.

    Args:
        level: Feature extraction level.

    Returns:
        Output dimension.

    Raises:
        ValueError: If level is unknown.
    """
    if level not in LEVEL_DIMS:
        raise ValueError(
            f"Unknown feature level: {level}. "
            f"Choose from: {list(LEVEL_DIMS.keys())}"
        )
    return LEVEL_DIMS[level]
