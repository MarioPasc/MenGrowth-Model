#!/usr/bin/env python
# experiments/sdp/visualize_latent.py
"""UMAP visualization of SDP latent space.

Loads a trained SDP checkpoint and precomputed features, projects through
the SDP, and generates a UMAP scatter plot colored by volume.

Methodology Revision R1: vol + residual partition only (location/shape removed).

Outputs:
    {output_dir}/latent_umap.png — UMAP colored by whole-tumor volume

Usage:
    python -m experiments.sdp.visualize_latent \
        --checkpoint outputs/sdp/phase2_sdp.pt \
        --features outputs/sdp/features/lora_val.h5 \
        --output-dir outputs/sdp/
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import umap

from growth.models.projection.partition import LatentPartition
from growth.models.projection.sdp import SDP, SDPWithHeads
from growth.models.projection.semantic_heads import SemanticHeads
from growth.training.train_sdp import load_precomputed_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sdp_from_checkpoint(
    checkpoint_path: str,
) -> tuple[SDPWithHeads, torch.Tensor, torch.Tensor]:
    """Load trained SDP model from checkpoint.

    Args:
        checkpoint_path: Path to phase2_sdp.pt checkpoint.

    Returns:
        Tuple of (model, h_mean, h_std).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    sdp_cfg = ckpt["sdp_config"]
    part_cfg = ckpt["partition_config"]
    tgt_cfg = ckpt["target_config"]

    partition = LatentPartition.from_config(
        vol_dim=part_cfg["vol_dim"],
        residual_dim=part_cfg["residual_dim"],
        n_vol=tgt_cfg["n_vol"],
    )
    sdp = SDP(
        in_dim=sdp_cfg["in_dim"],
        hidden_dim=sdp_cfg["hidden_dim"],
        out_dim=sdp_cfg["out_dim"],
        dropout=sdp_cfg["dropout"],
    )
    heads = SemanticHeads(
        vol_in=part_cfg["vol_dim"],
        vol_out=tgt_cfg["n_vol"],
    )

    model = SDPWithHeads(sdp=sdp, partition=partition, heads=heads)
    model.load_state_dict(ckpt["model_state_dict"])

    h_mean = ckpt["h_mean"]
    h_std = ckpt["h_std"]

    return model, h_mean, h_std


def main(
    checkpoint_path: str,
    features_path: str,
    output_dir: str,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    seed: int = 42,
) -> None:
    """Generate UMAP visualization of SDP latent space.

    Args:
        checkpoint_path: Path to phase2_sdp.pt.
        features_path: Path to .h5 file with encoder features and targets.
        output_dir: Output directory for latent_umap.png.
        umap_n_neighbors: UMAP n_neighbors parameter.
        umap_min_dist: UMAP min_dist parameter.
        seed: Random seed for reproducibility.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, h_mean, h_std = load_sdp_from_checkpoint(checkpoint_path)
    model.eval()

    h, targets = load_precomputed_features(features_path)

    # Normalize and project
    h_norm = (h - h_mean) / h_std.clamp(min=1e-8)
    with torch.no_grad():
        z, partitions, predictions = model(h_norm)

    z_np = z.numpy()

    # UMAP embedding
    logger.info(f"Running UMAP on {z_np.shape[0]} samples, {z_np.shape[1]} dims...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=seed,
    )
    embedding = reducer.fit_transform(z_np)

    # Semantic coloring: whole-tumor volume
    vol_target = targets["vol"][:, 0].numpy()  # log(V_WT + 1)

    # Create single-panel figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=vol_target,
        cmap="viridis",
        s=12,
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_title("Log Whole-Tumor Volume", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.colorbar(sc, ax=ax, shrink=0.8)

    fig.suptitle("SDP Latent Space (UMAP)", fontsize=15, y=1.02)
    fig.tight_layout()

    out_file = output_path / "latent_umap.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved UMAP visualization to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP visualization of SDP latent space")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to phase2_sdp.pt")
    parser.add_argument("--features", type=str, required=True, help="Path to features .h5 file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        features_path=args.features,
        output_dir=args.output_dir,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        seed=args.seed,
    )
