"""Latent space statistics and diagnostics for VAEs.

Provides pure metric computation functions for analyzing latent space properties,
including correlation, covariance, and sensitivity to spatial transformations.

Functions:
    compute_correlation: Correlation statistics of latent means
    compute_dipvae_covariance: DIP-VAE-II covariance metrics
    compute_shift_sensitivity: Sensitivity to spatial shifts
    compute_cov_batch: Helper to compute covariance matrix
"""

import torch
import torch.nn as nn
from typing import Dict
from torch.utils.data import DataLoader


def compute_correlation(
    mu: torch.Tensor,
    return_matrix: bool = False
) -> Dict[str, float]:
    """Compute correlation statistics of latent means.

    Args:
        mu: Latent means [N, z_dim]
        return_matrix: If True, also return full correlation matrix

    Returns:
        Dictionary with keys:
        - corr_offdiag_meanabs: Mean absolute off-diagonal correlation
        - corr_matrix (optional): Full correlation matrix [z_dim, z_dim]
    """
    # Standardize
    mu_mean = mu.mean(dim=0)
    mu_std_val = mu.std(dim=0, unbiased=True)
    mu_std = (mu - mu_mean) / (mu_std_val + 1e-8)

    # Correlation matrix
    N = mu_std.shape[0]
    corr = torch.mm(mu_std.T, mu_std) / (N - 1)  # [z_dim, z_dim]

    # Off-diagonal elements
    z_dim = corr.shape[0]
    mask = ~torch.eye(z_dim, dtype=torch.bool, device=corr.device)
    off_diag = corr[mask]

    result = {"corr_offdiag_meanabs": off_diag.abs().mean().item()}

    if return_matrix:
        result["corr_matrix"] = corr

    return result


def compute_dipvae_covariance(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    compute_in_fp32: bool = True
) -> Dict[str, float]:
    """Compute DIP-VAE-II covariance metrics.

    DIP-VAE-II covariance estimator (Kumar et al., 2018):
        Cov_q(z) = Cov_batch(μ) + mean_batch(diag(exp(logvar)))

    This captures both between-sample diversity and within-sample uncertainty,
    matching the training loss computation.

    Reference: https://ar5iv.org/pdf/1711.00848 (Eq. 8-10)

    Args:
        mu: Latent means [N, z_dim]
        logvar: Latent log-variances [N, z_dim]
        compute_in_fp32: Force FP32 computation for numerical stability

    Returns:
        Dictionary with keys:
        - cov_q_offdiag_meanabs: Mean absolute off-diagonal covariance
        - cov_q_offdiag_fro: Frobenius norm of off-diagonal
        - cov_q_diag_meanabs_error: |diag(Cov_q) - 1| mean
        - cov_q_diag_mean: Mean diagonal value (should → 1.0)
        - cov_q_matrix: Full covariance matrix [z_dim, z_dim]
    """
    N, d = mu.shape

    # Compute in FP32 for numerical stability
    if compute_in_fp32:
        with torch.cuda.amp.autocast(enabled=False):
            mu = mu.float()
            logvar = logvar.float()

            # Between-sample covariance: Cov(μ)
            mu_centered = mu - mu.mean(dim=0, keepdim=True)
            cov_mu = torch.mm(mu_centered.t(), mu_centered) / N  # [d, d]

            # Within-sample variance: E[diag(exp(logvar))]
            mean_encoder_var = torch.exp(logvar).mean(dim=0)  # [d]
            cov_var = torch.diag(mean_encoder_var)  # [d, d]

            # Total aggregated covariance (DIP-VAE-II)
            cov_q = cov_mu + cov_var  # [d, d]

            # Extract off-diagonal elements
            mask = ~torch.eye(d, dtype=torch.bool, device=cov_q.device)
            off_diag = cov_q[mask]

            # Extract diagonal elements
            diag_elems = torch.diag(cov_q)

            metrics = {
                "cov_q_offdiag_meanabs": off_diag.abs().mean().item(),
                "cov_q_offdiag_fro": torch.norm(off_diag, p="fro").item(),
                "cov_q_diag_meanabs_error": (diag_elems - 1.0).abs().mean().item(),
                "cov_q_diag_mean": diag_elems.mean().item(),
                "cov_q_matrix": cov_q,
            }
    else:
        # Regular precision computation
        mu_centered = mu - mu.mean(dim=0, keepdim=True)
        cov_mu = torch.mm(mu_centered.t(), mu_centered) / N  # [d, d]

        mean_encoder_var = torch.exp(logvar).mean(dim=0)  # [d]
        cov_var = torch.diag(mean_encoder_var)  # [d, d]

        cov_q = cov_mu + cov_var  # [d, d]

        mask = ~torch.eye(d, dtype=torch.bool, device=cov_q.device)
        off_diag = cov_q[mask]
        diag_elems = torch.diag(cov_q)

        metrics = {
            "cov_q_offdiag_meanabs": off_diag.abs().mean().item(),
            "cov_q_offdiag_fro": torch.norm(off_diag, p="fro").item(),
            "cov_q_diag_meanabs_error": (diag_elems - 1.0).abs().mean().item(),
            "cov_q_diag_mean": diag_elems.mean().item(),
            "cov_q_matrix": cov_q,
        }

    return metrics


def compute_shift_sensitivity(
    model: nn.Module,
    dataloader: DataLoader,
    shift_vox: int = 5,
    num_samples: int = 32,
    device: str = "cuda",
    seed: int = 42
) -> Dict[str, float]:
    """Measure latent sensitivity to spatial shifts.

    Computes how much the latent mean μ changes when the input image
    is randomly shifted by up to shift_vox voxels in each dimension.

    Args:
        model: VAE model with encode() method
        dataloader: DataLoader providing image tensors
        shift_vox: Maximum shift magnitude per axis
        num_samples: Number of samples to evaluate (uses first N from dataloader)
        device: Device to run model on
        seed: Random seed for reproducible shifts

    Returns:
        Dictionary with keys:
        - shift_sens_mu_l2: Mean L2 change in μ under translation
        - shift_sens_mu_rel: Relative L2 change (normalized by ||μ||)
    """
    delta_norms = []
    delta_rel_norms = []

    # Set generator for reproducible random shifts
    rng = torch.Generator().manual_seed(seed)

    model.eval()
    model.to(device)

    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            # Handle different batch formats (dict or tensor)
            if isinstance(batch, dict):
                x = batch.get("image", batch.get("vol", None))
            else:
                x = batch

            x = x.to(device)
            B = x.shape[0]

            for i in range(B):
                if sample_count >= num_samples:
                    break

                x_single = x[i]  # [C, D, H, W]

                # Encode original
                x_batch = x_single.unsqueeze(0)
                mu_orig, _ = model.encode(x_batch)
                mu_orig = mu_orig.squeeze(0).cpu().float()

                # Random shift
                shift_d = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()
                shift_h = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()
                shift_w = torch.randint(-shift_vox, shift_vox + 1, (1,), generator=rng).item()

                x_shifted = _shift_image_no_wrap(x_single, shift_d, shift_h, shift_w)

                # Encode shifted
                x_shifted_batch = x_shifted.unsqueeze(0).to(device)
                mu_shifted, _ = model.encode(x_shifted_batch)
                mu_shifted = mu_shifted.squeeze(0).cpu().float()

                # Compute change
                delta = mu_shifted - mu_orig
                delta_norm = delta.norm(p=2).item()
                mu_orig_norm = mu_orig.norm(p=2).item()

                delta_norms.append(delta_norm)
                delta_rel_norms.append(delta_norm / (mu_orig_norm + 1e-8))

                sample_count += 1

            if sample_count >= num_samples:
                break

    return {
        "shift_sens_mu_l2": float(sum(delta_norms) / len(delta_norms)),
        "shift_sens_mu_rel": float(sum(delta_rel_norms) / len(delta_rel_norms)),
    }


def _shift_image_no_wrap(
    x: torch.Tensor,
    shift_d: int,
    shift_h: int,
    shift_w: int
) -> torch.Tensor:
    """Shift image by (shift_d, shift_h, shift_w) voxels without wrap-around.

    Helper function for compute_shift_sensitivity.

    Args:
        x: [C, D, H, W] image tensor
        shift_d: Signed shift in depth dimension
        shift_h: Signed shift in height dimension
        shift_w: Signed shift in width dimension

    Returns:
        x_shifted: [C, D, H, W] with zeros in vacated regions
    """
    C, D, H, W = x.shape
    x_shifted = torch.zeros_like(x)

    # Source slice ranges
    d_src_start = max(0, -shift_d)
    d_src_end = min(D, D - shift_d)
    h_src_start = max(0, -shift_h)
    h_src_end = min(H, H - shift_h)
    w_src_start = max(0, -shift_w)
    w_src_end = min(W, W - shift_w)

    # Destination slice ranges
    d_dst_start = max(0, shift_d)
    d_dst_end = d_dst_start + (d_src_end - d_src_start)
    h_dst_start = max(0, shift_h)
    h_dst_end = h_dst_start + (h_src_end - h_src_start)
    w_dst_start = max(0, shift_w)
    w_dst_end = w_dst_start + (w_src_end - w_src_start)

    x_shifted[
        :, d_dst_start:d_dst_end, h_dst_start:h_dst_end, w_dst_start:w_dst_end
    ] = x[:, d_src_start:d_src_end, h_src_start:h_src_end, w_src_start:w_src_end]

    return x_shifted


def compute_cov_batch(x: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrix of [N, D] tensor.

    Helper function for covariance computations.

    Args:
        x: Input tensor [N, D]

    Returns:
        Covariance matrix [D, D]
    """
    N = x.shape[0]
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = torch.mm(x_centered.t(), x_centered) / N
    return cov
