"""Loss functions module."""

from .elbo import compute_elbo, get_beta_schedule, get_capacity_schedule
from .dipvae import compute_dipvae_loss, get_lambda_cov_schedule
from .tc import (
    compute_tc_loss,
    compute_tc_loss_on_subset,
    compute_tc_ddp_aware,
    compute_decomposed_kl,
)
from .cross_partition import (
    compute_cross_partition_loss,
    compute_cross_partition_covariance_loss,
    all_gather_with_grad,
)

__all__ = [
    "compute_elbo",
    "get_beta_schedule",
    "get_capacity_schedule",
    "compute_dipvae_loss",
    "get_lambda_cov_schedule",
    "compute_tc_loss",
    "compute_tc_loss_on_subset",
    "compute_tc_ddp_aware",
    "compute_decomposed_kl",
    "compute_cross_partition_loss",
    "compute_cross_partition_covariance_loss",
    "all_gather_with_grad",
]
