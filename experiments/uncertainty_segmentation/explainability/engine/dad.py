"""Domain Attention Divergence (DAD) between BraTS-GLI and BraTS-MEN.

For each ``(stage, block, head)`` we summarise the model's attention
on a scan as the *row-averaged* distribution over the ``N`` window
tokens::

    p_scan[s, b, h, :] = mean over all (window, query-row) of attn[s, b, h, q, :]

This collapses the ``[n_windows*B, H, N, N]`` tensor to one
probability vector of length ``N`` per head.  Because all scans in a
cohort are padded to the same ROI size, ``N`` (window volume) is
identical across scans, so per-scan vectors are directly comparable.

DAD between two cohorts is the symmetric KL divergence of the
cohort-averaged vectors.  Significance is assessed by a paired
permutation test: per-scan domain labels are shuffled (preserving the
total cohort size) and DAD is recomputed; the empirical p-value is
the fraction of permutations that match or exceed the observed DAD.

Notes
-----
- We keep one accumulator per scan (small: ``H × N`` floats per
  ``(stage, block)``) and reduce after the entire scan loop ends.
  Storing per-scan vectors enables the permutation test below.
- The per-row softmax sums to 1, so the row-average sums to 1 as
  well; we still re-normalise to be defensive against floating-point
  drift after summing across thousands of windows.
- Stage 1 / 2 windows have ``N = 343`` (full ``7×7×7``), so each
  scan's stage-1 signature is ``num_heads * 343 ≤ 24*343 ≈ 8 KB`` —
  trivial to keep in memory for both cohorts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Numerical floor inside log/log-ratio.
DAD_EPS = 1e-10


# -----------------------------------------------------------------------------
# Per-scan accumulator
# -----------------------------------------------------------------------------


@dataclass
class DADScanResult:
    """Row-averaged attention distributions for one scan.

    Attributes
    ----------
    row_avg : dict[str, np.ndarray]
        Mapping ``"stage_{s}_block_{b}" -> [num_heads, N]``.  Each row
        ``[h, :]`` is a probability vector summing to 1.
    n_rows : dict[str, int]
        Number of rows that were averaged for each key (sanity
        check; equals ``n_windows*B*N``).
    """

    row_avg: dict[str, np.ndarray] = field(default_factory=dict)
    n_rows: dict[str, int] = field(default_factory=dict)


class DADScanAccumulator:
    """Streaming accumulator that plugs into :class:`AttentionCapture`.

    Pass an instance as ``process_fn`` to ``AttentionCapture(...,
    mode="callback", process_fn=acc)``.  After the forward pass call
    :meth:`result` to get a :class:`DADScanResult`.  Re-use across
    scans by calling :meth:`reset`.
    """

    def __init__(self, target_stages: set[int] | None = None) -> None:
        self._target_stages = target_stages or {1, 2, 3, 4}
        self._sum: dict[str, torch.Tensor] = {}
        self._count: dict[str, int] = {}

    def reset(self) -> None:
        self._sum.clear()
        self._count.clear()

    # ------------------------------------------------------------------
    def __call__(
        self,
        key: str,
        attn: torch.Tensor,
        mask: torch.Tensor | None,  # noqa: ARG002 - unused
    ) -> None:
        parts = key.split("_")
        stage = int(parts[1])
        if stage not in self._target_stages:
            return
        # attn: [n_windows*B, H, N, N]
        if attn.dim() != 4:
            raise RuntimeError(f"{key}: expected 4-D attn, got {tuple(attn.shape)}")
        # Sum across (window, query) → [H, N]; cast to float32 for
        # numerical stability (the model may run in fp16 / bf16).
        summed = attn.to(torch.float32).sum(dim=(0, 2)).detach().cpu()
        n_rows_added = attn.shape[0] * attn.shape[2]
        if key not in self._sum:
            self._sum[key] = summed
            self._count[key] = n_rows_added
        else:
            self._sum[key] = self._sum[key] + summed
            self._count[key] += n_rows_added

    # ------------------------------------------------------------------
    def result(self) -> DADScanResult:
        out: dict[str, np.ndarray] = {}
        n_rows: dict[str, int] = {}
        for key, s in self._sum.items():
            avg = s.numpy() / float(self._count[key])
            # Re-normalise per head to compensate for float drift.
            row_sum = avg.sum(axis=-1, keepdims=True)
            row_sum = np.where(row_sum > 0, row_sum, 1.0)
            out[key] = avg / row_sum
            n_rows[key] = self._count[key]
        return DADScanResult(row_avg=out, n_rows=n_rows)


# -----------------------------------------------------------------------------
# Symmetric KL divergence
# -----------------------------------------------------------------------------


def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = DAD_EPS) -> float:
    """Symmetric KL divergence ``0.5 * (KL(p||q) + KL(q||p))``.

    Stable formulation: clamps both distributions at ``eps`` before
    taking logs, then renormalises.  Handles zero entries that arise
    from masked attention positions (the shifted-block ``compute_mask``
    sets some attention slots to ``-inf`` which become exact zeros
    after softmax).

    Parameters
    ----------
    p, q : np.ndarray
        Probability vectors of equal length.  Must be non-negative.
    eps : float
        Lower clamp on both distributions.

    Returns
    -------
    float
        Symmetric KL divergence in nats.
    """
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: p={p.shape}, q={q.shape}")
    p = np.clip(p.astype(np.float64), eps, None)
    q = np.clip(q.astype(np.float64), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = float(np.sum(p * (np.log(p) - np.log(q))))
    kl_qp = float(np.sum(q * (np.log(q) - np.log(p))))
    return 0.5 * (kl_pq + kl_qp)


# -----------------------------------------------------------------------------
# Cohort-level DAD with permutation test
# -----------------------------------------------------------------------------


@dataclass
class DADStats:
    """Per-head DAD with permutation-test p-value.

    Attributes
    ----------
    key : str
        ``"stage_{s}_block_{b}"``.
    head : int
    dad_observed : float
        Symmetric KL between cohort means.
    p_value : float
        Empirical p-value: fraction of permutations whose DAD ≥
        ``dad_observed``.  Lower-bounded by ``1 / (n_perm + 1)``.
    null_mean : float
    null_std : float
    """

    key: str
    head: int
    dad_observed: float
    p_value: float
    null_mean: float
    null_std: float


def _cohort_mean(vectors: np.ndarray) -> np.ndarray:
    """Mean of ``[n_scans, ...]`` along axis 0."""
    return vectors.mean(axis=0)


def compute_dad_with_permutation(
    cohort_a: list[DADScanResult],
    cohort_b: list[DADScanResult],
    n_perm: int = 1000,
    seed: int = 0,
) -> dict[str, list[DADStats]]:
    """Per (stage, block, head) DAD with permutation null distribution.

    Parameters
    ----------
    cohort_a, cohort_b : list[DADScanResult]
        Per-scan row-averaged distributions for each domain.
    n_perm : int
        Number of permutations of scan-level domain labels.  Each
        permutation re-splits the ``len(a) + len(b)`` scans into two
        cohorts of the same sizes as the originals.
    seed : int

    Returns
    -------
    dict[str, list[DADStats]]
        Mapping ``"stage_{s}_block_{b}" -> [DADStats per head]``.
    """
    if not cohort_a or not cohort_b:
        raise ValueError("Both cohorts must contain at least one scan")

    rng = np.random.RandomState(seed)
    keys = sorted(set(cohort_a[0].row_avg.keys()) & set(cohort_b[0].row_avg.keys()))
    if not keys:
        raise ValueError("No common (stage, block) keys between the two cohorts")

    n_a, n_b = len(cohort_a), len(cohort_b)
    out: dict[str, list[DADStats]] = {}

    for key in keys:
        # Stack per-scan vectors → [n_scans, H, N].
        a = np.stack([r.row_avg[key] for r in cohort_a], axis=0)
        b = np.stack([r.row_avg[key] for r in cohort_b], axis=0)
        if a.shape[1:] != b.shape[1:]:
            raise RuntimeError(
                f"{key}: shape mismatch {a.shape[1:]} vs {b.shape[1:]}"
            )
        h = a.shape[1]
        all_vecs = np.concatenate([a, b], axis=0)  # [n_a+n_b, H, N]

        observed = np.array([
            symmetric_kl(_cohort_mean(a)[head_i], _cohort_mean(b)[head_i])
            for head_i in range(h)
        ])

        null = np.empty((n_perm, h), dtype=np.float64)
        idx = np.arange(n_a + n_b)
        for p_i in range(n_perm):
            rng.shuffle(idx)
            perm_a = all_vecs[idx[:n_a]]
            perm_b = all_vecs[idx[n_a:]]
            for head_i in range(h):
                null[p_i, head_i] = symmetric_kl(
                    _cohort_mean(perm_a)[head_i],
                    _cohort_mean(perm_b)[head_i],
                )

        # One-sided p-value: ``Pr(null >= observed)``.
        p_values = (np.sum(null >= observed[None, :], axis=0) + 1) / (n_perm + 1)
        out[key] = [
            DADStats(
                key=key,
                head=head_i,
                dad_observed=float(observed[head_i]),
                p_value=float(p_values[head_i]),
                null_mean=float(null[:, head_i].mean()),
                null_std=float(null[:, head_i].std()),
            )
            for head_i in range(h)
        ]
    return out


def stack_per_scan_vectors(
    results: Sequence[DADScanResult], key: str
) -> np.ndarray:
    """Convenience: stack ``[n_scans, H, N]`` for one (stage, block) key."""
    return np.stack([r.row_avg[key] for r in results], axis=0)
