"""Domain divergence analysis for LoRA stage targeting.

Provides per-stage metrics that quantify how different the frozen encoder's
representations are for glioma (GLI) vs meningioma (MEN).  These metrics
predict which encoder stages need LoRA adaptation and what type (attention
vs feature recalibration).

Key metrics:
    - **Domain classifier accuracy**: Can a linear model distinguish GLI
      from MEN at stage *s*?  50% = domain-invariant, 100% = fully
      domain-specific.
    - **MMD with permutation test**: Distribution-level divergence with
      significance testing (RBF kernel, median heuristic bandwidth).
    - **Proxy-A Distance (PAD)**: PAD = 2·(1 − 2·ε) where ε is the domain
      classifier error.  Range [0, 2].
    - **Feature Statistics Divergence (FSD)**: Per-channel squared Cohen's d
      averaged across channels.  Catches magnitude shifts that geometric
      metrics miss.
    - **CKA (cross-stage / adaptation drift)**: Centered Kernel Alignment is
      valid ONLY for same-sample comparisons (cross-stage on one domain,
      frozen vs adapted on the same scans).  NOT for cross-domain.

References:
    Kornblith et al. (2019). "Similarity of Neural Network Representations
    Revisited." ICML.
    Ben-David et al. (2010). "A theory of learning from different domains."
    Machine Learning 79:151–175.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from growth.evaluation.latent_quality import (
    compute_cka,
    compute_domain_classifier_accuracy,
    compute_proxy_a_distance,
    mmd_permutation_test,
)

from .data_loader import LoadedDataset
from .tsi import extract_hidden_states

logger = logging.getLogger(__name__)

# Stage index → channel count for BrainSegFounder-Tiny (feature_size=48).
STAGE_CHANNELS: dict[int, int] = {0: 48, 1: 96, 2: 192, 3: 384, 4: 768}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_gap_features_per_stage(
    model: nn.Module,
    loaded: LoadedDataset,
    indices: list[int],
    device: str,
    stages: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[int, np.ndarray]:
    """Extract Global-Average-Pooled features for every encoder stage.

    Performs one forward pass per scan through the frozen encoder, collects
    the 5 SwinViT hidden states, and applies adaptive_avg_pool3d to
    produce a single ``[C_s]`` vector per stage per scan.

    Args:
        model: Frozen SwinUNETR or LoRAOriginalDecoderModel in eval mode.
        loaded: LoadedDataset with the domain's scan list.
        indices: Dataset indices to process (into ``loaded.dataset``).
        device: CUDA device string.
        stages: Which stages to collect (default: all 5).

    Returns:
        Dict mapping stage index to float32 ndarray ``[N, C_s]``.
    """
    accum: dict[int, list[np.ndarray]] = {s: [] for s in stages}
    n_total = len(indices)

    for i, ds_idx in enumerate(indices):
        sample = loaded.dataset[ds_idx]
        images = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device != "cpu")):
            hidden_states = extract_hidden_states(model, images)

        for s in stages:
            h = hidden_states[s]  # [1, C_s, D, H, W] on CPU
            gap = F.adaptive_avg_pool3d(h, 1).squeeze().float().numpy()  # [C_s]
            accum[s].append(gap)

        if device != "cpu":
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            logger.info("  GAP features: %d / %d scans", i + 1, n_total)

    return {s: np.stack(vecs, axis=0) for s, vecs in accum.items()}


# ---------------------------------------------------------------------------
# Domain divergence metrics
# ---------------------------------------------------------------------------


@dataclass
class StageDomainMetrics:
    """Domain divergence metrics for one encoder stage.

    Attributes:
        stage: Stage index 0-4.
        n_channels: C_s for this stage.
        n_gli: Number of GLI scans.
        n_men: Number of MEN scans.
        domain_acc_linear: 5-fold CV logistic regression accuracy.
        domain_acc_mlp: 5-fold CV shallow MLP accuracy.
        mmd: Unbiased MMD^2 (RBF kernel, median heuristic).
        mmd_p: Permutation-test p-value for MMD.
        pad: Proxy A-distance in [0, 2].
        fsd: Feature-Space Divergence.
        domain_acc_ci_lower: Bootstrap lower CI for linear probe accuracy.
        domain_acc_ci_upper: Bootstrap upper CI for linear probe accuracy.
    """

    stage: int
    n_channels: int
    n_gli: int
    n_men: int
    domain_acc_linear: float
    domain_acc_mlp: float
    mmd: float
    mmd_p: float
    pad: float
    fsd: float
    domain_acc_ci_lower: float
    domain_acc_ci_upper: float


def compute_fsd(
    features_a: np.ndarray,
    features_b: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Feature Statistics Divergence: per-channel squared Cohen's d averaged.

    .. math::

        \\text{FSD}_s = \\frac{1}{C_s} \\sum_{c=1}^{C_s}
            \\frac{(\\mu^{A}_{c} - \\mu^{B}_{c})^2}
                 {\\frac{1}{2}((\\sigma^{A}_{c})^2 + (\\sigma^{B}_{c})^2)
                  + \\varepsilon}

    FSD = 0 means identical channel means; FSD >> 1 means the mean shift
    exceeds typical within-domain variation.  Comparable across stages
    because it is a *mean* (not sum) over channels.

    Args:
        features_a: ``[N_a, C_s]`` float array for domain A.
        features_b: ``[N_b, C_s]`` float array for domain B.
        epsilon: Numerical stability added to denominator.

    Returns:
        FSD scalar in ``[0, inf)``.

    Raises:
        ValueError: If feature dimensionalities differ.
    """
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError(
            f"Channel dimensions must match: {features_a.shape[1]} "
            f"vs {features_b.shape[1]}"
        )
    mu_a = features_a.mean(axis=0)
    mu_b = features_b.mean(axis=0)
    var_a = features_a.var(axis=0)
    var_b = features_b.var(axis=0)
    return float(
        np.mean((mu_a - mu_b) ** 2 / (0.5 * (var_a + var_b) + epsilon))
    )


def _bootstrap_domain_acc(
    features_a: np.ndarray,
    features_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for domain classifier accuracy.

    Args:
        features_a: ``[N_a, C]`` GLI features.
        features_b: ``[N_b, C]`` MEN features.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``(ci_lower, ci_upper)`` for the 95% percentile interval.
    """
    rng = np.random.RandomState(seed)
    n_a, n_b = len(features_a), len(features_b)
    accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)
        accs[i] = compute_domain_classifier_accuracy(
            features_a[idx_a], features_b[idx_b], n_splits=3,
        )
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def _mlp_domain_accuracy(
    features_a: np.ndarray,
    features_b: np.ndarray,
    n_splits: int = 5,
) -> float:
    """Shallow MLP domain classifier accuracy (5-fold CV).

    Uses ``MLPClassifier(hidden_layer_sizes=(256,), max_iter=500)`` from
    scikit-learn.  Reports mean CV accuracy.

    Args:
        features_a: ``[N_a, C]`` domain A features.
        features_b: ``[N_b, C]`` domain B features.
        n_splits: Number of CV folds.

    Returns:
        Mean CV accuracy in ``[0, 1]``.
    """
    X = np.vstack([features_a, features_b])
    y = np.concatenate([np.zeros(len(features_a)), np.ones(len(features_b))])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clf = MLPClassifier(
        hidden_layer_sizes=(256,), max_iter=500, random_state=42,
    )
    scores = cross_val_score(clf, X, y, cv=n_splits)
    return float(scores.mean())


def compute_per_stage_domain_metrics(
    gli_features: dict[int, np.ndarray],
    men_features: dict[int, np.ndarray],
    stages: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_mmd_perm: int = 1000,
    n_bootstrap: int = 1000,
) -> dict[int, StageDomainMetrics]:
    """Compute the full battery of domain divergence metrics per stage.

    For each stage, features are standardised (``StandardScaler`` fit on
    combined GLI + MEN) before computing domain classifier accuracy, MMD,
    and PAD.  FSD operates on raw features (inherently normalised).

    Args:
        gli_features: ``{stage: [N_GLI, C_s]}`` for glioma.
        men_features: ``{stage: [N_MEN, C_s]}`` for meningioma.
        stages: Which stages to compute.
        n_mmd_perm: Permutation count for MMD significance test.
        n_bootstrap: Bootstrap count for domain accuracy CI.

    Returns:
        Dict mapping stage index to ``StageDomainMetrics``.
    """
    results: dict[int, StageDomainMetrics] = {}

    for s in stages:
        feat_g = gli_features[s]
        feat_m = men_features[s]
        logger.info(
            "Stage %d: GLI %s, MEN %s", s, feat_g.shape, feat_m.shape,
        )

        # Standardise for distance-based metrics
        scaler = StandardScaler()
        combined = np.vstack([feat_g, feat_m])
        combined_scaled = scaler.fit_transform(combined)
        g_scaled = combined_scaled[: len(feat_g)]
        m_scaled = combined_scaled[len(feat_g) :]

        # Domain classifier (linear)
        acc_lin = compute_domain_classifier_accuracy(g_scaled, m_scaled)
        # Domain classifier (MLP)
        acc_mlp = _mlp_domain_accuracy(feat_g, feat_m)

        # MMD
        mmd_sq, mmd_p = mmd_permutation_test(
            g_scaled, m_scaled, n_perm=n_mmd_perm,
        )

        # PAD
        pad = compute_proxy_a_distance(g_scaled, m_scaled)

        # FSD (on raw features — inherently normalised by within-domain var)
        fsd = compute_fsd(feat_g, feat_m)

        # Bootstrap CI for linear accuracy
        ci_lo, ci_hi = _bootstrap_domain_acc(
            g_scaled, m_scaled, n_bootstrap=n_bootstrap,
        )

        results[s] = StageDomainMetrics(
            stage=s,
            n_channels=feat_g.shape[1],
            n_gli=len(feat_g),
            n_men=len(feat_m),
            domain_acc_linear=acc_lin,
            domain_acc_mlp=acc_mlp,
            mmd=mmd_sq,
            mmd_p=mmd_p,
            pad=pad,
            fsd=fsd,
            domain_acc_ci_lower=ci_lo,
            domain_acc_ci_upper=ci_hi,
        )
        logger.info(
            "  Stage %d: DomainAcc=%.3f [%.3f, %.3f], "
            "MMD=%.4f (p=%.3f), PAD=%.3f, FSD=%.4f",
            s, acc_lin, ci_lo, ci_hi, mmd_sq, mmd_p, pad, fsd,
        )

    return results


# ---------------------------------------------------------------------------
# CKA analyses (same-sample only)
# ---------------------------------------------------------------------------


def compute_cka_cross_stage(
    features: dict[int, np.ndarray],
) -> np.ndarray:
    """Pairwise CKA between all stage representations for one domain.

    Valid because all entries correspond to the SAME set of scans at
    different encoder depths.

    Args:
        features: ``{stage: [N, C_s]}``.  All values must share N.

    Returns:
        ``[S, S]`` float32 matrix where ``[i, j] = CKA(stage_i, stage_j)``.

    Raises:
        ValueError: If any stage array has a different N.
    """
    stages = sorted(features.keys())
    n_stages = len(stages)
    n_samples = features[stages[0]].shape[0]
    for s in stages[1:]:
        if features[s].shape[0] != n_samples:
            raise ValueError(
                f"All stages must have the same N. Stage {stages[0]} has "
                f"{n_samples}, stage {s} has {features[s].shape[0]}."
            )

    matrix = np.eye(n_stages, dtype=np.float32)
    for i in range(n_stages):
        for j in range(i + 1, n_stages):
            cka_val = compute_cka(features[stages[i]], features[stages[j]])
            matrix[i, j] = cka_val
            matrix[j, i] = cka_val
    return matrix


def compute_cka_adaptation_drift(
    frozen_features: dict[int, np.ndarray],
    adapted_features: dict[int, np.ndarray],
) -> dict[int, float]:
    """Per-stage CKA between frozen and LoRA-adapted representations.

    Both feature sets must come from the SAME scans forwarded through the
    respective models.  Measures how much each stage's representation has
    drifted after LoRA adaptation.

    CKA ≈ 1.0 → minimal drift.  CKA ≈ 0.0 → maximal drift.

    Args:
        frozen_features: ``{stage: [N, C_s]}`` for frozen model.
        adapted_features: ``{stage: [N, C_s]}`` for adapted model.

    Returns:
        Dict mapping stage index to CKA value in ``[0, 1]``.
    """
    drift: dict[int, float] = {}
    for s in sorted(frozen_features.keys()):
        if s not in adapted_features:
            continue
        drift[s] = compute_cka(frozen_features[s], adapted_features[s])
    return drift


# ---------------------------------------------------------------------------
# Decoder patching sensitivity
# ---------------------------------------------------------------------------


def compute_decoder_patching_sensitivity(
    decoder: nn.Module,
    men_hidden_states: list[torch.Tensor],
    gli_hidden_states: list[torch.Tensor],
    men_input: torch.Tensor,
    device: str,
    stages: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[int, float]:
    """Measure decoder output change when patching one stage's hidden state.

    For each stage *s*:
      1. Start from the MEN scan's full hidden_states.
      2. Replace ``hidden_states[s]`` with the GLI scan's version.
      3. Feed the patched hidden states through the decoder.
      4. Compute the normalised L2 distance between patched and baseline
         logits.

    The score quantifies how sensitive the decoder output is to domain-
    specific information at stage *s*.  Higher score → stronger coupling.

    Args:
        decoder: ``OriginalDecoderWrapper`` in eval mode.
        men_hidden_states: 5 CPU tensors from a MEN scan.
        gli_hidden_states: 5 CPU tensors from a GLI scan.
        men_input: ``[1, 4, D, H, W]`` MEN input tensor on device.
        device: Computation device.
        stages: Which stages to patch.

    Returns:
        Dict mapping stage index to normalised L2 sensitivity.
    """
    # Baseline: unpatched MEN forward
    men_hs_device = [h.to(device) for h in men_hidden_states]
    with torch.no_grad():
        baseline_logits = decoder(men_input, men_hs_device)

    sensitivity: dict[int, float] = {}

    for s in stages:
        patched_hs = list(men_hs_device)  # shallow copy
        patched_hs[s] = gli_hidden_states[s].to(device)

        with torch.no_grad():
            patched_logits = decoder(men_input, patched_hs)

        diff = (patched_logits - baseline_logits).float()
        l2 = diff.norm().item() / diff.numel()
        sensitivity[s] = l2

    return sensitivity


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def apply_bh_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction on an array of p-values.

    Args:
        p_values: 1-D array of raw p-values.
        alpha: FDR threshold.

    Returns:
        ``(adjusted_p, is_significant)`` — both 1-D arrays, same length
        as input.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH adjusted p-values (step-up)
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            sorted_p[i] * n / (i + 1),
            adjusted[sorted_idx[i + 1]],
        )
    adjusted = np.clip(adjusted, 0.0, 1.0)
    significant = adjusted < alpha
    return adjusted, significant


def compute_drift_divergence_correlation(
    domain_acc_per_stage: dict[int, float],
    cka_drift_per_stage: dict[int, float],
) -> tuple[float, float]:
    """Spearman rank correlation between domain divergence and CKA drift.

    A positive ρ means stages with higher domain gap also show more
    adaptation drift (lower CKA).  Uses ``1 − CKA`` as the drift metric
    so that higher values mean more drift.

    Args:
        domain_acc_per_stage: ``{stage: domain_classifier_accuracy}``.
        cka_drift_per_stage: ``{stage: CKA(frozen, adapted)}``.

    Returns:
        ``(rho, p_value)`` from ``scipy.stats.spearmanr``.
    """
    shared_stages = sorted(
        set(domain_acc_per_stage) & set(cka_drift_per_stage)
    )
    if len(shared_stages) < 3:
        logger.warning(
            "Spearman correlation needs ≥ 3 stages; got %d",
            len(shared_stages),
        )
        return 0.0, 1.0

    divergence = [domain_acc_per_stage[s] for s in shared_stages]
    drift = [1.0 - cka_drift_per_stage[s] for s in shared_stages]
    rho, p = stats.spearmanr(divergence, drift)
    return float(rho), float(p)
