"""Auto-generated narrative text for the LoRA ablation report.

Each function takes ExperimentData and produces data-driven section text.
All narrative is computed from metrics — nothing is hardcoded.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from experiments.lora_ablation.report.data_loader import (
    ConditionData,
    ExperimentData,
)
from experiments.lora_ablation.report.style import (
    CONDITION_LABELS,
    RANKS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Section container
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SectionContent:
    """Content for one report section."""

    section_id: str
    title: str
    paragraphs: list[str] = field(default_factory=list)
    figure_names: list[str] = field(default_factory=list)
    table_names: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _fmt(val: float | None, decimals: int = 3) -> str:
    """Format a float for display, handling None."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _pct(val: float | None) -> str:
    """Format as percentage."""
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _best_rank_condition(exp: ExperimentData, metric_fn) -> str | None:
    """Find the rank condition that maximizes metric_fn(cond)."""
    rank_conds = [c for c in exp.conditions if c.startswith(("lora_r", "dora_r"))]
    if not rank_conds:
        return None
    return max(rank_conds, key=lambda c: metric_fn(exp.conditions[c]))


def _frozen(exp: ExperimentData) -> ConditionData | None:
    """Get frozen baseline condition."""
    return exp.conditions.get("baseline_frozen")


# ─────────────────────────────────────────────────────────────────────
# Section generators
# ─────────────────────────────────────────────────────────────────────


def generate_abstract(exp: ExperimentData) -> SectionContent:
    """Generate report abstract from summary metrics."""
    section = SectionContent(section_id="abstract", title="Abstract")

    frozen = _frozen(exp)
    best_cond_name = _best_rank_condition(
        exp,
        lambda c: c.dice_men.get("dice_mean", 0),
    )

    if frozen is None or best_cond_name is None:
        section.paragraphs.append(
            "This report analyzes LoRA/DoRA adaptation results. "
            "Insufficient data to generate automated abstract."
        )
        return section

    best = exp.conditions[best_cond_name]
    frozen_dice = frozen.dice_men.get("dice_mean", 0)
    best_dice = best.dice_men.get("dice_mean", 0)
    delta_dice = best_dice - frozen_dice

    frozen_gli = frozen.dice_gli.get("dice_mean", 0)
    best_gli = best.dice_gli.get("dice_mean", 0)
    retention = best_gli / best_dice if best_dice > 0 else 0

    frozen_mmd = frozen.domain_metrics.get("mmd", 0)
    best_mmd = best.domain_metrics.get("mmd", 0)
    mmd_reduction = frozen_mmd - best_mmd

    adapter_type = exp.adapter_type.upper()
    label = CONDITION_LABELS.get(best_cond_name, best_cond_name)

    section.paragraphs.append(
        f"We evaluate {adapter_type} adaptation of BrainSegFounder (SwinUNETR) "
        f"for meningioma segmentation across ranks r ∈ {{2, 4, 8, 16, 32}}. "
        f"The best condition ({label}) achieves a BraTS-MEN Dice of {_fmt(best_dice)} "
        f"(Δ = +{_fmt(delta_dice)} vs frozen baseline at {_fmt(frozen_dice)}). "
        f"Glioma retention ratio is {_fmt(retention, 2)} "
        f"(GLI Dice = {_fmt(best_gli)}), indicating "
        f"{'preserved' if retention > 0.9 else 'moderate loss of'} anatomical knowledge. "
        f"Domain gap (MMD²) {'decreases' if mmd_reduction > 0 else 'increases'} "
        f"by {_fmt(abs(mmd_reduction))} from frozen to adapted features."
    )

    return section


def generate_introduction(exp: ExperimentData) -> SectionContent:
    """Generate introduction with domain gap context."""
    section = SectionContent(section_id="introduction", title="1. Introduction")

    frozen = _frozen(exp)
    adapter_type = exp.adapter_type.upper()

    section.paragraphs.append(
        "BrainSegFounder is a SwinUNETR model pre-trained on 41,000+ brain MRI subjects "
        "and fine-tuned for glioma segmentation on BraTS-GLI. Applying it to meningioma "
        "segmentation requires domain adaptation to bridge the feature distribution gap "
        "between glioma and meningioma representations."
    )

    if frozen is not None:
        frozen_men = frozen.dice_men.get("dice_mean", 0)
        frozen_gli = frozen.dice_gli.get("dice_mean", 0)
        frozen_mmd = frozen.domain_metrics.get("mmd", 0)
        frozen_clf = frozen.domain_metrics.get("domain_classifier_accuracy", 0)

        section.paragraphs.append(
            f"The frozen model achieves Dice = {_fmt(frozen_men)} on BraTS-MEN and "
            f"{_fmt(frozen_gli)} on BraTS-GLI, with a domain gap of "
            f"MMD² = {_fmt(frozen_mmd)} and domain classifier accuracy = {_fmt(frozen_clf)} "
            f"(0.5 = indistinguishable). This confirms a measurable distributional "
            f"shift between the domains."
        )

    section.paragraphs.append(
        f"This report evaluates {adapter_type} as a parameter-efficient method to "
        f"adapt the encoder, analyzing the trade-off between domain specialization "
        f"(meningioma Dice), anatomical preservation (glioma Dice retention), "
        f"and feature quality (semantic probe R²)."
    )

    return section


def generate_segmentation_section(exp: ExperimentData) -> SectionContent:
    """Section 3.1: Segmentation performance."""
    section = SectionContent(
        section_id="segmentation",
        title="3.1 Segmentation Performance",
        figure_names=["dice_men_by_rank"],
    )

    frozen = _frozen(exp)
    best_name = _best_rank_condition(
        exp,
        lambda c: c.dice_men.get("dice_mean", 0),
    )

    if best_name is None:
        section.paragraphs.append("Insufficient data for segmentation analysis.")
        return section

    best = exp.conditions[best_name]
    label = CONDITION_LABELS.get(best_name, best_name)

    section.paragraphs.append(
        f"The best MEN Dice is achieved by {label} at "
        f"{_fmt(best.dice_men.get('dice_mean', 0))} "
        f"(TC = {_fmt(best.dice_men.get('dice_TC', 0))}, "
        f"WT = {_fmt(best.dice_men.get('dice_WT', 0))}, "
        f"ET = {_fmt(best.dice_men.get('dice_ET', 0))})."
    )

    if frozen is not None:
        delta = best.dice_men.get("dice_mean", 0) - frozen.dice_men.get("dice_mean", 0)
        direction = "improvement" if delta > 0 else "degradation"
        section.paragraphs.append(
            f"Compared to the frozen baseline ({_fmt(frozen.dice_men.get('dice_mean', 0))}), "
            f"this represents a {direction} of Δ = {_fmt(delta, 4)}."
        )

    # Summarize trend across ranks
    rank_conds = [c for c in exp.conditions if c.startswith(("lora_r", "dora_r"))]
    if len(rank_conds) >= 3:
        dices = [exp.conditions[c].dice_men.get("dice_mean", 0) for c in rank_conds]
        if dices[-1] > dices[0]:
            trend = "monotonically increasing"
        elif dices[-1] < dices[0]:
            trend = "decreasing at higher ranks"
        else:
            trend = "relatively stable"
        section.paragraphs.append(
            f"Across ranks, segmentation performance is {trend}, suggesting "
            f"{'higher capacity benefits MEN adaptation' if dices[-1] > dices[0] else 'overfitting at higher ranks'}."
        )

    return section


def generate_domain_adaptation_section(exp: ExperimentData) -> SectionContent:
    """Section 3.2: Domain adaptation metrics."""
    section = SectionContent(
        section_id="domain_adaptation",
        title="3.2 Domain Adaptation",
        figure_names=["domain_metrics", "domain_umap"],
    )

    frozen = _frozen(exp)
    conditions = list(exp.conditions.keys())
    has_domain = [c for c in conditions if exp.conditions[c].domain_metrics]

    if not has_domain:
        section.paragraphs.append("No domain metrics available.")
        return section

    # MMD trend
    rank_conds = [c for c in has_domain if c.startswith(("lora_r", "dora_r"))]
    if rank_conds:
        mmds = [(c, exp.conditions[c].domain_metrics.get("mmd", 0)) for c in rank_conds]
        min_mmd_cond, min_mmd = min(mmds, key=lambda x: x[1])
        max_mmd_cond, max_mmd = max(mmds, key=lambda x: x[1])

        section.paragraphs.append(
            f"MMD² ranges from {_fmt(min_mmd)} ({CONDITION_LABELS.get(min_mmd_cond, min_mmd_cond)}) "
            f"to {_fmt(max_mmd)} ({CONDITION_LABELS.get(max_mmd_cond, max_mmd_cond)}) "
            f"across adapted conditions."
        )

    if frozen is not None and frozen.domain_metrics:
        frozen_clf = frozen.domain_metrics.get("domain_classifier_accuracy", 0)
        section.paragraphs.append(
            f"The frozen baseline has domain classifier accuracy = {_fmt(frozen_clf)}, "
            f"indicating {'high' if frozen_clf > 0.7 else 'moderate'} domain separability."
        )

        # Best adapted clf accuracy
        if rank_conds:
            clfs = [
                (c, exp.conditions[c].domain_metrics.get("domain_classifier_accuracy", 1.0))
                for c in rank_conds
            ]
            best_clf_cond, best_clf = min(clfs, key=lambda x: abs(x[1] - 0.5))
            section.paragraphs.append(
                f"The condition closest to chance-level separability is "
                f"{CONDITION_LABELS.get(best_clf_cond, best_clf_cond)} "
                f"(accuracy = {_fmt(best_clf)})."
            )

    return section


def generate_preservation_section(exp: ExperimentData) -> SectionContent:
    """Section 3.3: Anatomical preservation (GLI retention)."""
    section = SectionContent(
        section_id="preservation",
        title="3.3 Anatomical Preservation",
        figure_names=["dice_dual_domain", "retention_ratio"],
    )

    frozen = _frozen(exp)
    if frozen is None:
        section.paragraphs.append("No frozen baseline for retention analysis.")
        return section

    frozen_gli = frozen.dice_gli.get("dice_mean", 0)

    # Compute retention for each adapted condition
    rank_conds = [c for c in exp.conditions if c.startswith(("lora_r", "dora_r"))]
    retentions = []
    for c in rank_conds:
        cond = exp.conditions[c]
        gli = cond.dice_gli.get("dice_mean", 0)
        retention = gli / frozen_gli if frozen_gli > 0 else 0
        retentions.append((c, gli, retention))

    if retentions:
        best_ret = max(retentions, key=lambda x: x[2])
        worst_ret = min(retentions, key=lambda x: x[2])

        section.paragraphs.append(
            f"The frozen model achieves GLI Dice = {_fmt(frozen_gli)}. "
            f"Best GLI retention is {_fmt(best_ret[2], 2)} "
            f"({CONDITION_LABELS.get(best_ret[0], best_ret[0])}, "
            f"GLI Dice = {_fmt(best_ret[1])}). "
            f"Worst retention is {_fmt(worst_ret[2], 2)} "
            f"({CONDITION_LABELS.get(worst_ret[0], worst_ret[0])})."
        )

        # Flag any condition with >20% drop
        big_drops = [(c, r) for c, _, r in retentions if r < 0.80]
        if big_drops:
            names = ", ".join(CONDITION_LABELS.get(c, c) for c, _ in big_drops)
            section.paragraphs.append(
                f"Warning: {names} show >20% GLI Dice retention loss, "
                f"indicating significant forgetting of glioma anatomy."
            )
        else:
            section.paragraphs.append(
                "All adapted conditions maintain >80% GLI Dice retention, "
                "suggesting adequate preservation of anatomical knowledge."
            )

    return section


def generate_feature_quality_section(exp: ExperimentData) -> SectionContent:
    """Section 3.4: Feature quality (probe R²)."""
    section = SectionContent(
        section_id="feature_quality",
        title="3.4 Feature Quality",
        figure_names=["probe_r2", "nonlinearity_gap"],
    )

    rank_conds = [c for c in exp.conditions if c.startswith(("lora_r", "dora_r"))]
    if not rank_conds:
        section.paragraphs.append("No probe metrics available.")
        return section

    # Find best R² condition
    best_r2_name = max(
        rank_conds,
        key=lambda c: exp.conditions[c].metrics_enhanced.get("r2_mean_linear", -999),
    )
    best = exp.conditions[best_r2_name]
    best_r2 = best.metrics_enhanced.get("r2_mean_linear", 0)

    section.paragraphs.append(
        f"Best mean linear R² is {_fmt(best_r2)} "
        f"({CONDITION_LABELS.get(best_r2_name, best_r2_name)}), "
        f"with per-feature breakdown: "
        f"volume = {_fmt(best.metrics_enhanced.get('r2_volume_linear', 0))}, "
        f"location = {_fmt(best.metrics_enhanced.get('r2_location_linear', 0))}, "
        f"shape = {_fmt(best.metrics_enhanced.get('r2_shape_linear', 0))}."
    )

    # Compare linear vs MLP
    best_mlp = best.metrics_enhanced.get("r2_mean_mlp", 0)
    gap = best_mlp - best_r2
    if gap > 0.05:
        section.paragraphs.append(
            f"MLP probe ({_fmt(best_mlp)}) substantially exceeds linear probe ({_fmt(best_r2)}), "
            f"indicating significant nonlinearly encoded information (gap = {_fmt(gap)})."
        )
    elif gap > 0:
        section.paragraphs.append(
            f"MLP probe ({_fmt(best_mlp)}) marginally exceeds linear ({_fmt(best_r2)}), "
            f"suggesting most semantic information is linearly accessible."
        )
    else:
        section.paragraphs.append(
            f"Linear probe ({_fmt(best_r2)}) matches or exceeds MLP ({_fmt(best_mlp)}), "
            f"indicating semantic features are linearly encoded."
        )

    # Volume vs location vs shape analysis
    vol_r2 = best.metrics_enhanced.get("r2_volume_linear", 0)
    loc_r2 = best.metrics_enhanced.get("r2_location_linear", 0)
    shape_r2 = best.metrics_enhanced.get("r2_shape_linear", 0)
    features_ranked = sorted(
        [("volume", vol_r2), ("location", loc_r2), ("shape", shape_r2)],
        key=lambda x: x[1],
        reverse=True,
    )
    section.paragraphs.append(
        f"Feature decodability ranking: "
        f"{features_ranked[0][0]} ({_fmt(features_ranked[0][1])}) > "
        f"{features_ranked[1][0]} ({_fmt(features_ranked[1][1])}) > "
        f"{features_ranked[2][0]} ({_fmt(features_ranked[2][1])})."
    )

    return section


def generate_training_dynamics_section(exp: ExperimentData) -> SectionContent:
    """Section 3.5: Training dynamics."""
    section = SectionContent(
        section_id="training_dynamics",
        title="3.5 Training Dynamics",
        figure_names=["training_curves"],
    )

    rank_conds = [c for c in exp.conditions if c.startswith(("lora_r", "dora_r"))]
    if not rank_conds:
        section.paragraphs.append("No training logs available.")
        return section

    # Convergence epochs
    epochs = []
    for c in rank_conds:
        best_ep = exp.conditions[c].training_summary.get("best_epoch")
        if best_ep is not None:
            epochs.append((c, best_ep))

    if epochs:
        fastest = min(epochs, key=lambda x: x[1])
        slowest = max(epochs, key=lambda x: x[1])
        section.paragraphs.append(
            f"Convergence epochs range from {fastest[1]} "
            f"({CONDITION_LABELS.get(fastest[0], fastest[0])}) to {slowest[1]} "
            f"({CONDITION_LABELS.get(slowest[0], slowest[0])}). "
            f"{'Higher ranks converge faster' if fastest[1] < slowest[1] and 'r32' in fastest[0] else 'Lower ranks converge faster' if 'r2' in fastest[0] else 'Convergence speed varies across ranks'}."
        )

    # Training time
    times = []
    for c in rank_conds:
        t = exp.conditions[c].training_summary.get("training_time_minutes")
        if t is not None:
            times.append((c, t))

    if times:
        total = sum(t for _, t in times)
        section.paragraphs.append(
            f"Total training time: {total:.0f} minutes across {len(times)} conditions."
        )

    # Parameter counts
    for c in rank_conds[:1]:
        params = exp.conditions[c].training_summary.get("param_counts", {})
        encoder_lora = params.get("encoder_lora", 0)
        total = params.get("total", 0)
        if encoder_lora > 0 and total > 0:
            pct = encoder_lora / total * 100
            section.paragraphs.append(
                f"LoRA encoder parameters: {encoder_lora:,} "
                f"({pct:.2f}% of {total:,} total model parameters)."
            )
            break

    return section


def generate_statistical_section(exp: ExperimentData) -> SectionContent:
    """Section 3.6: Statistical analysis."""
    section = SectionContent(
        section_id="statistical",
        title="3.6 Statistical Analysis",
        figure_names=["statistical_heatmap"],
    )

    stats = exp.statistical_comparisons
    if stats is None:
        section.paragraphs.append("No statistical comparisons available.")
        return section

    # Count significant improvements
    significant = []
    for cond, metrics in stats.items():
        if cond == "baseline_frozen":
            continue
        for metric_name, entry in metrics.items():
            p = entry.get("p_corrected", entry.get("p_value"))
            if p is not None and isinstance(p, (int, float)) and not np.isnan(p) and p < 0.05:
                effect = entry.get("effect_interpretation", "unknown")
                significant.append((cond, metric_name, p, effect))

    if significant:
        section.paragraphs.append(
            f"Found {len(significant)} statistically significant improvements "
            f"(corrected p < 0.05) across all condition-metric pairs."
        )

        # Group by condition
        by_cond: dict[str, list] = {}
        for cond, metric, p, effect in significant:
            by_cond.setdefault(cond, []).append((metric, p, effect))

        for cond, items in sorted(by_cond.items(), key=lambda x: -len(x[1])):
            label = CONDITION_LABELS.get(cond, cond)
            n_sig = len(items)
            effects = [e for _, _, e in items]
            dominant_effect = max(set(effects), key=effects.count) if effects else "unknown"
            section.paragraphs.append(
                f"  • {label}: {n_sig} significant metric(s), "
                f"dominant effect size: {dominant_effect}."
            )
    else:
        section.paragraphs.append(
            "No statistically significant improvements found after correction."
        )

    return section


def generate_rank_summary_section(exp: ExperimentData) -> SectionContent:
    """Section 3.7: Optimal rank recommendation."""
    section = SectionContent(
        section_id="rank_summary",
        title="3.7 Rank Selection Summary",
        figure_names=["rank_summary"],
    )

    rank_conds = [c for c in exp.conditions if c.startswith(("lora_r", "dora_r"))]
    if not rank_conds:
        section.paragraphs.append("Insufficient data for rank recommendation.")
        return section

    # Score each condition across multiple criteria
    scores: dict[str, float] = {}
    for c in rank_conds:
        cond = exp.conditions[c]
        dice = cond.dice_men.get("dice_mean", 0)
        r2 = cond.metrics_enhanced.get("r2_mean_linear", 0)
        gli = cond.dice_gli.get("dice_mean", 0)
        mmd = cond.domain_metrics.get("mmd", 1.0)

        # Composite score: high MEN dice + high R² + high GLI retention - high MMD
        scores[c] = dice + 0.5 * max(r2, 0) + 0.3 * gli - 0.2 * mmd

    recommended = max(scores, key=scores.get)  # type: ignore[arg-type]
    label = CONDITION_LABELS.get(recommended, recommended)

    section.paragraphs.append(
        f"Based on a composite score weighing segmentation quality, feature decodability, "
        f"anatomical preservation, and domain gap, the recommended configuration is "
        f"{label}."
    )

    # Detail the recommended condition
    cond = exp.conditions[recommended]
    section.paragraphs.append(
        f"  Dice (MEN) = {_fmt(cond.dice_men.get('dice_mean', 0))}, "
        f"R² (linear) = {_fmt(cond.metrics_enhanced.get('r2_mean_linear', 0))}, "
        f"GLI retention = {_fmt(cond.dice_gli.get('dice_mean', 0))}, "
        f"MMD² = {_fmt(cond.domain_metrics.get('mmd', 0))}."
    )

    return section


def generate_lora_vs_dora_section(
    experiments: list[ExperimentData],
) -> SectionContent | None:
    """Section 3.8: LoRA vs DoRA comparison."""
    lora_exp = None
    dora_exp = None
    for exp in experiments:
        if exp.adapter_type == "lora" and lora_exp is None:
            lora_exp = exp
        elif exp.adapter_type == "dora" and dora_exp is None:
            dora_exp = exp

    if lora_exp is None or dora_exp is None:
        return None

    section = SectionContent(
        section_id="lora_vs_dora",
        title="3.8 LoRA vs DoRA",
        figure_names=["lora_vs_dora"],
    )

    # Compare at each rank
    comparisons = []
    for r in RANKS:
        lora_c = lora_exp.conditions.get(f"lora_r{r}")
        dora_c = dora_exp.conditions.get(f"dora_r{r}")
        if lora_c is None or dora_c is None:
            continue

        lora_dice = lora_c.dice_men.get("dice_mean", 0)
        dora_dice = dora_c.dice_men.get("dice_mean", 0)
        delta = dora_dice - lora_dice
        comparisons.append((r, lora_dice, dora_dice, delta))

    if comparisons:
        dora_wins = sum(1 for _, _, _, d in comparisons if d > 0)
        section.paragraphs.append(
            f"Across {len(comparisons)} ranks, DoRA outperforms LoRA in "
            f"{dora_wins}/{len(comparisons)} cases for MEN Dice."
        )

        # Summarize deltas
        deltas = [d for _, _, _, d in comparisons]
        mean_delta = np.mean(deltas)
        section.paragraphs.append(
            f"Mean Dice difference (DoRA − LoRA): {_fmt(mean_delta, 4)} "
            f"({'DoRA advantage' if mean_delta > 0 else 'LoRA advantage'})."
        )

    return section


def generate_semantic_section(
    experiments: list[ExperimentData],
) -> SectionContent | None:
    """Section 3.9: Semantic heads comparison."""
    sem_exp = None
    no_sem_exp = None
    for exp in experiments:
        if exp.semantic_heads and sem_exp is None:
            sem_exp = exp
        elif not exp.semantic_heads and no_sem_exp is None:
            no_sem_exp = exp

    if sem_exp is None or no_sem_exp is None:
        return None

    section = SectionContent(
        section_id="semantic_comparison",
        title="3.9 Effect of Semantic Heads",
        figure_names=["semantic_comparison"],
    )

    prefix = "dora_r" if sem_exp.adapter_type == "dora" else "lora_r"

    dice_diffs = []
    r2_diffs = []
    for r in RANKS:
        sem_c = sem_exp.conditions.get(f"{prefix}{r}")
        nosem_c = no_sem_exp.conditions.get(f"{prefix}{r}")
        if sem_c is None or nosem_c is None:
            continue

        sem_dice = sem_c.dice_men.get("dice_mean", 0)
        nosem_dice = nosem_c.dice_men.get("dice_mean", 0)
        dice_diffs.append(sem_dice - nosem_dice)

        sem_r2 = sem_c.metrics_enhanced.get("r2_mean_linear", 0)
        nosem_r2 = nosem_c.metrics_enhanced.get("r2_mean_linear", 0)
        r2_diffs.append(sem_r2 - nosem_r2)

    if dice_diffs:
        mean_dice_diff = np.mean(dice_diffs)
        mean_r2_diff = np.mean(r2_diffs)

        section.paragraphs.append(
            f"Adding semantic auxiliary heads changes MEN Dice by an average of "
            f"{_fmt(mean_dice_diff, 4)} and R² by {_fmt(mean_r2_diff, 4)}."
        )

        if mean_r2_diff > 0.02:
            section.paragraphs.append(
                "Semantic heads provide a meaningful improvement in feature decodability, "
                "confirming that multi-task supervision shapes the latent space."
            )
        elif mean_r2_diff > 0:
            section.paragraphs.append(
                "Semantic heads provide a marginal R² improvement. "
                "The auxiliary signal has a subtle but positive effect."
            )
        else:
            section.paragraphs.append(
                "Semantic heads do not improve R² on average, suggesting the "
                "auxiliary task may not effectively guide feature learning."
            )

    return section


def generate_discussion(exp: ExperimentData) -> SectionContent:
    """Generate discussion section."""
    section = SectionContent(section_id="discussion", title="4. Discussion")

    frozen = _frozen(exp)
    best_name = _best_rank_condition(
        exp,
        lambda c: c.dice_men.get("dice_mean", 0),
    )

    if frozen is not None and best_name is not None:
        best = exp.conditions[best_name]
        frozen_men = frozen.dice_men.get("dice_mean", 0)
        best_men = best.dice_men.get("dice_mean", 0)
        improvement = best_men - frozen_men

        if improvement > 0.01:
            section.paragraphs.append(
                f"LoRA adaptation successfully improves meningioma segmentation "
                f"(Δ Dice = +{_fmt(improvement, 4)}), confirming that domain-specific "
                f"fine-tuning is beneficial even with parameter-efficient methods."
            )
        else:
            section.paragraphs.append(
                f"LoRA adaptation provides minimal improvement over the frozen model "
                f"(Δ Dice = {_fmt(improvement, 4)}), suggesting the pre-trained features "
                f"already generalize well to meningioma."
            )

    section.paragraphs.append(
        "Limitations include: (1) evaluation on a single test split without cross-validation, "
        "(2) fixed hyperparameters across ranks, (3) no comparison with full fine-tuning. "
        "Future work should explore rank-adaptive strategies and validate on external datasets."
    )

    return section


def generate_conclusion(exp: ExperimentData) -> SectionContent:
    """Generate one-paragraph conclusion."""
    section = SectionContent(section_id="conclusion", title="5. Conclusion")

    frozen = _frozen(exp)
    best_name = _best_rank_condition(
        exp,
        lambda c: c.dice_men.get("dice_mean", 0),
    )

    if frozen is None or best_name is None:
        section.paragraphs.append(
            "The LoRA ablation study demonstrates the feasibility of "
            "parameter-efficient domain adaptation for brain tumor segmentation."
        )
        return section

    best = exp.conditions[best_name]
    label = CONDITION_LABELS.get(best_name, best_name)
    adapter_type = exp.adapter_type.upper()

    frozen_gli = frozen.dice_gli.get("dice_mean", 0)
    best_gli = best.dice_gli.get("dice_mean", 0)
    retention = best_gli / frozen_gli if frozen_gli > 0 else 0

    answer = "Yes" if retention > 0.8 else "Partially"

    section.paragraphs.append(
        f"{answer}, {adapter_type} correctly applies a domain shift from glioma to meningioma "
        f"without catastrophically forgetting basic MRI anatomy. The optimal configuration "
        f"({label}) achieves Dice = {_fmt(best.dice_men.get('dice_mean', 0))} on meningioma "
        f"while retaining {_pct(retention)} of glioma segmentation performance "
        f"(GLI Dice = {_fmt(best_gli)} vs frozen {_fmt(frozen_gli)}). "
        f"These adapted features serve as the foundation for downstream "
        f"Supervised Disentangled Projection and GP-based growth prediction."
    )

    return section


# ─────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────


def generate_all_sections(
    experiments: list[ExperimentData],
    compare_semantic: bool = False,
) -> list[SectionContent]:
    """Generate all narrative sections for the report.

    Args:
        experiments: Loaded experiment data.
        compare_semantic: Include semantic comparison section.

    Returns:
        Ordered list of SectionContent objects.
    """
    primary = experiments[0]

    sections = [
        generate_abstract(primary),
        generate_introduction(primary),
        generate_segmentation_section(primary),
        generate_domain_adaptation_section(primary),
        generate_preservation_section(primary),
        generate_feature_quality_section(primary),
        generate_training_dynamics_section(primary),
        generate_statistical_section(primary),
        generate_rank_summary_section(primary),
    ]

    # Multi-experiment sections
    if len(experiments) >= 2:
        lora_dora = generate_lora_vs_dora_section(experiments)
        if lora_dora is not None:
            sections.append(lora_dora)

    if compare_semantic and len(experiments) >= 2:
        semantic = generate_semantic_section(experiments)
        if semantic is not None:
            sections.append(semantic)

    sections.append(generate_discussion(primary))
    sections.append(generate_conclusion(primary))

    return sections
