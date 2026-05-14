"""Model complexity vs performance table (Tab 3).

One row per adapter configuration (Frozen BSF + each LoRA rank).
Columns: Adapter, ΔW Params, Total Params, Dice (mean ± 95% CI),
Mean Variance (mean ± 95% CI), Mean Entropy (mean ± 95% CI).

``Mean Variance`` is the inter-member variance of per-scan Dice (a
performance-dispersion metric). ``Mean Entropy`` is the predictive binary
entropy of the ensemble-mean meningioma probability, averaged over voxels
inside the predicted meningioma mask and then over the test scans that have
per-voxel uncertainty saved (``ensemble_probs.nii.gz``; optionally the
per-member maps when ``tab3_max_member_scans`` > 0).

Produces CSV, Markdown, and LaTeX (booktabs) renditions.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import (
    InterLoraData,
    RankRun,
    compute_scan_mean_entropy,
)

logger = logging.getLogger(__name__)

_LORA_PARAMS_PER_RANK: int = 23_040
_BASE_MODEL_PARAMS: int = 62_191_941

# Default uncertainty settings (overridable via the render() config).
_DEFAULT_ENTROPY_CHANNEL: int = 0  # BSF ch0 = meningioma mass (TC)
_DEFAULT_N_MEMBERS: int = 20
# Scans whose only per-voxel artefact is the per-member probability stack are
# expensive to read. 0 = skip them entirely (use only ensemble_probs.nii.gz).
_DEFAULT_MAX_MEMBER_SCANS: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapter_label(rank: int) -> str:
    if rank == 0:
        return "Frozen BSF"
    return f"LoRA $r{{=}}{rank}$"


def _adapter_label_plain(rank: int) -> str:
    if rank == 0:
        return "Frozen BSF"
    return f"LoRA r={rank}"


def _fmt_params(n: int) -> str:
    if n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_pm(mean: float, ci_lo: float, ci_hi: float) -> str:
    """Format as 'mean ± half-width' using 95% CI bounds."""
    if any(math.isnan(v) for v in (mean, ci_lo, ci_hi)):
        return "—"
    hw = (ci_hi - ci_lo) / 2
    return f"{mean:.3f} ± {hw:.3f}"


def _fmt_pm_sci(mean: float, ci_lo: float, ci_hi: float) -> str:
    """Format as 'mean ± half-width' for small values (scientific)."""
    if any(math.isnan(v) for v in (mean, ci_lo, ci_hi)):
        return "—"
    hw = (ci_hi - ci_lo) / 2
    return f"{mean:.4f} ± {hw:.4f}"


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap mean and 95% CI for a 1-D array."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    mean = float(values.mean())
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def _compute_mean_variance(
    rank_run: RankRun,
) -> tuple[float, float, float]:
    """Compute mean inter-member Dice variance with bootstrap 95% CI.

    Returns (mean, ci_lo, ci_hi).  For baseline (no members), returns
    (0, 0, 0).
    """
    pm = rank_run.per_member_dice
    if pm.empty:
        return 0.0, 0.0, 0.0

    dice_cols = [c for c in ["dice_tc", "dice_wt", "dice_et"] if c in pm.columns]
    if not dice_cols:
        return float("nan"), float("nan"), float("nan")

    per_scan_var = pm.groupby("scan_id")[dice_cols].var()
    scan_mean_var = per_scan_var.mean(axis=1).values

    return _bootstrap_ci(scan_mean_var)


def _iter_scan_dirs(predictions_dir: Path):
    """Yield per-scan prediction directories under predictions/ (incl. brats_men_test/)."""
    seen: set[str] = set()
    for parent in (predictions_dir, predictions_dir / "brats_men_test"):
        if not parent.is_dir():
            continue
        for d in sorted(parent.iterdir()):
            if d.is_dir() and d.name not in seen:
                seen.add(d.name)
                yield d


def _compute_mean_entropy(
    rank_run: RankRun,
    *,
    entropy_channel: int,
    n_members: int,
    max_member_scans: int,
) -> tuple[float, float, float]:
    """Mean predictive entropy with bootstrap 95% CI for one rank.

    Per scan: mean predictive binary entropy of the ensemble-mean meningioma
    probability inside the predicted meningioma mask
    (``io_layer.compute_scan_mean_entropy``). Scans exposing
    ``ensemble_probs.nii.gz`` are always included; scans whose only per-voxel
    artefact is the per-member probability stack are included up to
    ``max_member_scans`` (deterministic sorted order) to bound runtime.

    Returns:
        ``(mean, ci_lo, ci_hi)`` across the contributing scans, or
        ``(nan, nan, nan)`` when fewer than two scans contribute.
    """
    if rank_run.predictions_dir is None:
        return float("nan"), float("nan"), float("nan")

    values: list[float] = []
    member_budget = max(0, int(max_member_scans))
    for scan_dir in _iter_scan_dirs(rank_run.predictions_dir):
        has_ensemble = (scan_dir / "ensemble_probs.nii.gz").exists()
        has_members = (scan_dir / "member_0_probs.nii.gz").exists()
        if not has_ensemble:
            if not (has_members and member_budget > 0):
                continue
            member_budget -= 1
        score = compute_scan_mean_entropy(scan_dir, entropy_channel, n_members)
        if np.isfinite(score):
            values.append(float(score))

    if len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    return _bootstrap_ci(np.asarray(values, dtype=np.float64))


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def _load_entropy_cache(cache_path: Path | None) -> dict[str, list[float]]:
    """Load the per-rank Mean Entropy cache, or an empty dict."""
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _build_table(
    data: InterLoraData,
    *,
    entropy_channel: int = _DEFAULT_ENTROPY_CHANNEL,
    n_members: int = _DEFAULT_N_MEMBERS,
    max_member_scans: int = _DEFAULT_MAX_MEMBER_SCANS,
    cache_path: Path | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Build the complexity table with one row per configuration.

    The ``Mean Entropy`` column is computed per rank via
    ``_compute_mean_entropy`` and cached to ``cache_path`` (keyed by rank) so
    repeated report runs do not re-read the prediction NIfTIs.
    """
    cm = data.compiled_metrics

    rows: list[dict] = []
    entropy_cache = {} if force_recompute else _load_entropy_cache(cache_path)
    cache_dirty = False

    # Frozen BSF row (rank = 0)
    bsf_dice = cm[(cm["rank"] == 0) & (cm["label"] == "mean")]
    if not bsf_dice.empty:
        r = bsf_dice.iloc[0]
        dice_str = _fmt_pm(r["dice_mean"], r["dice_ci_lo"], r["dice_ci_hi"])
    else:
        dice_str = "—"

    rows.append(
        {
            "rank_val": 0,
            "adapter": _adapter_label_plain(0),
            "adapter_latex": _adapter_label(0),
            "delta_w": 0,
            "delta_w_fmt": _fmt_params(0),
            "total_params": _BASE_MODEL_PARAMS,
            "total_params_fmt": _fmt_params(_BASE_MODEL_PARAMS),
            "dice": dice_str,
            "mean_var": "0.0000 ± 0.0000",
            # Frozen BSF is a single deterministic model — no ensemble entropy.
            "mean_entropy": "—",
        }
    )

    # LoRA rank rows
    for rr in data.ranks:
        if rr.rank == 0:
            continue

        lora_params = rr.rank * _LORA_PARAMS_PER_RANK
        total = _BASE_MODEL_PARAMS + lora_params

        rank_dice = cm[(cm["rank"] == rr.rank) & (cm["label"] == "mean")]
        if not rank_dice.empty:
            r = rank_dice.iloc[0]
            dice_str = _fmt_pm(r["dice_mean"], r["dice_ci_lo"], r["dice_ci_hi"])
        else:
            dice_str = "—"

        mv_mean, mv_lo, mv_hi = _compute_mean_variance(rr)
        var_str = _fmt_pm_sci(mv_mean, mv_lo, mv_hi)

        rank_key = str(rr.rank)
        if rank_key in entropy_cache:
            me_mean, me_lo, me_hi = entropy_cache[rank_key]
        else:
            me_mean, me_lo, me_hi = _compute_mean_entropy(
                rr,
                entropy_channel=entropy_channel,
                n_members=n_members,
                max_member_scans=max_member_scans,
            )
            entropy_cache[rank_key] = [me_mean, me_lo, me_hi]
            cache_dirty = True
        ent_str = _fmt_pm_sci(me_mean, me_lo, me_hi)

        rows.append(
            {
                "rank_val": rr.rank,
                "adapter": _adapter_label_plain(rr.rank),
                "adapter_latex": _adapter_label(rr.rank),
                "delta_w": lora_params,
                "delta_w_fmt": _fmt_params(lora_params),
                "total_params": total,
                "total_params_fmt": _fmt_params(total),
                "dice": dice_str,
                "mean_var": var_str,
                "mean_entropy": ent_str,
            }
        )

    if cache_dirty and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(entropy_cache, f, indent=2)

    df = pd.DataFrame(rows).sort_values("rank_val").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------


def _write_csv(table: pd.DataFrame, out_dir: Path) -> None:
    out = table[["adapter", "delta_w", "total_params", "dice", "mean_var", "mean_entropy"]].copy()
    out.columns = [
        "Adapter",
        "Delta_W_Params",
        "Total_Params",
        "Dice",
        "Mean_Dice_Variance",
        "Mean_Entropy",
    ]
    path = out_dir / "tab3_model_complexity.csv"
    out.to_csv(path, index=False)
    logger.info("CSV written to %s", path)


def _write_markdown(table: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "| Adapter | ΔW Params | Total Params | Dice (mean ± 95% CI) "
        "| Mean Dice Variance (mean ± 95% CI) | Mean Entropy (mean ± 95% CI) |",
        "|:--------|----------:|-------------:|:--------------------:"
        "|:---------------------------------:|:----------------------------:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['adapter']} | {row['delta_w_fmt']} | {row['total_params_fmt']} "
            f"| {row['dice']} | {row['mean_var']} | {row['mean_entropy']} |"
        )
    path = out_dir / "tab3_model_complexity.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Markdown written to %s", path)


def _write_latex(table: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Model complexity versus segmentation performance. "
        r"$\Delta W$ denotes the LoRA adapter parameters. "
        r"Dice, mean Dice variance, and mean predictive entropy are reported "
        r"as mean $\pm$ 95\% CI (bootstrap, $B{=}10\,000$). Mean Dice variance "
        r"is the inter-member variance of per-scan Dice; mean entropy is the "
        r"predictive binary entropy of the ensemble-mean meningioma "
        r"probability, averaged inside the predicted meningioma mask over the "
        r"test scans with per-voxel uncertainty saved.}",
        r"\label{tab:model_complexity}",
        r"\begin{tabular}{l r r c c c}",
        r"\toprule",
        r"Adapter & $\Delta W$ & Total & Dice & Mean Var. & Mean Ent. \\",
        r"\midrule",
    ]

    # Find best Dice row (excluding BSF)
    non_bsf = table[table["rank_val"] > 0]
    best_dice_idx = -1
    if not non_bsf.empty:
        best_dice_idx = int(non_bsf["rank_val"].iloc[0])
        best_dice_val = 0.0
        for _, row in non_bsf.iterrows():
            d = row["dice"]
            if d != "—":
                val = float(d.split("±")[0].strip())
                if val > best_dice_val:
                    best_dice_val = val
                    best_dice_idx = int(row["rank_val"])

    for i, (_, row) in enumerate(table.iterrows()):
        adapter = row["adapter_latex"]
        dw = row["delta_w_fmt"]
        tp = row["total_params_fmt"]
        dice = row["dice"].replace("±", r"$\pm$")
        mvar = row["mean_var"].replace("±", r"$\pm$")
        ment = row["mean_entropy"].replace("±", r"$\pm$")

        if int(row["rank_val"]) == best_dice_idx:
            dice = r"\textbf{" + dice + "}"

        line = f"{adapter} & {dw} & {tp} & {dice} & {mvar} & {ment} \\\\"

        if i == 0:
            line += "\n" + r"\midrule"

        lines.append(line)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path = out_dir / "tab3_model_complexity.tex"
    path.write_text("\n".join(lines) + "\n")
    logger.info("LaTeX written to %s", path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render(data: InterLoraData, config: dict, out_dir: Path) -> None:
    """Render Tab 3: model complexity vs performance.

    Args:
        data: Aggregated inter-LoRA data container.
        config: ``uncertainty`` config block. Recognised keys:
            ``entropy_channel`` (default 0), ``n_members`` (default 20),
            ``tab3_max_member_scans`` (default 0 — Mean Entropy uses only
            scans with ``ensemble_probs.nii.gz``), ``force_recompute``.
        out_dir: Output directory for table files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Rendering Tab 3 (model complexity) to %s", out_dir)

    config = config or {}
    entropy_channel = int(config.get("entropy_channel", _DEFAULT_ENTROPY_CHANNEL))
    n_members = int(config.get("n_members", _DEFAULT_N_MEMBERS))
    max_member_scans = int(config.get("tab3_max_member_scans", _DEFAULT_MAX_MEMBER_SCANS))
    force_recompute = bool(config.get("force_recompute", False))
    cache_path = out_dir.parent / "data" / "tab3_mean_entropy.json"

    table = _build_table(
        data,
        entropy_channel=entropy_channel,
        n_members=n_members,
        max_member_scans=max_member_scans,
        cache_path=cache_path,
        force_recompute=force_recompute,
    )

    if table.empty:
        logger.warning("Tab 3: empty table — skipping.")
        return

    _write_csv(table, out_dir)
    _write_markdown(table, out_dir)
    _write_latex(table, out_dir)

    logger.info("Tab 3 complete: %d rows.", len(table))
