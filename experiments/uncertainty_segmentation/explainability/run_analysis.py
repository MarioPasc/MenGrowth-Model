"""Orchestrator for the brain-masked TSI + ASI + DAD pipeline (spec §10).

This is the production entry point for the explainability refactor described
in ``docs/growth-related/lora/EXPLAINABILITY_ANALYSIS_SPEC.md`` (2026-04-14).
It supersedes the legacy ``run_tsi.py``.

Execution stages
----------------
1. **Frozen.**  Load BrainSegFounder once → for each test scan run a single
   forward pass with ``AttentionCapture`` in callback mode, accumulating
   brain-masked TSI (all stages) **and** ASI (stages 1-4) on the fly.
   Write ``raw/tsi_frozen_per_scan.csv`` and ``raw/asi_frozen_per_scan.csv``
   (``member_id = -1``).

2. **Adapted (multi-member).**  For each ``member_id`` in
   ``analysis.member_ids`` (default first 5 members of each rank in
   ``analysis.ranks``), repeat step 1 with the LoRA-adapted model and
   append rows to the same adapted CSVs, tagging each row with
   ``member_id`` and ``rank``.

3. **DAD.**  For one member of one rank (``analysis.dad_member_id``)
   compute row-averaged attention on a volume-matched MEN/GLI cohort
   (frozen *and* adapted) and run the permutation test.  Writes
   ``raw/dad_per_head.csv`` plus the null distribution NPZ.

4. **Tables and figures.**  Aggregate the per-scan CSVs into the
   spec §8 tables, then emit the panel + summary PDFs.

CLI
---
``--config``           Parent uncertainty_segmentation config.
``--analysis-config``  This module's config (default: ``./config.yaml``).
``--run-dir``          Optional explicit run directory (overrides the
                       rank-based pattern lookup).
``--output``           Destination directory for ``raw/``, ``tables/`` and
                       ``figures/``.
``--device``           Compute device (default ``cuda:0``).
``--ranks``            Subset of ranks to evaluate (default: all in config).
``--members``          Subset of member IDs (default: first ``n_adapted_members``).
``--phase``            ``all|frozen|adapted|dad|tables_figures`` for partial reruns.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from .engine.asi import ASIScanAccumulator, ASIScanResult
from .engine.brain_mask import brain_mask_coverage, derive_brain_mask
from .engine.dad import (
    DADScanAccumulator,
    DADScanResult,
    compute_dad_with_permutation,
)
from .engine.data_loader import (
    LoadedDataset,
    build_gli_loader,
    build_men_loader,
    get_wt_mask,
    match_by_volume,
    select_scan_indices,
)
from .engine.hooks import AttentionCapture
from .engine.model_loader import (
    get_checkpoint_path,
    get_run_dir,
    load_adapted_model,
    load_frozen_model,
)
from .engine.tsi import (
    ScanTSIResult,
    compute_tsi_single_scan,
    extract_hidden_states,
)
from .figures.fig_asi_panel import render_asi_panel
from .figures.fig_dad_bar import render_dad_bar
from .figures.fig_summary import render_summary
from .figures.fig_tsi_panel import render_tsi_panel
from .tables.generate_tables import generate_tables

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(output_dir: Path) -> None:
    """Configure root logger to write to console and ``analysis.log``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "analysis.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s — %(message)s")
    )
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------


def _tsi_results_to_df(
    results: list[ScanTSIResult],
    rank: int | None,
    member_id: int,
    thresholds: list[float],
) -> pd.DataFrame:
    """Flatten per-scan TSI results to a tidy CSV row format."""
    rows = []
    for scan_res in results:
        for sr in scan_res.stages:
            row = {
                "scan_id": scan_res.scan_id,
                "condition": scan_res.condition,
                "rank": rank if rank is not None else -1,
                "member_id": member_id,
                "stage": sr.stage,
                "n_channels": sr.n_channels,
                "resolution": f"{sr.resolution[0]}^3",
                "mean_tsi": sr.mean_tsi,
                "std_tsi": sr.std_tsi,
                "wilcoxon_p": sr.wilcoxon_p,
            }
            for tau in thresholds:
                row[f"frac_{tau}"] = sr.frac_above.get(tau, float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


def _asi_results_to_df(
    results: list[tuple[str, ASIScanResult]],
    condition: str,
    rank: int | None,
    member_id: int,
) -> pd.DataFrame:
    """Long-format ASI dataframe: one row per (scan, stage, block, head, window)."""
    rows = []
    for scan_id, sr in results:
        for key, arr in sr.per_block_per_head.items():
            if arr.size == 0:
                continue
            parts = key.split("_")
            stage = int(parts[1])
            block = int(parts[3])
            n_w, n_h = arr.shape
            for w_i in range(n_w):
                for h_i in range(n_h):
                    val = float(arr[w_i, h_i])
                    if not np.isfinite(val):
                        continue
                    rows.append({
                        "scan_id": scan_id,
                        "condition": condition,
                        "rank": rank if rank is not None else -1,
                        "member_id": member_id,
                        "stage": stage,
                        "block": block,
                        "head": h_i,
                        "window_idx": w_i,
                        "asi_value": val,
                    })
    return pd.DataFrame(rows)


def _dad_stats_to_df(
    stats_by_key: dict[str, list],
    condition: str,
    rank: int | None,
    member_id: int,
) -> pd.DataFrame:
    """Flatten ``DADStats`` lists to a CSV row format."""
    rows = []
    for key, stats_list in stats_by_key.items():
        parts = key.split("_")
        stage = int(parts[1])
        block = int(parts[3])
        for st in stats_list:
            rows.append({
                "condition": condition,
                "rank": rank if rank is not None else -1,
                "member_id": member_id,
                "stage": stage,
                "block": block,
                "head": st.head,
                "dad": st.dad_observed,
                "p_value": st.p_value,
                "null_mean": st.null_mean,
                "null_std": st.null_std,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-condition core loop: TSI + ASI in one forward
# ---------------------------------------------------------------------------


@dataclass
class PerConditionOutputs:
    """In-memory container for one condition's per-scan results."""

    tsi_results: list[ScanTSIResult]
    asi_results: list[tuple[str, ASIScanResult]]


def _run_tsi_asi(
    model: torch.nn.Module,
    loaded: LoadedDataset,
    scan_indices: list[int],
    condition: str,
    analysis_cfg: DictConfig,
    device: str,
) -> PerConditionOutputs:
    """Single forward pass per scan; capture attention for ASI, hidden states for TSI."""
    thresholds = list(analysis_cfg.tsi_thresholds)
    asi_stages = set(int(s) for s in analysis_cfg.asi_stages)
    tsi_eps = float(analysis_cfg.tsi_epsilon)

    bm_threshold = analysis_cfg.brain_mask_threshold
    if bm_threshold is not None:
        bm_threshold = float(bm_threshold)

    tsi_results: list[ScanTSIResult] = []
    asi_results: list[tuple[str, ASIScanResult]] = []

    for idx_i, ds_idx in enumerate(scan_indices):
        sample = loaded.dataset[ds_idx]
        images = sample["image"].unsqueeze(0).to(device)
        gt_mask = get_wt_mask(sample["seg"])
        brain_mask = derive_brain_mask(images.cpu(), threshold=bm_threshold)
        sid = loaded.all_scan_ids[loaded.test_indices[ds_idx]]

        coverage = brain_mask_coverage(brain_mask)
        if coverage < 0.05 or coverage > 0.95:
            logger.warning(
                "Scan %s: brain mask coverage %.2f%% — outside expected 30-60%% range",
                sid, 100 * coverage,
            )

        t0 = time.time()
        asi_acc = ASIScanAccumulator(
            gt_mask=gt_mask,
            target_stages=asi_stages,
            min_tumor=int(analysis_cfg.asi_min_tumor_tokens),
            min_nontumor=int(analysis_cfg.asi_min_nontumor_tokens),
        )

        # Hooks installed only for the duration of the forward pass.  We
        # call extract_hidden_states (no_grad) so TSI uses the canonical
        # SwinTransformer output path; ASI piggy-backs on that same forward.
        with AttentionCapture(
            model,
            mode="callback",
            process_fn=asi_acc,
            target_stages=asi_stages,
        ):
            with torch.amp.autocast("cuda", enabled=(device != "cpu")):
                hidden_states = extract_hidden_states(model, images)

        del images
        if device != "cpu":
            torch.cuda.empty_cache()

        scan_tsi = compute_tsi_single_scan(
            hidden_states,
            gt_mask,
            scan_id=sid,
            condition=condition,
            thresholds=thresholds,
            top_k=int(analysis_cfg.tsi_top_k),
            return_maps=(idx_i == 0),
            epsilon=tsi_eps,
            brain_mask=brain_mask,
        )
        tsi_results.append(scan_tsi)
        asi_results.append((sid, asi_acc.result()))

        elapsed = time.time() - t0
        mean_tsi_str = ", ".join(f"{sr.mean_tsi:.2f}" for sr in scan_tsi.stages)
        logger.info(
            "  [%s] scan %d/%d (%s): %.1fs — mean_TSI=[%s]",
            condition, idx_i + 1, len(scan_indices), sid, elapsed, mean_tsi_str,
        )
        del hidden_states

    return PerConditionOutputs(tsi_results=tsi_results, asi_results=asi_results)


# ---------------------------------------------------------------------------
# DAD core loop
# ---------------------------------------------------------------------------


def _run_dad_for_cohort(
    model: torch.nn.Module,
    loaded: LoadedDataset,
    scan_indices: list[int],
    analysis_cfg: DictConfig,
    device: str,
    label: str,
) -> list[DADScanResult]:
    """Per-scan row-averaged attention for one cohort."""
    target_stages = set(int(s) for s in analysis_cfg.dad_stages)
    results: list[DADScanResult] = []

    for idx_i, ds_idx in enumerate(scan_indices):
        sample = loaded.dataset[ds_idx]
        images = sample["image"].unsqueeze(0).to(device)
        sid = loaded.all_scan_ids[loaded.test_indices[ds_idx]]

        t0 = time.time()
        acc = DADScanAccumulator(target_stages=target_stages)
        with AttentionCapture(
            model,
            mode="callback",
            process_fn=acc,
            target_stages=target_stages,
        ):
            with torch.amp.autocast("cuda", enabled=(device != "cpu")):
                with torch.no_grad():
                    _ = extract_hidden_states(model, images)
        results.append(acc.result())

        del images
        if device != "cpu":
            torch.cuda.empty_cache()
        logger.info(
            "  [DAD-%s] scan %d/%d (%s): %.1fs",
            label, idx_i + 1, len(scan_indices), sid, time.time() - t0,
        )

    return results


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _save_tsi_channels(
    results: list[ScanTSIResult],
    out_path: Path,
) -> None:
    """Save per-channel TSI arrays as NPZ for figure regeneration."""
    arrays: dict[str, np.ndarray] = {}
    for scan_res in results:
        for sr in scan_res.stages:
            arrays[f"{scan_res.scan_id}_stage{sr.stage}"] = sr.tsi_per_channel
    np.savez_compressed(out_path, **arrays)


def _save_asi_window_stats(
    results: list[tuple[str, ASIScanResult]],
    out_path: Path,
) -> None:
    """Save per-window ASI arrays as NPZ for distribution plots."""
    arrays: dict[str, np.ndarray] = {}
    for sid, sr in results:
        for key, arr in sr.per_block_per_head.items():
            if arr.size == 0:
                continue
            arrays[f"{sid}__{key}"] = arr.astype(np.float32)
    np.savez_compressed(out_path, **arrays)


def _save_dad_null(
    null_per_key: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Save the permutation null distribution as NPZ (one array per key)."""
    np.savez_compressed(out_path, **null_per_key)


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append ``df`` to ``path`` (write header if file is new)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", index=False, header=not path.exists())


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def _resolve_member_ids(analysis_cfg: DictConfig) -> list[int]:
    explicit = analysis_cfg.get("member_ids", None)
    if explicit is not None:
        return [int(m) for m in explicit]
    n = int(analysis_cfg.n_adapted_members)
    return list(range(n))


def _resolve_ranks(
    analysis_cfg: DictConfig, ranks_override: list[int] | None
) -> list[int]:
    if ranks_override:
        return ranks_override
    return [int(r) for r in analysis_cfg.ranks]


def phase_frozen(
    config: DictConfig,
    analysis_cfg: DictConfig,
    loaded: LoadedDataset,
    scan_indices: list[int],
    raw_dir: Path,
    device: str,
) -> PerConditionOutputs:
    """Phase 1: frozen TSI + ASI in a single hooked forward per scan."""
    logger.info("=" * 60)
    logger.info("PHASE 1: FROZEN")
    logger.info("=" * 60)

    model = load_frozen_model(config, device=device)
    out = _run_tsi_asi(
        model, loaded, scan_indices,
        condition="frozen",
        analysis_cfg=analysis_cfg,
        device=device,
    )
    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    thresholds = list(analysis_cfg.tsi_thresholds)
    tsi_df = _tsi_results_to_df(out.tsi_results, rank=None, member_id=-1, thresholds=thresholds)
    asi_df = _asi_results_to_df(out.asi_results, condition="frozen", rank=None, member_id=-1)

    tsi_df.to_csv(raw_dir / "tsi_frozen_per_scan.csv", index=False)
    asi_df.to_csv(raw_dir / "asi_frozen_per_scan.csv", index=False)
    _save_tsi_channels(out.tsi_results, raw_dir / "tsi_frozen_channels.npz")
    _save_asi_window_stats(out.asi_results, raw_dir / "asi_frozen_window_stats.npz")
    logger.info("Frozen artefacts written to %s", raw_dir)
    return out


def phase_adapted(
    config: DictConfig,
    analysis_cfg: DictConfig,
    loaded: LoadedDataset,
    scan_indices: list[int],
    raw_dir: Path,
    device: str,
    ranks: list[int],
    member_ids: list[int],
    run_dir_override: Path | None,
) -> dict[tuple[int, int], PerConditionOutputs]:
    """Phase 2: per-member TSI + ASI on the same scans as frozen."""
    logger.info("=" * 60)
    logger.info("PHASE 2: ADAPTED (ranks=%s, members=%s)", ranks, member_ids)
    logger.info("=" * 60)

    thresholds = list(analysis_cfg.tsi_thresholds)
    tsi_csv = raw_dir / "tsi_adapted_per_scan.csv"
    asi_csv = raw_dir / "asi_adapted_per_scan.csv"
    # Reset target CSVs at the start of the phase so reruns don't duplicate rows.
    for p in (tsi_csv, asi_csv):
        if p.exists():
            p.unlink()

    all_outputs: dict[tuple[int, int], PerConditionOutputs] = {}
    all_tsi_for_npz: list[ScanTSIResult] = []
    all_asi_for_npz: list[tuple[str, ASIScanResult]] = []

    for rank in ranks:
        for mid in member_ids:
            condition = f"adapted_r{rank}_m{mid}"
            logger.info("--- Loading rank=%d member=%d ---", rank, mid)
            model = load_adapted_model(
                config, analysis_cfg, rank=rank,
                member_id=mid, device=device,
                run_dir=run_dir_override,
            )
            out = _run_tsi_asi(
                model, loaded, scan_indices,
                condition="adapted",
                analysis_cfg=analysis_cfg,
                device=device,
            )
            del model
            if device != "cpu":
                torch.cuda.empty_cache()

            tsi_df = _tsi_results_to_df(out.tsi_results, rank=rank, member_id=mid, thresholds=thresholds)
            asi_df = _asi_results_to_df(out.asi_results, condition="adapted", rank=rank, member_id=mid)
            _append_csv(tsi_df, tsi_csv)
            _append_csv(asi_df, asi_csv)
            all_outputs[(rank, mid)] = out
            all_tsi_for_npz.extend(out.tsi_results)
            all_asi_for_npz.extend(out.asi_results)

    if all_tsi_for_npz:
        _save_tsi_channels(all_tsi_for_npz, raw_dir / "tsi_adapted_channels.npz")
    if all_asi_for_npz:
        _save_asi_window_stats(all_asi_for_npz, raw_dir / "asi_adapted_window_stats.npz")
    logger.info("Adapted artefacts written to %s", raw_dir)
    return all_outputs


def phase_dad(
    config: DictConfig,
    analysis_cfg: DictConfig,
    raw_dir: Path,
    device: str,
    ranks: list[int],
    run_dir_override: Path | None,
) -> None:
    """Phase 3: DAD on volume-matched MEN/GLI cohorts (frozen + one adapted member)."""
    logger.info("=" * 60)
    logger.info("PHASE 3: DAD")
    logger.info("=" * 60)

    roi_size = tuple(int(x) for x in analysis_cfg.roi_size)
    n_scans = int(analysis_cfg.n_scans_dad)
    seed = int(analysis_cfg.seed)
    selection = str(analysis_cfg.scan_selection)
    n_perm = int(analysis_cfg.dad_permutations)
    dad_member_id = int(analysis_cfg.dad_member_id)

    men_loader = build_men_loader(config, roi_size=roi_size)
    gli_loader = build_gli_loader(config, roi_size=roi_size)

    men_indices = select_scan_indices(len(men_loader.dataset), n_scans, selection, seed)
    # Pick a generous GLI candidate pool (>= n_scans) to allow volume matching.
    gli_pool_size = min(len(gli_loader.dataset), max(n_scans * 4, n_scans + 10))
    gli_candidate_indices = select_scan_indices(
        len(gli_loader.dataset), gli_pool_size, selection, seed
    )
    pairs = match_by_volume(men_loader, men_indices, gli_loader, gli_candidate_indices)
    men_paired = [p[0] for p in pairs]
    gli_paired = [p[1] for p in pairs]

    dad_csv = raw_dir / "dad_per_head.csv"
    if dad_csv.exists():
        dad_csv.unlink()
    null_arrays: dict[str, np.ndarray] = {}

    def _run_one(model: torch.nn.Module, condition: str, rank: int | None, member_id: int) -> None:
        logger.info("DAD MEN cohort (%s)", condition)
        men_results = _run_dad_for_cohort(
            model, men_loader, men_paired, analysis_cfg, device, "MEN",
        )
        logger.info("DAD GLI cohort (%s)", condition)
        gli_results = _run_dad_for_cohort(
            model, gli_loader, gli_paired, analysis_cfg, device, "GLI",
        )
        stats = compute_dad_with_permutation(
            cohort_a=men_results, cohort_b=gli_results,
            n_perm=n_perm, seed=seed,
        )
        df = _dad_stats_to_df(stats, condition=condition, rank=rank, member_id=member_id)
        _append_csv(df, dad_csv)
        # Stash null mean/std into the NPZ per (condition, key) for diagnostic plots.
        for key in stats:
            null_arrays[f"{condition}__{key}__null_mean"] = np.array(
                [s.null_mean for s in stats[key]]
            )
            null_arrays[f"{condition}__{key}__null_std"] = np.array(
                [s.null_std for s in stats[key]]
            )

    # Frozen
    frozen = load_frozen_model(config, device=device)
    _run_one(frozen, condition="frozen", rank=None, member_id=-1)
    del frozen
    if device != "cpu":
        torch.cuda.empty_cache()

    # One adapted member per rank.
    for rank in ranks:
        adapted = load_adapted_model(
            config, analysis_cfg, rank=rank,
            member_id=dad_member_id, device=device,
            run_dir=run_dir_override,
        )
        _run_one(adapted, condition=f"adapted_r{rank}", rank=rank, member_id=dad_member_id)
        del adapted
        if device != "cpu":
            torch.cuda.empty_cache()

    if null_arrays:
        _save_dad_null(null_arrays, raw_dir / "dad_permutation_null.npz")
    logger.info("DAD artefacts written to %s", raw_dir)


def phase_tables_figures(
    config: DictConfig,
    analysis_cfg: DictConfig,
    raw_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    """Phase 4: tables (spec §8) + figures (spec §7)."""
    logger.info("=" * 60)
    logger.info("PHASE 4: TABLES + FIGURES")
    logger.info("=" * 60)
    written = generate_tables(raw_dir=raw_dir, out_dir=tables_dir)
    logger.info("Tables written: %s", list(written.keys()))

    save_dpi = int(analysis_cfg.get("figure", {}).get("save_dpi", 300)) \
        if "figure" in analysis_cfg else 300
    fmt = analysis_cfg.get("figure", {}).get("save_format", "pdf") \
        if "figure" in analysis_cfg else "pdf"
    lora_stages = set(int(s) for s in config.lora.target_stages)

    # ---- TSI panels ----
    for cond in ("frozen", "adapted"):
        csv = raw_dir / f"tsi_{cond}_per_scan.csv"
        if not csv.exists():
            logger.info("Skipping TSI panel for %s (CSV missing)", cond)
            continue
        df = pd.read_csv(csv)
        out_path = figures_dir / f"tsi_{cond}_brainmasked.{fmt}"
        render_tsi_panel(
            df, out_path, title=f"Brain-masked TSI — {cond}",
            lora_stages=lora_stages, dpi=save_dpi,
        )

    # ---- ASI panels ----
    for cond in ("frozen", "adapted"):
        csv = raw_dir / f"asi_{cond}_per_scan.csv"
        if not csv.exists():
            logger.info("Skipping ASI panel for %s (CSV missing)", cond)
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        out_path = figures_dir / f"asi_{cond}.{fmt}"
        render_asi_panel(df, out_path, title=f"ASI per stage — {cond}", dpi=save_dpi)

    # ---- DAD bar ----
    dad_csv = raw_dir / "dad_per_head.csv"
    if dad_csv.exists():
        dad_df = pd.read_csv(dad_csv)
        for cond in dad_df["condition"].unique():
            out_path = figures_dir / f"dad_{cond}.{fmt}"
            render_dad_bar(
                dad_df, out_path, title=f"DAD — {cond} (MEN vs GLI)",
                condition=cond, dpi=save_dpi,
            )

    # ---- Combined summary ----
    tsi_frozen_csv = raw_dir / "tsi_frozen_per_scan.csv"
    asi_frozen_csv = raw_dir / "asi_frozen_per_scan.csv"
    if tsi_frozen_csv.exists():
        tsi_df = pd.read_csv(tsi_frozen_csv)
        if (raw_dir / "tsi_adapted_per_scan.csv").exists():
            tsi_df = pd.concat(
                [tsi_df, pd.read_csv(raw_dir / "tsi_adapted_per_scan.csv")],
                ignore_index=True,
            )
        asi_df = None
        if asi_frozen_csv.exists():
            asi_df = pd.read_csv(asi_frozen_csv)
            if (raw_dir / "asi_adapted_per_scan.csv").exists():
                asi_df = pd.concat(
                    [asi_df, pd.read_csv(raw_dir / "asi_adapted_per_scan.csv")],
                    ignore_index=True,
                )
        dad_df = pd.read_csv(dad_csv) if dad_csv.exists() else None
        for cond in ("frozen", "adapted"):
            tsi_sub = tsi_df[tsi_df["condition"] == cond]
            if tsi_sub.empty:
                continue
            out_path = figures_dir / f"summary_combined_{cond}.{fmt}"
            render_summary(
                tsi_df=tsi_df, asi_df=asi_df, dad_df=dad_df,
                lora_stages=lora_stages, out_path=out_path,
                condition=cond, dpi=save_dpi,
            )

    logger.info("Figures written to %s", figures_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brain-masked TSI + ASI + DAD pipeline (spec §10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Parent config.yaml")
    parser.add_argument(
        "--analysis-config", default=None,
        help="Analysis config (default: explainability/config.yaml)",
    )
    parser.add_argument("--run-dir", default=None, help="Override LoRA run directory")
    parser.add_argument("--output", default=None, help="Override output_dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", type=int, nargs="+", default=None)
    parser.add_argument("--members", type=int, nargs="+", default=None)
    parser.add_argument(
        "--phase", default="all",
        choices=["all", "frozen", "adapted", "dad", "tables_figures"],
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    analysis_path = args.analysis_config or (
        Path(args.config).parent / "explainability" / "config.yaml"
    )
    analysis_config = OmegaConf.load(analysis_path)

    output_dir = Path(args.output or analysis_config.paths.output_dir)
    raw_dir = output_dir / "raw"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for d in (raw_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    OmegaConf.save(
        OmegaConf.merge(config, analysis_config),
        output_dir / "config_snapshot.yaml",
        resolve=True,
    )

    ranks = _resolve_ranks(analysis_config.analysis, args.ranks)
    member_ids = args.members if args.members else _resolve_member_ids(analysis_config.analysis)
    run_dir_override = Path(args.run_dir) if args.run_dir else None

    logger.info("Pipeline phase=%s ranks=%s members=%s output=%s",
                args.phase, ranks, member_ids, output_dir)
    logger.info("Checkpoint: %s", get_checkpoint_path(config))
    if run_dir_override is None:
        for r in ranks:
            logger.info("Run dir (rank=%d): %s", r, get_run_dir(config, analysis_config, r))
    else:
        logger.info("Run dir override: %s", run_dir_override)

    # Load MEN test cohort once (frozen + adapted phases share it).
    roi_size = tuple(int(x) for x in analysis_config.analysis.roi_size)
    men_loader = build_men_loader(config, roi_size=roi_size)
    scan_indices = select_scan_indices(
        len(men_loader.dataset),
        int(analysis_config.analysis.n_scans_tsi),
        str(analysis_config.analysis.scan_selection),
        int(analysis_config.analysis.seed),
    )
    logger.info("MEN scans selected: %d / %d", len(scan_indices), len(men_loader.dataset))

    t_start = time.time()

    if args.phase in ("all", "frozen"):
        phase_frozen(config, analysis_config.analysis, men_loader, scan_indices, raw_dir, args.device)
    if args.phase in ("all", "adapted"):
        phase_adapted(
            config, analysis_config.analysis, men_loader, scan_indices, raw_dir,
            args.device, ranks, member_ids, run_dir_override,
        )
    if args.phase in ("all", "dad"):
        phase_dad(config, analysis_config.analysis, raw_dir, args.device, ranks, run_dir_override)
    if args.phase in ("all", "tables_figures"):
        phase_tables_figures(
            config, analysis_config.analysis, raw_dir, tables_dir, figures_dir,
        )

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
