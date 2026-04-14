"""Single-metric debugging entry point.

Useful for iteration on one piece of the pipeline (e.g. swapping the
brain-mask threshold) without paying the full TSI + ASI + DAD cost.

Usage::

    python -m experiments.uncertainty_segmentation.explainability.run_single_metric \
        --metric tsi --condition frozen --n-scans 2 \
        --config experiments/uncertainty_segmentation/config.yaml

The output CSV(s) land under ``{output_dir}/raw/`` so they overwrite
the same files that ``run_analysis.py`` writes — be careful when
mixing in an existing run directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from .engine.data_loader import build_men_loader, select_scan_indices
from .engine.model_loader import load_adapted_model, load_frozen_model
from .run_analysis import (
    _asi_results_to_df,
    _run_dad_for_cohort,
    _run_tsi_asi,
    _save_asi_window_stats,
    _save_tsi_channels,
    _tsi_results_to_df,
    phase_dad,
)

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single explainability metric")
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-config", default=None)
    parser.add_argument("--metric", choices=["tsi", "asi", "dad"], required=True)
    parser.add_argument(
        "--condition", default="frozen", choices=["frozen", "adapted"],
        help="Model variant to use; ignored for --metric dad (runs both)",
    )
    parser.add_argument("--n-scans", type=int, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--member-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    _setup_logging()

    config = OmegaConf.load(args.config)
    analysis_path = args.analysis_config or (
        Path(args.config).parent / "explainability" / "config.yaml"
    )
    analysis_config = OmegaConf.load(analysis_path)
    output_dir = Path(args.output or analysis_config.paths.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_dir_override = Path(args.run_dir) if args.run_dir else None

    if args.metric == "dad":
        # DAD always exercises both frozen and one adapted member.
        ranks = [args.rank]
        # Override n_scans_dad if user requested.
        if args.n_scans is not None:
            analysis_config.analysis.n_scans_dad = args.n_scans
        phase_dad(
            config, analysis_config, raw_dir, args.device,
            ranks=ranks, run_dir_override=run_dir_override,
        )
        return

    # TSI / ASI: load one model and run a forward sweep.
    roi_size = tuple(int(x) for x in analysis_config.analysis.roi_size)
    men_loader = build_men_loader(config, roi_size=roi_size)
    n_scans = args.n_scans or int(analysis_config.analysis.n_scans_tsi)
    scan_indices = select_scan_indices(
        len(men_loader.dataset), n_scans,
        str(analysis_config.analysis.scan_selection),
        int(analysis_config.analysis.seed),
    )
    logger.info("Selected %d MEN scans", len(scan_indices))

    if args.condition == "frozen":
        model = load_frozen_model(config, device=args.device)
        rank_label, member_label, condition_csv = None, -1, "frozen"
    else:
        model = load_adapted_model(
            config, analysis_config, rank=args.rank,
            member_id=args.member_id, device=args.device,
            run_dir=run_dir_override,
        )
        rank_label, member_label, condition_csv = args.rank, args.member_id, "adapted"

    out = _run_tsi_asi(
        model, men_loader, scan_indices,
        condition=condition_csv,
        analysis_cfg=analysis_config.analysis,
        device=args.device,
    )

    thresholds = list(analysis_config.analysis.tsi_thresholds)
    if args.metric == "tsi":
        df = _tsi_results_to_df(
            out.tsi_results, rank=rank_label, member_id=member_label, thresholds=thresholds,
        )
        path = raw_dir / f"tsi_{condition_csv}_per_scan.csv"
        df.to_csv(path, index=False)
        _save_tsi_channels(out.tsi_results, raw_dir / f"tsi_{condition_csv}_channels.npz")
        logger.info("TSI written to %s (%d rows)", path, len(df))
    else:  # asi
        df = _asi_results_to_df(
            out.asi_results, condition=condition_csv,
            rank=rank_label, member_id=member_label,
        )
        path = raw_dir / f"asi_{condition_csv}_per_scan.csv"
        df.to_csv(path, index=False)
        _save_asi_window_stats(out.asi_results, raw_dir / f"asi_{condition_csv}_window_stats.npz")
        logger.info("ASI written to %s (%d rows)", path, len(df))


if __name__ == "__main__":
    main()
