"""CLI entry point for the BraTS-MEN benchmark analysis pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .aggregate import aggregate_pairwise, model_order_from_df, select_best_median_worst
from .compute import compute_all
from .io import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_GT_ROOT,
    DEFAULT_MODELS_ROOT,
    DEFAULT_OUR_RUN,
    discover_models,
)
from .metrics import LABELS
from .plot_qualitative import write_qualitative
from .plot_quantitative import write_quantitative
from .plot_size_sensitivity import write_size_sensitivity

logger = logging.getLogger("benchmark.analysis")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS-MEN benchmark analysis pipeline")
    p.add_argument(
        "stage",
        choices=("compute", "aggregate", "plot", "all"),
        nargs="?",
        default="all",
    )
    p.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    p.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    p.add_argument("--our-run", type=Path, default=DEFAULT_OUR_RUN)
    p.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--heatmap-label", choices=LABELS, default="TC")
    p.add_argument(
        "--zoom-dice",
        action="store_true",
        help="Add an inset zoom on the largest GT/pred disagreement in each cell of the dice qualitative figure.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose >= 2 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args.analysis_root.mkdir(parents=True, exist_ok=True)

    df = None
    pairwise = None
    bmw = None

    if args.stage in ("compute", "all"):
        df = compute_all(
            analysis_root=args.analysis_root,
            models_root=args.models_root,
            our_run=args.our_run,
            gt_root=args.gt_root,
            force=args.force_recompute,
            workers=args.workers,
        )

    if args.stage in ("aggregate", "plot", "all"):
        if df is None:
            from .compute import _load_existing_metrics
            df = _load_existing_metrics(args.analysis_root)
            if df.empty:
                logger.error("No cached metrics; run 'compute' first.")
                return 2
        model_order = model_order_from_df(df)
        pairwise = aggregate_pairwise(df, model_order, analysis_root=args.analysis_root)
        bmw = select_best_median_worst(df, model_order, analysis_root=args.analysis_root, rank_label="TC")

    if args.stage in ("plot", "all"):
        assert df is not None and pairwise is not None and bmw is not None
        model_order = model_order_from_df(df)
        write_quantitative(
            df,
            pairwise,
            model_order,
            analysis_root=args.analysis_root,
            heatmap_label=args.heatmap_label,
            seed=args.seed,
        )
        # Discover entries again so we have schema/pred_dir for image loading.
        entries = discover_models(models_root=args.models_root, our_run=args.our_run)
        # Re-order entries to match `model_order`.
        by_id = {e.model_id: e for e in entries}
        ordered = [by_id[m] for m in model_order if m in by_id]
        write_qualitative(
            ordered,
            df,
            bmw,
            analysis_root=args.analysis_root,
            gt_root=args.gt_root,
            zoom_dice=args.zoom_dice,
        )
        write_size_sensitivity(
            df,
            model_order,
            analysis_root=args.analysis_root,
            entries=ordered,
            gt_root=args.gt_root,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
