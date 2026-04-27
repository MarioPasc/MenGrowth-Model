"""Top-level orchestrator for inter-LoRA comparison report generation."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.uncertainty_segmentation.plotting.inter_lora.compile import (
    build_compiled_metrics,
    validate_compiled_metrics,
)
from experiments.uncertainty_segmentation.plotting.inter_lora.figures import (
    FIGURE_OUTPUT_NAMES,
    INTER_LORA_FIGURE_REGISTRY,
    INTER_LORA_TABLE_REGISTRY,
)
from experiments.uncertainty_segmentation.plotting.inter_lora.io_layer import (
    InterLoraData,
    discover_ranks,
    load_rank_run,
    select_subjects_auto,
    validate_baseline_consistency,
)
from experiments.uncertainty_segmentation.plotting.inter_lora.style import (
    save_figure,
    setup_inter_lora_style,
)

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _setup_logging(log_dir: Path) -> None:
    """Configure file logging to log_dir/plotting.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "plotting.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )
    logging.getLogger("experiments.uncertainty_segmentation.plotting.inter_lora").addHandler(fh)


def _build_report_summary(data: InterLoraData, out_root: Path) -> None:
    """Emit report_summary.md with detected r* and narrative."""
    cm = data.compiled_metrics
    lines = ["# Inter-LoRA Report Summary\n"]
    lines.append(f"- **Generated**: {data.run_timestamp}")
    lines.append(f"- **Git SHA**: {data.git_sha}")
    lines.append(f"- **Ranks**: {data.rank_values}\n")

    for label in ["TC", "WT", "ET"]:
        subset = cm[(cm["label"] == label) & (cm["rank"] > 0)]
        if subset.empty:
            continue
        best_row = subset.loc[subset["dice_mean"].idxmax()]
        lines.append(
            f"- **{label}**: best ensemble Dice = {best_row['dice_mean']:.3f} "
            f"at r={int(best_row['rank'])}",
        )

    non_bl = cm[(cm["rank"] > 0) & (cm["label"] != "mean")]
    if not non_bl.empty:
        ece_best = non_bl.loc[non_bl["ece"].idxmin()]
        lines.append(f"\n- **Lowest ECE**: {ece_best['ece']:.2e} at r={int(ece_best['rank'])}")

        if "pct_bias_dominated" in non_bl.columns:
            bias_best = non_bl.loc[non_bl["pct_bias_dominated"].idxmin()]
            lines.append(
                f"- **Lowest bias-dominated fraction**: "
                f"{bias_best['pct_bias_dominated']:.1%} at r={int(bias_best['rank'])}",
            )

    summary_path = out_root / "report_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Report summary written to %s", summary_path)


def generate_inter_lora_report(
    root_dir: str | Path,
    out_root: str | Path | None = None,
    config_path: str | Path | None = None,
    ranks: list[int] | None = None,
    baseline_dir: str | Path | None = None,
    select_subjects: str | list[str] = "auto",
    bootstrap_n: int = 1000,
    seed: int = 42,
    dpi: int = 600,
    fmt: str = "pdf",
    skip: set[str] | None = None,
    strict: bool = False,
    dry_run: bool = False,
    force_recompute: bool = False,
) -> None:
    """Generate the complete inter-LoRA comparison report.

    Args:
        root_dir: Root directory containing r*_M*_s* subdirectories.
        out_root: Output directory. Defaults to {root_dir}/_inter_lora_report/.
        config_path: Optional config YAML path.
        ranks: If provided, filter to only these ranks.
        baseline_dir: Explicit baseline directory override.
        select_subjects: "auto" or list of scan IDs for Qual1.
        bootstrap_n: Number of bootstrap resamples.
        seed: Random seed.
        dpi: Output DPI for PNG.
        fmt: Primary output format (pdf, png, svg).
        skip: Set of items to skip: {"quant1", "quant2", "qual1", "qual2", "tables"}.
        strict: If True, fail on missing artefacts.
        dry_run: If True, discover and validate only, no rendering.
        force_recompute: If True, rebuild compiled_metrics even if cached.
    """
    root_dir = Path(root_dir)
    out_root = Path(out_root) if out_root else root_dir / "_inter_lora_report"
    skip = skip or set()

    # Load config — fall back to the default config.yaml bundled with the
    # plotting package when no explicit path is given.
    config: dict = {}
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    import yaml

    cp = Path(config_path)
    if cp.exists():
        with open(cp) as f:
            full_cfg = yaml.safe_load(f) or {}
        config = full_cfg.get("inter_lora", {})

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(exist_ok=True)
    (out_root / "tables").mkdir(exist_ok=True)
    (out_root / "data").mkdir(exist_ok=True)

    _setup_logging(out_root / "logs")

    t_start = time.time()
    logger.info("=== Inter-LoRA Report Generation ===")
    logger.info("Root: %s", root_dir)
    logger.info("Output: %s", out_root)

    # 1. Discover ranks
    expected = frozenset(ranks) if ranks else None
    rank_dirs = discover_ranks(root_dir, expected=expected)
    logger.info("Discovered %d rank directories", len(rank_dirs))

    # 2. Load all RankRun objects
    rank_runs = [load_rank_run(d, strict=strict) for d in rank_dirs]

    # 3. Validate baseline consistency
    validate_baseline_consistency(rank_runs)

    if dry_run:
        logger.info("Dry run complete — %d ranks validated", len(rank_runs))
        return

    # 4. Build compiled_metrics
    csv_path = out_root / "data" / "compiled_metrics.csv"
    if csv_path.exists() and not force_recompute:
        import pandas as pd

        compiled = pd.read_csv(csv_path)
        logger.info("Loaded cached compiled_metrics (%d rows)", len(compiled))
    else:
        compiled = build_compiled_metrics(rank_runs, n_boot=bootstrap_n, seed=seed)
        validate_compiled_metrics(compiled, n_ranks=len(rank_runs))
        compiled.to_csv(csv_path, index=False)
        logger.info("Compiled metrics saved to %s", csv_path)

    # 5. Select subjects for Qual1
    selected_path = out_root / "data" / "selected_slices.json"
    if isinstance(select_subjects, str) and select_subjects == "auto":
        selected = select_subjects_auto(tuple(rank_runs))
    else:
        selected = {
            "brats_men": {"scan_id": select_subjects[0], "slice_idx": None},
        }
        if len(select_subjects) > 1:
            selected["mengrowth"] = {"scan_id": select_subjects[1], "slice_idx": None}

    with open(selected_path, "w") as f:
        json.dump(selected, f, indent=2)
    logger.info("Selected slices persisted to %s", selected_path)

    # 6. Construct InterLoraData
    inter_data = InterLoraData(
        root_dir=root_dir,
        out_root=out_root,
        ranks=tuple(rank_runs),
        compiled_metrics=compiled,
        git_sha=_git_sha(),
        run_timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        selected_slices=selected,
    )

    # 7. Setup style
    setup_inter_lora_style()

    # 8. Generate figures
    n_ok = 0
    n_skip = 0
    n_fail = 0

    skip_map = {
        "quant1": "quant1_dice_vs_rank",
        "quant2": "quant2_calib_epistemic",
        "qual1": "qual1_slice_grid",
        "qual2": "qual2_clustered_heatmap",
    }
    skip_names = {skip_map.get(s, s) for s in skip}

    from experiments.uncertainty_segmentation.plotting.inter_lora.figures.qual1_slice_grid import (
        ALL_VARIANTS as QUAL1_VARIANTS,
    )

    fig_dir = out_root / "figures"

    for name, module in INTER_LORA_FIGURE_REGISTRY.items():
        if name in skip_names:
            logger.info("[SKIP] %s (in --skip list)", name)
            n_skip += 1
            continue

        # qual1 generates 8 variants (4 vertical + 4 horizontal)
        if name == "qual1_slice_grid":
            for variant in QUAL1_VARIANTS:
                variant_name = f"qual1_{variant}"
                t0 = time.time()
                fig_config = {
                    "variant": variant,
                    "brats_h5": config.get("brats_h5"),
                    "mengrowth_h5": config.get("mengrowth_h5"),
                    "force_recompute": force_recompute,
                }
                try:
                    fig = module.plot(inter_data, fig_config)
                except Exception:
                    logger.exception("[FAIL] %s", variant_name)
                    n_fail += 1
                    continue
                if fig is None:
                    logger.info("[SKIP] %s (data not available)", variant_name)
                    n_skip += 1
                    continue
                save_figure(
                    fig,
                    fig_dir / f"{variant_name}.{fmt}",
                    title=variant_name,
                    description=f"Inter-LoRA qual1 {variant}",
                    dpi=dpi,
                )
                if fmt != "png":
                    save_figure(
                        fig,
                        fig_dir / f"{variant_name}.png",
                        title=variant_name,
                        description=f"Inter-LoRA qual1 {variant}",
                        dpi=dpi,
                    )
                plt.close(fig)
                elapsed = time.time() - t0
                logger.info("[OK]   %s (%.1fs)", variant_name, elapsed)
                n_ok += 1
            continue

        t0 = time.time()
        try:
            fig = module.plot(inter_data, {})
        except Exception:
            logger.exception("[FAIL] %s", name)
            n_fail += 1
            continue

        if fig is None:
            logger.info("[SKIP] %s (data not available)", name)
            n_skip += 1
            continue

        out_name = FIGURE_OUTPUT_NAMES.get(name, name)

        save_figure(
            fig,
            fig_dir / f"{out_name}.{fmt}",
            title=out_name,
            description=f"Inter-LoRA {name}",
            dpi=dpi,
        )
        if fmt != "png":
            save_figure(
                fig,
                fig_dir / f"{out_name}.png",
                title=out_name,
                description=f"Inter-LoRA {name}",
                dpi=dpi,
            )

        plt.close(fig)
        elapsed = time.time() - t0
        logger.info("[OK]   %s (%.1fs)", name, elapsed)
        n_ok += 1

    # 9. Generate tables
    if "tables" not in skip:
        for name, module in INTER_LORA_TABLE_REGISTRY.items():
            try:
                module.render(inter_data, {}, out_root / "tables")
                logger.info("[OK]   %s", name)
            except Exception:
                logger.exception("[FAIL] %s", name)
                n_fail += 1
    else:
        logger.info("[SKIP] tables (in --skip list)")

    # 10. Report summary
    _build_report_summary(inter_data, out_root)

    total_time = time.time() - t_start
    logger.info(
        "Done: %d figures, %d skipped, %d failed (%.1fs total)",
        n_ok,
        n_skip,
        n_fail,
        total_time,
    )
    logger.info("Output at %s/", out_root)
