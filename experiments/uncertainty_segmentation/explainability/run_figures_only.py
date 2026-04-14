"""Re-render figures + tables from cached ``raw/`` artefacts (no GPU).

Use this script after a successful ``run_analysis.py`` to iterate on
figure formatting on a CPU-only laptop.  It reads:

    {data_dir}/tsi_{frozen,adapted}_per_scan.csv
    {data_dir}/asi_{frozen,adapted}_per_scan.csv
    {data_dir}/dad_per_head.csv

and re-emits the spec §7 PDFs and the spec §8 LaTeX tables.

The ``--config`` argument supplies the parent config so the LoRA
target stages can be highlighted; if absent we fall back to
``[3, 4]`` (the historical default).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from .figures.fig_asi_panel import render_asi_panel
from .figures.fig_dad_bar import render_dad_bar
from .figures.fig_summary import render_summary
from .figures.fig_tsi_panel import render_tsi_panel
from .tables.generate_tables import generate_tables

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


def _resolve_lora_stages(config_path: Path | None, raw_dir: Path) -> set[int]:
    """Return the LoRA target stages from a config file or the cached snapshot."""
    if config_path is not None and config_path.exists():
        cfg = OmegaConf.load(config_path)
        if "lora" in cfg and "target_stages" in cfg.lora:
            return {int(s) for s in cfg.lora.target_stages}
    snapshot = raw_dir.parent / "config_snapshot.yaml"
    if snapshot.exists():
        cfg = OmegaConf.load(snapshot)
        if "lora" in cfg and "target_stages" in cfg.lora:
            return {int(s) for s in cfg.lora.target_stages}
    logger.warning("No LoRA target_stages found; falling back to [3, 4]")
    return {3, 4}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate explainability figures and tables without GPU",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing the cached raw/ CSVs",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: ../figures relative to data-dir)",
    )
    parser.add_argument(
        "--tables-dir", default=None,
        help="Tables output directory (default: ../tables relative to data-dir)",
    )
    parser.add_argument("--config", default=None, help="Parent config for LoRA stages")
    parser.add_argument("--format", default="pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    _setup_logging()

    raw_dir = Path(args.data_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {raw_dir}")
    figures_dir = Path(args.output) if args.output else raw_dir.parent / "figures"
    tables_dir = Path(args.tables_dir) if args.tables_dir else raw_dir.parent / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else None
    lora_stages = _resolve_lora_stages(config_path, raw_dir)
    fmt = args.format
    dpi = args.dpi

    logger.info("Regenerating from %s (LoRA stages=%s)", raw_dir, sorted(lora_stages))
    written = generate_tables(raw_dir=raw_dir, out_dir=tables_dir)
    logger.info("Tables: %s", list(written.keys()))

    for cond in ("frozen", "adapted"):
        csv = raw_dir / f"tsi_{cond}_per_scan.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            render_tsi_panel(
                df, figures_dir / f"tsi_{cond}_brainmasked.{fmt}",
                title=f"Brain-masked TSI — {cond}",
                lora_stages=lora_stages, dpi=dpi,
            )
        csv = raw_dir / f"asi_{cond}_per_scan.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            if not df.empty:
                render_asi_panel(
                    df, figures_dir / f"asi_{cond}.{fmt}",
                    title=f"ASI per stage — {cond}", dpi=dpi,
                )

    dad_csv = raw_dir / "dad_per_head.csv"
    if dad_csv.exists():
        dad_df = pd.read_csv(dad_csv)
        for cond in dad_df["condition"].unique():
            render_dad_bar(
                dad_df, figures_dir / f"dad_{cond}.{fmt}",
                title=f"DAD — {cond} (MEN vs GLI)",
                condition=cond, dpi=dpi,
            )

    # Combined summary
    tsi_frozen = raw_dir / "tsi_frozen_per_scan.csv"
    if tsi_frozen.exists():
        tsi_df = pd.read_csv(tsi_frozen)
        if (raw_dir / "tsi_adapted_per_scan.csv").exists():
            tsi_df = pd.concat(
                [tsi_df, pd.read_csv(raw_dir / "tsi_adapted_per_scan.csv")],
                ignore_index=True,
            )
        asi_df = None
        if (raw_dir / "asi_frozen_per_scan.csv").exists():
            asi_df = pd.read_csv(raw_dir / "asi_frozen_per_scan.csv")
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
            render_summary(
                tsi_df=tsi_df, asi_df=asi_df, dad_df=dad_df,
                lora_stages=lora_stages,
                out_path=figures_dir / f"summary_combined_{cond}.{fmt}",
                condition=cond, dpi=dpi,
            )

    logger.info("Done. Figures in %s", figures_dir)


if __name__ == "__main__":
    main()
