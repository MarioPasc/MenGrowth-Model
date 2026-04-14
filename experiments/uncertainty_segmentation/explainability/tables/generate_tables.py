"""Build the spec §8 tables (TSI+ASI combined, DAD with permutation p)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _wilcoxon_one_sided_greater(values: np.ndarray, mu0: float = 1.0) -> float:
    """One-sided Wilcoxon ``H1: median > mu0``; ``NaN`` if not enough data."""
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return float("nan")
    diffs = finite - mu0
    diffs = diffs[diffs != 0]
    if len(diffs) < 3:
        return float("nan")
    try:
        _, p = scipy.stats.wilcoxon(diffs, alternative="greater")
        return float(p)
    except ValueError:
        return float("nan")


def _aggregate_tsi(per_scan_df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, stage): mean ± SD TSI across scans."""
    rows = []
    for (cond, stage), grp in per_scan_df.groupby(["condition", "stage"]):
        vals = grp["mean_tsi"].dropna().to_numpy()
        rows.append({
            "condition": cond,
            "stage": int(stage),
            "n_scans": int(len(vals)),
            "mean_tsi": float(np.nanmean(vals)) if len(vals) else float("nan"),
            "std_tsi": float(np.nanstd(vals)) if len(vals) else float("nan"),
            "frac_tsi_gt_1.5": float(grp["frac_1.5"].mean()) if "frac_1.5" in grp else float("nan"),
            "wilcoxon_p_tsi": _wilcoxon_one_sided_greater(vals, mu0=1.0),
        })
    return pd.DataFrame(rows).sort_values(["condition", "stage"]).reset_index(drop=True)


def _aggregate_asi(per_scan_df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, stage): mean ± SD ASI across scans×blocks×heads."""
    if per_scan_df is None or per_scan_df.empty:
        return pd.DataFrame(
            columns=["condition", "stage", "n_obs", "mean_asi", "std_asi",
                     "frac_asi_gt_1.5", "wilcoxon_p_asi"]
        )
    rows = []
    for (cond, stage), grp in per_scan_df.groupby(["condition", "stage"]):
        vals = grp["asi_value"].dropna().to_numpy()
        rows.append({
            "condition": cond,
            "stage": int(stage),
            "n_obs": int(len(vals)),
            "mean_asi": float(np.nanmean(vals)) if len(vals) else float("nan"),
            "std_asi": float(np.nanstd(vals)) if len(vals) else float("nan"),
            "frac_asi_gt_1.5": float(np.mean(vals > 1.5)) if len(vals) else float("nan"),
            "wilcoxon_p_asi": _wilcoxon_one_sided_greater(vals, mu0=1.0),
        })
    return pd.DataFrame(rows).sort_values(["condition", "stage"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combined TSI+ASI table
# ---------------------------------------------------------------------------


def build_tsi_asi_table(
    tsi_per_scan: pd.DataFrame,
    asi_per_scan: pd.DataFrame | None,
) -> pd.DataFrame:
    """Spec §8.1: per-stage TSI + ASI combined per condition."""
    tsi_agg = _aggregate_tsi(tsi_per_scan)
    asi_agg = _aggregate_asi(asi_per_scan) if asi_per_scan is not None else pd.DataFrame()
    if asi_agg.empty:
        return tsi_agg
    return tsi_agg.merge(
        asi_agg, on=["condition", "stage"], how="outer"
    ).sort_values(["condition", "stage"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# DAD table
# ---------------------------------------------------------------------------


def build_dad_table(dad_per_head: pd.DataFrame) -> pd.DataFrame:
    """Spec §8.2: per-stage DAD with permutation significance."""
    rows = []
    for (cond, stage, block), grp in dad_per_head.groupby(["condition", "stage", "block"]):
        rows.append({
            "condition": cond,
            "stage": int(stage),
            "block": int(block),
            "n_heads": int(len(grp)),
            "mean_dad": float(grp["dad"].mean()),
            "std_dad": float(grp["dad"].std()),
            "min_p_value": float(grp["p_value"].min()),
            "n_significant_heads": int((grp["p_value"] < 0.05).sum()),
        })
    return pd.DataFrame(rows).sort_values(["condition", "stage", "block"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return r"$<\!10^{-3}$"
    return f"{p:.3f}"


def _format_mean_sd(m: float, s: float) -> str:
    if not np.isfinite(m):
        return "—"
    return f"{m:.2f} $\\pm$ {s:.2f}"


def write_tsi_asi_latex(df: pd.DataFrame, out_path: Path) -> None:
    """Write the TSI+ASI combined table as a LaTeX tabular."""
    cols = [
        "stage", "condition", "mean_tsi", "std_tsi", "frac_tsi_gt_1.5",
        "wilcoxon_p_tsi", "mean_asi", "std_asi", "frac_asi_gt_1.5",
        "wilcoxon_p_asi",
    ]
    df = df.reindex(columns=cols)

    lines = [
        r"\begin{tabular}{cccccc}",
        r"\hline",
        r"Stage & Condition & TSI (mean $\pm$ SD) & "
        r"Frac(TSI$>$1.5) & ASI (mean $\pm$ SD) & Wilcoxon $p$ \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        tsi = _format_mean_sd(r["mean_tsi"], r["std_tsi"])
        asi = _format_mean_sd(r.get("mean_asi", np.nan), r.get("std_asi", np.nan))
        frac_t = (
            "—" if not np.isfinite(r["frac_tsi_gt_1.5"]) else f"{r['frac_tsi_gt_1.5']:.2f}"
        )
        p_a = _format_p(r.get("wilcoxon_p_asi", np.nan))
        lines.append(
            f"{int(r['stage'])} & {r['condition']} & {tsi} & {frac_t} & {asi} & {p_a} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines))


def write_dad_latex(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{ccccccc}",
        r"\hline",
        r"Stage & Block & Condition & \#Heads & DAD (mean $\pm$ SD) & "
        r"min $p$ & Sig. heads \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{int(r['stage'])} & {int(r['block'])} & {r['condition']} & "
            f"{int(r['n_heads'])} & "
            f"{_format_mean_sd(r['mean_dad'], r['std_dad'])} & "
            f"{_format_p(r['min_p_value'])} & {int(r['n_significant_heads'])} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def generate_tables(
    raw_dir: Path,
    out_dir: Path,
) -> dict[str, Path]:
    """Build all spec §8 tables from the per-scan CSVs in ``raw_dir``.

    Reads
    -----
    - ``raw/tsi_frozen_per_scan.csv``, ``raw/tsi_adapted_per_scan.csv``
    - ``raw/asi_frozen_per_scan.csv`` (optional), ``raw/asi_adapted_per_scan.csv`` (optional)
    - ``raw/dad_per_head.csv``       (optional)

    Writes
    ------
    ``out_dir/{tsi_asi_table.csv,tsi_asi_table.tex,dad_table.csv,dad_table.tex}``
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    tsi_dfs: list[pd.DataFrame] = []
    for name in ("tsi_frozen_per_scan.csv", "tsi_adapted_per_scan.csv"):
        p = raw_dir / name
        if p.exists():
            tsi_dfs.append(pd.read_csv(p))
    asi_dfs: list[pd.DataFrame] = []
    for name in ("asi_frozen_per_scan.csv", "asi_adapted_per_scan.csv"):
        p = raw_dir / name
        if p.exists():
            asi_dfs.append(pd.read_csv(p))
    dad_path = raw_dir / "dad_per_head.csv"

    if tsi_dfs:
        tsi_concat = pd.concat(tsi_dfs, ignore_index=True)
        asi_concat = pd.concat(asi_dfs, ignore_index=True) if asi_dfs else None
        tsi_asi = build_tsi_asi_table(tsi_concat, asi_concat)
        csv_path = out_dir / "tsi_asi_table.csv"
        tex_path = out_dir / "tsi_asi_table.tex"
        tsi_asi.to_csv(csv_path, index=False)
        write_tsi_asi_latex(tsi_asi, tex_path)
        written["tsi_asi_csv"] = csv_path
        written["tsi_asi_tex"] = tex_path
        logger.info("Wrote %s and %s", csv_path.name, tex_path.name)
    else:
        logger.warning("No TSI per-scan CSVs found in %s", raw_dir)

    if dad_path.exists():
        dad_df = pd.read_csv(dad_path)
        dad_table = build_dad_table(dad_df)
        csv_path = out_dir / "dad_table.csv"
        tex_path = out_dir / "dad_table.tex"
        dad_table.to_csv(csv_path, index=False)
        write_dad_latex(dad_table, tex_path)
        written["dad_csv"] = csv_path
        written["dad_tex"] = tex_path
        logger.info("Wrote %s and %s", csv_path.name, tex_path.name)
    else:
        logger.info("No DAD CSV at %s — skipping DAD table", dad_path)

    return written
