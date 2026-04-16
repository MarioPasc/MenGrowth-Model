"""Generate LaTeX and CSV tables for domain divergence analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Interpretation thresholds for domain classifier accuracy.
_INTERP_HIGH = 0.80
_INTERP_MODERATE = 0.65


def _interpret_domain_acc(acc: float) -> str:
    """Rule-based interpretation of domain classifier accuracy."""
    if acc >= _INTERP_HIGH:
        return "High"
    elif acc >= _INTERP_MODERATE:
        return "Moderate"
    else:
        return "Low"


def generate_domain_metrics_table(
    metrics_csv: Path,
    out_dir: Path,
) -> None:
    """Generate domain metrics summary table in CSV and LaTeX.

    Columns: Stage, C_s, DomainAcc, MLP Acc, MMD (p), PAD, FSD, Interpretation.

    Args:
        metrics_csv: Path to ``domain_metrics.csv``.
        out_dir: Output directory for table files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Stage": int(r["stage"]),
            "C_s": int(r["n_channels"]),
            "N_GLI": int(r["n_gli"]),
            "N_MEN": int(r["n_men"]),
            "DomainAcc": f"{r['domain_acc_linear']:.3f}",
            "DomainAcc_CI": f"[{r['domain_acc_ci_lower']:.3f}, {r['domain_acc_ci_upper']:.3f}]",
            "MLP_Acc": f"{r['domain_acc_mlp']:.3f}",
            "MMD": f"{r['mmd']:.4f}",
            "MMD_p": f"{r['mmd_p']:.3f}",
            "PAD": f"{r['pad']:.3f}",
            "FSD": f"{r['fsd']:.4f}",
            "Interpretation": _interpret_domain_acc(r["domain_acc_linear"]),
        })
    table_df = pd.DataFrame(rows)

    # CSV
    csv_path = out_dir / "domain_metrics_table.csv"
    table_df.to_csv(csv_path, index=False)
    logger.info("Domain metrics CSV table saved to %s", csv_path)

    # LaTeX
    tex_path = out_dir / "domain_metrics_table.tex"
    _write_domain_metrics_latex(table_df, tex_path)
    logger.info("Domain metrics LaTeX table saved to %s", tex_path)


def _write_domain_metrics_latex(df: pd.DataFrame, tex_path: Path) -> None:
    """Write LaTeX tabular for the domain metrics summary."""
    header = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{Per-stage domain divergence metrics (frozen BrainSegFounder, GLI vs MEN).}" "\n"
        r"\label{tab:domain_divergence}" "\n"
        r"\begin{tabular}{c c c c c c c c l}" "\n"
        r"\toprule" "\n"
        r"Stage & $C_s$ & DomainAcc & 95\% CI & MLP Acc & MMD ($p$) & PAD & FSD & Interp. \\" "\n"
        r"\midrule" "\n"
    )
    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    lines = [header]
    for _, r in df.iterrows():
        line = (
            f"{r['Stage']} & {r['C_s']} & {r['DomainAcc']} & {r['DomainAcc_CI']} & "
            f"{r['MLP_Acc']} & {r['MMD']} ({r['MMD_p']}) & {r['PAD']} & {r['FSD']} & "
            f"{r['Interpretation']} \\\\\n"
        )
        lines.append(line)
    lines.append(footer)
    tex_path.write_text("".join(lines))


def generate_drift_table(
    drift_csv: Path,
    correlation_json: Path | None,
    out_dir: Path,
) -> None:
    """Generate CKA drift summary table in CSV and LaTeX.

    Columns: Config, Stage 0..4, Spearman rho.

    Args:
        drift_csv: Path to ``cka_drift.csv``.
        correlation_json: Optional path to ``drift_divergence_correlation.json``.
        out_dir: Output directory for table files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(drift_csv)

    correlations: dict = {}
    if correlation_json is not None and correlation_json.exists():
        with open(correlation_json) as f:
            corr_data = json.load(f)
        correlations = corr_data.get("correlations", {})

    # Detect stage columns
    stage_cols = [c for c in df.columns if c.startswith("cka_stage_")]
    stage_indices = [int(c.split("_")[-1]) for c in stage_cols]

    rows = []
    for _, r in df.iterrows():
        config_name = str(r["config_name"])
        row: dict = {"Config": config_name}
        for s, col in zip(stage_indices, stage_cols):
            row[f"Stage {s}"] = f"{float(r[col]):.4f}"
        # Spearman rho if available
        if config_name in correlations:
            rho = correlations[config_name]["spearman_rho"]
            p = correlations[config_name]["spearman_p"]
            row["Spearman_rho"] = f"{rho:.3f} (p={p:.3f})"
        else:
            row["Spearman_rho"] = "—"
        rows.append(row)

    table_df = pd.DataFrame(rows)

    # CSV
    csv_path = out_dir / "cka_drift_table.csv"
    table_df.to_csv(csv_path, index=False)
    logger.info("CKA drift CSV table saved to %s", csv_path)

    # LaTeX
    tex_path = out_dir / "cka_drift_table.tex"
    _write_drift_latex(table_df, stage_indices, tex_path)
    logger.info("CKA drift LaTeX table saved to %s", tex_path)


def _write_drift_latex(df: pd.DataFrame, stages: list[int], tex_path: Path) -> None:
    """Write LaTeX tabular for the CKA drift summary."""
    n_cols = len(stages) + 2  # Config + stages + Spearman
    col_spec = "l " + "c " * len(stages) + "c"
    header = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{CKA adaptation drift per stage (1.0 = no drift, lower = more adaptation).}" "\n"
        r"\label{tab:cka_drift}" "\n"
        rf"\begin{{tabular}}{{{col_spec}}}" "\n"
        r"\toprule" "\n"
    )
    col_headers = "Config & " + " & ".join(f"Stage {s}" for s in stages) + r" & Spearman $\rho$ \\"
    header += col_headers + "\n" + r"\midrule" + "\n"

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )

    lines = [header]
    for _, r in df.iterrows():
        vals = [str(r["Config"])]
        for s in stages:
            vals.append(str(r[f"Stage {s}"]))
        vals.append(str(r["Spearman_rho"]))
        lines.append(" & ".join(vals) + " \\\\\n")
    lines.append(footer)
    tex_path.write_text("".join(lines))
