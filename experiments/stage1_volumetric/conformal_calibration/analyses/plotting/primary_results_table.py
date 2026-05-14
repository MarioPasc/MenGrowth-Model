"""Primary-results table for the BSc thesis results section.

Emits a booktabs LaTeX table comparing the two models retained as the primary
showcase -- the homoscedastic LME baseline (``lme_homo``) and the ensemble
Bayesian model averaging model (``ensemble_bma``) -- both evaluated with the
native parametric (Gaussian) 95% predictive interval under ``last_from_rest``
LOPO-CV on the N=56 MenGrowth cohort.

Columns: coefficient of determination on log-volume (``R^2_{\\log V}``),
Interval Score IS@95 (Gneiting & Raftery, 2007; lower is better), empirical
95% coverage with a beta-binomial credible interval, mean interval width, and
CRPS. The caption note carries the paired BCa-bootstrap Delta IS@95 of BMA vs
the baseline (the headline model_lift comparison).

The parametric layer is seed-deterministic at tau=0, so the 20-seed mean is
the exact value and no across-seed dispersion is reported.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_propagation_volume_prediction/conformal_calibration"
)
LAYER = "parametric"
MODELS: tuple[tuple[str, str], ...] = (
    ("lme_homo", r"LME homoscedastic (baseline)"),
    ("ensemble_bma", r"Ensemble BMA"),
)


def _marginal_metrics() -> pd.DataFrame:
    """Mean marginal metrics per model for the parametric layer.

    Returns:
        Frame indexed by ``base_model`` with one column per metric, averaged
        over seeds (the parametric layer is seed-invariant, so the mean is
        exact).
    """
    df = pd.read_parquet(RESULTS_ROOT / "aggregated" / "results_table.parquet")
    sub = df[(df["scope"] == "marginal") & (df["layer"] == LAYER)]
    wide = sub.pivot_table(index="base_model", columns="metric", values="value", aggfunc="mean")
    return wide


def _delta_is_note() -> str:
    """Build the caption note with the paired-bootstrap Delta IS@95 (BMA - homo).

    Returns:
        LaTeX string, or an empty string if the model_lift family is absent.
    """
    stats_path = RESULTS_ROOT / "aggregated" / "statistics.json"
    if not stats_path.exists():
        return ""
    stats = json.loads(stats_path.read_text())
    rows = [
        r
        for r in stats.get("bootstrap", [])
        if r.get("family") == "model_lift"
        and r.get("base_model_b") == "ensemble_bma"
        and r.get("layer") == LAYER
        and r.get("scope") == "marginal"
        and r.get("metric") == "is_95"
        and r.get("seed") == 0
    ]
    if not rows:
        return ""
    r = rows[0]
    return (
        rf"Paired BCa-bootstrap $\Delta\mathrm{{IS}}@95$ (Ensemble BMA $-$ baseline) "
        rf"$= {r['delta']:.2f}$, 95\% CI $[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]$, "
        rf"$p = {r['p_value']:.3f}$ (B$=10{{,}}000$): the point estimate favours BMA but "
        rf"the interval includes $0$ at $N=56$."
    )


def build_table() -> str:
    """Assemble the LaTeX table body.

    Returns:
        Full ``table`` environment as a string.
    """
    wide = _marginal_metrics()
    note = _delta_is_note()

    lines: list[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Primary results: homoscedastic LME baseline versus the "
        r"ensemble Bayesian model averaging (BMA) model, both with the native "
        r"parametric 95\% predictive interval, under \texttt{last\_from\_rest} "
        r"LOPO-CV on the $N=56$ MenGrowth cohort. IS@95 and width are in "
        r"$\log(V_{\mathrm{MEN}}+1)$ units; coverage is reported with a 95\% "
        r"beta-binomial credible interval (nominal $0.95$). " + note + r"}",
        r"  \label{tab:conformal-primary-results}",
        r"  \begin{tabular}{lccccc}",
        r"    \toprule",
        r"    Model & $R^2_{\log V}$ & IS@95 $\downarrow$ & Coverage@95 "
        r"& Mean width & CRPS $\downarrow$ \\",
        r"    \midrule",
    ]

    for base_model, label in MODELS:
        m = wide.loc[base_model]
        cov = m["coverage_95"]
        cov_lo = m.get("coverage_95_ci_low", float("nan"))
        cov_hi = m.get("coverage_95_ci_high", float("nan"))
        lines.append(
            rf"    {label} & {m['r2_log']:.3f} & {m['is_95']:.2f} & "
            rf"{cov:.3f} \, {{\scriptsize $[{cov_lo:.2f}, {cov_hi:.2f}]$}} & "
            rf"{m['mean_width']:.2f} & {m['crps']:.3f} \\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(out_tex: Path) -> None:
    """Write the LaTeX table and echo it to the log.

    Args:
        out_tex: Destination ``.tex`` path.
    """
    table = build_table()
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(table + "\n")
    logger.info("Wrote %s", out_tex)
    logger.info("Table body:\n%s", table)


if __name__ == "__main__":
    main(out_tex=RESULTS_ROOT / "tables" / "primary_results_table.tex")
