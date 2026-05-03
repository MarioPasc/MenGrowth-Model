# experiments/stage1_volumetric/analysis/summary.py
"""Terminal summary tables for Stage 1 UQ growth prediction results."""

from __future__ import annotations

from growth.shared.lopo import LOPOResults


def print_summary_table(
    model_names: list[str],
    lopo_results: dict[str, LOPOResults],
    calib_metrics: dict[str, dict],
) -> None:
    """Print formatted results table to stdout.

    Args:
        model_names: Ordered list of model names (preserves config order).
        lopo_results: Dict mapping model name to LOPOResults.
        calib_metrics: Dict mapping model name to calibration metrics.
    """
    print("\n" + "=" * 110)
    print("STAGE 1: UNCERTAINTY-PROPAGATED VOLUME PREDICTION — RESULTS")
    print("=" * 110)

    header = (
        f"  {'Model':<22} {'R2_log':>8} {'MAE_log':>8} {'CRPS':>8} "
        f"{'DSS':>8} {'NLPD':>8} "
        f"{'Cov_50':>7} {'Cov_80':>7} {'Cov_90':>7} {'Cov_95':>7} "
        f"{'IS_95':>8} {'CI_w':>8}"
    )
    print(f"\n{header}")
    print("  " + "-" * 112)

    for model_name in model_names:
        if model_name not in lopo_results:
            continue
        m = lopo_results[model_name].aggregate_metrics
        cm = calib_metrics.get(model_name, {})

        print(
            f"  {model_name:<22} "
            f"{m.get('last_from_rest/r2_log', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/mae_log', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/crps', float('nan')):>8.4f} "
            f"{cm.get('dss', float('nan')):>8.4f} "
            f"{cm.get('nlpd', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/coverage_50', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/coverage_80', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/coverage_90', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/coverage_95', float('nan')):>7.3f} "
            f"{m.get('last_from_rest/is_95', float('nan')):>8.4f} "
            f"{m.get('last_from_rest/mean_ci_width_log', float('nan')):>8.4f}"
        )


def print_comparison_tables(
    homo_hetero: list[dict],
    classical_propagated: list[dict],
    analytical_homo: list[dict],
) -> None:
    """Print paired comparison tables to stdout.

    Args:
        homo_hetero: Homo vs hetero comparisons.
        classical_propagated: NLME vs hetero comparisons.
        analytical_homo: NLME vs homo comparisons.
    """
    sections = [
        ("Homo vs Hetero Paired Comparisons", homo_hetero),
        ("Classical (NLME) vs Propagated (Hetero)", classical_propagated),
        ("Analytical (NLME) vs Homoscedastic", analytical_homo),
    ]

    for title, comparisons in sections:
        if not comparisons:
            continue
        print(f"\n--- {title} ---")
        print(f"  {'Pair':<40} {'dR2':>8} {'dCRPS':>8} {'dCov95':>8} {'p':>8}")
        print("  " + "-" * 74)
        for c in comparisons:
            pair_str = f"{c['pair'][0]} -> {c['pair'][1]}"
            print(
                f"  {pair_str:<40} "
                f"{c['delta_r2']:>+8.4f} "
                f"{c['delta_crps']:>+8.4f} "
                f"{c['delta_coverage_95']:>+8.3f} "
                f"{c['p_value_errors']:>8.4f}"
            )
