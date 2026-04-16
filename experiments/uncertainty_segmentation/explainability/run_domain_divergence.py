"""Orchestrator for domain divergence analysis pipeline.

Quantifies per-stage representation divergence between glioma (GLI) and
meningioma (MEN) in the frozen BrainSegFounder encoder, and measures CKA
adaptation drift across trained LoRA configurations.

Execution phases
----------------
1. **extract** (GPU)
   Load frozen model → forward-pass GLI and MEN scans → GAP per stage
   → ``raw/gap_features_{men,gli}.npz``.

2. **metrics** (CPU)
   Load NPZ features → domain classifier accuracy (linear + MLP),
   MMD with permutation test, PAD, FSD, bootstrap CIs per stage
   → ``raw/domain_metrics.csv``.

3. **cka_cross_stage** (CPU)
   CKA matrix across stages for MEN domain
   → ``raw/cka_cross_stage_men.npy``.

4. **cka_drift** (GPU)
   For each adapted config, extract MEN features and compute per-stage
   CKA(frozen, adapted) → ``raw/cka_drift.csv``.

5. **decoder_patch** (GPU, optional)
   Patch MEN hidden states with GLI's, measure decoder sensitivity
   → ``raw/decoder_patching.csv``.

6. **correlate** (CPU)
   Spearman rank correlation between domain divergence and CKA drift
   → ``raw/drift_divergence_correlation.json``.

7. **figures** (CPU)
   Generate all PDF figures and LaTeX tables.

CLI
---
``--config``           Parent uncertainty_segmentation config.
``--analysis-config``  Domain divergence config.
``--output``           Destination directory.
``--device``           Compute device (default ``cuda:0``).
``--phase``            ``all`` or a specific phase name for partial reruns.
``--smoke-test``       Use N=3 scans at 64³ for quick validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from .engine.data_loader import (
    build_gli_loader,
    build_men_loader,
    select_scan_indices,
)
from .engine.domain_divergence import (
    StageDomainMetrics,
    apply_bh_correction,
    compute_cka_adaptation_drift,
    compute_cka_cross_stage,
    compute_decoder_patching_sensitivity,
    compute_drift_divergence_correlation,
    compute_per_stage_domain_metrics,
    extract_gap_features_per_stage,
)
from .engine.model_loader import load_adapted_model, load_frozen_model
from .engine.tsi import extract_hidden_states

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup (same pattern as run_analysis.py)
# ---------------------------------------------------------------------------


def _setup_logging(output_dir: Path) -> None:
    """Configure root logger to write to console and ``domain_divergence.log``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "domain_divergence.log"
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
# Phase implementations
# ---------------------------------------------------------------------------


def phase_extract(
    config: DictConfig,
    dd_cfg: DictConfig,
    raw_dir: Path,
    device: str,
) -> None:
    """Phase 1: extract GAP features for MEN and GLI domains."""
    logger.info("=" * 60)
    logger.info("PHASE: EXTRACT")
    logger.info("=" * 60)

    roi_size = tuple(int(x) for x in dd_cfg.roi_size)
    n_scans = int(dd_cfg.n_scans_per_domain)
    selection = str(dd_cfg.scan_selection)
    seed = int(dd_cfg.seed)
    stages = tuple(int(s) for s in dd_cfg.stages)

    model = load_frozen_model(config, device=device)

    for domain, builder in [("men", build_men_loader), ("gli", build_gli_loader)]:
        logger.info("Extracting %s features (N=%d, ROI=%s)", domain.upper(), n_scans, roi_size)
        loaded = builder(config, roi_size=roi_size)
        indices = select_scan_indices(len(loaded.dataset), n_scans, selection, seed)
        features = extract_gap_features_per_stage(
            model, loaded, indices, device, stages=stages,
        )

        # Collect scan IDs for provenance.
        # Use dataset.subject_ids which correctly handles longitudinal
        # expansion (patient→scan), unlike test_indices which may be
        # patient-level for longitudinal datasets (e.g. GLI).
        dataset_ids = loaded.dataset.subject_ids
        scan_ids = [dataset_ids[i] for i in indices]

        save_dict = {f"stage_{s}": features[s] for s in stages}
        save_dict["scan_ids"] = np.array(scan_ids)
        out_path = raw_dir / f"gap_features_{domain}.npz"
        np.savez_compressed(out_path, **save_dict)
        logger.info("Saved %s features to %s", domain.upper(), out_path)

    del model
    if device != "cpu":
        torch.cuda.empty_cache()


def phase_metrics(
    dd_cfg: DictConfig,
    raw_dir: Path,
) -> None:
    """Phase 2: compute per-stage domain divergence metrics from NPZ."""
    logger.info("=" * 60)
    logger.info("PHASE: METRICS")
    logger.info("=" * 60)

    stages = tuple(int(s) for s in dd_cfg.stages)
    n_mmd_perm = int(dd_cfg.mmd_permutations)
    n_bootstrap = int(dd_cfg.bootstrap_ci_samples)

    gli_npz = np.load(raw_dir / "gap_features_gli.npz", allow_pickle=True)
    men_npz = np.load(raw_dir / "gap_features_men.npz", allow_pickle=True)

    gli_features = {s: gli_npz[f"stage_{s}"] for s in stages}
    men_features = {s: men_npz[f"stage_{s}"] for s in stages}

    results = compute_per_stage_domain_metrics(
        gli_features, men_features,
        stages=stages,
        n_mmd_perm=n_mmd_perm,
        n_bootstrap=n_bootstrap,
    )

    # Save as CSV
    rows = [asdict(m) for m in results.values()]
    df = pd.DataFrame(rows)
    df.to_csv(raw_dir / "domain_metrics.csv", index=False)
    logger.info("Domain metrics saved to %s", raw_dir / "domain_metrics.csv")


def phase_cka_cross_stage(
    dd_cfg: DictConfig,
    raw_dir: Path,
) -> None:
    """Phase 3: cross-stage CKA matrix for MEN domain."""
    logger.info("=" * 60)
    logger.info("PHASE: CKA CROSS-STAGE")
    logger.info("=" * 60)

    stages = tuple(int(s) for s in dd_cfg.stages)
    men_npz = np.load(raw_dir / "gap_features_men.npz", allow_pickle=True)
    features = {s: men_npz[f"stage_{s}"] for s in stages}

    matrix = compute_cka_cross_stage(features)
    out_path = raw_dir / "cka_cross_stage_men.npy"
    np.save(out_path, matrix)
    logger.info("CKA cross-stage matrix (%d×%d) saved to %s", *matrix.shape, out_path)


def phase_cka_drift(
    config: DictConfig,
    dd_cfg: DictConfig,
    analysis_config: DictConfig,
    raw_dir: Path,
    device: str,
) -> None:
    """Phase 4: CKA adaptation drift for each adapted config."""
    logger.info("=" * 60)
    logger.info("PHASE: CKA DRIFT")
    logger.info("=" * 60)

    stages = tuple(int(s) for s in dd_cfg.stages)
    adapted_configs = dd_cfg.adapted_configs
    base_results_dir = Path(analysis_config.paths.base_results_dir)

    # Load frozen MEN features
    men_npz = np.load(raw_dir / "gap_features_men.npz", allow_pickle=True)
    frozen_features = {s: men_npz[f"stage_{s}"] for s in stages}
    men_scan_ids = list(men_npz["scan_ids"])

    # Load frozen MEN dataset to get scan indices
    roi_size = tuple(int(x) for x in dd_cfg.roi_size)
    men_loaded = build_men_loader(config, roi_size=roi_size)

    # Reconstruct the same indices used in extraction
    n_scans = int(dd_cfg.n_scans_per_domain)
    selection = str(dd_cfg.scan_selection)
    seed = int(dd_cfg.seed)
    indices = select_scan_indices(len(men_loaded.dataset), n_scans, selection, seed)

    drift_rows: list[dict] = []

    for ac in adapted_configs:
        name = str(ac.name)
        run_dir = base_results_dir / str(ac.run_dir)
        rank = int(ac.rank)
        member_id = int(ac.member_id)
        logger.info("CKA drift: %s (run_dir=%s, rank=%d, member=%d)", name, run_dir, rank, member_id)

        model = load_adapted_model(
            config, analysis_config, rank=rank,
            member_id=member_id, device=device,
            run_dir=run_dir,
        )

        adapted_features = extract_gap_features_per_stage(
            model, men_loaded, indices, device, stages=stages,
        )

        del model
        if device != "cpu":
            torch.cuda.empty_cache()

        drift = compute_cka_adaptation_drift(frozen_features, adapted_features)

        row = {"config_name": name, "rank": rank, "member_id": member_id}
        for s in stages:
            row[f"cka_stage_{s}"] = drift.get(s, float("nan"))
        drift_rows.append(row)
        logger.info("  Drift: %s", {s: f"{v:.4f}" for s, v in drift.items()})

    df = pd.DataFrame(drift_rows)
    df.to_csv(raw_dir / "cka_drift.csv", index=False)
    logger.info("CKA drift saved to %s", raw_dir / "cka_drift.csv")


def phase_decoder_patch(
    config: DictConfig,
    dd_cfg: DictConfig,
    analysis_config: DictConfig,
    raw_dir: Path,
    device: str,
) -> None:
    """Phase 5 (optional): decoder patching sensitivity."""
    logger.info("=" * 60)
    logger.info("PHASE: DECODER PATCHING")
    logger.info("=" * 60)

    stages = tuple(int(s) for s in dd_cfg.stages)
    roi_size = tuple(int(x) for x in dd_cfg.roi_size)
    seed = int(dd_cfg.seed)
    n_pairs = min(20, int(dd_cfg.n_scans_per_domain))

    men_loaded = build_men_loader(config, roi_size=roi_size)
    gli_loaded = build_gli_loader(config, roi_size=roi_size)

    men_indices = select_scan_indices(len(men_loaded.dataset), n_pairs, "first", seed)
    gli_indices = select_scan_indices(len(gli_loaded.dataset), n_pairs, "first", seed)

    # Use adapted model for decoder patching (first adapted config)
    adapted_configs = dd_cfg.adapted_configs
    ac = adapted_configs[0]
    base_results_dir = Path(analysis_config.paths.base_results_dir)
    run_dir = base_results_dir / str(ac.run_dir)

    model = load_adapted_model(
        config, analysis_config, rank=int(ac.rank),
        member_id=int(ac.member_id), device=device,
        run_dir=run_dir,
    )

    # Access the decoder wrapper
    decoder = model.decoder
    decoder.eval()

    sensitivity_rows: list[dict] = []

    for pair_i, (men_idx, gli_idx) in enumerate(zip(men_indices, gli_indices)):
        men_sample = men_loaded.dataset[men_idx]
        gli_sample = gli_loaded.dataset[gli_idx]

        men_images = men_sample["image"].unsqueeze(0).to(device)
        gli_images = gli_sample["image"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device != "cpu")):
            men_hs = extract_hidden_states(model, men_images)
            gli_hs = extract_hidden_states(model, gli_images)

        sensitivity = compute_decoder_patching_sensitivity(
            decoder, men_hs, gli_hs, men_images, device, stages=stages,
        )

        men_sid = men_loaded.dataset.subject_ids[men_idx]
        gli_sid = gli_loaded.dataset.subject_ids[gli_idx]

        row = {
            "pair_idx": pair_i,
            "men_scan_id": men_sid,
            "gli_scan_id": gli_sid,
        }
        for s in stages:
            row[f"sensitivity_stage_{s}"] = sensitivity.get(s, float("nan"))
        sensitivity_rows.append(row)

        del men_images, gli_images, men_hs, gli_hs
        if device != "cpu":
            torch.cuda.empty_cache()

        if (pair_i + 1) % 5 == 0:
            logger.info("  Decoder patching: %d / %d pairs", pair_i + 1, len(men_indices))

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    df = pd.DataFrame(sensitivity_rows)
    df.to_csv(raw_dir / "decoder_patching.csv", index=False)
    logger.info("Decoder patching saved to %s", raw_dir / "decoder_patching.csv")


def phase_correlate(
    dd_cfg: DictConfig,
    raw_dir: Path,
) -> None:
    """Phase 6: Spearman correlation between divergence and drift."""
    logger.info("=" * 60)
    logger.info("PHASE: CORRELATE")
    logger.info("=" * 60)

    stages = tuple(int(s) for s in dd_cfg.stages)

    metrics_csv = raw_dir / "domain_metrics.csv"
    drift_csv = raw_dir / "cka_drift.csv"

    if not metrics_csv.exists() or not drift_csv.exists():
        logger.warning("Skipping correlation: metrics or drift CSV missing")
        return

    metrics_df = pd.read_csv(metrics_csv)
    drift_df = pd.read_csv(drift_csv)

    domain_acc = {int(row["stage"]): float(row["domain_acc_linear"]) for _, row in metrics_df.iterrows()}

    # BH correction on MMD p-values
    mmd_p_raw = np.array([float(metrics_df.loc[metrics_df["stage"] == s, "mmd_p"].values[0]) for s in stages])
    adj_p, is_sig = apply_bh_correction(mmd_p_raw)
    bh_results = {int(s): {"adjusted_p": float(adj_p[i]), "significant": bool(is_sig[i])} for i, s in enumerate(stages)}

    correlation_results: dict[str, dict] = {}
    for _, row in drift_df.iterrows():
        config_name = str(row["config_name"])
        cka_drift = {s: float(row[f"cka_stage_{s}"]) for s in stages if f"cka_stage_{s}" in row}
        rho, p = compute_drift_divergence_correlation(domain_acc, cka_drift)
        correlation_results[config_name] = {
            "spearman_rho": rho,
            "spearman_p": p,
        }
        logger.info("  %s: Spearman rho=%.4f (p=%.4f)", config_name, rho, p)

    output = {
        "domain_acc_per_stage": {str(k): v for k, v in domain_acc.items()},
        "bh_correction": {str(k): v for k, v in bh_results.items()},
        "correlations": correlation_results,
    }

    out_path = raw_dir / "drift_divergence_correlation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Correlation results saved to %s", out_path)


def phase_figures(
    dd_cfg: DictConfig,
    fig_cfg: DictConfig | None,
    raw_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    """Phase 7: generate figures and tables from raw artefacts."""
    logger.info("=" * 60)
    logger.info("PHASE: FIGURES + TABLES")
    logger.info("=" * 60)

    fmt = "pdf"
    dpi = 300
    if fig_cfg is not None:
        fmt = str(fig_cfg.get("save_format", "pdf"))
        dpi = int(fig_cfg.get("save_dpi", 300))

    stages = tuple(int(s) for s in dd_cfg.stages)

    # Import figure/table generators (lazy so we don't fail if matplotlib missing)
    from .figures.fig_domain_divergence_panel import render_domain_divergence_panel
    from .figures.fig_cka_drift import render_cka_drift
    from .tables.generate_domain_divergence_tables import (
        generate_domain_metrics_table,
        generate_drift_table,
    )

    # Domain divergence panel (4-panel figure)
    metrics_csv = raw_dir / "domain_metrics.csv"
    cka_matrix_path = raw_dir / "cka_cross_stage_men.npy"
    dad_csv_path = dd_cfg.get("dad_csv_path", None)

    if metrics_csv.exists() and cka_matrix_path.exists():
        render_domain_divergence_panel(
            metrics_csv=metrics_csv,
            cka_matrix_path=cka_matrix_path,
            dad_csv_path=dad_csv_path,
            stages=stages,
            out_path=figures_dir / f"domain_divergence_panel.{fmt}",
            figsize=tuple(fig_cfg.get("panel_figsize", [14, 10])) if fig_cfg else (14, 10),
            dpi=dpi,
        )

    # CKA drift figure
    drift_csv = raw_dir / "cka_drift.csv"
    if drift_csv.exists():
        render_cka_drift(
            drift_csv=drift_csv,
            stages=stages,
            out_path=figures_dir / f"cka_drift.{fmt}",
            figsize=tuple(fig_cfg.get("drift_figsize", [10, 5])) if fig_cfg else (10, 5),
            dpi=dpi,
        )

    # Tables
    if metrics_csv.exists():
        generate_domain_metrics_table(
            metrics_csv=metrics_csv,
            out_dir=tables_dir,
        )
    if drift_csv.exists():
        corr_path = raw_dir / "drift_divergence_correlation.json"
        generate_drift_table(
            drift_csv=drift_csv,
            correlation_json=corr_path if corr_path.exists() else None,
            out_dir=tables_dir,
        )

    logger.info("Figures written to %s", figures_dir)
    logger.info("Tables written to %s", tables_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


PHASES = [
    "extract", "metrics", "cka_cross_stage", "cka_drift",
    "decoder_patch", "correlate", "figures",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Domain divergence analysis for LoRA stage targeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Parent config.yaml")
    parser.add_argument(
        "--analysis-config", default=None,
        help="Domain divergence config (default: explainability/config/domain_divergence.yaml)",
    )
    parser.add_argument("--output", default=None, help="Override output_dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--phase", default="all",
        choices=["all"] + PHASES,
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick validation: N=3 scans at 64^3, reduced permutations",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    analysis_path = args.analysis_config or (
        Path(args.config).parent / "explainability" / "config" / "domain_divergence.yaml"
    )
    analysis_config = OmegaConf.load(analysis_path)

    # Apply smoke-test overrides
    if args.smoke_test:
        smoke_overrides = OmegaConf.create({
            "domain_divergence": {
                "n_scans_per_domain": 3,
                "roi_size": [64, 64, 64],
                "mmd_permutations": 50,
                "bootstrap_ci_samples": 50,
                "decoder_patching": False,
            }
        })
        analysis_config = OmegaConf.merge(analysis_config, smoke_overrides)
        logger.info("SMOKE TEST MODE: N=3, ROI=64^3, reduced permutations")

    dd_cfg = analysis_config.domain_divergence
    fig_cfg = analysis_config.get("figure", None)

    output_dir = Path(args.output or analysis_config.paths.output_dir)
    raw_dir = output_dir / "raw"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for d in (raw_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    # Save merged config snapshot
    OmegaConf.save(
        OmegaConf.merge(config, analysis_config),
        output_dir / "config_snapshot.yaml",
        resolve=True,
    )

    logger.info("Domain divergence pipeline: phase=%s, device=%s, output=%s",
                args.phase, args.device, output_dir)
    logger.info("N_scans=%d, ROI=%s, stages=%s",
                int(dd_cfg.n_scans_per_domain),
                list(dd_cfg.roi_size),
                list(dd_cfg.stages))

    t_start = time.time()

    run_all = args.phase == "all"

    if run_all or args.phase == "extract":
        phase_extract(config, dd_cfg, raw_dir, args.device)

    if run_all or args.phase == "metrics":
        phase_metrics(dd_cfg, raw_dir)

    if run_all or args.phase == "cka_cross_stage":
        phase_cka_cross_stage(dd_cfg, raw_dir)

    if run_all or args.phase == "cka_drift":
        phase_cka_drift(config, dd_cfg, analysis_config, raw_dir, args.device)

    if (run_all or args.phase == "decoder_patch") and dd_cfg.get("decoder_patching", False):
        phase_decoder_patch(config, dd_cfg, analysis_config, raw_dir, args.device)

    if run_all or args.phase == "correlate":
        phase_correlate(dd_cfg, raw_dir)

    if run_all or args.phase == "figures":
        phase_figures(dd_cfg, fig_cfg, raw_dir, tables_dir, figures_dir)

    elapsed = time.time() - t_start
    logger.info("Domain divergence pipeline complete in %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
