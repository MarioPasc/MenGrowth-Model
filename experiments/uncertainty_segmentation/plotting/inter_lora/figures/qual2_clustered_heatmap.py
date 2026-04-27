"""Clustered heatmap of per-scan Dice across LoRA ranks.

Three stacked heatmaps (TC / WT / ET) sharing a hierarchically-clustered
row order.  Rows are BraTS-MEN test scans; columns progress from frozen BSF
(rank=0) through each non-zero rank in ascending order.  A dendrogram is
rendered to the left of the top panel only.  A log10-volume strip is appended
to the right margin of every panel.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage

from ..io_layer import InterLoraData
from ..style import (
    FULL_PAGE_MM,
    LABEL_COLORS,
    MM_TO_INCH,
    setup_inter_lora_style,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CMAP_HEATMAP: str = "RdYlGn"
_CMAP_VOLUME: str = "cividis"
_VMIN: float = 0.0
_VMAX: float = 1.0
_VCENTER: float = 0.7
_FAILURE_THRESHOLD: float = 0.4
_LINKAGE_METHOD: str = "ward"
_LINKAGE_METRIC: str = "euclidean"


# ---------------------------------------------------------------------------
# Helper: build per-label Dice matrix
# ---------------------------------------------------------------------------


def _build_dice_matrix(
    data: InterLoraData,
    label: str,
) -> tuple[list[str], list[str], np.ndarray]:
    """Assemble the scan × column Dice matrix for one BraTS-hierarchical label.

    Column order: frozen BSF (rank=0) followed by non-zero ranks ascending.
    Rows are all scan_ids present in the baseline DataFrame, in their natural
    (un-clustered) order.

    Args:
        data: Aggregated inter-rank data.
        label: One of ``"tc"``, ``"wt"``, ``"et"``.

    Returns:
        Tuple of:
            - scan_ids: list of scan_id strings (row labels).
            - col_labels: list of column label strings (``"BSF"``, ``"r=2"``, …).
            - matrix: float32 array of shape (n_scans, n_cols).
    """
    dice_col = f"dice_{label}"

    # Baseline column (rank == 0 or first run's baseline_dice)
    if 0 in data.all_rank_values:
        baseline_run = data.get_rank(0)
    else:
        baseline_run = data.get_rank(data.rank_values[0])
    baseline_df = baseline_run.baseline_dice.copy()

    # Canonical scan_id order from baseline
    scan_ids: list[str] = baseline_df["scan_id"].tolist()
    n_scans = len(scan_ids)

    # Column definitions: baseline + non-zero ranks
    col_labels: list[str] = ["BSF"]
    col_arrays: list[np.ndarray] = []

    # Baseline column
    base_indexed = baseline_df.set_index("scan_id")
    base_col = np.full(n_scans, np.nan, dtype=np.float32)
    for i, sid in enumerate(scan_ids):
        if sid in base_indexed.index and dice_col in base_indexed.columns:
            base_col[i] = float(base_indexed.loc[sid, dice_col])
    col_arrays.append(base_col)

    # Non-zero rank columns
    for rank in data.rank_values:
        col_labels.append(f"r={rank}")
        rr = data.get_rank(rank)
        ens_indexed = rr.ensemble_dice.set_index("scan_id")
        rank_col = np.full(n_scans, np.nan, dtype=np.float32)
        for i, sid in enumerate(scan_ids):
            if sid in ens_indexed.index and dice_col in ens_indexed.columns:
                rank_col[i] = float(ens_indexed.loc[sid, dice_col])
        col_arrays.append(rank_col)

    matrix = np.stack(col_arrays, axis=1)  # (n_scans, n_cols)
    assert matrix.shape == (n_scans, len(col_labels)), (
        f"Matrix shape mismatch: {matrix.shape} vs ({n_scans}, {len(col_labels)})"
    )
    return scan_ids, col_labels, matrix


# ---------------------------------------------------------------------------
# Helper: row clustering across all labels
# ---------------------------------------------------------------------------


def _cluster_rows(matrices: list[np.ndarray]) -> tuple[np.ndarray, Any]:
    """Compute hierarchical clustering order from concatenated label matrices.

    Clustering is performed on the horizontally-concatenated matrix
    ``[D_TC | D_WT | D_ET]`` (shape n_scans × 3·n_cols).  NaN values are
    replaced with the column mean before clustering.

    Args:
        matrices: List of per-label matrices, each shape (n_scans, n_cols).

    Returns:
        Tuple of:
            - order: int array of shape (n_scans,) giving the clustered row
              order (leaf order from the dendrogram).
            - Z: linkage matrix (for dendrogram rendering).
    """
    combined = np.concatenate(matrices, axis=1)  # (n_scans, 3*n_cols)

    # Impute NaN column-wise with column mean
    col_means = np.nanmean(combined, axis=0)
    nan_mask = np.isnan(combined)
    combined_filled = combined.copy()
    for j in range(combined.shape[1]):
        combined_filled[nan_mask[:, j], j] = col_means[j]

    # Fallback: if any value is still NaN, replace with 0.5
    combined_filled = np.nan_to_num(combined_filled, nan=0.5)

    Z = linkage(combined_filled, method=_LINKAGE_METHOD, metric=_LINKAGE_METRIC)
    dend = dendrogram(Z, no_plot=True)
    order = np.array(dend["leaves"], dtype=int)
    return order, Z


# ---------------------------------------------------------------------------
# Helper: draw a single heatmap panel
# ---------------------------------------------------------------------------


def _draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    order: np.ndarray,
    col_labels: list[str],
    *,
    label_color: str,
    panel_title: str,
    show_col_labels: bool,
    norm: TwoSlopeNorm,
    cmap: Any,
) -> Any:
    """Render one Dice heatmap panel onto *ax*.

    Args:
        ax: Target axes.
        matrix: Raw (un-ordered) matrix of shape (n_scans, n_cols).
        order: Row permutation from clustering.
        col_labels: Column label strings.
        label_color: Title colour for this BraTS region.
        panel_title: Panel title string (e.g. ``"(a) TC"``).
        show_col_labels: Whether to draw column tick labels on the x-axis.
        norm: TwoSlopeNorm instance.
        cmap: Matplotlib colormap.

    Returns:
        The AxesImage returned by ``imshow``.
    """
    ordered = matrix[order, :]  # apply clustering

    im = ax.imshow(
        ordered,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=0,
    )
    im.set_rasterized(True)

    n_scans, n_cols = ordered.shape

    # White-outlined cells for failure cases (Dice < threshold)
    for row_i in range(n_scans):
        for col_j in range(n_cols):
            val = ordered[row_i, col_j]
            if not np.isnan(val) and val < _FAILURE_THRESHOLD:
                rect = plt.Rectangle(
                    (col_j - 0.5, row_i - 0.5),
                    1,
                    1,
                    linewidth=0.6,
                    edgecolor="white",
                    facecolor="none",
                    zorder=2,
                )
                ax.add_patch(rect)

    # Axes decoration
    ax.set_title(panel_title, color=label_color, fontsize=9, pad=3, loc="left")
    ax.set_yticks([])

    if show_col_labels:
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    else:
        ax.set_xticks([])

    ax.tick_params(axis="both", which="both", length=0)

    return im


# ---------------------------------------------------------------------------
# Helper: draw volume strip
# ---------------------------------------------------------------------------


def _draw_volume_strip(
    ax: plt.Axes,
    scan_ids: list[str],
    order: np.ndarray,
    volumes: np.ndarray,
) -> None:
    """Render a per-scan log10-volume colour strip on *ax*.

    Args:
        ax: Target axes (should be a single-column narrow axes).
        scan_ids: Ordered list of scan_ids (pre-clustering order).
        order: Row permutation from clustering.
        volumes: Raw (un-ordered) 1-D array of tumour volumes in mm³.
    """
    ordered_vols = volumes[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vols = np.where(ordered_vols > 0, np.log10(ordered_vols), np.nan)

    # Replace NaN with column min for display purposes
    finite_mask = np.isfinite(log_vols)
    if finite_mask.any():
        fill_val = float(np.nanmin(log_vols[finite_mask]))
    else:
        fill_val = 0.0
    log_vols_display = np.where(np.isfinite(log_vols), log_vols, fill_val)

    strip = log_vols_display.reshape(-1, 1)  # (n_scans, 1)

    im = ax.imshow(
        strip,
        aspect="auto",
        cmap=_CMAP_VOLUME,
        interpolation="nearest",
        zorder=0,
    )
    im.set_rasterized(True)

    ax.set_xticks([0])
    ax.set_xticklabels(["Vol"], fontsize=6, rotation=45, ha="right")
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot(data: InterLoraData, config: dict[str, Any]) -> Figure | None:
    """Render the clustered heatmap of per-scan Dice across LoRA ranks.

    Layout: 3 stacked heatmap panels (TC, WT, ET) sharing hierarchically-
    clustered row order, with a dendrogram to the left of the top panel and a
    log10-volume strip to the right of every panel.

    Args:
        data: Aggregated inter-rank data (``InterLoraData``).
        config: Dict from the orchestrator config section
            ``qual2_clustered_heatmap``.  Recognised keys:

            - ``figsize``: (width_mm, height_mm) override (default
              ``FULL_PAGE_MM``).
            - ``failure_threshold``: Dice below which cells get a white
              outline (default 0.4).
            - ``vcenter``: Centre of the ``TwoSlopeNorm`` (default 0.7).

    Returns:
        Matplotlib ``Figure``, or ``None`` if fewer than 2 non-baseline ranks
        are available.
    """
    setup_inter_lora_style(config.get("style"))

    rank_values = data.rank_values
    if len(rank_values) < 1:
        logger.warning(
            "qual2_clustered_heatmap: no non-baseline ranks found; skipping.",
        )
        return None

    failure_thr: float = float(config.get("failure_threshold", _FAILURE_THRESHOLD))
    vcenter: float = float(config.get("vcenter", _VCENTER))
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=_VMIN, vmax=_VMAX)
    cmap = plt.get_cmap(_CMAP_HEATMAP)

    # Determine figure size
    size_mm = config.get("figsize", list(FULL_PAGE_MM))
    figsize = (size_mm[0] * MM_TO_INCH, size_mm[1] * MM_TO_INCH)

    # ------------------------------------------------------------------
    # 1. Build matrices for all three labels
    # ------------------------------------------------------------------
    scan_ids_tc, col_labels, mat_tc = _build_dice_matrix(data, "tc")
    _, _, mat_wt = _build_dice_matrix(data, "wt")
    _, _, mat_et = _build_dice_matrix(data, "et")

    # All matrices share the same scan_ids (from baseline); use TC list.
    scan_ids = scan_ids_tc
    n_scans = len(scan_ids)
    n_cols = len(col_labels)

    logger.info("qual2_clustered_heatmap: %d scans × %d columns", n_scans, n_cols)

    # ------------------------------------------------------------------
    # 2. Hierarchical clustering on concatenated matrix
    # ------------------------------------------------------------------
    order, Z = _cluster_rows([mat_tc, mat_wt, mat_et])

    # ------------------------------------------------------------------
    # 3. Extract per-scan GT volumes from the highest-rank ensemble_dice
    # ------------------------------------------------------------------
    max_rank = max(rank_values)
    max_rr = data.get_rank(max_rank)
    ens_indexed = max_rr.ensemble_dice.set_index("scan_id")
    volumes = np.zeros(n_scans, dtype=np.float32)
    vol_col = "volume_gt" if "volume_gt" in max_rr.ensemble_dice.columns else None
    if vol_col is not None:
        for i, sid in enumerate(scan_ids):
            if sid in ens_indexed.index:
                volumes[i] = float(ens_indexed.loc[sid, vol_col])
    else:
        logger.warning(
            "qual2_clustered_heatmap: 'volume_gt' column not found in "
            "ensemble_dice for rank %d; volume strip will be uniform.",
            max_rank,
        )

    # ------------------------------------------------------------------
    # 4. Figure layout
    # ------------------------------------------------------------------
    # Width ratios: [dendrogram | heatmap | volume_strip]
    # Dendrogram only rendered in row 0; hidden (invisible) in rows 1, 2.
    dend_width_ratio = 0.12
    vol_width_ratio = 0.04
    heat_width_ratio = 1.0 - dend_width_ratio - vol_width_ratio

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(
        left=0.02,
        right=0.90,
        top=0.96,
        bottom=0.06,
        hspace=0.04,
    )

    # GridSpec: 3 rows × 3 cols
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[dend_width_ratio, heat_width_ratio, vol_width_ratio],
        hspace=0.04,
        wspace=0.015,
        left=0.02,
        right=0.88,
        top=0.96,
        bottom=0.08,
    )

    labels_info = [
        ("tc", "(a) TC"),
        ("wt", "(b) WT"),
        ("et", "(c) ET"),
    ]
    matrices = [mat_tc, mat_wt, mat_et]
    heatmap_axes: list[plt.Axes] = []
    vol_axes: list[plt.Axes] = []
    last_im = None

    for row_idx, ((lbl, title), mat) in enumerate(zip(labels_info, matrices)):
        lbl_color = LABEL_COLORS[lbl]
        ax_dend = fig.add_subplot(gs[row_idx, 0])
        ax_heat = fig.add_subplot(gs[row_idx, 1])
        ax_vol = fig.add_subplot(gs[row_idx, 2])

        heatmap_axes.append(ax_heat)
        vol_axes.append(ax_vol)

        # Dendrogram: only in top panel
        if row_idx == 0:
            dendrogram(
                Z,
                ax=ax_dend,
                orientation="left",
                no_labels=True,
                color_threshold=0,
                above_threshold_color="#555555",
                link_color_func=lambda _k: "#555555",
            )
            ax_dend.invert_yaxis()
            ax_dend.set_axis_off()
        else:
            ax_dend.set_visible(False)

        # Heatmap
        show_col = row_idx == 2  # column labels only on bottom panel
        im = _draw_heatmap(
            ax_heat,
            mat,
            order,
            col_labels,
            label_color=lbl_color,
            panel_title=title,
            show_col_labels=show_col,
            norm=norm,
            cmap=cmap,
        )
        last_im = im

        # Volume strip
        _draw_volume_strip(ax_vol, scan_ids, order, volumes)
        # Show x-label only on bottom strip
        if row_idx < 2:
            ax_vol.set_xticks([])

    # ------------------------------------------------------------------
    # 5. Colorbars
    # ------------------------------------------------------------------
    # Dice colorbar
    cbar_ax = fig.add_axes([0.90, 0.25, 0.012, 0.50])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Dice", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([0.0, 0.4, 0.7, 1.0])

    # Volume colorbar (share vol strip mappable from last row)
    vol_sm = plt.cm.ScalarMappable(cmap=_CMAP_VOLUME)
    vol_sm.set_array(np.log10(volumes[volumes > 0]) if (volumes > 0).any() else [0])
    cbar_vol_ax = fig.add_axes([0.905, 0.06, 0.012, 0.15])
    cbar_vol = fig.colorbar(vol_sm, cax=cbar_vol_ax)
    cbar_vol.set_label(r"$\log_{10}V$", fontsize=6)
    cbar_vol.ax.tick_params(labelsize=5)

    logger.info(
        "qual2_clustered_heatmap: rendered %d scans, %d cols, order via ward linkage",
        n_scans,
        n_cols,
    )
    return fig
