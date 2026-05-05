"""Tests for the benchmark analysis pipeline (metrics + aggregation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.uncertainty_segmentation.benchmark.analysis.aggregate import (
    model_order_from_df,
    pairwise_stats,
    select_best_median_worst,
)
from experiments.uncertainty_segmentation.benchmark.analysis.metrics import (
    HIGHER_IS_BETTER,
    LABELS,
    METRICS,
    compute_case_metrics,
    dice,
    hd95,
    label_mask,
    lesion_recall,
)

pytestmark = [pytest.mark.experiment, pytest.mark.unit]


def _make_volume(shape: tuple[int, int, int], boxes: list[tuple[slice, slice, slice, int]]) -> np.ndarray:
    vol = np.zeros(shape, dtype=np.int8)
    for s0, s1, s2, val in boxes:
        vol[s0, s1, s2] = val
    return vol


def test_dice_known_values() -> None:
    a = np.zeros((4, 4, 4), dtype=bool)
    b = np.zeros((4, 4, 4), dtype=bool)
    a[1:3, 1:3, 1:3] = True  # 8 voxels
    b[1:3, 1:3, 1:4] = True  # 12 voxels (slice 1:4 in last axis)
    assert dice(a, b) == pytest.approx(2 * 8 / (8 + 12))
    assert np.isnan(dice(np.zeros_like(a), np.zeros_like(b)))


def test_lesion_recall_any_overlap() -> None:
    gt = np.zeros((6, 6, 6), dtype=bool)
    gt[0:2, 0:2, 0:2] = True  # CC1
    gt[4:6, 4:6, 4:6] = True  # CC2
    pred = np.zeros_like(gt)
    pred[0, 0, 0] = True  # touches only CC1
    assert lesion_recall(gt, pred) == pytest.approx(0.5)
    pred[5, 5, 5] = True
    assert lesion_recall(gt, pred) == pytest.approx(1.0)
    assert np.isnan(lesion_recall(np.zeros_like(gt), pred))


def test_hd95_disjoint_far_apart() -> None:
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[0, 0, 0] = True
    b[9, 9, 9] = True
    val = hd95(a, b, spacing=(1.0, 1.0, 1.0))
    assert val == pytest.approx(np.linalg.norm([9, 9, 9]), rel=0.01)


def test_hd95_returns_nan_when_empty() -> None:
    a = np.zeros((4, 4, 4), dtype=bool)
    b = np.zeros((4, 4, 4), dtype=bool)
    a[0, 0, 0] = True
    assert np.isnan(hd95(a, b))
    assert np.isnan(hd95(b, a))


def test_label_mask_definitions() -> None:
    vol = np.array([0, 1, 2, 3], dtype=np.int8).reshape(1, 1, 4)
    assert label_mask(vol, "NETC").sum() == 1
    assert label_mask(vol, "SNFH").sum() == 1
    assert label_mask(vol, "ET").sum() == 1
    assert label_mask(vol, "TC").sum() == 2  # NETC + ET
    assert label_mask(vol, "WT").sum() == 3  # all foreground


def test_compute_case_metrics_returns_all_labels() -> None:
    gt = _make_volume((8, 8, 8), [(slice(2, 5), slice(2, 5), slice(2, 5), 3)])
    pred = _make_volume((8, 8, 8), [(slice(3, 6), slice(2, 5), slice(2, 5), 3)])
    rows = compute_case_metrics(gt, pred, spacing=(1.0, 1.0, 1.0))
    assert {r["label"] for r in rows} == set(LABELS)
    for row in rows:
        for metric in METRICS:
            assert metric in row


def test_pairwise_stats_anti_symmetric_d() -> None:
    rng = np.random.default_rng(0)
    n_cases, n_models = 20, 3
    values = rng.normal(size=(n_cases, n_models))
    values[:, 2] += 0.5  # third model systematically higher
    p, d = pairwise_stats(values, ["m1", "m2", "m3"])
    np.testing.assert_allclose(d, -d.T, atol=1e-12)
    assert (p == p.T).all()
    assert p[0, 1] > 0.0
    # m3 should be statistically distinguishable from m1
    assert p[0, 2] < 0.05


def test_pairwise_stats_handles_constant_diff() -> None:
    values = np.tile(np.array([0.5, 0.5, 0.5]), (10, 1))
    p, d = pairwise_stats(values, ["a", "b", "c"])
    assert (d == 0).all()
    assert (p == 1.0).all()


def test_select_best_median_worst_orders_by_metric() -> None:
    rows = []
    for case_idx in range(5):
        for model in ("M1", "Ours"):
            score = case_idx if model == "Ours" else case_idx * 0.5
            rows.append(
                {
                    "model": model,
                    "case_id": f"BraTS-MEN-0000{case_idx}-000",
                    "label": "TC",
                    "dice": score / 5,
                    "hd95": (5 - case_idx) * 1.0,
                    "lesion_recall": score / 5,
                }
            )
    df = pd.DataFrame(rows)
    order = model_order_from_df(df)
    assert order == ["M1", "Ours"]
    bmw = select_best_median_worst(df, order, analysis_root=_tmp_root(), rank_label="TC")
    assert bmw["dice"]["best"] == "BraTS-MEN-00004-000"
    assert bmw["dice"]["worst"] == "BraTS-MEN-00000-000"
    # HD95 lower is better → best at low index, worst at high index
    assert bmw["hd95"]["best"] == "BraTS-MEN-00004-000"
    assert bmw["hd95"]["worst"] == "BraTS-MEN-00000-000"
    assert HIGHER_IS_BETTER["hd95"] is False


def _tmp_root():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())
