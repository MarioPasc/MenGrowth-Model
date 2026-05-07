"""Unit tests for statistics utilities (BH-FDR + Cohen's d helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.stage1_volumetric.main_experiment.modules.statistics import bh_fdr

pytestmark = [pytest.mark.unit]


def test_bh_fdr_all_null():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200)
    rej, padj = bh_fdr(p, q=0.05)
    # Under uniform p-values, BH should reject very few.
    assert rej.sum() <= 0.1 * len(p)
    assert padj.shape == p.shape


def test_bh_fdr_known_signal():
    p = np.array([0.001, 0.002, 0.04, 0.6, 0.9])
    rej, padj = bh_fdr(p, q=0.05)
    # BH p_adj for p=0.04: 5*0.04/3 = 0.0667 > 0.05 -> NOT rejected.
    # Only the two smallest pass at q=0.05.
    assert rej.tolist() == [True, True, False, False, False]
    assert (padj <= 1.0).all() and (padj >= 0.0).all()
    # Raise q and the third should pass.
    rej_high, _ = bh_fdr(p, q=0.10)
    assert rej_high[2]


def test_bh_fdr_handles_nans():
    p = np.array([0.001, np.nan, 0.5, 0.9])
    rej, padj = bh_fdr(p, q=0.05)
    assert not rej[1]  # NaN never rejected
    assert padj[1] == padj[1]  # not nan; treated as 1


def test_bh_fdr_monotone_in_q():
    p = np.linspace(0.001, 0.5, 50)
    rej_low, _ = bh_fdr(p, q=0.01)
    rej_high, _ = bh_fdr(p, q=0.20)
    # Higher q ⇒ at least as many rejections.
    assert rej_high.sum() >= rej_low.sum()
