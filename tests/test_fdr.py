# tests/test_fdr.py
"""Unit tests for pairs.stats.cointegration.benjamini_hochberg_fdr."""
import numpy as np
import pytest

from pairs.stats.cointegration import benjamini_hochberg_fdr


class TestBenjaminiHochbergFdr:
    # ── basic contracts ───────────────────────────────────────────────────────

    def test_returns_reject_and_padj_arrays(self):
        pvals = np.array([0.01, 0.04, 0.20, 0.50, 0.90])
        reject, padj = benjamini_hochberg_fdr(pvals, alpha=0.05)
        assert reject.shape == pvals.shape
        assert padj.shape == pvals.shape

    def test_reject_is_bool_array(self):
        pvals = np.array([0.01, 0.10, 0.50])
        reject, _ = benjamini_hochberg_fdr(pvals)
        assert reject.dtype == bool

    def test_padj_clipped_to_unit_interval(self):
        pvals = np.array([1e-300, 0.5, 0.999, 1.0])
        _, padj = benjamini_hochberg_fdr(pvals)
        assert (padj >= 0.0).all() and (padj <= 1.0).all()

    def test_empty_input(self):
        reject, padj = benjamini_hochberg_fdr(np.array([]))
        assert len(reject) == 0
        assert len(padj) == 0

    def test_all_nan_input(self):
        pvals = np.array([np.nan, np.nan, np.nan])
        reject, padj = benjamini_hochberg_fdr(pvals)
        assert not reject.any()

    def test_single_significant_pvalue(self):
        pvals = np.array([0.001])
        reject, _ = benjamini_hochberg_fdr(pvals, alpha=0.05)
        assert reject[0]

    def test_single_nonsignificant_pvalue(self):
        pvals = np.array([0.99])
        reject, _ = benjamini_hochberg_fdr(pvals, alpha=0.05)
        assert not reject[0]

    # ── correctness ──────────────────────────────────────────────────────────

    def test_known_example_rejects_correct_hypotheses(self):
        # Example from Benjamini & Hochberg (1995), Table 1
        # 15 p-values; at alpha=0.05 the BH procedure rejects the first 4
        raw_pvals = np.array([
            0.0001, 0.0004, 0.0019, 0.0095,   # these 4 should be rejected
            0.0201, 0.0278, 0.0298, 0.0344,
            0.0459, 0.3240, 0.4262, 0.5719,
            0.6528, 0.7590, 1.000,
        ])
        reject, padj = benjamini_hochberg_fdr(raw_pvals, alpha=0.05)
        assert reject[:4].all(), "First 4 should be rejected at BH alpha=0.05"
        assert not reject[4:].any(), "Remaining 11 should not be rejected"

    def test_adjusted_pvalues_monotone_nondecreasing_after_sort(self):
        """BH-adjusted p-values must be non-decreasing when inputs are sorted."""
        pvals = np.sort(np.random.default_rng(0).uniform(0, 1, 50))
        _, padj = benjamini_hochberg_fdr(pvals)
        # padj is already in the same order as pvals (both sorted)
        assert (np.diff(padj) >= -1e-12).all()

    def test_all_rejected_when_all_tiny(self):
        pvals = np.full(20, 1e-10)
        reject, _ = benjamini_hochberg_fdr(pvals, alpha=0.05)
        assert reject.all()

    def test_none_rejected_when_all_large(self):
        pvals = np.full(20, 0.90)
        reject, _ = benjamini_hochberg_fdr(pvals, alpha=0.05)
        assert not reject.any()

    def test_nan_entries_not_rejected(self):
        pvals = np.array([0.001, np.nan, 0.002])
        reject, _ = benjamini_hochberg_fdr(pvals, alpha=0.05)
        assert not reject[1]   # NaN position must never be rejected

    def test_fdr_more_lenient_than_bonferroni(self):
        """BH must reject at least as many as Bonferroni at the same alpha."""
        rng = np.random.default_rng(42)
        pvals = np.concatenate([rng.uniform(0, 0.01, 10),    # true signals
                                rng.uniform(0.5, 1.0, 90)])  # true nulls
        alpha = 0.05
        reject_bh, _ = benjamini_hochberg_fdr(pvals, alpha=alpha)
        reject_bonf = pvals <= (alpha / len(pvals))
        assert reject_bh.sum() >= reject_bonf.sum()

    def test_reject_consistent_with_padj(self):
        """reject[i] must be True iff padj[i] <= alpha."""
        pvals = np.random.default_rng(7).uniform(0, 1, 30)
        alpha = 0.05
        reject, padj = benjamini_hochberg_fdr(pvals, alpha=alpha)
        np.testing.assert_array_equal(reject, padj <= alpha)
