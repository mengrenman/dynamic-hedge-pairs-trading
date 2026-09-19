"""Tests for pairs.stats.network — the TMFG/PMFG filters and the Pozzi X/Y indices.

The headline test replicates the worked example published as Fig. 2 of Grande & Borondo (2025),
whose X, Y and X+Y values are printed in the figure; reproducing all 21 of them exactly is what
licenses using this code to test the paper's claim on other data.
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pairs.stats.network import tmfg, pmfg, pozzi_xy, classify_nodes


@pytest.fixture
def paper_toy():
    """Fig. 2: hub C joined to O1..O5, with a leaf P hanging off O5."""
    G = nx.Graph()
    for o in ("O1", "O2", "O3", "O4", "O5"):
        G.add_edge("C", o, weight=1.0)
    G.add_edge("O5", "P", weight=1.0)
    return G


PAPER_FIG2 = pd.DataFrame(
    {"X":  [0.66, 0.66, 0.66, 0.66, 0.16, 0.00, 0.66],
     "Y":  [0.50, 0.50, 0.50, 0.50, 0.33, 0.33, 0.55],
     "XY": [1.16, 1.16, 1.16, 1.16, 0.50, 0.33, 1.22]},
    index=["O1", "O2", "O3", "O4", "O5", "C", "P"])


def test_reproduces_the_papers_published_toy_table(paper_toy):
    got = pozzi_xy(paper_toy).loc[PAPER_FIG2.index, ["X", "Y", "XY"]]
    trunc = np.floor(got * 100) / 100          # the paper truncates its printed values
    pd.testing.assert_frame_equal(trunc, PAPER_FIG2, atol=1e-9, check_dtype=False)


def test_small_xy_is_central_large_is_peripheral(paper_toy):
    xy = pozzi_xy(paper_toy)["XY"]
    assert xy.idxmin() == "C", "the hub must score lowest"
    assert xy["P"] == xy.max(), "the leaf must score highest"
    assert xy["C"] < xy["O5"] < xy["O1"]


def test_tie_rule_changes_values_but_not_the_central_node(paper_toy):
    floor_ = pozzi_xy(paper_toy, tie="floor")["XY"]
    avg = pozzi_xy(paper_toy, tie="average")["XY"]
    assert not np.allclose(floor_, avg), "the two conventions should differ on a tied graph"
    assert floor_.idxmin() == avg.idxmin() == "C"


def test_tie_must_be_a_known_rule(paper_toy):
    with pytest.raises(ValueError, match="tie must be"):
        pozzi_xy(paper_toy, tie="dense")


# ── the filters ──────────────────────────────────────────────────────────────

@pytest.fixture
def weights():
    rng = np.random.default_rng(0)
    n = 40
    A = rng.random((n, n))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0.0)
    names = [f"T{i:02d}" for i in range(n)]
    return pd.DataFrame(A, index=names, columns=names)


@pytest.mark.parametrize("filt", [tmfg, pmfg])
def test_filter_keeps_exactly_3n_minus_6_edges(weights, filt):
    G = filt(weights)
    assert G.number_of_nodes() == len(weights)
    assert G.number_of_edges() == 3 * (len(weights) - 2)


@pytest.mark.parametrize("filt", [tmfg, pmfg])
def test_filter_output_is_planar(weights, filt):
    assert nx.check_planarity(filt(weights), counterexample=False)[0]


def test_tmfg_is_chordal_and_connected(weights):
    G = tmfg(weights)
    assert nx.is_connected(G)
    assert nx.is_chordal(G), "the TMFG is built from 3-cliques, so it must be chordal"


def test_filters_prefer_strong_edges(weights):
    """A filter that kept arbitrary edges would not beat a random subgraph of the same size."""
    G = tmfg(weights)
    kept = np.mean([d["weight"] for *_, d in G.edges(data=True)])
    allw = weights.to_numpy()[np.triu_indices(len(weights), 1)]
    assert kept > np.mean(allw) + np.std(allw), f"kept mean {kept:.3f} vs all {allw.mean():.3f}"


def test_pmfg_candidate_cap_is_a_speed_tradeoff_not_a_correctness_change(weights):
    """Capping candidates may retain fewer edges, but never a non-planar or larger graph."""
    full, capped = pmfg(weights), pmfg(weights, max_candidates=200)
    assert capped.number_of_edges() <= full.number_of_edges()
    assert nx.check_planarity(capped, counterexample=False)[0]


@pytest.mark.parametrize("filt,n", [(tmfg, 3), (pmfg, 2)])
def test_filters_reject_graphs_that_are_too_small(filt, n):
    W = pd.DataFrame(np.ones((n, n)), index=range(n), columns=range(n))
    with pytest.raises(ValueError, match="needs at least"):
        filt(W)


def test_nan_weights_are_dropped_not_read_as_zero():
    n = 8
    A = np.full((n, n), 0.5)
    np.fill_diagonal(A, 0.0)
    A[0, 1] = A[1, 0] = np.nan
    W = pd.DataFrame(A, index=range(n), columns=range(n))
    assert not pmfg(W).has_edge(0, 1)


# ── classification ───────────────────────────────────────────────────────────

def test_classify_splits_into_quartiles(weights):
    xy = pozzi_xy(tmfg(weights))
    role = classify_nodes(xy, q=0.25)
    assert set(role.unique()) <= {"peripheral", "central", "middle"}
    assert (role == "peripheral").sum() >= 1 and (role == "central").sum() >= 1
    # peripheral nodes must all score above every central one
    assert xy.loc[role == "peripheral", "XY"].min() >= xy.loc[role == "central", "XY"].max()


def test_disconnected_nodes_are_treated_as_maximally_peripheral():
    G = nx.Graph()
    nx.add_path(G, ["a", "b", "c", "d"], weight=1.0)
    G.add_node("island")
    xy = pozzi_xy(G)
    assert xy.loc["island", "XY"] == pytest.approx(2.0)
    assert xy["XY"].idxmax() == "island"
