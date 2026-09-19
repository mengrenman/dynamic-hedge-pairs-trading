"""
Network filtering and centrality for cointegration graphs.

Implements the machinery of Grande & Borondo (2025), *Embedding pairs trading in market
networks: a network science approach to portfolio construction*, Humanities & Social Sciences
Communications 12:1477, so that its central claim -- that pairs built from **peripheral** assets
outperform pairs built from central ones -- can be tested on this repository's own data.

Exports
-------
- tmfg(W)              : Triangular Maximally Filtered Graph (Massara, Di Matteo & Aste 2016)
- pmfg(W)              : Planar Maximally Filtered Graph (Tumminello et al. 2005)
- pozzi_xy(G)          : the X and Y centrality/peripherality indices of Pozzi, Di Matteo &
                         Aste (2013), as the paper uses them
- classify_nodes(xy)   : split nodes into peripheral / central quartiles by X + Y

Conventions
-----------
Edge weights are *similarities*: larger means a stronger relationship, so both filters keep the
largest weights. For a cointegration network the natural weight is -log10(p), which is what
`notebooks/network_pairs_day_lake.ipynb` uses.

In the X/Y indices, **small X + Y means central and large X + Y means peripheral** -- the sign
convention of the paper, and the opposite of what "centrality" alone would suggest.
"""
from __future__ import annotations

from typing import Dict, Hashable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["tmfg", "pmfg", "pozzi_xy", "classify_nodes"]


def _as_frame(W) -> pd.DataFrame:
    if isinstance(W, pd.DataFrame):
        if list(W.index) != list(W.columns):
            raise ValueError("W must be square with matching index and columns")
        return W
    A = np.asarray(W, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"W must be a square matrix, got shape {A.shape}")
    return pd.DataFrame(A, index=range(A.shape[0]), columns=range(A.shape[0]))


def _edge_list(W: pd.DataFrame) -> list[tuple[Hashable, Hashable, float]]:
    """Upper-triangle edges, strongest first. NaN weights are dropped, not treated as zero."""
    A = W.to_numpy(dtype=float)
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    w = A[iu]
    keep = np.isfinite(w)
    i, j, w = iu[0][keep], iu[1][keep], w[keep]
    order = np.argsort(-w, kind="stable")
    names = list(W.index)
    return [(names[i[k]], names[j[k]], float(w[k])) for k in order]


def tmfg(W) -> "nx.Graph":
    """Triangular Maximally Filtered Graph.

    Greedy construction of a chordal planar graph made only of 3-cliques (Massara, Di Matteo &
    Aste 2016): start from the 4-clique of the four nodes with the largest total weight, then
    repeatedly insert the vertex that gains the most weight into the triangular face that gains
    the most. Retains 3(N-2) edges, the same count as the PMFG, but is far cheaper because no
    planarity test is ever run.
    """
    import networkx as nx

    W = _as_frame(W)
    names = list(W.index)
    n = len(names)
    if n < 4:
        raise ValueError(f"TMFG needs at least 4 nodes, got {n}")
    A = np.nan_to_num(W.to_numpy(dtype=float), nan=0.0).copy()
    np.fill_diagonal(A, 0.0)

    # seed: the four vertices with the largest sum of weights to all others
    seed = list(np.argsort(-A.sum(axis=1))[:4])
    G = nx.Graph()
    G.add_nodes_from(names)
    for a in range(4):
        for b in range(a + 1, 4):
            G.add_edge(names[seed[a]], names[seed[b]], weight=A[seed[a], seed[b]])

    # the four triangular faces of the seed tetrahedron
    faces = [tuple(sorted(c)) for c in
             ((seed[0], seed[1], seed[2]), (seed[0], seed[1], seed[3]),
              (seed[0], seed[2], seed[3]), (seed[1], seed[2], seed[3]))]
    remaining = [i for i in range(n) if i not in seed]

    # gain[v, f] = total weight from v into face f; track the best face per vertex incrementally
    while remaining:
        best_v, best_f, best_gain = None, None, -np.inf
        for v in remaining:
            for f in faces:
                g = A[v, f[0]] + A[v, f[1]] + A[v, f[2]]
                if g > best_gain:
                    best_v, best_f, best_gain = v, f, g
        for u in best_f:
            G.add_edge(names[best_v], names[u], weight=A[best_v, u])
        faces.remove(best_f)
        faces.extend([tuple(sorted((best_v, best_f[0], best_f[1]))),
                      tuple(sorted((best_v, best_f[0], best_f[2]))),
                      tuple(sorted((best_v, best_f[1], best_f[2])))])
        remaining.remove(best_v)
    return G


def pmfg(W, *, max_candidates: Optional[int] = None) -> "nx.Graph":
    """Planar Maximally Filtered Graph (Tumminello et al. 2005).

    Add edges in decreasing weight order, keeping an edge only if the graph stays planar; stop at
    3(N-2) edges. Unlike the TMFG this admits 4-node cliques as well as triangles, so it is a
    superset in spirit but not in general a superset in fact.

    `max_candidates` caps how many of the strongest edges are considered. The PMFG is complete
    only when every edge is offered, but the weakest edges are almost never accepted once the
    graph is dense, so a cap trades a small amount of fidelity for a large amount of time. Leave
    it None for the exact filter.
    """
    import networkx as nx

    W = _as_frame(W)
    n = len(W)
    if n < 3:
        raise ValueError(f"PMFG needs at least 3 nodes, got {n}")
    target = 3 * (n - 2)

    G = nx.Graph()
    G.add_nodes_from(W.index)
    edges = _edge_list(W)
    if max_candidates is not None:
        edges = edges[:max_candidates]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
        if not nx.check_planarity(G, counterexample=False)[0]:
            G.remove_edge(u, v)
        elif G.number_of_edges() >= target:
            break
    return G


def _ranks(values: Dict[Hashable, float], nodes: list, *, descending: bool,
           tie: str = "floor") -> Dict[Hashable, float]:
    """Tied ranks, 1-based. descending=True gives rank 1 to the largest value.

    ``tie="average"`` is the ordinary midrank. ``tie="floor"`` truncates the midrank, which is
    what Grande & Borondo's published Fig. 2 table does -- with it, every value in that table is
    reproduced exactly; with midranks the tied nodes differ by 0.5 of a rank. It is not the
    principled choice, but it is theirs, so it is the default here for replication.
    """
    if tie not in ("average", "floor"):
        raise ValueError(f"tie must be 'average' or 'floor', got {tie!r}")
    s = pd.Series([values[k] for k in nodes], index=nodes, dtype=float)
    r = s.rank(method="average", ascending=not descending)
    if tie == "floor":
        r = np.floor(r)
    return r.to_dict()


def pozzi_xy(G, *, weight: str = "weight", tie: str = "floor") -> pd.DataFrame:
    """The X and Y indices of Pozzi, Di Matteo & Aste (2013).

    Five centrality measures -- degree (D), betweenness (BC), eccentricity (E), closeness (C) and
    eigenvector centrality (EC) -- each computed on the weighted and the unweighted graph, then
    converted to tied ranks and combined:

        X = (C^w_D + C^u_D + C^w_BC + C^u_BC - 4) / (4 (N-1))
        Y = (C^w_E + C^u_E + C^w_C + C^u_C + C^w_EC + C^u_EC - 6) / (6 (N-1))

    D, BC and EC are ranked in descending order and E and C in ascending order, so that the most
    central node scores 0 on X and the most peripheral scores 1. **Small X + Y is central.**

    For the distance-based measures (E, C) an edge's *weight* is a similarity, so the distance
    used is its reciprocal.

    ``tie`` selects the tie-breaking rule -- see :func:`_ranks`. The default reproduces the
    paper's Fig. 2 exactly; pass ``tie="average"`` for ordinary midranks.

    Returns a frame indexed by node with columns D, BC, E, C, EC (the summed weighted+unweighted
    ranks, for inspection) plus X, Y and XY.
    """
    import networkx as nx

    nodes = list(G.nodes())
    N = len(nodes)
    if N < 2:
        raise ValueError(f"need at least 2 nodes, got {N}")
    if not nx.is_connected(G):
        # eccentricity and closeness are undefined across components; work on the giant component
        # and give every other node the most-peripheral rank, which is what they are.
        giant = max(nx.connected_components(G), key=len)
        if len(giant) < N:
            H = G.subgraph(giant).copy()
            sub = pozzi_xy(H, weight=weight, tie=tie)
            out = pd.DataFrame(index=nodes, columns=sub.columns, dtype=float)
            out.loc[sub.index] = sub
            # A node outside the giant component is as peripheral as a node can be: X = Y = 1,
            # hence XY = 2. The per-measure rank columns take the largest rank available so that
            # they stay on the same scale as the connected nodes' ranks.
            off = [v for v in nodes if v not in giant]
            for col in ("D", "BC", "E", "C", "EC"):
                out.loc[off, col] = 2.0 * N
            out.loc[off, "X"] = 1.0
            out.loc[off, "Y"] = 1.0
            out.loc[off, "XY"] = 2.0
            return out.loc[nodes]

    dist = {(u, v): 1.0 / w for u, v, w in G.edges(data=weight, default=1.0) if w and w > 0}
    Gd = G.copy()
    for (u, v), d in dist.items():
        Gd[u][v]["_dist"] = d

    measures: Dict[str, Tuple[Dict, Dict, bool]] = {}
    #                               name: (weighted, unweighted, descending)
    measures["D"] = (dict(G.degree(weight=weight)), dict(G.degree()), True)
    measures["BC"] = (nx.betweenness_centrality(Gd, weight="_dist"),
                      nx.betweenness_centrality(G), True)
    measures["E"] = (nx.eccentricity(Gd, weight="_dist"), nx.eccentricity(G), False)
    measures["C"] = (nx.closeness_centrality(Gd, distance="_dist"),
                     nx.closeness_centrality(G), False)
    try:
        ec_w = nx.eigenvector_centrality_numpy(G, weight=weight)
        ec_u = nx.eigenvector_centrality_numpy(G)
    except Exception:                                   # pragma: no cover - tiny/degenerate graphs
        ec_w = nx.eigenvector_centrality(G, weight=weight, max_iter=5000, tol=1e-8)
        ec_u = nx.eigenvector_centrality(G, max_iter=5000, tol=1e-8)
    measures["EC"] = (ec_w, ec_u, True)

    out = pd.DataFrame(index=nodes, dtype=float)
    for name, (wv, uv, desc) in measures.items():
        rw = _ranks(wv, nodes, descending=desc, tie=tie)
        ru = _ranks(uv, nodes, descending=desc, tie=tie)
        out[name] = [rw[v] + ru[v] for v in nodes]

    out["X"] = (out["D"] + out["BC"] - 4.0) / (4.0 * (N - 1))
    out["Y"] = (out["E"] + out["C"] + out["EC"] - 6.0) / (6.0 * (N - 1))
    out["XY"] = out["X"] + out["Y"]
    return out


def classify_nodes(xy: pd.DataFrame, *, q: float = 0.25) -> pd.Series:
    """Label the top `q` of X+Y "peripheral" and the bottom `q` "central", the rest "middle"."""
    s = xy["XY"] if isinstance(xy, pd.DataFrame) else pd.Series(xy)
    lo, hi = s.quantile(q), s.quantile(1.0 - q)
    return pd.Series(np.where(s >= hi, "peripheral", np.where(s <= lo, "central", "middle")),
                     index=s.index, name="role")
