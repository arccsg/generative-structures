"""Build 09 graphs v2: exact |Aut(G)| via twin-class reduction.

Vertices with identical open neighborhoods (false twins — mutually
non-adjacent) or identical closed neighborhoods (true twins — mutually
adjacent) form automorphism modules: any permutation within a class is an
automorphism, and Aut(G) = (prod Sym(C_i)) x| Aut(quotient with classes as
size-colored vertices). So |Aut(G)| = prod k_i! * |Aut_colored(quotient)|,
applied recursively until no twins remain; the terminal quotient is small
enough for BLISS generators + sympy Schreier-Sims. Factorial exponents come
from Legendre's formula — the order is never materialized as one integer.

Cross-check: igraph's BLISS count_automorphisms() (float) is logged next to
our exact log10 for every graph.
"""
import gzip
import json
import math
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, OUT
import build02_lib as lib
from stage16_hunt09 import Null, SNAP_GRAPHS

LN10 = math.log(10)


def legendre_factorial_exp(n, exp):
    """Multiply exp map by the factorization of n! (Legendre)."""
    if n < 2:
        return
    from sympy import primerange
    for p in primerange(2, n + 1):
        e, q = 0, p
        while q <= n:
            e += n // q
            q *= p
        exp[p] = exp.get(p, 0) + e


def twin_reduce(adj, colors, exp):
    """One pass of false+true twin collapse. adj: list[set]. Returns
    (new_adj, new_colors, changed)."""
    n = len(adj)
    changed = False
    for mode in ("false", "true"):
        sig = {}
        for v in range(len(adj)):
            key = (frozenset(adj[v]) if mode == "false"
                   else frozenset(adj[v]) | {v}, colors[v])
            sig.setdefault(key, []).append(v)
        classes = [vs for vs in sig.values()]
        if all(len(vs) == 1 for vs in classes):
            continue
        changed = True
        rep = {}
        newid = {}
        for ci, vs in enumerate(classes):
            k = len(vs)
            if k > 1:
                legendre_factorial_exp(k, exp)
            for v in vs:
                rep[v] = ci
        new_n = len(classes)
        new_adj = [set() for _ in range(new_n)]
        new_colors = []
        for ci, vs in enumerate(classes):
            v0 = vs[0]
            # color encodes (old color, class size) so quotient autos
            # respect class structure
            new_colors.append(hash((colors[v0], len(vs))) & 0x7FFFFFFF)
            for u in adj[v0]:
                cu = rep[u]
                if cu != ci:
                    new_adj[ci].add(cu)
        adj, colors = new_adj, new_colors
    return adj, colors, changed


def aut_order_exp(edges, n_nodes):
    """Exact factorization {p: e} of |Aut(G)|."""
    import igraph
    from sympy.combinatorics import Permutation, PermutationGroup
    from sympy import factorint
    adj = [set() for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    colors = [0] * n_nodes
    exp = {}
    for _ in range(30):
        adj, colors, changed = twin_reduce(adj, colors, exp)
        if not changed:
            break
    # normalize colors to 0..k
    uniq = {c: i for i, c in enumerate(sorted(set(colors)))}
    colors = [uniq[c] for c in colors]
    q_edges = [(a, b) for a in range(len(adj)) for b in adj[a] if a < b]
    g = igraph.Graph(n=len(adj), edges=q_edges, directed=False)
    gens = g.automorphism_group(color=colors)
    if gens:
        G = PermutationGroup([Permutation(p) for p in gens])
        for p, e in factorint(int(G.order())).items():
            exp[int(p)] = exp.get(int(p), 0) + int(e)
    return exp, len(adj)


def main():
    import igraph
    lib.init_worker()
    null = Null()
    rows = []
    for name in SNAP_GRAPHS:
        try:
            url = f"https://snap.stanford.edu/data/{name}.txt.gz"
            with urllib.request.urlopen(url, timeout=90) as r:
                raw = gzip.decompress(r.read()).decode("utf-8",
                                                       errors="replace")
            nodes, edges = {}, set()
            for line in raw.splitlines():
                if line.startswith("#"):
                    continue
                ab = line.split()
                if len(ab) < 2:
                    continue
                a = nodes.setdefault(ab[0], len(nodes))
                b = nodes.setdefault(ab[1], len(nodes))
                if a != b:
                    edges.add((min(a, b), max(a, b)))
            exp, q_size = aut_order_exp(list(edges), len(nodes))
            # BLISS cross-check on the full graph (may return a huge int —
            # never convert to float; math.log10 takes big ints directly)
            gfull = igraph.Graph(n=len(nodes), edges=list(edges),
                                 directed=False)
            try:
                bc = gfull.count_automorphisms()
                bliss_log10 = math.log10(bc) if bc and bc > 0 else np.nan
            except (OverflowError, ValueError, TypeError):
                bliss_log10 = np.nan
            if not exp:
                rows.append(dict(graph=name, n_nodes=len(nodes),
                                 n_edges=len(edges), quotient_nodes=q_size,
                                 aut_order_log10=0.0, d=1, keff=np.nan,
                                 bliss_log10=bliss_log10, trivial=True))
                print(f"  {name}: trivial Aut", flush=True)
                continue
            ln_n = sum(e * math.log(p) for p, e in exp.items())
            contribs = sorted((e * math.log(p) for p, e in exp.items()),
                              reverse=True)
            L = [c / ln_n for c in contribs]
            H2 = sum(x * x for x in L)
            d = int(ln_n / LN10) + 1
            rows.append(dict(graph=name, n_nodes=len(nodes),
                             n_edges=len(edges), quotient_nodes=q_size,
                             aut_order_log10=ln_n / LN10, d=d, L1=L[0],
                             Tail=1.0 - L[0] - (L[1] if len(L) > 1
                                                else 0.0),
                             H2=H2, keff=null.get(d, "E_H2") / H2,
                             omega=len(exp), maxexp=max(exp.values()),
                             bliss_log10=bliss_log10, trivial=False))
            print(f"  {name}: |V|={len(nodes):,} quotient={q_size:,} "
                  f"|Aut|~10^{ln_n/LN10:.1f} "
                  f"(bliss 10^{bliss_log10 if np.isfinite(bliss_log10) else float('nan'):.1f}) "
                  f"keff={null.get(d, 'E_H2')/H2:.2f}", flush=True)
        except Exception as e:
            print(f"  {name} failed: {e!r}", flush=True)
    gdf = pd.DataFrame(rows)
    gdf.round(5).to_csv(os.path.join(TABLES, "hunt09_highkeff.csv"),
                        index=False, encoding="utf-8")
    nt = gdf[~gdf.trivial] if len(gdf) else gdf
    if len(nt):
        big = nt[nt.aut_order_log10 >= 4]
        print(f"\ngraphs: {len(gdf)} total, {len(nt)} non-trivial, "
              f"{len(big)} with |Aut|>=1e4; keff of those "
              f"{big.keff.mean():.2f}±{big.keff.std():.2f}")


if __name__ == "__main__":
    main()
