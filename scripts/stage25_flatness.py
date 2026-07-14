"""TEST 3 (v10.3, EXPLORATORY): local-flatness stress simulation.

Latent densities with curvature comparable to the grid width: x ~
Exponential(scale = r*g) rounded onto grid g, ratios r in {0.5,...,50}; plus
a zero-inflated Poisson count case. Measures how non-generic the recorded
quotient q = n/g becomes (c2/c5 log-share and keff vs the digit-matched
uniform null) and whether the paper's de-round + digit-conditioning sequence
absorbs the deviation. Plan frozen in config/run_config_flat25.json.

Output: tables/flat25_results.csv
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import build02_lib as lib
import stage22_geo15 as g15

CFG = json.load(open(os.path.join(common.CONFIG, "run_config_flat25.json")))
SEED = CFG["seed"]
N = 40_000


def quotient_stats(q, base):
    """Profile + grid-share stats of an integer array vs digit-matched null."""
    q = q[q > 1]
    uniq, cnt = np.unique(q, return_counts=True)
    tot = cnt.sum()
    s = np.zeros(5)
    c25 = 0.0
    dh = np.zeros(18)
    for v, c in zip(uniq.tolist(), cnt.tolist()):
        l1, l2, l3, tail, h2, r = lib.lprofile(int(v))
        s += np.array([l1, l2, l3, tail, h2]) * c
        pairs = dict(lib.factor_pairs(int(v)))
        c25 += c * (pairs.get(2, 0) * math.log(2) + pairs.get(5, 0) * math.log(5)) / math.log(v)
        dh[min(len(str(v)), 18) - 1] += c
    m = s / tot
    w = dh / tot
    eL1 = float(np.dot(w, base["E_L1"].reindex(range(1, 19)).to_numpy()))
    eH2 = float(np.dot(w, base["E_H2"].reindex(range(1, 19)).to_numpy()))
    # null c2/c5 share of a generic integer: E[v_2] ln2 + E[v_5] ln5 over ln n
    # = (sum_k 2^-k) ln2 + (sum_k 5^-k) ln5 = 1*ln2 + 0.25*ln5 per value,
    # normalized by mean ln value of the sample
    mean_ln = float(np.average(np.log(uniq.astype(float)), weights=cnt))
    c25_null = (1.0 * math.log(2) + 0.25 * math.log(5)) / mean_ln
    sdL1 = float(np.dot(w, base["sd_L1"].reindex(range(1, 19)).to_numpy())) / math.sqrt(tot)
    return dict(n=int(tot), L1=m[0], dL1=m[0] - eL1, dL1_se=sdL1,
                keff=eH2 / m[4], c25_share=c25 / tot, c25_null=c25_null)


def main():
    lib.init_worker()
    base = g15.load_baseline()
    rng = np.random.default_rng(SEED)
    rows = []
    for gwidth in (10, 100):
        for r in (0.5, 1, 2, 5, 10, 50):
            scale = r * gwidth
            x = rng.exponential(scale, size=N) + 20 * gwidth  # keep q >= ~20
            n = (np.round(x / gwidth) * gwidth).astype(np.int64)
            q = (n // gwidth).astype(np.int64)
            st = quotient_stats(q, base)
            adequate = abs(st["dL1"]) < 2 * st["dL1_se"]
            rows.append(dict(case="exponential", grid=gwidth, ratio=r,
                             **st, flat_adequate=adequate))
            print(f"exp g={gwidth:3d} r={r:5.1f} dL1(q)={st['dL1']:+.4f} "
                  f"(2se={2*st['dL1_se']:.4f}) keff(q)={st['keff']:.3f} "
                  f"c25 {st['c25_share']:.4f} vs null {st['c25_null']:.4f} "
                  f"adequate={adequate}", flush=True)
    # zero-inflated Poisson counts on a 10-grid
    lam, zfrac = 30, 0.6
    x = rng.poisson(lam, size=N).astype(float)
    x[rng.random(N) < zfrac] = 0.0
    n = (np.round(x / 10) * 10).astype(np.int64)
    q = (n // 10).astype(np.int64)
    st = quotient_stats(q, base)
    rows.append(dict(case="zero_inflated_poisson", grid=10, ratio=lam / 10,
                     **st, flat_adequate=abs(st["dL1"]) < 2 * st["dL1_se"]))
    print(f"ZIP g=10 lam=30 zfrac=0.6: dL1(q)={st['dL1']:+.4f} "
          f"keff(q)={st['keff']:.3f} n={st['n']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(common.TABLES, "flat25_results.csv"),
                              index=False)
    print("written flat25_results.csv")


if __name__ == "__main__":
    main()
