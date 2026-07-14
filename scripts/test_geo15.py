"""Unit tests for stage22_geo15 (run: python3 scripts/test_geo15.py)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build02_lib as lib
import stage22_geo15 as g


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, (a, b)


def test_profile_from_pairs():
    # 720 = 2^4 * 3^2 * 5 -> contributions 4ln2, 2ln3, ln5
    pairs = [(2, 4), (3, 2), (5, 1)]
    ln = 4 * math.log(2) + 2 * math.log(3) + math.log(5)
    L1, L2, L3, tail, h2, r = g.profile_from_pairs(pairs)
    approx(L1, 4 * math.log(2) / ln)
    approx(L2, 2 * math.log(3) / ln)
    approx(L3, math.log(5) / ln)
    approx(tail, 1 - L1 - L2)
    assert r == 3
    # matches the canonical lib.lprofile on the same integer
    lib.init_worker()
    l1, l2, l3, t2, hh, rr = lib.lprofile(720)
    approx(L1, l1); approx(L3, l3); assert rr == 3


def test_legendre():
    # v_2(10!) = 8, v_5(10!) = 2
    pairs = dict(g.legendre_pairs(10, [2, 3, 5, 7]))
    assert pairs[2] == 8 and pairs[5] == 2 and pairs[3] == 4 and pairs[7] == 1


def test_binom_pairs():
    # C(10,5) = 252 = 2^2 * 3^2 * 7
    from sympy import primerange
    pairs = dict(g._binom_pairs(5, list(primerange(2, 12))))
    assert pairs == {2: 2, 3: 2, 7: 1}


def test_channel_means_structural_zeros():
    lib.init_worker()
    # 6 = 2*3 has 2 parts (L3 undefined/0); 30 = 2*3*5 has 3 parts
    cm = g.channel_means(np.array([6, 6, 30, 30], dtype=np.int64))
    approx(cm["frac_parts_lt3"], 0.5)
    l1, l2, l3, *_ = lib.lprofile(30)
    approx(cm["L3_cond"], l3)          # conditional mean excludes the 6s
    approx(cm["L3"], l3 / 2)           # all-records mean includes zeros


def test_transforms():
    v = np.array([100, 47, 250], dtype=np.int64)
    assert list(g._transform(v, "plus1", 1)) == [101, 48, 251]
    assert list(g._transform(v, "round10", 1)) == [100, 50, 250]
    assert list(g._transform(v, "grid_quotient", 50)) == [2, 5]
    assert g.deround_int(720) == 9
    assert g.strip_pairs([(2, 4), (3, 2), (5, 1), (11, 1)], 7) == [(11, 1)]


def test_digit_residual():
    import pandas as pd
    base = g.load_baseline()
    w = np.zeros(18); w[3] = 1.0     # all records 4-digit
    resid, exp = g.digit_residual(0.9, w, base, "E_L1")
    approx(exp, float(base.loc[4, "E_L1"]))
    approx(resid, 0.9 - exp)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")
