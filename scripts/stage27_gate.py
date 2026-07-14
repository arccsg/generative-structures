"""TESTS 5+7 (v10.3, EXPLORATORY): co-occurrence adversarial worked example
and the ground-truth grid library. Plan, modulus set, g0 recipe, and flag
rule frozen in config/run_config_gate27.json BEFORE computation.

TEST 5: fabricated-and-gridded synthetic series through the two-step gate --
the gate reports "grid-compatible": it attributes the ALARM, not the DATA.
TEST 7: sensitivity/specificity of the gate across documented-grid series and
non-grid lookalikes already cached in the project (no new acquisition).

Outputs: tables/gate27_library.csv (one row per series with all gate fields)
"""
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import build02_lib as lib

CFG = json.load(open(os.path.join(common.CONFIG, "run_config_gate27.json")))
SEED = CFG["seed"]
T = common.TABLES
CACHE = os.path.join(common.OUT, "diagnostics", "intermediate02", "emp11_cache")
MODULI = CFG["modulus_set"]
G0_CAND = [1000, 500, 250, 200, 100, 50, 25, 20, 10, 5, 4, 2]
FLAG_TV = 0.05


def bs_chi2(v):
    """Beber-Scacco last-digit chi2 on records >= 1000."""
    w = v[v >= 1000]
    if len(w) < 100:
        return np.nan, np.nan, len(w)
    obs = np.bincount(w % 10, minlength=10)
    exp = len(w) / 10.0
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    return chi2, float(stats.chi2.sf(chi2, 9)), len(w)


def tv_excess(v, rng):
    """Residue TV per modulus minus the digit-matched uniform-null mean."""
    v = v[v > 1]
    digs = np.array([len(str(int(x))) for x in v])
    out = {}
    for m in MODULI:
        c = np.bincount(v % m, minlength=m) / len(v)
        tv = 0.5 * np.abs(c - 1.0 / m).sum()
        nulls = []
        for _ in range(20):
            u = np.concatenate([rng.integers(max(10 ** (d - 1), 2), 10 ** d, size=n)
                                for d, n in zip(*np.unique(digs, return_counts=True))])
            cu = np.bincount(u % m, minlength=m) / len(u)
            nulls.append(0.5 * np.abs(cu - 1.0 / m).sum())
        out[m] = float(tv - np.mean(nulls))
    return out


def g0_of(v):
    for g in G0_CAND:
        if np.mean(v % g == 0) >= 0.90:
            return g
    return 1


def gate(name, truth, v, rng):
    v = np.asarray(v, dtype=np.int64)
    v = v[v > 1]
    chi2, p, n_el = bs_chi2(v)
    ex = tv_excess(v, rng)
    g0 = g0_of(v)
    flagged = (max(ex.values()) >= FLAG_TV) or (g0 > 1)
    row = dict(series=name, truth=truth, n=len(v), bs_chi2=chi2, bs_p=p,
               n_bs=n_el, g0=g0, flag_grid_compatible=bool(flagged),
               max_tv_excess=float(max(ex.values())),
               argmax_modulus=int(max(ex, key=ex.get)),
               **{f"tv{m}": ex[m] for m in MODULI})
    print(f"{name:34s} truth={truth:8s} g0={g0:5d} maxTVex={row['max_tv_excess']:+.3f}"
          f"@{row['argmax_modulus']:3d} chi2={chi2 if chi2==chi2 else -1:9.1f} "
          f"flag={flagged}", flush=True)
    return row


def main():
    lib.init_worker()
    rng = np.random.default_rng(SEED)
    rows = []

    # ---------------- TEST 5: fabricated AND gridded
    x = rng.lognormal(mean=9.0, sigma=1.0, size=3000)
    fab5 = (np.round(x / 5) * 5).astype(np.int64)
    rows.append(gate("FABRICATED_lognormal_5grid", "grid", fab5, rng))

    # ---------------- TEST 7 library
    panel = pd.read_csv(os.path.join(T, "vote14_panel.csv"))

    def cs(**flt):
        m = pd.Series(True, index=panel.index)
        for k, val in flt.items():
            m &= panel[k] == val
        return panel[m]["count"].to_numpy(np.int64)

    rows.append(gate("CVAP_county_modeled", "grid",
                     cs(level="county", contest="cvap_2018_2022"), rng))
    ces = pd.read_csv(os.path.join(T, "emp11_panel.csv"))
    cesv = (pd.to_numeric(ces[ces.source == "CES"].value_thousands,
                          errors="coerce").dropna().to_numpy() * 1000)
    rows.append(gate("CES_survey_employment", "grid",
                     np.rint(cesv).astype(np.int64), rng))
    # ACS margin-of-error column (cached ACS detailed-table .dat)
    try:
        acs = pd.read_csv(os.path.join(CACHE, "acsdt5y2022-b01003.dat"), sep="|")
        moe = pd.to_numeric(acs["B01003_M001"], errors="coerce").dropna()
        rows.append(gate("ACS_margin_of_error", "grid",
                         moe[moe > 1].astype(np.int64).to_numpy(), rng))
    except Exception as e:
        print("ACS MOE unavailable:", repr(e)[:80])
    # binary-aligned allocation demo (factor-2 grid): powers-of-two blocks
    q = rng.integers(2, 5000, size=4000)
    rows.append(gate("binary_aligned_allocations", "grid",
                     (q * 4096).astype(np.int64), rng))
    # constructed non-decimal ticks on real county totals
    tot = cs(level="county", contest="president_2024", candidate="total")
    rows.append(gate("constructed_47tick", "grid", (tot // 47) * 47, rng))
    rows.append(gate("constructed_43tick", "grid", (tot // 43) * 43, rng))

    # non-grid lookalikes
    rows.append(gate("county_pres_total_2020", "nongrid",
                     cs(level="county", contest="president_2020", candidate="total"), rng))
    rows.append(gate("county_pres_total_2024", "nongrid", tot, rng))
    rows.append(gate("nc_precinct_2024", "nongrid",
                     cs(level="precinct", contest="president_2024"), rng))
    try:
        co = pd.read_csv(os.path.join(CACHE, "co-est2019-alldata.csv"),
                         encoding="latin1", usecols=["CENSUS2010POP"])
        rows.append(gate("county_population_census2010", "nongrid",
                         co.CENSUS2010POP.dropna().astype(np.int64).to_numpy(), rng))
    except Exception as e:
        print("census pop unavailable:", repr(e)[:80])
    rows.append(gate("additive_sums_synthetic", "nongrid",
                     rng.integers(1, 10 ** 4, size=(3000, 12)).sum(axis=1), rng))
    rows.append(gate("uniform_null", "nongrid",
                     rng.integers(10 ** 4, 10 ** 7, size=4000), rng))

    df = pd.DataFrame(rows)
    sens = df[(df.truth == "grid")].flag_grid_compatible.mean()
    spec = 1 - df[(df.truth == "nongrid")].flag_grid_compatible.mean()
    print(f"\nsensitivity {sens:.2f} over {int((df.truth=='grid').sum())} grid series; "
          f"specificity {spec:.2f} over {int((df.truth=='nongrid').sum())} non-grid series")
    df.to_csv(os.path.join(T, "gate27_library.csv"), index=False)
    print("written gate27_library.csv")


if __name__ == "__main__":
    main()
