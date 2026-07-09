"""Build 08 (run hunt08): first real hunt — structured products in the wild.

Stages (python stage15_hunt08.py <stage>):
  structured — exact-factorization families (Legendre/Kummer): factorials,
               binomials, multinomials from real margins, group orders,
               anchors + prime-pair control. No value factorization needed.
  separation — magnitude/moment-matched single & sum nulls (factorized) for
               the in-range (d <= 18) subset; keff ROC-AUC per family.
  cod        — Crystallography Open Database sample: space-group orders,
               Wyckoff site multiplicities, per-structure multiplicity
               products. Polite fetch (~3,000 CIFs, 6 threads) — reduced
               from the >=20k ask out of courtesy; flagged in outputs.
  controls   — real additive negative controls from the frozen corpus
               (telemetry byte/packet counts, census counts) via existing
               channel_profiles_v3 keff.
  readout    — assemble read-out.

Null convention: E[X|d] from tables/baseline.csv for d <= 18; for d > 18 an
extrapolated null E[X|d] = X_inf + a/d fitted on strata 10..18 with the
asymptote PINNED to exact values (E[H2]->0.5 (Ewens theta=1), E[L1]->
0.6243299885, E[L2]->0.1860231051, E[Tail]->1-L1-L2). Rows using the
extrapolated null are marked null_source="extrapolated".

Pre-registered predictions are written to config/run_config_hunt08.json at
import time, before any computation.
"""
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT
import build02_lib as lib

DIAG = os.path.join(OUT, "diagnostics")
FROZEN = os.path.join(OUT, "frozen")
INTER = os.path.join(DIAG, "intermediate02")
RUN = "hunt08"
SEED = 20260708
WORKERS = 14
KTP = {"L1": 0.6243299885, "L2": 0.1860231051}
ASYMPT = {"E_L1": KTP["L1"], "E_L2": KTP["L2"],
          "E_Tail": 1.0 - KTP["L1"] - KTP["L2"], "E_H2": 0.5}

SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
GRAY = "#898781"

# ---- pre-registration (written before any computation)
PREREG = {
    "run": RUN, "seed": SEED, "workers": WORKERS,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "preregistered_predictions_from_gate07": {
        "structured_composite_products": "keff elevated; AUC >= 0.85 vs "
                                         "magnitude/moment-matched single "
                                         "and sum nulls",
        "prime_pair_products": "keff ~= 1.1 (near-null, reconfirming the "
                               "gate's surprise)",
        "count_additive_fields": "keff ~= 1.0",
        "rounded_anything": "collapses toward null (not tested here; "
                            "gate07 result stands)",
    },
    "null_extension": "d>18: E[X|d] = X_inf + a/d fitted on strata 10..18, "
                      "asymptotes pinned to PD(0,1)/Ewens exact values",
    "cod_note": "sample reduced to ~3,000 CIFs (6 threads, polite) from "
                "the >=20k ask — flagged; bulk pull left to the user",
}
with open(os.path.join(CONFIG, "run_config_hunt08.json"), "w") as _f:
    json.dump(PREREG, _f, indent=2)


# ------------------------------------------------------------- null model
class Null:
    def __init__(self):
        bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
        bl = bl[bl.stratum != "PD(0,1)"].copy()
        bl["d"] = bl.stratum.astype(int)
        self.tab = bl.set_index("d")
        self.coef = {}
        fit = bl[bl.d >= 10]
        for c in ("E_L1", "E_L2", "E_Tail", "E_H2"):
            a = np.mean((fit[c] - ASYMPT[c]) * fit.d)
            self.coef[c] = a

    def get(self, d, col):
        if d <= 18:
            return float(self.tab.loc[d, col]), "empirical"
        return ASYMPT[col] + self.coef[col] / d, "extrapolated"


def profile_from_pairs(pairs):
    """L-profile from an exact factorization {p: e}. Returns
    (log10_n, L1, L2, Tail, H2)."""
    contrib = [(e * math.log(p)) for p, e in pairs.items() if e > 0]
    ln_n = sum(contrib)
    log10 = ln_n / math.log(10)
    contrib.sort(reverse=True)
    L = [c / ln_n for c in contrib]
    L1 = L[0]
    L2 = L[1] if len(L) > 1 else 0.0
    H2 = sum(x * x for x in L)
    return log10, L1, L2, 1.0 - L1 - L2, H2


# ----------------------------------------------------- exact factorizers
def legendre_factorial(n, primes):
    out = {}
    for p in primes:
        if p > n:
            break
        e, q = 0, p
        while q <= n:
            e += n // q
            q *= p
        out[p] = e
    return out


def sub_pairs(a, b):
    out = dict(a)
    for p, e in b.items():
        out[p] = out.get(p, 0) - e
        if out[p] == 0:
            del out[p]
    return out


def make_structured():
    from sympy import primerange, factorint, randprime
    primes_20k = list(primerange(2, 20001))
    rows = []
    null = Null()

    def add(family, label, pairs):
        pairs = {p: e for p, e in pairs.items() if e > 0}
        if not pairs:            # value == 1 (degenerate) — skip
            return
        log10, L1, L2, T, H2 = profile_from_pairs(pairs)
        d = int(log10) + 1
        nH2, src = null.get(d, "E_H2")
        nL1, _ = null.get(d, "E_L1")
        nT, _ = null.get(d, "E_Tail")
        rows.append(dict(family=family, label=label, log10=log10, d=d,
                         L1=L1, L2=L2, Tail=T, H2=H2, keff=nH2 / H2,
                         dL1=L1 - nL1, dTail=T - nT, null_source=src))

    # factorials 8..120
    for n in range(8, 121):
        add("factorial", f"{n}!", legendre_factorial(n, primes_20k))
    # central binomials C(2n,n), n = 10..300 step 10
    for n in range(10, 301, 10):
        pairs = sub_pairs(legendre_factorial(2 * n, primes_20k),
                          legendre_factorial(n, primes_20k))
        pairs = sub_pairs(pairs, legendre_factorial(n, primes_20k))
        add("central_binomial", f"C({2*n},{n})", pairs)
    # general binomials
    for n in (50, 100, 200, 300):
        for k in (5, 10, 25, 50):
            if k >= n:
                continue
            pairs = sub_pairs(legendre_factorial(n, primes_20k),
                              legendre_factorial(k, primes_20k))
            pairs = sub_pairs(pairs, legendre_factorial(n - k, primes_20k))
            add("binomial", f"C({n},{k})", pairs)

    # multinomials from REAL margins (our own frozen corpus counts)
    obs = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                      usecols=["dataset_family", "domain"],
                      low_memory=False)
    margins = {
        "domain_channels": obs.domain.value_counts().tolist(),
        "domain_families": obs.groupby("domain").dataset_family.nunique()
        .tolist(),
    }
    rng = np.random.default_rng(SEED)
    fam_counts = obs.dataset_family.value_counts().to_numpy()
    for N in (50, 200, 1000, 5000):
        # subsampled real channel-per-family margins; cap any single margin
        # at N//10 so the vector stays genuinely multi-category
        cap = max(1, N // 10)
        take, tot = [], 0
        for c in rng.permutation(fam_counts):
            c = int(min(c, cap, N - tot))
            if c > 0:
                take.append(c)
                tot += c
            if tot >= N:
                break
        margins[f"family_margins_N{N}"] = take
    for name, ms in margins.items():
        N = sum(ms)
        pairs = legendre_factorial(N, primes_20k)
        for m in ms:
            pairs = sub_pairs(pairs, legendre_factorial(m, primes_20k))
        add("multinomial_real_margins", f"{name}(N={N})", pairs)

    # group orders: GL(k,q)
    for q, qp, qm in [(2, 2, 1), (3, 3, 1), (4, 2, 2), (5, 5, 1),
                      (7, 7, 1), (8, 2, 3), (9, 3, 2)]:
        for k in range(2, 7):
            pairs = {qp: qm * k * (k - 1) // 2}
            for i in range(1, k + 1):
                for p, e in factorint(q ** i - 1).items():
                    pairs[p] = pairs.get(p, 0) + e
            add("GL_order", f"GL({k},{q})", pairs)
    # sporadic groups (verified factorizations)
    sporadics = {
        "M11": {2: 4, 3: 2, 5: 1, 11: 1},
        "M12": {2: 6, 3: 3, 5: 1, 11: 1},
        "M24": {2: 10, 3: 3, 5: 1, 7: 1, 11: 1, 23: 1},
        "J1": {2: 3, 3: 1, 5: 1, 7: 1, 11: 1, 19: 1},
        "Monster": {2: 46, 3: 20, 5: 9, 7: 6, 11: 2, 13: 3, 17: 1, 19: 1,
                    23: 1, 29: 1, 31: 1, 41: 1, 47: 1, 59: 1, 71: 1},
    }
    for name, pairs in sporadics.items():
        add("sporadic_group_order", name, pairs)

    # anchors: primorials
    prim = {}
    for p in primes_20k:
        if p > 300:
            break
        prim[p] = 1
        add("primorial", f"{p}#", dict(prim))
    # B-smooth anchor (products of primes <= 31, in 1e10..1e14)
    r2 = np.random.default_rng([SEED, 2])
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for i in range(300):
        pairs = {}
        ln = 0.0
        while ln < math.log(10) * r2.uniform(10, 14):
            p = int(r2.choice(small_primes))
            pairs[p] = pairs.get(p, 0) + 1
            ln += math.log(p)
        add("bsmooth_31", f"smooth_{i}", pairs)
    # prime-pair control: balanced large primes, product in 1e14..1e16
    for i in range(300):
        lo = 10 ** r2.uniform(7.0, 8.0)
        p1 = randprime(int(lo), int(lo * 1.6))
        p2 = randprime(int(lo), int(lo * 1.6))
        add("prime_pair", f"pq_{i}", {int(p1): 1, int(p2): 1})

    df = pd.DataFrame(rows)
    df.round(6).to_csv(os.path.join(TABLES, "hunt_structured.csv"),
                       index=False, encoding="utf-8")
    print(f"structured: {len(df)} values across "
          f"{df.family.nunique()} families "
          f"(magnitude 10^{df.log10.min():.0f}..10^{df.log10.max():.0f})")
    return df


# ------------------------------------------------------------- separation
def _factor_chunk(vals):
    out = np.empty(len(vals))
    for i, v in enumerate(vals.tolist()):
        out[i] = lib.lprofile(v)[4]     # H2
    return out


def separation(df):
    from sklearn.metrics import roc_auc_score
    null = Null()
    rng = np.random.default_rng([SEED, 3])
    pool = ProcessPoolExecutor(max_workers=WORKERS,
                               initializer=lib.init_worker)
    rows = []
    for fam, g in df.groupby("family"):
        sub = g[g.d <= 18]
        if len(sub) < 5:
            rows.append(dict(family=fam, n_in_range=len(sub),
                             auc_vs_single=np.nan, auc_vs_sum=np.nan,
                             note="too few in-range values for AUC"))
            continue
        mu, sd = sub.log10.mean(), max(0.15, sub.log10.std())
        n_null = 1500
        t = np.clip(rng.normal(mu, sd, n_null), 2, 17.9)
        singles = np.floor(10 ** t).astype(np.int64)
        a = np.floor(10 ** np.clip(rng.normal(mu - 0.301, sd, n_null),
                                   1.5, 17.5)).astype(np.int64)
        b = np.floor(10 ** np.clip(rng.normal(mu - 0.301, sd, n_null),
                                   1.5, 17.5)).astype(np.int64)
        sums = a + b
        H2s = np.concatenate(list(pool.map(
            _factor_chunk, np.array_split(singles, WORKERS))))
        H2m = np.concatenate(list(pool.map(
            _factor_chunk, np.array_split(sums, WORKERS))))
        # keff score comparable across mixed magnitudes: keff_rec
        ks = np.array([null.get(len(str(v)), "E_H2")[0]
                       for v in singles]) / H2s
        km = np.array([null.get(len(str(v)), "E_H2")[0]
                       for v in sums]) / H2m
        kf = sub.keff.to_numpy()
        auc_s = roc_auc_score(np.r_[np.ones(len(kf)), np.zeros(len(ks))],
                              np.r_[kf, ks])
        auc_m = roc_auc_score(np.r_[np.ones(len(kf)), np.zeros(len(km))],
                              np.r_[kf, km])
        rows.append(dict(family=fam, n_in_range=len(sub),
                         mean_keff=float(sub.keff.mean()),
                         auc_vs_single=float(auc_s),
                         auc_vs_sum=float(auc_m), note=""))
        print(f"  {fam:<26} n={len(sub):<4} keff={sub.keff.mean():.2f} "
              f"AUC(single)={auc_s:.3f} AUC(sum)={auc_m:.3f}", flush=True)
    pool.shutdown()
    sep = pd.DataFrame(rows)
    sep.round(4).to_csv(os.path.join(TABLES, "hunt_separation.csv"),
                        index=False, encoding="utf-8")
    return sep


def fig_structured(df, sep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    gate = pd.read_csv(os.path.join(TABLES, "gate_rungs.csv"))
    fams = df.groupby("family").keff.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE_C)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    y = np.arange(len(fams))
    for yi, (fam, k) in enumerate(fams.items()):
        vals = df[df.family == fam].keff
        ax.scatter(vals, np.full(len(vals), yi), s=9, color=PALETTE[0],
                   alpha=0.35, linewidths=0, zorder=2)
        ax.scatter([k], [yi], s=55, marker="D", color=PALETTE[5],
                   edgecolors=SURFACE, zorder=4)
    for rung, colr in [("single", GRAY), ("prod2", PALETTE[2]),
                       ("prod4", PALETTE[3])]:
        kv = float(gate[gate.rung == rung].keff.iloc[0])
        ax.axvline(kv, color=colr, linewidth=1.0, linestyle="--")
        ax.text(kv, len(fams) - 0.2, f"gate {rung} ({kv:.2f})", fontsize=7,
                color=colr, rotation=90, va="top", ha="right")
    ax.set_yticks(y, fams.index, fontsize=8.5, color=INK)
    ax.set_xlabel("keff (diamonds = family mean; dots = values)",
                  fontsize=9, color=MUTED)
    ax.set_title("Structured products in the wild — keff by family, gate "
                 "rungs overlaid", fontsize=11, color=INK, loc="left",
                 pad=12)
    fig.text(0.01, 0.01, f"run {RUN} · seed {SEED} · exact factorizations "
             "(Legendre/Kummer); d>18 nulls extrapolated (pinned "
             "asymptotes)", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig08_keff_structured.png"),
                facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------- COD
COD_QUERIES = [
    {"el1": "Si", "el2": "O", "nel1": "2", "nel2": "2"},
    {"el1": "Fe", "el2": "O", "nel1": "2", "nel2": "3"},
    {"el1": "C", "el2": "H", "el3": "N", "el4": "O", "nel1": "4",
     "nel2": "4"},
    {"el1": "Ti", "el2": "O", "nel1": "2", "nel2": "3"},
    {"el1": "Ca", "el2": "C", "el3": "O", "nel1": "3", "nel2": "3"},
]
N_CIF = 3000


def cod_fetch_ids():
    import urllib.parse
    import urllib.request
    ids = set()
    for q in COD_QUERIES:
        params = dict(q)
        params["format"] = "lst"
        url = ("https://www.crystallography.net/cod/result.php?" +
               urllib.parse.urlencode(params))
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                for line in r.read().decode().splitlines():
                    line = line.strip()
                    if line.isdigit():
                        ids.add(line)
        except Exception as e:
            print(f"  COD id query failed ({q}): {e!r}", flush=True)
    return sorted(ids)


def _fetch_cif(cid):
    import urllib.request
    try:
        url = f"https://www.crystallography.net/cod/{cid}.cif"
        with urllib.request.urlopen(url, timeout=25) as r:
            return cid, r.read().decode("utf-8", errors="replace")
    except Exception:
        return cid, None


def parse_cif(text):
    """Return (n_symops, [site multiplicities]) from CIF text."""
    n_symop = 0
    for tag in ("_space_group_symop_operation_xyz",
                "_symmetry_equiv_pos_as_xyz"):
        if tag in text:
            after = text.split(tag, 1)[1]
            for line in after.splitlines()[1:]:
                s = line.strip()
                if not s or s.startswith(("_", "loop_", "#", ";")):
                    break
                n_symop += 1
            break
    mults = []
    if "_atom_site_symmetry_multiplicity" in text or \
            "_atom_site_site_symmetry_multiplicity" in text:
        m = re.search(r"loop_\s*((?:\s*_atom_site\S*\n)+)((?:[^\n_]*\n)+)",
                      text)
        if m:
            headers = [h.strip() for h in m.group(1).split()]
            idx = None
            for cand in ("_atom_site_symmetry_multiplicity",
                         "_atom_site_site_symmetry_multiplicity"):
                if cand in headers:
                    idx = headers.index(cand)
                    break
            if idx is not None:
                for line in m.group(2).splitlines():
                    parts = line.split()
                    if len(parts) == len(headers):
                        try:
                            mults.append(int(float(parts[idx])))
                        except ValueError:
                            pass
    return n_symop, mults


def cod():
    lib.init_worker()
    null = Null()
    print("COD: querying id lists...", flush=True)
    ids = cod_fetch_ids()
    print(f"COD: {len(ids):,} ids pooled from {len(COD_QUERIES)} queries",
          flush=True)
    if not ids:
        print("COD UNAVAILABLE — flagged for user bulk pull; proceeding "
              "without Stage 3", flush=True)
        pd.DataFrame([{"status": "cod_unreachable"}]).to_csv(
            os.path.join(TABLES, "hunt_crystallography.csv"), index=False)
        return None
    rng = np.random.default_rng([SEED, 7])
    sample = list(rng.choice(ids, size=min(N_CIF, len(ids)),
                             replace=False))
    rows = []
    got = 0
    with ThreadPoolExecutor(max_workers=6) as tp:
        for cid, text in tp.map(_fetch_cif, sample):
            if text is None:
                continue
            got += 1
            n_symop, mults = parse_cif(text)
            prod = 1
            for m in mults:
                if 1 <= m <= 10 ** 6:
                    prod *= m
                if prod > 10 ** 17:
                    break
            rows.append(dict(cod_id=cid, sg_order=n_symop,
                             n_sites=len(mults),
                             mult_product=prod if mults else None,
                             mults=";".join(map(str, mults[:50]))))
            if got % 500 == 0:
                print(f"  fetched {got}", flush=True)
    cdf = pd.DataFrame(rows)
    print(f"COD: parsed {len(cdf)} structures "
          f"({(cdf.sg_order > 0).sum()} with symops, "
          f"{cdf.mult_product.notna().sum()} with site multiplicities)",
          flush=True)

    # profile the three integer types
    out = []
    for typ, vals in [
            ("sg_order", cdf[cdf.sg_order > 1].sg_order.astype(int)),
            ("site_multiplicity",
             pd.Series([int(x) for s in cdf.mults.dropna()
                        for x in s.split(";") if x and int(x) > 1])),
            ("structure_mult_product",
             cdf[cdf.mult_product.notna() & (cdf.mult_product > 1)]
             .mult_product.astype(np.int64))]:
        for v in vals:
            l1, l2, l3, tail, h2, r = lib.lprofile(int(v))
            d = len(str(int(v)))
            nH2, src = null.get(d, "E_H2")
            out.append(dict(integer_type=typ, value=int(v), d=d, L1=l1,
                            Tail=tail, H2=h2, keff=nH2 / h2,
                            null_source=src))
    prof = pd.DataFrame(out)
    prof.round(5).to_csv(os.path.join(TABLES, "hunt_crystallography.csv"),
                         index=False, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE_C)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    colors = {"sg_order": PALETTE[0], "site_multiplicity": PALETTE[2],
              "structure_mult_product": PALETTE[5]}
    for typ, g in prof.groupby("integer_type"):
        jitter = np.random.default_rng(1).uniform(-0.12, 0.12, len(g))
        ax.scatter(np.log10(g.value.astype(float)) + jitter, g.keff, s=8,
                   color=colors[typ], alpha=0.35, linewidths=0, label=typ)
        gm = g.groupby("d").keff.mean()
        ax.plot(gm.index - 0.5, gm.values, color=colors[typ], linewidth=1.6,
                marker="D", markersize=5, markeredgecolor=SURFACE)
    ax.axhline(1.0, color=BASELINE_C, linewidth=1.0)
    gate = pd.read_csv(os.path.join(TABLES, "gate_rungs.csv"))
    ax.axhline(float(gate[gate.rung == "prod2"].keff.iloc[0]),
               color=GRAY, linewidth=1.0, linestyle="--")
    ax.text(0.15, float(gate[gate.rung == "prod2"].keff.iloc[0]) + 0.03,
            "gate prod2", fontsize=7, color=GRAY)
    ax.set_xlabel("log10(value) — magnitude caveat: multiplicities are "
                  "1–3 digit integers", fontsize=9, color=MUTED)
    ax.set_ylabel("keff", fontsize=9, color=MUTED)
    ax.set_title("Crystallography (COD sample): keff vs magnitude for the "
                 "three integer types", fontsize=10.5, color=INK,
                 loc="left", pad=12)
    ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK)
    fig.text(0.01, 0.01, f"run {RUN} · COD CC0 sample n={len(cdf)} "
             "structures (reduced from >=20k for server courtesy)",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig08_crystallography.png"),
                facecolor=SURFACE)
    plt.close(fig)
    return prof


# -------------------------------------------------------------- controls
def controls():
    ch = pd.read_csv(os.path.join(TABLES, "channel_profiles_v3.csv"),
                     low_memory=False)
    obs2 = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                       usecols=["dataset_id", "domain"], low_memory=False)
    ch = ch.merge(obs2, on="dataset_id", suffixes=("", "_v2"))
    tele = ch[(ch.domain_v2 == "network_telemetry") &
              ch.column_name.isin(["n_bytes", "n_packets", "n_flows"])]
    census = ch[(ch.domain_v2 == "census") & (ch.channel_kind == "count")]
    rows = []
    for name, g in [("telemetry_bytes_packets_flows", tele),
                    ("census_counts", census)]:
        rows.append(dict(control=name, channels=len(g),
                         keff_mean=float(g.keff.mean()),
                         keff_sd=float(g.keff.std()),
                         keff_p90=float(g.keff.quantile(0.9))))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(TABLES,
                                     "hunt_negative_controls.csv"),
                        index=False, encoding="utf-8")
    return out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    df = sep = prof = ctr = None
    if stage in ("structured", "all"):
        df = make_structured()
    if stage in ("separation", "all"):
        if df is None:
            df = pd.read_csv(os.path.join(TABLES, "hunt_structured.csv"))
        sep = separation(df)
        fig_structured(df, sep)
    if stage in ("cod", "all"):
        prof = cod()
    if stage in ("controls", "all"):
        ctr = controls()
    if stage in ("readout", "all"):
        if df is None:
            df = pd.read_csv(os.path.join(TABLES, "hunt_structured.csv"))
        if sep is None:
            sep = pd.read_csv(os.path.join(TABLES, "hunt_separation.csv"))
        if prof is None and os.path.exists(
                os.path.join(TABLES, "hunt_crystallography.csv")):
            prof = pd.read_csv(os.path.join(TABLES,
                                            "hunt_crystallography.csv"))
            if "integer_type" not in prof.columns:
                prof = None
        if ctr is None:
            ctr = pd.read_csv(os.path.join(TABLES,
                                           "hunt_negative_controls.csv"))
        print("\n================ HUNT08 READ-OUT ================")
        print("1. structured products vs matched nulls "
              "(pre-registered AUC >= 0.85):")
        for r in sep.sort_values("family").itertuples():
            if np.isnan(r.auc_vs_single):
                print(f"   {r.family:<26} {r.note}")
            else:
                ok = "PASS" if (r.auc_vs_single >= 0.85 and
                                r.auc_vs_sum >= 0.85) else "MISS"
                print(f"   {r.family:<26} keff={r.mean_keff:.2f} "
                      f"AUC(single)={r.auc_vs_single:.3f} "
                      f"AUC(sum)={r.auc_vs_sum:.3f}  {ok}")
        pp = df[df.family == "prime_pair"].keff
        print(f"2. prime-pair control: keff={pp.mean():.3f} ± "
              f"{pp.std():.3f} (pre-registered ~1.1) -> "
              f"{'RECONFIRMED' if abs(pp.mean()-1.1) < 0.15 else 'DEVIATES'}")
        if prof is not None:
            for typ, g in prof.groupby("integer_type"):
                print(f"3. COD {typ:<26} n={len(g):<6} "
                      f"median value d={int(g.d.median())} "
                      f"keff={g.keff.mean():.2f}±{g.keff.std():.2f}")
        else:
            print("3. COD unavailable — flagged for user bulk pull")
        print("4. negative controls:")
        for r in ctr.itertuples():
            ok = "PASS" if abs(r.keff_mean - 1.0) < 0.08 else "MISS"
            print(f"   {r.control:<32} n={r.channels} "
                  f"keff={r.keff_mean:.3f}±{r.keff_sd:.3f}  {ok}")


if __name__ == "__main__":
    main()
