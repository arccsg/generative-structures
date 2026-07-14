"""TEST 4 (v10.3, EXPLORATORY): quotient-level spike-in on count-kind hosts.

Extends the power13 tier-(b) magnitude-matched in-place design to count-kind
host channels. Host-selection recipe, injection design, detection rule, and
holds-rule frozen in config/run_config_count26.json BEFORE computation.
Either outcome is reported and scopes Section 7.

Outputs: tables/count26_hosts.csv, tables/count26_spikein.csv,
         tables/count26_confound.csv
"""
import json
import math
import os
import random
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import build02_lib as lib
import stage22_geo15 as g15

CFG = json.load(open(os.path.join(common.CONFIG, "run_config_count26.json")))
SEED = CFG["seed"]
T = common.TABLES
MODULI = (100, 25, 49, 47, 43)


def load_coremag():
    b = pd.read_csv(os.path.join(T, "baseline_coremag.csv"))
    b = b[b.cut == 7].set_index("core_d")
    return b


def core7(pairs):
    return [(p, e) for p, e in pairs if p > 7]


def value_core_stats(vals):
    """Per-unique-value core H2 and core digit length (cached arrays)."""
    uniq, cnt = np.unique(vals[vals > 1], return_counts=True)
    h2 = np.full(len(uniq), np.nan)
    cd = np.zeros(len(uniq), dtype=int)
    for i, v in enumerate(uniq.tolist()):
        kp = core7(lib.factor_pairs(int(v)))
        if not kp:
            continue
        l1, l2, l3, tail, hh, r = g15.profile_from_pairs(kp)
        h2[i] = hh
        cd[i] = int(sum(e * math.log10(p) for p, e in kp)) + 1
    return uniq, cnt, h2, cd


def keff_core_from(cnts, h2, cd, cm):
    ok = np.isfinite(h2) & (cnts > 0)
    if ok.sum() == 0:
        return np.nan
    w = cnts[ok].astype(float)
    eh2 = cm["E_H2"].reindex(cd[ok]).to_numpy()
    m = np.isfinite(eh2)
    if m.sum() == 0:
        return np.nan
    return float(np.average(eh2[m], weights=w[m]) /
                 np.average(h2[ok][m], weights=w[m]))


def coords11(vals):
    v = vals[vals > 1]
    ln = np.log(v.astype(float))
    v2 = np.zeros(len(v)); v5 = np.zeros(len(v))
    x = v.copy()
    while (x % 2 == 0).any():
        m = x % 2 == 0
        v2[m] += 1; x[m] //= 2
    x = v.copy()
    while (x % 5 == 0).any():
        m = x % 5 == 0
        v5[m] += 1; x[m] //= 5
    c2 = float(np.mean(v2 * math.log(2) / ln)); c5 = float(np.mean(v5 * math.log(5) / ln))
    trail10 = float(np.mean(np.minimum(v2, v5)))
    conform10 = float(np.mean(v % 10 == 0))
    digs = np.array([len(str(int(t))) for t in v])
    med = np.median(digs)
    rm = v2 * math.log(2) / ln + v5 * math.log(5) / ln
    split = abs(float(rm[digs <= med].mean() - (rm[digs > med].mean() if (digs > med).any() else rm.mean())))
    out = [c2 + c5, trail10, conform10, c2, c5]
    for m in MODULI:
        cnts = np.bincount(v % m, minlength=m) / len(v)
        out.append(float(0.5 * np.abs(cnts - 1.0 / m).sum()))
    out.append(split)
    return np.array(out)


def balanced_product(rng_r, t):
    from sympy import randprime
    half = t / 2.0
    a = randprime(int(10 ** max(half - 0.15, 0.8)), int(10 ** (half + 0.15)) + 3)
    b = randprime(int(10 ** max(half - 0.15, 0.8)), int(10 ** (half + 0.15)) + 3)
    return int(a) * int(b)


def main():
    lib.init_worker()
    random.seed(SEED)
    cm = load_coremag()
    corpus = pd.read_csv(CFG_INPUT := "frozen/observational_corpus_v2.csv")
    cand = corpus[(corpus.channel_kind == "count") & (corpus.n_records >= 3000)]
    cand = cand[cand.file_path.map(os.path.exists)]
    cand = cand.sort_values("n_records", ascending=False)
    log = lambda m: print(m, flush=True)
    log(f"count-kind candidates: {len(cand)}")

    hosts, seen_fam = [], set()
    for _, r in cand.iterrows():
        if len(hosts) >= 3 or r.dataset_family in seen_fam:
            continue
        try:
            ext = os.path.splitext(r.archive_member if isinstance(r.archive_member, str)
                                   and r.archive_member else r.file_path)[1].lower()
            df, _ = lib.read_table(r.file_path, r.archive_member if isinstance(r.archive_member, str) else None,
                                   r.sheet_or_table if isinstance(r.sheet_or_table, str) else None, ext, SEED)
            if df is None or r.column_name not in df.columns:
                continue
            vals, _ = lib.coerce_ints(df[r.column_name], False)
        except Exception:
            continue
        if len(vals) < 3000 or np.median(vals) < 1e5:
            continue
        rng = np.random.default_rng([SEED, zlib.crc32(r.dataset_id.encode())])
        if len(vals) > 8000:
            vals = rng.choice(vals, size=8000, replace=False)
        # eligibility: 2*5-free core >= 300
        core25 = np.array([g15.deround_int(int(v)) for v in vals])
        elig = core25 >= 300
        if elig.sum() < 500:
            log(f"  skip {r.dataset_id}: eligible {elig.sum()}")
            continue
        hosts.append(dict(rec=r, vals=vals, elig=elig, core25=core25))
        seen_fam.add(r.dataset_family)
        log(f"  host {len(hosts)}: {r.dataset_id} ({r.domain}) n={len(vals)} "
            f"eligible={int(elig.sum())}")
    pd.DataFrame([dict(dataset_id=h["rec"].dataset_id, domain=h["rec"].domain,
                       family=h["rec"].dataset_family, n=len(h["vals"]),
                       eligible=int(h["elig"].sum())) for h in hosts]
                 ).to_csv(os.path.join(T, "count26_hosts.csv"), index=False)

    spike_rows, conf_rows = [], []
    for h in hosts:
        vals, elig, core25 = h["vals"], h["elig"], h["core25"]
        did = h["rec"].dataset_id
        rng = np.random.default_rng([SEED, 7, zlib.crc32(did.encode())])
        uniq, cnt, h2, cd = value_core_stats(vals)
        idx_of = {int(v): i for i, v in enumerate(uniq.tolist())}
        # f=0 bootstrap of keff_core (R=200, n-matched, record-level)
        boots = []
        for _ in range(200):
            pick = rng.integers(0, len(vals), len(vals))
            bc = np.bincount([idx_of[int(v)] for v in vals[pick]], minlength=len(uniq))
            boots.append(keff_core_from(bc, h2, cd, cm))
        q95 = float(np.nanpercentile(boots, 95))
        base_keff = keff_core_from(cnt, h2, cd, cm)
        log(f"{did}: keff_core={base_keff:.4f} q95={q95:.4f}")
        # host coords bootstrap for the confound check
        host_coords = np.array([coords11(vals[rng.integers(0, len(vals), len(vals))])
                                for _ in range(200)])
        elig_idx = np.where(elig)[0]
        for f in CFG["mixture_grid"]:
            k = int(f * len(vals))
            det = []
            for rep in range(CFG["replicates"]):
                rrep = np.random.default_rng([SEED, 11, zlib.crc32(did.encode()), rep, int(f * 1000)])
                take = rrep.choice(elig_idx, size=min(k, len(elig_idx)), replace=False)
                mix = vals.copy()
                for i in take:
                    G = int(vals[i] // core25[i])
                    t = math.log10(core25[i])
                    mix[i] = G * balanced_product(rrep, t)
                u2, c2_, hh2, cdd = value_core_stats(mix)
                det.append(keff_core_from(c2_, hh2, cdd, cm))
            med = float(np.nanmedian(det))
            spike_rows.append(dict(dataset_id=did, f=f, keff_core_median=med,
                                   q95=q95, recovered=med > q95))
            log(f"  f={f}: median keff_core {med:.4f} recovered={med > q95}")
            if f == 0.10:
                rrep = np.random.default_rng([SEED, 13, zlib.crc32(did.encode())])
                take = rrep.choice(elig_idx, size=min(k, len(elig_idx)), replace=False)
                mix = vals.copy()
                for i in take:
                    G = int(vals[i] // core25[i])
                    mix[i] = G * balanced_product(rrep, math.log10(core25[i]))
                z = (coords11(mix) - host_coords.mean(0)) / (host_coords.std(0) + 1e-12)
                conf_rows.append(dict(dataset_id=did, max_abs_z=float(np.abs(z).max()),
                                      **{f"z{i}": float(v) for i, v in enumerate(z)}))
                log(f"  confound max|z| = {np.abs(z).max():.2f}")
    pd.DataFrame(spike_rows).to_csv(os.path.join(T, "count26_spikein.csv"), index=False)
    pd.DataFrame(conf_rows).to_csv(os.path.join(T, "count26_confound.csv"), index=False)
    log("written count26_*.csv")


if __name__ == "__main__":
    main()
