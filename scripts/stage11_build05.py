"""Build 05: decimalization as feature + label-free pattern discovery.

Stages (python stage11_build05.py <stage>):
  pass       — one corpus pass: per-channel decimalization stats (from RAW
               stored values, pre-coercion) + true de-rounded core stats
               incl. per-record H2' (14 workers, memoized factorizer)
  baseline   — de-rounded baseline v2 with E[H2'|d] (seed 20260703 — same
               draws as Build 03's, new columns) -> baseline_derounded_v2.csv
  features   — decimalization.csv, scale-invariance check, shape/encoding
               feature blocks (channel + family), dL2' canonicalized
  discovery  — PCA/UMAP/HDBSCAN/gap/TwoNN intrinsic dim, IsolationForest+LOF
               anomalies (NO labels), then Stage-4 interpretation (labels
               ONLY here) + read-out
  all        — everything in order

GEOMETRY HYGIENE: stages pass/baseline/features/discovery-embedding use no
labels. domain / generating_process / corpus_role / decimals overlays enter
only in Stage 4, after discovery, for interpretation and coloring.
"""
import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT, ext_of
import build02_lib as lib

DIAG = os.path.join(OUT, "diagnostics")
FROZEN = os.path.join(OUT, "frozen")
INTER = os.path.join(DIAG, "intermediate02")
PARENT_RUN = "lpg01-0643fff9"
SUBRUN = "05"
SEED = 20260705
SEED_DER = 20260703          # reuse Build-03 baseline draws, add H2'
BASELINE_N = 200_000
WORKERS = 14
LN2, LN3, LN5 = math.log(2), math.log(3), math.log(5)

SHAPE_COLS = ["dL1_der", "dL2_der", "dTail_der", "H2_der", "keff_der",
              "L1_der", "L2_der"]
ENC_COLS = ["decimals", "tick_log10", "rounding_mass", "v3_share",
            "mean_log10_digits", "sd_log10_digits"]

SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
GRAY = "#898781"


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE_C)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


# ================================================== stage: corpus pass
def detect_decimals(series):
    """(decimals, tick, native_integer) from RAW stored values."""
    s = series.dropna()
    if len(s) > 20000:
        s = s.iloc[:20000]
    if len(s) == 0:
        return 0, np.nan, True
    if pd.api.types.is_integer_dtype(s):
        vals = np.abs(s.to_numpy(dtype=np.int64, na_value=0))
        vals = vals[vals > 0]
        g = int(np.gcd.reduce(vals)) if len(vals) else 0
        return 0, float(g) if g else np.nan, True
    if pd.api.types.is_numeric_dtype(s):
        v = np.abs(s.to_numpy(dtype=float))
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return 0, np.nan, True
        decs = np.full(len(v), 9, dtype=int)
        for k in range(0, 9):
            scaled = v * (10.0 ** k)
            ok = np.abs(scaled - np.rint(scaled)) < 1e-6 * np.maximum(
                1.0, np.abs(scaled))
            decs = np.where((decs == 9) & ok, k, decs)
        decs = decs[decs < 9]
        if len(decs) == 0:
            return 0, np.nan, False
        d = int(Counter(decs.tolist()).most_common(1)[0][0])
        scaled = np.rint(v * (10.0 ** d))
        scaled = scaled[(scaled > 0) & (scaled < 9e15)].astype(np.int64)
        g = int(np.gcd.reduce(scaled)) if len(scaled) else 0
        return d, (g / 10.0 ** d) if g else np.nan, d == 0
    # strings: read decimal places from the literal representation
    t = s.astype(str).str.strip().str.replace(r"^[\$£€¥]", "", regex=True) \
        .str.replace(",", "", regex=False)
    m = t.str.extract(r"\.(\d+)\s*$")[0]
    dec_len = m.str.len().fillna(0).astype(int)
    num = pd.to_numeric(t, errors="coerce")
    ok = num.notna()
    if ok.sum() == 0:
        return 0, np.nan, True
    d = int(dec_len[ok].mode().iloc[0])
    v = np.abs(num[ok].to_numpy(dtype=float))
    scaled = np.rint(v * (10.0 ** d))
    scaled = scaled[(scaled > 0) & (scaled < 9e15)].astype(np.int64)
    g = int(np.gcd.reduce(scaled)) if len(scaled) else 0
    return d, (g / 10.0 ** d) if g else np.nan, d == 0


def core_stats(ints):
    """De-rounded core stats incl. per-record H2' (means over records)."""
    n_used = len(ints)
    if n_used == 0:
        return None
    uniq, counts = np.unique(ints, return_counts=True)
    sums = np.zeros(4)          # L1', L2', Tail', H2'
    rm_sum = 0.0
    n_core = n_core1 = 0
    for v, c in zip(uniq.tolist(), counts.tolist()):
        pairs = lib.factor_pairs(v)
        ln_n = math.log(v)
        a = b = 0
        contrib = []
        for p, e in pairs:
            if p == 2:
                a = e
            elif p == 5:
                b = e
            else:
                contrib.append(e * math.log(p))
        rm_sum += ((a * LN2 + b * LN5) / ln_n) * c
        ln_core = ln_n - (a * LN2 + b * LN5)
        if not contrib or ln_core <= 1e-12:
            n_core1 += c
            continue
        contrib.sort(reverse=True)
        L = [x / ln_core for x in contrib]
        h2 = sum(x * x for x in L)
        sums += np.array([L[0], L[1] if len(L) > 1 else 0.0,
                          1.0 - L[0] - (L[1] if len(L) > 1 else 0.0),
                          h2]) * c
        n_core += c
    out = dict(n_used=int(n_used), rounding_mass=rm_sum / n_used,
               frac_core1=n_core1 / n_used)
    if n_core:
        out.update(L1_der=sums[0] / n_core, L2_der=sums[1] / n_core,
                   Tail_der=sums[2] / n_core, H2_der=sums[3] / n_core)
    else:
        out.update(L1_der=np.nan, L2_der=np.nan, Tail_der=np.nan,
                   H2_der=np.nan)
    return out


def _pass_task(task):
    out, errors, mon_sample = [], [], []
    for meta, channels in task:
        fp, member, sheet, ext, seed = meta
        want = [ch["column_name"] for ch in channels]
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from stage9_build03 import _read_with_fallback
            df, sampled = _read_with_fallback(fp, member, sheet, ext, seed,
                                              want)
        except Exception as e:
            for ch in channels:
                errors.append((fp, member, ch["column_name"], repr(e)[:300]))
            continue
        for ch in channels:
            col = ch["column_name"]
            if col not in df.columns:
                errors.append((fp, member, col, "column missing"))
                continue
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            try:
                dec, tick, native = detect_decimals(series)
                ints, _ = lib.coerce_ints(series, ch["monetary"])
                prof = core_stats(ints)
            except Exception as e:
                errors.append((fp, member, col, repr(e)[:300]))
                continue
            row = dict(dataset_id=ch["dataset_id"], decimals=dec,
                       tick=tick, native_integer=bool(native),
                       scale_applied=100 if ch["monetary"] else 1)
            if prof:
                row.update(prof)
            out.append(row)
            if ch["monetary"] and len(mon_sample) < 800 and prof:
                nz = series.dropna()
                take = pd.to_numeric(nz.astype(str).str.replace(
                    r"^[\$£€¥]", "", regex=True).str.replace(
                    ",", "", regex=False), errors="coerce").dropna()
                mon_sample.extend(np.abs(take.to_numpy(dtype=float))
                                  [:40].tolist())
    return out, errors, mon_sample


def corpus_pass():
    all_mon = []
    for fname, out_pkl in [("observational_corpus.csv", "obs_v5.pkl"),
                           ("synthetic_control.csv", "syn_v5.pkl")]:
        dd = pd.read_csv(os.path.join(FROZEN, fname), low_memory=False)
        for c in ("archive_member", "sheet_or_table"):
            dd[c] = dd[c].fillna("")
        dd["monetary"] = (dd.channel_kind == "amount") | \
            dd.looks_monetary.astype(bool)
        tables = []
        for (fp, member, sheet), g in dd.groupby(
                ["file_path", "archive_member", "sheet_or_table"], sort=True):
            seed = int(hashlib.md5(f"{fp}|{member}".encode())
                       .hexdigest()[:8], 16)
            channels = [dict(dataset_id=r.dataset_id,
                             column_name=r.column_name,
                             monetary=bool(r.monetary))
                        for r in g.itertuples()]
            tables.append(((fp, member, sheet, ext_of(member or fp), seed),
                           channels))
        tasks, cur = [], []
        for t in tables:
            cur.append(t)
            if len(cur) >= 40:
                tasks.append(cur)
                cur = []
        if cur:
            tasks.append(cur)
        print(f"[{fname}] {len(dd):,} channels, {len(tasks)} tasks, "
              f"{WORKERS} workers", flush=True)
        results, errors = [], []
        with ProcessPoolExecutor(max_workers=WORKERS,
                                 initializer=lib.init_worker) as pool:
            for i, (rows, errs, mon) in enumerate(
                    pool.map(_pass_task, tasks)):
                results.extend(rows)
                errors.extend(errs)
                all_mon.extend(mon)
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(tasks)}", flush=True)
        with open(os.path.join(INTER, out_pkl), "wb") as f:
            pickle.dump(results, f)
        print(f"[{fname}] done: {len(results):,} channels, "
              f"{len(errors)} errors", flush=True)
    with open(os.path.join(INTER, "monetary_sample_v5.pkl"), "wb") as f:
        pickle.dump(all_mon[:10000], f)


# ================================================== stage: baseline v2
def _baseline_task(args):
    d, chunk_idx, n = args
    lo = max(2, 10 ** (d - 1))
    rng = np.random.default_rng([SEED_DER, d, chunk_idx])
    vals = rng.integers(lo, 10 ** d, size=n, dtype=np.int64)
    sums = np.zeros(4)
    sq = np.zeros(4)
    n_core = n_core1 = 0
    for v in vals.tolist():
        pairs = lib.factor_pairs(v)
        ln_n = math.log(v)
        a = b = 0
        contrib = []
        for p, e in pairs:
            if p == 2:
                a = e
            elif p == 5:
                b = e
            else:
                contrib.append(e * math.log(p))
        ln_core = ln_n - (a * LN2 + b * LN5)
        if not contrib or ln_core <= 1e-12:
            n_core1 += 1
            continue
        contrib.sort(reverse=True)
        L = [x / ln_core for x in contrib]
        h2 = sum(x * x for x in L)
        x = np.array([L[0], L[1] if len(L) > 1 else 0.0,
                      1.0 - L[0] - (L[1] if len(L) > 1 else 0.0), h2])
        sums += x
        sq += x * x
        n_core += 1
    return d, n, sums, sq, n_core, n_core1


def baseline_v2():
    D, chunks = 18, 10
    per = BASELINE_N // chunks
    tasks = [(d, c, per) for d in range(1, D + 1) for c in range(chunks)]
    acc = {d: [0, np.zeros(4), np.zeros(4), 0, 0] for d in range(1, D + 1)}
    print(f"baseline v2 (adds H2'): {D}×{BASELINE_N:,}, seed={SEED_DER}",
          flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=lib.init_worker) as pool:
        for d, n, sums, sq, nc, nc1 in pool.map(_baseline_task, tasks):
            a = acc[d]
            a[0] += n
            a[1] += sums
            a[2] += sq
            a[3] += nc
            a[4] += nc1
    rows = []
    for d in range(1, D + 1):
        n, sums, sq, nc, nc1 = acc[d]
        mean = sums / max(1, nc)
        sd = np.sqrt(np.maximum(sq / max(1, nc) - mean ** 2, 0))
        rows.append(dict(stratum=d, n_draws=n, n_core_valid=nc,
                         frac_core1=nc1 / n,
                         E_L1_der=mean[0], E_L2_der=mean[1],
                         E_Tail_der=mean[2], E_H2_der=mean[3],
                         sd_L1_der=sd[0], sd_L2_der=sd[1],
                         sd_Tail_der=sd[2], sd_H2_der=sd[3]))
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "baseline_derounded_v2.csv"), index=False,
        encoding="utf-8")
    print("baseline_derounded_v2.csv written")


# ================================================== stage: features
def features():
    lib.init_worker()
    chp = pd.read_csv(os.path.join(TABLES, "channel_profiles_v3.csv"),
                      low_memory=False)
    with open(os.path.join(INTER, "obs_v5.pkl"), "rb") as f:
        v5o = pd.DataFrame(pickle.load(f))
    with open(os.path.join(INTER, "syn_v5.pkl"), "rb") as f:
        v5s = pd.DataFrame(pickle.load(f))
    with open(os.path.join(INTER, "obs_raw.pkl"), "rb") as f:
        dh = {r["dataset_id"]: r["digit_hist"]
              for r in pickle.load(f)["results"]}
    with open(os.path.join(INTER, "syn_raw.pkl"), "rb") as f:
        dh.update({r["dataset_id"]: r["digit_hist"]
                   for r in pickle.load(f)["results"]})
    bln = pd.read_csv(os.path.join(TABLES, "baseline_derounded_v2.csv")) \
        .sort_values("stratum")
    labels = pd.read_csv(os.path.join(FROZEN, "ml_labels.csv"),
                         low_memory=False)
    mlf = pd.read_csv(os.path.join(FROZEN, "ml_features.csv"),
                      low_memory=False)

    # ---- decimalization table (one row per channel, obs + synthetic)
    dec = pd.concat([v5o, v5s], ignore_index=True)
    dec_out = dec[["dataset_id", "decimals", "tick", "native_integer",
                   "scale_applied"]].rename(
        columns={"dataset_id": "channel_id"}).sort_values("channel_id")
    dec_out.to_csv(os.path.join(TABLES, "decimalization.csv"), index=False,
                   encoding="utf-8")

    # ---- invariance check
    with open(os.path.join(INTER, "monetary_sample_v5.pkl"), "rb") as f:
        mon = pickle.load(f)
    rng = np.random.default_rng(SEED)
    mon = [m for m in mon if np.isfinite(m) and m > 0][:10000]
    ok = tot = 0
    for v in mon:
        d = 0
        while d < 6 and abs(v * 10 ** d - round(v * 10 ** d)) > 1e-6:
            d += 1
        base = int(round(v * 10 ** d))
        if base <= 1 or base > 10 ** 14:
            continue

        def core(n):
            return tuple(sorted((p, e) for p, e in lib.factor_pairs(n)
                                if p not in (2, 5)))
        # invariance claim: the 2·5-free core is unchanged under any 10^k
        # scaling — ×1 (10^decimals-integerized), ×10, ×100
        variants = [core(base), core(base * 10), core(base * 100)]
        tot += 1
        ok += int(all(x == variants[0] for x in variants))
    inv_pass = (ok == tot and tot > 0)
    print(f"invariance check: {ok}/{tot} monetary records have identical "
          f"2·5-free cores under ×1/×100/×10^decimals -> "
          f"{'PASS' if inv_pass else 'FAIL'}")

    # ---- scale-clean shape block (channel level)
    keep = dec.merge(labels.rename(columns={"channel_id": "dataset_id"}),
                     on="dataset_id", how="left")
    hists = [np.asarray(dh.get(i, [1]), dtype=float) for i in keep.dataset_id]
    D = len(bln)
    W = np.vstack([h[:D] / max(1, h.sum()) for h in
                   [np.pad(h, (0, max(0, D - len(h)))) for h in hists]])
    for col, ecol in [("L1_der", "E_L1_der"), ("L2_der", "E_L2_der"),
                      ("Tail_der", "E_Tail_der"), ("H2_der", "E_H2_der")]:
        keep[f"null_{col}"] = W @ bln[ecol].to_numpy()
    keep["dL1_der"] = keep.L1_der - keep.null_L1_der
    keep["dL2_der"] = keep.L2_der - keep.null_L2_der
    keep["dTail_der"] = keep.Tail_der - keep.null_Tail_der
    keep["keff_der"] = keep.null_H2_der / keep.H2_der
    keep["mean_log10_digits"] = W @ (np.arange(1, D + 1) - 0.5)
    dvals = np.arange(1, D + 1) - 0.5
    keep["sd_log10_digits"] = np.sqrt(np.maximum(
        W @ (dvals ** 2) - (W @ dvals) ** 2, 0))
    keep["tick_log10"] = np.log10(keep.tick.replace(0, np.nan))
    keep["v3_share"] = keep.dataset_id.map(
        dict(zip(mlf.channel_id, mlf.v3_share)))

    shape = keep[["dataset_id", "dataset_family"] + SHAPE_COLS].rename(
        columns={"dataset_id": "channel_id"}).sort_values("channel_id")
    shape.round(6).to_csv(os.path.join(FROZEN, "shape_features.csv"),
                          index=False, encoding="utf-8")
    enc = keep[["dataset_id", "dataset_family"] + ENC_COLS].rename(
        columns={"dataset_id": "channel_id"}).sort_values("channel_id")
    enc.round(6).to_csv(os.path.join(FROZEN, "encoding_features.csv"),
                        index=False, encoding="utf-8")

    fam_shape = keep.groupby("dataset_family")[SHAPE_COLS].mean() \
        .reset_index().sort_values("dataset_family")
    fam_shape.round(6).to_csv(
        os.path.join(FROZEN, "shape_features_family.csv"), index=False,
        encoding="utf-8")
    fam_enc = keep.groupby("dataset_family")[ENC_COLS].mean() \
        .reset_index().sort_values("dataset_family")
    fam_enc.round(6).to_csv(
        os.path.join(FROZEN, "encoding_features_family.csv"), index=False,
        encoding="utf-8")

    # ---- read-out item 2: raw dL2 vs canonical dL2'
    ch3 = chp[["dataset_id", "dL2", "channel_kind"]].merge(
        keep[["dataset_id", "dL2_der"]], on="dataset_id")
    mon_k = ch3.channel_kind == "amount"
    cnt_k = ch3.channel_kind == "count"
    print("\nraw dL2 vs scale-clean dL2' (mean ± sd):")
    for name, mask in [("amount(monetary)", mon_k), ("count", cnt_k)]:
        s = ch3[mask]
        print(f"  {name:<18} raw dL2 {s.dL2.mean():+.4f}±{s.dL2.std():.4f}"
              f"   dL2' {s.dL2_der.mean():+.4f}±{s.dL2_der.std():.4f}")
    return inv_pass


# ================================================== stage: discovery
def discovery():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.metrics import adjusted_mutual_info_score as ami
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    FOOT = (f"run {PARENT_RUN} · sub-run {SUBRUN} · geometry: scale-clean "
            "shape features only, no labels; overlays applied post-hoc")

    fam_shape = pd.read_csv(os.path.join(FROZEN,
                                         "shape_features_family.csv"))
    fam_enc = pd.read_csv(os.path.join(FROZEN,
                                       "encoding_features_family.csv"))
    labels = pd.read_csv(os.path.join(FROZEN, "ml_labels.csv"),
                         low_memory=False)
    def safe_mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else ""

    fam_lab = labels.groupby("dataset_family").agg(
        domain=("domain", "first"),
        generating_process=("generating_process", safe_mode),
        corpus_role=("corpus_role", "first")).reset_index()

    f = fam_shape.dropna(subset=SHAPE_COLS).reset_index(drop=True)
    X = f[SHAPE_COLS].to_numpy(dtype=float)
    Xs = (X - X.mean(0)) / X.std(0)
    print(f"discovery on {len(f)} families × {len(SHAPE_COLS)} scale-clean "
          f"shape coords (no labels)", flush=True)

    # ---- PCA
    pca = PCA(n_components=4, random_state=SEED)
    P = pca.fit_transform(Xs)
    load = pd.DataFrame(pca.components_.T, index=SHAPE_COLS,
                        columns=[f"PC{i+1}" for i in range(4)])
    load.round(4).to_csv(os.path.join(TABLES, "discovery_pca_loadings.csv"),
                         encoding="utf-8")
    evr = pca.explained_variance_ratio_

    # ---- UMAP
    NN, MD = 20, 0.1
    um = umap.UMAP(n_neighbors=NN, min_dist=MD, n_components=2,
                   random_state=SEED)
    U = um.fit_transform(Xs)

    # ---- HDBSCAN grid + bootstrap stability
    grid_rows, best = [], None
    for mcs in (5, 8, 10, 15, 20):
        lab = HDBSCAN(min_cluster_size=mcs).fit_predict(Xs)
        k = len(set(lab)) - (1 if -1 in lab else 0)
        noise = float(np.mean(lab == -1))
        rng = np.random.default_rng([SEED, mcs])
        aris = []
        for _ in range(50):
            idx = rng.choice(len(Xs), int(0.8 * len(Xs)), replace=False)
            lb = HDBSCAN(min_cluster_size=mcs).fit_predict(Xs[idx])
            m = (lab[idx] != -1) & (lb != -1)
            if m.sum() > 10 and len(set(lb[m])) > 1 and \
                    len(set(lab[idx][m])) > 1:
                aris.append(adjusted_rand_score(lab[idx][m], lb[m]))
        stab = float(np.mean(aris)) if aris else np.nan
        grid_rows.append(dict(min_cluster_size=mcs, clusters=k,
                              noise_frac=noise, bootstrap_stability=stab))
        score = (stab if not np.isnan(stab) else 0) * (k >= 2)
        if best is None or score > best[0]:
            best = (score, mcs, lab)
    _, mcs_best, lab_h = best
    pd.DataFrame(grid_rows).round(4).to_csv(
        os.path.join(TABLES, "discovery_hdbscan_grid.csv"), index=False,
        encoding="utf-8")

    # ---- gap statistic
    rng = np.random.default_rng(SEED)
    lo, hi = Xs.min(0), Xs.max(0)
    gaps, sks, wks = [], [], []
    for k in range(1, 13):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        wk = np.log(km.inertia_)
        ref = []
        for b in range(20):
            R = rng.uniform(lo, hi, Xs.shape)
            ref.append(np.log(KMeans(n_clusters=k, n_init=4,
                                     random_state=b).fit(R).inertia_))
        gaps.append(np.mean(ref) - wk)
        sks.append(np.std(ref) * np.sqrt(1 + 1 / 20))
        wks.append(wk)
    gap_k = next((k for k in range(1, 12)
                  if gaps[k - 1] >= gaps[k] - sks[k]), 12)

    # ---- intrinsic dimension (TwoNN, Facco et al.)
    nn = NearestNeighbors(n_neighbors=3).fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    r1, r2 = dist[:, 1], dist[:, 2]
    m = (r1 > 1e-9) & (r2 > r1)
    mu = r2[m] / r1[m]
    id_twonn = float(len(mu) / np.sum(np.log(mu)))

    # ---- anomalies (label-free)
    iso = IsolationForest(n_estimators=500, random_state=SEED,
                          n_jobs=WORKERS).fit(Xs)
    iso_s = -iso.score_samples(Xs)
    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit(Xs)
    lof_s = -lof.negative_outlier_factor_
    rank = pd.DataFrame({
        "dataset_family": f.dataset_family,
        "iso_score": iso_s, "lof_score": lof_s,
        "combined_rank": (pd.Series(iso_s).rank(ascending=False) +
                          pd.Series(lof_s).rank(ascending=False)) / 2})
    rank = rank.sort_values("combined_rank")

    # ---------- Stage 4: labels enter ONLY from here ----------
    f2 = f.merge(fam_lab, on="dataset_family", how="left") \
        .merge(fam_enc, on="dataset_family", how="left")
    f2["cluster"] = lab_h
    f2["PC1"], f2["PC2"] = P[:, 0], P[:, 1]
    f2["UMAP1"], f2["UMAP2"] = U[:, 0], U[:, 1]

    f2[["dataset_family", "cluster", "PC1", "PC2", "UMAP1", "UMAP2"]] \
        .sort_values("dataset_family").round(5).to_csv(
        os.path.join(TABLES, "discovery_clusters.csv"), index=False,
        encoding="utf-8")
    anom = rank.merge(fam_lab, on="dataset_family", how="left").merge(
        fam_enc[["dataset_family", "decimals", "rounding_mass"]],
        on="dataset_family", how="left")
    anom.round(4).to_csv(os.path.join(TABLES, "discovery_anomalies.csv"),
                         index=False, encoding="utf-8")

    # overlay AMI per cluster structure
    def binify(x, q=4):
        return pd.qcut(x, q, labels=False, duplicates="drop")

    overlays = {
        "domain": f2.domain.astype(str),
        "generating_process": f2.generating_process.astype(str),
        "corpus_role": f2.corpus_role.astype(str),
        "decimals": f2.decimals.round().astype("Int64").astype(str),
        "rounding_mass_q": binify(f2.rounding_mass).astype(str),
        "magnitude_q": binify(f2.mean_log10_digits).astype(str),
    }
    mask = f2.cluster != -1
    ami_rows = [dict(overlay=name,
                     ami_vs_clusters=ami(overlays[name][mask],
                                         f2.cluster[mask]))
                for name in overlays]
    per_cl = []
    for cl, g in f2[mask].groupby("cluster"):
        per_cl.append(dict(
            cluster=int(cl), families=len(g),
            top_domain=g.domain.mode().iloc[0],
            domain_purity=float((g.domain == g.domain.mode().iloc[0])
                                .mean()),
            modal_decimals=int(g.decimals.round().mode().iloc[0]),
            mean_rounding_mass=float(g.rounding_mass.mean()),
            mean_log10=float(g.mean_log10_digits.mean()),
            synthetic_frac=float((g.corpus_role == "synthetic_control")
                                 .mean())))
    pd.concat([pd.DataFrame(ami_rows), pd.DataFrame(per_cl)],
              axis=0, ignore_index=True).round(4).to_csv(
        os.path.join(TABLES, "pattern_interpretation.csv"), index=False,
        encoding="utf-8")

    # ---------- figures ----------
    def scatter(ax, xy, cvals, cmapping, size=20):
        for key, colr in cmapping.items():
            m2 = cvals == key
            ax.scatter(xy[m2, 0], xy[m2, 1], s=size, color=colr,
                       edgecolors=SURFACE, linewidths=0.5, zorder=3)

    # PCA biplot
    fig, ax = plt.subplots(figsize=(7.8, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    cl_colors = {cl: (GRAY if cl == -1 else PALETTE[cl % len(PALETTE)])
                 for cl in sorted(set(lab_h))}
    scatter(ax, P[:, :2], pd.Series(lab_h), cl_colors)
    sc = 2.8
    for feat in SHAPE_COLS:
        vx, vy = load.loc[feat, "PC1"] * sc, load.loc[feat, "PC2"] * sc
        ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
        ax.text(vx * 1.12, vy * 1.12, feat, fontsize=7.5, color=INK,
                ha="center")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var)", fontsize=9, color=MUTED)
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})", fontsize=9, color=MUTED)
    ax.set_title("PCA of scale-clean shape space (color = HDBSCAN cluster, "
                 "arrows = loadings)", fontsize=10.5, color=INK,
                 loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig05_pca_embedding.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # UMAP by cluster (and the dedicated hdbscan figure)
    for fname, ttl in [("fig05_umap_embedding.png",
                        f"UMAP (n_neighbors={NN}, min_dist={MD}) — color = "
                        f"HDBSCAN cluster (mcs={mcs_best})"),
                       ("fig05_hdbscan_clusters.png",
                        f"HDBSCAN clusters in UMAP plane (mcs={mcs_best}, "
                        "gray = noise)")]:
        fig, ax = plt.subplots(figsize=(7.8, 6.2), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        style_ax(ax)
        scatter(ax, U, pd.Series(lab_h), cl_colors)
        if "hdbscan" in fname:
            for cl in sorted(set(lab_h)):
                if cl == -1:
                    continue
                cx, cy = U[lab_h == cl].mean(0)
                ax.text(cx, cy, str(cl), fontsize=11, color=INK,
                        weight="bold", ha="center", va="center", zorder=5)
        ax.set_xlabel("UMAP1", fontsize=9, color=MUTED)
        ax.set_ylabel("UMAP2", fontsize=9, color=MUTED)
        ax.set_title(ttl, fontsize=10.5, color=INK, loc="left", pad=12)
        fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(DIAG, fname), facecolor=SURFACE)
        plt.close(fig)

    # overlay recolors
    dom_top = f2.domain.value_counts().index.tolist()[:7]
    dom_cmap = {d: PALETTE[i] for i, d in enumerate(dom_top)}

    def overlay_fig(fname, series, title, categorical, cmap_dict=None):
        fig, ax = plt.subplots(figsize=(7.8, 6.2), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        style_ax(ax)
        if categorical:
            keys = series.value_counts().index.tolist()
            for k in keys:
                m2 = (series == k).to_numpy()
                c = cmap_dict.get(k, GRAY) if cmap_dict else GRAY
                ax.scatter(U[m2, 0], U[m2, 1], s=20, color=c,
                           edgecolors=SURFACE, linewidths=0.5, zorder=3,
                           label=str(k) if (cmap_dict and k in cmap_dict)
                           else None)
            ax.legend(fontsize=7, frameon=False, loc="best",
                      labelcolor=INK)
        else:
            sc2 = ax.scatter(U[:, 0], U[:, 1], s=20,
                             c=series.to_numpy(dtype=float),
                             cmap="viridis", edgecolors=SURFACE,
                             linewidths=0.4, zorder=3)
            fig.colorbar(sc2, ax=ax, shrink=0.75)
        ax.set_xlabel("UMAP1", fontsize=9, color=MUTED)
        ax.set_ylabel("UMAP2", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=12)
        fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(DIAG, fname), facecolor=SURFACE)
        plt.close(fig)

    overlay_fig("fig05_embedding_by_domain.png", f2.domain,
                "UMAP overlay: domain (top 7 colored)", True, dom_cmap)
    overlay_fig("fig05_embedding_by_decimals.png",
                f2.decimals.round().astype(int).astype(str),
                "UMAP overlay: detected decimals", True,
                {"0": PALETTE[0], "1": PALETTE[2], "2": PALETTE[5],
                 "3": PALETTE[4]})
    overlay_fig("fig05_embedding_by_roundingmass.png", f2.rounding_mass,
                "UMAP overlay: rounding_mass", False)
    overlay_fig("fig05_embedding_by_magnitude.png", f2.mean_log10_digits,
                "UMAP overlay: mean log10 magnitude", False)

    # ---------- read-out ----------
    k_h = len(set(lab_h)) - (1 if -1 in lab_h else 0)
    noise = float(np.mean(lab_h == -1))
    print("\n================ BUILD 05 READ-OUT (discovery) ============")
    print(f"3. PCA explained variance PC1-4: "
          + ", ".join(f"{v:.1%}" for v in evr))
    print("   PC1 loadings: " + ", ".join(
        f"{c}={load.loc[c, 'PC1']:+.2f}" for c in SHAPE_COLS))
    print("   PC2 loadings: " + ", ".join(
        f"{c}={load.loc[c, 'PC2']:+.2f}" for c in SHAPE_COLS))
    print(f"4. HDBSCAN(mcs={mcs_best}): {k_h} clusters, noise={noise:.1%}, "
          f"bootstrap stability={best[0]:.2f}; gap-statistic k={gap_k}; "
          f"TwoNN intrinsic dimension={id_twonn:.2f}")
    print("5. overlay AMI vs discovered clusters:")
    for r in sorted(ami_rows, key=lambda r: -r["ami_vs_clusters"]):
        print(f"   {r['overlay']:<20} AMI={r['ami_vs_clusters']:.3f}")
    print("   per-cluster interpretation -> pattern_interpretation.csv")
    print("6. top 10 blind anomalies:")
    for r in anom.head(10).itertuples():
        print(f"   {r.dataset_family:<52} {str(r.domain):<20} "
              f"iso={r.iso_score:.3f} lof={r.lof_score:.2f}")
    return ami_rows, anom


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    with open(os.path.join(CONFIG, "run_config_05.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN, "seed": SEED,
            "workers": WORKERS,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "decimals": "detected from raw stored values (string decimal "
                        "places, or float with 1e-6 relative tolerance); "
                        "tick = gcd of values scaled by 10^decimals",
            "shape_block": SHAPE_COLS,
            "encoding_block": ENC_COLS,
            "dL2_der": "canonical de-rounded L2 residual vs "
                       "baseline_derounded_v2 (raw dL2 retired from the "
                       "process space)",
            "baseline_v2": f"seed {SEED_DER} (identical draws to Build 03) "
                           "+ E[H2'|d]; keff' = H2'_null_w / H2'_obs",
            "umap": {"n_neighbors": 20, "min_dist": 0.1},
            "geometry_hygiene": "no labels in pass/baseline/features/"
                                "embedding/clustering/anomaly stages; "
                                "overlays post-hoc only",
        }, f, indent=2)
    if stage in ("pass", "all"):
        corpus_pass()
    if stage in ("baseline", "all"):
        baseline_v2()
    if stage in ("features", "all"):
        features()
    if stage in ("discovery", "all"):
        discovery()


if __name__ == "__main__":
    main()
