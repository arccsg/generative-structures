"""Build 03: de-rounding decomposition, confound tests, first topology.

Stages (python stage9_build03.py <stage>):
  deround      — one pass over frozen corpora: per record split n = 2^a 5^b * core,
                 rounding_mass, v3 share, de-rounded core L-profile (14 workers)
  baseline_der — de-rounded baseline per ORIGINAL digit stratum (fresh, seeded)
  profiles     — channel_profiles_v3 / family_profiles_v3 + arm decomposition
                 (fig03_arm_before_after, fig03_roundingmass_vs_dtail,
                 derounding_effect.csv)
  topology     — HDBSCAN + Ward in raw vs de-rounded residuals (+rounding_mass
                 variant), magnitude bands, bootstrap CIs, ML feature/label split
  all          — everything in order

GEOMETRY HYGIENE: residuals/clustering/projections use only L-profile-derived
coordinates. domain / dataset_family / channel_kind / generating_process /
corpus_role never enter any distance, clustering, or projection — they are
attached AFTER clustering for coloring and composition reporting only.

Factorization trick: factorizing original n once yields the de-rounded core
profile directly (core factor list = factors(n) minus {2,5};
ln core = ln n - a ln2 - b ln5). Records with core == 1 (pure 2^a 5^b) carry
rounding_mass = 1 and are excluded from core-profile means (fraction logged).
"""
import hashlib
import json
import math
import os
import pickle
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
SUBRUN = "03"
SEED_DER = 20260703          # de-rounded baseline seed
BASELINE_N = 200_000
WORKERS = 14
LN2, LN3, LN5 = math.log(2), math.log(3), math.log(5)

SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
GRAY = "#898781"


# ------------------------------------------------------------- core stats
def deround_values(ints):
    """Per-channel de-rounding stats over an int64 array (values >= 2)."""
    n_used = len(ints)
    if n_used == 0:
        return None
    uniq, counts = np.unique(ints, return_counts=True)
    rm_sum = v3_sum = 0.0
    core_sums = np.zeros(3)          # L1', L2', Tail'
    n_core = 0
    n_core1 = 0
    for v, c in zip(uniq.tolist(), counts.tolist()):
        pairs = lib.factor_pairs(v)
        ln_n = math.log(v)
        a = b = v3 = 0
        core_contrib = []
        for p, e in pairs:
            if p == 2:
                a = e
            elif p == 5:
                b = e
            else:
                if p == 3:
                    v3 = e
                core_contrib.append(e * math.log(p))
        rm = (a * LN2 + b * LN5) / ln_n
        rm_sum += rm * c
        v3_sum += (v3 * LN3 / ln_n) * c
        ln_core = ln_n - (a * LN2 + b * LN5)
        if not core_contrib or ln_core <= 1e-12:
            n_core1 += c
            continue
        core_contrib.sort(reverse=True)
        l1 = core_contrib[0] / ln_core
        l2 = core_contrib[1] / ln_core if len(core_contrib) > 1 else 0.0
        core_sums += np.array([l1, l2, 1.0 - l1 - l2]) * c
        n_core += c
    out = dict(n_used=int(n_used),
               rounding_mass=rm_sum / n_used,
               v3_share=v3_sum / n_used,
               frac_core1=n_core1 / n_used)
    if n_core:
        out.update(L1_der=core_sums[0] / n_core,
                   L2_der=core_sums[1] / n_core,
                   Tail_der=core_sums[2] / n_core)
    else:
        out.update(L1_der=np.nan, L2_der=np.nan, Tail_der=np.nan)
    return out


# --------------------------------------------------------------- deround
def _read_with_fallback(fp, member, sheet, ext, seed, want_cols):
    """Fast reader with the Build-02 rescue paths built in."""
    try:
        df, sampled = lib.read_table(fp, member, sheet, ext, seed)
        if any(c in df.columns for c in want_cols):
            return df, sampled
    except Exception:
        pass
    # rescue 1: quote-aware pandas read
    import zipfile
    kw = dict(nrows=lib.TOTAL_ROWS, on_bad_lines="skip",
              encoding_errors="replace")
    try:
        if member and fp.lower().endswith(".zip"):
            with zipfile.ZipFile(fp) as zf, zf.open(member) as f:
                df = pd.read_csv(f, **kw)
        else:
            df = pd.read_csv(fp, **kw)
        if any(c in df.columns for c in want_cols):
            return df, len(df) >= lib.TOTAL_ROWS
    except Exception:
        pass
    # rescue 2: whitespace-delimited headerless
    kw.update(sep=r"\s+", header=None)
    if member and fp.lower().endswith(".zip"):
        with zipfile.ZipFile(fp) as zf, zf.open(member) as f:
            df = pd.read_csv(f, **kw)
    else:
        df = pd.read_csv(fp, **kw)
    df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df, len(df) >= lib.TOTAL_ROWS


def _deround_task(task):
    out, errors = [], []
    for meta, channels in task:
        fp, member, sheet, ext, seed = meta
        want = [ch["column_name"] for ch in channels]
        try:
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
                ints, _ = lib.coerce_ints(series, ch["monetary"])
                prof = deround_values(ints)
            except Exception as e:
                errors.append((fp, member, col, repr(e)[:300]))
                continue
            if prof is None:
                errors.append((fp, member, col, "no usable integers >= 2"))
                continue
            prof["dataset_id"] = ch["dataset_id"]
            out.append(prof)
    return out, errors


def deround():
    for fname, out_pkl in [("observational_corpus.csv", "obs_der.pkl"),
                           ("synthetic_control.csv", "syn_der.pkl")]:
        dd = pd.read_csv(os.path.join(FROZEN, fname), low_memory=False)
        for c in ("archive_member", "sheet_or_table"):
            dd[c] = dd[c].fillna("")
        dd["monetary"] = (dd.channel_kind == "amount") | \
            dd.looks_monetary.astype(bool)
        tables = []
        for (fp, member, sheet), g in dd.groupby(
                ["file_path", "archive_member", "sheet_or_table"], sort=True):
            seed = int(hashlib.md5(f"{fp}|{member}".encode())
                       .hexdigest()[:8], 16)   # identical to Build 02
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
            for i, (rows, errs) in enumerate(pool.map(_deround_task, tasks)):
                results.extend(rows)
                errors.extend(errs)
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(tasks)} tasks, "
                          f"{len(results):,} channels", flush=True)
        with open(os.path.join(INTER, out_pkl), "wb") as f:
            pickle.dump(results, f)
        pd.DataFrame(errors, columns=["file_path", "archive_member",
                                      "column_name", "error"]).to_csv(
            os.path.join(DIAG, f"deround_errors_{fname[:3]}.csv"),
            index=False, encoding="utf-8")
        print(f"[{fname}] done: {len(results):,} channels, "
              f"{len(errors)} errors", flush=True)


# ---------------------------------------------------------- baseline_der
def _baseline_der_task(args):
    d, chunk_idx, n = args
    lo = max(2, 10 ** (d - 1))
    rng = np.random.default_rng([SEED_DER, d, chunk_idx])
    vals = rng.integers(lo, 10 ** d, size=n, dtype=np.int64)
    sums = np.zeros(3)
    sq = np.zeros(3)
    rm_sum = 0.0
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
        rm_sum += (a * LN2 + b * LN5) / ln_n
        ln_core = ln_n - (a * LN2 + b * LN5)
        if not contrib or ln_core <= 1e-12:
            n_core1 += 1
            continue
        contrib.sort(reverse=True)
        l1 = contrib[0] / ln_core
        l2 = contrib[1] / ln_core if len(contrib) > 1 else 0.0
        x = np.array([l1, l2, 1.0 - l1 - l2])
        sums += x
        sq += x * x
        n_core += 1
    return d, n, sums, sq, n_core, n_core1, rm_sum


def baseline_der():
    D = 18
    chunks = 10
    per = BASELINE_N // chunks
    tasks = [(d, c, per) for d in range(1, D + 1) for c in range(chunks)]
    acc = {d: [0, np.zeros(3), np.zeros(3), 0, 0, 0.0]
           for d in range(1, D + 1)}
    print(f"de-rounded baseline: {D} strata × {BASELINE_N:,}, "
          f"seed={SEED_DER}, {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=lib.init_worker) as pool:
        for d, n, sums, sq, nc, nc1, rm in pool.map(_baseline_der_task,
                                                    tasks):
            a = acc[d]
            a[0] += n
            a[1] += sums
            a[2] += sq
            a[3] += nc
            a[4] += nc1
            a[5] += rm
    rows = []
    for d in range(1, D + 1):
        n, sums, sq, nc, nc1, rm = acc[d]
        mean = sums / max(1, nc)
        sd = np.sqrt(np.maximum(sq / max(1, nc) - mean ** 2, 0))
        rows.append(dict(stratum=d, n_draws=n, n_core_valid=nc,
                         frac_core1=nc1 / n,
                         E_rounding_mass=rm / n,
                         E_L1_der=mean[0], E_L2_der=mean[1],
                         E_Tail_der=mean[2],
                         sd_L1_der=sd[0], sd_L2_der=sd[1], sd_Tail_der=sd[2]))
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "baseline_derounded.csv"), index=False,
        encoding="utf-8")
    print("baseline_derounded.csv written (indexed by ORIGINAL digit "
          "stratum)")


# -------------------------------------------------------------- profiles
def _weighted_null(ch_digit_hists, bln, cols):
    D = len(bln)
    W = np.vstack([np.asarray(h[:D], dtype=float) / max(1, sum(h))
                   for h in ch_digit_hists])
    return {c: W @ bln[c].to_numpy() for c in cols}, W


def profiles():
    chp = pd.read_csv(os.path.join(TABLES, "channel_profiles.csv"),
                      low_memory=False)
    with open(os.path.join(INTER, "obs_der.pkl"), "rb") as f:
        der = pd.DataFrame(pickle.load(f))
    with open(os.path.join(INTER, "obs_raw.pkl"), "rb") as f:
        raw = pickle.load(f)["results"]
    dh = {r["dataset_id"]: r["digit_hist"] for r in raw}

    bln = pd.read_csv(os.path.join(TABLES, "baseline_derounded.csv"))
    bln = bln.sort_values("stratum")

    ch = chp.merge(der.drop(columns=["n_used"]), on="dataset_id",
                   how="inner")
    hists = [dh[i] for i in ch.dataset_id]
    null, W = _weighted_null(hists, bln,
                             ["E_L1_der", "E_Tail_der", "sd_L1_der",
                              "sd_Tail_der"])
    ch["null_L1_der"] = null["E_L1_der"]
    ch["null_Tail_der"] = null["E_Tail_der"]
    ch["dL1_der"] = ch.L1_der - ch.null_L1_der
    ch["dTail_der"] = ch.Tail_der - ch.null_Tail_der
    ch["z_L1_der"] = ch.dL1_der / null["sd_L1_der"]
    ch["z_Tail_der"] = ch.dTail_der / null["sd_Tail_der"]
    # per-channel magnitude: digit-weighted mean stratum midpoint (~log10)
    ch["digit_mean_log10"] = W @ (np.arange(1, W.shape[1] + 1) - 0.5)

    num = ch.select_dtypes(include=[float]).columns
    ch[num] = ch[num].round(6)
    ch.sort_values(["top_project", "dataset_family", "column_name",
                    "dataset_id"], kind="mergesort").to_csv(
        os.path.join(TABLES, "channel_profiles_v3.csv"), index=False,
        encoding="utf-8")

    core = ch[~ch.low_info].copy()
    coords = ["L1", "L2", "L3", "Tail", "H2", "keff", "dL1", "dL2", "dTail",
              "z_L1", "z_L2", "rounding_mass", "v3_share", "frac_core1",
              "L1_der", "L2_der", "Tail_der", "dL1_der", "dTail_der",
              "z_L1_der", "z_Tail_der"]
    g = core.groupby("dataset_family")
    famp = g[coords].mean()
    famp["channels"] = g.size()
    famp["records"] = g.n_records_used.sum()
    famp["domain"] = g.domain.first()
    famp["family_label"] = g.family_label.first()
    famp["mean_log10"] = g.digit_mean_log10.median()  # banding coordinate
    famp = famp.reset_index().sort_values("dataset_family")
    famp.round(6).to_csv(os.path.join(TABLES, "family_profiles_v3.csv"),
                         index=False, encoding="utf-8")
    return ch, famp


def synth_families():
    syn = pd.read_csv(os.path.join(FROZEN, "synthetic_control.csv"),
                      low_memory=False)
    with open(os.path.join(INTER, "syn_raw.pkl"), "rb") as f:
        raw = pd.DataFrame(pickle.load(f)["results"])
    with open(os.path.join(INTER, "syn_der.pkl"), "rb") as f:
        der = pd.DataFrame(pickle.load(f))
    bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
    bl = bl[bl.stratum != "PD(0,1)"].copy()
    bl["d"] = bl.stratum.astype(int)
    bl = bl.sort_values("d")
    bld = pd.read_csv(os.path.join(TABLES, "baseline_derounded.csv")) \
        .sort_values("stratum")
    m = syn[["dataset_id", "dataset_family", "domain", "channel_kind",
             "generating_process"]] \
        .merge(raw, on="dataset_id").merge(
        der.drop(columns=["n_used"]), on="dataset_id")
    hists = list(m.digit_hist)
    D = len(bl)
    W = np.vstack([np.asarray(h[:D], dtype=float) / max(1, sum(h))
                   for h in hists])
    m["dL1"] = m.L1 - W @ bl.E_L1.to_numpy()
    m["dL2"] = m.L2 - W @ bl.E_L2.to_numpy()
    m["dTail"] = m.Tail - W @ bl.E_Tail.to_numpy()
    m["keff"] = (W @ bl.E_H2.to_numpy()) / m.H2
    m["dL1_der"] = m.L1_der - W @ bld.E_L1_der.to_numpy()
    m["dTail_der"] = m.Tail_der - W @ bld.E_Tail_der.to_numpy()
    g = m.groupby("dataset_family")
    sf = g[["dL1", "dL2", "dTail", "keff", "dL1_der", "dTail_der",
            "rounding_mass", "L1", "Tail"]].mean()
    sf["domain"] = g.domain.first()
    sf["corpus_role"] = "synthetic_control"
    return sf.reset_index(), m


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


def domain_cmap(famp):
    top = famp.domain.value_counts().index.tolist()[:7]
    return {d: PALETTE[i] for i, d in enumerate(top)}


def stage4(ch, famp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    FOOT = (f"run {PARENT_RUN} · sub-run {SUBRUN} · geometry = L-derived "
            "coordinates only; labels are color only")
    cmap = domain_cmap(famp)
    sf, _ = synth_families()

    def col(d):
        return cmap.get(d, GRAY)

    # ---- fig: arm before/after
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=200,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (xc, yc, ttl) in zip(axes, [
            ("dL1", "dTail", "raw residual (dL1, dTail)"),
            ("dL1_der", "dTail_der",
             "de-rounded residual (dL1', dTail')")]):
        style_ax(ax)
        ax.axhline(0, color=BASELINE_C, linewidth=1.0)
        ax.axvline(0, color=BASELINE_C, linewidth=1.0)
        for d in famp.domain.unique():
            sub = famp[famp.domain == d]
            ax.scatter(sub[xc], sub[yc], s=20, color=col(d),
                       edgecolors=SURFACE, linewidths=0.5, zorder=3)
        ax.scatter(sf[xc], sf[yc], s=26, facecolors="none",
                   edgecolors=INK, linewidths=0.7, zorder=4)
        ax.set_title(ttl, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(xc, fontsize=9, color=MUTED)
    axes[0].set_ylabel("residual Tail", fontsize=9, color=MUTED)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      markerfacecolor=v, markeredgecolor=SURFACE, label=k)
               for k, v in cmap.items()]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                          markerfacecolor=GRAY, markeredgecolor=SURFACE,
                          label="other domains"))
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                          markerfacecolor="none", markeredgecolor=INK,
                          label="synthetic_control"))
    fig.legend(handles=handles, fontsize=7.5, frameon=False,
               loc="upper center", ncols=5, bbox_to_anchor=(0.5, 0.06),
               labelcolor=INK)
    fig.suptitle("Does de-rounding collapse the arm?", fontsize=12,
                 color=INK, x=0.01, ha="left")
    fig.text(0.99, 0.005, FOOT, fontsize=6.5, color=MUTED, ha="right")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(DIAG, "fig03_arm_before_after.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- rounding mass vs dTail
    x = famp.rounding_mass.to_numpy()
    y = famp.dTail.to_numpy()
    rho, pval = spearmanr(x, y)
    beta = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - np.polyval(beta, x)) ** 2) / \
        np.sum((y - y.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    for d in famp.domain.unique():
        sub = famp[famp.domain == d]
        ax.scatter(sub.rounding_mass, sub.dTail, s=20, color=col(d),
                   edgecolors=SURFACE, linewidths=0.5, zorder=3)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(beta, xs), color=INK, linewidth=1.2, zorder=4)
    ax.set_xlabel("family mean rounding_mass (share of ln n in 2s & 5s)",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("raw dTail", fontsize=9, color=MUTED)
    ax.set_title(f"rounding_mass vs raw dTail — Spearman ρ={rho:.3f}, "
                 f"univariate R²={r2:.3f}", fontsize=10.5, color=INK,
                 loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig03_roundingmass_vs_dtail.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- derounding_effect.csv
    g = famp.groupby("domain")
    eff = pd.DataFrame({
        "families": g.size(),
        "mean_dTail_raw": g.dTail.mean(),
        "mean_dTail_derounded": g.dTail_der.mean(),
        "mean_dL1_raw": g.dL1.mean(),
        "mean_dL1_derounded": g.dL1_der.mean(),
        "mean_rounding_mass": g.rounding_mass.mean(),
    })
    eff["arm_shrinkage"] = np.where(
        eff.mean_dTail_raw.abs() > 1e-4,
        1.0 - eff.mean_dTail_derounded / eff.mean_dTail_raw, np.nan)
    eff = eff.reset_index().sort_values("domain")
    eff.round(6).to_csv(os.path.join(TABLES, "derounding_effect.csv"),
                        index=False, encoding="utf-8")
    return rho, r2, eff


def topology(ch, famp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    FOOT = (f"run {PARENT_RUN} · sub-run {SUBRUN} · clustering input = "
            "L-derived coordinates only (standardized); labels attached "
            "after, for color/composition only")
    cmap = domain_cmap(famp)

    sf, syn_ch = synth_families()
    obs = famp.copy()
    obs["corpus_role"] = "observational"
    keep = ["dataset_family", "domain", "corpus_role", "dL1", "dL2", "dTail",
            "keff", "dL1_der", "dTail_der", "rounding_mass"]
    allf = pd.concat([obs[keep], sf[keep]], ignore_index=True).dropna(
        subset=["dL1", "dL2", "dTail", "keff", "dL1_der", "dTail_der"])
    allf = allf.sort_values("dataset_family").reset_index(drop=True)

    VARIANTS = {
        "raw": ["dL1", "dL2", "dTail", "keff"],
        "derounded": ["dL1_der", "dL2", "dTail_der", "keff"],
        "raw_plus_roundingmass": ["dL1", "dL2", "dTail", "keff",
                                  "rounding_mass"],
    }
    results, sil_report = {}, []
    for name, cols in VARIANTS.items():
        X = allf[cols].to_numpy()
        X = (X - X.mean(0)) / X.std(0)
        best = None
        for mcs in (5, 8, 10, 15, 20):
            lab = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
            k = len(set(lab)) - (1 if -1 in lab else 0)
            mask = lab != -1
            if k >= 2 and mask.sum() > k:
                sil = silhouette_score(X[mask], lab[mask])
            else:
                sil = np.nan
            if best is None or (not np.isnan(sil) and
                                (np.isnan(best[2]) or sil > best[2])):
                best = (mcs, lab, sil, k)
        mcs, lab, sil, k = best
        # Ward cross-check at the same k (>=2)
        kw = max(2, k)
        wlab = AgglomerativeClustering(n_clusters=kw,
                                       linkage="ward").fit_predict(X)
        wsil = silhouette_score(X, wlab)
        results[name] = lab
        noise = float(np.mean(lab == -1))
        sil_report.append((name, mcs, k, sil, noise, kw, wsil))
        allf[f"cluster_{name}"] = lab

    # ---- cluster composition (labels attached AFTER clustering)
    comp_rows = []
    for name in VARIANTS:
        for cl, g in allf.groupby(f"cluster_{name}"):
            top_doms = g.domain.value_counts().head(3)
            comp_rows.append(dict(
                coordinate_set=name, cluster=int(cl), families=len(g),
                synthetic_fraction=float(
                    (g.corpus_role == "synthetic_control").mean()),
                mean_rounding_mass=float(g.rounding_mass.mean()),
                top_domains="; ".join(f"{d}({n})"
                                      for d, n in top_doms.items())))
    comp = pd.DataFrame(comp_rows).sort_values(
        ["coordinate_set", "cluster"])
    comp.round(4).to_csv(os.path.join(TABLES, "cluster_composition.csv"),
                         index=False, encoding="utf-8")

    allf.round(6).to_csv(os.path.join(TABLES, "cluster_membership.csv"),
                         index=False, encoding="utf-8")

    # ---- survival: raw clusters vs derounded clusters (Jaccard overlap)
    surv_rows = []
    for cl in sorted(set(results["raw"])):
        if cl == -1:
            continue
        a = set(np.where(results["raw"] == cl)[0])
        best_j, best_cl = 0.0, None
        for cl2 in sorted(set(results["derounded"])):
            if cl2 == -1:
                continue
            b = set(np.where(results["derounded"] == cl2)[0])
            j = len(a & b) / len(a | b)
            if j > best_j:
                best_j, best_cl = j, cl2
        sub = allf.iloc[sorted(a)]
        surv_rows.append(dict(
            raw_cluster=cl, families=len(a),
            best_derounded_match=best_cl, jaccard=round(best_j, 3),
            survives=best_j >= 0.5,
            synthetic_fraction=round(
                float((sub.corpus_role == "synthetic_control").mean()), 3),
            mean_rounding_mass=round(float(sub.rounding_mass.mean()), 4)))
    surv = pd.DataFrame(surv_rows)
    surv.to_csv(os.path.join(TABLES, "cluster_survival.csv"), index=False,
                encoding="utf-8")

    # ---- fig: clusters raw vs derounded
    shapes = ["o", "s", "^", "D", "v", "P", "X", "*"]
    top_dom = allf.domain.value_counts().index.tolist()[:8]
    smap = {d: shapes[i] for i, d in enumerate(top_dom)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, name, (xc, yc) in zip(
            axes, ["raw", "derounded"],
            [("dL1", "dTail"), ("dL1_der", "dTail_der")]):
        style_ax(ax)
        ax.axhline(0, color=BASELINE_C, linewidth=1.0)
        ax.axvline(0, color=BASELINE_C, linewidth=1.0)
        labs = allf[f"cluster_{name}"]
        for cl in sorted(labs.unique()):
            sub = allf[labs == cl]
            c = GRAY if cl == -1 else PALETTE[cl % len(PALETTE)]
            for d, g in sub.groupby("domain"):
                ax.scatter(g[xc], g[yc], s=24, color=c,
                           marker=smap.get(d, "o"),
                           edgecolors=SURFACE, linewidths=0.5, zorder=3)
        ax.set_xlabel(xc, fontsize=9, color=MUTED)
        ax.set_ylabel(yc, fontsize=9, color=MUTED)
        k = len(set(labs)) - (1 if -1 in set(labs) else 0)
        ax.set_title(f"{name}: {k} clusters (gray = noise)", fontsize=10,
                     color=INK, loc="left")
    handles = [Line2D([], [], marker=m, linestyle="", markersize=6,
                      markerfacecolor=INK, markeredgecolor=SURFACE, label=d)
               for d, m in smap.items()]
    fig.legend(handles=handles, fontsize=7.5, frameon=False,
               loc="upper center", ncols=8, bbox_to_anchor=(0.5, 0.06),
               labelcolor=INK, title="marker = domain (label only)",
               title_fontsize=7.5)
    fig.suptitle("Clusters: raw residual vs de-rounded residual "
                 "(color = cluster)", fontsize=12, color=INK, x=0.01,
                 ha="left")
    fig.text(0.99, 0.005, FOOT, fontsize=6.5, color=MUTED, ha="right")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(os.path.join(DIAG, "fig03_clusters_raw_vs_derounded.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- magnitude bands (small multiples, derounded residual map)
    bands = [(0, 3, "log10 < 3"), (3, 6, "3–6"), (6, 9, "6–9"),
             (9, 99, "≥ 9")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), dpi=200,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (lo, hi, ttl) in zip(axes, bands):
        style_ax(ax)
        ax.axhline(0, color=BASELINE_C, linewidth=0.9)
        ax.axvline(0, color=BASELINE_C, linewidth=0.9)
        sub = famp[(famp.mean_log10 >= lo) & (famp.mean_log10 < hi)]
        for d in sub.domain.unique():
            s2 = sub[sub.domain == d]
            ax.scatter(s2.dL1_der, s2.dTail_der, s=16,
                       color=cmap.get(d, GRAY), edgecolors=SURFACE,
                       linewidths=0.4, zorder=3)
        ax.set_title(f"{ttl}  (n={len(sub)})", fontsize=9, color=INK)
        ax.set_xlabel("dL1'", fontsize=8, color=MUTED)
    axes[0].set_ylabel("dTail'", fontsize=8, color=MUTED)
    fig.suptitle("De-rounded residual map by magnitude band", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.text(0.99, 0.005, FOOT, fontsize=6.5, color=MUTED, ha="right")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(os.path.join(DIAG, "fig03_by_magnitude_band.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- bootstrap centroids (1000 resamples, 14 workers)
    doms = sorted(famp.domain.unique())
    args = [(d, famp[famp.domain == d][["dL1_der", "dTail_der", "dL2"]]
             .to_numpy(), 1000 // WORKERS + 1, c) for d in doms
            for c in range(WORKERS)]
    boot = {d: [] for d in doms}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for d, samples in pool.map(_bootstrap_task, args):
            boot[d].append(samples)
    ci_rows = []
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(0, color=BASELINE_C, linewidth=1.0)
    ax.axvline(0, color=BASELINE_C, linewidth=1.0)
    for d in doms:
        S = np.vstack(boot[d])[:1000]
        c = cmap.get(d, GRAY)
        ax.scatter(S[:, 0], S[:, 1], s=3, color=c, alpha=0.10, zorder=2,
                   linewidths=0)
        mx, my = famp[famp.domain == d][["dL1_der", "dTail_der"]] \
            .mean().to_numpy()
        ax.scatter(mx, my, s=42, color=c, edgecolors=INK, linewidths=0.7,
                   zorder=4)
        if d in cmap or d in ("accounting", "taxi_trips"):
            ax.annotate(d, (mx, my), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color=INK)
        lo = np.percentile(S, 5, axis=0)
        hi = np.percentile(S, 95, axis=0)
        n_fam = int((famp.domain == d).sum())
        ci_rows.append(dict(
            domain=d, families=n_fam,
            dL1_der=mx, dL1_der_ci5=lo[0], dL1_der_ci95=hi[0],
            dTail_der=my, dTail_der_ci5=lo[1], dTail_der_ci95=hi[1],
            dL2=float(famp[famp.domain == d].dL2.mean()),
            dL2_ci5=lo[2], dL2_ci95=hi[2]))
    ax.set_xlabel("dL1' (de-rounded residual)", fontsize=9, color=MUTED)
    ax.set_ylabel("dTail'", fontsize=9, color=MUTED)
    ax.set_title("Domain centroids with 1,000-resample bootstrap clouds "
                 "(90% bands in table)", fontsize=11, color=INK,
                 loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG,
                             "fig03_domain_centroids_bootstrap.png"),
                facecolor=SURFACE)
    plt.close(fig)
    ci = pd.DataFrame(ci_rows).sort_values("domain")
    ci.round(6).to_csv(os.path.join(TABLES, "domain_centroids_ci.csv"),
                       index=False, encoding="utf-8")

    # ---- ML matrix (features and labels SEPARATED)
    syn_full = syn_ch
    obs_ch = ch.copy()
    obs_ch["corpus_role"] = "observational"
    syn_full["corpus_role"] = "synthetic_control"
    frames = []
    for src in (obs_ch, syn_full):
        f = pd.DataFrame({"channel_id": src.dataset_id,
                          "dataset_family": src.dataset_family})
        for c in ["L1", "L2", "L3", "Tail", "H2", "keff", "dL1", "dL2",
                  "dTail", "L1_der", "Tail_der", "dL1_der", "dTail_der",
                  "rounding_mass", "v3_share", "L1_p10", "L1_p50",
                  "L1_p90"]:
            f[c] = src[c] if c in src.columns else np.nan
        if "digit_hist" in src.columns:
            hs = src.digit_hist
        else:
            with open(os.path.join(INTER, "obs_raw.pkl"), "rb") as fh:
                dh2 = {r["dataset_id"]: r["digit_hist"]
                       for r in pickle.load(fh)["results"]}
            hs = src.dataset_id.map(dh2)
        stats = []
        for h in hs:
            h = np.asarray(h, dtype=float)
            w = h / max(1, h.sum())
            d = np.arange(1, len(h) + 1)
            mu = float((w * d).sum())
            stats.append((mu, float(np.sqrt(max(0, (w * d * d).sum()
                                                - mu * mu)))))
        f["mean_log10_digits"] = [s[0] for s in stats]
        f["sd_log10_digits"] = [s[1] for s in stats]
        frames.append((f, src))
    feats = pd.concat([f for f, _ in frames], ignore_index=True) \
        .sort_values("channel_id")
    feats.round(6).to_csv(os.path.join(FROZEN, "ml_features.csv"),
                          index=False, encoding="utf-8")
    # observational channel_profiles lack generating_process — join it from
    # the frozen corpus (labels file only; never touches features)
    gp = pd.read_csv(os.path.join(FROZEN, "observational_corpus.csv"),
                     usecols=["dataset_id", "generating_process"],
                     low_memory=False)
    gp_map = dict(zip(gp.dataset_id, gp.generating_process))
    lab_frames = []
    for _, s in frames:
        lf = s.copy()
        if "generating_process" not in lf.columns:
            lf["generating_process"] = lf.dataset_id.map(gp_map)
        lab_frames.append(lf[["dataset_id", "dataset_family", "domain",
                              "channel_kind", "generating_process",
                              "corpus_role"]].rename(
            columns={"dataset_id": "channel_id"}))
    labels = pd.concat(lab_frames, ignore_index=True) \
        .sort_values("channel_id")
    labels.to_csv(os.path.join(FROZEN, "ml_labels.csv"), index=False,
                  encoding="utf-8")

    return sil_report, surv, comp, ci, allf


def _bootstrap_task(args):
    import zlib
    domain, X, n, chunk = args
    rng = np.random.default_rng(
        [20260703, zlib.crc32(domain.encode()), chunk])
    out = np.empty((n, X.shape[1]))
    for i in range(n):
        idx = rng.integers(0, len(X), len(X))
        out[i] = X[idx].mean(0)
    return domain, out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    with open(os.path.join(CONFIG, "run_config_03.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workers": WORKERS, "seed_derounded_baseline": SEED_DER,
            "corpus_sampling": "identical per-file seeds to Build 02 "
                               "(md5(file|member)) — same 200k samples",
            "derounding": "n = 2^a 5^b * core; rounding_mass = "
                          "(a ln2 + b ln5)/ln n; core L-profile from the "
                          "same factorization; core==1 rows excluded from "
                          "core means (frac_core1 logged); v3 share "
                          "recorded, not stripped",
            "baseline_derounded": "indexed by ORIGINAL digit stratum d; "
                                  "uniform draws de-rounded identically",
            "clustering": "HDBSCAN (min_cluster_size tuned by silhouette) "
                          "+ Ward cross-check; inputs standardized "
                          "L-derived coordinates ONLY; labels attached "
                          "after clustering",
            "geometry_hygiene": "domain/family/kind/process/corpus_role "
                                "never enter residuals, clustering, or any "
                                "distance; color/composition only",
        }, f, indent=2)

    if stage in ("deround", "all"):
        deround()
    if stage in ("baseline_der", "all"):
        baseline_der()
    if stage in ("profiles", "topology", "all"):
        ch, famp = profiles()
    if stage in ("profiles", "all"):
        rho, r2, eff = stage4(ch, famp)
        print(f"\nrounding_mass vs raw dTail: Spearman rho={rho:.3f}, "
              f"univariate R2={r2:.3f}")
        print("\narm shrinkage by domain (1 - dTail'/dTail):")
        for r in eff.sort_values("arm_shrinkage",
                                 ascending=False).itertuples():
            print(f"  {r.domain:<22} dTail {r.mean_dTail_raw:+.4f} -> "
                  f"{r.mean_dTail_derounded:+.4f}  rm="
                  f"{r.mean_rounding_mass:.3f}  shrink="
                  f"{'' if pd.isna(r.arm_shrinkage) else format(r.arm_shrinkage, '+.1%')}")
    if stage in ("topology", "all"):
        sil, surv, comp, ci, allf = topology(ch, famp)
        print("\nclustering (HDBSCAN tuned / Ward cross-check):")
        for name, mcs, k, s, noise, kw, wsil in sil:
            print(f"  {name:<24} mcs={mcs:<3} k={k} sil={s:.3f} "
                  f"noise={noise:.1%} | ward k={kw} sil={wsil:.3f}")
        print("\nraw clusters -> de-rounded survival:")
        for r in surv.itertuples():
            print(f"  raw#{r.raw_cluster} (n={r.families}, synth="
                  f"{r.synthetic_fraction:.0%}, rm={r.mean_rounding_mass:.3f})"
                  f" -> der#{r.best_derounded_match} J={r.jaccard}"
                  f" {'SURVIVES' if r.survives else 'dissolves'}")
        acc = ci[ci.domain == "accounting"]
        if len(acc):
            a = acc.iloc[0]
            print(f"\naccounting (n={int(a.families)} families): dL2="
                  f"{a.dL2:+.4f}  90% CI [{a.dL2_ci5:+.4f}, "
                  f"{a.dL2_ci95:+.4f}]  dTail'={a.dTail_der:+.4f} "
                  f"[{a.dTail_der_ci5:+.4f}, {a.dTail_der_ci95:+.4f}]")
        print("\nGEOMETRY HYGIENE: clustering/residual inputs were "
              "standardized L-derived coordinates only; domain/kind/"
              "process/corpus_role attached post-hoc for color and "
              "composition.")


if __name__ == "__main__":
    main()
