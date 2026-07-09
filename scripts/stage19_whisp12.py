"""Build 12 (run whisp12): replicate or retire the multiplicative whisper.

Build 11 found preliminary CES estimates carried a small CI-clean
multiplicative non-genericity (dL1' +0.016, dL2' -0.015 on stored
integers) that vanished after benchmarking. Hypothesis: the multiplicative
axis detects estimation-multiplication surviving to the stored integer.

Stages (python stage19_whisp12.py <stage>):
  censuspep — Stage 1: census enumeration vs PEP cohort-component
              estimates, county level, both decades
  sansa     — Stage 2: SA vs NSA state CES (the isolated multiplicative
              step); SA panel fetched from ALFRED, NSA from emp11 cache
  acs       — Stage 3: ACS 5-yr estimate + MOE vs decennial (encoding)
  synthetic — Stage 4: generative-model integers vs real (exploratory)
  readout   — Stage 5: verdict against the frozen rule

Predictions frozen in config/run_config_whisp12.json BEFORE any fetch.
Instruments unchanged from the corpus builds (imported from emp11).
Fully autonomous: deviations logged to
config/run_config_whisp12_acquisition.json and the run continues.
"""
import io
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT
import build02_lib as lib
import stage18_emp11 as e18

DIAG = os.path.join(OUT, "diagnostics")
CACHE = os.path.join(DIAG, "intermediate02", "emp11_cache")
RUN = "whisp12"
SEED = 20260712
WORKERS = 14
SA_VINTAGE = "2026-06-25"       # matches emp11 v3 (NSA comparator)
CES_WHISPER = {"dL1p": 0.016, "dL2p": -0.015}   # Build-11 reference

SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
GRAY = "#898781"
FOOT = f"run {RUN} · seed {SEED} · predictions frozen before any fetch"
ACQ = os.path.join(CONFIG, "run_config_whisp12_acquisition.json")


def acq_log(msg):
    log = {"deviations": []}
    if os.path.exists(ACQ):
        with open(ACQ) as f:
            log = json.load(f)
    log.setdefault("deviations", []).append(
        f"[{datetime.now(timezone.utc).isoformat()[:19]}] {msg}")
    with open(ACQ, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  DEVIATION LOGGED: {msg}", flush=True)


def fetch_cached(url, name):
    """Bulk-file fetch, bytes-safe (Census CSVs are latin-1)."""
    import subprocess
    import time as _t
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, name)
    if os.path.exists(cpath) and os.path.getsize(cpath) > 1000:
        return open(cpath, encoding="latin-1").read()
    for k in range(3):
        r = subprocess.run(["curl", "-fsS", "--max-time", "120", url],
                           capture_output=True)
        if r.returncode == 0 and len(r.stdout) > 1000:
            with open(cpath, "wb") as f:
                f.write(r.stdout)
            return r.stdout.decode("latin-1")
        _t.sleep(15 * (k + 1))
    raise RuntimeError(f"bulk fetch failed: {url}")


def profile(vals, label, rng, bl):
    return e18.cross_section_profile(np.asarray(vals, dtype=np.int64),
                                     bl, rng, label)


def paired_contrast(a_vals, b_vals, bl, rng, label, n_boot=2000):
    """Paired (same-unit) bootstrap of the multiplicative-axis DIFFERENCE
    a - b: units resampled jointly, so magnitude is matched exactly."""
    a = np.asarray(a_vals, dtype=np.int64)
    b = np.asarray(b_vals, dtype=np.int64)
    assert len(a) == len(b)
    ok = (a > 1) & (b > 1)
    a, b = a[ok], b[ok]
    Sa, Sb = e18.lstats(a), e18.lstats(b)

    def axis(S, idx):
        s = S[idx]
        ds = np.clip(s[:, 4].astype(int), 1, 18)
        eh2 = bl.raw.E_H2.reindex(np.arange(1, 19)).to_numpy()[ds - 1]
        keff = float(eh2.mean() / s[:, 3].mean())
        m = np.isfinite(s[:, 5])
        dsm = np.clip(s[m, 4].astype(int), 1, 18)
        e1 = bl.der.E_L1_der.reindex(np.arange(1, 19)).to_numpy()[dsm - 1]
        e2 = bl.der.E_L2_der.reindex(np.arange(1, 19)).to_numpy()[dsm - 1]
        return np.array([keff, float((s[m, 5] - e1).mean()),
                         float((s[m, 6] - e2).mean())])

    full = np.arange(len(a))
    point = axis(Sa, full) - axis(Sb, full)
    boots = np.empty((n_boot, 3))
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        boots[i] = axis(Sa, idx) - axis(Sb, idx)
    lo = np.percentile(boots, 5, axis=0)
    hi = np.percentile(boots, 95, axis=0)
    names = ["d_keff", "d_dL1p", "d_dL2p"]
    row = dict(contrast=label, n=len(a))
    for i, nm in enumerate(names):
        row[nm] = float(point[i])
        row[f"{nm}_lo"] = float(lo[i])
        row[f"{nm}_hi"] = float(hi[i])
    return row


def ci_fig(ax, rows, stat, colors, title):
    e18.style_ax(ax)
    ax.axhline(0.0 if stat != "keff" else 1.0, color=BASELINE_C,
               linewidth=1.0)
    for i, r in enumerate(rows):
        c = colors[i % len(colors)]
        ax.errorbar([i], [r[stat]],
                    yerr=[[r[stat] - r[f"{stat}_lo"]],
                          [r[f"{stat}_hi"] - r[stat]]],
                    marker="o", markersize=6, color=c, capsize=3,
                    markeredgecolor=SURFACE, linewidth=1.6)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["cross_section"].replace("_", "\n")
                        for r in rows], fontsize=7)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")


# =========================================================== Stage 1
def censuspep():
    lib.init_worker()
    bl = e18.Baselines()
    rng = np.random.default_rng([SEED, 1])
    c19 = pd.read_csv(io.StringIO(fetch_cached(
        "https://www2.census.gov/programs-surveys/popest/datasets/"
        "2010-2019/counties/totals/co-est2019-alldata.csv",
        "co-est2019-alldata.csv")), dtype={"STATE": str, "COUNTY": str})
    c24 = pd.read_csv(io.StringIO(fetch_cached(
        "https://www2.census.gov/programs-surveys/popest/datasets/"
        "2020-2024/counties/totals/co-est2024-alldata.csv",
        "co-est2024-alldata.csv")), dtype={"STATE": str, "COUNTY": str})
    c19 = c19[c19.SUMLEV == 50]
    c24 = c24[c24.SUMLEV == 50]
    acq_log("2020 pure enumeration unavailable keylessly (Census API now "
            "requires a key); ESTIMATESBASE2020 from co-est2024 used as "
            "the 2020 census comparator (April 1 count with minor "
            "geographic modifications). 2010 CENSUS2010POP is the pure "
            "enumeration.")

    def col_ints(df, col):
        v = pd.to_numeric(df[col], errors="coerce")
        return v

    d1 = pd.DataFrame({
        "census2010": col_ints(c19, "CENSUS2010POP"),
        "pep2015": col_ints(c19, "POPESTIMATE2015")}).dropna()
    d2 = pd.DataFrame({
        "censusbase2020": col_ints(c24, "ESTIMATESBASE2020"),
        "pep2022": col_ints(c24, "POPESTIMATE2022")}).dropna()
    d1 = d1[(d1 > 1).all(axis=1)].astype(np.int64)
    d2 = d2[(d2 > 1).all(axis=1)].astype(np.int64)
    print(f"counties: decade1 n={len(d1)}, decade2 n={len(d2)}",
          flush=True)

    rows = []
    for name, vals in (("census2010_enumeration", d1.census2010),
                       ("pep2015_estimate", d1.pep2015),
                       ("censusbase2020", d2.censusbase2020),
                       ("pep2022_estimate", d2.pep2022)):
        rows.append(profile(vals.to_numpy(), name, rng, bl))
        r = rows[-1]
        print(f"  {name:<24} keff={r['keff']:.3f} dL1'={r['dL1p']:+.4f} "
              f"[{r['dL1p_lo']:+.4f},{r['dL1p_hi']:+.4f}] "
              f"dL2'={r['dL2p']:+.4f} rm={r['rounding_mass']:.4f}",
              flush=True)
    # magnitude-matched synthetic null (log-uniform at census2010 digits)
    t = np.log10(d1.census2010.to_numpy(float))
    null_vals = np.maximum(np.floor(
        10 ** (t + rng.uniform(-0.02, 0.02, len(t)))), 2).astype(np.int64)
    rows.append(profile(null_vals, "matched_synthetic_null", rng, bl))
    r = rows[-1]
    print(f"  matched_synthetic_null   keff={r['keff']:.3f} "
          f"dL1'={r['dL1p']:+.4f} dL2'={r['dL2p']:+.4f}", flush=True)

    contrasts = [
        paired_contrast(d1.pep2015, d1.census2010, bl, rng,
                        "pep2015_minus_census2010"),
        paired_contrast(d2.pep2022, d2.censusbase2020, bl, rng,
                        "pep2022_minus_censusbase2020")]
    for c in contrasts:
        print(f"  PAIRED {c['contrast']}: d_dL1'={c['d_dL1p']:+.4f} "
              f"[{c['d_dL1p_lo']:+.4f},{c['d_dL1p_hi']:+.4f}] "
              f"d_dL2'={c['d_dL2p']:+.4f} "
              f"[{c['d_dL2p_lo']:+.4f},{c['d_dL2p_hi']:+.4f}]",
              flush=True)
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "whisp12_census_pep.csv"), index=False,
        encoding="utf-8")
    pd.DataFrame(contrasts).round(6).to_csv(
        os.path.join(TABLES, "whisp12_census_pep_contrasts.csv"),
        index=False, encoding="utf-8")

    plt = e18.get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, stat, ttl in zip(
            axes, ("dL1p", "dL2p", "keff"),
            ("dL1′ (CES whisper: +0.016)", "dL2′ (CES whisper: −0.015)",
             "keff")):
        ci_fig(ax, rows, stat, PALETTE, ttl)
        if stat in ("dL1p", "dL2p"):
            ax.axhline(CES_WHISPER[stat], color=GRAY, linewidth=0.9,
                       linestyle="--")
    fig.suptitle("Census enumeration vs PEP cohort-component estimates — "
                 "county cross-sections (90% CIs; dashed = Build-11 CES "
                 "whisper)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(os.path.join(DIAG, "fig12_census_pep.png"),
                facecolor=SURFACE)
    print("stage 1 done", flush=True)


# =========================================================== Stage 2
def sa_series_list():
    sa_mnem = {mn[:-1]: ss for mn, ss in e18.MNEMONICS.items()}
    series = [(f"SMS{st}00000{ss}01", st, ss) for st in e18.STATE_FIPS
              for ss in e18.SMU_SUPERSECTORS] + \
        [(f"{e18.STATE_ABBR[st]}{mn}", st, ss) for st in e18.STATE_FIPS
         for mn, ss in sa_mnem.items()]
    return series


def sansa():
    lib.init_worker()
    bl = e18.Baselines()
    rng = np.random.default_rng([SEED, 2])
    series = sa_series_list()
    invalid = set()
    log = {"deviations": []}

    def cache_read(sid):
        p = os.path.join(CACHE, f"{sid}_{SA_VINTAGE}.csv")
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            return None
        body = open(p).read().strip()
        if body == "invalid":
            invalid.add(sid)
            return "invalid"
        try:
            return float(body)
        except ValueError:
            return None

    for sweep in range(8):
        todo = [sid for sid, st, ss in series
                if sid not in invalid and cache_read(sid) is None]
        if len(series) - len(todo) >= 500 and todo:
            acq_log(f"SA panel proceeding at {len(series)-len(todo)}/816 "
                    f"coverage: the ALFRED edge re-blocked mid-fetch and "
                    f"{len(todo)} cells would cost hours at breaker "
                    f"cadence; CI-width consequence reported in the note")
            break
        if len(todo) <= 12:
            if todo:
                acq_log(f"{len(todo)} SA cells left unfetched "
                        f"(timeout-instead-of-404 ids): {todo[:6]}...")
            break
        got = {}
        B = 10 if len(todo) > 100 else 1
        for i in range(0, len(todo), B):
            batch = [s for s in todo[i:i + B] if s not in invalid]
            if batch:
                e18.fetch_batch(batch, SA_VINTAGE, got, invalid, log)
            if (i // B) % 10 == 0:
                print(f"  SA sweep{sweep}: {min(i + B, len(todo))}/"
                      f"{len(todo)} (+{len(got)})", flush=True)
        for sid in todo:
            p = os.path.join(CACHE, f"{sid}_{SA_VINTAGE}.csv")
            if sid in invalid:
                open(p, "w").write("invalid")
            elif sid in got:
                open(p, "w").write(str(got[sid]))
    for d in log["deviations"][:10]:
        acq_log(f"SA fetch: {d}")

    # assemble matched SA/NSA pairs by (state, supersector)
    nsa = {}
    for sid, st, ss in ([(f"SMU{st}00000{ss}01", st, ss)
                         for st in e18.STATE_FIPS
                         for ss in e18.SMU_SUPERSECTORS] +
                        [(f"{e18.STATE_ABBR[st]}{mn}", st, ss)
                         for st in e18.STATE_FIPS
                         for mn, ss in e18.MNEMONICS.items()]):
        p = os.path.join(CACHE, f"{sid}_{SA_VINTAGE}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                v = float(open(p).read().strip())
                if np.isfinite(v):
                    nsa[(st, ss)] = v
            except ValueError:
                pass
    pairs = []
    for sid, st, ss in series:
        v = cache_read(sid)
        if isinstance(v, float) and np.isfinite(v) and (st, ss) in nsa:
            pairs.append(dict(state=st, industry=ss, sa=v,
                              nsa=nsa[(st, ss)]))
    pr = pd.DataFrame(pairs)
    print(f"matched SA/NSA pairs: {len(pr)}", flush=True)
    acq_log(f"SA storage precision: identical to NSA (thousands with one "
            f"decimal = 100-person grid; 3-6 significant figures by state "
            f"size) — the seasonal multiply is rounded to this grid at "
            f"publication (Build 07/10 cliff applies at ~4-5 sig figs). "
            f"Matched pairs n={len(pr)}.")

    sa10 = np.rint(pr.sa.to_numpy() * 10).astype(np.int64)
    nsa10 = np.rint(pr.nsa.to_numpy() * 10).astype(np.int64)
    sa_p = np.rint(pr.sa.to_numpy() * 1000).astype(np.int64)
    nsa_p = np.rint(pr.nsa.to_numpy() * 1000).astype(np.int64)
    rows = [profile(sa10, "SA_x10", rng, bl),
            profile(nsa10, "NSA_x10", rng, bl),
            profile(sa_p, "SA_persons", rng, bl),
            profile(nsa_p, "NSA_persons", rng, bl)]
    for r in rows:
        print(f"  {r['cross_section']:<12} keff={r['keff']:.3f} "
              f"dL1'={r['dL1p']:+.4f} [{r['dL1p_lo']:+.4f},"
              f"{r['dL1p_hi']:+.4f}] dL2'={r['dL2p']:+.4f} "
              f"rm={r['rounding_mass']:.4f}", flush=True)
    con = paired_contrast(sa10, nsa10, bl, rng, "SA_minus_NSA_x10")
    print(f"  PAIRED SA−NSA (x10): d_dL1'={con['d_dL1p']:+.4f} "
          f"[{con['d_dL1p_lo']:+.4f},{con['d_dL1p_hi']:+.4f}] "
          f"d_dL2'={con['d_dL2p']:+.4f} [{con['d_dL2p_lo']:+.4f},"
          f"{con['d_dL2p_hi']:+.4f}]", flush=True)
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "whisp12_sa_nsa.csv"), index=False,
        encoding="utf-8")
    pd.DataFrame([con]).round(6).to_csv(
        os.path.join(TABLES, "whisp12_sa_nsa_contrast.csv"), index=False,
        encoding="utf-8")

    plt = e18.get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    x10rows = rows[:2]
    for ax, stat, ttl in zip(
            axes, ("dL1p", "dL2p", "keff"),
            ("dL1′ (stored-integer)", "dL2′", "keff")):
        ci_fig(ax, x10rows, stat, [PALETTE[5], PALETTE[0]], ttl)
        if stat in ("dL1p", "dL2p"):
            ax.axhline(CES_WHISPER[stat], color=GRAY, linewidth=0.9,
                       linestyle="--")
    fig.suptitle("SA vs NSA state CES (Dec 2025, matched pairs) — the "
                 "isolated seasonal-multiply step (dashed = CES whisper)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(os.path.join(DIAG, "fig12_sa_nsa.png"),
                facecolor=SURFACE)
    print("stage 2 done", flush=True)


# =========================================================== Stage 3
def acs():
    lib.init_worker()
    bl = e18.Baselines()
    rng = np.random.default_rng([SEED, 3])
    body = fetch_cached(
        "https://www2.census.gov/programs-surveys/acs/summary_file/2022/"
        "table-based-SF/data/5YRData/acsdt5y2022-b01003.dat",
        "acsdt5y2022-b01003.dat")
    est, moe, fips = [], [], []
    for line in body.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) >= 3 and parts[0].startswith("0500000US"):
            try:
                e_ = int(parts[1])
                m_ = int(parts[2])
            except ValueError:
                continue
            fips.append(parts[0][-5:])
            est.append(e_)
            moe.append(m_)
    a = pd.DataFrame(dict(fips=fips, est=est, moe=moe))
    c24 = pd.read_csv(io.StringIO(fetch_cached(
        "https://www2.census.gov/programs-surveys/popest/datasets/"
        "2020-2024/counties/totals/co-est2024-alldata.csv",
        "co-est2024-alldata.csv")), dtype={"STATE": str, "COUNTY": str})
    c24 = c24[c24.SUMLEV == 50].copy()
    c24["fips"] = c24.STATE.str.zfill(2) + c24.COUNTY.str.zfill(3)
    m = a.merge(c24[["fips", "ESTIMATESBASE2020"]], on="fips")
    m = m[(m.est > 1) & (m.ESTIMATESBASE2020 > 1)]
    print(f"ACS x decennial matched counties: {len(m)} "
          f"(ACS county rows {len(a)})", flush=True)
    rows = [profile(m.ESTIMATESBASE2020.astype(np.int64).to_numpy(),
                    "census2020_count", rng, bl),
            profile(m.est.astype(np.int64).to_numpy(),
                    "acs5yr_estimate", rng, bl),
            profile(m.moe[m.moe > 1].astype(np.int64).to_numpy(),
                    "acs5yr_MOE", rng, bl)]
    for r in rows:
        print(f"  {r['cross_section']:<18} keff={r['keff']:.3f} "
              f"dL1'={r['dL1p']:+.4f} rm={r['rounding_mass']:.4f} "
              f"[{r['rounding_mass_lo']:.4f},{r['rounding_mass_hi']:.4f}] "
              f"trail10={r['trail10']:.3f}", flush=True)
    con = paired_contrast(m.est, m.ESTIMATESBASE2020, bl, rng,
                          "acs_minus_census2020")
    print(f"  PAIRED ACS−census: d_dL1'={con['d_dL1p']:+.4f} "
          f"[{con['d_dL1p_lo']:+.4f},{con['d_dL1p_hi']:+.4f}]",
          flush=True)
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "whisp12_acs.csv"), index=False,
        encoding="utf-8")
    pd.DataFrame([con]).round(6).to_csv(
        os.path.join(TABLES, "whisp12_acs_contrast.csv"), index=False,
        encoding="utf-8")

    plt = e18.get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ci_fig(axes[0], rows, "rounding_mass", PALETTE,
           "encoding axis: rounding_mass")
    axes[0].axhline(0, color=BASELINE_C, linewidth=0)
    ci_fig(axes[1], rows, "dL1p", PALETTE,
           "multiplicative axis: dL1′")
    fig.suptitle("ACS 5-yr estimate & MOE vs decennial count — provenance "
                 "channel (90% CIs)", fontsize=10.5, color=INK, x=0.01,
                 ha="left")
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(os.path.join(DIAG, "fig12_acs.png"), facecolor=SURFACE)
    print("stage 3 done", flush=True)


# =========================================================== Stage 4
def synthetic():
    lib.init_worker()
    bl = e18.Baselines()
    rng = np.random.default_rng([SEED, 4])
    import stage17_hard10 as h10
    hosts = h10.select_hosts()
    row = list(hosts.itertuples())[0]
    real = h10.load_channel(row)
    real = real[real > 1]
    acq_log(f"synthetic generator: log10-KDE (scipy gaussian_kde) fitted "
            f"to real channel {row.dataset_id}, samples snapped to the "
            f"host grid by digit-conditional trailing-zero resampling; "
            f"sdv/CTGAN not installed (heavy torch dependency) — "
            f"generator choice logged per protocol.")
    from scipy.stats import gaussian_kde
    t = np.log10(real.astype(float))
    kde = gaussian_kde(t, bw_method=0.15)
    ts = kde.resample(len(real), seed=int(rng.integers(2**31)))[0]
    ts = np.clip(ts, t.min(), t.max())
    syn = np.maximum(np.floor(10 ** ts), 2).astype(np.int64)
    # grid-snap: sample trailing-10 depth conditional on digit length
    zr = h10.valuation(real, 10)
    dr = h10.digits_of(real)
    ds = h10.digits_of(syn)
    z_by_d = {d: zr[dr == d] for d in np.unique(dr)}
    zs = np.zeros(len(syn), dtype=np.int64)
    for d in np.unique(ds):
        pool = z_by_d.get(d)
        if pool is None or not len(pool):
            near = min(z_by_d, key=lambda x: abs(x - d))
            pool = z_by_d[near]
        sel = ds == d
        zs[sel] = pool[rng.integers(0, len(pool), sel.sum())]
    zs = np.minimum(zs, np.maximum(ds - 2, 0))
    g = (10 ** zs).astype(np.int64)
    syn = np.maximum(np.round(syn / g).astype(np.int64) * g, 2)

    rows = [profile(real, "real_channel", rng, bl),
            profile(syn, "synthetic_kde_gridsnap", rng, bl)]
    for r in rows:
        print(f"  {r['cross_section']:<24} keff={r['keff']:.3f} "
              f"dL1'={r['dL1p']:+.4f} [{r['dL1p_lo']:+.4f},"
              f"{r['dL1p_hi']:+.4f}] dL2'={r['dL2p']:+.4f} "
              f"rm={r['rounding_mass']:.4f}", flush=True)
    con = paired_contrast(syn, real[:len(syn)], bl, rng,
                          "synthetic_minus_real")
    print(f"  PAIRED syn−real: d_dL1'={con['d_dL1p']:+.4f} "
          f"[{con['d_dL1p_lo']:+.4f},{con['d_dL1p_hi']:+.4f}] "
          f"d_keff={con['d_keff']:+.4f}", flush=True)
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "whisp12_synthetic.csv"), index=False,
        encoding="utf-8")
    pd.DataFrame([con]).round(6).to_csv(
        os.path.join(TABLES, "whisp12_synthetic_contrast.csv"),
        index=False, encoding="utf-8")

    plt = e18.get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, stat, ttl in zip(axes, ("dL1p", "dL2p", "keff"),
                             ("dL1′", "dL2′", "keff")):
        ci_fig(ax, rows, stat, [PALETTE[3], PALETTE[6]], ttl)
    fig.suptitle("EXPLORATORY: real channel vs magnitude/grid-matched "
                 "KDE synthetic (90% CIs)", fontsize=10.5, color=INK,
                 x=0.01, ha="left")
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(os.path.join(DIAG, "fig12_synthetic.png"),
                facecolor=SURFACE)
    print("stage 4 done", flush=True)


# =========================================================== Stage 5
def readout():
    cp = pd.read_csv(os.path.join(TABLES, "whisp12_census_pep.csv"))
    cc = pd.read_csv(os.path.join(TABLES,
                                  "whisp12_census_pep_contrasts.csv"))
    sn = pd.read_csv(os.path.join(TABLES, "whisp12_sa_nsa.csv"))
    sc = pd.read_csv(os.path.join(TABLES, "whisp12_sa_nsa_contrast.csv"))
    ac = pd.read_csv(os.path.join(TABLES, "whisp12_acs.csv"))
    sy = pd.read_csv(os.path.join(TABLES,
                                  "whisp12_synthetic_contrast.csv"))
    print("\n================ WHISP12 READ-OUT ================")
    print("1. CENSUS vs PEP (paired, identical counties):")
    for r in cc.itertuples():
        rep = (r.d_dL1p_lo > 0 and r.d_dL2p_hi < 0 and
               r.d_dL1p >= 0.01)
        print(f"   {r.contrast}: d_dL1'={r.d_dL1p:+.4f} "
              f"[{r.d_dL1p_lo:+.4f},{r.d_dL1p_hi:+.4f}], "
              f"d_dL2'={r.d_dL2p:+.4f} [{r.d_dL2p_lo:+.4f},"
              f"{r.d_dL2p_hi:+.4f}] -> "
              f"{'REPLICATES' if rep else 'does not replicate'}")
    s = sc.iloc[0]
    sa_rep = (s.d_dL1p_lo > 0 and s.d_dL2p_hi < 0 and s.d_dL1p >= 0.01)
    print(f"2. SA vs NSA (paired): d_dL1'={s.d_dL1p:+.4f} "
          f"[{s.d_dL1p_lo:+.4f},{s.d_dL1p_hi:+.4f}], "
          f"d_dL2'={s.d_dL2p:+.4f} [{s.d_dL2p_lo:+.4f},{s.d_dL2p_hi:+.4f}]"
          f" -> {'REPLICATES' if sa_rep else 'does not replicate'}")
    est = ac[ac.cross_section == "acs5yr_estimate"].iloc[0]
    cen = ac[ac.cross_section == "census2020_count"].iloc[0]
    moe = ac[ac.cross_section == "acs5yr_MOE"].iloc[0]
    enc_fires = moe.rounding_mass_lo > cen.rounding_mass_hi
    print(f"3. ACS: est rm={est.rounding_mass:.4f}, census "
          f"rm={cen.rounding_mass:.4f}, MOE rm={moe.rounding_mass:.4f} "
          f"-> encoding {'FIRES (MOE)' if enc_fires else 'does not fire'};"
          f" est dL1'={est.dL1p:+.4f} [{est.dL1p_lo:+.4f},"
          f"{est.dL1p_hi:+.4f}]")
    y = sy.iloc[0]
    print(f"4. SYNTHETIC (exploratory): d_dL1'={y.d_dL1p:+.4f} "
          f"[{y.d_dL1p_lo:+.4f},{y.d_dL1p_hi:+.4f}], "
          f"d_keff={y.d_keff:+.4f}")
    promoted = any(
        (r.d_dL1p_lo > 0 and r.d_dL2p_hi < 0 and r.d_dL1p >= 0.01)
        for r in cc.itertuples()) or sa_rep
    print(f"5. VERDICT (frozen rule): "
          f"{'PROMOTED' if promoted else 'RETIRED'}")
    nul = cp[cp.cross_section == "matched_synthetic_null"].iloc[0]
    print(f"   (matched synthetic null reads dL1'={nul.dL1p:+.4f}, "
          f"dL2'={nul.dL2p:+.4f} — instrument calibration)")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("censuspep", "all"):
        censuspep()
    if stage in ("sansa", "all"):
        sansa()
    if stage in ("acs", "all"):
        acs()
    if stage in ("synthetic", "all"):
        synthetic()
    if stage in ("readout", "all"):
        readout()
