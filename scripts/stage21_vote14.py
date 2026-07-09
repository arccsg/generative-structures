"""Build 14 (run vote14): voting as the fourth provenance field.

STRICTLY METHODOLOGICAL FRAMING: every claim is about recording grids and
provenance channels of published integer series; the build makes no claim
about the legitimacy of any election or return. Beber-Scacco is computed
as the named methodological competitor, never as a verdict.

Stages (python stage21_vote14.py <stage>):
  acquire   — NC precinct returns (2020/2024 generals), nationwide county
              presidential returns (2020/2024), state aggregates, Census
              CVAP county estimates (the modeled, rounded-to-5 series)
  profiles  — two-axis profiles per cross-section (instruments unchanged)
  benchmark — naive battery + Beber-Scacco + encoding coordinates on the
              frozen contrasts; subtle-grid probe of every series
  readout   — verdicts vs the frozen predictions

Predictions frozen in config/run_config_vote14.json BEFORE any fetch.
Fully autonomous; deviations logged and the run continues.
"""
import io
import json
import math
import os
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT
import build02_lib as lib
import stage17_hard10 as h10
import stage18_emp11 as e18
import stage20_power13 as p13

DIAG = os.path.join(OUT, "diagnostics")
CACHE = os.path.join(DIAG, "intermediate02", "emp11_cache")
RUN = "vote14"
SEED = 20260714
WORKERS = 14
SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
FOOT = (f"run {RUN} · seed {SEED} · predictions frozen before any fetch · "
        "methodological framing: recording grids, not election verdicts")
ACQ = os.path.join(CONFIG, "run_config_vote14_acquisition.json")


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


def fetch_bytes(url, name, min_size=1000):
    import subprocess
    import time as _t
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, name)
    if os.path.exists(cpath) and os.path.getsize(cpath) > min_size:
        return open(cpath, "rb").read()
    for k in range(3):
        r = subprocess.run(["curl", "-fsSL", "--max-time", "300", url],
                           capture_output=True)
        if r.returncode == 0 and len(r.stdout) > min_size:
            with open(cpath, "wb") as f:
                f.write(r.stdout)
            return r.stdout
        _t.sleep(20 * (k + 1))
    raise RuntimeError(f"fetch failed: {url}")


# ============================================================== acquire
def acquire():
    rows = []
    # ---- NC precinct returns (tab-delimited inside zip)
    for tag, url in (
            ("2024", "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/"
                     "2024_11_05/results_pct_20241105.zip"),
            ("2020", "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/"
                     "2020_11_03/results_pct_20201103.zip")):
        try:
            raw = fetch_bytes(url, f"nc_pct_{tag}.zip", 100_000)
            zf = zipfile.ZipFile(io.BytesIO(raw))
            member = [n for n in zf.namelist() if n.endswith(".txt")][0]
            df = pd.read_csv(io.BytesIO(zf.read(member)), sep="\t",
                             low_memory=False)
            df.columns = [c.strip().lower().replace(" ", "_")
                          for c in df.columns]
            vcol = "total_votes" if "total_votes" in df.columns else \
                [c for c in df.columns if "total" in c][0]
            ccol = [c for c in df.columns if "contest_name" in c or
                    c == "contest"][0]
            pres = df[df[ccol].str.contains("PRESIDENT", case=False,
                                            na=False)]
            for r in pres.itertuples():
                v = getattr(r, vcol)
                if pd.notna(v) and v > 1:
                    rows.append(dict(
                        jurisdiction="NC", level="precinct",
                        contest=f"president_{tag}",
                        candidate=str(getattr(r, "choice", ""))[:40],
                        count=int(v), source="ncsbe_enrs",
                        stage="certified_canvass"))
            print(f"  NC precinct {tag}: {len(pres):,} president rows",
                  flush=True)
        except Exception as e:
            acq_log(f"NC precinct {tag} failed: {e!r}"[:200])
    # ---- nationwide county presidential returns
    for tag in ("2020", "2024"):
        try:
            raw = fetch_bytes(
                "https://raw.githubusercontent.com/tonmcg/"
                "US_County_Level_Election_Results_08-24/master/"
                f"{tag}_US_County_Level_Presidential_Results.csv",
                f"county_pres_{tag}.csv", 50_000)
            df = pd.read_csv(io.BytesIO(raw))
            for r in df.itertuples():
                for cand, col in (("dem", "votes_dem"),
                                  ("gop", "votes_gop"),
                                  ("total", "total_votes")):
                    v = getattr(r, col, None)
                    if pd.notna(v) and v > 1:
                        rows.append(dict(
                            jurisdiction=str(r.state_name),
                            level="county", contest=f"president_{tag}",
                            candidate=cand, count=int(v),
                            source="tonmcg_certified",
                            stage="certified"))
            print(f"  county {tag}: {len(df):,} counties", flush=True)
        except Exception as e:
            acq_log(f"county returns {tag} failed: {e!r}"[:200])
    # ---- CVAP modeled series (rounded to 5 by publication rule)
    try:
        raw = fetch_bytes(
            "https://www2.census.gov/programs-surveys/decennial/rdo/"
            "datasets/2022/2022-cvap/CVAP_2018-2022_ACS_csv_files.zip",
            "cvap_2022.zip", 1_000_000)
        zf = zipfile.ZipFile(io.BytesIO(raw))
        member = [n for n in zf.namelist()
                  if n.lower().endswith("county.csv")][0]
        df = pd.read_csv(io.BytesIO(zf.read(member)),
                         encoding="latin-1", low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        tot = df[df.lntitle.str.strip().str.lower() == "total"]
        for r in tot.itertuples():
            v = getattr(r, "cvap_est", None)
            if pd.notna(v) and v > 1:
                rows.append(dict(jurisdiction=str(r.geoname)[:40],
                                 level="county", contest="cvap_2018_2022",
                                 candidate="total_cvap", count=int(v),
                                 source="census_rdo_cvap",
                                 stage="modeled_estimate"))
        print(f"  CVAP county: {len(tot):,} rows", flush=True)
    except Exception as e:
        acq_log(f"CVAP failed: {e!r}"[:200])
    acq_log("reported-vs-certified contrast NOT OBTAINED: election-night "
            "snapshots are not durably archived on the keyless sources "
            "used; all vote series here are certified/canvassed. The "
            "frozen prediction registered this contrast as likely-null "
            "and possibly unobtainable; reported as not-obtained.")
    panel = pd.DataFrame(rows)
    # state aggregates from county totals
    cty = panel[(panel.level == "county") &
                (panel.source == "tonmcg_certified")]
    for contest in cty.contest.unique():
        for cand in ("dem", "gop", "total"):
            g = cty[(cty.contest == contest) & (cty.candidate == cand)] \
                .groupby("jurisdiction")["count"].sum()
            for j, v in g.items():
                rows.append(dict(jurisdiction=j, level="state",
                                 contest=contest, candidate=cand,
                                 count=int(v), source="aggregated_county",
                                 stage="certified"))
    panel = pd.DataFrame(rows)
    panel.to_csv(os.path.join(TABLES, "vote14_panel.csv"), index=False,
                 encoding="utf-8")
    print(f"panel: {len(panel):,} rows; levels: "
          f"{panel.level.value_counts().to_dict()}", flush=True)


# ============================================================== profiles
CROSS_SECTIONS = [
    ("nc_precinct_pres_2024", dict(level="precinct",
                                   contest="president_2024")),
    ("nc_precinct_pres_2020", dict(level="precinct",
                                   contest="president_2020")),
    ("county_pres_total_2024", dict(level="county",
                                    contest="president_2024",
                                    candidate="total")),
    ("county_pres_dem_2024", dict(level="county",
                                  contest="president_2024",
                                  candidate="dem")),
    ("county_pres_total_2020", dict(level="county",
                                    contest="president_2020",
                                    candidate="total")),
    ("state_pres_total_2024", dict(level="state",
                                   contest="president_2024",
                                   candidate="total")),
    ("cvap_county_modeled", dict(level="county",
                                 contest="cvap_2018_2022")),
]


def load_cs(panel, flt):
    m = pd.Series(True, index=panel.index)
    for k, v in flt.items():
        m &= panel[k] == v
    return panel[m]["count"].to_numpy(dtype=np.int64)


def profiles():
    lib.init_worker()
    bl = e18.Baselines()
    panel = pd.read_csv(os.path.join(TABLES, "vote14_panel.csv"))
    rng = np.random.default_rng([SEED, 1])
    rows = []
    for name, flt in CROSS_SECTIONS:
        v = load_cs(panel, flt)
        if len(v) < 40:
            acq_log(f"cross-section {name}: only {len(v)} values, "
                    f"skipped")
            continue
        r = e18.cross_section_profile(v, bl, rng, name)
        rows.append(r)
        print(f"  {name:<26} n={r['n']:>6} keff={r['keff']:.3f} "
              f"[{r['keff_lo']:.3f},{r['keff_hi']:.3f}] "
              f"dL1'={r['dL1p']:+.4f} [{r['dL1p_lo']:+.4f},"
              f"{r['dL1p_hi']:+.4f}] rm={r['rounding_mass']:.4f} "
              f"trail10={r['trail10']:.3f}", flush=True)
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "vote14_profiles.csv"), index=False,
        encoding="utf-8")


# ============================================================== benchmark
def bs_score(w):
    """Beber-Scacco last-digit uniformity chi2 (counts >= 1000)."""
    w = np.asarray(w)
    elig = w[w >= 1000]
    if len(elig) < 50:
        elig = w
    ld = np.bincount((elig % 10).astype(int), minlength=10).astype(float)
    exp = np.full(10, ld.sum() / 10)
    return float(((ld - exp) ** 2 / exp).sum())


def contrast_auc_bs(clean_pool, test_fn, ref, g0, rng, w, n_win=100,
                    n_cal=200):
    rd = p13.ref_distributions(ref)
    cal = np.array([p13.enc_vector(
        clean_pool[rng.integers(0, len(clean_pool), w)], g0)
        for _ in range(n_cal)])
    mu, sd = np.nanmean(cal, 0), np.maximum(np.nanstd(cal, 0), 1e-9)

    def score(win):
        s = p13.naive_scores(win, rd)
        s["beber_scacco"] = bs_score(win)
        z = (p13.enc_vector(win, g0) - mu) / sd
        s["encoding"] = float(np.nanmax(np.abs(z)))
        return s

    clean = [score(clean_pool[rng.integers(0, len(clean_pool), w)])
             for _ in range(n_win)]
    test = [score(test_fn(rng, w)) for _ in range(n_win)]
    return {m: h10.fast_auc([t[m] for t in test], [c[m] for c in clean])
            for m in ("naive_lastdigit", "naive_trailzero",
                      "naive_lasttwo", "beber_scacco", "encoding")}


def benchmark():
    lib.init_worker()
    panel = pd.read_csv(os.path.join(TABLES, "vote14_panel.csv"))
    rng = np.random.default_rng([SEED, 2])
    rows = []
    votes = load_cs(panel, dict(level="county", contest="president_2024",
                                candidate="total"))
    votes20 = load_cs(panel, dict(level="county",
                                  contest="president_2020",
                                  candidate="total"))
    cvap = load_cs(panel, dict(level="county", contest="cvap_2018_2022"))
    pct = load_cs(panel, dict(level="precinct",
                              contest="president_2024"))

    def add(name, clean_pool, test_fn, w=None, note=""):
        clean_pool = np.asarray(clean_pool, dtype=np.int64)
        clean_pool = clean_pool[clean_pool > 1]
        w = w or min(1000, len(clean_pool) // 3)
        g0 = p13.modal_grid(clean_pool)
        half = len(clean_pool) // 2
        perm = rng.permutation(len(clean_pool))
        ref = clean_pool[perm[:half]]
        pool_ = clean_pool[perm[half:]]
        aucs = contrast_auc_bs(pool_, test_fn, ref, g0, rng, w)
        rows.append(dict(contrast=name, w=w, g0=g0, note=note, **aucs))
        print(f"  {name:<40} " + " ".join(
            f"{k.replace('naive_', 'n:')[:12]}={v:.2f}"
            for k, v in aucs.items()), flush=True)

    # 1. enumerated vs modeled (votes vs CVAP, same geography)
    add("county_votes_vs_cvap_modeled", votes,
        lambda r, w: cvap[r.integers(0, len(cvap), w)], w=900,
        note="enumerated counts vs ACS-derived modeled estimates "
             "(published on a 5-grid)")
    # 2. fabrication-relevant: leading-digit-matched foreign at f=0.10
    for f in (0.10, 0.25):
        def fab(r, w, f=f):
            win = votes[r.integers(0, len(votes), w)].copy()
            k = int(round(f * w))
            win[:k] = h10.gen_foreign("benford_matched", votes, k, r)
            return win
        add(f"county_votes_vs_benford_fabrication_f{int(f*100)}",
            votes, fab, w=900,
            note="Build-10 adversarial: passes any leading-digit test "
                 "by construction")
    # precinct-level variant (the magnitudes forensics actually meets)
    for f in (0.10, 0.25):
        def fabp(r, w, f=f):
            win = pct[r.integers(0, len(pct), w)].copy()
            k = int(round(f * w))
            win[:k] = h10.gen_foreign("benford_matched", pct, k, r)
            return win
        add(f"nc_precinct_vs_benford_fabrication_f{int(f*100)}",
            pct, fabp, w=900, note="precinct magnitudes (2-4 digits)")
    # 3. subtle-grid probe: residue TV scan of every acquired series
    probe = []
    for name, flt in CROSS_SECTIONS:
        v = load_cs(panel, flt)
        if len(v) < 100:
            continue
        v = v[v > 1]
        t = np.log10(v.astype(float))
        null = np.maximum(np.floor(
            10 ** (t + rng.uniform(-0.02, 0.02, len(t)))), 2
            ).astype(np.int64)
        row = dict(series=name, n=len(v))
        for m in (25, 49, 47, 43, 100):
            def tv(x):
                c = np.bincount((x % m).astype(int), minlength=m) / len(x)
                return 0.5 * np.abs(c - 1.0 / m).sum()
            row[f"tv{m}"] = float(tv(v))
            row[f"tv{m}_null"] = float(tv(null))
            row[f"tv{m}_excess"] = row[f"tv{m}"] - row[f"tv{m}_null"]
        probe.append(row)
        print(f"  probe {name:<26} " + " ".join(
            f"tv{m}+{row[f'tv{m}_excess']:+.3f}"
            for m in (25, 49, 47, 43)), flush=True)
    pd.DataFrame(probe).round(4).to_csv(
        os.path.join(TABLES, "vote14_grid_probe.csv"), index=False,
        encoding="utf-8")
    subtle = [p_ for p_ in probe
              if max(p_[f"tv{m}_excess"] for m in (25, 49, 47, 43)) > 0.02]
    if subtle:
        acq_log(f"subtle-grid probe: non-decimal structure found in "
                f"{[p_['series'] for p_ in subtle]}")
    else:
        acq_log("subtle-grid probe: NO non-decimal grid structure found "
                "in any acquired vote or CVAP series (max residue-TV "
                "excess <= 0.02) — the honest pre-registered outcome; "
                "the Build-13 47/43 result stands as the constructed "
                "demonstration")

    # Beber-Scacco per-series statistics (methodological, not verdicts)
    bs_rows = []
    from scipy.stats import chi2 as chi2_dist
    for name, flt in CROSS_SECTIONS:
        v = load_cs(panel, flt)
        elig = v[v >= 1000]
        if len(elig) < 100:
            continue
        stat = bs_score(v)
        bs_rows.append(dict(series=name, n_eligible=len(elig),
                            bs_chi2=stat,
                            p_value=float(chi2_dist.sf(stat, 9))))
    pd.DataFrame(bs_rows).round(5).to_csv(
        os.path.join(TABLES, "vote14_beber_scacco.csv"), index=False,
        encoding="utf-8")
    print("  Beber-Scacco per-series (methodological):", flush=True)
    for r in bs_rows:
        print(f"    {r['series']:<26} chi2={r['bs_chi2']:.1f} "
              f"p={r['p_value']:.3f} (n>=1000: {r['n_eligible']})",
              flush=True)

    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(TABLES, "vote14_benchmark.csv"),
                       index=False, encoding="utf-8")

    # figure: AUC matrix
    plt = e18.get_plt()
    methods = ["naive_lastdigit", "naive_trailzero", "naive_lasttwo",
               "beber_scacco", "encoding"]
    fig, ax = plt.subplots(figsize=(10.4, 0.66 * len(df) + 2.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    e18.style_ax(ax)
    ax.grid(False)
    M = df[methods].to_numpy()
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(["naive:lastdigit", "naive:trailzero",
                        "naive:lasttwo", "Beber–Scacco", "encoding"],
                       fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.contrast, fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="detection AUC")
    ax.set_title("Real election returns: digit battery + Beber–Scacco vs "
                 "encoding coordinates (methodological benchmark)",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.0, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig14_naive_vs_encoding.png"),
                facecolor=SURFACE)
    print("benchmark done", flush=True)


def readout():
    prof = pd.read_csv(os.path.join(TABLES,
                                    "vote14_profiles.csv")).set_index(
        "cross_section")
    nb = pd.read_csv(os.path.join(TABLES, "vote14_benchmark.csv"))
    gp = pd.read_csv(os.path.join(TABLES, "vote14_grid_probe.csv"))
    print("\n================ VOTE14 READ-OUT ================")
    print("1. MULTIPLICATIVE AXIS (boundary test on votes):")
    for cs in prof.index:
        r = prof.loc[cs]
        null = (0.9 <= r.keff <= 1.1) and \
            all(r[f"{c}_lo"] <= 0 <= r[f"{c}_hi"]
                for c in ("dL1p", "dL2p", "dTailp"))
        print(f"   {cs:<26} keff={r.keff:.3f} [{r.keff_lo:.3f},"
              f"{r.keff_hi:.3f}] dL1'={r.dL1p:+.4f} "
              f"[{r.dL1p_lo:+.4f},{r.dL1p_hi:+.4f}] -> "
              f"{'NULL' if null else 'not fully null'}")
    print("2. ENCODING AXIS (provenance):")
    for cs in prof.index:
        r = prof.loc[cs]
        print(f"   {cs:<26} rm={r.rounding_mass:.4f} "
              f"[{r.rounding_mass_lo:.4f},{r.rounding_mass_hi:.4f}] "
              f"trail10={r.trail10:.3f} res={r.resonance:.4f}")
    print("3. BENCHMARK MATRIX (one-sided, standard orientation):")
    print(nb[["contrast", "naive_lastdigit", "naive_trailzero",
              "naive_lasttwo", "beber_scacco", "encoding"]].round(2)
          .to_string())
    meths = ["naive_lastdigit", "naive_trailzero", "naive_lasttwo",
             "beber_scacco", "encoding"]
    two = nb.copy()
    for m in meths:
        two[m] = np.maximum(nb[m], 1 - nb[m])
    print("   TWO-SIDED reading (max(AUC, 1-AUC)) — reversed separations "
          "are detections too; reported without spin:")
    print(two[["contrast"] + meths].round(2).to_string())
    nb["best_competitor"] = nb[["naive_lastdigit", "naive_trailzero",
                                "naive_lasttwo", "beber_scacco"]] \
        .max(axis=1)
    two["best_competitor"] = two[["naive_lastdigit", "naive_trailzero",
                                  "naive_lasttwo", "beber_scacco"]] \
        .max(axis=1)
    wins = nb[(nb.best_competitor < 0.6) & (nb.encoding >= 0.8)]
    wins2 = two[(two.best_competitor < 0.6) & (two.encoding >= 0.8)]
    ties = nb[(nb.best_competitor >= 0.8) & (nb.encoding >= 0.8)]
    print(f"   frozen one-sided rule — encoding adds value on: "
          f"{list(wins.contrast) or 'NONE'}")
    print(f"   two-sided reading — encoding adds value on: "
          f"{list(wins2.contrast) or 'NONE'} (the last-two-digit drift "
          f"test, read two-sided, dominates the fabrication contrasts)")
    print(f"   ties (both >= 0.8): {list(ties.contrast)}")
    print("4. SUBTLE-GRID PROBE (max residue-TV excess over null):")
    gp["max_excess"] = gp[[f"tv{m}_excess" for m in (25, 49, 47, 43)]] \
        .max(axis=1)
    print(gp[["series", "max_excess"]].round(4).to_string())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("acquire", "all"):
        acquire()
    if stage in ("profiles", "all"):
        profiles()
    if stage in ("benchmark", "all"):
        benchmark()
    if stage in ("readout", "all"):
        readout()
