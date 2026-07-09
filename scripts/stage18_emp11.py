"""Build 11 (run emp11): the employment-revisions probe.

Sample vs administrative on both axes: CES state x supersector employment
at three ALFRED real-time vintages (preliminary / 1st revision / 2nd+
revision) vs the QCEW administrative near-census at the same reference
month (December 2025).

Stages (python stage18_emp11.py <stage>):
  acquire  — CES vintages via alfredgraph.csv (throttled, cached,
             resumable) + QCEW open CSV slices; writes tables/emp11_panel.csv
  profile  — L-profile / keff / scale-clean residuals / encoding
             coordinates per vintage cross-section; emp11_profiles.csv
  contrast — two-axis tests + bootstrap CIs + figures; emp11_contrast.csv
  readout  — verdicts vs the frozen predictions

Predictions were frozen in config/run_config_emp11.json BEFORE any fetch.
"""
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT
import build02_lib as lib

DIAG = os.path.join(OUT, "diagnostics")
CACHE = os.path.join(DIAG, "intermediate02", "emp11_cache")
RUN = "emp11"
SEED = 20260711
WORKERS = 14
LN10 = math.log(10)
LN2, LN5 = math.log(2), math.log(5)
REF_MONTH = "2025-12-01"          # reference month: December 2025
VINTAGES = {"v1_preliminary": "2026-02-05",   # state prelim ~Jan 27
            "v2_revision1": "2026-03-25",     # revised with Jan release
            "v3_revision2": "2026-06-25"}     # post-benchmark, latest
STATE_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY"}
STATE_FIPS = sorted(STATE_ABBR)
# FRED aliases the classic state supersectors to mnemonic ids
# ({ABBR}CONSN etc.); only the private/goods/service aggregates resolve
# in SMU form. Wrong-form ids are silently dropped from multi-series
# responses (and 404 alone).
SMU_SUPERSECTORS = ["05000000", "06000000", "07000000", "08000000"]
MNEMONICS = {"NAN": "00000000", "NRMNN": "10000000", "CONSN": "20000000",
             "MFGN": "30000000", "TRADN": "40000000", "INFON": "50000000",
             "FIREN": "55000000", "PBSVN": "60000000", "EDUHN": "65000000",
             "LEIHN": "70000000", "SRVON": "80000000", "GOVTN": "90000000"}
# combined NAICS sectors (31-33, 44-45, 48-49) are not valid QCEW slice
# slugs; manufacturing enters via supersector 1013, trade/transport via
# retail+wholesale-adjacent single sectors covered below
QCEW_SECTORS = ["10", "11", "21", "22", "23", "1013", "42", "51", "52",
                "53", "54", "55", "56", "61", "62", "71", "72", "81"]

SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
GRAY = "#898781"
FOOT = f"run {RUN} · seed {SEED} · predictions frozen before acquisition"


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


def get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    return plt


# ============================================================== acquisition
def _fetch(url, tries=2, base_sleep=20):
    """Fetch via curl subprocess: the St. Louis Fed edge accepts curl's
    TLS fingerprint but times out python-urllib requests (observed
    2026-07-03; deviation logged). Fail fast — the caller's circuit
    breaker handles rate-limit blocks (retrying while blocked extends
    the block)."""
    import subprocess
    for k in range(tries):
        r = subprocess.run(["curl", "-fsS", "--max-time", "40", url],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
        if r.returncode == 22:      # HTTP 404: series/vintage absent
            return ""
        if k == tries - 1:
            raise RuntimeError(f"curl failed rc={r.returncode} "
                               f"{r.stderr[:120]}")
        time.sleep(base_sleep)
    return None


COOLDOWN = [0.0, 180.0]   # [resume-time, current backoff seconds]


def _breaker_fetch(url):
    wait = COOLDOWN[0] - time.time()
    if wait > 0:
        print(f"    [breaker: cooling {wait:.0f}s]", flush=True)
        time.sleep(wait)
    try:
        body = _fetch(url)
        COOLDOWN[1] = 180.0                     # success: reset backoff
        return body
    except Exception:
        COOLDOWN[0] = time.time() + COOLDOWN[1]
        COOLDOWN[1] = min(COOLDOWN[1] * 2, 2700.0)  # 3min -> 45min cap
        raise


PACE = [0]      # last-request wall time (module state for throttling)


def _paced_fetch(url):
    wait = PACE[0] + 8.0 - time.time()
    if wait > 0:
        time.sleep(wait)
    try:
        return _breaker_fetch(url)
    finally:
        PACE[0] = time.time()


def fetch_batch(ids, vdate, out, invalid, log):
    """Multi-series ALFRED request; per-series vintage_date list; bisect
    on 404 (any nonexistent id 404s the whole batch)."""
    url = ("https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=" +
           ",".join(ids) + "&vintage_date=" +
           ",".join([vdate] * len(ids)))
    try:
        body = _paced_fetch(url)
    except Exception as e:
        log["deviations"].append(f"batch failed n={len(ids)}@{vdate}: "
                                 f"{e!r}"[:160])
        return                          # left uncached; rerun refetches
    if body == "":                      # 404 -> at least one bad id
        if len(ids) == 1:
            invalid.add(ids[0])
            return
        mid = len(ids) // 2
        fetch_batch(ids[:mid], vdate, out, invalid, log)
        fetch_batch(ids[mid:], vdate, out, invalid, log)
        return
    lines = body.splitlines()
    if not lines or not lines[0].startswith("observation_date"):
        log["deviations"].append(f"non-CSV response (bot page?) "
                                 f"n={len(ids)}@{vdate}")
        return                          # left uncached; rerun refetches
    cols = lines[0].split(",")[1:]
    vals = {c: np.nan for c in cols}
    for line in lines[1:]:
        parts = line.split(",")
        if parts[0] == REF_MONTH:
            for c, v in zip(cols, parts[1:]):
                try:
                    vals[c] = float(v)
                except ValueError:
                    pass
    for c, v in vals.items():
        out[c.rsplit("_", 1)[0]] = v
    # ids the server silently dropped (unresolvable at this vintage):
    # record as missing-value so they are cached and not refetched
    returned = {c.rsplit("_", 1)[0] for c in cols}
    for sid in ids:
        if sid not in returned:
            out[sid] = np.nan


def acquire():
    os.makedirs(CACHE, exist_ok=True)
    log = {"started_utc": datetime.now(timezone.utc).isoformat(),
           "deviations": [
               "fetches via curl subprocess (urllib TLS fingerprint is "
               "bot-blocked by the StLouisFed edge)",
               "batched multi-series requests with per-series vintage "
               "lists; 404 batches bisected to isolate nonexistent "
               "state x supersector series"]}
    rows = []
    # ---------------- CES vintages via ALFRED (batched, cached, paced)
    series = [(f"SMU{st}00000{ss}01", st, ss) for st in STATE_FIPS
              for ss in SMU_SUPERSECTORS] + \
        [(f"{STATE_ABBR[st]}{mn}", st, ss) for st in STATE_FIPS
         for mn, ss in MNEMONICS.items()]
    meta = {sid: (st, ss) for sid, st, ss in series}
    invalid = set()
    B = 10          # batches of >~12 series time out at the ALFRED edge

    def cache_read(sid, vdate):
        cpath = os.path.join(CACHE, f"{sid}_{vdate}.csv")
        if not (os.path.exists(cpath) and os.path.getsize(cpath) > 0):
            return None
        body = open(cpath).read().strip()
        if body == "invalid":
            invalid.add(sid)
            return "invalid"
        try:
            return float(body)
        except ValueError:
            return None

    for sweep in range(10):
        pre_missing = sum(
            1 for sid, st, ss in series for vdate in VINTAGES.values()
            if sid not in invalid and cache_read(sid, vdate) is None)
        if pre_missing <= 30:   # residual timeout-instead-of-404 ids;
            # dropped by the matched-panel rule anyway — logged
            if pre_missing:
                log["deviations"].append(
                    f"{pre_missing} series-vintage cells left unfetched "
                    "(timeout-instead-of-404 ids); dropped by matched-"
                    "panel rule")
            break
        n_missing = 0
        for vname, vdate in VINTAGES.items():
            todo = [sid for sid, st, ss in series
                    if sid not in invalid and
                    cache_read(sid, vdate) is None]
            if not todo:
                continue
            # mop-up singly: mixed batches containing a nonexistent id
            # time out instead of 404ing, so bisection never triggers
            Bs = B if len(todo) > 100 and sweep < 1 else 1
            n_missing += len(todo)
            got = {}
            for i in range(0, len(todo), Bs):
                batch = [s for s in todo[i:i + Bs] if s not in invalid]
                if not batch:
                    continue
                fetch_batch(batch, vdate, got, invalid, log)
                if (i // Bs) % 8 == 0:
                    print(f"  sweep{sweep} {vname}: "
                          f"{min(i + Bs, len(todo))}/{len(todo)} "
                          f"(+{len(got)} got, {len(invalid)} invalid)",
                          flush=True)
            for sid in todo:
                cpath = os.path.join(CACHE, f"{sid}_{vdate}.csv")
                if sid in invalid:
                    with open(cpath, "w") as f:
                        f.write("invalid")
                elif sid in got:        # fetched (value or genuine nan)
                    with open(cpath, "w") as f:
                        f.write(str(got[sid]))
                # else: batch failed -> refetched next sweep
        print(f"sweep {sweep}: {n_missing} missing at start", flush=True)
        if n_missing == 0:
            break
    # build rows purely from cache
    for sid, (st, ss) in meta.items():
        for vname, vdate in VINTAGES.items():
            v = cache_read(sid, vdate)
            if isinstance(v, float) and np.isfinite(v):
                rows.append(dict(source="CES", vintage=vname,
                                 vintage_date=vdate, series_id=sid,
                                 state=st, industry=ss,
                                 ref_month=REF_MONTH,
                                 value_thousands=v))
    log["n_invalid_series"] = len(invalid)
    log["invalid_series"] = sorted(invalid)
    print(f"CES: {len(rows)} observations; {len(invalid)} nonexistent "
          f"series", flush=True)

    # ---------------- QCEW 2025 Q4 (administrative), open CSV slices
    for sector in QCEW_SECTORS:
        url = f"https://data.bls.gov/cew/data/api/2025/4/industry/{sector}.csv"
        try:
            body = _fetch(url)
        except Exception as e:
            log["deviations"].append(f"QCEW fetch failed {sector}: "
                                     f"{e!r}"[:160])
            continue
        if not body or not body.strip():
            log["deviations"].append(f"QCEW sector {sector}: empty/404")
            continue
        cpath = os.path.join(CACHE, f"qcew_2025q4_{sector}.csv")
        with open(cpath, "w") as f:
            f.write(body)
        import io
        q = pd.read_csv(io.StringIO(body), dtype={"area_fips": str},
                        low_memory=False)
        # state rows: private ownership (5) for sectors, total (0) for
        # industry 10; agglvl 5x = statewide, 7x = county
        for lvl, unit in (("5", "state"), ("7", "county")):
            sub = q[q.agglvl_code.astype(str).str.startswith(lvl)]
            own = 0 if sector == "10" else 5
            sub = sub[sub.own_code == own]
            for r in sub.itertuples():
                v = int(r.month3_emplvl)
                if v <= 1:
                    continue
                rows.append(dict(source="QCEW", vintage="v4_administrative",
                                 vintage_date="2026-06-03(release)",
                                 series_id=f"QCEW_{r.area_fips}_{sector}",
                                 state=str(r.area_fips)[:2],
                                 industry=f"{unit}:{sector}",
                                 ref_month=REF_MONTH,
                                 value_thousands=v / 1000.0))
        print(f"  QCEW sector {sector}: ok", flush=True)
        time.sleep(1.0)
    panel = pd.DataFrame(rows)
    # matched CES panel: series present at all three vintages
    ces = panel[panel.source == "CES"]
    counts = ces.groupby("series_id").vintage.nunique()
    full = set(counts[counts == 3].index)
    dropped = counts[counts < 3]
    if len(dropped):
        log["deviations"].append(
            f"{len(dropped)} CES series missing >=1 vintage, dropped from "
            f"matched panel: {sorted(dropped.index)[:8]}...")
    panel["in_matched_panel"] = panel.series_id.isin(full) | \
        (panel.source == "QCEW")
    panel.to_csv(os.path.join(TABLES, "emp11_panel.csv"), index=False,
                 encoding="utf-8")
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    log["n_ces_series_matched"] = len(full)
    log["n_qcew_rows"] = int((panel.source == "QCEW").sum())
    with open(os.path.join(CONFIG, "run_config_emp11_acquisition.json"),
              "w") as f:
        json.dump(log, f, indent=2)
    print(f"panel written: {len(panel)} rows; matched CES series: "
          f"{len(full)}; QCEW rows: {log['n_qcew_rows']}", flush=True)


# ============================================================== profiling
class Baselines:
    def __init__(self):
        bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
        bl = bl[bl.stratum != "PD(0,1)"].copy()
        bl["d"] = bl.stratum.astype(int)
        self.raw = bl.set_index("d")
        der = pd.read_csv(os.path.join(TABLES, "baseline_derounded_v2.csv"))
        der = der[der.stratum.astype(str).str.isdigit()].copy()
        der["d"] = der.stratum.astype(int)
        self.der = der.set_index("d")


def lstats(v):
    """Per-record raw + 2*5-core stats for an int64 array."""
    out = []
    for x in np.asarray(v, dtype=np.int64).tolist():
        pairs = lib.factor_pairs(int(x))
        ln_n = math.log(x)
        c = sorted((e * math.log(p) for p, e in pairs), reverse=True)
        L = [y / ln_n for y in c]
        L1 = L[0]
        L2 = L[1] if len(L) > 1 else 0.0
        H2 = sum(y * y for y in L)
        d = int(ln_n / LN10) + 1
        core = [(p, e) for p, e in pairs if p not in (2, 5)]
        if core:
            ln_c = sum(e * math.log(p) for p, e in core)
            Lc = sorted((e * math.log(p) / ln_c for p, e in core),
                        reverse=True)
            l1d = Lc[0]
            l2d = Lc[1] if len(Lc) > 1 else 0.0
            taild = 1.0 - l1d - l2d
        else:
            l1d = l2d = taild = np.nan
        a2 = sum(e for p, e in pairs if p == 2)
        a5 = sum(e for p, e in pairs if p == 5)
        out.append((L1, L2, 1.0 - L1 - L2, H2, d, l1d, l2d, taild,
                    (a2 * LN2 + a5 * LN5) / ln_n, min(a2, a5)))
    return np.array(out)


def resonance_score(v):
    """Mean TV of value mod p from uniform, primes/powers as in
    residue_structure.csv, scored where 5*m <= max(value)."""
    v = np.asarray(v, dtype=np.int64)
    mods = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
            4, 8, 9, 25, 27, 49]
    tvs = []
    for m in mods:
        if 5 * m > v.max():
            continue
        cnt = np.bincount((v % m).astype(int), minlength=m) / len(v)
        tvs.append(0.5 * np.abs(cnt - 1.0 / m).sum())
    return float(np.mean(tvs)) if tvs else np.nan


def cross_section_profile(vals, bl, rng, label):
    """Channel-level two-axis profile of one cross-section + bootstrap."""
    vals = np.asarray(vals, dtype=np.int64)
    vals = vals[vals > 1]
    S = lstats(vals)

    def agg(idx):
        s = S[idx]
        ds = np.clip(s[:, 4].astype(int), 1, 18)
        eh2 = bl.raw.E_H2.reindex(np.arange(1, 19)).to_numpy()[ds - 1]
        keff = float(eh2.mean() / s[:, 3].mean())
        m = np.isfinite(s[:, 5])
        if m.sum() >= 20:
            dsm = np.clip(s[m, 4].astype(int), 1, 18)
            e1 = bl.der.E_L1_der.reindex(np.arange(1, 19)).to_numpy()[
                dsm - 1]
            e2 = bl.der.E_L2_der.reindex(np.arange(1, 19)).to_numpy()[
                dsm - 1]
            et = bl.der.E_Tail_der.reindex(np.arange(1, 19)).to_numpy()[
                dsm - 1]
            dL1p = float((s[m, 5] - e1).mean())
            dL2p = float((s[m, 6] - e2).mean())
            dTailp = float((s[m, 7] - et).mean())
        else:
            dL1p = dL2p = dTailp = np.nan
        return (keff, dL1p, dL2p, dTailp, float(s[:, 8].mean()),
                float(s[:, 9].mean()))

    point = agg(np.arange(len(S)))
    boots = np.array([agg(rng.integers(0, len(S), len(S)))
                      for _ in range(2000)])
    lo = np.nanpercentile(boots, 5, axis=0)
    hi = np.nanpercentile(boots, 95, axis=0)
    names = ["keff", "dL1p", "dL2p", "dTailp", "rounding_mass", "trail10"]
    row = dict(cross_section=label, n=len(vals),
               log10_mean=float(np.log10(vals.astype(float)).mean()),
               resonance=resonance_score(vals))
    for i, nm in enumerate(names):
        row[nm] = point[i]
        row[f"{nm}_lo"] = float(lo[i])
        row[f"{nm}_hi"] = float(hi[i])
    return row


def profile():
    lib.init_worker()
    bl = Baselines()
    panel = pd.read_csv(os.path.join(TABLES, "emp11_panel.csv"),
                        dtype={"state": str, "industry": str})
    panel = panel[panel.in_matched_panel]
    rows = []
    rng = np.random.default_rng([SEED, 1])
    # CES vintages: person units (x1000 of thousands; the stored grid is
    # 100 persons). Sensitivity: stored-integer x10 variant.
    for vname in sorted(panel[panel.source == "CES"].vintage.unique()):
        sub = panel[(panel.source == "CES") & (panel.vintage == vname)]
        persons = np.rint(sub.value_thousands.to_numpy() * 1000
                          ).astype(np.int64)
        rows.append(cross_section_profile(persons, bl, rng,
                                          f"CES_{vname}_persons"))
        x10 = np.rint(sub.value_thousands.to_numpy() * 10).astype(np.int64)
        rows.append(cross_section_profile(x10, bl, rng,
                                          f"CES_{vname}_x10"))
        # detected decimals from raw values
        dec = (sub.value_thousands * 10 % 10 != 0).mean()
        rows[-2]["frac_nonzero_decimal"] = float(
            (sub.value_thousands * 10 % 10 != 0).mean())
        print(f"  {vname}: n={len(sub)}", flush=True)
    # QCEW cross-sections: exact person counts
    q = panel[panel.source == "QCEW"].copy()
    q["unit"] = q.industry.str.split(":").str[0]
    q["sector"] = q.industry.str.split(":").str[1]
    for unit in ("state", "county"):
        sub = q[q.unit == unit]
        if not len(sub):
            continue
        persons = np.rint(sub.value_thousands.to_numpy() * 1000
                          ).astype(np.int64)
        rows.append(cross_section_profile(
            persons, bl, rng, f"QCEW_administrative_{unit}"))
        print(f"  QCEW {unit}: n={len(sub)}", flush=True)
    prof = pd.DataFrame(rows)
    prof.round(6).to_csv(os.path.join(TABLES, "emp11_profiles.csv"),
                         index=False, encoding="utf-8")
    print(prof[["cross_section", "n", "keff", "dL2p", "rounding_mass",
                "trail10", "resonance"]].round(4).to_string(), flush=True)


# ============================================================== contrast
# Encoding axis reads person units (the storage grid vs QCEW's exact
# counts). Multiplicative axis reads the stored-integer representation
# (x10 for CES, native for QCEW): person-unit x1000 injects 2^3*5^3 whose
# shrunken cores leak the encoding into original-digit-indexed residuals
# (the Build-09 reference-misspecification lesson, reproduced on a null
# mock before analysis).
# December 2025 state CES had exactly TWO published states (preliminary;
# benchmark-revised from the mid-April 2026 state release) — the normal
# monthly first revision was folded into the benchmark cycle, so vintage
# v2 (2026-03-25) duplicates v1 for all 805 series (verified; deviation
# logged). The drift sequence uses the distinct stages.
SEQ = ["CES_v1_preliminary_persons", "CES_v3_revision2_persons",
       "QCEW_administrative_state"]
MULT_SEQ = ["CES_v1_preliminary_x10", "CES_v3_revision2_x10",
            "QCEW_administrative_state"]
SEQ_LBL = ["initial\n(CES prelim)", "benchmark-revised\n(CES final)",
           "administrative\n(QCEW)"]


def contrast():
    prof = pd.read_csv(os.path.join(TABLES, "emp11_profiles.csv"))
    prof = prof.set_index("cross_section")
    panel = pd.read_csv(os.path.join(TABLES, "emp11_panel.csv"),
                        dtype={"state": str})
    rows = []
    for cs in sorted(set(SEQ + MULT_SEQ + ["QCEW_administrative_county"])):
        if cs not in prof.index:
            continue
        r = prof.loc[cs]
        mult_null = (0.9 <= r.keff <= 1.1) and \
            all(r[f"{c}_lo"] <= 0 <= r[f"{c}_hi"]
                for c in ("dL1p", "dL2p", "dTailp"))
        rows.append(dict(
            cross_section=cs, n=int(r.n), keff=r.keff,
            keff_ci=f"[{r.keff_lo:.3f},{r.keff_hi:.3f}]",
            dL1p=r.dL1p, dL1p_ci=f"[{r.dL1p_lo:.4f},{r.dL1p_hi:.4f}]",
            dL2p=r.dL2p, dL2p_ci=f"[{r.dL2p_lo:.4f},{r.dL2p_hi:.4f}]",
            dTailp=r.dTailp,
            dTailp_ci=f"[{r.dTailp_lo:.4f},{r.dTailp_hi:.4f}]",
            multiplicative_null=bool(mult_null),
            rounding_mass=r.rounding_mass,
            rm_ci=f"[{r.rounding_mass_lo:.4f},{r.rounding_mass_hi:.4f}]",
            trail10=r.trail10, resonance=r.resonance))
    ct = pd.DataFrame(rows)
    # sample-vs-administrative sharp contrast + drift monotonicity
    seq = prof.loc[[s for s in SEQ if s in prof.index]]
    rm = seq.rounding_mass.to_numpy()
    from scipy.stats import kendalltau
    tau = kendalltau(np.arange(len(rm)), rm).statistic
    init, adm = prof.loc[SEQ[0]], prof.loc[SEQ[-1]]
    sep = (init.rounding_mass_lo > adm.rounding_mass_hi)
    ct.attrs = {}
    summary = dict(
        encoding_separation_initial_vs_admin=bool(sep),
        rm_initial=f"{init.rounding_mass:.4f} [{init.rounding_mass_lo:.4f},"
                   f"{init.rounding_mass_hi:.4f}]",
        rm_admin=f"{adm.rounding_mass:.4f} [{adm.rounding_mass_lo:.4f},"
                 f"{adm.rounding_mass_hi:.4f}]",
        drift_kendall_tau=float(tau) if tau == tau else np.nan,
        drift_all_steps_down=bool(np.all(np.diff(rm) < 0)))
    ct.round(5).to_csv(os.path.join(TABLES, "emp11_contrast.csv"),
                       index=False, encoding="utf-8")
    with open(os.path.join(TABLES, "emp11_contrast_summary.json"),
              "w") as f:
        json.dump(summary, f, indent=2)
    print(ct.to_string(), flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    # ---------------- figures
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(len(SEQ))
    seqd = prof.loc[[s for s in SEQ if s in prof.index]]
    multd = prof.loc[[s for s in MULT_SEQ if s in prof.index]]
    ax = axes[0]
    style_ax(ax)
    ax.axhline(1.0, color=BASELINE_C, linewidth=1.0)
    ax.errorbar(xs, multd.keff,
                yerr=[multd.keff - multd.keff_lo,
                      multd.keff_hi - multd.keff],
                marker="o", markersize=6, linewidth=1.8, color=PALETTE[0],
                markeredgecolor=SURFACE, capsize=3, label="keff")
    ax2v = multd.dL2p
    ax.errorbar(xs + 0.06, 1.0 + ax2v * 5,
                yerr=[(ax2v - multd.dL2p_lo) * 5,
                      (multd.dL2p_hi - ax2v) * 5],
                marker="D", markersize=5, linewidth=1.4, color=PALETTE[4],
                markeredgecolor=SURFACE, capsize=3, linestyle="--",
                label="1 + 5·dL2′ (scale-clean)")
    ax.set_xticks(xs)
    ax.set_xticklabels(SEQ_LBL, fontsize=7.5)
    ax.set_ylabel("multiplicative axis (stored-integer representation)",
                  fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="best", labelcolor=INK)
    ax.set_title("Multiplicative axis: null at every vintage?",
                 fontsize=10, color=INK, loc="left", pad=10)
    ax = axes[1]
    style_ax(ax)
    ax.errorbar(xs, seqd.rounding_mass,
                yerr=[seqd.rounding_mass - seqd.rounding_mass_lo,
                      seqd.rounding_mass_hi - seqd.rounding_mass],
                marker="o", markersize=6, linewidth=1.8, color=PALETTE[2],
                markeredgecolor=SURFACE, capsize=3, label="rounding_mass")
    ax.plot(xs, seqd.trail10 / 10, marker="D", markersize=5,
            linewidth=1.4, linestyle="--", color=PALETTE[1],
            markeredgecolor=SURFACE, label="trailing-10 depth / 10")
    ax.set_xticks(xs)
    ax.set_xticklabels(SEQ_LBL, fontsize=7.5)
    ax.set_ylabel("encoding axis", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="best", labelcolor=INK)
    ax.set_title("Encoding axis: precision fingerprint by provenance",
                 fontsize=10, color=INK, loc="left", pad=10)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig11_two_axis.png"),
                facecolor=SURFACE)

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    coords = [("rounding_mass", PALETTE[2], "o"),
              ("trail10", PALETTE[1], "D"),
              ("resonance", PALETTE[4], "s")]
    for nm, c, mk in coords:
        y = seqd[nm].to_numpy()
        y0 = y / (abs(y[0]) if abs(y[0]) > 1e-12 else 1.0)
        ax.plot(xs, y0, marker=mk, markersize=6, linewidth=1.8, color=c,
                markeredgecolor=SURFACE,
                label=f"{nm} (initial = {y[0]:.3f})")
    ax.axhline(0, color=BASELINE_C, linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(SEQ_LBL, fontsize=8)
    ax.set_ylabel("encoding coordinate, relative to the initial estimate",
                  fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="best", labelcolor=INK)
    ax.set_title("Revision drift: does the encoding converge to the "
                 "administrative count?", fontsize=10.5, color=INK,
                 loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig11_revision_drift.png"),
                facecolor=SURFACE)
    print("figures written: fig11_two_axis.png, fig11_revision_drift.png",
          flush=True)


def readout():
    prof = pd.read_csv(os.path.join(TABLES,
                                    "emp11_profiles.csv")).set_index(
        "cross_section")
    with open(os.path.join(TABLES, "emp11_contrast_summary.json")) as f:
        summary = json.load(f)
    print("\n================ EMP11 READ-OUT ================")
    for cs in SEQ + MULT_SEQ[:3] + ["QCEW_administrative_county"]:
        if cs not in prof.index:
            continue
        r = prof.loc[cs]
        print(f"  {cs:<34} n={int(r.n):>6} keff={r.keff:.3f} "
              f"[{r.keff_lo:.3f},{r.keff_hi:.3f}] dL2'={r.dL2p:+.4f} "
              f"rm={r.rounding_mass:.4f} trail10={r.trail10:.2f} "
              f"res={r.resonance:.4f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("acquire", "all"):
        acquire()
    if stage in ("profile", "all"):
        profile()
    if stage in ("contrast", "all"):
        contrast()
    if stage in ("readout", "all"):
        readout()
