"""Build 02: freeze observational corpus, regenerate magnitude baseline,
factorize, compute L-profiles (raw + residual), draw the first geography.

Stages (run via: python stage8_build02.py <stage>):
  stage0   — finalize domains, quarantine synthetic, freeze corpus
  corpus   — factorize observational + synthetic channels (heavy)
  baseline — regenerate magnitude-conditioned null (fresh, seeded)
  profiles — channel/family/domain profiles + 4 figures + summary
  all      — everything in order

GEOMETRY HYGIENE: this build computes NO clustering, NO projection, NO
distance. Figures plot L-coordinates only. domain / dataset_family /
channel_kind / generating_process are used exclusively for grouping and
color. Definitions follow prime-factorization/CANONICAL_DEFINITIONS.md
(L_j from prime-power contributions; theta_hat = 1/H2_mean - 1;
k_eff = (theta_hat+1)/N_eff(B) with N_eff = 1 + theta_hat_1(B), which
digit-weighted reduces to k_eff = H2_null_weighted / H2_observed).
Note: the build-02 prompt wrote theta_hat = (1-H2)/(H2-1/n); the canonical
definition file supersedes it and is used here.
"""
import hashlib
import json
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
SUBRUN = "02"
SEED = 20260702
BASELINE_N = 200_000
MIN_D = 13

# Knuth–Trabb Pardo constants as recorded in the project's canonical
# definitions (prime-power grouping)
KTP = {"L1": 0.6243299885, "L2": 0.1860231051, "L3": 0.0849465984}

SYNTH_RE = re.compile(r"kprime|k_prime|keff|k_eff|\bgen\b|generator|"
                      r"synthetic|simulate|_sim_|benchmark|fixture", re.I)
SYNTH_DIRS = {"reports", "results", "output"}
REAL_COL_HINTS = {"fare", "price", "amount", "date", "year", "zip", "county",
                  "state", "population", "sale", "value", "income", "name",
                  "address", "city", "id", "total"}
GEN_PARAM_HINTS = {"seed", "sigma", "theta", "param", "params", "trial",
                   "rep", "iter", "n_sim", "alpha", "beta"}

NEW_DOMAIN_RULES = [
    ("real_estate", r"apartment|estate|listing|rent|realtor|zillow|redfin"),
    ("food_nutrition", r"food|nutrition|recipe|calorie|usda_fdc|ingredient"),
]


# ================================================================ stage 0
def stage0():
    os.makedirs(FROZEN, exist_ok=True)
    os.makedirs(INTER, exist_ok=True)
    dd = pd.read_csv(os.path.join(TABLES, "analysis_corpus_resolved.csv"),
                     low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        dd[c] = dd[c].fillna("")
    fam = pd.read_csv(os.path.join(TABLES, "family_inventory.csv"))
    colsets = dict(zip(fam.dataset_family, fam.column_set.fillna("")))

    # -- 1. name the unresolved wave + 3. re-run token map over columns
    dom_rules = NEW_DOMAIN_RULES + [
        ("taxi_trips", r"taxi|tlc|tripdata|fare|pickup|dropoff"),
        ("procurement", r"procure|contract|award|obligation|solicitation"),
        ("census", r"census|\bacs\b|pums|psam|tract|decennial|puma"),
        ("payments_fraud", r"fraud|ieee|transactionid|card1"),
        ("equity_markets", r"equity|ohlc|ticker|close|volume"),
        ("network_telemetry", r"netflow|n_bytes|n_packets|sum_n_dest"),
        ("crime", r"crime|arrest|incident"),
        ("seismology", r"quake|earthquake|seismic|magnitude|depth_km"),
        ("genomics", r"gene|\bcds\b|intron|exon|transcript|ensembl"),
        ("admin_codes", r"naics|agency_code|duns|\bcik\b|filer"),
        ("accounting", r"ledger|journal|invoice"),
        ("health", r"patient|\bicd\b|hospital|bene_"),
        ("campaign_finance", r"cmte_id|transaction_amt|itcont|\bfec\b"),
        ("sports", r"\bmlb\b|marathon|batting|pitching"),
        ("corporate_financials", r"wrds|compustat|gvkey"),
    ]
    relabels = 0
    for f in dd[dd.domain == "unresolved"].dataset_family.unique():
        rows = dd.dataset_family == f
        r0 = dd[rows].iloc[0]
        blob = " ".join([str(r0.file_path), str(r0.archive_member),
                         colsets.get(f, ""),
                         " ".join(dd[rows].column_name.astype(str))]).lower()
        for dom, pat in dom_rules:
            m = re.search(pat, blob)
            if m:
                dd.loc[rows, "domain"] = dom
                dd.loc[rows, "domain_confidence"] = "med"
                dd.loc[rows, "domain_rule"] = \
                    f"{dom}:stage0_colpath({m.group(0).strip()})"
                relabels += int(rows.sum())
                break

    still = dd[dd.domain == "unresolved"]
    still[["dataset_id", "dataset_family", "family_label", "top_project",
           "file_path", "archive_member", "column_name", "domain_hint"]] \
        .sort_values(["dataset_family", "column_name"], kind="mergesort") \
        .to_csv(os.path.join(DIAG, "still_unresolved_02.csv"), index=False,
                encoding="utf-8")

    # -- 2. quarantine synthetic/derived (with gen-override inspection)
    blob = (dd.file_path.astype(str) + "\x00" +
            dd.archive_member.astype(str)).str.lower()
    fname = dd.apply(lambda r: os.path.basename(r.archive_member or
                                                r.file_path), axis=1)
    hit_re = blob.str.contains(SYNTH_RE) | fname.str.contains(SYNTH_RE)
    comps = (dd.file_path.str.lower().str.split("/") +
             dd.archive_member.str.lower().str.split("/"))
    hit_dir = comps.map(lambda p: bool(SYNTH_DIRS & set(p[:-1])))
    synthetic = hit_re | hit_dir

    overrides = []
    gen_only = blob.str.contains(r"\bgen\b", regex=True) & ~hit_dir
    for f in dd[synthetic & gen_only].dataset_family.unique():
        cols = {t for t in re.split(r"[^a-z0-9_]+", colsets.get(f, "").lower())
                if t}
        col_tokens = {p for c in cols for p in c.split("_")}
        real = len(col_tokens & REAL_COL_HINTS)
        genp = len(col_tokens & GEN_PARAM_HINTS)
        if real >= 2 and genp == 0:
            rows = (dd.dataset_family == f) & synthetic
            overrides.append((f, real, genp, int(rows.sum())))
            synthetic &= ~(dd.dataset_family == f)
    if overrides:
        pd.DataFrame(overrides, columns=[
            "dataset_family", "real_col_hits", "gen_param_hits",
            "rows_overridden"]).to_csv(
            os.path.join(DIAG, "corpus_role_overrides.csv"), index=False)

    dd["corpus_role"] = np.where(synthetic, "synthetic_control",
                                 "observational")

    # -- 4. low-info flag + landmark tag
    lname = dd.column_name.astype(str).str.lower()
    phone_name = lname.str.contains(r"phone|tel\b|fax")
    ten_digit = (pd.to_numeric(dd["min"], errors="coerce") >= 1e9) & \
        (pd.to_numeric(dd["max"], errors="coerce") < 1e10)
    dd["low_info"] = (dd.channel_kind == "identifier") & \
        (phone_name | ten_digit)
    dd["is_landmark"] = dd.channel_kind.isin(
        ["identifier", "geocode", "calendar"])

    # -- 5. freeze
    obs = dd[dd.corpus_role == "observational"].sort_values(
        ["top_project", "file_path", "archive_member", "sheet_or_table",
         "column_name"], kind="mergesort")
    syn = dd[dd.corpus_role == "synthetic_control"].sort_values(
        ["top_project", "file_path", "archive_member", "sheet_or_table",
         "column_name"], kind="mergesort")
    obs_path = os.path.join(FROZEN, "observational_corpus.csv")
    syn_path = os.path.join(FROZEN, "synthetic_control.csv")
    obs.to_csv(obs_path, index=False, encoding="utf-8")
    syn.to_csv(syn_path, index=False, encoding="utf-8")

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "run_id": PARENT_RUN, "subrun": SUBRUN,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "observational_corpus": {
            "path": obs_path, "sha256": sha(obs_path), "rows": len(obs),
            "domains": obs.domain.value_counts().to_dict(),
            "low_info_rows": int(obs.low_info.sum()),
            "landmark_rows": int(obs.is_landmark.sum())},
        "synthetic_control": {
            "path": syn_path, "sha256": sha(syn_path), "rows": len(syn)},
        "stage0_relabels": relabels,
        "gen_overrides": len(overrides),
        "still_unresolved_rows": len(still),
    }
    with open(os.path.join(FROZEN, "freeze_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"stage0: observational={len(obs):,} synthetic={len(syn):,} "
          f"relabelled={relabels} gen_overrides={len(overrides)} "
          f"still_unresolved={len(still):,} low_info="
          f"{int(obs.low_info.sum())}")


# ================================================================ corpus
def _corpus_task(task):
    """task = list of (table_meta, channels). One zip open per task."""
    out, errors, cache = [], [], Counter()
    for meta, channels in task:
        fp, member, sheet, ext, seed = meta
        try:
            df, sampled = lib.read_table(fp, member, sheet, ext, seed)
        except Exception as e:
            for ch in channels:
                errors.append((fp, member, ch["column_name"],
                               repr(e)[:300]))
            continue
        for ch in channels:
            col = ch["column_name"]
            if col not in df.columns:
                errors.append((fp, member, col, "column missing on re-read"))
                continue
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            try:
                ints, n_ceil = lib.coerce_ints(series, ch["monetary"])
                prof = lib.profile_values(ints)
            except Exception as e:
                errors.append((fp, member, col, repr(e)[:300]))
                continue
            if prof is None:
                errors.append((fp, member, col, "no usable integers >= 2"))
                continue
            prof.update(dataset_id=ch["dataset_id"], sampled=bool(sampled),
                        unit_scaled=bool(ch["monetary"]), n_ceiling=n_ceil)
            out.append(prof)
    cache.update(lib.pop_cache_stats())
    return out, errors, dict(cache)


def corpus():
    for role, fname, out_pkl in [
            ("observational", "observational_corpus.csv", "obs_raw.pkl"),
            ("synthetic_control", "synthetic_control.csv", "syn_raw.pkl")]:
        dd = pd.read_csv(os.path.join(FROZEN, fname), low_memory=False)
        for c in ("archive_member", "sheet_or_table"):
            dd[c] = dd[c].fillna("")
        dd["monetary"] = (dd.channel_kind == "amount") | \
            dd.looks_monetary.astype(bool)

        tables = []
        for (fp, member, sheet), g in dd.groupby(
                ["file_path", "archive_member", "sheet_or_table"],
                sort=True):
            ext = ext_of(member or fp)
            seed = int(hashlib.md5(f"{fp}|{member}".encode())
                       .hexdigest()[:8], 16)
            channels = [dict(dataset_id=r.dataset_id,
                             column_name=r.column_name,
                             monetary=bool(r.monetary))
                        for r in g.itertuples()]
            tables.append(((fp, member, sheet, ext, seed), channels))

        # chunk tables; keep same-zip tables adjacent (already sorted)
        tasks, cur = [], []
        for t in tables:
            cur.append(t)
            if len(cur) >= 40:
                tasks.append(cur)
                cur = []
        if cur:
            tasks.append(cur)

        print(f"[{role}] {len(dd):,} channels over {len(tables):,} tables "
              f"in {len(tasks):,} tasks", flush=True)
        results, errors, cache = [], [], Counter()
        with ProcessPoolExecutor(max_workers=10,
                                 initializer=lib.init_worker) as pool:
            for i, (rows, errs, cst) in enumerate(
                    pool.map(_corpus_task, tasks)):
                results.extend(rows)
                errors.extend(errs)
                cache.update(cst)
                if (i + 1) % 20 == 0:
                    print(f"  [{role}] {i+1}/{len(tasks)} tasks, "
                          f"{len(results):,} channels done", flush=True)

        with open(os.path.join(INTER, out_pkl), "wb") as f:
            pickle.dump({"results": results, "cache": dict(cache)}, f)
        pd.DataFrame(errors, columns=[
            "file_path", "archive_member", "column_name", "error"]).to_csv(
            os.path.join(DIAG, f"factorization_errors_{role}.csv"),
            index=False, encoding="utf-8")
        ceil_rows = [(r["dataset_id"], r["n_ceiling"]) for r in results
                     if r.get("n_ceiling")]
        mode = "w" if role == "observational" else "a"
        with open(os.path.join(DIAG, "factorization_ceiling.csv"), mode,
                  encoding="utf-8") as f:
            if role == "observational":
                f.write("dataset_id,n_over_1e18\n")
            for did, n in sorted(ceil_rows):
                f.write(f"{did},{n}\n")
        hits, miss = cache.get("hits", 0), cache.get("misses", 0)
        print(f"[{role}] done: {len(results):,} channels, "
              f"{len(errors)} errors, large-value cache hit rate "
              f"{hits/max(1,hits+miss):.1%}", flush=True)


# ================================================================ baseline
def _baseline_task(args):
    d, chunk_idx, n = args
    lo = max(2, 10 ** (d - 1))
    hi = 10 ** d
    rng = np.random.default_rng([SEED, d, chunk_idx])
    vals = rng.integers(lo, hi, size=n, dtype=np.int64)
    sums = np.zeros(5)
    sq = np.zeros(5)
    ksum = 0.0
    for v in vals.tolist():
        l1, l2, l3, tail, h2, r = lib.lprofile(v)
        x = np.array([l1, l2, l3, tail, h2])
        sums += x
        sq += x * x
        ksum += r
    return d, n, sums, sq, ksum


def baseline():
    # D = max digit length present in the corpus (>= 13)
    with open(os.path.join(INTER, "obs_raw.pkl"), "rb") as f:
        obs = pickle.load(f)["results"]
    maxd = MIN_D
    for r in obs:
        h = r["digit_hist"]
        nz = [i + 1 for i, c in enumerate(h) if c > 0]
        if nz:
            maxd = max(maxd, max(nz))
    print(f"baseline: D={maxd} strata × {BASELINE_N:,} draws, seed={SEED}",
          flush=True)

    chunks = 10
    per = BASELINE_N // chunks
    tasks = [(d, c, per) for d in range(1, maxd + 1) for c in range(chunks)]
    acc = {d: [0, np.zeros(5), np.zeros(5), 0.0] for d in range(1, maxd + 1)}
    with ProcessPoolExecutor(max_workers=10,
                             initializer=lib.init_worker) as pool:
        for d, n, sums, sq, ksum in pool.map(_baseline_task, tasks):
            a = acc[d]
            a[0] += n
            a[1] += sums
            a[2] += sq
            a[3] += ksum
            done = sum(v[0] for v in acc.values())
            if done % 400_000 == 0:
                print(f"  baseline {done:,}/{maxd*BASELINE_N:,}", flush=True)

    rows = []
    for d in range(1, maxd + 1):
        n, sums, sq, ksum = acc[d]
        mean = sums / n
        var = np.maximum(sq / n - mean ** 2, 0)
        sd = np.sqrt(var)
        rows.append(dict(stratum=str(d), n_draws=n,
                         E_L1=mean[0], E_L2=mean[1], E_L3=mean[2],
                         E_Tail=mean[3], E_H2=mean[4],
                         sd_L1=sd[0], sd_L2=sd[1], sd_L3=sd[2],
                         sd_Tail=sd[3], sd_H2=sd[4],
                         E_k_parts=ksum / n))
    rows.append(dict(stratum="PD(0,1)", n_draws=0,
                     E_L1=KTP["L1"], E_L2=KTP["L2"], E_L3=KTP["L3"],
                     E_Tail=1.0 - KTP["L1"] - KTP["L2"], E_H2=np.nan,
                     sd_L1=np.nan, sd_L2=np.nan, sd_L3=np.nan,
                     sd_Tail=np.nan, sd_H2=np.nan, E_k_parts=np.nan))
    bl = pd.DataFrame(rows).round(6)
    bl.to_csv(os.path.join(TABLES, "baseline.csv"), index=False,
              encoding="utf-8")
    print(f"baseline written: {maxd} strata + PD(0,1) reference row")


# ================================================================ profiles
KIND_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
                "#e34948", "#e87ba4"]
GRAY = "#898781"
SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")


def _load_channel_profiles():
    dd = pd.read_csv(os.path.join(FROZEN, "observational_corpus.csv"),
                     low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        dd[c] = dd[c].fillna("")
    with open(os.path.join(INTER, "obs_raw.pkl"), "rb") as f:
        obs = pickle.load(f)
    raw = pd.DataFrame(obs["results"])
    bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
    bln = bl[bl.stratum != "PD(0,1)"].copy()
    bln["d"] = bln.stratum.astype(int)
    bln = bln.sort_values("d")
    D = int(bln.d.max())

    ch = dd.merge(raw, on="dataset_id", how="inner")
    W = np.vstack(ch.digit_hist.map(
        lambda h: np.asarray(h[:D], dtype=float) /
        max(1, sum(h))).to_numpy())
    EB = {c: bln[c].to_numpy() for c in
          ("E_L1", "E_L2", "E_L3", "E_Tail", "E_H2", "E_k_parts",
           "sd_L1", "sd_L2", "sd_Tail")}
    ch["null_L1"] = W @ EB["E_L1"]
    ch["null_L2"] = W @ EB["E_L2"]
    ch["null_Tail"] = W @ EB["E_Tail"]
    ch["null_H2"] = W @ EB["E_H2"]
    ch["dL1"] = ch.L1 - ch.null_L1
    ch["dL2"] = ch.L2 - ch.null_L2
    ch["dTail"] = ch.Tail - ch.null_Tail
    ch["sw_L1"] = W @ EB["sd_L1"]
    ch["sw_L2"] = W @ EB["sd_L2"]
    ch["z_L1"] = ch.dL1 / ch.sw_L1
    ch["z_L2"] = ch.dL2 / ch.sw_L2
    # canonical: theta_hat = 1/H2_mean - 1 ; k_eff = (theta+1)/N_eff(B)
    #            = (1/H2_obs) / (1/H2_null_weighted) = H2_null / H2_obs
    ch["theta_hat"] = 1.0 / ch.H2 - 1.0
    ch["keff"] = ch.null_H2 / ch.H2
    for i in range(D):
        ch[f"digit_frac_d{i+1}"] = W[:, i].round(6)
    return ch, bln, D


def profiles():
    ch, bln, D = _load_channel_profiles()

    ident = ["dataset_id", "dataset_family", "family_label", "domain",
             "channel_kind", "column_name", "top_project", "low_info",
             "is_landmark", "unit_scaled", "sampled", "n_used", "n_distinct",
             "log10_min", "log10_max"]
    shape = ["L1", "L2", "L3", "Tail", "H2", "theta_hat", "keff",
             "null_L1", "null_L2", "null_Tail", "null_H2",
             "dL1", "dL2", "dTail", "z_L1", "z_L2",
             "L1_p10", "L1_p50", "L1_p90", "k_parts_mean", "n_ceiling"]
    digit_cols = [f"digit_frac_d{i+1}" for i in range(D)]
    out = ch[ident + shape + digit_cols].rename(
        columns={"n_used": "n_records_used"}).copy()
    num = out.select_dtypes(include=[float]).columns
    out[num] = out[num].round(6)
    out.sort_values(["top_project", "dataset_family", "column_name",
                     "dataset_id"], kind="mergesort").to_csv(
        os.path.join(TABLES, "channel_profiles.csv"), index=False,
        encoding="utf-8")

    # ---- family / domain aggregates (low_info excluded by default)
    core = ch[~ch.low_info].copy()
    coords = ["L1", "L2", "L3", "Tail", "H2", "keff", "dL1", "dL2", "dTail",
              "z_L1", "z_L2"]
    g = core.groupby("dataset_family")
    famp = g[coords].mean()
    famp["channels"] = g.size()
    famp["records"] = g.n_used.sum()
    famp["domain"] = g.domain.first()
    famp["family_label"] = g.family_label.first()
    famp["log10_min"] = g.log10_min.min()
    famp["log10_max"] = g.log10_max.max()
    famp = famp.reset_index().sort_values("dataset_family")
    famp.round(6).to_csv(os.path.join(TABLES, "family_profiles.csv"),
                         index=False, encoding="utf-8")

    gd = famp.groupby("domain")
    domp = gd[coords].mean()
    domp = domp.join(gd[coords].std().rename(
        columns={c: f"sd_{c}" for c in coords}))
    domp["families"] = gd.size()
    domp["channels"] = gd.channels.sum()
    domp["records"] = gd.records.sum()
    domp = domp.reset_index().sort_values("domain")
    domp.round(6).to_csv(os.path.join(TABLES, "domain_profiles.csv"),
                         index=False, encoding="utf-8")

    make_figures(ch, famp, bln)

    # ---- summary
    with open(os.path.join(INTER, "obs_raw.pkl"), "rb") as f:
        cache = pickle.load(f)["cache"]
    hits, miss = cache.get("hits", 0), cache.get("misses", 0)
    total_fact = int(ch.n_used.sum())
    zdom = core.groupby("domain").z_L2.apply(
        lambda s: float(np.mean(np.abs(s))))
    print("\n================ BUILD 02 SUMMARY ================")
    print(f"run {PARENT_RUN} sub-run {SUBRUN} seed={SEED}")
    print(f"observational channels factorized: {len(ch):,}")
    print(f"total record-factorizations: {total_fact:,} "
          f"(large-value cache hit rate {hits/max(1,hits+miss):.1%})")
    print(f"channels sampled (200k cap): {int(ch.sampled.sum()):,}")
    print(f"baseline strata: {D} × {BASELINE_N:,} draws + PD(0,1) row")
    print("\nGEOMETRY HYGIENE: no clustering/projection/distance computed "
          "in this build; domain/family/kind/generating_process were used "
          "for grouping and color only.")
    print("\ndomain centroids (family-level means), sorted by dL2:")
    show = domp.sort_values("dL2")
    for r in show.itertuples():
        print(f"  {r.domain:<22} L1={r.L1:.4f} Tail={r.Tail:.4f}   "
              f"dL1={r.dL1:+.4f} dL2={r.dL2:+.4f}  fams={r.families}")
    flag = zdom[zdom > 3]
    if len(flag):
        print("\ndomains with mean |z_L2| > 3 (first candidate structure):")
        for d, z in flag.sort_values(ascending=False).items():
            print(f"  {d:<22} mean|z_L2|={z:.1f}")
    else:
        print("\nno domain has mean |z_L2| > 3")


def make_figures(ch, famp, bln):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

    FOOT = (f"run {PARENT_RUN} · sub-run {SUBRUN} · geometry = L-coordinates "
            "only; domain is color/grouping only (never an input)")
    top = famp.domain.value_counts().index.tolist()
    colored = top[:7]
    cmap = {d: KIND_PALETTE[i] for i, d in enumerate(colored)}

    def col(d):
        return cmap.get(d, GRAY)

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASELINE_C)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)

    def legend_domains(ax, extra=None):
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], marker="o", linestyle="", markersize=6,
                     markerfacecolor=cmap[d], markeredgecolor=SURFACE,
                     label=d) for d in colored]
        hs.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                         markerfacecolor=GRAY, markeredgecolor=SURFACE,
                         label="other domains"))
        if extra:
            hs += extra
        ax.legend(handles=hs, fontsize=7.5, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  labelcolor=INK)

    from matplotlib.lines import Line2D
    pd01 = (KTP["L1"], 1.0 - KTP["L1"] - KTP["L2"])

    # ---- fig 1: raw family map
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for d in famp.domain.unique():
        sub = famp[famp.domain == d]
        ax.scatter(sub.L1, sub.Tail, s=22, color=col(d),
                   edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.plot(bln.E_L1, bln.E_Tail, color=INK, linewidth=1.2, zorder=4,
            alpha=0.75)
    for r in bln[bln.d.isin([1, 3, 6, 9, 12, 15, 18])].itertuples():
        ax.annotate(f"d={r.d}", (r.E_L1, r.E_Tail), fontsize=6.5,
                    color=MUTED, xytext=(4, -7), textcoords="offset points")
    ax.scatter(*pd01, marker="*", s=170, color=INK, zorder=5)
    ax.annotate("PD(0,1)", pd01, xytext=(8, 4), textcoords="offset points",
                fontsize=8, color=INK)
    ax.set_xlabel("mean L1 (dominant prime-power share)", fontsize=9,
                  color=MUTED)
    ax.set_ylabel("mean Tail = Σ L≥3", fontsize=9, color=MUTED)
    ax.set_title("Raw family map — (L1, Tail) plane, magnitude baseline as "
                 "landmark", fontsize=11, color=INK, loc="left", pad=12)
    legend_domains(ax, extra=[
        Line2D([], [], color=INK, linewidth=1.2,
               label="uniform baseline (d=1…D)"),
        Line2D([], [], marker="*", linestyle="", markersize=11,
               color=INK, label="PD(0,1)")])
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.82, 1))
    fig.savefig(os.path.join(DIAG, "fig02_map_raw_family.pdf"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- fig 2: residual family map
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.axhline(0, color=BASELINE_C, linewidth=1.0, zorder=2)
    ax.axvline(0, color=BASELINE_C, linewidth=1.0, zorder=2)
    for d in famp.domain.unique():
        sub = famp[famp.domain == d]
        ax.scatter(sub.dL1, sub.dTail, s=22, color=col(d),
                   edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.set_xlabel("dL1 (observed − digit-weighted baseline)", fontsize=9,
                  color=MUTED)
    ax.set_ylabel("dTail (observed − digit-weighted baseline)", fontsize=9,
                  color=MUTED)
    ax.set_title("Residual (structural) family map — confound-controlled "
                 "(dL1, dTail)", fontsize=11, color=INK, loc="left", pad=12)
    legend_domains(ax)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.82, 1))
    fig.savefig(os.path.join(DIAG, "fig02_map_residual_family.pdf"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- fig 3: channel-level hexbin + synthetic overlay
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    hb = ax.hexbin(ch.L1, ch.Tail, gridsize=48, cmap="Blues",
                   bins="log", mincnt=1, linewidths=0.2,
                   edgecolors=SURFACE)
    syn_path = os.path.join(INTER, "syn_raw.pkl")
    if os.path.exists(syn_path):
        with open(syn_path, "rb") as f:
            syn = pd.DataFrame(pickle.load(f)["results"])
        if len(syn):
            ax.scatter(syn.L1, syn.Tail, s=9, marker="x",
                       color="#e34948", linewidths=0.8, zorder=4,
                       label="synthetic_control")
            ax.legend(fontsize=8, frameon=False, loc="upper right",
                      labelcolor=INK)
    ax.scatter(*pd01, marker="*", s=170, color=INK, zorder=5)
    fig.colorbar(hb, ax=ax, label="channels (log)", shrink=0.8)
    ax.set_xlabel("mean L1 per channel", fontsize=9, color=MUTED)
    ax.set_ylabel("mean Tail per channel", fontsize=9, color=MUTED)
    ax.set_title("Channel density in raw (L1, Tail) — mass visible, not "
                 "defining", fontsize=11, color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig02_channel_density.pdf"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- fig 4: dL2 by domain
    core = ch[~ch.low_info]
    doms = (core.groupby("domain").size().sort_values(ascending=False)
            .index.tolist())
    data = [core[core.domain == d].dL2.dropna().to_numpy() for d in doms]
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.axhline(0, color=BASELINE_C, linewidth=1.0)
    bp = ax.boxplot(data, tick_labels=doms, showfliers=False, whis=(10, 90),
                    patch_artist=True, medianprops=dict(color=INK))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=col(doms[i]), alpha=0.85, edgecolor=SURFACE)
    ax.set_ylabel("dL2 per channel (observed − baseline)", fontsize=9,
                  color=MUTED)
    ax.set_title("Residual dL2 by domain — the most sensitive coordinate",
                 fontsize=11, color=INK, loc="left", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7.5,
             color=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig02_dL2_by_domain.pdf"),
                facecolor=SURFACE)
    plt.close(fig)


def write_config():
    with open(os.path.join(CONFIG, "run_config_02.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN, "seed": SEED,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "build": "02-factorize-and-first-geography",
            "definitions_source": "prime-factorization/"
                                  "CANONICAL_DEFINITIONS.md",
            "lprofile": "L_j = a_j ln(p_j)/ln(n), prime-power parts, sorted "
                        "desc; H2 over all parts; Tail = 1 - L1 - L2",
            "theta_hat": "1/H2_mean - 1 (canonical; prompt's (1-H2)/"
                         "(H2-1/n) variant superseded)",
            "keff": "H2_null_digitweighted / H2_obs (== (theta+1)/N_eff(B))",
            "factorizer": "SPF sieve < 1e7; sympy isprime/factorint above; "
                          "exact; values > 1e18 logged+skipped",
            "monetary": "amount/looks_monetary channels ×100 cents, "
                        "round-half-to-even",
            "sampling": "per-channel cap 200k records (head 100k + seeded "
                        "random 100k for large seekable files)",
            "baseline": f"{BASELINE_N} uniform draws per digit stratum, "
                        f"strata 1..D (D >= {MIN_D}), + PD(0,1) KTP row",
            "geometry_hygiene": "no clustering/projection/distance in this "
                                "build; metadata is color/grouping only",
        }, f, indent=2)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    write_config()
    if stage in ("stage0", "all"):
        stage0()
    if stage in ("corpus", "all"):
        corpus()
    if stage in ("baseline", "all"):
        baseline()
    if stage in ("profiles", "all"):
        profiles()
