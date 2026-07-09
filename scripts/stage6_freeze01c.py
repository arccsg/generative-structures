"""Build 01c: apply weather + SEC exclusions, dedup, tag channel_kind,
freeze the analysis-corpus candidate. Supersedes 01b.

Reads ONLY Build 01 outputs (tables/catalog.csv + column_profiles.csv for
n_distinct_int). No re-walk, no factorization, no L-profile. Exclusion is a
flag written to new tables — nothing existing is moved or deleted.

channel_kind is METADATA ONLY — never fed into clustering, projection,
distance, or geometry downstream.

dup_key (from Build 01) is a content FINGERPRINT, not a path key:
md5(basename(file or member) | column_name | frac_integer | min | max |
n_integer). GSOD-style station-years differ in sampled stats, so they remain
distinct dup_keys unless stats collide exactly.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT

DIAG = os.path.join(OUT, "diagnostics")
PARENT_RUN = "lpg01-0643fff9"
SUBRUN = "01c"

WEATHER_PATH_RE = re.compile(
    r"gsod|noaa|ncdc|ghcn|isd_|weather|climate|station-year", re.I)
SEC_FORM_RE = re.compile(
    r"\b(10[- ]?k|8[- ]?k|def[- ]?14a|proxy[- ]?statement)\b", re.I)

# ------------------------------------------------------------- rule sets
CAL_NAMES = {"date", "yearmoda", "yrmoda", "yr_mo_da", "datetime", "day",
             "mo_da"}
GEO_NAMES = {"zip", "zipcode", "fips", "geoid", "tract", "block",
             "blockgroup", "cbsa", "county_fips", "state_fips", "place_fips"}
ID_NAMES = {"id", "stn", "wban", "station", "uid", "guid", "key", "cik",
            "duns", "filer", "filer_id", "fec_id", "case_id", "parcel",
            "account"}
AMOUNT_TOKENS = {"amount", "amt", "value", "price", "cost", "pay", "salary",
                 "sal", "wage", "oblig", "obligation", "award", "funding",
                 "dollars", "revenue", "income", "assessed", "market", "sale"}
MEAS_TOKENS = {"temp", "dewp", "elev", "elevation", "depth", "pressure",
               "wind", "precip", "mass", "length", "height", "duration",
               "magnitude", "distance", "discharge", "flow"}
COUNT_TOKENS = {"count", "cnt", "n", "num", "nobs", "obs", "freq", "qty",
                "quantity", "pop", "population", "total", "records", "size",
                "length_bp"}
INDEX_TOKENS = {"index", "idx", "rank", "score"}
B01_ID_NAME_SET = {"id", "index", "key", "code", "uid", "guid"}
COLN_RE = re.compile(r"^col_\d+$")


def norm_name(name):
    s = str(name).strip().lower()
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)


def name_parts(norm):
    return [p for p in re.split(r"[^a-z0-9]+", norm) if p]


def is_yyyymmdd(v):
    if not (10000000 <= v <= 99999999):
        return False
    y, m, d = v // 10000, (v // 100) % 100, v % 100
    return 1800 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31


def is_yyyymm(v):
    if not (100000 <= v <= 999999):
        return False
    y, m = v // 100, v % 100
    return 1800 <= y <= 2100 and 1 <= m <= 12


def tag_row(r):
    """(channel_kind, confidence, rule) — name rules 1-7 first, then
    value/range rules in precedence order."""
    norm = norm_name(r.column_name)
    joined = "_".join(name_parts(norm))
    parts = set(name_parts(norm))

    if norm in CAL_NAMES or joined in CAL_NAMES:
        return "calendar", "high", f"calendar:name({norm})"
    if norm in GEO_NAMES or joined in GEO_NAMES or r.looks_like_geocode:
        return "geocode", "high", f"geocode:name({norm})"
    if norm in ID_NAMES or joined in ID_NAMES:
        return "identifier", "high", f"identifier:name({norm})"
    if parts & AMOUNT_TOKENS or any(p.startswith("oblig") for p in parts):
        tok = sorted(parts & AMOUNT_TOKENS)[0] if parts & AMOUNT_TOKENS \
            else "oblig*"
        return "amount", "high", f"amount:name({tok})"
    if parts & MEAS_TOKENS:
        return "measurement", "high", \
            f"measurement:name({sorted(parts & MEAS_TOKENS)[0]})"
    if parts & COUNT_TOKENS:
        return "count", "high", f"count:name({sorted(parts & COUNT_TOKENS)[0]})"
    if parts & INDEX_TOKENS:
        return "index", "high", f"index:name({sorted(parts & INDEX_TOKENS)[0]})"

    mn, mx = r.min, r.max
    if pd.notna(mn) and pd.notna(mx):
        mn_i, mx_i = int(mn), int(mx)
        if is_yyyymmdd(mn_i) and is_yyyymmdd(mx_i):
            return "calendar", "med", "calendar:range(yyyymmdd)"
        if is_yyyymm(mn_i) and is_yyyymm(mx_i):
            return "calendar", "med", "calendar:range(yyyymm)"
    if r.looks_like_id and norm not in B01_ID_NAME_SET:
        return "identifier", "med", "identifier:increasing_by_1"
    if COLN_RE.match(norm) and pd.notna(r.n_distinct_int) and \
            r.n_integer and r.n_distinct_int / r.n_integer > 0.95:
        return "identifier", "med", "identifier:colN_near_unique"
    if bool(r.looks_monetary):
        return "amount", "med", "amount:monetary_flag"
    return "other", "low", "other:no_rule_matched"


# ---------------------------------------------------------------- figures
PALETTE = {"calendar": "#2a78d6", "geocode": "#1baf7a",
           "identifier": "#eda100", "amount": "#008300",
           "measurement": "#4a3aa7", "count": "#e34948",
           "index": "#e87ba4", "other": "#898781"}
KIND_ORDER = ["calendar", "geocode", "identifier", "amount", "measurement",
              "count", "index", "other"]
SURFACE, GRID, BASELINE, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                       "#898781", "#0b0b0b")
FOOT = (f"run {PARENT_RUN} · sub-run {SUBRUN} · exclusions applied: weather, "
        "sec_filing · channel_kind is metadata only (never used in geometry)")


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def make_figures(dd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

    # -- composition by topic
    ct = (dd.groupby(["topic", "channel_kind"]).size().unstack(fill_value=0)
          .reindex(columns=KIND_ORDER, fill_value=0))
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    left = np.zeros(len(ct))
    y = np.arange(len(ct))
    for kind in KIND_ORDER:
        vals = ct[kind].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.62, color=PALETTE[kind],
                edgecolor=SURFACE, linewidth=0.8, label=kind)
        left += vals
    totals = ct.sum(axis=1).to_numpy()
    for yi, t in zip(y, totals):
        ax.text(t + max(totals) * 0.012, yi, f"{int(t):,}", va="center",
                fontsize=8, color=INK)
    ax.set_yticks(y, ct.index, fontsize=9, color=INK)
    ax.set_xlim(0, max(totals) * 1.12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("distinct kept channels (dup_key groups)", fontsize=9,
                  color=MUTED)
    ax.set_title("Kept distinct-channel composition by topic, segmented by "
                 "channel_kind (post-exclusion)", fontsize=11, color=INK,
                 loc="left", pad=14)
    ax.legend(ncols=4, fontsize=8, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, -0.02), labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig01c_composition_by_topic.pdf"),
                facecolor=SURFACE)
    plt.close(fig)

    # -- magnitude coverage
    d = dd[pd.notna(dd.band_num)]
    ct2 = (d.groupby(["band_num", "channel_kind"]).size()
           .unstack(fill_value=0).reindex(columns=KIND_ORDER, fill_value=0)
           .sort_index())
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ends = {}
    for kind in KIND_ORDER:
        s = ct2[kind].replace(0, np.nan)
        if s.notna().sum() == 0:
            continue
        ax.plot(ct2.index, s, color=PALETTE[kind], linewidth=1.6, marker="o",
                markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=kind)
        last = s.dropna().index[-1]
        ends[kind] = (last, s[last])
    ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(ct2.index.astype(int))
    ax.set_xlabel("magnitude band  —  floor(log10(max sampled value))",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("distinct kept channels (log scale)", fontsize=9,
                  color=MUTED)
    ax.set_title("Magnitude coverage of the kept distinct-channel set, by "
                 "channel_kind", fontsize=11, color=INK, loc="left", pad=14)
    top4 = ct2.sum().sort_values(ascending=False).index[:4]
    for kind in top4:
        if kind in ends:
            x, v = ends[kind]
            ax.annotate(kind, (x, v), xytext=(6, 0),
                        textcoords="offset points", fontsize=8, color=INK,
                        va="center")
    ax.legend(ncols=4, fontsize=8, frameon=False, loc="upper right",
              labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig01c_magnitude_coverage.pdf"),
                facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main

def modal(s):
    vc = s.value_counts()
    return vc.index[0] if len(vc) else ""


def band_sort(series):
    return series.map(lambda x: (x == "no_positive_max",
                                 int(x) if str(x).lstrip("-").isdigit()
                                 else 0))


def main():
    os.makedirs(DIAG, exist_ok=True)
    with open(os.path.join(CONFIG, "run_config_01c.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN,
            "build": "01c-apply-exclusions-freeze-analysis-corpus",
            "supersedes": "01b",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": ["tables/catalog.csv",
                       "tables/column_profiles.csv (n_distinct_int only)"],
            "exclusions": {
                "weather": "topic==meteorology OR path/member matches "
                           "gsod|noaa|ncdc|ghcn|isd_|weather|climate|"
                           "station-year (case-insensitive)",
                "sec_filing": "file_path contains risk-10-k, or a path "
                              "component 'sec', or filename matches "
                              r"\b(10[- ]?k|8[- ]?k|def[- ]?14a|"
                              r"proxy[- ]?statement)\b",
            },
            "dup_key_definition": "md5(basename(file or member)|column_name|"
                                  "frac_integer|min|max|n_integer) — content "
                                  "fingerprint, NOT path; station-years stay "
                                  "distinct unless sampled stats collide",
            "representative_rule": "lexicographically smallest (file_path, "
                                   "archive_member, sheet_or_table, "
                                   "column_name) per dup_key",
            "notes": "exclusion is a flag; nothing moved/deleted. "
                     "channel_kind + generating_process are metadata only.",
        }, f, indent=2)

    print("loading catalog...", flush=True)
    cat = pd.read_csv(os.path.join(TABLES, "catalog.csv"), low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        cat[c] = cat[c].fillna("")

    # ---- Step 1: exclusions
    pathblob = (cat.file_path.astype(str) + "\x00" +
                cat.archive_member.astype(str))
    weather = (cat.topic == "meteorology") | \
        pathblob.str.contains(WEATHER_PATH_RE)

    fname = cat.apply(lambda r: os.path.basename(r.archive_member or
                                                 r.file_path), axis=1)
    comps = cat.file_path.str.lower().str.split("/") + \
        cat.archive_member.str.lower().str.split("/")
    has_sec_comp = comps.map(lambda p: "sec" in p)
    sec = (cat.file_path.str.contains("risk-10-k", case=False) |
           has_sec_comp | fname.str.contains(SEC_FORM_RE))

    cat["excluded"] = weather | sec
    cat["exclude_rule"] = np.where(weather, "weather",
                                   np.where(sec, "sec_filing", ""))

    exc = cat[cat.excluded].sort_values(
        ["exclude_rule", "top_project", "file_path", "archive_member",
         "column_name"], kind="mergesort")
    exc[["exclude_rule", "topic", "top_project", "file_path",
         "archive_member", "column_name"]].to_csv(
        os.path.join(DIAG, "excluded_channels.csv"), index=False,
        encoding="utf-8")

    kept = cat[~cat.excluded].copy()
    removed_keys = set(cat.dup_key) - set(kept.dup_key)
    n_w = int((cat.exclude_rule == "weather").sum())
    n_s = int((cat.exclude_rule == "sec_filing").sum())
    print(f"excluded rows: weather={n_w:,}  sec_filing={n_s:,}")
    print(f"distinct channels fully removed: {len(removed_keys):,}")

    # ---- Step 2: dedup kept set
    keys = ["file_path", "archive_member", "sheet_or_table", "column_name"]
    prof = pd.read_csv(os.path.join(TABLES, "column_profiles.csv"),
                       usecols=keys + ["n_distinct_int"], low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        prof[c] = prof[c].fillna("")
    prof = prof.drop_duplicates(keys)
    kept = kept.merge(prof, on=keys, how="left")

    kept = kept.sort_values(keys, kind="mergesort")
    dd = kept.drop_duplicates("dup_key", keep="first").copy()
    dc = kept.groupby("dup_key").size()
    print(f"kept distinct channels: {len(dd):,} (from {len(kept):,} kept "
          f"rows); dup_count min={int(dc.min())} "
          f"median={float(dc.median()):.0f} max={int(dc.max())}")

    # ---- Step 3: channel_kind
    tags = [tag_row(r) for r in dd.itertuples()]
    dd["channel_kind"] = [t[0] for t in tags]
    dd["channel_kind_confidence"] = [t[1] for t in tags]
    dd["channel_kind_rule"] = [t[2] for t in tags]

    # ---- Step 4: composition tables
    dd["band_num"] = np.floor(pd.to_numeric(dd.log10_max, errors="coerce"))
    dd["n_records_num"] = pd.to_numeric(dd.n_records, errors="coerce")

    def agg_by(col):
        g = dd.groupby(col)
        out = pd.DataFrame({
            "distinct_channels": g.size(),
            "total_records": g.n_records_num.sum(),
            "median_log10_max": g.log10_max.median(),
        }).reset_index().sort_values(col)
        out["total_records"] = out.total_records.round(0).astype("Int64")
        out["median_log10_max"] = out.median_log10_max.round(3)
        return out

    agg_by("topic").to_csv(
        os.path.join(TABLES, "diag01c_topic_counts.csv"), index=False,
        encoding="utf-8")
    agg_by("channel_kind").to_csv(
        os.path.join(TABLES, "diag01c_kind_counts.csv"), index=False,
        encoding="utf-8")
    (dd.groupby(["topic", "channel_kind"]).size().unstack(fill_value=0)
       .reindex(columns=KIND_ORDER, fill_value=0).sort_index().reset_index()
       .to_csv(os.path.join(TABLES, "diag01c_topic_by_kind.csv"),
               index=False, encoding="utf-8"))

    bands = dd.copy()
    bands["magnitude_band"] = bands.band_num.map(
        lambda v: str(int(v)) if pd.notna(v) else "no_positive_max")
    (bands.groupby("magnitude_band").size().rename("distinct_channels")
     .reset_index().sort_values("magnitude_band", key=band_sort)
     .to_csv(os.path.join(TABLES, "diag01c_magnitude_bands.csv"),
             index=False, encoding="utf-8"))
    (bands.groupby(["magnitude_band", "channel_kind"]).size()
     .unstack(fill_value=0).reindex(columns=KIND_ORDER, fill_value=0)
     .reset_index().sort_values("magnitude_band", key=band_sort)
     .to_csv(os.path.join(TABLES, "diag01c_magnitude_by_kind.csv"),
             index=False, encoding="utf-8"))

    # ---- Step 4b: finance breakdown
    fin = dd[dd.topic == "finance"]
    blocks = []
    b = fin.groupby("top_project").size().rename("distinct_channels") \
        .reset_index()
    b.insert(0, "section", "by_top_project")
    b = b.rename(columns={"top_project": "key"})
    blocks.append(b.sort_values(["distinct_channels", "key"],
                                ascending=[False, True]))
    b = fin.groupby("channel_kind").size().rename("distinct_channels") \
        .reset_index()
    b.insert(0, "section", "by_channel_kind")
    b = b.rename(columns={"channel_kind": "key"})
    blocks.append(b.sort_values(["distinct_channels", "key"],
                                ascending=[False, True]))
    b = fin.groupby("column_name").size().rename("distinct_channels") \
        .reset_index().sort_values(["distinct_channels", "column_name"],
                                   ascending=[False, True]).head(40)
    b.insert(0, "section", "top40_column_names")
    b = b.rename(columns={"column_name": "key"})
    blocks.append(b)
    cell = fin.groupby(["top_project", "column_name"]).size()
    if len(cell):
        (cp, cc), cn = cell.idxmax(), int(cell.max())
        fin_share = cn / len(fin)
        b = pd.DataFrame([{"section": "largest_project_x_column_cell",
                           "key": f"{cp} × {cc}",
                           "distinct_channels": cn}])
        b["share_of_finance"] = round(fin_share, 4)
        blocks.append(b)
    else:
        fin_share, cp, cc = float("nan"), "", ""
    pd.concat(blocks, ignore_index=True).to_csv(
        os.path.join(TABLES, "diag01c_finance_breakdown.csv"), index=False,
        encoding="utf-8")

    # ---- Step 4b: numeric_analysis probe
    na = dd[dd.topic == "numeric_analysis"].copy()
    SYNTH_RE = re.compile(r"roughness|complexity|synthetic|sim|simulated|"
                          r"generated|test|benchmark|fixture", re.I)
    na_blob = na.file_path.astype(str) + "\x00" + \
        na.archive_member.astype(str)
    na["looks_synthetic"] = na_blob.str.contains(SYNTH_RE)
    blocks = []
    b = na.groupby("top_project").agg(
        distinct_channels=("dup_key", "size"),
        frac_looks_synthetic=("looks_synthetic", "mean")).reset_index()
    b.insert(0, "section", "by_top_project")
    b = b.rename(columns={"top_project": "key"})
    blocks.append(b.sort_values(["distinct_channels", "key"],
                                ascending=[False, True]))
    na["path_key"] = np.where(na.archive_member != "", na.archive_member,
                              na.file_path)
    b = na.groupby(na.path_key.map(os.path.dirname)).agg(
        distinct_channels=("dup_key", "size"),
        frac_looks_synthetic=("looks_synthetic", "mean")).reset_index() \
        .rename(columns={"path_key": "key"}) \
        .sort_values(["distinct_channels", "key"],
                     ascending=[False, True]).head(25)
    b.insert(0, "section", "top_dirs")
    blocks.append(b)
    b = na.groupby("column_name").agg(
        distinct_channels=("dup_key", "size"),
        modal_channel_kind=("channel_kind", modal),
        frac_looks_synthetic=("looks_synthetic", "mean")).reset_index() \
        .rename(columns={"column_name": "key"}) \
        .sort_values(["distinct_channels", "key"],
                     ascending=[False, True]).head(40)
    b.insert(0, "section", "top_column_names")
    blocks.append(b)
    out = pd.concat(blocks, ignore_index=True)
    for c in ("frac_looks_synthetic",):
        out[c] = out[c].round(4)
    out.to_csv(os.path.join(TABLES, "diag01c_numeric_analysis_probe.csv"),
               index=False, encoding="utf-8")

    # ---- Step 5: frozen candidate
    dd["excluded"] = False
    drop_helpers = ("band_num", "n_records_num", "path_key",
                    "looks_synthetic")
    out_cols = [c for c in dd.columns if c not in drop_helpers]
    dd[out_cols].sort_values(
        ["top_project", "file_path", "archive_member", "sheet_or_table",
         "column_name"], kind="mergesort").to_csv(
        os.path.join(TABLES, "analysis_corpus_candidate.csv"), index=False,
        encoding="utf-8")

    # ---- Step 6: figures
    make_figures(dd)

    # ---- summary
    kinds = dd.channel_kind.value_counts()
    topics = dd.topic.value_counts()
    na_synth = float(na.looks_synthetic.mean()) if len(na) else float("nan")
    print("\n================ BUILD 01c SUMMARY ================")
    print(f"run {PARENT_RUN} sub-run {SUBRUN} (supersedes 01b)")
    print(f"distinct channels removed: weather + sec_filing = "
          f"{len(removed_keys):,}")
    print(f"distinct channels KEPT: {len(dd):,}")
    print("top 8 remaining topics: " + ", ".join(
        f"{t}={n:,}" for t, n in topics.head(8).items()))
    print("channel_kind: " + ", ".join(
        f"{k}={int(n):,}" for k, n in kinds.reindex(KIND_ORDER).dropna()
        .items()))
    if len(fin):
        print(f"finance largest (project × column) cell: {cp} × {cc} = "
              f"{fin_share:.1%} of {len(fin):,} finance channels")
    print(f"numeric_analysis looks_synthetic: {na_synth:.1%} of {len(na):,} "
          f"channels (mostly synthetic: {na_synth > 0.5})")


if __name__ == "__main__":
    main()
