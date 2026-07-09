"""Build 01b: corpus diagnostic & channel-kind tagging.

Reads ONLY Build 01 outputs (tables/catalog.csv + tables/column_profiles.csv
for n_distinct_int). No filesystem re-walk, no factorization, no L-profile.

channel_kind is METADATA ONLY — like generating_process, it must never be fed
into any clustering, projection, distance, or geometry downstream.

dup_key (from Build 01) is a content FINGERPRINT, not a full content hash:
md5(basename(file or member) | column_name | frac_integer | min | max |
n_integer). Two GSOD station-years share a basename + columns but differ in
their value fingerprints, so station-years remain distinct dup_keys unless
their sampled stats collide exactly.
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
SUBRUN = "01b"

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
MEAS_TOKENS = {"temp", "dewp", "slp", "stp", "visib", "wdsp", "mxspd", "gust",
               "max", "min", "prcp", "sndp", "elev", "elevation", "depth",
               "pressure", "wind", "precip", "mass", "length", "height",
               "duration", "magnitude", "distance", "discharge", "flow"}
COUNT_TOKENS = {"count", "cnt", "n", "num", "nobs", "obs", "freq", "qty",
                "quantity", "pop", "population", "total", "records", "size",
                "length_bp"}
INDEX_TOKENS = {"index", "idx", "rank", "score"}
# build-01 looks_like_id fires on these NAMES too; excluding them isolates
# the strictly-increasing-by-1 evidence
B01_ID_NAME_SET = {"id", "index", "key", "code", "uid", "guid"}

COLN_RE = re.compile(r"^col_\d+$")


def norm_name(name):
    s = str(name).strip().lower()
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)   # strip STN--- style edges
    return s


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
    """Return (channel_kind, confidence, rule). Name-match beats range-match:
    all name rules (precedence 1-7) run before any value/range rule."""
    norm = norm_name(r.column_name)
    joined = "_".join(name_parts(norm))
    parts = set(name_parts(norm))

    # ---- pass A: name matches, precedence order 1..7
    if norm in CAL_NAMES or joined in CAL_NAMES:
        return "calendar", "high", f"calendar:name({norm})"
    if norm in GEO_NAMES or joined in GEO_NAMES or r.looks_like_geocode:
        return "geocode", "high", f"geocode:name({norm})"
    if norm in ID_NAMES or joined in ID_NAMES:
        return "identifier", "high", f"identifier:name({norm})"
    amount_hit = parts & AMOUNT_TOKENS or \
        any(p.startswith("oblig") for p in parts)
    if amount_hit:
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

    # ---- pass B: value/range evidence, precedence order
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

PALETTE = {  # validated light-mode categorical slots, fixed order
    "calendar": "#2a78d6", "geocode": "#1baf7a", "identifier": "#eda100",
    "amount": "#008300", "measurement": "#4a3aa7", "count": "#e34948",
    "index": "#e87ba4", "other": "#898781",   # residual category: muted ink
}
KIND_ORDER = ["calendar", "geocode", "identifier", "amount", "measurement",
              "count", "index", "other"]
SURFACE, GRID, BASELINE, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                       "#898781", "#0b0b0b")


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def fig_composition(dd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

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
    ax.set_xlabel("distinct channels (dup_key groups)", fontsize=9,
                  color=MUTED)
    ax.set_title("Distinct-channel composition by topic, segmented by "
                 "channel_kind", fontsize=11, color=INK, loc="left", pad=14)
    ax.legend(ncols=4, fontsize=8, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, -0.02), labelcolor=INK)
    fig.text(0.01, 0.01, f"run {PARENT_RUN} · sub-run {SUBRUN} · "
             "channel_kind is metadata only (never used in any geometry)",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig_composition_by_topic.pdf"),
                facecolor=SURFACE)
    plt.close(fig)


def fig_magnitude(dd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

    d = dd[pd.notna(dd.band_num)].copy()
    ct = (d.groupby(["band_num", "channel_kind"]).size().unstack(fill_value=0)
          .reindex(columns=KIND_ORDER, fill_value=0).sort_index())

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ends = {}
    for kind in KIND_ORDER:
        s = ct[kind].replace(0, np.nan)
        if s.notna().sum() == 0:
            continue
        ax.plot(ct.index, s, color=PALETTE[kind], linewidth=1.6,
                marker="o", markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, label=kind)
        last = s.dropna().index[-1]
        ends[kind] = (last, s[last])
    ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(ct.index.astype(int))
    ax.set_xlabel("magnitude band  —  floor(log10(max sampled value))",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("distinct channels (log scale)", fontsize=9, color=MUTED)
    ax.set_title("Magnitude coverage of the distinct-channel set, by "
                 "channel_kind", fontsize=11, color=INK, loc="left", pad=14)
    # direct labels for the four largest series (relief for low-contrast hues)
    top4 = ct.sum().sort_values(ascending=False).index[:4]
    for kind in top4:
        if kind in ends:
            x, v = ends[kind]
            ax.annotate(kind, (x, v), xytext=(6, 0),
                        textcoords="offset points", fontsize=8,
                        color=INK, va="center")
    ax.legend(ncols=4, fontsize=8, frameon=False, loc="upper right",
              labelcolor=INK)
    fig.text(0.01, 0.01, f"run {PARENT_RUN} · sub-run {SUBRUN} · "
             "channel_kind is metadata only (never used in any geometry)",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig_magnitude_coverage.pdf"),
                facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main

def modal(s):
    vc = s.value_counts()
    return vc.index[0] if len(vc) else ""


def main():
    os.makedirs(DIAG, exist_ok=True)
    with open(os.path.join(CONFIG, "run_config_01b.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN,
            "build": "01b-corpus-diagnostic-and-channel-kind-tagging",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": ["tables/catalog.csv",
                       "tables/column_profiles.csv (n_distinct_int only)"],
            "dup_key_definition": "md5(basename(file or member)|column_name|"
                                  "frac_integer|min|max|n_integer) — content "
                                  "fingerprint, NOT path: GSOD station-years "
                                  "stay distinct unless sampled stats collide",
            "representative_rule": "lexicographically smallest (file_path, "
                                   "archive_member, sheet_or_table, "
                                   "column_name) per dup_key",
            "notes": "channel_kind + generating_process are metadata only; "
                     "never fed to clustering/projection/distance/geometry",
        }, f, indent=2)

    print("loading catalog...", flush=True)
    cat = pd.read_csv(os.path.join(TABLES, "catalog.csv"), low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        cat[c] = cat[c].fillna("")

    print("joining n_distinct_int from column_profiles...", flush=True)
    keys = ["file_path", "archive_member", "sheet_or_table", "column_name"]
    prof = pd.read_csv(os.path.join(TABLES, "column_profiles.csv"),
                       usecols=keys + ["n_distinct_int"], low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        prof[c] = prof[c].fillna("")
    prof = prof.drop_duplicates(keys)
    cat = cat.merge(prof, on=keys, how="left")

    # ---- Step 1: dedup to distinct contents
    cat = cat.sort_values(["file_path", "archive_member", "sheet_or_table",
                           "column_name"], kind="mergesort")
    dd = cat.drop_duplicates("dup_key", keep="first").copy()
    dc = cat.groupby("dup_key").size()
    print(f"distinct dup_keys: {len(dd):,} (from {len(cat):,} catalog rows)")
    print(f"dup_count distribution: min={int(dc.min())} "
          f"median={float(dc.median()):.0f} max={int(dc.max())}")

    # ---- Step 2: tag channel_kind
    print("tagging channel_kind...", flush=True)
    tags = [tag_row(r) for r in dd.itertuples()]
    dd["channel_kind"] = [t[0] for t in tags]
    dd["channel_kind_confidence"] = [t[1] for t in tags]
    dd["channel_kind_rule"] = [t[2] for t in tags]

    # ---- Step 3: composition & magnitude tables
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

    agg_by("topic").to_csv(os.path.join(TABLES, "diag_topic_counts.csv"),
                           index=False, encoding="utf-8")
    agg_by("channel_kind").to_csv(
        os.path.join(TABLES, "diag_kind_counts.csv"), index=False,
        encoding="utf-8")

    (dd.groupby(["topic", "channel_kind"]).size().unstack(fill_value=0)
       .reindex(columns=KIND_ORDER, fill_value=0).sort_index().reset_index()
       .to_csv(os.path.join(TABLES, "diag_topic_by_kind.csv"), index=False,
               encoding="utf-8"))

    bands = dd.copy()
    bands["magnitude_band"] = bands.band_num.map(
        lambda v: str(int(v)) if pd.notna(v) else "no_positive_max")
    (bands.groupby("magnitude_band").size().rename("distinct_channels")
     .reset_index()
     .sort_values("magnitude_band",
                  key=lambda s: s.map(lambda x: (x == "no_positive_max",
                                                 int(x) if x.lstrip("-")
                                                 .isdigit() else 0)))
     .to_csv(os.path.join(TABLES, "diag_magnitude_bands.csv"), index=False,
             encoding="utf-8"))

    (bands.groupby(["magnitude_band", "channel_kind"]).size()
     .unstack(fill_value=0).reindex(columns=KIND_ORDER, fill_value=0)
     .reset_index()
     .sort_values("magnitude_band",
                  key=lambda s: s.map(lambda x: (x == "no_positive_max",
                                                 int(x) if x.lstrip("-")
                                                 .isdigit() else 0)))
     .to_csv(os.path.join(TABLES, "diag_magnitude_by_kind.csv"), index=False,
             encoding="utf-8"))

    def colname_table(sub, top_n):
        g = sub.groupby("column_name")
        t = pd.DataFrame({
            "count": g.size(),
            "modal_channel_kind": g.channel_kind.agg(modal),
            "example_min": g["min"].first(),
            "example_max": g["max"].first(),
            "example_log10_min": g.log10_min.first(),
            "example_log10_max": g.log10_max.first(),
        }).reset_index()
        return t.sort_values(["count", "column_name"],
                             ascending=[False, True]).head(top_n)

    met = dd[dd.topic == "meteorology"]
    colname_table(met, 50).to_csv(
        os.path.join(TABLES, "diag_meteorology_columns.csv"), index=False,
        encoding="utf-8")
    colname_table(dd, 100)[["column_name", "count", "modal_channel_kind"]] \
        .to_csv(os.path.join(TABLES, "diag_column_names_overall.csv"),
                index=False, encoding="utf-8")

    # ---- Step 4: figures
    print("rendering figures...", flush=True)
    fig_composition(dd)
    fig_magnitude(dd)

    # ---- Step 5: frozen deduped tagged catalog
    out_cols = [c for c in dd.columns if c not in
                ("band_num", "n_records_num", "n_distinct_int")] + \
               ["n_distinct_int"]
    dd_out = dd[out_cols].sort_values(
        ["top_project", "file_path", "archive_member", "sheet_or_table",
         "column_name"], kind="mergesort")
    dd_out.to_csv(os.path.join(TABLES, "catalog_dedup_tagged.csv"),
                  index=False, encoding="utf-8")

    unresolved = dd_out[dd_out.channel_kind == "other"][
        ["dataset_id", "top_project", "file_path", "archive_member",
         "column_name", "min", "max", "n_integer", "n_distinct_int",
         "topic", "notes"]]
    unresolved.to_csv(os.path.join(DIAG, "kind_unresolved.csv"), index=False,
                      encoding="utf-8")

    # ---- summary
    met_calid = float((met.channel_kind.isin(["calendar", "identifier"]))
                      .mean()) if len(met) else float("nan")
    kinds = dd.channel_kind.value_counts()
    topics = dd.topic.value_counts()
    print("\n================ BUILD 01b SUMMARY ================")
    print(f"run {PARENT_RUN} sub-run {SUBRUN}")
    print(f"distinct channels: {len(dd):,}")
    print("top 6 topics: " + ", ".join(
        f"{t}={n:,}" for t, n in topics.head(6).items()))
    print("channel_kind: " + ", ".join(
        f"{k}={n:,}" for k, n in kinds.reindex(KIND_ORDER).dropna()
        .astype(int).items()))
    print(f"meteorology channels tagged calendar/identifier: "
          f"{met_calid:.1%}  ({len(met):,} meteorology channels total)")
    print(f"unresolved (channel_kind=other): {len(unresolved):,} "
          f"→ diagnostics/kind_unresolved.csv")


if __name__ == "__main__":
    main()
