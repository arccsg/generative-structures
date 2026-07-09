"""Build 01d: resolve domains & dataset-families over the kept channel set.

Read-only against analysis_corpus_candidate.csv (+ column_profiles.csv for
full per-file column sets, + nothing else). Nothing dropped, capped, or
supplemented — labels and groups only. No factorization, no L-profile.

domain / dataset_family are structural bookkeeping; generating_process and
channel_kind remain metadata only, never fed into geometry.
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT, ROOT

DIAG = os.path.join(OUT, "diagnostics")
PARENT_RUN = "lpg01-0643fff9"
SUBRUN = "01d"

GENERIC_DIRS = {"data", "raw", "csv", "processed", "corpus", "files",
                "downloads", "tmp", "out", "output", "final", "clean",
                "staging"}
GENERIC_LABEL_TOKENS = GENERIC_DIRS | {"public", "private", "archive", "agg",
                                       "full", "daily", "test", "misc", "db"}

# ---- domain token map: (domain, regex on path+stem blob, regex on colset
#      blob). Path/stem hit = high confidence; column-only hit = med.
#      First rule that fires wins (spec order).
DOMAIN_RULES = [
    ("taxi_trips", r"taxi|tlc|tripdata|\btrips?\b|fare|pickup|dropoff",
     r"fare|pickup|dropoff|trip_distance"),
    # extensions mined from the first unresolved pass (spec: "extend as you
    # find more"):
    ("campaign_finance", r"itcont|\bfec\b|indiv2\d|campaign",
     r"cmte_id|transaction_amt"),
    ("corporate_financials", r"wrds|compustat|10x_|edgar",
     r"\bcik\b.*\bgvkey\b|gvkey"),
    ("sports", r"\bmlb\b|marathon|batting|pitching|llimllib",
     r"\bab\b.*\bhr\b|at_bats|innings"),
    ("procurement", r"procure|contract|award|obligation|solicitation|"
     r"usaspending", r"obligation|award|solicitation"),
    ("census", r"census|\bacs\b|pums|psam|\btract\b|decennial",
     r"pums|tract|puma"),
    ("payments_fraud", r"fraud|ieee", r"transactionid|card1"),
    ("equity_markets", r"equity|\btick\b|ohlc|ticker|_daily_full|"
     r"stock|sp500|nasdaq", r"\bclose\b|close_tick|ohlc|\bvolume\b|ticker"),
    ("network_telemetry", r"netflow|n_bytes|n_packets|sum_n_dest",
     r"n_bytes|n_packets|n_flows|sum_n_dest"),
    ("crime", r"crime|arrest|incident", r"arrest|incident"),
    ("seismology", r"quake|earthquake|seismic|usgs",
     r"magnitude|depth_km|seismic"),
    ("genomics", r"gene|genome|\bcds\b|intron|exon|transcript|ensembl",
     r"\bcds\b|intron|exon|transcript|ensembl|length_bp"),
    ("admin_codes", r"naics|agency_code|duns|\bcik\b|filer",
     r"naics|agency_code|duns|\bcik\b|filer"),
    ("accounting", r"accounting|ledger|gl_|journal|invoice",
     r"ledger|journal|invoice"),
    ("health", r"health|patient|\bclaim s?\b|\bclaims?\b|icd|hospital|"
     r"medicare|medicaid|\bcms\b|leie|mup_phy",
     r"patient|icd|hospital|bene_"),
]
# "flow" alone is too weak for network_telemetry (streamflow/cash flow);
# it only counts alongside another telemetry token — recorded in run config.

QUANTITIES = {"n_bytes", "n_packets", "n_flows", "sum_n_dest_ip",
              "sum_n_dest_asn", "sum_n_dest_ports", "id_time"}
GRAIN_RE = re.compile(r"agg_\d+_(minute|min|hour|day)s?|\b(minute|hourly|"
                      r"hour|daily|day)\b", re.I)

STOP_HINTS = GENERIC_LABEL_TOKENS | {"users", "<ANON>", "projects",
                                     "zip", "col", "unnamed", "value", "id",
                                     "name", "type", "code", "cbad",
                                     "ourosboros", "deprroughness", "index",
                                     "roughness", "complexity", "methods",
                                     "concentration", "prime", "depr"}


def slugify(s, maxlen=48):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-").lower()
    return s[:maxlen] or "x"


def rel_dir_of(r):
    """Directory path relative to top_project, generic components stripped;
    for archive members, the zip's dir (relative) + the member's inner dir."""
    proj_root = os.path.join(ROOT, r.top_project)
    comps = []
    d = os.path.relpath(os.path.dirname(r.file_path), proj_root)
    if d != ".":
        comps += d.split(os.sep)
    if r.archive_member:
        md = os.path.dirname(r.archive_member)
        if md:
            comps += md.split("/")
    return "/".join(c for c in comps if c.lower() not in GENERIC_DIRS)


def stem_of(r):
    base = os.path.basename(r.archive_member or r.file_path)
    base = os.path.splitext(base)[0]
    if re.fullmatch(r"\d+", base):
        return ""
    # strip trailing shard/window/date suffixes, then mask long digit runs
    base = re.sub(r"(?:[_\-\. ]+(?:\d+|\d{4}-\d{2}(?:-\d{2})?|[qQ]\d))+$",
                  "", base)
    if re.fullmatch(r"[\d#_\-. ]*", base):   # bare shard number (e.g. 100__1)
        return ""
    return re.sub(r"\d{4,}", "#", base)


def family_label(stem, rel_dir, top_project):
    cands = ([stem] if stem else []) + list(reversed(rel_dir.split("/")))
    for c in cands:
        toks = [t for t in re.split(r"[^A-Za-z0-9#]+", c) if t]
        if not toks:
            continue
        informative = [t for t in toks
                       if t.lower() not in GENERIC_LABEL_TOKENS and
                       not t.replace("#", "").isdigit()]
        if informative:
            return c
    return top_project


def hint_tokens(blob, n=3):
    toks = [t.lower() for t in re.split(r"[^A-Za-z]+", blob)
            if len(t) >= 3 and t.lower() not in STOP_HINTS]
    return [t for t, _ in Counter(toks).most_common(n)]


def conf_median(series):
    m = {"high": 3, "med": 2, "low": 1}
    inv = {3: "high", 2: "med", 1: "low"}
    vals = series.map(m).dropna()
    return inv[int(np.median(vals))] if len(vals) else ""


def main():
    os.makedirs(DIAG, exist_ok=True)
    with open(os.path.join(CONFIG, "run_config_01d.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN,
            "build": "01d-resolve-domains-and-dataset-families",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": ["tables/analysis_corpus_candidate.csv",
                       "tables/column_profiles.csv (column sets only)"],
            "family_definition": "(top_project, rel_dir stripped of generic "
                                 "components, filename stem with numeric/"
                                 "date/window suffixes stripped and long "
                                 "digit runs masked, colset_sig = md5 of the "
                                 "file's full sorted column-name set)",
            "deviations": ["'flow' alone does not fire network_telemetry "
                           "(would hit streamflow/cash-flow); it requires a "
                           "second telemetry token (n_bytes/n_packets/"
                           "sum_n_dest/netflow)"],
            "notes": "nothing dropped or capped; domain/dataset_family are "
                     "structural bookkeeping; channel_kind and "
                     "generating_process stay metadata-only.",
        }, f, indent=2)

    print("loading candidate set...", flush=True)
    dd = pd.read_csv(os.path.join(TABLES, "analysis_corpus_candidate.csv"),
                     low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        dd[c] = dd[c].fillna("")
    assert len(dd) > 0

    # ---- full column set per file (from column_profiles)
    keys = ["file_path", "archive_member", "sheet_or_table"]
    fkeys = dd[keys].drop_duplicates()
    prof = pd.read_csv(os.path.join(TABLES, "column_profiles.csv"),
                       usecols=keys + ["column_name"], low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        prof[c] = prof[c].fillna("")
    prof = prof.merge(fkeys, on=keys, how="inner")
    colsets = (prof.groupby(keys).column_name
               .agg(lambda s: sorted(set(s.astype(str)))).rename("colset")
               .reset_index())
    dd = dd.merge(colsets, on=keys, how="left")
    dd["colset"] = dd.colset.map(
        lambda v: v if isinstance(v, list) else [])
    dd["colset_sig"] = dd.colset.map(
        lambda cs: hashlib.md5("|".join(cs).encode()).hexdigest()[:8])

    # ---- Step 1: dataset_family
    dd["rel_dir"] = [rel_dir_of(r) for r in dd.itertuples()]
    dd["stem"] = [stem_of(r) for r in dd.itertuples()]
    dd["dataset_family"] = [
        f"{slugify(r.top_project, 24)}__{slugify(r.rel_dir)}__"
        f"{slugify(r.stem, 32)}__{r.colset_sig[:6]}"
        for r in dd.itertuples()]
    dd["family_label"] = [family_label(r.stem, r.rel_dir, r.top_project)
                          for r in dd.itertuples()]

    # ---- Step 2: domain (family-level, independent of Build-01 topic)
    fam_blob = {}
    for fam, g in dd.groupby("dataset_family"):
        r = g.iloc[0]
        pathblob = f"{r.rel_dir} {r.stem} " + \
            os.path.basename(r.archive_member or r.file_path)
        colblob = " ".join(r.colset)
        fam_blob[fam] = (pathblob.lower(), colblob.lower())

    fam_domain = {}
    for fam, (pblob, cblob) in fam_blob.items():
        hit = None
        for dom, ppat, cpat in DOMAIN_RULES:
            m = re.search(ppat, pblob)
            if m:
                hit = (dom, "high", f"{dom}:path({m.group(0).strip()})")
                break
            m = re.search(cpat, cblob)
            if m:
                hit = (dom, "med", f"{dom}:column({m.group(0).strip()})")
                break
        if hit is None:
            hints = hint_tokens(pblob + " " + cblob)
            hit = ("unresolved", "low", "unresolved", ",".join(hints))
        fam_domain[fam] = hit

    dd["domain"] = dd.dataset_family.map(lambda f: fam_domain[f][0])
    dd["domain_confidence"] = dd.dataset_family.map(lambda f: fam_domain[f][1])
    dd["domain_rule"] = dd.dataset_family.map(lambda f: fam_domain[f][2])
    dd["domain_hint"] = dd.dataset_family.map(
        lambda f: fam_domain[f][3] if len(fam_domain[f]) > 3 else "")

    # ---- Step 3: telemetry substructure
    tel = dd.domain == "network_telemetry"
    dd["quantity"] = np.where(
        tel, dd.column_name.where(dd.column_name.isin(QUANTITIES), "other"),
        "")

    def grain_of(r):
        m = GRAIN_RE.search(f"{r.file_path} {r.archive_member}")
        if not m:
            return "unknown"
        u = (m.group(1) or m.group(2)).lower()
        return {"min": "minute", "minute": "minute", "hourly": "hour",
                "hour": "hour", "daily": "day", "day": "day"}[u]

    dd["grain"] = [grain_of(r) if t else "" for r, t in
                   zip(dd.itertuples(), tel)]

    dd["n_records_num"] = pd.to_numeric(dd.n_records, errors="coerce")
    sub = dd[tel].groupby(["quantity", "grain"]).agg(
        channels=("dup_key", "size"),
        total_records=("n_records_num", "sum"),
        log10_min=("log10_min", "min"),
        log10_max=("log10_max", "max")).reset_index() \
        .sort_values(["quantity", "grain"])
    sub["total_records"] = sub.total_records.round(0).astype("Int64")
    for c in ("log10_min", "log10_max"):
        sub[c] = sub[c].round(3)
    sub.to_csv(os.path.join(TABLES, "telemetry_substructure.csv"),
               index=False, encoding="utf-8")

    # ---- Step 4: inventories
    g = dd.groupby("domain")
    dom_inv = pd.DataFrame({
        "dataset_families": g.dataset_family.nunique(),
        "channels": g.size(),
        "total_records": g.n_records_num.sum(),
        "log10_min": g.log10_min.min(),
        "log10_max": g.log10_max.max(),
        "dominant_channel_kind": g.channel_kind.agg(
            lambda s: s.value_counts().index[0]),
        "median_domain_confidence": g.domain_confidence.agg(conf_median),
    }).reset_index().sort_values(["dataset_families", "domain"],
                                 ascending=[False, True])
    dom_inv["total_records"] = dom_inv.total_records.round(0).astype("Int64")
    for c in ("log10_min", "log10_max"):
        dom_inv[c] = dom_inv[c].round(3)
    dom_inv.to_csv(os.path.join(TABLES, "domain_inventory.csv"), index=False,
                   encoding="utf-8")

    g = dd.groupby("dataset_family")
    fam_inv = pd.DataFrame({
        "family_label": g.family_label.first(),
        "domain": g.domain.first(),
        "top_project": g.top_project.first(),
        "channels": g.size(),
        "column_set": g.colset.first().map(
            lambda cs: "|".join(cs)[:400]),
        "total_records": g.n_records_num.sum(),
        "log10_min": g.log10_min.min(),
        "log10_max": g.log10_max.max(),
    }).reset_index().sort_values("dataset_family")
    fam_inv["total_records"] = fam_inv.total_records.round(0).astype("Int64")
    for c in ("log10_min", "log10_max"):
        fam_inv[c] = fam_inv[c].round(3)
    fam_inv.to_csv(os.path.join(TABLES, "family_inventory.csv"), index=False,
                   encoding="utf-8")

    # ---- topic ↔ domain disagreement (vocabulary-aware)
    EQUIV = {"census": {"census"}, "crime": {"crime"}, "health": {"health"},
             "accounting": {"accounting"}, "geophysics": {"seismology"},
             "biology": {"genomics"},
             "government_procurement": {"procurement", "admin_codes"},
             "finance": {"equity_markets", "payments_fraud"}}
    named = dd.domain != "unresolved"
    equiv_ok = [r.domain in EQUIV.get(r.topic, set())
                for r in dd.itertuples()]
    dis = dd[named & ~pd.Series(equiv_ok, index=dd.index)].copy()
    dis["disagreement_type"] = [
        "refined" if r.topic not in EQUIV else "conflict"
        for r in dis.itertuples()]
    dis[["dataset_id", "top_project", "file_path", "archive_member",
         "column_name", "topic", "topic_confidence", "domain",
         "domain_confidence", "domain_rule", "dataset_family",
         "disagreement_type"]].sort_values(
        ["disagreement_type", "topic", "domain", "file_path", "column_name"],
        kind="mergesort").to_csv(
        os.path.join(TABLES, "domain_topic_disagreement.csv"), index=False,
        encoding="utf-8")

    unres = dd[~named]
    hint_rank = Counter()
    for h in unres.domain_hint:
        for t in str(h).split(","):
            if t:
                hint_rank[t] += 1
    u = unres[["dataset_id", "top_project", "file_path", "archive_member",
               "column_name", "dataset_family", "family_label",
               "domain_hint", "topic"]].copy()
    u["hint_freq"] = u.domain_hint.map(
        lambda h: max([hint_rank[t] for t in str(h).split(",") if t],
                      default=0))
    u.sort_values(["hint_freq", "dataset_family", "column_name"],
                  ascending=[False, True, True], kind="mergesort").to_csv(
        os.path.join(DIAG, "unresolved_domains.csv"), index=False,
        encoding="utf-8")

    # ---- Step 5: figures
    make_figures(dom_inv)

    # ---- Step 6: resolved corpus
    drop_helpers = {"colset", "colset_sig", "rel_dir", "stem",
                    "n_records_num"}
    out_cols = [c for c in dd.columns if c not in drop_helpers]
    dd[out_cols].sort_values(
        ["top_project", "file_path", "archive_member", "sheet_or_table",
         "column_name"], kind="mergesort").to_csv(
        os.path.join(TABLES, "analysis_corpus_resolved.csv"), index=False,
        encoding="utf-8")

    # ---- summary
    coarse = dd.topic.isin(["unknown", "numeric_analysis"]) | \
        (dd.topic_confidence == "low")
    moved = int((coarse & named).sum())
    print("\n================ BUILD 01d SUMMARY ================")
    print(f"run {PARENT_RUN} sub-run {SUBRUN}")
    print(f"distinct domains resolved (excl. unresolved): "
          f"{dd[named].domain.nunique()}")
    print(f"dataset-families: {dd.dataset_family.nunique():,} "
          f"(over {len(dd):,} channels)")
    print("\ndomain inventory (by dataset-families):")
    for r in dom_inv.itertuples():
        print(f"  {r.domain:<20} families={r.dataset_families:<5} "
              f"channels={r.channels:<7} kind={r.dominant_channel_kind:<11} "
              f"conf={r.median_domain_confidence}")
    print(f"\nrows moved out of unknown/coarse topics into a named domain: "
          f"{moved:,}")
    print(f"unresolved rows: {len(unres):,} in "
          f"{unres.dataset_family.nunique()} families; top 10 hint tokens: "
          + ", ".join(f"{t}({n})" for t, n in hint_rank.most_common(10)))


def make_figures(dom_inv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    SURFACE, GRID, BASELINE, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                           "#898781", "#0b0b0b")
    BLUE, AQUA = "#2a78d6", "#1baf7a"
    FOOT = (f"run {PARENT_RUN} · sub-run {SUBRUN} · unit of analysis: "
            "dataset-family · nothing dropped or capped")

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASELINE)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)

    inv = dom_inv.sort_values("dataset_families", ascending=True)
    y = np.arange(len(inv))

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.barh(y, inv.dataset_families, height=0.6, color=BLUE,
            edgecolor=SURFACE, linewidth=0.8)
    for yi, v in zip(y, inv.dataset_families):
        ax.text(v + inv.dataset_families.max() * 0.012, yi, f"{int(v):,}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(y, inv.domain, fontsize=9, color=INK)
    ax.set_xlim(0, inv.dataset_families.max() * 1.12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("dataset-families", fontsize=9, color=MUTED)
    ax.set_title("Dataset-families per domain — the corpus's honest "
                 "diversity", fontsize=11, color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig01d_families_per_domain.pdf"),
                facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    h = 0.36
    ax.barh(y + h / 2 + 0.02, inv.channels, height=h, color=BLUE,
            edgecolor=SURFACE, linewidth=0.8, label="channels")
    ax.barh(y - h / 2 - 0.02, inv.dataset_families, height=h, color=AQUA,
            edgecolor=SURFACE, linewidth=0.8, label="dataset-families")
    for yi, v in zip(y + h / 2 + 0.02, inv.channels):
        ax.text(v * 1.08, yi, f"{int(v):,}", va="center", fontsize=7,
                color=INK)
    for yi, v in zip(y - h / 2 - 0.02, inv.dataset_families):
        ax.text(v * 1.08, yi, f"{int(v):,}", va="center", fontsize=7,
                color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y, inv.domain, fontsize=9, color=INK)
    ax.set_xlim(0.8, inv.channels.max() * 3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("count (log scale)", fontsize=9, color=MUTED)
    ax.set_title("Channels vs dataset-families per domain — why the unit "
                 "of analysis matters", fontsize=11, color=INK, loc="left",
                 pad=12)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig01d_channels_vs_families.pdf"),
                facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
