"""Stage 4: assemble the channel-candidate catalog with best-effort
interpretive labels (topic / subtopic / generating_process / source).

generating_process is metadata only — recorded, never to be conditioned on
downstream (no clustering, projection, or geometry may use it).
"""
import csv
import hashlib
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, TABLES, LOGS, top_project

csv.field_size_limit(10_000_000)

# ---------------------------------------------------------- topic rules
# (token-in-path/filename, topic, subtopic, confidence) — first match wins;
# ordered most-specific first.
TOPIC_RULES = [
    (r"gsod", "meteorology", "surface_weather_daily", "high"),
    (r"noaa|weather|climate", "meteorology", "weather_observations", "med"),
    (r"ieee-fraud|train_transaction|train_identity|test_transaction|test_identity", "finance", "payment_fraud_detection", "med"),
    (r"institutions/agg_", "finance", "aggregated_time_series", "low"),
    (r"psam_pus|psam_hus|pums", "census", "acs_pums_microdata", "high"),
    (r"chicago_crimes", "crime", "incident_reports", "high"),
    (r"traffic_violations", "crime", "traffic_citations", "high"),
    (r"fec|itcont|indiv2[0-9]|campaign", "finance", "campaign_contributions", "high"),
    (r"all_assistance|assistance_primeaward", "government_procurement", "federal_assistance", "high"),
    (r"all_contracts|usaspending|fpds", "government_procurement", "federal_contracts", "high"),
    (r"cms_leie|leie", "health", "provider_exclusions", "high"),
    (r"mup_phy|medicare|medicaid|cms", "health", "medicare_utilization", "med"),
    (r"mortality|death|cdc|wonder", "health", "mortality", "med"),
    (r"10-?k|edgar|sec_filing|risk-10-k", "finance", "sec_filings", "med"),
    (r"stock|ticker|ohlc|sp500|nasdaq|djia", "finance", "equity_markets", "med"),
    (r"acs_|american_community|census|cbp_|county_business", "census", "population_business", "med"),
    (r"qcew|bls|unemploy|employment|labor|wage", "employment", "labor_statistics", "med"),
    (r"irs|tax|soi_", "taxation", "tax_records", "med"),
    (r"earthquake|seismic|usgs|quake|geomag", "geophysics", "seismology", "med"),
    (r"genome|gene_|protein|rna|dna", "biology", "genomics", "med"),
    (r"msa|cbsa|metro", "geography", "metro_areas", "med"),
    # NB: must not match the ".zip" archive extension present in every
    # zip-member path — only zip-as-a-word/token
    (r"zip_?code|(^|[_\s/])zip([_\s/]|$)|fips|geoid|(^|[_\s/])tract([_\s/]|$)|(^|[_\s/])county([_\s/]|$)", "geography", "geocoding", "low"),
    (r"abm|agent_|simulation|synthetic|montecarlo|monte_carlo", "simulation", "generated_data", "med"),
    (r"audit", "accounting", "audit_analytics", "med"),
    (r"benford|digit", "numeric_analysis", "digit_distributions", "low"),
]

PROJECT_TOPIC = {
    "audit-analytics": ("accounting", "audit_analytics", "med"),
    "cbad-audit": ("accounting", "audit_analytics", "med"),
    "risk-10-k": ("finance", "sec_filings", "med"),
    "strategic-mortality": ("health", "mortality", "med"),
    "erm-co-indicators": ("risk_management", "enterprise_risk", "med"),
    "erm-hidden-network": ("risk_management", "enterprise_risk", "med"),
    "spire-abm": ("simulation", "agent_based_model", "med"),
    "spire-abm-1000": ("simulation", "agent_based_model", "med"),
    "waterfall-msa-deployed": ("geography", "metro_areas", "low"),
    "msa": ("geography", "metro_areas", "low"),
    "structural-risk": ("risk_management", "structural_risk", "low"),
    "<GEN-SIM-PROJECT>": ("simulation", "generated_data", "low"),
    "prime-factorization": ("numeric_analysis", "factorization_inputs", "low"),
    "cp-discovery": ("numeric_analysis", "constructed_process_discovery", "low"),
    "data-hold": ("unknown", "unlabeled_holding_area", "low"),
    "Archive": ("unknown", "archived_project", "low"),
    "cns-finalized-snapshot": ("unknown", "snapshot", "low"),
}

# ------------------------------------------- generating_process rules
GP_RULES = [
    # (regex on column name, process, confidence)
    (r"(^|_)(n|num|count|cnt|freq|qty|quantity|total)(_|$)|population|(^|_)pop(_|$)|employees|employment|deaths|births|households|establishments|units|firms|members|votes", "count", "med"),
    (r"amount|amt|price|cost|salary|wage|revenue|income|obligat|outlay|spend|dollar|usd|fee|payment|paid", "price", "med"),
    (r"estimate|(^|_)est(_|$)|projection|forecast|imput", "estimate", "med"),
    (r"magnitude|depth|length|weight|height|temp|pressure|distance|area|volume|duration|latency", "measurement", "med"),
    (r"rank|index_val|score", "index", "low"),
    (r"rate|ratio|pct|percent", "index", "low"),
]

URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")

# portals inferable from unambiguous path/filename tokens (med confidence —
# inferred, not verified against a README)
SOURCE_RULES = [
    (r"gsod", "https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-day"),
    (r"ieee-fraud|train_transaction|train_identity|test_transaction|test_identity", "https://www.kaggle.com/c/ieee-fraud-detection"),
    (r"psam_pus|psam_hus|pums", "https://www.census.gov/programs-surveys/acs/microdata.html"),
    (r"chicago_crimes", "https://data.cityofchicago.org"),
    (r"itcont|indiv2[0-9]|(^|/| )fec(/|_| )", "https://www.fec.gov/data/browse-data/?tab=bulk-data"),
    (r"all_contracts|all_assistance|assistance_primeaward|usaspending|fpds", "https://www.usaspending.gov/download_center"),
    (r"cms_leie", "https://oig.hhs.gov/exclusions/exclusions_list.asp"),
    (r"mup_phy", "https://data.cms.gov"),
]


def slugify(s, maxlen=40):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-").lower()
    return s[:maxlen] or "x"


_readme_cache = {}


def find_source_url(file_path):
    """Look for a README in the file's dir, walking up to the project root;
    return the first URL found."""
    d = os.path.dirname(file_path)
    proj_root = os.path.join(ROOT, top_project(file_path))
    tried = []
    while True:
        if d in _readme_cache:
            hit = _readme_cache[d]
        else:
            hit = None
            try:
                for fn in sorted(os.listdir(d)):
                    if fn.lower().startswith("readme"):
                        p = os.path.join(d, fn)
                        try:
                            with open(p, "r", errors="replace") as f:
                                m = URL_RE.search(f.read(200_000))
                            if m:
                                hit = (m.group(0).rstrip(".,;"), p)
                                break
                        except OSError:
                            pass
            except OSError:
                pass
            _readme_cache[d] = hit
        if hit:
            return hit
        tried.append(d)
        if os.path.normpath(d) == os.path.normpath(proj_root) or \
                len(d) <= len(ROOT):
            return None
        d = os.path.dirname(d)


def label_topic(path_blob):
    for pat, topic, sub, conf in TOPIC_RULES:
        m = re.search(pat, path_blob)
        if m:
            return topic, sub, conf, f"topic from path/name token '{m.group(0)}'"
    return None


def label_row(r):
    """Return interpretive dict for a candidate row."""
    blob = " ".join([str(r.file_path), str(r.archive_member or ""),
                     str(r.column_name)]).lower()
    notes = []

    hit = label_topic(blob)
    if hit:
        topic, subtopic, tconf, note = hit
        notes.append(note)
    else:
        topic, subtopic, tconf = PROJECT_TOPIC.get(
            r.top_project, ("unknown", "unknown", "low"))
        notes.append(f"topic from project fallback '{r.top_project}'")

    col = str(r.column_name).lower()
    if r.looks_like_id:
        gp, gconf = "identifier", "med"
        notes.append("gp: id-flagged column")
    elif r.looks_like_geocode:
        gp, gconf = "administrative", "med"
        notes.append("gp: geocode-flagged column")
    elif r.looks_like_year or re.search(
            r"(^|_)(year|yr|fy|month|day|date|qtr|quarter)(_|$)", col):
        gp, gconf = "fixed_rule", "med"
        notes.append("gp: calendar field")
    elif r.looks_monetary:
        gp, gconf = "price", "med"
        notes.append("gp: monetary-flagged column")
    else:
        gp, gconf = "other", "low"
        for pat, proc, conf in GP_RULES:
            m = re.search(pat, col)
            if m:
                gp, gconf = proc, conf
                notes.append(f"gp from column token '{m.group(0)}'")
                break
        if gp == "other":
            notes.append("gp: no column-name evidence")

    source = None
    for pat, url in SOURCE_RULES:
        if re.search(pat, blob):
            source, sconf = url, "med"
            notes.append("source portal inferred from filename token")
            break
    if source is None:
        src = find_source_url(str(r.file_path)) if not r.archive_member else None
        if src:
            source, sconf = src[0], "med"
            notes.append(f"source URL from {os.path.relpath(src[1], ROOT)}")
        else:
            source, sconf = f"local:{r.top_project}", "low"

    if str(r.rows_basis) == "estimated":
        notes.append("n_records estimated from byte length")

    return dict(topic=topic, topic_confidence=tconf,
                subtopic=subtopic, subtopic_confidence=tconf,
                generating_process=gp, generating_process_confidence=gconf,
                source=source, source_confidence=sconf,
                notes="; ".join(notes))


CATALOG_FIELDS = [
    "dataset_id", "top_project", "file_path", "archive_member",
    "sheet_or_table", "sha256", "dup_key", "dup_count", "column_name",
    "n_records", "n_records_basis",
    "n_integer", "frac_integer", "min", "max", "log10_min", "log10_max",
    "looks_monetary", "looks_like_year", "looks_like_id",
    "looks_like_geocode", "looks_categorical",
    "topic", "topic_confidence", "subtopic", "subtopic_confidence",
    "generating_process", "generating_process_confidence",
    "source", "source_confidence", "notes",
]


def main():
    prof = pd.read_csv(os.path.join(TABLES, "column_profiles.csv"),
                       low_memory=False)
    cand = prof[prof.channel_candidate == True].copy()
    cand["archive_member"] = cand["archive_member"].fillna("")
    cand["sheet_or_table"] = cand["sheet_or_table"].fillna("")

    inv = pd.read_csv(os.path.join(TABLES, "inventory.csv"), low_memory=False)
    finv = inv[inv.row_type == "file"][["file_path", "sha256"]] \
        .drop_duplicates("file_path")
    cand = cand.merge(finv, on="file_path", how="left")

    rows = []
    for r in cand.itertuples():
        lbl = label_row(r)
        fname = os.path.basename(str(r.archive_member) or str(r.file_path))
        uniq = hashlib.sha256(
            f"{r.file_path}|{r.archive_member}|{r.sheet_or_table}|"
            f"{r.column_name}".encode()).hexdigest()[:8]
        rows.append({
            "dataset_id": f"{slugify(r.top_project)}__{slugify(fname)}__"
                          f"{slugify(r.column_name)}__{uniq}",
            "top_project": r.top_project,
            "file_path": r.file_path,
            "archive_member": r.archive_member,
            "sheet_or_table": r.sheet_or_table,
            "sha256": r.sha256,
            "column_name": r.column_name,
            "n_records": r.est_total_rows,
            "n_records_basis": r.rows_basis,
            "n_integer": r.n_integer,
            "frac_integer": r.frac_integer,
            "min": r.min, "max": r.max,
            "log10_min": r.log10_min, "log10_max": r.log10_max,
            "looks_monetary": r.looks_monetary,
            "looks_like_year": r.looks_like_year,
            "looks_like_id": r.looks_like_id,
            "looks_like_geocode": r.looks_like_geocode,
            "looks_categorical": r.looks_categorical,
            **lbl,
        })

    cat = pd.DataFrame(rows, columns=CATALOG_FIELDS)
    # dup_key groups identical column content across the many copied corpora
    # (Archive snapshots, zips): same base filename + column + fingerprint
    cat["dup_key"] = [
        hashlib.md5("|".join(str(v) for v in t).encode()).hexdigest()[:12]
        for t in zip(
            [os.path.basename(str(m) if m else str(f))
             for m, f in zip(cat.archive_member, cat.file_path)],
            cat.column_name, cat.frac_integer, cat["min"], cat["max"],
            cat.n_integer)]
    cat["dup_count"] = cat.groupby("dup_key")["dup_key"].transform("size")
    cat = cat.sort_values(["top_project", "file_path", "archive_member",
                           "sheet_or_table", "column_name"])
    cat.to_csv(os.path.join(TABLES, "catalog.csv"), index=False,
               encoding="utf-8")

    conf_cols = ["topic_confidence", "subtopic_confidence",
                 "generating_process_confidence", "source_confidence"]
    reasons = pd.Series([""] * len(cat), index=cat.index)

    def add_reason(mask, label):
        nonlocal reasons
        reasons = reasons.where(~mask, reasons + label + ";")

    add_reason(cat.looks_like_year.astype(bool), "year")
    add_reason(cat.looks_like_id.astype(bool), "id")
    add_reason(cat.looks_like_geocode.astype(bool), "geocode")
    for c in conf_cols:
        add_reason(cat[c] == "low", c.replace("_confidence", "_low"))
    review = cat[reasons != ""].copy()
    review["review_reasons"] = reasons[reasons != ""].str.rstrip(";")
    review.to_csv(os.path.join(TABLES, "catalog_review.csv"), index=False,
                  encoding="utf-8")
    n_flag_review = int((cat.looks_like_year | cat.looks_like_id |
                         cat.looks_like_geocode).sum())

    # ------------------------------------------------- final counts
    n_files = int((inv.row_type == "file").sum())
    n_members = int((inv.row_type == "archive_member").sum())
    n_data = int(((inv.is_probably_data == True) &
                  (inv.is_derived_artifact == False)).sum())
    n_derived = int((inv.is_derived_artifact == True).sum())
    err_p = os.path.join(LOGS, "read_errors.csv")
    n_err = max(0, sum(1 for _ in open(err_p, encoding="utf-8")) - 1) \
        if os.path.exists(err_p) else 0

    summary = f"""FINAL COUNTS (build 01)
  files inventoried (on disk):      {n_files:,}
  archive members listed:           {n_members:,}
  rows labeled probably-data:       {n_data:,}
  rows labeled derived-artifact:    {n_derived:,}  (label only — nothing moved)
  columns profiled:                 {len(prof):,}
  channel candidates:               {len(cat):,}
  distinct channel contents:        {cat.dup_key.nunique():,}  (dup_key groups; corpus is heavily copied across Archive snapshots)
  candidates needing review:        {len(review):,}
    of which year/id/geocode flags: {n_flag_review:,}
  read errors logged:               {n_err:,}
"""
    print(summary)
    with open(os.path.join(LOGS, "final_counts.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
