"""Build 09 rescue: (a) remap channels whose on-disk Archive files were
deleted since the freeze to their identical zip-member duplicates (via
dup_key) and re-run the mechanism pass on them; (b) NIST retry with a
browser User-Agent (urllib's default is 403-blocked; curl's UA works)."""
import hashlib
import math
import os
import pickle
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, OUT, ext_of
import build02_lib as lib
import stage16_hunt09 as h9

DIAG = os.path.join(OUT, "diagnostics")
FROZEN = os.path.join(OUT, "frozen")
INTER = os.path.join(DIAG, "intermediate02")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 research-script (contact: "
                    "claude.rubbing413@passmail.net)"}


def remap():
    with open(os.path.join(INTER, "mech_v9.pkl"), "rb") as f:
        results = pickle.load(f)
    got = {r["dataset_id"] for r in results}
    dd = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                     low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        dd[c] = dd[c].fillna("")
    missing = dd[~dd.dataset_id.isin(got)].copy()
    print(f"remapping {len(missing)} channels with deleted source files")

    cat = pd.read_csv(os.path.join(TABLES, "catalog.csv"),
                      usecols=["dataset_id", "dup_key", "file_path",
                               "archive_member", "sheet_or_table",
                               "column_name"], low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        cat[c] = cat[c].fillna("")
    key2rows = cat.groupby("dup_key")
    dk = dict(zip(cat.dataset_id, cat.dup_key))

    remap_rows, unmapped = [], 0
    for r in missing.itertuples():
        k = dk.get(r.dataset_id)
        alt = None
        if k is not None:
            for a in key2rows.get_group(k).itertuples():
                if a.dataset_id == r.dataset_id:
                    continue
                if os.path.exists(a.file_path):
                    alt = a
                    break
        if alt is None:
            unmapped += 1
            continue
        remap_rows.append(dict(
            dataset_id=r.dataset_id, monetary=(r.channel_kind == "amount")
            or bool(r.looks_monetary),
            old_path=r.file_path,
            file_path=alt.file_path, archive_member=alt.archive_member,
            sheet_or_table=alt.sheet_or_table,
            column_name=alt.column_name))
    rm = pd.DataFrame(remap_rows)
    rm.to_csv(os.path.join(DIAG, "path_remap_09.csv"), index=False,
              encoding="utf-8")
    print(f"  remapped {len(rm)}, unmappable {unmapped} "
          f"-> diagnostics/path_remap_09.csv")

    tables = []
    for (fp, member, sheet), g in rm.groupby(
            ["file_path", "archive_member", "sheet_or_table"], sort=True):
        seed = int(hashlib.md5(f"{fp}|{member}".encode()).hexdigest()[:8],
                   16)
        channels = [dict(dataset_id=x.dataset_id,
                         column_name=x.column_name,
                         monetary=bool(x.monetary)) for x in g.itertuples()]
        tables.append(((fp, member, sheet, ext_of(member or fp), seed),
                       channels))
    tasks = [tables[i:i + 20] for i in range(0, len(tables), 20)]
    new, errs = [], []
    with ProcessPoolExecutor(max_workers=14,
                             initializer=lib.init_worker) as pool:
        for rows, e in pool.map(h9._corpus_task, tasks):
            new.extend(rows)
            errs.extend(e)
    print(f"  re-profiled {len(new)} channels ({len(errs)} errors)")
    results.extend(new)
    with open(os.path.join(INTER, "mech_v9.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"  mech_v9 now {len(results):,} channels")


def nist_retry():
    lib.init_worker()
    null = h9.Null()
    rows = []
    for sp in h9.NIST_SPECIES:
        url = ("https://physics.nist.gov/cgi-bin/ASD/energy1.pl?" +
               urllib.parse.urlencode({
                   "de": "0", "spectrum": sp, "units": "1", "format": "2",
                   "output": "0", "page_size": "15", "conf_out": "on",
                   "term_out": "on", "level_out": "on", "j_out": "on",
                   "submit": "Retrieve Data"}))
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  NIST fetch failed {sp}: {e!r}", flush=True)
            continue
        confs = {}
        for line in text.splitlines()[1:]:
            parts = [p.replace('="', "").replace('"', "").strip()
                     for p in line.split(",")]
            if len(parts) < 3 or not parts[0] or not parts[2]:
                continue
            try:
                twoJp1 = int(2 * Fraction(parts[2]) + 1)
            except (ValueError, ZeroDivisionError):
                continue
            if twoJp1 >= 1:
                confs.setdefault(parts[0], []).append(twoJp1)
        for conf, gs in confs.items():
            gs = [g for g in gs if g > 1]
            if len(gs) < 2:
                continue
            exp = {}
            for g in gs:
                for p, e in lib.factor_pairs(g):
                    exp[p] = exp.get(p, 0) + e
            ln_n = sum(e * math.log(p) for p, e in exp.items())
            if ln_n <= 0:
                continue
            contribs = sorted((e * math.log(p) for p, e in exp.items()),
                              reverse=True)
            L = [c / ln_n for c in contribs]
            H2 = sum(x * x for x in L)
            d = int(ln_n / math.log(10)) + 1
            rows.append(dict(species=sp, configuration=conf[:60],
                             n_levels=len(gs), log10=ln_n / math.log(10),
                             d=d, L1=L[0], H2=H2,
                             keff=null.get(d, "E_H2") / H2,
                             omega=len(exp), maxexp=max(exp.values())))
    ndf = pd.DataFrame(rows)
    ndf.round(5).to_csv(os.path.join(TABLES, "hunt09_nist.csv"),
                        index=False, encoding="utf-8")
    if len(ndf):
        print(f"NIST retry OK: {len(ndf)} configuration products, "
              f"{ndf.species.nunique()} species, "
              f"keff={ndf.keff.mean():.3f}±{ndf.keff.std():.3f}")
    else:
        print("NIST still unavailable — flagged")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("remap", "both"):
        remap()
    if which in ("nist", "both"):
        nist_retry()
