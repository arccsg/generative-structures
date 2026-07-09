"""Build 02 rescue: re-profile channels that failed the fast reader —
quoted-newline CSVs (quote-aware pandas read) and whitespace-delimited
tables (sep='\\s+'). Appends to obs_raw.pkl and rewrites the error log
with only genuinely failed rows."""
import os
import pickle
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
import build02_lib as lib

DIAG = os.path.join(OUT, "diagnostics")
INTER = os.path.join(DIAG, "intermediate02")
FROZEN = os.path.join(OUT, "frozen")


def read_rescue(fp, member, whitespace):
    kw = dict(nrows=lib.TOTAL_ROWS, on_bad_lines="skip",
              encoding_errors="replace")
    if whitespace:
        kw.update(sep=r"\s+", header=None)
    if member:
        with zipfile.ZipFile(fp) as zf, zf.open(member) as f:
            df = pd.read_csv(f, **kw)
    else:
        df = pd.read_csv(fp, **kw)
    if whitespace:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df, len(df) >= lib.TOTAL_ROWS


def main():
    lib.init_worker()
    errs = pd.read_csv(os.path.join(DIAG,
                                    "factorization_errors_observational.csv"))
    errs["archive_member"] = errs["archive_member"].fillna("")
    corpus = pd.read_csv(os.path.join(FROZEN, "observational_corpus.csv"),
                         low_memory=False)
    corpus["archive_member"] = corpus["archive_member"].fillna("")
    corpus["monetary"] = (corpus.channel_kind == "amount") | \
        corpus.looks_monetary.astype(bool)
    meta = {(r.file_path, r.archive_member, r.column_name):
            (r.dataset_id, bool(r.monetary)) for r in corpus.itertuples()}

    new_rows, fixed = [], set()
    for (fp, member), g in errs.groupby(["file_path", "archive_member"]):
        ws = g.error.str.contains("column missing").any()
        try:
            df, sampled = read_rescue(fp, member, ws)
        except Exception as e:
            print(f"  still failing: {fp}::{member} — {e!r}")
            continue
        for r in g.itertuples():
            key = (fp, member, r.column_name)
            if key not in meta or r.column_name not in df.columns:
                print(f"  unrescuable column: {key}")
                continue
            did, monetary = meta[key]
            series = df[r.column_name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            ints, n_ceil = lib.coerce_ints(series, monetary)
            prof = lib.profile_values(ints)
            if prof is None:
                print(f"  no integers >= 2: {key}")
                continue
            prof.update(dataset_id=did, sampled=bool(sampled),
                        unit_scaled=monetary, n_ceiling=n_ceil)
            new_rows.append(prof)
            fixed.add((fp, member, r.column_name))
        print(f"rescued {fp}::{member} ({ws and 'whitespace' or 'quoted'})")

    pkl = os.path.join(INTER, "obs_raw.pkl")
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    data["results"].extend(new_rows)
    with open(pkl, "wb") as f:
        pickle.dump(data, f)

    keep = errs[~errs.apply(lambda r: (r.file_path, r.archive_member,
                                       r.column_name) in fixed, axis=1)]
    keep.to_csv(os.path.join(DIAG,
                             "factorization_errors_observational.csv"),
                index=False, encoding="utf-8")
    print(f"rescued {len(new_rows)} channels; {len(keep)} errors remain; "
          f"obs_raw now {len(data['results']):,} channels")


if __name__ == "__main__":
    main()
