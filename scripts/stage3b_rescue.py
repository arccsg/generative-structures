"""Stage 3b: rescue quoted-newline CSVs that the line-based sampler broke.

Re-reads every 'EOF inside string' failure with pandas' quote-aware parser
(first 50k rows), appends the profiles, re-sorts column_profiles.csv
deterministically, and removes rescued rows from read_errors.csv.
"""
import csv
import io
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, LOGS, SAMPLE_TOTAL_ROWS, top_project
from stage3_profile import profile_df, PROFILE_FIELDS

csv.field_size_limit(10_000_000)


def read_quoted(fp, member):
    if member:
        with zipfile.ZipFile(fp) as zf, zf.open(member) as f:
            return pd.read_csv(f, nrows=SAMPLE_TOTAL_ROWS,
                               on_bad_lines="skip", encoding_errors="replace")
    return pd.read_csv(fp, nrows=SAMPLE_TOTAL_ROWS, on_bad_lines="skip",
                       encoding_errors="replace")


def main():
    err_path = os.path.join(LOGS, "read_errors.csv")
    errs = pd.read_csv(err_path)
    errs["archive_member"] = errs["archive_member"].fillna("")
    mask = errs.exception.str.contains("EOF inside string", na=False)
    targets = errs[mask]
    print(f"rescuing {len(targets)} quoted-newline tables")

    new_rows, rescued = [], []
    for r in targets.itertuples():
        try:
            df = read_quoted(r.file_path, r.archive_member)
            meta = dict(top_project=top_project(r.file_path),
                        file_path=r.file_path,
                        archive_member=r.archive_member, sheet_or_table="")
            rows = profile_df(df, meta)
            full = len(df) < SAMPLE_TOTAL_ROWS
            for p in rows:
                p.update(sample_method="pandas_quoted_rescue",
                         n_rows_sampled=len(df),
                         est_total_rows=len(df) if full else "",
                         rows_basis="exact" if full else "sampled")
            new_rows.extend(rows)
            rescued.append((r.file_path, r.archive_member))
            print(f"  ok: {r.file_path} :: {r.archive_member} "
                  f"({len(rows)} cols)")
        except Exception as e:
            print(f"  still failing: {r.file_path} :: {r.archive_member} "
                  f"— {e!r}")

    if new_rows:
        prof_path = os.path.join(TABLES, "column_profiles.csv")
        prof = pd.read_csv(prof_path, low_memory=False)
        add = pd.DataFrame(new_rows)[PROFILE_FIELDS]
        prof = pd.concat([prof, add], ignore_index=True)
        for c in ["archive_member", "sheet_or_table"]:
            prof[c] = prof[c].fillna("")
        prof = prof.sort_values(["top_project", "file_path", "archive_member",
                                 "sheet_or_table", "column_name"])
        prof.to_csv(prof_path, index=False, encoding="utf-8")
        print(f"column_profiles.csv now {len(prof):,} rows (re-sorted)")

        keys = set(rescued)
        keep = errs[~errs.apply(lambda r: (r.file_path, r.archive_member)
                                in keys, axis=1)]
        keep.to_csv(err_path, index=False, encoding="utf-8")
        print(f"read_errors.csv now {len(keep)} rows")


if __name__ == "__main__":
    main()
