"""Stage 2: label every inventory row as derived-artifact and/or probably-data.

Label only — nothing is moved, deleted, or acted on. Rewrites
tables/inventory.csv in place WITH ADDED COLUMNS ONLY (same rows, same order).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, TABLES, DERIVED_NAME_TOKENS, DERIVED_DIR_NAMES,
                    TABULAR_EXTS, STRUCTURED_JSON_EXTS, EXCLUDE_DIRS)

csv.field_size_limit(10_000_000)


def classify(row):
    """Return (is_derived_artifact, is_probably_data, reason)."""
    # the "filename" for an archive member is the member's basename; the
    # directory rule applies to the member's internal dirs + the on-disk path
    if row["archive_member"]:
        fname = os.path.basename(row["archive_member"]).lower()
        member_parts = os.path.dirname(row["archive_member"]).lower().split("/")
        dir_parts = (os.path.dirname(row["file_path"]).lower().split(os.sep)
                     + member_parts)
        # apply the walk's exclude-dir rule inside archives too: venv /
        # site-packages trees bundled into project zips are not corpus data
        excl = sorted({p for p in member_parts if p in EXCLUDE_DIRS})
        if excl:
            return False, False, (f"archive member under excluded dir "
                                  f"(walk exclusion applied in-archive): {excl}")
    else:
        fname = os.path.basename(row["file_path"]).lower()
        dir_parts = os.path.dirname(row["file_path"]).lower().split(os.sep)

    reasons = []
    name_hits = [t for t in DERIVED_NAME_TOKENS if t in fname]
    if name_hits:
        reasons.append(f"filename matches derived tokens: {sorted(set(name_hits))}")
    dir_hits = sorted({p for p in dir_parts if p in DERIVED_DIR_NAMES})
    if dir_hits:
        reasons.append(f"under derived-artifact directory: {dir_hits}")
    is_derived = bool(reasons)

    ext = row["extension"].lower()
    if is_derived:
        is_data = False
        reason = "; ".join(reasons)
    elif ext in TABULAR_EXTS:
        is_data = True
        reason = f"tabular extension {ext}, no derived-artifact signals"
    elif ext in STRUCTURED_JSON_EXTS:
        is_data = True
        reason = f"structured {ext}, no derived-artifact signals"
    else:
        is_data = False
        if row["row_type"] == "file" and ext in {".zip", ".gz"}:
            reason = "archive container (members classified separately)"
        else:
            reason = f"extension {ext or '(none)'} not a data format"
    return is_derived, is_data, reason


def main():
    inv_path = os.path.join(TABLES, "inventory.csv")
    with open(inv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fields = r.fieldnames

    n_derived = n_data = 0
    for row in rows:
        d, p, reason = classify(row)
        row["is_derived_artifact"] = d
        row["is_probably_data"] = p
        row["classify_reason"] = reason
        n_derived += d
        n_data += p

    out_fields = list(fields)
    for c in ["is_derived_artifact", "is_probably_data", "classify_reason"]:
        if c not in out_fields:
            out_fields.append(c)

    with open(inv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    print(f"rows classified: {len(rows):,}")
    print(f"is_derived_artifact=True: {n_derived:,}")
    print(f"is_probably_data=True (and not derived): {n_data:,}")


if __name__ == "__main__":
    main()
