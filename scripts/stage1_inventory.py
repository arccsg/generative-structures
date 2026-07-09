"""Stage 1: raw inventory of every candidate data file under ROOT.

Non-destructive: reads files, writes only under lprofile-geography/.
One row per file; additional rows for zip/gz archive members (not extracted).
"""
import csv
import gzip
import hashlib
import json
import os
import struct
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, OUT, TABLES, LOGS, CONFIG, INCLUDE_EXTS,
                    EXCLUDE_DIRS, EXCLUDE_PATHS, HASH_FULL_LIMIT,
                    HASH_PARTIAL_BYTES, ext_of, top_project)


def freeze_config():
    import pandas
    ts = datetime.now(timezone.utc).isoformat()
    run_id = "lpg01-" + hashlib.sha256(ts.encode()).hexdigest()[:8]
    cfg = {
        "run_id": run_id,
        "build": "01-corpus-discovery-and-catalog",
        "timestamp_utc": ts,
        "root_walked": ROOT,
        "output_tree": OUT,
        "include_extensions": sorted(INCLUDE_EXTS),
        "exclude_dir_names": sorted(EXCLUDE_DIRS),
        "exclude_paths": sorted(EXCLUDE_PATHS),
        "hash_full_limit_bytes": HASH_FULL_LIMIT,
        "hash_partial_bytes": HASH_PARTIAL_BYTES,
        "archives": "zip/gz members listed, never extracted to disk",
        "python_version": sys.version,
        "pandas_version": pandas.__version__,
        "notes": "Build 01: discovery + catalog only. No factorization, no "
                 "L-profile, no figures. Label-only classification; nothing "
                 "moved, deleted, or overwritten.",
    }
    with open(os.path.join(CONFIG, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def walk_files():
    """Yield absolute paths of candidate files."""
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in EXCLUDE_DIRS
                       and os.path.join(dirpath, d) not in EXCLUDE_PATHS]
        for fn in filenames:
            if ext_of(fn) in INCLUDE_EXTS:
                yield os.path.join(dirpath, fn)


def hash_file(path, size):
    h = hashlib.sha256()
    partial = size > HASH_FULL_LIMIT
    limit = HASH_PARTIAL_BYTES if partial else None
    read = 0
    with open(path, "rb") as f:
        while True:
            chunk_size = 4 * 1024 * 1024
            if limit is not None:
                chunk_size = min(chunk_size, limit - read)
                if chunk_size <= 0:
                    break
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest(), partial


def gz_isize(path):
    """Uncompressed size mod 2^32 from the gzip trailer."""
    try:
        with open(path, "rb") as f:
            f.seek(-4, os.SEEK_END)
            return struct.unpack("<I", f.read(4))[0]
    except Exception:
        return ""


def process_one(path):
    """Return (rows, error) for a single file. rows includes archive members."""
    rows, errors = [], []
    try:
        st = os.stat(path)
        size = st.st_size
        mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
        sha, partial = hash_file(path, size)
        ext = ext_of(path)
        proj = top_project(path)
        base = dict(file_path=path, top_project=proj, extension=ext,
                    size_bytes=size, mtime=mtime, sha256=sha,
                    hash_partial=partial)
        rows.append({**base, "row_type": "file", "archive_member": "",
                     "member_uncompressed_size": ""})
        if ext == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        rows.append({**base, "row_type": "archive_member",
                                     "extension": ext_of(info.filename),
                                     "archive_member": info.filename,
                                     "member_uncompressed_size": info.file_size})
            except Exception as e:
                errors.append((path, "", f"zip member listing failed: {e!r}"))
        elif ext == ".gz":
            member = os.path.basename(path)[:-3]  # strip .gz
            rows.append({**base, "row_type": "archive_member",
                         "extension": ext_of(member),
                         "archive_member": member,
                         "member_uncompressed_size": gz_isize(path)})
    except Exception as e:
        errors.append((path, "", f"stat/hash failed: {e!r}"))
    return rows, errors


FIELDS = ["row_type", "file_path", "archive_member", "top_project",
          "extension", "size_bytes", "member_uncompressed_size", "mtime",
          "sha256", "hash_partial"]


def main():
    cfg = freeze_config()
    print(f"run_id={cfg['run_id']}  root={ROOT}", flush=True)

    paths = sorted(walk_files())
    print(f"walk complete: {len(paths)} candidate files; hashing...", flush=True)

    all_rows, all_errors = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rows, errors in pool.map(process_one, paths, chunksize=64):
            all_rows.extend(rows)
            all_errors.extend(errors)
            done += 1
            if done % 10000 == 0:
                print(f"  hashed {done}/{len(paths)}", flush=True)

    all_rows.sort(key=lambda r: (r["top_project"], r["file_path"],
                                 r["archive_member"]))
    inv_path = os.path.join(TABLES, "inventory.csv")
    with open(inv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    if all_errors:
        with open(os.path.join(LOGS, "inventory_errors.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_path", "archive_member", "exception"])
            w.writerows(all_errors)

    write_checkpoint(cfg, all_rows, all_errors)


def write_checkpoint(cfg, rows, errors):
    files = [r for r in rows if r["row_type"] == "file"]
    members = [r for r in rows if r["row_type"] == "archive_member"]
    total_bytes = sum(r["size_bytes"] for r in files)

    by_ext, by_proj = {}, {}
    for r in files:
        by_ext[r["extension"]] = by_ext.get(r["extension"], 0) + 1
        p = by_proj.setdefault(r["top_project"], [0, 0])
        p[0] += 1
        p[1] += r["size_bytes"]
    member_ext = {}
    for r in members:
        member_ext[r["extension"]] = member_ext.get(r["extension"], 0) + 1

    known = {
        "cp-discovery": os.path.join(ROOT, "cp-discovery"),
        "cbad-audit": os.path.join(ROOT, "cbad-audit"),
        "audit-analytics": os.path.join(ROOT, "audit-analytics"),
        "Archive/cbad": os.path.join(ROOT, "Archive", "cbad"),
        "Archive/ourosboros": os.path.join(ROOT, "Archive", "ourosboros"),
    }

    def gb(n):
        return f"{n / 1e9:.2f} GB"

    L = []
    L.append(f"# Checkpoint — Stage 1 raw inventory (run {cfg['run_id']})")
    L.append(f"\nGenerated {cfg['timestamp_utc']} — root `{ROOT}`\n")
    L.append(f"- **Total files found:** {len(files):,}")
    L.append(f"- **Total bytes:** {total_bytes:,} ({gb(total_bytes)})")
    L.append(f"- **Archive members listed (not extracted):** {len(members):,}")
    L.append(f"- **Files that failed stat/hash/list:** {len(errors)}")

    L.append("\n## Counts by extension (files on disk)\n")
    L.append("| extension | count |")
    L.append("|---|---|")
    for ext, n in sorted(by_ext.items(), key=lambda kv: -kv[1]):
        L.append(f"| {ext} | {n:,} |")
    if member_ext:
        L.append("\n### Archive-member extensions (inside zip/gz)\n")
        L.append("| extension | count |")
        L.append("|---|---|")
        for ext, n in sorted(member_ext.items(), key=lambda kv: -kv[1]):
            L.append(f"| {ext or '(none)'} | {n:,} |")

    L.append("\n## Counts by top-level project\n")
    L.append("| project | files | total bytes |")
    L.append("|---|---|---|")
    for proj, (n, b) in sorted(by_proj.items(), key=lambda kv: -kv[1][1]):
        L.append(f"| {proj} | {n:,} | {b:,} ({gb(b)}) |")

    L.append("\n## Known-data directory confirmation\n")
    L.append("| directory | exists | candidate files (on disk) | archive members |")
    L.append("|---|---|---|---|")
    for label, path in known.items():
        exists = os.path.isdir(path)
        prefix = path + os.sep
        nf = sum(1 for r in files if r["file_path"].startswith(prefix))
        nm = sum(1 for r in members if r["file_path"].startswith(prefix))
        L.append(f"| {label} | {'YES' if exists else '**MISSING**'} | {nf:,} | {nm:,} |")
    L.append("\n(Note: `Archive/ourosboros` is the on-disk spelling; "
             "`Archive/ourosboros.zip` and other Archive zips are inventoried "
             "as archives with members listed.)")

    L.append("\n## 20 largest files\n")
    L.append("| size | path |")
    L.append("|---|---|")
    for r in sorted(files, key=lambda r: -r["size_bytes"])[:20]:
        L.append(f"| {gb(r['size_bytes'])} | {r['file_path']} |")

    L.append("\n## 20 most recently modified\n")
    L.append("| mtime | path |")
    L.append("|---|---|")
    for r in sorted(files, key=lambda r: r["mtime"], reverse=True)[:20]:
        L.append(f"| {r['mtime'][:19]} | {r['file_path']} |")

    text = "\n".join(L) + "\n"
    with open(os.path.join(LOGS, "checkpoint_inventory.md"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
