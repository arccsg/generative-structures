"""Stage 3: memory-safe column profiling + integer-channel detection.

Profiles every inventory row with is_probably_data=True and
is_derived_artifact=False. Sampling: header + up to 50k rows (first 25k +
random 25k by byte-offset for large seekable files; sequential 50k for
compressed/zip streams). Archive members are read through temp handles,
never extracted to disk.

Outputs: tables/column_profiles.csv, logs/read_errors.csv,
logs/profile_skips.csv.
"""
import csv
import gzip
import io
import json
import math
import os
import random
import re
import sys
import zipfile
import zlib
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TABLES, LOGS, SAMPLE_HEAD_ROWS, SAMPLE_TOTAL_ROWS,
                    top_project)

warnings.filterwarnings("ignore")
csv.field_size_limit(10_000_000)

JSON_PARSE_LIMIT = 200 * 1024 * 1024
CURRENCY_RE = re.compile(r"^[\$£€¥]")
THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
NUMERIC_NAME_RE = re.compile(r"^-?\d+(\.\d+)?$")
INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?\d*\.\d+([eE][+-]?\d+)?$|^[+-]?\d+\.\d*([eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$")

ID_NAMES = {"id", "index", "key", "code", "uid", "guid"}
GEO_NAMES = {"zip", "zipcode", "fips", "geoid", "tract", "block", "cbsa"}
MONEY_NAME_KW = ("price", "cost", "amount", "amt", "salary", "wage", "revenue",
                 "income", "usd", "dollar", "fee", "pay", "obligat", "outlay",
                 "spend", "value_usd")

PROFILE_FIELDS = [
    "top_project", "file_path", "archive_member", "sheet_or_table",
    "column_name", "dtype_as_read", "sample_method", "n_rows_sampled",
    "est_total_rows", "rows_basis", "n_nonnull", "n_integer", "frac_integer",
    "n_whole", "frac_whole", "min", "max", "log10_min", "log10_max",
    "n_distinct_int", "looks_like_year", "looks_like_id",
    "looks_like_geocode", "looks_monetary", "looks_categorical",
    "channel_candidate",
]


# ---------------------------------------------------------------- sampling

def sniff(text_block, ext):
    """Return (delimiter, ok). delimiter 'ws' means whitespace."""
    lines = [ln for ln in text_block.splitlines()[:40] if ln.strip()]
    if not lines:
        return None, False
    best = None
    for d in [",", "\t", ";", "|"]:
        counts = [ln.count(d) for ln in lines]
        pos = [c for c in counts if c > 0]
        if len(pos) >= max(1, int(0.8 * len(lines))):
            from collections import Counter
            common, freq = Counter(pos).most_common(1)[0]
            score = (freq / len(pos), common)
            if freq / len(pos) >= 0.7 and (best is None or score > best[1]):
                best = (d, score)
    if best:
        return best[0], True
    ws_counts = [len(ln.split()) for ln in lines]
    if len(lines) >= 2 and min(ws_counts) >= 2 and max(ws_counts) == min(ws_counts):
        return "ws", True
    if ext in (".csv", ".tsv"):
        return ("\t" if ext == ".tsv" else ","), True  # single-column ok
    return None, False


def decode(b):
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1")


def sample_lines_stream(fh, head_rows, total_rows):
    """Sequentially read raw lines from a binary stream (non-seekable ok).
    Returns (header, lines, bytes_consumed, eof)."""
    header = fh.readline()
    lines, consumed, eof = [], len(header), False
    for _ in range(total_rows):
        ln = fh.readline()
        if not ln:
            eof = True
            break
        consumed += len(ln)
        lines.append(ln)
    else:
        if not fh.readline():
            eof = True
    return header, lines, consumed, eof


def random_line_sample(path, start, size, n, seed):
    rng = random.Random(seed)
    offsets = sorted(rng.randrange(start, size) for _ in range(n))
    out, last_line_start = [], -1
    with open(path, "rb") as f:
        for off in offsets:
            if last_line_start >= 0 and off < last_line_start:
                continue
            f.seek(off)
            f.readline()  # discard partial line
            pos = f.tell()
            if pos == last_line_start:
                continue
            ln = f.readline()
            if ln:
                last_line_start = pos
                out.append(ln)
    return out


def read_delimited(open_binary, path_for_seek, size, ext, seed):
    """Return (df, sample_method, n_rows_sampled, est_total_rows, rows_basis)
    or raises SkipTable."""
    with open_binary() as fh:
        block = fh.read(256 * 1024)
    if b"\x00" in block[:65536]:
        raise SkipTable("binary content (NUL bytes) — not a text table")
    delim, ok = sniff(decode(block), ext)
    if not ok:
        raise SkipTable("no consistent delimiter — not tabular")

    with open_binary() as fh:
        header, lines, consumed, eof = sample_lines_stream(
            fh, SAMPLE_HEAD_ROWS,
            SAMPLE_HEAD_ROWS if path_for_seek else SAMPLE_TOTAL_ROWS)

    method = "head"
    if eof:
        basis, est = "exact", len(lines)
    elif path_for_seek:
        extra = random_line_sample(path_for_seek, consumed, size,
                                   SAMPLE_TOTAL_ROWS - SAMPLE_HEAD_ROWS, seed)
        lines += extra
        method = "head25k+random25k"
        avg = consumed / max(1, SAMPLE_HEAD_ROWS)
        basis, est = "estimated", int(size / max(1.0, avg))
    else:
        method = "head50k_stream"
        avg = (consumed) / max(1, len(lines))
        basis, est = "estimated", int(size / max(1.0, avg)) if size else ""

    text = decode(header + b"".join(lines))
    kw = dict(engine="python", sep=r"\s+") if delim == "ws" else dict(sep=delim)
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", **kw)

    def is_data_value(c):
        c = str(c).strip()
        return bool(re.match(r"^-?\d{5,}$", c) or re.match(r"^-?\d*\.\d+$", c))

    if len(df.columns) and (
            all(NUMERIC_NAME_RE.match(str(c).strip()) for c in df.columns)
            or any(is_data_value(c) for c in df.columns)):
        df = pd.read_csv(io.StringIO(text), header=None, on_bad_lines="skip",
                         **kw)
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        method += "+noheader"
        if basis == "exact":
            est = est + 1
    return df, method, len(df), est, basis


class SkipTable(Exception):
    pass


def read_json_obj(raw):
    obj = json.loads(raw)
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj[:100]):
            return pd.json_normalize(obj[:SAMPLE_TOTAL_ROWS]), len(obj)
        if obj and all(isinstance(x, (list, tuple)) for x in obj[:100]):
            return pd.DataFrame(obj[:SAMPLE_TOTAL_ROWS]), len(obj)
        raise SkipTable("json list of scalars/mixed — not tabular")
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and all(isinstance(v, list) for v in vals) and \
                len({len(v) for v in vals}) == 1:
            return (pd.DataFrame({k: v[:SAMPLE_TOTAL_ROWS]
                                  for k, v in obj.items()}), len(vals[0]))
        if vals and all(isinstance(v, dict) for v in vals):
            return (pd.DataFrame.from_dict(obj, orient="index")
                    .head(SAMPLE_TOTAL_ROWS), len(vals))
        raise SkipTable("json dict — no tabular structure")
    raise SkipTable("json scalar — not tabular")


def read_jsonl_stream(fh):
    rows = []
    for i, ln in enumerate(fh):
        if i >= SAMPLE_TOTAL_ROWS:
            break
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(decode(ln) if isinstance(ln, bytes)
                                       else ln))
            except json.JSONDecodeError:
                pass
    if not rows:
        raise SkipTable("jsonl: no parseable json lines")
    return pd.json_normalize(rows)


# ------------------------------------------------------------- per-column

def profile_column(name, series, head_n):
    s = series.dropna()
    n_nonnull = int(len(s))
    out = dict(column_name=str(name), dtype_as_read=str(series.dtype),
               n_nonnull=n_nonnull)
    monetary_sym = False
    nums = np.array([], dtype=float)

    if n_nonnull:
        if pd.api.types.is_bool_dtype(s):
            pass
        elif pd.api.types.is_numeric_dtype(s):
            nums = s.to_numpy(dtype=float, na_value=np.nan)
        elif pd.api.types.is_datetime64_any_dtype(s) or \
                pd.api.types.is_timedelta64_dtype(s):
            pass
        else:
            # conservative string coercion: strip a leading currency symbol
            # and thousands separators; never invent values
            t = s.astype(str).str.strip()
            cur = t.str.match(CURRENCY_RE)
            if cur.any():
                monetary_sym = True
                t = t.where(~cur, t.str.replace(CURRENCY_RE, "", n=1,
                                                regex=True).str.strip())
            thou = t.str.match(THOUSANDS_RE)
            if thou.any():
                t = t.where(~thou, t.str.replace(",", "", regex=False))
            nums = pd.to_numeric(t, errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(nums)
    fin = nums[finite]
    whole_mask = fin == np.floor(fin)
    small_mask = np.abs(fin) < 1e18
    whole = fin[whole_mask & small_mask]
    n_decimal = int((~whole_mask).sum())

    n_whole = int(len(whole))
    n_integer = int((whole >= 2).sum())
    out["n_whole"] = n_whole
    out["frac_whole"] = round(n_whole / n_nonnull, 6) if n_nonnull else 0.0
    out["n_integer"] = n_integer
    out["frac_integer"] = round(n_integer / n_nonnull, 6) if n_nonnull else 0.0

    if n_whole:
        mn, mx = int(whole.min()), int(whole.max())
        out["min"], out["max"] = mn, mx
        out["log10_min"] = round(math.log10(mn), 4) if mn > 0 else ""
        out["log10_max"] = round(math.log10(mx), 4) if mx > 0 else ""
        distinct = int(len(np.unique(whole)))
        out["n_distinct_int"] = distinct
        out["looks_like_year"] = bool(n_whole == n_nonnull and
                                      1800 <= mn and mx <= 2100)
        out["looks_categorical"] = bool(distinct <= 25 and n_nonnull >= 50)
    else:
        out.update({"min": "", "max": "", "log10_min": "", "log10_max": "",
                    "n_distinct_int": 0, "looks_like_year": False,
                    "looks_categorical": False})
        mx = None

    lname = str(name).strip().lower()
    inc_by_one = False
    if n_whole >= 2 and n_whole == n_nonnull:
        head = whole[:min(head_n, n_whole)]
        if len(head) >= 2:
            inc_by_one = bool(np.all(np.diff(head) == 1.0))
    out["looks_like_id"] = bool(lname in ID_NAMES or inc_by_one)
    out["looks_like_geocode"] = bool(lname in GEO_NAMES)

    money_name = any(k in lname for k in MONEY_NAME_KW)
    out["looks_monetary"] = bool(
        monetary_sym or (money_name and n_decimal > 0 and n_nonnull and
                         n_decimal / n_nonnull > 0.3))
    out["channel_candidate"] = bool(
        out["frac_integer"] >= 0.99 and n_nonnull >= 100 and
        mx is not None and mx >= 2)
    return out


def profile_df(df, meta):
    rows = []
    head_n = min(len(df), SAMPLE_HEAD_ROWS)
    for col in df.columns:
        series = df[col]
        if isinstance(series, pd.DataFrame):  # duplicate column names
            series = series.iloc[:, 0]
        p = profile_column(col, series, head_n)
        p.update(meta)
        rows.append(p)
    rows.sort(key=lambda r: r["column_name"])
    return rows


# ------------------------------------------------------------- dispatch

def profile_source(task):
    """task = dict(kind, file_path, archive_member, extension, size).
    Returns (profile_rows, error_rows, skip_rows)."""
    fp = task["file_path"]
    member = task["archive_member"]
    ext = task["extension"]
    size = task["size"]
    seed = zlib.crc32(f"{fp}::{member}".encode()) ^ 42
    meta = dict(top_project=top_project(fp), file_path=fp,
                archive_member=member, sheet_or_table="")
    try:
        if task["kind"] == "file":
            return profile_plain_file(fp, ext, size, seed, meta)
        elif task["kind"] == "gz_member":
            return profile_gz_member(fp, member, ext, size, seed, meta)
        else:
            raise ValueError(f"bad kind {task['kind']}")
    except SkipTable as e:
        return [], [], [(fp, member, str(e))]
    except Exception as e:
        return [], [(fp, member, repr(e)[:500])], []


def finish(df, meta, method, n, est, basis):
    rows = profile_df(df, meta)
    for r in rows:
        r.update(sample_method=method, n_rows_sampled=n,
                 est_total_rows=est, rows_basis=basis)
    return rows, [], []


def profile_plain_file(fp, ext, size, seed, meta):
    if ext in (".csv", ".tsv", ".txt", ".dat"):
        df, method, n, est, basis = read_delimited(
            lambda: open(fp, "rb"), fp, size, ext, seed)
        return finish(df, meta, method, n, est, basis)
    if ext == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(fp)
        total = pf.metadata.num_rows
        batches, got = [], 0
        for b in pf.iter_batches(batch_size=8192):
            batches.append(b)
            got += b.num_rows
            if got >= SAMPLE_TOTAL_ROWS:
                break
        import pyarrow as pa
        df = pa.Table.from_batches(batches).to_pandas() if batches else \
            pf.schema_arrow.empty_table().to_pandas()
        df = df.head(SAMPLE_TOTAL_ROWS)
        return finish(df, meta, "parquet_head", len(df), total, "exact")
    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(fp, sheet_name=None, nrows=SAMPLE_TOTAL_ROWS)
        rows = []
        for sname in sorted(sheets):
            m = dict(meta, sheet_or_table=str(sname))
            df = sheets[sname]
            rr = profile_df(df, m)
            for r in rr:
                r.update(sample_method="excel_sheet", n_rows_sampled=len(df),
                         est_total_rows=len(df),
                         rows_basis="exact" if len(df) < SAMPLE_TOTAL_ROWS
                         else "sampled")
            rows.extend(rr)
        if not rows:
            raise SkipTable("excel: no sheets with data")
        return rows, [], []
    if ext == ".json":
        if size > JSON_PARSE_LIMIT:
            raise SkipTable(f"json too large to parse safely ({size} bytes)")
        with open(fp, "rb") as f:
            df, total = read_json_obj(decode(f.read()))
        return finish(df, meta, "json_full", len(df), total,
                      "exact" if total <= SAMPLE_TOTAL_ROWS else "sampled")
    if ext in (".jsonl", ".ndjson"):
        with open(fp, "rb") as f:
            df = read_jsonl_stream(f)
        full = len(df) < SAMPLE_TOTAL_ROWS
        return finish(df, meta, "jsonl_head", len(df),
                      len(df) if full else "", "exact" if full else "sampled")
    raise SkipTable(f"unhandled extension {ext}")


def profile_gz_member(fp, member, ext, size, seed, meta):
    if ext in (".csv", ".tsv", ".txt", ".dat"):
        df, method, n, est, basis = read_delimited(
            lambda: gzip.open(fp, "rb"), None, size or 0, ext, seed)
        return finish(df, meta, "gz_" + method, n, est, basis)
    if ext in (".jsonl", ".ndjson"):
        with gzip.open(fp, "rb") as f:
            df = read_jsonl_stream(f)
        return finish(df, meta, "gz_jsonl_head", len(df),
                      len(df) if len(df) < SAMPLE_TOTAL_ROWS else "",
                      "exact" if len(df) < SAMPLE_TOTAL_ROWS else "sampled")
    if ext == ".json":
        with gzip.open(fp, "rb") as f:
            raw = f.read(JSON_PARSE_LIMIT + 1)
        if len(raw) > JSON_PARSE_LIMIT:
            raise SkipTable("json.gz too large to parse safely")
        df, total = read_json_obj(decode(raw))
        return finish(df, meta, "gz_json_full", len(df), total,
                      "exact" if total <= SAMPLE_TOTAL_ROWS else "sampled")
    raise SkipTable(f"unhandled gz member extension {ext}")


def profile_zip_chunk(chunk):
    """chunk = (zip_path, [ (member, ext, size), ... ]) — one open per chunk."""
    zip_path, members = chunk
    out_rows, out_errs, out_skips = [], [], []
    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception as e:
        return [], [(zip_path, m[0], f"zip open failed: {e!r}"[:500])
                    for m in members], []
    with zf:
        for member, ext, size in members:
            meta = dict(top_project=top_project(zip_path), file_path=zip_path,
                        archive_member=member, sheet_or_table="")
            seed = zlib.crc32(f"{zip_path}::{member}".encode()) ^ 42
            try:
                if ext in (".csv", ".tsv", ".txt", ".dat"):
                    df, method, n, est, basis = read_delimited(
                        lambda: zf.open(member), None, size or 0, ext, seed)
                    rows, _, _ = finish(df, meta, "zip_" + method, n, est,
                                        basis)
                elif ext in (".jsonl", ".ndjson"):
                    with zf.open(member) as f:
                        df = read_jsonl_stream(f)
                    full = len(df) < SAMPLE_TOTAL_ROWS
                    rows, _, _ = finish(df, meta, "zip_jsonl_head", len(df),
                                        len(df) if full else "",
                                        "exact" if full else "sampled")
                elif ext == ".json":
                    if size and size > JSON_PARSE_LIMIT:
                        raise SkipTable("zip json member too large")
                    with zf.open(member) as f:
                        df, total = read_json_obj(decode(f.read()))
                    rows, _, _ = finish(df, meta, "zip_json_full", len(df),
                                        total,
                                        "exact" if total <= SAMPLE_TOTAL_ROWS
                                        else "sampled")
                elif ext in (".parquet", ".xlsx", ".xls"):
                    with zf.open(member) as f:
                        buf = io.BytesIO(f.read(JSON_PARSE_LIMIT + 1))
                    if buf.getbuffer().nbytes > JSON_PARSE_LIMIT:
                        raise SkipTable("zip binary member too large")
                    if ext == ".parquet":
                        import pyarrow.parquet as pq
                        pf = pq.ParquetFile(buf)
                        total = pf.metadata.num_rows
                        got, batches = 0, []
                        for b in pf.iter_batches(batch_size=8192):
                            batches.append(b)
                            got += b.num_rows
                            if got >= SAMPLE_TOTAL_ROWS:
                                break
                        import pyarrow as pa
                        df = (pa.Table.from_batches(batches).to_pandas()
                              .head(SAMPLE_TOTAL_ROWS)) if batches else \
                            pf.schema_arrow.empty_table().to_pandas()
                        rows, _, _ = finish(df, meta, "zip_parquet_head",
                                            len(df), total, "exact")
                    else:
                        sheets = pd.read_excel(buf, sheet_name=None,
                                               nrows=SAMPLE_TOTAL_ROWS)
                        rows = []
                        for sname in sorted(sheets):
                            m = dict(meta, sheet_or_table=str(sname))
                            sdf = sheets[sname]
                            rr = profile_df(sdf, m)
                            for r in rr:
                                r.update(sample_method="zip_excel_sheet",
                                         n_rows_sampled=len(sdf),
                                         est_total_rows=len(sdf),
                                         rows_basis="exact"
                                         if len(sdf) < SAMPLE_TOTAL_ROWS
                                         else "sampled")
                            rows.extend(rr)
                else:
                    raise SkipTable(f"unhandled zip member extension {ext}")
                out_rows.extend(rows)
            except SkipTable as e:
                out_skips.append((zip_path, member, str(e)))
            except Exception as e:
                out_errs.append((zip_path, member, repr(e)[:500]))
    return out_rows, out_errs, out_skips


def run_task(task):
    if task[0] == "zip":
        return profile_zip_chunk(task[1])
    return profile_source(task[1])


# ------------------------------------------------------------------ main

def build_tasks():
    inv = pd.read_csv(os.path.join(TABLES, "inventory.csv"), low_memory=False)
    data = inv[(inv.is_probably_data == True) &
               (inv.is_derived_artifact == False)].copy()
    data["archive_member"] = data["archive_member"].fillna("")
    data = data.sort_values(["top_project", "file_path", "archive_member"])

    tasks = []
    pending_zip = None  # (zip_path, [members])

    def flush():
        nonlocal pending_zip
        if pending_zip and pending_zip[1]:
            tasks.append(("zip", pending_zip))
        pending_zip = None

    for r in data.itertuples():
        size = r.member_uncompressed_size if r.archive_member else r.size_bytes
        try:
            size = int(float(size))
        except (TypeError, ValueError):
            size = 0
        if r.row_type == "archive_member" and r.file_path.lower().endswith(".zip"):
            if pending_zip is None or pending_zip[0] != r.file_path or \
                    len(pending_zip[1]) >= 200:
                flush()
                pending_zip = (r.file_path, [])
            pending_zip[1].append((r.archive_member, r.extension, size))
        elif r.row_type == "archive_member":  # .gz member
            flush()
            tasks.append(("one", dict(kind="gz_member", file_path=r.file_path,
                                      archive_member=r.archive_member,
                                      extension=r.extension, size=size)))
        else:
            flush()
            tasks.append(("one", dict(kind="file", file_path=r.file_path,
                                      archive_member="",
                                      extension=r.extension, size=size)))
    flush()
    return tasks


def main():
    tasks = build_tasks()
    n_sources = sum(len(t[1][1]) if t[0] == "zip" else 1 for t in tasks)
    print(f"profiling {n_sources:,} table sources in {len(tasks):,} tasks",
          flush=True)

    prof_path = os.path.join(TABLES, "column_profiles.csv")
    err_path = os.path.join(LOGS, "read_errors.csv")
    skip_path = os.path.join(LOGS, "profile_skips.csv")

    n_cols = n_err = n_skip = n_done = 0
    with open(prof_path, "w", newline="", encoding="utf-8") as pf, \
         open(err_path, "w", newline="", encoding="utf-8") as ef, \
         open(skip_path, "w", newline="", encoding="utf-8") as sf, \
         ProcessPoolExecutor(max_workers=10) as pool:
        pw = csv.DictWriter(pf, fieldnames=PROFILE_FIELDS,
                            extrasaction="ignore")
        pw.writeheader()
        ew = csv.writer(ef)
        ew.writerow(["file_path", "archive_member", "exception"])
        sw = csv.writer(sf)
        sw.writerow(["file_path", "archive_member", "skip_reason"])
        for rows, errs, skips in pool.map(run_task, tasks, chunksize=4):
            pw.writerows(rows)
            ew.writerows(errs)
            sw.writerows(skips)
            n_cols += len(rows)
            n_err += len(errs)
            n_skip += len(skips)
            n_done += 1
            if n_done % 2000 == 0:
                print(f"  {n_done}/{len(tasks)} tasks — {n_cols:,} columns, "
                      f"{n_err} errors, {n_skip} skips", flush=True)

    print(f"DONE: {n_cols:,} column profiles, {n_err} read errors, "
          f"{n_skip} non-tabular skips", flush=True)


if __name__ == "__main__":
    main()
