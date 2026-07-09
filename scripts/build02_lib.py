"""Build 02 compute library: canonical L-profile, hybrid factorizer, and
memory-safe value extraction.

Definitions follow /Users/<ANON>/Projects/prime-factorization/
CANONICAL_DEFINITIONS.md exactly:
  n = p1^a1 ... pr^ar;  contributions c_i = a_i * ln(p_i)
  L_j = j-th largest c_i / ln(n)         (prime-POWER parts, not copies)
  H2(n) = sum_j L_j^2  (over ALL r parts) ;  Tail = sum_{j>=3} L_j = 1-L1-L2
  digit stratum d(n) = floor(log10 n) + 1
  monetary: * 100, round-half-to-even (np.rint), then integerize
  exclusions: abs value first; drop |n| <= 1; non-integers dropped
Factorization: SPF sieve for n < 1e7; sympy isprime/factorint above
(exact — sympy uses trial division + Pollard rho/p-1/ECM). Values > 1e18
are logged, not factorized.
"""
import gzip
import io
import math
import re
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

SPF_LIMIT = 10_000_000
CEILING = 10 ** 18
HEAD_ROWS = 100_000
TOTAL_ROWS = 200_000
MAX_DIGITS = 18

_SPF = None
_memo = {}
_stats = Counter()


def init_worker():
    """Build the smallest-prime-factor sieve once per worker process."""
    global _SPF
    spf = np.zeros(SPF_LIMIT, dtype=np.int32)
    spf[1] = 1
    for i in range(2, int(SPF_LIMIT ** 0.5) + 1):
        if spf[i] == 0:
            spf[i::i][spf[i::i] == 0] = i
    spf[spf == 0] = 0  # remaining zeros are primes > sqrt(limit)
    # mark remaining primes as their own spf
    rem = np.nonzero(spf == 0)[0]
    spf[rem] = rem
    _SPF = spf


def factor_pairs(n):
    """Exact prime factorization -> list of (prime, exponent)."""
    global _memo
    if n < SPF_LIMIT:
        spf = _SPF
        out = {}
        while n > 1:
            p = int(spf[n])
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            out[p] = e
        return list(out.items())
    hit = _memo.get(n)
    if hit is not None:
        _stats["hits"] += 1
        return hit
    _stats["misses"] += 1
    from sympy import isprime, factorint
    m, out = n, {}
    for p in (2, 3, 5, 7, 11, 13):
        while m % p == 0:
            out[p] = out.get(p, 0) + 1
            m //= p
    if m > 1:
        if m < SPF_LIMIT:
            for p, e in factor_pairs(m):
                out[p] = out.get(p, 0) + e
        elif isprime(m):
            out[m] = out.get(m, 0) + 1
        else:
            for p, e in factorint(m).items():
                out[p] = out.get(p, 0) + int(e)
    pairs = list(out.items())
    if len(_memo) > 4_000_000:      # memory backstop
        _memo.clear()
        _stats["cache_clears"] += 1
    _memo[n] = pairs
    return pairs


def lprofile(n):
    """Return (L1, L2, L3, Tail, H2, r_parts) for integer n >= 2."""
    pairs = factor_pairs(n)
    ln_n = math.log(n)
    c = sorted((e * math.log(p) for p, e in pairs), reverse=True)
    inv = 1.0 / ln_n
    L = [x * inv for x in c]
    L1 = L[0]
    L2 = L[1] if len(L) > 1 else 0.0
    L3 = L[2] if len(L) > 2 else 0.0
    H2 = sum(x * x for x in L)
    return L1, L2, L3, 1.0 - L1 - L2, H2, len(L)


def profile_values(ints):
    """Accumulate per-record L-profiles over an int array.

    Returns dict of summary stats (means, digit histogram over 1..18,
    L1 percentiles, part-count mean) plus factorizer cache counters.
    """
    n_used = len(ints)
    if n_used == 0:
        return None
    uniq, counts = np.unique(ints, return_counts=True)
    L1s = np.empty(len(uniq))
    sums = np.zeros(5)          # L1, L2, L3, Tail, H2
    kparts = 0.0
    digit_hist = np.zeros(MAX_DIGITS, dtype=np.int64)
    for i, (v, c) in enumerate(zip(uniq.tolist(), counts.tolist())):
        l1, l2, l3, tail, h2, r = lprofile(v)
        L1s[i] = l1
        sums += np.array([l1, l2, l3, tail, h2]) * c
        kparts += r * c
        digit_hist[min(len(str(v)), MAX_DIGITS) - 1] += c
    means = sums / n_used
    # weighted percentiles of L1 over records
    order = np.argsort(L1s)
    cw = np.cumsum(counts[order]) / n_used
    def wpct(q):
        return float(L1s[order][np.searchsorted(cw, q)])
    return dict(n_used=int(n_used), n_distinct=int(len(uniq)),
                L1=float(means[0]), L2=float(means[1]), L3=float(means[2]),
                Tail=float(means[3]), H2=float(means[4]),
                k_parts_mean=float(kparts / n_used),
                digit_hist=digit_hist.tolist(),
                L1_p10=wpct(0.10), L1_p50=wpct(0.50), L1_p90=wpct(0.90))


def coerce_ints(series, monetary):
    """Canonical integerization. Returns (int64 array >=2, n_ceiling)."""
    s = series.dropna()
    if len(s) == 0:
        return np.array([], dtype=np.int64), 0
    if pd.api.types.is_bool_dtype(s):
        return np.array([], dtype=np.int64), 0
    if pd.api.types.is_integer_dtype(s):
        v = np.abs(s.to_numpy(dtype=np.int64, na_value=0))
        if monetary:
            over = v > CEILING // 100
            n_ceil = int(over.sum())
            v = v[~over] * 100
        else:
            over = v > CEILING
            n_ceil = int(over.sum())
            v = v[~over]
        return v[v > 1], n_ceil
    if pd.api.types.is_numeric_dtype(s):
        f = np.abs(s.to_numpy(dtype=float))
    else:
        t = s.astype(str).str.strip()
        cur = t.str.match(r"^[\$£€¥]")
        if cur.any():
            t = t.where(~cur, t.str.replace(r"^[\$£€¥]", "", n=1,
                                            regex=True).str.strip())
        thou = t.str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
        if thou.any():
            t = t.where(~thou, t.str.replace(",", "", regex=False))
        f = np.abs(pd.to_numeric(t, errors="coerce").to_numpy(dtype=float))
    f = f[np.isfinite(f)]
    if monetary:
        f = np.rint(f * 100.0)          # cents, round-half-to-even
    else:
        f = f[f == np.floor(f)]         # non-integers dropped
    over = f > CEILING
    n_ceil = int(over.sum())
    f = f[~over]
    v = f.astype(np.int64)
    return v[v > 1], n_ceil


def pop_cache_stats():
    global _stats
    out = dict(_stats)
    _stats = Counter()
    return out


# ------------------------------------------------------------ file readers
def decode(b):
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1")


def read_table(file_path, archive_member, sheet, ext, seed):
    """Read up to TOTAL_ROWS rows of the given table.
    Returns (df, sampled: bool)."""
    if archive_member and file_path.lower().endswith(".zip"):
        with zipfile.ZipFile(file_path) as zf:
            return _read_stream(lambda: zf.open(archive_member), ext, sheet)
    if archive_member:  # .gz member
        return _read_stream(lambda: gzip.open(file_path, "rb"), ext, sheet)
    if ext in (".csv", ".tsv", ".txt", ".dat"):
        return _read_seekable_text(file_path, ext, seed)
    return _read_stream(lambda: open(file_path, "rb"), ext, sheet)


def _sniff_sep(block, ext):
    lines = [ln for ln in decode(block).splitlines()[:40] if ln.strip()]
    best = None
    for d in [",", "\t", ";", "|"]:
        counts = [ln.count(d) for ln in lines]
        pos = [c for c in counts if c > 0]
        if len(pos) >= max(1, int(0.8 * len(lines))):
            common, freq = Counter(pos).most_common(1)[0]
            if freq / len(pos) >= 0.7 and (best is None or
                                           (freq / len(pos), common) > best[1]):
                best = (d, (freq / len(pos), common))
    if best:
        return best[0]
    return "\t" if ext == ".tsv" else ","


def _noheader_fix(df, text, kw):
    def is_data_value(c):
        c = str(c).strip()
        return bool(re.match(r"^-?\d{5,}$", c) or re.match(r"^-?\d*\.\d+$", c))
    if len(df.columns) and (
            all(re.match(r"^-?\d+(\.\d+)?$", str(c).strip())
                for c in df.columns)
            or any(is_data_value(c) for c in df.columns)):
        df = pd.read_csv(io.StringIO(text), header=None,
                         on_bad_lines="skip", **kw)
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df


def _read_seekable_text(path, ext, seed):
    import os
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        block = f.read(256 * 1024)
    sep = _sniff_sep(block, ext)
    with open(path, "rb") as f:
        header = f.readline()
        lines, consumed, eof = [], len(header), False
        for _ in range(HEAD_ROWS):
            ln = f.readline()
            if not ln:
                eof = True
                break
            consumed += len(ln)
            lines.append(ln)
    sampled = False
    if not eof and consumed < size:
        rng = np.random.default_rng([seed, 2026])
        offs = np.sort(rng.integers(consumed, size,
                                    TOTAL_ROWS - HEAD_ROWS))
        last = -1
        with open(path, "rb") as f:
            for off in offs.tolist():
                if last >= 0 and off < last:
                    continue
                f.seek(off)
                f.readline()
                pos = f.tell()
                if pos == last:
                    continue
                ln = f.readline()
                if ln:
                    last = pos
                    lines.append(ln)
        sampled = True
    text = decode(header + b"".join(lines))
    kw = dict(sep=sep)
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", **kw)
    df = _noheader_fix(df, text, kw)
    return df, sampled


def _read_stream(open_fn, ext, sheet):
    if ext in (".csv", ".tsv", ".txt", ".dat"):
        with open_fn() as fh:
            block = fh.read(256 * 1024)
        sep = _sniff_sep(block, ext)
        with open_fn() as fh:
            header = fh.readline()
            lines, eof = [], False
            for _ in range(TOTAL_ROWS):
                ln = fh.readline()
                if not ln:
                    eof = True
                    break
                lines.append(ln)
        text = decode(header + b"".join(lines))
        kw = dict(sep=sep)
        df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", **kw)
        df = _noheader_fix(df, text, kw)
        return df, not eof
    if ext == ".parquet":
        import pyarrow.parquet as pq
        import pyarrow as pa
        with open_fn() as fh:
            buf = io.BytesIO(fh.read())
        pf = pq.ParquetFile(buf)
        total = pf.metadata.num_rows
        got, batches = 0, []
        for b in pf.iter_batches(batch_size=16384):
            batches.append(b)
            got += b.num_rows
            if got >= TOTAL_ROWS:
                break
        df = (pa.Table.from_batches(batches).to_pandas().head(TOTAL_ROWS)
              if batches else pf.schema_arrow.empty_table().to_pandas())
        return df, total > TOTAL_ROWS
    if ext in (".xlsx", ".xls"):
        with open_fn() as fh:
            buf = io.BytesIO(fh.read())
        df = pd.read_excel(buf, sheet_name=sheet or 0, nrows=TOTAL_ROWS)
        return df, len(df) >= TOTAL_ROWS
    if ext in (".jsonl", ".ndjson"):
        import json as _json
        rows = []
        with open_fn() as fh:
            for i, ln in enumerate(fh):
                if i >= TOTAL_ROWS:
                    break
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(_json.loads(decode(ln)))
                    except Exception:
                        pass
        return pd.json_normalize(rows), len(rows) >= TOTAL_ROWS
    if ext == ".json":
        import json as _json
        with open_fn() as fh:
            obj = _json.loads(decode(fh.read()))
        if isinstance(obj, list):
            return pd.json_normalize(obj[:TOTAL_ROWS]), len(obj) > TOTAL_ROWS
        if isinstance(obj, dict):
            vals = list(obj.values())
            if vals and all(isinstance(v, list) for v in vals) and \
                    len({len(v) for v in vals}) == 1:
                return (pd.DataFrame({k: v[:TOTAL_ROWS]
                                      for k, v in obj.items()}),
                        len(vals[0]) > TOTAL_ROWS)
            if vals and all(isinstance(v, dict) for v in vals):
                return (pd.DataFrame.from_dict(obj, orient="index")
                        .head(TOTAL_ROWS), len(vals) > TOTAL_ROWS)
        raise ValueError("json not tabular")
    raise ValueError(f"unhandled extension {ext}")
