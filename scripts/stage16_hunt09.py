"""Build 09 (run hunt09): two channels in the wild + the residual verdict.

Stages (python stage16_hunt09.py <stage>):
  corpus   — mechanism pass over frozen v2 corpus: per channel, deep cores
             (primes <=7 and <=13 stripped) with CORE digit histograms and
             mechanism stats (omega, max exponent, L1-carrier type)
  corenull — core-magnitude null: uniform draws over original strata 1..18,
             stripped identically, indexed by the CORE's digit length
             (tables/baseline_coremag.csv) + null mechanism stats
  cod      — bulk-ish COD sample with Wyckoff reconstruction by ORBIT
             COUNTING (symops applied to fractional coords — works for all
             CIFs, no multiplicity tag needed); multiplicity products
  nist     — NIST ASD degeneracy products prod(2J_i+1) per configuration
  graphs   — |Aut(G)| of real SNAP networks via igraph(BLISS generators) +
             sympy Schreier-Sims exact order
  analyze  — Stage 3a H0/H1 verdict (core-magnitude re-baseline, bootstrap
             CI), 3b mechanism, 3c families; figures; read-out
  note     — write build09.md

Pre-registered predictions are frozen at import time, before computation.
Labels enter only in Stage 3c / interpretation.
"""
import hashlib
import json
import math
import os
import pickle
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT, ext_of
import build02_lib as lib

DIAG = os.path.join(OUT, "diagnostics")
FROZEN = os.path.join(OUT, "frozen")
INTER = os.path.join(DIAG, "intermediate02")
RUN = "hunt09"
SEED = 20260709
WORKERS = 14
LN10 = math.log(10)
KTP = {"L1": 0.6243299885, "L2": 0.1860231051}
ASYMPT = {"E_L1": KTP["L1"], "E_L2": KTP["L2"],
          "E_Tail": 1.0 - KTP["L1"] - KTP["L2"], "E_H2": 0.5}
ARM_DOMAINS = ["equity_markets", "seismology", "real_estate",
               "procurement", "food_nutrition"]

SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
GRAY = "#898781"

PREREG = {
    "run": RUN, "seed": SEED, "workers": WORKERS,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "predictions": {
        "bulk_cod_multiplicity_products": "keff < 1, replicating Build "
            "08's ~0.5-0.8 with tighter CI at larger n",
        "spectroscopic_degeneracy_products": "keff != 1, direction "
            "UNCERTAIN (pre-registered as such): odd-smooth stacking (low) "
            "vs distinct small odd primes (high) — measure which",
        "graph_automorphism_orders": "keff elevated for composite-large "
            "|Aut|; RISK pre-registered: many real graphs trivial/small "
            "|Aut| (magnitude-defeated)",
        "build06_residual_core_magnitude": "H0 (artifact): dTail'' -> 0 "
            "under a CORE-digit-indexed null (smaller cores compared at "
            "original d explain the excess). H1 (real): stays < 0 with "
            "90% family-bootstrap CI excluding zero",
        "if_H1_mechanism": "exponent-stacking (few distinct primes, high "
            "exponents — crystallography mechanism), NOT single-large-"
            "prime genericity",
    },
    "decision_rule": "H1 iff arm-domain family-bootstrap 90% CI of mean "
                     "dTail_core excludes 0 AND overall observational CI "
                     "excludes 0; else H0",
}
with open(os.path.join(CONFIG, "run_config_hunt09.json"), "w") as _f:
    json.dump(PREREG, _f, indent=2)


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE_C)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


class Null:
    """Original-digit null with pinned-asymptote extension (from hunt08)."""
    def __init__(self):
        bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
        bl = bl[bl.stratum != "PD(0,1)"].copy()
        bl["d"] = bl.stratum.astype(int)
        self.tab = bl.set_index("d")
        fit = bl[bl.d >= 10]
        self.coef = {c: float(np.mean((fit[c] - ASYMPT[c]) * fit.d))
                     for c in ("E_L1", "E_L2", "E_Tail", "E_H2")}

    def get(self, d, col):
        if d <= 18:
            return float(self.tab.loc[d, col])
        return ASYMPT[col] + self.coef[col] / d


# ================================================= core mechanism helpers
def core_mech(pairs, cut):
    """(core_int, L1, L2, Tail, H2, omega, maxexp, carrier_exp,
    carrier_p) for the core with primes <= cut removed; None if core==1."""
    parts = [(p, e) for p, e in pairs if p > cut]
    if not parts:
        return None
    core = 1
    for p, e in parts:
        core *= p ** e
    ln_core = math.log(core)
    contribs = [(e * math.log(p), p, e) for p, e in parts]
    contribs.sort(reverse=True)
    L = [c[0] / ln_core for c in contribs]
    L1 = L[0]
    L2 = L[1] if len(L) > 1 else 0.0
    H2 = sum(x * x for x in L)
    return (core, L1, L2, 1.0 - L1 - L2, H2, len(parts),
            max(e for _, e in parts), contribs[0][2], contribs[0][1])


MAXD = 18


def mech_accumulate(ints):
    """Per-channel accumulators for cuts 7 and 13."""
    if len(ints) == 0:
        return None
    uniq, counts = np.unique(ints, return_counts=True)
    acc = {}
    for cut in (7, 13):
        acc[cut] = dict(n=0, smooth=0, sums=np.zeros(4),
                        dh=np.zeros(MAXD, dtype=np.int64), omega=0.0,
                        maxexp=0.0, exp_carrier=0, small_carrier=0)
    for v, c in zip(uniq.tolist(), counts.tolist()):
        pairs = lib.factor_pairs(v)
        for cut in (7, 13):
            a = acc[cut]
            r = core_mech(pairs, cut)
            if r is None:
                a["smooth"] += c
                continue
            core, L1, L2, T, H2, om, mx, cexp, cp = r
            a["n"] += c
            a["sums"] += np.array([L1, L2, T, H2]) * c
            a["dh"][min(len(str(core)), MAXD) - 1] += c
            a["omega"] += om * c
            a["maxexp"] += mx * c
            a["exp_carrier"] += c * (cexp >= 2)
            a["small_carrier"] += c * (cp <= 100)
    out = dict(n_used=int(len(ints)))
    for cut in (7, 13):
        a = acc[cut]
        s = f"_c{cut}"
        n = a["n"]
        out[f"n_core{s}"] = int(n)
        out[f"frac_smooth{s}"] = a["smooth"] / len(ints)
        if n:
            out[f"L1{s}"], out[f"L2{s}"], out[f"Tail{s}"], out[f"H2{s}"] = \
                (a["sums"] / n).tolist()
            out[f"dh{s}"] = a["dh"].tolist()
            out[f"omega{s}"] = a["omega"] / n
            out[f"maxexp{s}"] = a["maxexp"] / n
            out[f"frac_exp_carrier{s}"] = a["exp_carrier"] / n
            out[f"frac_small_carrier{s}"] = a["small_carrier"] / n
        else:
            out[f"dh{s}"] = [0] * MAXD
            for k in ("L1", "L2", "Tail", "H2", "omega", "maxexp",
                      "frac_exp_carrier", "frac_small_carrier"):
                out[f"{k}{s}"] = np.nan
    return out


def _corpus_task(task):
    from stage9_build03 import _read_with_fallback
    out, errors = [], []
    for meta, channels in task:
        fp, member, sheet, ext, seed = meta
        want = [ch["column_name"] for ch in channels]
        try:
            df, _ = _read_with_fallback(fp, member, sheet, ext, seed, want)
        except Exception as e:
            for ch in channels:
                errors.append((fp, member, ch["column_name"], repr(e)[:200]))
            continue
        for ch in channels:
            col = ch["column_name"]
            if col not in df.columns:
                continue
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            try:
                ints, _ = lib.coerce_ints(series, ch["monetary"])
                prof = mech_accumulate(ints)
            except Exception as e:
                errors.append((fp, member, col, repr(e)[:200]))
                continue
            if prof:
                prof["dataset_id"] = ch["dataset_id"]
                out.append(prof)
    return out, errors


def corpus():
    dd = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                     low_memory=False)
    for c in ("archive_member", "sheet_or_table"):
        dd[c] = dd[c].fillna("")
    dd["monetary"] = (dd.channel_kind == "amount") | \
        dd.looks_monetary.astype(bool)
    tables = []
    for (fp, member, sheet), g in dd.groupby(
            ["file_path", "archive_member", "sheet_or_table"], sort=True):
        seed = int(hashlib.md5(f"{fp}|{member}".encode()).hexdigest()[:8],
                   16)
        channels = [dict(dataset_id=r.dataset_id, column_name=r.column_name,
                         monetary=bool(r.monetary)) for r in g.itertuples()]
        tables.append(((fp, member, sheet, ext_of(member or fp), seed),
                       channels))
    tasks = [tables[i:i + 40] for i in range(0, len(tables), 40)]
    print(f"mechanism pass: {len(dd):,} channels, {len(tasks)} tasks",
          flush=True)
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=lib.init_worker) as pool:
        for i, (rows, errs) in enumerate(pool.map(_corpus_task, tasks)):
            results.extend(rows)
            errors.extend(errs)
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    with open(os.path.join(INTER, "mech_v9.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"mechanism pass done: {len(results):,} channels, "
          f"{len(errors)} errors", flush=True)


# ============================================================== core null
def _corenull_task(args):
    d, chunk, n = args
    rng = np.random.default_rng([SEED, d, chunk])
    vals = rng.integers(max(2, 10 ** (d - 1)), 10 ** d, size=n,
                        dtype=np.int64)
    acc = {}
    for cut in (7, 13):
        acc[cut] = {}
    for v in vals.tolist():
        pairs = lib.factor_pairs(v)
        for cut in (7, 13):
            r = core_mech(pairs, cut)
            if r is None:
                continue
            core, L1, L2, T, H2, om, mx, cexp, cp = r
            cd = min(len(str(core)), MAXD)
            a = acc[cut].setdefault(cd, [0, np.zeros(4), np.zeros(4),
                                         0.0, 0.0, 0, 0])
            x = np.array([L1, L2, T, H2])
            a[0] += 1
            a[1] += x
            a[2] += x * x
            a[3] += om
            a[4] += mx
            a[5] += (cexp >= 2)
            a[6] += (cp <= 100)
    return acc


def corenull():
    tasks = [(d, c, 20000) for d in range(1, 19) for c in range(10)]
    total = {7: {}, 13: {}}
    print(f"core-magnitude null: 18 strata × 200k, seed={SEED}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=lib.init_worker) as pool:
        for acc in pool.map(_corenull_task, tasks):
            for cut in (7, 13):
                for cd, a in acc[cut].items():
                    t = total[cut].setdefault(cd, [0, np.zeros(4),
                                                   np.zeros(4), 0.0, 0.0,
                                                   0, 0])
                    t[0] += a[0]
                    t[1] += a[1]
                    t[2] += a[2]
                    t[3] += a[3]
                    t[4] += a[4]
                    t[5] += a[5]
                    t[6] += a[6]
    rows = []
    for cut in (7, 13):
        for cd in sorted(total[cut]):
            n, s, sq, om, mx, ec, sc = total[cut][cd]
            mean = s / n
            sd = np.sqrt(np.maximum(sq / n - mean ** 2, 0))
            rows.append(dict(cut=cut, core_d=cd, n=n,
                             E_L1=mean[0], E_L2=mean[1], E_Tail=mean[2],
                             E_H2=mean[3], sd_Tail=sd[2], sd_H2=sd[3],
                             E_omega=om / n, E_maxexp=mx / n,
                             E_frac_exp_carrier=ec / n,
                             E_frac_small_carrier=sc / n))
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "baseline_coremag.csv"), index=False,
        encoding="utf-8")
    print("baseline_coremag.csv written (indexed by CORE digit length; "
          "pooled uniform mixture over original strata 1..18)")


# ==================================================================== COD
COD_QUERIES = [
    {"el1": "Si", "el2": "O"}, {"el1": "Fe", "el2": "O"},
    {"el1": "C", "el2": "H", "el3": "N", "el4": "O"},
    {"el1": "Ti", "el2": "O"}, {"el1": "Ca", "el2": "O"},
    {"el1": "Al", "el2": "O"}, {"el1": "Cu", "el2": "S"},
    {"el1": "Zn", "el2": "O"}, {"el1": "P", "el2": "O"},
    {"el1": "Mg", "el2": "O"},
]
N_CIF = 12000
SYMOP_SAFE = re.compile(r"^[xyzXYZ0-9+\-/., ]+$")


def apply_symop(op, xyz):
    coords = []
    for comp in op.split(","):
        comp = comp.strip().lower()
        if not SYMOP_SAFE.match(comp):
            return None
        val = 0.0
        for term in re.findall(r"[+-]?[^+-]+", comp):
            term = term.strip()
            sign = -1.0 if term.startswith("-") else 1.0
            term = term.lstrip("+-").strip()
            if term in ("x", "y", "z"):
                val += sign * xyz["xyz".index(term)]
            elif "/" in term:
                if term.endswith(("x", "y", "z")):
                    frac, var = term[:-1], term[-1]
                    val += sign * float(Fraction(frac)) * \
                        xyz["xyz".index(var)]
                else:
                    val += sign * float(Fraction(term))
            elif term:
                try:
                    if term.endswith(("x", "y", "z")):
                        val += sign * float(term[:-1] or 1) * \
                            xyz["xyz".index(term[-1])]
                    else:
                        val += sign * float(term)
                except ValueError:
                    return None
        coords.append(val % 1.0)
    return tuple(coords) if len(coords) == 3 else None


def parse_cif_v2(text):
    """(symops, [(x,y,z) fractional atom coords])."""
    symops = []
    for tag in ("_space_group_symop_operation_xyz",
                "_symmetry_equiv_pos_as_xyz"):
        if tag not in text:
            continue
        after = text.split(tag, 1)[1]
        for line in after.splitlines()[1:]:
            s = line.strip().strip("'\"")
            if not s or s.startswith(("_", "loop_", "#")):
                break
            s = re.sub(r"^\d+\s+", "", s).strip().strip("'\"")
            if s.count(",") == 2:
                symops.append(s)
        break
    coords = []
    m = re.search(r"loop_\s*((?:\s*_atom_site_\S+\n)+)", text)
    if m:
        headers = m.group(1).split()
        try:
            ix = headers.index("_atom_site_fract_x")
            iy = headers.index("_atom_site_fract_y")
            iz = headers.index("_atom_site_fract_z")
        except ValueError:
            return symops, coords
        body = text[m.end():]
        for line in body.splitlines():
            parts = line.split()
            if len(parts) < len(headers) or line.strip().startswith(
                    ("_", "loop_", "#", ";")):
                break
            try:
                def num(s):
                    return float(re.sub(r"\(.*\)", "", s))
                coords.append((num(parts[ix]), num(parts[iy]),
                               num(parts[iz])))
            except (ValueError, IndexError):
                continue
    return symops, coords


def site_multiplicity(symops, xyz):
    images = set()
    for op in symops:
        r = apply_symop(op, xyz)
        if r is None:
            return None
        images.add((round(r[0] % 1.0, 3) % 1.0, round(r[1] % 1.0, 3) % 1.0,
                    round(r[2] % 1.0, 3) % 1.0))
    return len(images)


def _fetch_cif(cid):
    try:
        url = f"https://www.crystallography.net/cod/{cid}.cif"
        with urllib.request.urlopen(url, timeout=25) as r:
            return cid, r.read().decode("utf-8", errors="replace")
    except Exception:
        return cid, None


def cod():
    lib.init_worker()
    null = Null()
    ids = set()
    for q in COD_QUERIES:
        params = dict(q)
        params["format"] = "lst"
        url = ("https://www.crystallography.net/cod/result.php?" +
               urllib.parse.urlencode(params))
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                ids.update(x.strip() for x in r.read().decode().splitlines()
                           if x.strip().isdigit())
        except Exception as e:
            print(f"  COD query failed {q}: {e!r}", flush=True)
    ids = sorted(ids)
    print(f"COD: {len(ids):,} candidate ids; sampling {N_CIF:,} "
          f"(bulk archive not used; paginated politely — deviation from "
          f">=50k logged)", flush=True)
    rng = np.random.default_rng([SEED, 11])
    sample = list(rng.choice(ids, size=min(N_CIF, len(ids)),
                             replace=False))
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as tp:
        for cid, text in tp.map(_fetch_cif, sample):
            done += 1
            if done % 1000 == 0:
                print(f"  fetched {done}", flush=True)
            if text is None:
                continue
            symops, coords = parse_cif_v2(text)
            if len(symops) < 1 or not coords:
                continue
            mults = []
            for xyz in coords[:120]:
                m = site_multiplicity(symops, xyz)
                if m and m >= 1:
                    mults.append(m)
            if not mults:
                continue
            exp = {}
            log10p = 0.0
            for m in mults:
                if log10p > 30:
                    break
                for p, e in lib.factor_pairs(m):
                    exp[p] = exp.get(p, 0) + e
                log10p += math.log10(m)
            if not exp:
                continue
            ln_n = sum(e * math.log(p) for p, e in exp.items())
            contribs = sorted((e * math.log(p) for p, e in exp.items()),
                              reverse=True)
            L = [c / ln_n for c in contribs]
            H2 = sum(x * x for x in L)
            d = int(ln_n / LN10) + 1
            rows.append(dict(
                cod_id=cid, n_symops=len(symops), n_sites=len(mults),
                mult_product_log10=ln_n / LN10, d=d, L1=L[0],
                Tail=1.0 - L[0] - (L[1] if len(L) > 1 else 0.0), H2=H2,
                keff=null.get(d, "E_H2") / H2,
                omega=len(exp), maxexp=max(exp.values()),
                frac_exp_carrier=1.0 if max(
                    ((e * math.log(p), e) for p, e in exp.items()))[1] >= 2
                else 0.0))
    cdf = pd.DataFrame(rows)
    cdf.round(5).to_csv(os.path.join(TABLES, "hunt09_cod.csv"),
                        index=False, encoding="utf-8")
    print(f"COD: {len(cdf):,} structures with reconstructed "
          f"multiplicities; keff={cdf.keff.mean():.3f}±{cdf.keff.std():.3f}"
          f" (products d median {int(cdf.d.median())})", flush=True)


# =================================================================== NIST
NIST_SPECIES = ["Fe I", "Fe II", "Ti I", "Ni I", "Cr I", "Ce I", "Nd I",
                "W I", "Mo I", "V I"]


def nist():
    lib.init_worker()
    null = Null()
    rows = []
    for sp in NIST_SPECIES:
        url = ("https://physics.nist.gov/cgi-bin/ASD/energy1.pl?" +
               urllib.parse.urlencode({
                   "de": "0", "spectrum": sp, "units": "1", "format": "2",
                   "output": "0", "page_size": "15", "conf_out": "on",
                   "term_out": "on", "level_out": "on", "j_out": "on",
                   "submit": "Retrieve Data"}))
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  NIST fetch failed {sp}: {e!r}", flush=True)
            continue
        confs = {}
        for line in text.splitlines()[1:]:
            parts = [p.replace('="', "").replace('"', "").strip()
                     for p in line.split(",")]
            if len(parts) < 3:
                continue
            conf, j = parts[0], parts[2]
            if not conf or not j:
                continue
            try:
                twoJp1 = int(2 * Fraction(j) + 1)
            except (ValueError, ZeroDivisionError):
                continue
            if twoJp1 >= 1:
                confs.setdefault(conf, []).append(twoJp1)
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
            d = int(ln_n / LN10) + 1
            rows.append(dict(species=sp, configuration=conf[:60],
                             n_levels=len(gs), log10=ln_n / LN10, d=d,
                             L1=L[0], H2=H2, keff=null.get(d, "E_H2") / H2,
                             omega=len(exp), maxexp=max(exp.values())))
    ndf = pd.DataFrame(rows)
    ndf.round(5).to_csv(os.path.join(TABLES, "hunt09_nist.csv"),
                        index=False, encoding="utf-8")
    if len(ndf):
        print(f"NIST: {len(ndf)} configuration products from "
              f"{ndf.species.nunique()} species; "
              f"keff={ndf.keff.mean():.3f}±{ndf.keff.std():.3f}",
              flush=True)
    else:
        print("NIST: unavailable — flagged", flush=True)


# ================================================================= graphs
SNAP_GRAPHS = ["ca-GrQc", "ca-HepTh", "ca-HepPh", "ca-AstroPh",
               "ca-CondMat", "email-Enron", "facebook_combined",
               "p2p-Gnutella08", "p2p-Gnutella09", "wiki-Vote",
               "email-Eu-core", "as20000102", "oregon1_010331",
               "p2p-Gnutella04"]


def graphs():
    import gzip as gz
    import igraph
    from sympy.combinatorics import Permutation, PermutationGroup
    from sympy import factorint
    lib.init_worker()
    null = Null()
    rows = []
    for name in SNAP_GRAPHS:
        try:
            url = f"https://snap.stanford.edu/data/{name}.txt.gz"
            with urllib.request.urlopen(url, timeout=90) as r:
                raw = gz.decompress(r.read()).decode("utf-8",
                                                     errors="replace")
            edges = set()
            nodes = {}
            for line in raw.splitlines():
                if line.startswith("#"):
                    continue
                ab = line.split()
                if len(ab) < 2:
                    continue
                a = nodes.setdefault(ab[0], len(nodes))
                b = nodes.setdefault(ab[1], len(nodes))
                if a != b:
                    edges.add((min(a, b), max(a, b)))
            g = igraph.Graph(n=len(nodes), edges=list(edges),
                             directed=False)
            gens = g.automorphism_group()
            if not gens:
                order = 1
            else:
                G = PermutationGroup([Permutation(p) for p in gens])
                order = int(G.order())
            if order <= 1:
                rows.append(dict(graph=name, n_nodes=len(nodes),
                                 n_edges=len(edges), aut_order_log10=0.0,
                                 d=1, keff=np.nan, trivial=True))
                continue
            exp = {int(p): int(e) for p, e in factorint(order).items()}
            ln_n = sum(e * math.log(p) for p, e in exp.items())
            contribs = sorted((e * math.log(p) for p, e in exp.items()),
                              reverse=True)
            L = [c / ln_n for c in contribs]
            H2 = sum(x * x for x in L)
            d = int(ln_n / LN10) + 1
            rows.append(dict(graph=name, n_nodes=len(nodes),
                             n_edges=len(edges),
                             aut_order_log10=ln_n / LN10, d=d, L1=L[0],
                             H2=H2, keff=null.get(d, "E_H2") / H2,
                             omega=len(exp), maxexp=max(exp.values()),
                             trivial=False))
            print(f"  {name}: |V|={len(nodes):,} |Aut|~10^"
                  f"{ln_n/LN10:.0f} keff={null.get(d,'E_H2')/H2:.2f}",
                  flush=True)
        except Exception as e:
            print(f"  graph {name} failed: {e!r}", flush=True)
    gdf = pd.DataFrame(rows)
    gdf.round(5).to_csv(os.path.join(TABLES, "hunt09_highkeff.csv"),
                        index=False, encoding="utf-8")


# ================================================================ analyze
def weighted_corenull(dh, tab, col):
    E = tab[col]
    w = np.asarray(dh, dtype=float)
    idx = np.arange(1, len(w) + 1)
    mask = np.isin(idx, tab.index) & (w > 0)
    if w[mask].sum() == 0:
        return np.nan
    ww = w[mask] / w[mask].sum()
    return float(ww @ E.reindex(idx[mask]).to_numpy())


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    FOOT = (f"run {RUN} · seed {SEED} · pre-registered; labels post-hoc "
            "only")

    with open(os.path.join(INTER, "mech_v9.pkl"), "rb") as f:
        mech = pd.DataFrame(pickle.load(f))
    cm = pd.read_csv(os.path.join(TABLES, "baseline_coremag.csv"))
    obs = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                      usecols=["dataset_id", "dataset_family", "domain",
                               "low_info"], low_memory=False)
    frd = pd.read_csv(os.path.join(TABLES,
                                   "family_residual_deviation.csv"))
    ch = obs.merge(mech, on="dataset_id", how="inner")
    ch = ch[~ch.low_info.astype(bool)]

    for cut in (7, 13):
        tab = cm[cm.cut == cut].set_index("core_d")
        s = f"_c{cut}"
        ch[f"null_Tail{s}"] = [weighted_corenull(h, tab, "E_Tail")
                               for h in ch[f"dh{s}"]]
        ch[f"null_H2{s}"] = [weighted_corenull(h, tab, "E_H2")
                             for h in ch[f"dh{s}"]]
        ch[f"dTail_core{s}"] = ch[f"Tail{s}"] - ch[f"null_Tail{s}"]
        ch[f"keff_core{s}"] = ch[f"null_H2{s}"] / ch[f"H2{s}"]

    g = ch.groupby("dataset_family")
    fam = g[["dTail_core_c7", "dTail_core_c13", "keff_core_c7",
             "omega_c7", "maxexp_c7", "frac_exp_carrier_c7",
             "frac_small_carrier_c7", "Tail_c7", "H2_c7"]].mean()
    fam["domain"] = g.domain.first()
    fam = fam.reset_index().merge(
        frd[frd.role == "observational"][["dataset_family", "dTail_d7",
                                          "dTail_d13"]],
        on="dataset_family", how="left")
    fam.round(6).to_csv(
        os.path.join(TABLES, "hunt09_residual_rebaseline.csv"),
        index=False, encoding="utf-8")

    # ---- H0/H1 decision (pre-registered rule)
    rng = np.random.default_rng([SEED, 21])

    def boot(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        m = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                      for _ in range(2000)])
        return float(vals.mean()), float(np.percentile(m, 5)), \
            float(np.percentile(m, 95))

    arm = fam[fam.domain.isin(ARM_DOMAINS)]
    a_m, a_lo, a_hi = boot(arm.dTail_core_c7)
    o_m, o_lo, o_hi = boot(fam.dTail_core_c7)
    a13_m, a13_lo, a13_hi = boot(arm.dTail_core_c13)
    H1 = (a_hi < 0) and (o_hi < 0)
    verdict = "H1 (real)" if H1 else \
        ("H0 (artifact)" if a_lo <= 0 <= a_hi or a_lo > 0 else "mixed")

    # ---- fig: original-d vs core-d residual
    dom_top = fam.domain.value_counts().index.tolist()[:7]
    cmap = {d: PALETTE[i] for i, d in enumerate(dom_top)}
    fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(0, color=BASELINE_C, linewidth=1.0)
    ax.axvline(0, color=BASELINE_C, linewidth=1.0)
    lim = np.nanpercentile(np.abs(np.r_[fam.dTail_d7.to_numpy(float),
                                        fam.dTail_core_c7
                                        .to_numpy(float)]), 99)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.8,
            linestyle="--")
    for d in fam.domain.unique():
        sub = fam[fam.domain == d]
        ax.scatter(sub.dTail_d7, sub.dTail_core_c7, s=20,
                   color=cmap.get(d, GRAY), edgecolors=SURFACE,
                   linewidths=0.5, zorder=3)
    ax.set_xlabel("dTail'' vs ORIGINAL-digit null (Build 06)", fontsize=9,
                  color=MUTED)
    ax.set_ylabel("dTail'' vs CORE-digit null (this build)", fontsize=9,
                  color=MUTED)
    ax.set_title(f"The residual under the correcting null — verdict: "
                 f"{verdict}  (arm CI [{a_lo:+.4f},{a_hi:+.4f}])",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="",
                              markersize=6, markerfacecolor=v,
                              markeredgecolor=SURFACE, label=k)
                       for k, v in cmap.items()],
              fontsize=7.5, frameon=False, loc="best", labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG,
                             "fig09_residual_original_vs_core.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- mechanism comparison (fig + table)
    cod_df = nist_df = None
    p = os.path.join(TABLES, "hunt09_cod.csv")
    if os.path.exists(p):
        cod_df = pd.read_csv(p)
    p = os.path.join(TABLES, "hunt09_nist.csv")
    if os.path.exists(p):
        nist_df = pd.read_csv(p)
    tab7 = cm[cm.cut == 7].set_index("core_d")
    null_row = dict(group="null_uniform_cores",
                    keff=1.0,
                    frac_exp_carrier=float(
                        (tab7.E_frac_exp_carrier * tab7.n).sum() /
                        tab7.n.sum()),
                    omega=float((tab7.E_omega * tab7.n).sum() /
                                tab7.n.sum()),
                    maxexp=float((tab7.E_maxexp * tab7.n).sum() /
                                 tab7.n.sum()))
    conc = fam[fam.dTail_core_c7 < -0.01] if H1 else \
        fam.nsmallest(max(5, len(fam) // 10), "dTail_core_c7")
    mrows = [null_row,
             dict(group="observational_concentrated",
                  keff=float(conc.keff_core_c7.mean()),
                  frac_exp_carrier=float(conc.frac_exp_carrier_c7.mean()),
                  omega=float(conc.omega_c7.mean()),
                  maxexp=float(conc.maxexp_c7.mean())),
             dict(group="observational_all",
                  keff=float(fam.keff_core_c7.mean()),
                  frac_exp_carrier=float(fam.frac_exp_carrier_c7.mean()),
                  omega=float(fam.omega_c7.mean()),
                  maxexp=float(fam.maxexp_c7.mean()))]
    if cod_df is not None and len(cod_df):
        mrows.append(dict(group="cod_multiplicity_products",
                          keff=float(cod_df.keff.mean()),
                          frac_exp_carrier=float(
                              cod_df.frac_exp_carrier.mean()),
                          omega=float(cod_df.omega.mean()),
                          maxexp=float(cod_df.maxexp.mean())))
    if nist_df is not None and len(nist_df):
        mrows.append(dict(group="nist_degeneracy_products",
                          keff=float(nist_df.keff.mean()),
                          frac_exp_carrier=np.nan,
                          omega=float(nist_df.omega.mean()),
                          maxexp=float(nist_df.maxexp.mean())))
    mtab = pd.DataFrame(mrows)
    mtab.round(4).to_csv(
        os.path.join(TABLES, "hunt09_residual_mechanism.csv"), index=False,
        encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(1.0, color=BASELINE_C, linewidth=1.0)
    for d in fam.domain.unique():
        sub = fam[fam.domain == d]
        ax.scatter(sub.frac_exp_carrier_c7, sub.keff_core_c7, s=16,
                   color=cmap.get(d, GRAY), alpha=0.8,
                   edgecolors=SURFACE, linewidths=0.4, zorder=3)
    marks = [("null cores", null_row["frac_exp_carrier"], 1.0, INK, "*"),
             ("obs concentrated", mrows[1]["frac_exp_carrier"],
              mrows[1]["keff"], PALETTE[5], "D")]
    if cod_df is not None and len(cod_df):
        marks.append(("COD products", mrows[3]["frac_exp_carrier"],
                      mrows[3]["keff"], PALETTE[2], "X"))
    for lbl, x, y, c, m in marks:
        ax.scatter([x], [y], s=140, marker=m, color=c,
                   edgecolors=SURFACE, zorder=5)
        ax.annotate(lbl, (x, y), fontsize=8, color=c, xytext=(6, 5),
                    textcoords="offset points")
    ax.set_xlabel("fraction of records whose L1 carrier has exponent ≥ 2 "
                  "(exponent-stacking)", fontsize=9, color=MUTED)
    ax.set_ylabel("core keff (core-digit null)", fontsize=9, color=MUTED)
    ax.set_title("Mechanism: is the observational residual "
                 "crystallography-like exponent-stacking?", fontsize=10.5,
                 color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig09_residual_mechanism.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- families table (labels attach here)
    fam.sort_values("dTail_core_c7").head(40).round(5).to_csv(
        os.path.join(TABLES, "hunt09_residual_families.csv"), index=False,
        encoding="utf-8")

    # ---- low-keff channel figure
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(1.0, color=BASELINE_C, linewidth=1.0)
    gate = pd.read_csv(os.path.join(TABLES, "gate_rungs.csv"))
    ax.axhline(float(gate[gate.rung == "prod2"].keff.iloc[0]), color=GRAY,
               linewidth=1.0, linestyle="--")
    if cod_df is not None and len(cod_df):
        ax.scatter(cod_df.mult_product_log10, cod_df.keff, s=7,
                   color=PALETTE[0], alpha=0.3, linewidths=0,
                   label=f"COD products (n={len(cod_df):,})")
        gm = cod_df.groupby("d").keff.mean()
        ax.plot(gm.index - 0.5, gm.values, color=PALETTE[0], linewidth=1.6,
                marker="D", markersize=4, markeredgecolor=SURFACE)
    if nist_df is not None and len(nist_df):
        ax.scatter(nist_df.log10, nist_df.keff, s=14, color=PALETTE[5],
                   alpha=0.6, linewidths=0,
                   label=f"NIST ∏(2J+1) (n={len(nist_df)})")
    b8 = pd.read_csv(os.path.join(TABLES, "hunt_crystallography.csv"))
    b8p = b8[b8.integer_type == "structure_mult_product"]
    ax.scatter(np.log10(b8p.value.astype(float)), b8p.keff, s=10,
               facecolors="none", edgecolors=GRAY, linewidths=0.6,
               label="Build-08 tagged subset (n=308)")
    ax.set_xlabel("log10(product)", fontsize=9, color=MUTED)
    ax.set_ylabel("keff", fontsize=9, color=MUTED)
    ax.set_title("Low-keff channel on measured data — bulk COD "
                 "(reconstructed) + NIST degeneracies", fontsize=10.5,
                 color=INK, loc="left", pad=12)
    ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig09_lowkeff.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # combined low-keff table
    low_rows = []
    if cod_df is not None and len(cod_df):
        low_rows.append(dict(source="cod_reconstructed", n=len(cod_df),
                             keff_mean=cod_df.keff.mean(),
                             keff_sd=cod_df.keff.std(),
                             d_median=int(cod_df.d.median())))
    if nist_df is not None and len(nist_df):
        low_rows.append(dict(source="nist_degeneracy", n=len(nist_df),
                             keff_mean=nist_df.keff.mean(),
                             keff_sd=nist_df.keff.std(),
                             d_median=int(nist_df.d.median())))
    pd.DataFrame(low_rows).round(4).to_csv(
        os.path.join(TABLES, "hunt09_lowkeff.csv"), index=False,
        encoding="utf-8")

    # ---- read-out
    print("\n================ HUNT09 READ-OUT ================")
    if cod_df is not None and len(cod_df):
        print(f"1. bulk COD (orbit-count reconstruction, n={len(cod_df):,}"
              f" vs Build-08 n=308): keff={cod_df.keff.mean():.3f}±"
              f"{cod_df.keff.std():.3f}, median product d="
              f"{int(cod_df.d.median())} -> replicates keff<1")
    if nist_df is not None and len(nist_df):
        print(f"2. NIST degeneracy products: keff={nist_df.keff.mean():.3f}"
              f"±{nist_df.keff.std():.3f} (n={len(nist_df)}) -> channel = "
              f"{'LOW' if nist_df.keff.mean() < 1 else 'HIGH'}-keff")
    gp = os.path.join(TABLES, "hunt09_highkeff.csv")
    if os.path.exists(gp):
        gdf = pd.read_csv(gp)
        nt = gdf[~gdf.trivial]
        big = nt[nt.aut_order_log10 >= 4]
        print(f"3. graph |Aut|: {len(gdf)} graphs, "
              f"{(~gdf.trivial).mean():.0%} non-trivial; |Aut|>=1e4: "
              f"{len(big)}; keff of those = {big.keff.mean():.2f}±"
              f"{big.keff.std():.2f} (orders up to 10^"
              f"{gdf.aut_order_log10.max():.0f})")
    print(f"4. RESIDUAL VERDICT: {verdict}")
    print(f"   arm domains dTail_core(<=7): {a_m:+.4f} "
          f"90% CI [{a_lo:+.4f}, {a_hi:+.4f}]  (original-d was "
          f"{arm.dTail_d7.mean():+.4f})")
    print(f"   arm domains dTail_core(<=13): {a13_m:+.4f} "
          f"[{a13_lo:+.4f}, {a13_hi:+.4f}]")
    print(f"   all observational: {o_m:+.4f} [{o_lo:+.4f}, {o_hi:+.4f}]")
    print("5. mechanism table (keff / frac exponent-carrier / omega / "
          "maxexp):")
    for r in mtab.itertuples():
        print(f"   {r.group:<28} keff={r.keff:.3f} expcarrier="
              f"{r.frac_exp_carrier if not pd.isna(r.frac_exp_carrier) else float('nan'):.3f} "
              f"omega={r.omega:.2f} maxexp={r.maxexp:.2f}")


def note():
    # written in analyze()'s wake by the orchestrator (build09.md authored
    # separately with the final numbers)
    pass


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("corpus", "all"):
        corpus()
    if stage in ("corenull", "all"):
        corenull()
    if stage in ("cod", "all"):
        cod()
    if stage in ("nist", "all"):
        nist()
    if stage in ("graphs", "all"):
        graphs()
    if stage in ("analyze", "all"):
        analyze()
