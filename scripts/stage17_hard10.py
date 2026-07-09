"""Build 10 (run hard10): instrument hardening & the integrity channel.

Stages (python stage17_hard10.py <stage>):
  semiprime   — Stage 1: keff blind spot vs full-profile statistics
  spikein     — Stage 2: planted-signal recovery through the full pipeline
  binarygrid  — Stage 3: binary-grid out-of-sample test of the encoding account
  degradation — Stage 4: continuous rounding/noise degradation curve
  integrity   — Stage 5: encoding-drift injection detection vs Benford
  readout     — verdicts vs frozen predictions

Stage 0 (pre-registration) executes at import time, before any computation:
all predictions and decision rules are frozen in config/run_config_hard10.json.
Non-destructive; figures PNG dpi 200; 14 workers; seeded.
"""
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT, ext_of
import build02_lib as lib

DIAG = os.path.join(OUT, "diagnostics")
FROZEN = os.path.join(OUT, "frozen")
RUN = "hard10"
SEED = 20260710
WORKERS = 14
LN10 = math.log(10)
LN2, LN5 = math.log(2), math.log(5)

SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
GRAY = "#898781"

# ============================================================= Stage 0
PREREG = {
    "run": RUN, "seed": SEED, "workers": WORKERS,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage1_semiprime": {
        "design": "balanced (rho=1) semiprimes p*q at product digits d in "
                  "{8,10,12,14,16} and imbalanced (rho=2,4) at d=12; nulls: "
                  "matched log-uniform single draws and uniform-in-stratum "
                  "generic integers; 4000 positives + 4000 per null per cell",
        "statistics": "keff_rec=E[H2|d]/H2 (score as-is); -Tail; -atoms; "
                      "-L3; -H3; group-level (m=200) energy distance of the "
                      "padded sorted profile to a held-out null cloud",
        "predictions": {
            "keff": "AUC ~ 0.5 (0.40-0.60) for balanced semiprimes — blind, "
                    "an identity: H2=0.5 sits on E[H2] under PD(0,1)",
            "full_profile": "at least one of {Tail, atoms, L3, H3, energy} "
                            "reaches AUC >= 0.8 on balanced semiprimes (the "
                            "profile is (~.5,~.5,0,...): two atoms, zero "
                            "tail — visible to statistics keff discards)",
        },
        "consequence": "if confirmed, the paper's 'L-profile is blind to "
                       "prime pairs' is corrected to 'keff is blind; the "
                       "full profile is not' and cryptographic-modulus "
                       "detection is un-ruled-out",
    },
    "stage2_spikein": {
        "host_selection_rule": "channel_profiles_v3 x corpus v2, file still "
            "on disk, channel_kind=='amount', not low_info, n_records_used "
            ">= 50000, n_distinct >= 5000, 0.2 <= rounding_mass <= 0.9, "
            "log10_max >= 5; dedupe by dataset_family keeping highest "
            "n_records_used, then dedupe by lowercase column_name (the "
            "corpus is ~8x duplicated across archive snapshot generations; "
            "family dedupe alone leaves snapshot copies of the same file); "
            "sort rounding_mass desc; take top 6 (criteria frozen before "
            "any Stage-2 statistic is computed; column-name dedupe added "
            "as a pre-computation amendment after the first selection "
            "returned 5 snapshot copies of one CT sale-amount file)",
        "injection": {
            "unrounded": "exact balanced prod2 (two composite factors, "
                         "jitter 0.08) magnitude-matched to host records",
            "postrounded": "same products rounded to the host grid "
                           "(trailing-10 depth resampled from host marginal)"
                           " — gate07 physics bound, EXPECTED destroyed; "
                           "reported, excluded from the powered decision",
            "covarying": "priced-quantity products a*b where the price a is "
                         "rounded to a host-like grid with (magnitude, "
                         "depth) sampled JOINTLY from host records — the "
                         "product is exact and carries 2*5 mass covarying "
                         "with magnitude, the adversarial over-control case",
        },
        "fractions": [0.01, 0.05, 0.10, 0.25],
        "pipeline": "unchanged: canonical integerization, 2*5 de-rounding "
                    "(keff' vs baseline_derounded_v2 by original d), deep "
                    "strip <=7 with core-magnitude re-indexing (keff_core, "
                    "dTail_core vs baseline_coremag cut=7)",
        "decision_rule": "channel flags at f iff keff_core_c7(mixed) > q95 "
            "of the channel's own f=0 record-bootstrap (R=400, n-matched); "
            "'recovered at f' = flagged in > half of host channels; POWERED "
            "iff unrounded recovered at f=0.05 AND covarying recovered at "
            "f=0.10; else the pipeline over-controls and the negative "
            "verdict is softened",
        "secondary": "dTail_core_c7 > q95 and keff' > q95 reported",
    },
    "stage3_binarygrid": {
        "binary_sources": "real file sizes (inventory.csv, dedup sha256), "
            "block-aligned disk allocations (st_blocks*512 over on-disk "
            "corpus files), archive-member uncompressed sizes, telemetry "
            "byte counts (frozen corpus)",
        "decimal_comparators": "the Stage-2 host amount channels; "
            "log-uniform synthetic control",
        "statistics": "c2/c5 shares (mean a_p*ln p/ln n) minus digit-"
            "weighted uniform-null expectation; frac divisible by 8/16 and "
            "25/125 vs null",
        "prediction": "binary-grid sources show elevated c2 (excess > 0, "
            ">= 3x their c5 excess) with c5 near baseline; decimal channels "
            "elevated on BOTH c2 and c5 (5 especially) — encoding-specific "
            "arms, out of sample",
    },
    "stage4_degradation": {
        "base": "balanced prod2 (gate07 g_prod2_ratio(1.0)) and matched "
                "single null, band [1e6,1e7), bin-matched, n=30000 each",
        "perturbations": {
            "grid": [2, 5, 10, 25, 100, 1000],
            "sigfig": [5, 4, 3, 2],
            "additive": "n + U{1..E}, E in {1,3,10,100,1000,10000} "
                        "(E=1 is the exact n -> n+1 case)",
        },
        "score": "AUC of per-record keff vs the SAME-perturbed single null "
                 "(the honest comparison, per gate07)",
        "predictions": {
            "monotone": "AUC non-increasing in perturbation size within "
                        "each family (tolerance 0.02)",
            "n_plus_1": "a single additive unit collapses detection: "
                        "AUC(n+1) <= 0.55 — multiplicative independence of "
                        "neighbors; 'one digit of rounding' strengthens to "
                        "'one additive unit'",
        },
    },
    "stage5_integrity": {
        "hosts": "first 4 channels of the Stage-2 host list (stable "
                 "encoding signature: rounded amounts)",
        "window": "w=2000 test windows vs the channel's reference half; "
                  "encoding coordinates calibrated on 300 clean windows; "
                  "AUC over 120 clean vs 120 injected windows per cell",
        "injections": {
            "real_donor": "real telemetry byte-count records, digit-matched "
                          "to the host (foreign real data, fine grid)",
            "synthetic_finegrid": "log-uniform unrounded integers matched "
                                  "to host magnitudes",
            "benford_matched": "THE SHARP CASE: first digit AND digit "
                               "length matched to the host empirical "
                               "distribution, mantissa uniform (passes any "
                               "leading-digit test by construction), "
                               "off-grid",
        },
        "fractions": [0.02, 0.05, 0.10, 0.20],
        "methods": {
            "encoding": "max |z| over {rounding_mass, mean trailing-10 "
                        "depth, frac on host modal grid, TV(mod 100) drift} "
                        "z-scored on clean-window calibration",
            "benford": "chi2 of window first-digit distribution vs the "
                       "reference half (drift form; vs-law also reported)",
            "lastdigit": "chi2 of window last-digit distribution vs "
                         "reference (vs-uniform also reported)",
        },
        "predictions": {
            "grid_mismatch": "encoding detects grid-mismatched injection "
                             "(AUC >= 0.8 by f=0.10 on real_donor and "
                             "synthetic_finegrid)",
            "sharp_case": "on benford_matched, Benford is at chance "
                          "(AUC 0.4-0.6) while encoding stays >= 0.8 at "
                          "f >= 0.10",
            "redundancy_rule": "if encoding cannot beat or complement "
                               "Benford anywhere, it is redundant and we "
                               "say so",
            "stated_limit": "a fabricator who matches the host grid "
                            "defeats both channels — stated explicitly",
        },
    },
    "hygiene": "matched nulls on magnitude/first moments; Benford and "
               "last-digit implemented as standard baselines; predictions "
               "frozen before computation; labels never enter geometry",
}
os.makedirs(CONFIG, exist_ok=True)
_cfg = os.path.join(CONFIG, "run_config_hard10.json")
if not os.path.exists(_cfg):
    with open(_cfg, "w") as _f:
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


def get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    return plt


FOOT = f"run {RUN} · seed {SEED} · pre-registered before computation"


class Null:
    """Original-digit null (baseline.csv), asymptote-pinned beyond d=18."""
    ASYMPT = {"E_L1": 0.6243299885, "E_L2": 0.1860231051,
              "E_Tail": 1.0 - 0.6243299885 - 0.1860231051, "E_H2": 0.5}

    def __init__(self):
        bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
        bl = bl[bl.stratum != "PD(0,1)"].copy()
        bl["d"] = bl.stratum.astype(int)
        self.tab = bl.set_index("d")
        fit = bl[bl.d >= 10]
        self.coef = {c: float(np.mean((fit[c] - self.ASYMPT[c]) * fit.d))
                     for c in ("E_L1", "E_L2", "E_Tail", "E_H2")}

    def get(self, d, col):
        if d <= 18:
            return float(self.tab.loc[d, col])
        return self.ASYMPT[col] + self.coef[col] / d

    def arr(self, ds, col):
        return np.array([self.get(int(d), col) for d in ds])


def fast_auc(pos, neg):
    """Rank AUC of score arrays (higher score = positive-like)."""
    from scipy.stats import rankdata
    s = np.r_[np.asarray(pos, float), np.asarray(neg, float)]
    ok = np.isfinite(s)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))][ok]
    r = rankdata(s[ok])
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def valuation(v, p):
    """p-adic valuation of an int64 array (vectorized)."""
    v = np.asarray(v, dtype=np.int64)
    a = np.zeros(len(v), dtype=np.int64)
    x = v.copy()
    m = (x > 0) & (x % p == 0)
    while m.any():
        x[m] //= p
        a[m] += 1
        m = (x > 0) & (x % p == 0)
    return a


def digits_of(v):
    return np.floor(np.log10(np.maximum(np.asarray(v, float), 2.0))
                    ).astype(int) + 1


# ---------------------------------------------------------- record profiling
def _prof_chunk(vals):
    """Full-profile per-record stats: L1,L2,L3,Tail,H2,H3,atoms,d, padded
    sorted profile (6 slots)."""
    n = len(vals)
    out = np.zeros((n, 8))
    pads = np.zeros((n, 6))
    for i, v in enumerate(np.asarray(vals).tolist()):
        pairs = lib.factor_pairs(int(v))
        ln_n = math.log(v)
        c = sorted((e * math.log(p) for p, e in pairs), reverse=True)
        L = [x / ln_n for x in c]
        L1 = L[0]
        L2 = L[1] if len(L) > 1 else 0.0
        L3 = L[2] if len(L) > 2 else 0.0
        H2 = sum(x * x for x in L)
        H3 = sum(x ** 3 for x in L)
        out[i] = (L1, L2, L3, 1.0 - L1 - L2, H2, H3, len(L),
                  int(ln_n / LN10) + 1)
        for j, x in enumerate(L[:6]):
            pads[i, j] = x
    return out, pads


def profile_parallel(vals, pool):
    chunks = np.array_split(np.asarray(vals, dtype=np.int64),
                            max(1, WORKERS * 3))
    parts = list(pool.map(_prof_chunk, chunks))
    return (np.vstack([p[0] for p in parts]),
            np.vstack([p[1] for p in parts]))


# ============================================================ Stage 1
def _semiprime_chunk(args):
    d, rho, n, key = args
    from sympy import randprime
    rng = np.random.default_rng([SEED, 1, key])
    rows, pads = [], []
    while len(rows) < n:
        t = rng.uniform(d - 1 + 0.02, d - 0.02)
        t1 = t * rho / (1 + rho) + rng.uniform(-0.04, 0.04)
        t2 = t - t1
        if t2 < 0.35:           # smallest prime magnitude floor
            t2 = 0.35
        try:
            p = randprime(int(10 ** (t1 - 0.02)), int(10 ** (t1 + 0.02)) + 3)
            q = randprime(int(10 ** (t2 - 0.02)), int(10 ** (t2 + 0.02)) + 3)
        except ValueError:
            continue
        if p is None or q is None:
            continue
        v = int(p) * int(q)
        ln_n = math.log(v)
        if p == q:
            L = [1.0]
        else:
            c = sorted([math.log(p), math.log(q)], reverse=True)
            L = [x / ln_n for x in c]
        L1 = L[0]
        L2 = L[1] if len(L) > 1 else 0.0
        H2 = sum(x * x for x in L)
        H3 = sum(x ** 3 for x in L)
        rows.append((L1, L2, 0.0, 1.0 - L1 - L2, H2, H3, len(L),
                     int(ln_n / LN10) + 1))
        pads.append(L[:6] + [0.0] * (6 - len(L)))
    return np.array(rows), np.array(pads)


def energy_distance(X, Y):
    from scipy.spatial.distance import cdist
    dxy = cdist(X, Y).mean()
    dxx = cdist(X, X).mean()
    dyy = cdist(Y, Y).mean()
    return 2 * dxy - dxx - dyy


def semiprime():
    lib.init_worker()
    null = Null()
    N_POS, N_NULL, M_GRP, N_GRP = 4000, 4000, 200, 60
    cells = [(8, 1.0), (10, 1.0), (12, 1.0), (14, 1.0), (16, 1.0),
             (12, 2.0), (12, 4.0)]
    pool = ProcessPoolExecutor(max_workers=WORKERS,
                               initializer=lib.init_worker)
    rows = []
    for ci, (d, rho) in enumerate(cells):
        # positives (construction-known factorization)
        tasks = [(d, rho, N_POS // WORKERS + 1, ci * 100 + w)
                 for w in range(WORKERS)]
        parts = list(pool.map(_semiprime_chunk, tasks))
        P = np.vstack([p[0] for p in parts])[:N_POS]
        Ppad = np.vstack([p[1] for p in parts])[:N_POS]
        # nulls
        rng = np.random.default_rng([SEED, 2, ci])
        nl = np.floor(10 ** rng.uniform(d - 1, d, N_NULL)).astype(np.int64)
        nu = rng.integers(10 ** (d - 1), 10 ** d, N_NULL, dtype=np.int64)
        NL, NLpad = profile_parallel(np.maximum(nl, 2), pool)
        NU, NUpad = profile_parallel(np.maximum(nu, 2), pool)
        stats = {}
        for null_name, Ng, Ngpad in (("loguniform", NL, NLpad),
                                     ("uniform", NU, NUpad)):
            keff_p = null.arr(P[:, 7], "E_H2") / P[:, 4]
            keff_n = null.arr(Ng[:, 7], "E_H2") / Ng[:, 4]
            aucs = {
                "keff": fast_auc(keff_p, keff_n),
                "tail": fast_auc(-P[:, 3], -Ng[:, 3]),
                "atoms": fast_auc(-P[:, 6], -Ng[:, 6]),
                "L3": fast_auc(-P[:, 2], -Ng[:, 2]),
                "H3": fast_auc(-P[:, 5], -Ng[:, 5]),
            }
            # group-level energy distance to a held-out null cloud
            rng2 = np.random.default_rng([SEED, 3, ci])
            half = N_NULL // 2
            ref = Ngpad[:half]
            hold = Ngpad[half:]
            e_pos = [energy_distance(
                Ppad[rng2.integers(0, len(Ppad), M_GRP)], ref)
                for _ in range(N_GRP)]
            e_neg = [energy_distance(
                hold[rng2.integers(0, len(hold), M_GRP)], ref)
                for _ in range(N_GRP)]
            aucs["energy"] = fast_auc(e_pos, e_neg)
            stats[null_name] = aucs
            for stat, auc in aucs.items():
                rows.append(dict(
                    d=d, rho=rho, null=null_name, statistic=stat, auc=auc,
                    n_pos=len(P), n_null=len(Ng),
                    pos_mean_keff=float(keff_p.mean()),
                    pos_mean_tail=float(P[:, 3].mean()),
                    pos_mean_atoms=float(P[:, 6].mean()),
                    null_mean_atoms=float(Ng[:, 6].mean())))
        print(f"  cell d={d} rho={rho}: " + " ".join(
            f"{k}={v:.3f}" for k, v in stats["loguniform"].items()),
            flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(TABLES, "hard10_semiprime.csv"),
                       index=False, encoding="utf-8")
    pool.shutdown()

    # -------- figure: AUC by statistic
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    stats_order = ["keff", "tail", "atoms", "L3", "H3", "energy"]
    cmap = {s: PALETTE[i] for i, s in enumerate(stats_order)}
    bal = df[(df.rho == 1.0) & (df.null == "loguniform")]
    ax = axes[0]
    style_ax(ax)
    for s in stats_order:
        sub = bal[bal.statistic == s].sort_values("d")
        ax.plot(sub.d, sub.auc, marker="o", markersize=4.5, linewidth=1.8,
                color=cmap[s], markeredgecolor=SURFACE, label=s)
    ax.axhline(0.5, color=BASELINE_C, linewidth=1.0)
    ax.axhline(0.8, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("product digits d (balanced ρ=1)", fontsize=9,
                  color=MUTED)
    ax.set_ylabel("AUC vs matched log-uniform null", fontsize=9,
                  color=MUTED)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right",
              labelcolor=INK)
    ax.set_title("Balanced semiprimes: keff blind, profile not",
                 fontsize=10, color=INK, loc="left", pad=10)
    ax = axes[1]
    style_ax(ax)
    d12 = df[(df.d == 12) & (df.null == "loguniform")]
    width = 0.26
    for j, rho in enumerate([1.0, 2.0, 4.0]):
        sub = d12[d12.rho == rho].set_index("statistic").reindex(
            stats_order)
        ax.bar(np.arange(len(stats_order)) + (j - 1) * width, sub.auc,
               width * 0.92, color=PALETTE[j], label=f"ρ={rho:.0f}",
               edgecolor=SURFACE, linewidth=0.5)
    ax.axhline(0.5, color=BASELINE_C, linewidth=1.0)
    ax.axhline(0.8, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(stats_order)))
    ax.set_xticklabels(stats_order, fontsize=8)
    ax.set_ylabel("AUC (d=12)", fontsize=9, color=MUTED)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)
    ax.set_title("Imbalance: full-profile statistics persist", fontsize=10,
                 color=INK, loc="left", pad=10)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig10_semiprime.png"),
                facecolor=SURFACE)
    print("stage 1 done -> hard10_semiprime.csv, fig10_semiprime.png",
          flush=True)


# ============================================================ Stage 2
FRACTIONS = [0.01, 0.05, 0.10, 0.25]
CAP_N = 30_000
BOOT_R = 400


def select_hosts():
    """Frozen selection rule (see PREREG stage2)."""
    prof = pd.read_csv(os.path.join(TABLES, "channel_profiles_v3.csv"),
                       low_memory=False)
    dd = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                     low_memory=False)
    dd["on_disk"] = dd.file_path.map(os.path.exists)
    m = prof.merge(dd[["dataset_id", "on_disk", "file_path",
                       "archive_member", "sheet_or_table", "looks_monetary",
                       "channel_kind"]].rename(
                           columns={"channel_kind": "kind_c"}),
                   on="dataset_id", how="left")
    cand = m[(m.on_disk == True) & (m.channel_kind == "amount") &  # noqa
             (~m.low_info.astype(bool)) & (m.n_records_used >= 50_000) &
             (m.n_distinct >= 5_000) & (m.rounding_mass.between(0.2, 0.9)) &
             (m.log10_max >= 5)]
    cand = cand.sort_values("n_records_used", ascending=False) \
               .drop_duplicates("dataset_family")
    cand["_col"] = cand.column_name.str.lower().str.strip()
    cand = cand.sort_values("rounding_mass", ascending=False) \
               .drop_duplicates("_col").head(6)
    return cand


def load_channel(row):
    from stage9_build03 import _read_with_fallback
    fp = row.file_path
    member = row.archive_member if isinstance(row.archive_member, str) \
        else ""
    sheet = row.sheet_or_table if isinstance(row.sheet_or_table, str) \
        else ""
    seed = int(hashlib.md5(f"{fp}|{member}".encode()).hexdigest()[:8], 16)
    df, _ = _read_with_fallback(fp, member, sheet,
                                ext_of(member or fp), seed,
                                [row.column_name])
    series = df[row.column_name]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    monetary = bool(row.looks_monetary) or row.channel_kind == "amount"
    ints, _ = lib.coerce_ints(series, monetary)
    rng = np.random.default_rng([SEED, seed])
    if len(ints) > CAP_N:
        ints = ints[rng.choice(len(ints), CAP_N, replace=False)]
    return np.asarray(ints, dtype=np.int64)


def _pipe_chunk(vals):
    """Pipeline per-value stats: for each value ->
    (d_orig, H2p 2*5-core (nan if core==1), core7_d, Tail7, H2_7
     (nan if 7-smooth))."""
    out = np.full((len(vals), 5), np.nan)
    for i, v in enumerate(np.asarray(vals).tolist()):
        v = int(v)
        pairs = lib.factor_pairs(v)
        d = int(math.log(v) / LN10) + 1
        out[i, 0] = d
        core25 = [(p, e) for p, e in pairs if p not in (2, 5)]
        if core25:
            ln_c = sum(e * math.log(p) for p, e in core25)
            out[i, 1] = sum((e * math.log(p) / ln_c) ** 2
                            for p, e in core25)
        core7 = [(p, e) for p, e in pairs if p > 7]
        if core7:
            ln_c = sum(e * math.log(p) for p, e in core7)
            L = sorted((e * math.log(p) / ln_c for p, e in core7),
                       reverse=True)
            L1 = L[0]
            L2 = L[1] if len(L) > 1 else 0.0
            out[i, 2] = int(ln_c / LN10) + 1
            out[i, 3] = 1.0 - L1 - L2
            out[i, 4] = sum(x * x for x in L)
    return out


def pipeline_stats(vals, pool):
    """Per-record pipeline arrays via unique-value factorization."""
    uniq, inv = np.unique(vals, return_inverse=True)
    chunks = np.array_split(uniq, max(1, WORKERS * 2))
    parts = list(pool.map(_pipe_chunk, chunks))
    per_uniq = np.vstack(parts)
    return per_uniq[inv]


class PipeNulls:
    def __init__(self):
        der = pd.read_csv(os.path.join(TABLES,
                                       "baseline_derounded_v2.csv"))
        der = der[der.stratum.astype(str).str.isdigit()].copy()
        der["d"] = der.stratum.astype(int)
        self.der = der.set_index("d")
        cm = pd.read_csv(os.path.join(TABLES, "baseline_coremag.csv"))
        self.cm = cm[cm.cut == 7].set_index("core_d")

    def channel_stats(self, rec):
        """rec: per-record array (d_orig, H2p, core7_d, Tail7, H2_7).
        Returns keff_p, keff_c7, dTail_c7 (channel-level)."""
        out = {}
        m = np.isfinite(rec[:, 1])
        if m.sum() >= 50:
            ds = np.clip(rec[m, 0].astype(int), 1, 18)
            eh = self.der.E_H2_der.reindex(np.arange(1, 19)).to_numpy()
            out["keff_p"] = float(eh[ds - 1].mean() / rec[m, 1].mean())
        else:
            out["keff_p"] = np.nan
        m = np.isfinite(rec[:, 4])
        if m.sum() >= 50:
            cds = np.clip(rec[m, 2].astype(int), 2, 18)
            idx = np.arange(2, 19)
            eh = self.cm.E_H2.reindex(idx).ffill().to_numpy()
            et = self.cm.E_Tail.reindex(idx).ffill().to_numpy()
            out["keff_c7"] = float(eh[cds - 2].mean() / rec[m, 4].mean())
            out["dTail_c7"] = float(rec[m, 3].mean() - et[cds - 2].mean())
        else:
            out["keff_c7"] = np.nan
            out["dTail_c7"] = np.nan
        return out


def gen_products(host_vals, n, rng, variant):
    """Composite-factor products magnitude-matched to host records."""
    idx = rng.integers(0, len(host_vals), n)
    hv = host_vals[idx].astype(float)
    t = np.log10(hv)
    z_all = valuation(host_vals, 10)
    if variant == "covarying":
        z = z_all[idx]              # joint (magnitude, depth)
    else:
        z = z_all[rng.integers(0, len(host_vals), n)]   # marginal
    u = rng.uniform(-0.08, 0.08, n)
    if variant == "covarying":
        # priced quantity: price a on a host-like grid, quantity b generic
        ta = t * rng.uniform(0.35, 0.65, n)
        a = np.maximum(np.floor(10 ** (ta + u)), 2).astype(np.int64)
        za = np.minimum(z, np.maximum(digits_of(a) - 2, 0))
        g = 10 ** za.astype(np.int64)
        a = np.maximum(np.round(a / g).astype(np.int64) * g, g)
        tb = t - np.log10(np.maximum(a, 2))
        b = np.maximum(np.floor(10 ** np.maximum(tb, 0.31)), 2
                       ).astype(np.int64)
        return a * b
    a = np.maximum(np.floor(10 ** (t / 2 + u)), 2).astype(np.int64)
    b = np.maximum(np.floor(10 ** (t / 2 - u)), 2).astype(np.int64)
    v = a * b
    if variant == "postrounded":
        g = 10 ** np.minimum(z, np.maximum(digits_of(v) - 2, 0)
                             ).astype(np.int64)
        v = np.maximum(np.round(v / g).astype(np.int64) * g, g)
    return np.maximum(v, 2)


def spikein():
    lib.init_worker()
    hosts = select_hosts()
    hosts.to_csv(os.path.join(DIAG, "hard10_hosts.csv"), index=False)
    print("hosts:\n" + hosts[["dataset_id", "domain", "column_name",
                              "rounding_mass",
                              "n_records_used"]].to_string(), flush=True)
    nulls = PipeNulls()
    pool = ProcessPoolExecutor(max_workers=WORKERS,
                               initializer=lib.init_worker)
    rows = []
    for hi, row in enumerate(hosts.itertuples()):
        try:
            vals = load_channel(row)
        except Exception as e:
            print(f"  LOAD FAIL {row.dataset_id}: {e!r}", flush=True)
            continue
        if len(vals) < 10_000:
            print(f"  skip {row.dataset_id}: n={len(vals)}", flush=True)
            continue
        rec0 = pipeline_stats(vals, pool)
        base = nulls.channel_stats(rec0)
        # f=0 bootstrap thresholds (record bootstrap, n-matched)
        rngb = np.random.default_rng([SEED, 4, hi])
        boots = {"keff_p": [], "keff_c7": [], "dTail_c7": []}
        for _ in range(BOOT_R):
            bs = nulls.channel_stats(
                rec0[rngb.integers(0, len(rec0), len(rec0))])
            for k in boots:
                boots[k].append(bs[k])
        q95 = {k: float(np.nanpercentile(v, 95)) for k, v in boots.items()}
        rows.append(dict(dataset_id=row.dataset_id, domain=row.domain,
                         variant="none", f=0.0, n=len(vals), **base,
                         **{f"q95_{k}": v for k, v in q95.items()},
                         det_keff_c7=False, det_keff_p=False,
                         det_dTail_c7=False))
        print(f"  {row.dataset_id[:50]} f=0: keff_p={base['keff_p']:.3f} "
              f"keff_c7={base['keff_c7']:.3f} q95={q95['keff_c7']:.3f}",
              flush=True)
        for variant in ("unrounded", "postrounded", "covarying"):
            for f in FRACTIONS:
                rngi = np.random.default_rng(
                    [SEED, 5, hi, int(f * 100),
                     ("unrounded", "postrounded",
                      "covarying").index(variant)])
                k = int(round(f * len(vals)))
                inj = gen_products(vals, k, rngi, variant)
                mixed = vals.copy()
                pos = rngi.choice(len(vals), k, replace=False)
                mixed[pos] = inj
                rec = pipeline_stats(mixed, pool)
                st = nulls.channel_stats(rec)
                rows.append(dict(
                    dataset_id=row.dataset_id, domain=row.domain,
                    variant=variant, f=f, n=len(vals), **st,
                    **{f"q95_{k2}": v for k2, v in q95.items()},
                    det_keff_c7=bool(st["keff_c7"] > q95["keff_c7"]),
                    det_keff_p=bool(st["keff_p"] > q95["keff_p"]),
                    det_dTail_c7=bool(st["dTail_c7"] > q95["dTail_c7"])))
            got = [r for r in rows if r["dataset_id"] == row.dataset_id
                   and r["variant"] == variant]
            print(f"    {variant}: keff_c7 " + " ".join(
                f"f={r['f']:.2f}:{r['keff_c7']:.3f}"
                f"{'*' if r['det_keff_c7'] else ''}" for r in got),
                flush=True)
    pool.shutdown()
    df = pd.DataFrame(rows)
    df.round(5).to_csv(os.path.join(TABLES, "hard10_spikein.csv"),
                       index=False, encoding="utf-8")

    # decision
    rec = {}
    for variant in ("unrounded", "postrounded", "covarying"):
        rec[variant] = {}
        for f in FRACTIONS:
            sub = df[(df.variant == variant) & (df.f == f)]
            rec[variant][f] = float(sub.det_keff_c7.mean()) if len(sub) \
                else np.nan
    powered = (rec["unrounded"].get(0.05, 0) > 0.5 and
               rec["covarying"].get(0.10, 0) > 0.5)
    print(f"RECOVERY (frac channels flagged, keff_c7 rule): "
          f"{json.dumps(rec, default=float)}")
    print(f"VERDICT: {'POWERED' if powered else 'OVER-CONTROLS'}",
          flush=True)

    # -------- figure
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    vari_c = {"unrounded": PALETTE[0], "covarying": PALETTE[2],
              "postrounded": GRAY}
    ax = axes[0]
    style_ax(ax)
    for variant, c in vari_c.items():
        ys = [rec[variant][f] for f in FRACTIONS]
        ax.plot(FRACTIONS, ys, marker="o", markersize=5, linewidth=1.8,
                color=c, markeredgecolor=SURFACE, label=variant)
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("injection fraction f", fontsize=9, color=MUTED)
    ax.set_ylabel("fraction of host channels flagged (keff_core rule)",
                  fontsize=9, color=MUTED)
    ax.set_ylim(-0.04, 1.04)
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title(f"Spike-in recovery — verdict: "
                 f"{'POWERED' if powered else 'OVER-CONTROLS'}",
                 fontsize=10, color=INK, loc="left", pad=10)
    ax = axes[1]
    style_ax(ax)
    for variant, c in vari_c.items():
        sub = df[df.variant == variant].groupby("f").keff_c7.mean()
        base_m = df[df.variant == "none"].keff_c7.mean()
        ax.plot([0] + list(sub.index), [base_m] + list(sub.values),
                marker="o", markersize=5, linewidth=1.8, color=c,
                markeredgecolor=SURFACE, label=variant)
    ax.axhline(df[df.variant == "none"].q95_keff_c7.mean(), color=MUTED,
               linewidth=0.8, linestyle="--")
    ax.set_xlabel("injection fraction f", fontsize=9, color=MUTED)
    ax.set_ylabel("mean core keff (cut ≤7, core-digit null)", fontsize=9,
                  color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("keff elevation vs f (mean q95 threshold dashed)",
                 fontsize=10, color=INK, loc="left", pad=10)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig10_spikein.png"),
                facecolor=SURFACE)
    print("stage 2 done -> hard10_spikein.csv, fig10_spikein.png",
          flush=True)


# ============================================================ Stage 3
def _c25_null_chunk(args):
    d, n, key = args
    rng = np.random.default_rng([SEED, 6, d, key])
    v = rng.integers(max(2, 10 ** (d - 1)), 10 ** d, n, dtype=np.int64)
    a2 = valuation(v, 2)
    a5 = valuation(v, 5)
    ln = np.log(v.astype(float))
    return d, float((a2 * LN2 / ln).mean()), float((a5 * LN5 / ln).mean()), \
        float((v % 8 == 0).mean()), float((v % 25 == 0).mean())


def c25_stats(v):
    v = np.asarray(v, dtype=np.int64)
    v = v[v > 1]
    a2 = valuation(v, 2)
    a5 = valuation(v, 5)
    ln = np.log(v.astype(float))
    return dict(n=len(v),
                c2_share=float((a2 * LN2 / ln).mean()),
                c5_share=float((a5 * LN5 / ln).mean()),
                frac_div8=float((v % 8 == 0).mean()),
                frac_div25=float((v % 25 == 0).mean()),
                dhist=np.bincount(np.clip(digits_of(v), 1, 18) - 1,
                                  minlength=18))


def binarygrid():
    lib.init_worker()
    pool = ProcessPoolExecutor(max_workers=WORKERS)
    # null expectations per digit stratum
    tasks = [(d, 100_000, k) for d in range(1, 19) for k in range(2)]
    acc = {}
    for d, c2, c5, f8, f25 in pool.map(_c25_null_chunk, tasks):
        acc.setdefault(d, []).append((c2, c5, f8, f25))
    nul = {d: np.mean(acc[d], axis=0) for d in acc}

    def excess(st):
        w = st["dhist"] / st["dhist"].sum()
        e = np.array([nul.get(d + 1, nul[18]) for d in range(18)])
        base = w @ e
        return dict(c2_null=float(base[0]), c5_null=float(base[1]),
                    c2_excess=st["c2_share"] - float(base[0]),
                    c5_excess=st["c5_share"] - float(base[1]),
                    div8_excess=st["frac_div8"] - float(base[2]),
                    div25_excess=st["frac_div25"] - float(base[3]))

    rows = []
    inv = pd.read_csv(os.path.join(TABLES, "inventory.csv"),
                      usecols=["row_type", "file_path", "size_bytes",
                               "member_uncompressed_size", "sha256"],
                      low_memory=False)
    files = inv[inv.row_type == "file"].drop_duplicates("sha256")
    fs = files.size_bytes.dropna().astype(np.int64)
    st = c25_stats(fs[fs > 1].to_numpy())
    rows.append(dict(source="file_sizes_bytes", cls="binary", **{
        k: v for k, v in st.items() if k != "dhist"}, **excess(st)))

    # block-aligned allocations of on-disk files
    rng = np.random.default_rng([SEED, 7])
    paths = files.file_path.dropna().unique()
    paths = paths[rng.permutation(len(paths))][:60_000]
    alloc = []
    for p in paths:
        try:
            s = os.stat(p)
            alloc.append(s.st_blocks * 512)
        except OSError:
            pass
    st = c25_stats(np.array(alloc, dtype=np.int64))
    rows.append(dict(source="disk_alloc_blocks512", cls="binary", **{
        k: v for k, v in st.items() if k != "dhist"}, **excess(st)))

    mem = inv.member_uncompressed_size.dropna().astype(np.int64)
    mem = mem[mem > 1].to_numpy()
    if len(mem) > 200_000:
        mem = mem[rng.choice(len(mem), 200_000, replace=False)]
    st = c25_stats(mem)
    rows.append(dict(source="archive_member_sizes", cls="binary", **{
        k: v for k, v in st.items() if k != "dhist"}, **excess(st)))

    # telemetry byte counts (real corpus channels)
    dd = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                     low_memory=False)
    dd["on_disk"] = dd.file_path.map(os.path.exists)
    tel = dd[(dd.on_disk) & (dd.column_name == "n_bytes")].head(25)
    tvals = []
    for r in tel.itertuples():
        try:
            tvals.append(load_channel(r))
        except Exception:
            pass
    if tvals:
        st = c25_stats(np.concatenate(tvals))
        rows.append(dict(source="telemetry_n_bytes", cls="telemetry", **{
            k: v for k, v in st.items() if k != "dhist"}, **excess(st)))

    # decimal comparators: stage-2 hosts
    hosts = select_hosts()
    for r in hosts.itertuples():
        try:
            v = load_channel(r)
        except Exception:
            continue
        st = c25_stats(v)
        rows.append(dict(source=f"decimal:{r.dataset_id[:44]}",
                         cls="decimal", **{k: v2 for k, v2 in st.items()
                                           if k != "dhist"}, **excess(st)))

    # synthetic log-uniform control
    v = np.floor(10 ** rng.uniform(2, 9, 200_000)).astype(np.int64)
    st = c25_stats(np.maximum(v, 2))
    rows.append(dict(source="loguniform_control", cls="control", **{
        k: v2 for k, v2 in st.items() if k != "dhist"}, **excess(st)))
    pool.shutdown()

    df = pd.DataFrame(rows)
    df.round(5).to_csv(os.path.join(TABLES, "hard10_binarygrid.csv"),
                       index=False, encoding="utf-8")
    print(df[["source", "cls", "n", "c2_excess", "c5_excess",
              "div8_excess", "div25_excess"]].to_string(), flush=True)
    fig_binarygrid(df)
    print("stage 3 done -> hard10_binarygrid.csv, fig10_binarygrid.png",
          flush=True)


def fig_binarygrid(df):
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(0, color=BASELINE_C, linewidth=1.0)
    ax.axvline(0, color=BASELINE_C, linewidth=1.0)
    cls_c = {"binary": PALETTE[0], "decimal": PALETTE[2],
             "telemetry": PALETTE[1], "control": GRAY}
    offsets = [(6, 6), (6, -12), (-6, 12), (6, 20), (-40, -14), (6, -22)]
    oi = 0
    for cls, c in cls_c.items():
        sub = df[df.cls == cls]
        ax.scatter(sub.c2_excess, sub.c5_excess, s=55, color=c,
                   edgecolors=SURFACE, linewidths=0.7, zorder=3,
                   label=cls)
        for r in sub.itertuples():
            if cls in ("binary", "control", "telemetry"):
                dx, dy = offsets[oi % len(offsets)]
                oi += 1
                ax.annotate(r.source.replace("decimal:", ""),
                            (r.c2_excess, r.c5_excess), fontsize=6.5,
                            color=MUTED, xytext=(dx, dy),
                            textcoords="offset points")
    ax.set_xlabel("c2 excess (share of ln n on prime 2, minus "
                  "magnitude-matched null)", fontsize=9, color=MUTED)
    ax.set_ylabel("c5 excess", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("Binary grids resonate at 2 without 5; decimal grids at "
                 "both", fontsize=10.5, color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig10_binarygrid.png"),
                facecolor=SURFACE)


# ============================================================ Stage 4
BINS4 = np.linspace(6.0, 7.0, 51)
LO4, HI4 = 10 ** 6, 10 ** 7


def bin_match(pool_fn, n, rng, max_rounds=12):
    quota = n // (len(BINS4) - 1)
    buckets = [[] for _ in range(len(BINS4) - 1)]
    for _ in range(max_rounds):
        need = sum(max(0, quota - len(b)) for b in buckets)
        if need == 0:
            break
        vals = pool_fn(rng, max(200_000, 4 * need))
        vals = vals[(vals >= LO4) & (vals < HI4)]
        idx = np.digitize(np.log10(vals), BINS4) - 1
        for v, i in zip(vals.tolist(), idx.tolist()):
            if 0 <= i < len(buckets) and len(buckets[i]) < quota:
                buckets[i].append(v)
    out = np.array([v for b in buckets for v in b], dtype=np.int64)
    rng.shuffle(out)
    return out


def g_single4(rng, m):
    return np.floor(10 ** rng.uniform(6, 7, m)).astype(np.int64)


def g_prod2_bal(rng, m, jitter=0.08):
    t = rng.uniform(6, 7, m)
    u = rng.uniform(-jitter, jitter, m)
    a = np.maximum(np.floor(10 ** (t / 2 + u)), 2)
    b = np.maximum(np.floor(10 ** (t / 2 - u)), 2)
    return (a * b).astype(np.int64)


def _h2_chunk(vals):
    H = np.empty(len(vals))
    D = np.empty(len(vals), dtype=int)
    for i, v in enumerate(np.asarray(vals).tolist()):
        pairs = lib.factor_pairs(int(v))
        ln_n = math.log(v)
        H[i] = sum((e * math.log(p) / ln_n) ** 2 for p, e in pairs)
        D[i] = int(ln_n / LN10) + 1
    return H, D


def h2_parallel(vals, pool):
    chunks = np.array_split(np.asarray(vals, dtype=np.int64),
                            max(1, WORKERS * 3))
    parts = list(pool.map(_h2_chunk, chunks))
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


def degradation():
    lib.init_worker()
    null = Null()
    N4 = 30_000
    rng = np.random.default_rng([SEED, 8])
    prod = bin_match(g_prod2_bal, N4, rng)
    single = bin_match(g_single4, N4, rng)
    pool = ProcessPoolExecutor(max_workers=WORKERS,
                               initializer=lib.init_worker)

    def score(vals):
        H, D = h2_parallel(vals, pool)
        keff_rec = null.arr(D, "E_H2") / H
        keff_ch = float(null.arr(D, "E_H2").mean() / H.mean())
        return keff_rec, keff_ch

    perturbs = [("none", "0", lambda v, r: v)]
    for g in (2, 5, 10, 25, 100, 1000):
        perturbs.append(("grid", str(g), lambda v, r, g=g: np.maximum(
            np.round(v / g).astype(np.int64) * g, 2)))
    for s in (5, 4, 3, 2):
        def sig(v, r, s=s):
            mag = 10 ** np.maximum(digits_of(v) - s, 0).astype(np.int64)
            return np.maximum(np.round(v / mag).astype(np.int64) * mag, 2)
        perturbs.append(("sigfig", str(s), sig))
    for E in (1, 3, 10, 100, 1000, 10000):
        perturbs.append(("additive", str(E), lambda v, r, E=E:
                         v + r.integers(1, E + 1, len(v))))
    rows = []
    for fam, param, fn in perturbs:
        r1 = np.random.default_rng([SEED, 9, len(rows)])
        r2 = np.random.default_rng([SEED, 10, len(rows)])
        pv = fn(prod, r1)
        sv = fn(single, r2)
        kp, kcp = score(pv)
        kn, kcn = score(sv)
        auc = fast_auc(kp, kn)
        rows.append(dict(family=fam, param=param, keff_prod=kcp,
                         keff_null=kcn, auc_same_perturbed=auc))
        print(f"  {fam:<9}{param:>6}: keff(prod)={kcp:.3f} "
              f"keff(null)={kcn:.3f} AUC={auc:.3f}", flush=True)
    pool.shutdown()
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(TABLES, "hard10_degradation.csv"),
                       index=False, encoding="utf-8")

    plt = get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4), dpi=200,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    base_auc = float(df[df.family == "none"].auc_same_perturbed.iloc[0])
    specs = [("grid", "round to nearest g", PALETTE[0]),
             ("sigfig", "significant figures kept", PALETTE[2]),
             ("additive", "additive noise bound E  (E=1 is n→n+1)",
              PALETTE[5])]
    for ax, (fam, xlab, c) in zip(axes, specs):
        style_ax(ax)
        sub = df[df.family == fam]
        x = sub.param.astype(int)
        ax.plot(x, sub.auc_same_perturbed, marker="o", markersize=5,
                linewidth=1.8, color=c, markeredgecolor=SURFACE)
        ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
        ax.axhline(base_auc, color=BASELINE_C, linewidth=0.8)
        ax.set_xscale("log")
        if fam == "sigfig":
            ax.set_xscale("linear")
            ax.invert_xaxis()
        ax.set_xlabel(xlab, fontsize=8.5, color=MUTED)
        ax.set_title(fam, fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("AUC vs same-perturbed null (keff score)",
                       fontsize=9, color=MUTED)
    axes[0].set_ylim(0.44, 0.95)
    fig.suptitle("Degradation of product detection — balanced prod2, "
                 "[1e6,1e7)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(os.path.join(DIAG, "fig10_degradation.png"),
                facecolor=SURFACE)
    print("stage 4 done -> hard10_degradation.csv, fig10_degradation.png",
          flush=True)


# ============================================================ Stage 5
W5 = 2000
N_CAL, N_TEST = 300, 120
F5 = [0.02, 0.05, 0.10, 0.20]


def enc_coords(v, host_grid):
    v = np.asarray(v, dtype=np.int64)
    ln = np.log(np.maximum(v, 2).astype(float))
    a2 = valuation(v, 2)
    a5 = valuation(v, 5)
    z10 = np.minimum(a2, a5)
    tv100 = np.bincount((v % 100).astype(int), minlength=100) / len(v)
    return dict(rounding_mass=float(((a2 * LN2 + a5 * LN5) / ln).mean()),
                trail10=float(z10.mean()),
                grid_conform=float((v % host_grid == 0).mean()),
                mod100=tv100)


def first_digit(v):
    v = np.asarray(v, dtype=np.int64)
    return (v // 10 ** (digits_of(v) - 1)).astype(int)


def chi2_drift(counts_w, dist_ref):
    exp = dist_ref * counts_w.sum()
    exp = np.maximum(exp, 0.5)
    return float(((counts_w - exp) ** 2 / exp).sum())


def gen_foreign(kind, host, n, rng, donor=None):
    if kind == "real_donor":
        hd = digits_of(host)
        dd_ = digits_of(donor)
        by_d = {d: donor[dd_ == d] for d in np.unique(dd_)}
        want = hd[rng.integers(0, len(hd), n)]
        out = np.empty(n, dtype=np.int64)
        for i, d in enumerate(want.tolist()):
            pool_ = by_d.get(d)
            if pool_ is None or len(pool_) == 0:
                near = min(by_d, key=lambda x: abs(x - d))
                pool_ = by_d[near]
            out[i] = pool_[rng.integers(0, len(pool_))]
        return out
    if kind == "synthetic_finegrid":
        t = np.log10(host[rng.integers(0, len(host), n)].astype(float))
        t = t + rng.uniform(-0.05, 0.05, n)
        return np.maximum(np.floor(10 ** t), 2).astype(np.int64)
    if kind == "benford_matched":
        hd = digits_of(host)
        fd = first_digit(host)
        idx = rng.integers(0, len(host), n)
        d, D = hd[idx], fd[idx]
        lo = D * 10 ** (d - 1)
        hi = (D + 1) * 10 ** (d - 1)
        return (lo + (rng.random(n) * (hi - lo)).astype(np.int64)
                ).astype(np.int64)
    raise ValueError(kind)


def integrity():
    lib.init_worker()
    hosts = select_hosts().head(4)
    # donor: pooled telemetry byte counts
    dd = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                     low_memory=False)
    dd["on_disk"] = dd.file_path.map(os.path.exists)
    tel = dd[(dd.on_disk) & (dd.column_name == "n_bytes")].head(20)
    donor = []
    for r in tel.itertuples():
        try:
            donor.append(load_channel(r))
        except Exception:
            pass
    donor = np.concatenate(donor)
    rows = []
    for hi, hrow in enumerate(hosts.itertuples()):
        try:
            vals = load_channel(hrow)
        except Exception as e:
            print(f"  LOAD FAIL {hrow.dataset_id}: {e!r}", flush=True)
            continue
        rng = np.random.default_rng([SEED, 11, hi])
        perm = rng.permutation(len(vals))
        ref = vals[perm[:len(vals) // 2]]
        test_pool = vals[perm[len(vals) // 2:]]
        z_ref = np.minimum(valuation(ref, 2), valuation(ref, 5))
        zz = z_ref[z_ref > 0]
        host_grid = 10 ** max(1, int(np.median(zz)) if len(zz) else 1)
        ref_enc = enc_coords(ref, host_grid)
        fd_ref = np.bincount(first_digit(ref), minlength=10)[1:10]
        fd_ref = fd_ref / fd_ref.sum()
        ld_ref = np.bincount((ref % 10).astype(int), minlength=10)
        ld_ref = ld_ref / ld_ref.sum()
        benford_law = np.log10(1 + 1 / np.arange(1, 10))
        keys = ["rounding_mass", "trail10", "grid_conform", "tv100"]

        def window_scores(w):
            e = enc_coords(w, host_grid)
            tv = 0.5 * np.abs(e["mod100"] - ref_enc["mod100"]).sum()
            enc = np.array([e["rounding_mass"], e["trail10"],
                            e["grid_conform"], tv])
            fd = np.bincount(first_digit(w), minlength=10)[1:10].astype(
                float)
            ld = np.bincount((w % 10).astype(int), minlength=10).astype(
                float)
            return (enc, chi2_drift(fd, fd_ref),
                    chi2_drift(fd, benford_law),
                    chi2_drift(ld, ld_ref),
                    chi2_drift(ld, np.full(10, 0.1)))

        # calibration on clean windows
        cal = np.array([window_scores(
            test_pool[rng.integers(0, len(test_pool), W5)])[0]
            for _ in range(N_CAL)])
        mu, sd = cal.mean(axis=0), np.maximum(cal.std(axis=0), 1e-9)

        def all_scores(w):
            enc, b_dr, b_law, l_dr, l_un = window_scores(w)
            return dict(encoding=float(np.abs((enc - mu) / sd).max()),
                        benford=b_dr, benford_law=b_law,
                        lastdigit=l_dr, lastdigit_uniform=l_un)

        clean = [all_scores(test_pool[rng.integers(0, len(test_pool), W5)])
                 for _ in range(N_TEST)]
        for kind in ("real_donor", "synthetic_finegrid", "benford_matched"):
            for f in F5:
                k = int(round(f * W5))
                inj_scores = []
                for _ in range(N_TEST):
                    w = test_pool[rng.integers(0, len(test_pool), W5)]
                    w = w.copy()
                    w[:k] = gen_foreign(kind, ref, k, rng, donor=donor)
                    inj_scores.append(all_scores(w))
                for method in ("encoding", "benford", "benford_law",
                               "lastdigit", "lastdigit_uniform"):
                    auc = fast_auc([s[method] for s in inj_scores],
                                   [s[method] for s in clean])
                    rows.append(dict(dataset_id=hrow.dataset_id,
                                     injection=kind, f=f, method=method,
                                     auc=auc, host_grid=host_grid,
                                     w=W5, n_test=N_TEST))
            got = [r for r in rows if r["dataset_id"] == hrow.dataset_id
                   and r["injection"] == kind and r["f"] == 0.10]
            print(f"  {hrow.dataset_id[:44]} {kind} f=0.10: " + " ".join(
                f"{r['method']}={r['auc']:.2f}" for r in got), flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(TABLES, "hard10_integrity.csv"),
                       index=False, encoding="utf-8")
    fig_integrity(df)
    print("stage 5 done -> hard10_integrity.csv, fig10_integrity.png",
          flush=True)


def fig_integrity(df):
    plt = get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4), dpi=200,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    # small x-dodge so coincident lines (encoding and lastdigit both hit
    # AUC 1.0) stay visible; dodge is visual only, values in the CSV
    meth_c = {"lastdigit": (PALETTE[1], -0.0022, "s", 6.5, 2.6),
              "encoding": (PALETTE[0], 0.0022, "o", 5.0, 1.6),
              "benford": (PALETTE[2], 0.0, "o", 5.0, 1.8)}
    titles = {"real_donor": "real foreign donor (telemetry)",
              "synthetic_finegrid": "synthetic fine-grid",
              "benford_matched": "Benford-passing (sharp case)"}
    for ax, kind in zip(axes, titles):
        style_ax(ax)
        sub = df[df.injection == kind]
        for method, (c, dx, mk, ms, lw) in meth_c.items():
            g = sub[sub.method == method].groupby("f").auc
            m, lo_, hi_ = g.mean(), g.min(), g.max()
            x = m.index.to_numpy() + dx
            ax.fill_between(x, lo_, hi_, color=c, alpha=0.15, linewidth=0)
            ax.plot(x, m.values, marker=mk, markersize=ms, linewidth=lw,
                    color=c, markeredgecolor=SURFACE, label=method,
                    zorder=4 if method == "encoding" else 3)
        ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel("injection fraction f", fontsize=8.5, color=MUTED)
        ax.set_title(titles[kind], fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(0.25, 1.05)
    axes[0].set_ylabel("detection AUC (windows w=2000)", fontsize=9,
                       color=MUTED)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right",
                   labelcolor=INK)
    fig.suptitle("The integrity channel: encoding drift vs Benford vs "
                 "last digit (mean over 4 real channels; band = min–max; "
                 "lines dodged where coincident)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(os.path.join(DIAG, "fig10_integrity.png"),
                facecolor=SURFACE)


# ============================================================ read-out
def readout():
    print("\n================ HARD10 READ-OUT ================")
    sp = pd.read_csv(os.path.join(TABLES, "hard10_semiprime.csv"))
    bal = sp[(sp.rho == 1.0) & (sp.null == "loguniform")]
    kf = bal[bal.statistic == "keff"].auc
    best = bal[bal.statistic != "keff"].groupby("statistic").auc.mean() \
        .sort_values(ascending=False)
    print(f"1. SEMIPRIME: keff AUC {kf.min():.3f}-{kf.max():.3f} "
          f"(pred ~0.5) | best full-profile: "
          + ", ".join(f"{k}={v:.3f}" for k, v in best.head(3).items())
          + f" -> prediction {'CONFIRMED' if best.iloc[0] >= 0.8 else 'FAILED'}")
    sk = pd.read_csv(os.path.join(TABLES, "hard10_spikein.csv"))
    rec = sk[sk.variant != "none"].groupby(["variant", "f"]) \
        .det_keff_c7.mean().unstack()
    print("2. SPIKE-IN recovery (frac channels flagged):")
    print(rec.round(2).to_string())
    powered = (rec.loc["unrounded", 0.05] > 0.5 and
               rec.loc["covarying", 0.10] > 0.5)
    print(f"   -> {'POWERED' if powered else 'OVER-CONTROLS'}")
    bg = pd.read_csv(os.path.join(TABLES, "hard10_binarygrid.csv"))
    print("3. BINARY GRID:")
    print(bg[["source", "cls", "c2_excess", "c5_excess"]].round(4)
          .to_string())
    dg = pd.read_csv(os.path.join(TABLES, "hard10_degradation.csv"))
    n1 = dg[(dg.family == "additive") & (dg.param == 1)]
    print(f"4. DEGRADATION: n+1 AUC = "
          f"{float(n1.auc_same_perturbed.iloc[0]):.3f} (pred <= 0.55)")
    ig = pd.read_csv(os.path.join(TABLES, "hard10_integrity.csv"))
    sharp = ig[(ig.injection == "benford_matched") & (ig.f == 0.10)]
    print("5. INTEGRITY (sharp case, f=0.10): " + ", ".join(
        f"{m}={sharp[sharp.method == m].auc.mean():.3f}"
        for m in ("encoding", "benford", "lastdigit")))


def refig():
    fig_binarygrid(pd.read_csv(os.path.join(TABLES,
                                            "hard10_binarygrid.csv")))
    fig_integrity(pd.read_csv(os.path.join(TABLES,
                                           "hard10_integrity.csv")))
    print("figures re-rendered from CSVs")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "refig":
        refig()
    if stage in ("semiprime", "all"):
        semiprime()
    if stage in ("spikein", "all"):
        spikein()
    if stage in ("binarygrid", "all"):
        binarygrid()
    if stage in ("degradation", "all"):
        degradation()
    if stage in ("integrity", "all"):
        integrity()
    if stage in ("readout", "all"):
        readout()
