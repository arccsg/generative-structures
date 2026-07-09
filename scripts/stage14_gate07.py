"""Build 07 (run gate07): ground-truth gate — does the L-profile detect
multiplication?

Entirely synthetic. All rungs live in [10^6, 10^7) and are BIN-MATCHED on
log10 to Uniform(6,7) (50 bins, equal quotas) — identical magnitude, sd,
skew, and digit distribution by construction, so any separation is
construction, not moments. No rounding in core rungs; rounding is the
Stage-3 manipulation. Values < 10^7 factorize via the SPF sieve (exact).

keff conventions (canonical): per-record keff_rec = E[H2 | d=7] / H2(rec)
(all values are 7-digit); rung keff = E[H2|d=7] / mean H2. "keff-only"
record-level classifier scores = keff_rec; "full shape" = logistic on
(L1, L2, Tail), 50/50 train/test.
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT
import build02_lib as lib

DIAG = os.path.join(OUT, "diagnostics")
RUN = "gate07"
SEED = 20260707
WORKERS = 14
N_RUNG = 120_000
N_SWEEP = 40_000
BINS = np.linspace(6.0, 7.0, 51)
LO, HI = 10 ** 6, 10 ** 7

SURFACE, GRID, BASELINE_C, MUTED, INK = ("#fcfcfb", "#e1e0d9", "#c3c2b7",
                                         "#898781", "#0b0b0b")
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
           "#e87ba4", "#eb6834"]
GRAY = "#898781"


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


# ------------------------------------------------------------- generators
def bin_match(pool_fn, n, rng, max_rounds=12):
    """Draw values from pool_fn(rng, m) until every log10 bin in [6,7) has
    its equal quota. Returns int64 array of length ~n (deficits logged)."""
    quota = n // (len(BINS) - 1)
    buckets = [[] for _ in range(len(BINS) - 1)]
    for _ in range(max_rounds):
        need = sum(max(0, quota - len(b)) for b in buckets)
        if need == 0:
            break
        vals = pool_fn(rng, max(200_000, 4 * need))
        vals = vals[(vals >= LO) & (vals < HI)]
        idx = np.digitize(np.log10(vals), BINS) - 1
        for v, i in zip(vals.tolist(), idx.tolist()):
            if 0 <= i < len(buckets) and len(buckets[i]) < quota:
                buckets[i].append(v)
    out = np.array([v for b in buckets for v in b], dtype=np.int64)
    rng.shuffle(out)
    return out


def g_single(rng, m):
    return np.floor(10 ** rng.uniform(6, 7, m)).astype(np.int64)


def g_sum(k):
    def f(rng, m):
        parts = 10 ** rng.uniform(6 - math.log10(k) - 0.35,
                                  7 - math.log10(k) + 0.05, (k, m))
        return np.floor(parts.sum(axis=0)).astype(np.int64)
    return f


def g_count(rng, m):
    # realistic tally: Poisson event count with log-uniform intensity
    lam = 10 ** rng.uniform(6, 7, m)
    return rng.poisson(lam).astype(np.int64)


def g_prod(k, jitter=0.12):
    def f(rng, m):
        t = rng.uniform(6, 7, m)
        w = rng.dirichlet(np.full(k, 40.0), size=m)  # near-equal splits
        logs = t[:, None] * w
        f_ = np.floor(10 ** (logs + rng.uniform(-jitter, jitter,
                                                (m, k)))).astype(np.int64)
        f_ = np.maximum(f_, 2)
        prod = np.ones(m, dtype=np.int64)
        for j in range(k):
            prod = prod * f_[:, j]
        return prod
    return f


def g_prod2_ratio(rho, jitter=0.08):
    """prod2 with factor log-ratio rho = log(a)/log(b)."""
    def f(rng, m):
        t = rng.uniform(6, 7, m)
        t1 = t * rho / (1 + rho)
        t2 = t / (1 + rho)
        a = np.maximum(np.floor(10 ** (t1 + rng.uniform(-jitter, jitter,
                                                        m))), 2)
        b = np.maximum(np.floor(10 ** (t2 + rng.uniform(-jitter, jitter,
                                                        m))), 2)
        return (a * b).astype(np.int64)
    return f


def g_prod_small(lo_f, hi_f):
    """Products of factors uniform in [lo_f, hi_f]: keep multiplying while
    below the band; overshoots past 1e7 are filtered by bin_match."""
    def f(rng, m):
        out = np.ones(m, dtype=np.int64)
        for _ in range(15):
            active = out < LO
            if not active.any():
                break
            fac = rng.integers(lo_f, hi_f + 1, m)
            out = np.where(active, out * fac, out)
        return out
    return f


# ------------------------------------------------------------ profiling
def _profile_chunk(vals):
    L1 = np.empty(len(vals))
    L2 = np.empty(len(vals))
    T = np.empty(len(vals))
    H = np.empty(len(vals))
    for i, v in enumerate(vals.tolist()):
        l1, l2, l3, tail, h2, r = lib.lprofile(v)
        L1[i] = l1
        L2[i] = l2
        T[i] = tail
        H[i] = h2
    return L1, L2, T, H


def profile_values(vals, pool):
    chunks = np.array_split(vals, WORKERS * 4)
    parts = list(pool.map(_profile_chunk, chunks))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def auc_keff(H2_pos, H2_neg):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(H2_pos)), np.zeros(len(H2_neg))]
    s = -np.r_[H2_pos, H2_neg]           # lower H2 => more product-like
    return float(roc_auc_score(y, s))


def auc_shape(P, Ng, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = np.vstack([np.column_stack(P[:3]), np.column_stack(Ng[:3])])
    y = np.r_[np.ones(len(P[0])), np.zeros(len(Ng[0]))]
    idx = rng.permutation(len(y))
    half = len(y) // 2
    tr, te = idx[:half], idx[half:]
    m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))


def main():
    os.makedirs(DIAG, exist_ok=True)
    with open(os.path.join(CONFIG, "run_config_gate07.json"), "w") as f:
        json.dump({
            "run": RUN, "seed": SEED, "workers": WORKERS,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "band": "[1e6, 1e7), all rungs bin-matched on log10 to "
                    "Uniform(6,7) over 50 bins (identical magnitude, "
                    "moments, digit distribution)",
            "rungs": "single, sum2, sum3, count(Poisson tally), prod2..6",
            "keff": "per-record keff_rec = E[H2|d=7]/H2(rec); rung keff = "
                    "E[H2|d=7]/mean H2; keff-only classifier = keff_rec",
            "no_rounding_in_core_rungs": True,
            "notes": "anchors (RSA/smooth/primorial/factorial) regenerated "
                     "synthetically as the same constructions",
        }, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    FOOT = f"run {RUN} · seed {SEED} · synthetic, moment/digit-matched"

    bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
    H2_null = float(bl[bl.stratum == "7"].E_H2.iloc[0])

    rng = np.random.default_rng(SEED)
    pool = ProcessPoolExecutor(max_workers=WORKERS,
                               initializer=lib.init_worker)
    lib.init_worker()

    # ---------------- Stage 1: rungs
    specs = [("single", g_single), ("sum2", g_sum(2)), ("sum3", g_sum(3)),
             ("count", g_count)] + \
        [(f"prod{k}", g_prod(k)) for k in range(2, 7)]
    rungs = {}
    rows = []
    for name, fn in specs:
        vals = bin_match(fn, N_RUNG, rng)
        P = profile_values(vals, pool)
        rungs[name] = P
        lg = np.log10(vals.astype(float))
        from scipy.stats import skew
        keff = H2_null / P[3].mean()
        rows.append(dict(
            rung=name, n=len(vals), log10_mean=lg.mean(), log10_sd=lg.std(),
            log10_skew=float(skew(lg)), L1=P[0].mean(), L2=P[1].mean(),
            Tail=P[2].mean(), H2=P[3].mean(), keff=keff,
            dL1=P[0].mean() - float(bl[bl.stratum == "7"].E_L1.iloc[0]),
            dTail=P[2].mean() - float(bl[bl.stratum == "7"].E_Tail.iloc[0])))
        print(f"  rung {name:<7} n={len(vals):,} log10 "
              f"{lg.mean():.3f}±{lg.std():.3f} keff={keff:.3f}", flush=True)
    pd.DataFrame(rows).round(5).to_csv(
        os.path.join(TABLES, "gate_rungs.csv"), index=False,
        encoding="utf-8")

    # ---------------- Stage 2: separation
    sep_rows = []
    negatives = ["single", "sum2", "sum3", "count"]
    for pk in [f"prod{k}" for k in range(2, 7)]:
        for ng in negatives:
            a_k = auc_keff(rungs[pk][3], rungs[ng][3])
            a_s = auc_shape(rungs[pk], rungs[ng],
                            np.random.default_rng([SEED, 1]))
            sep_rows.append(dict(product_rung=pk, vs=ng, auc_keff=a_k,
                                 auc_shape=a_s))
    sep = pd.DataFrame(sep_rows)
    sep.round(4).to_csv(os.path.join(TABLES, "gate_separation.csv"),
                        index=False, encoding="utf-8")

    # violin figure
    order = ["single", "sum2", "sum3", "count"] + \
        [f"prod{k}" for k in range(2, 7)]
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    data = [np.clip(H2_null / rungs[r][3], 0, 8) for r in order]
    vp = ax.violinplot(data, showmedians=True, widths=0.8)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(PALETTE[0] if order[i].startswith("prod")
                        else GRAY)
        b.set_alpha(0.75)
        b.set_edgecolor(SURFACE)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[part].set_color(INK)
        vp[part].set_linewidth(0.8)
    for k in range(1, 7):
        ax.axhline(k, color=GRID, linewidth=0.8)
        ax.text(len(order) + 0.42, k, f"keff={k}", fontsize=7, color=MUTED,
                va="center")
    agg = [H2_null / rungs[r][3].mean() for r in order]
    ax.scatter(np.arange(1, len(order) + 1), agg, s=42, marker="D",
               color=PALETTE[5], edgecolors=SURFACE, zorder=5,
               label="rung keff (aggregate)")
    ax.set_xticks(np.arange(1, len(order) + 1), order, fontsize=9,
                  color=INK)
    ax.set_ylabel("per-record keff  (E[H2|d=7] / H2)", fontsize=9,
                  color=MUTED)
    ax.set_title("keff by rung — matched magnitude, no rounding",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig07_keff_by_rung.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---------------- Stage 3: detection floor
    floor_rows = []
    # (a) balance sweep
    rhos = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0]
    bal_auc = []
    for rho in rhos:
        vals = bin_match(g_prod2_ratio(rho), N_SWEEP, rng)
        P = profile_values(vals, pool)
        a = auc_keff(P[3], rungs["single"][3][:N_SWEEP])
        bal_auc.append(a)
        floor_rows.append(dict(sweep="balance", param=rho,
                               keff=H2_null / P[3].mean(), auc_vs_single=a))
        print(f"  balance rho={rho}: keff={H2_null/P[3].mean():.3f} "
              f"AUC={a:.3f}", flush=True)
    # (b) factor size
    for name, lo_f, hi_f in [("small_10_100", 10, 100),
                             ("large_1e3_1e4", 1000, 10000)]:
        vals = bin_match(g_prod_small(lo_f, hi_f), N_SWEEP, rng)
        P = profile_values(vals, pool)
        a = auc_keff(P[3], rungs["single"][3][:N_SWEEP])
        floor_rows.append(dict(sweep="factor_size", param=name,
                               keff=H2_null / P[3].mean(), auc_vs_single=a))
        print(f"  size {name}: keff={H2_null/P[3].mean():.3f} AUC={a:.3f}",
              flush=True)
    # (c) factor count
    cnt_auc = []
    for k in range(2, 9):
        vals = bin_match(g_prod(k), N_SWEEP, rng)
        P = profile_values(vals, pool)
        a = auc_keff(P[3], rungs["single"][3][:N_SWEEP])
        cnt_auc.append((k, a, H2_null / P[3].mean()))
        floor_rows.append(dict(sweep="factor_count", param=k,
                               keff=H2_null / P[3].mean(), auc_vs_single=a))
    # (d) rounding tolerance (balanced large-factor prod2)
    base_vals = bin_match(g_prod2_ratio(1.0), N_SWEEP, rng)
    single_vals = bin_match(g_single, N_SWEEP, rng)
    rnd_rows = []
    for depth in (0, 1, 2, 3):
        q = 10 ** depth
        pv = (np.round(base_vals / q) * q).astype(np.int64)
        sv = (np.round(single_vals / q) * q).astype(np.int64)
        pv, sv = pv[pv > 1], sv[sv > 1]
        Pp = profile_values(pv, pool)
        Ps = profile_values(sv, pool)
        a_same = auc_keff(Pp[3], Ps[3])
        a_raw = auc_keff(Pp[3], rungs["single"][3][:N_SWEEP])
        keff_r = H2_null / Pp[3].mean()
        rnd_rows.append((depth, keff_r, a_same, a_raw))
        floor_rows.append(dict(sweep="rounding", param=depth, keff=keff_r,
                               auc_vs_single=a_same,
                               auc_vs_unrounded_single=a_raw))
        print(f"  rounding 10^{depth}: keff={keff_r:.3f} "
              f"AUC(same-rounded null)={a_same:.3f}", flush=True)
    pd.DataFrame(floor_rows).round(4).to_csv(
        os.path.join(TABLES, "gate_detection_floor.csv"), index=False,
        encoding="utf-8")

    # floor figures
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.plot(rhos, bal_auc, color=PALETTE[0], linewidth=1.8, marker="o",
            markersize=5, markeredgecolor=SURFACE)
    ax.axhline(0.6, color=PALETTE[5], linewidth=1.0, linestyle="--")
    ax.text(rhos[-1], 0.605, "AUC = 0.6 floor", fontsize=7.5,
            color=PALETTE[5], ha="right", va="bottom")
    ax.axhline(0.5, color=BASELINE_C, linewidth=0.8)
    ax.set_xlabel("factor log-ratio ρ = log(a)/log(b)", fontsize=9,
                  color=MUTED)
    ax.set_ylabel("AUC: prod2(ρ) vs single (keff score)", fontsize=9,
                  color=MUTED)
    ax.set_title("Detection floor — factor balance", fontsize=11,
                 color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig07_floor_balance.png"),
                facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ks = [c[0] for c in cnt_auc]
    ax.plot(ks, [c[1] for c in cnt_auc], color=PALETTE[0], linewidth=1.8,
            marker="o", markersize=5, markeredgecolor=SURFACE,
            label="AUC vs single")
    ax2 = ax.twinx()
    ax2.plot(ks, [c[2] for c in cnt_auc], color=PALETTE[2], linewidth=1.4,
             marker="s", markersize=4, markeredgecolor=SURFACE,
             label="rung keff")
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.spines["right"].set_color(BASELINE_C)
    ax2.set_ylabel("rung keff", fontsize=9, color="#8a6d00")
    ax.axhline(0.6, color=PALETTE[5], linewidth=1.0, linestyle="--")
    ax.set_xlabel("factor count k (fixed total magnitude — factors shrink "
                  "as k grows)", fontsize=9, color=MUTED)
    ax.set_ylabel("AUC: prod_k vs single", fontsize=9, color=PALETTE[0])
    ax.set_title("Detection floor — factor count and size", fontsize=11,
                 color=INK, loc="left", pad=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False,
              loc="center right", labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig07_floor_size_count.png"),
                facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    depths = [r[0] for r in rnd_rows]
    ax.plot(depths, [r[2] for r in rnd_rows], color=PALETTE[0],
            linewidth=1.8, marker="o", markersize=5,
            markeredgecolor=SURFACE, label="AUC vs same-rounded single")
    ax.axhline(0.6, color=PALETTE[5], linewidth=1.0, linestyle="--")
    ax2 = ax.twinx()
    ax2.plot(depths, [r[1] for r in rnd_rows], color=PALETTE[2],
             linewidth=1.4, marker="s", markersize=4,
             markeredgecolor=SURFACE, label="keff of rounded prod2")
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.spines["right"].set_color(BASELINE_C)
    ax2.set_ylabel("keff", fontsize=9, color="#8a6d00")
    ax.set_xticks(depths, [f"10^{d}" for d in depths], fontsize=9,
                  color=INK)
    ax.set_xlabel("rounding grain (nearest 10^d)", fontsize=9, color=MUTED)
    ax.set_ylabel("AUC", fontsize=9, color=PALETTE[0])
    ax.set_title("Rounding tolerance — balanced large-factor prod2",
                 fontsize=11, color=INK, loc="left", pad=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False,
              loc="center left", labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig07_rounding_tolerance.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---------------- Stage 4: anchors
    from sympy import randprime, primorial
    r2 = np.random.default_rng([SEED, 4])
    rsa = np.array([randprime(1000, 3163) * randprime(1000, 3163)
                    for _ in range(4000)], dtype=np.int64)
    rsa = rsa[(rsa >= LO) & (rsa < HI)]
    smooth7 = bin_match(g_prod_small(2, 7), 20_000,
                        np.random.default_rng([SEED, 5]))
    anchors = {
        "rsa_semiprime": profile_values(rsa, pool),
        "bsmooth_7": profile_values(smooth7, pool),
        "primorial_9699690": profile_values(
            np.array([9699690], dtype=np.int64), pool),   # 2·3·…·19
        "factorial_10!": profile_values(
            np.array([3628800], dtype=np.int64), pool),
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    for i, r in enumerate(order):
        P = rungs[r]
        ax.scatter(P[0].mean(), H2_null / P[3].mean(), s=42,
                   color=(PALETTE[0] if r.startswith("prod") else GRAY),
                   edgecolors=SURFACE, zorder=4)
        ax.annotate(r, (P[0].mean(), H2_null / P[3].mean()), fontsize=7.5,
                    color=INK, xytext=(5, 3), textcoords="offset points")
    for name, P in anchors.items():
        ax.scatter(P[0].mean(), H2_null / P[3].mean(), s=70, marker="X",
                   color=PALETTE[5], edgecolors=SURFACE, zorder=5)
        ax.annotate(name, (P[0].mean(), H2_null / P[3].mean()),
                    fontsize=7.5, color=PALETTE[5], xytext=(5, -8),
                    textcoords="offset points")
    ax.set_xlabel("mean L1", fontsize=9, color=MUTED)
    ax.set_ylabel("keff", fontsize=9, color=MUTED)
    ax.set_title("Synthetic rungs anchored against known extreme "
                 "constructions", fontsize=11, color=INK, loc="left",
                 pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig07_extremes_anchor.png"),
                facecolor=SURFACE)
    plt.close(fig)
    pool.shutdown()

    # ---------------- read-out
    rr = pd.DataFrame(rows).set_index("rung")
    s = sep.set_index(["product_rung", "vs"])
    rho_floor = next((rho for rho, a in zip(rhos, bal_auc) if a < 0.6),
                     None)
    print("\n================ GATE07 READ-OUT ================")
    print("1. rung keff: " + ", ".join(
        f"{r}={rr.loc[r].keff:.2f}" for r in order))
    print(f"   prod2 vs single: AUC_keff="
          f"{s.loc[('prod2', 'single')].auc_keff:.3f}, AUC_shape="
          f"{s.loc[('prod2', 'single')].auc_shape:.3f}; "
          f"prod2 vs sum2: AUC_keff="
          f"{s.loc[('prod2', 'sum2')].auc_keff:.3f}")
    ktrack = [f"k={k}:{rr.loc[f'prod{k}'].keff:.2f}" for k in range(2, 7)]
    print("   keff tracking: " + ", ".join(ktrack))
    print(f"2. detection floor: balance rho floor (AUC<0.6) = "
          f"{rho_floor if rho_floor else '>9 (never fails in sweep)'}")
    fs = pd.DataFrame(floor_rows)
    for r in fs[fs.sweep == "factor_size"].itertuples():
        print(f"   factor size {r.param}: keff={r.keff:.2f} "
              f"AUC={r.auc_vs_single:.3f}")
    cnt = fs[fs.sweep == "factor_count"]
    print("   factor count AUC: " + ", ".join(
        f"k={int(r.param)}:{r.auc_vs_single:.3f}" for r in
        cnt.itertuples()))
    print("3. rounding tolerance (AUC vs same-rounded single): " +
          ", ".join(f"10^{r[0]}:{r[2]:.3f}(keff {r[1]:.2f})"
                    for r in rnd_rows))
    print("4. anchors: " + ", ".join(
        f"{n}: keff={H2_null/P[3].mean():.2f}, L1={P[0].mean():.2f}"
        for n, P in anchors.items()))


if __name__ == "__main__":
    main()
