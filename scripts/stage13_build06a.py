"""Build 06A: is resonance the driver in the residual?

Stages (python stage13_build06a.py <stage>):
  residues  — per-channel residue-class structure over raw integer values:
              value mod p for primes 2..47 + powers/composites 4,8,9,25,27,49;
              TV distance + standardized chi^2 per modulus; lattice/AP flags
              (14 workers; no factorization needed)
  converge  — empirical convergence rate: D_base(B) ~ C·B^(-alpha) per
              qualifying family (log10 support >= 2, >= 5,000 records);
              factorizations memoized (14 workers)
  controls  — planted resonant/generic synthetic families + the
              RSA/kprime/bsmooth constructed families, scored by the same
              instruments
  relate    — does resonance explain Build 06's residual? correlations,
              partial correlations, variance lift, figure, read-out

Input mapping (recorded): the prompt's `family_residual_deviation.csv` is
produced by Build 06 analyze = norm of the deep-core (<=7-strip) residual
vector — deviation left after magnitude + rounding + composition. The
prompt's `deviation_metrics.csv` / `property_battery.csv` do not exist in
this pipeline; their role (magnitude/rounding covariates) is filled by
frozen/encoding_features_family.csv. No labels enter any computation before
the 'relate' stage's interpretation step.
"""
import hashlib
import json
import math
import os
import pickle
import re
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
INTER = os.path.join(DIAG, "intermediate02")
PARENT_RUN = "lpg01-0643fff9"
SUBRUN = "06A"
SEED = 202607061
WORKERS = 14

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
POWERS = [4, 8, 9, 25, 27, 49]
ALL_MODS = PRIMES + POWERS

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


# ============================================================ residue math
def residue_row(v):
    """Residue-structure stats for an int64 array of values >= 2.
    Returns dict of per-modulus TV & z plus lattice flags. n-independent
    effect sizes (TV) drive the aggregate; chi^2 z recorded alongside."""
    n = len(v)
    out = {"n_res": int(n)}
    if n < 100:
        return None
    vmax = int(v.max())
    tvs = []
    for m in ALL_MODS:
        # a modulus larger than the value range is trivially non-uniform;
        # only score moduli the data can actually equidistribute over
        if m * 5 > vmax:
            out[f"tv_mod{m}"] = np.nan
            out[f"z_mod{m}"] = np.nan
            continue
        counts = np.bincount((v % m).astype(np.int64), minlength=m)
        e = n / m
        chi2 = float(((counts - e) ** 2 / e).sum())
        df = m - 1
        z = (chi2 - df) / math.sqrt(2 * df)
        tv = 0.5 * float(np.abs(counts / n - 1.0 / m).sum())
        out[f"tv_mod{m}"] = tv
        out[f"z_mod{m}"] = z
        if m in PRIMES:
            tvs.append(tv)
    out["resonance_score"] = float(np.mean(tvs)) if tvs else np.nan
    out["n_moduli_scored"] = len(tvs)
    su = np.unique(v)
    if len(su) >= 3:
        g = int(np.gcd.reduce(np.diff(su)))
        out["lattice_gcd"] = g
        out["lattice_offset"] = int(su[0] % g) if g > 1 else 0
    else:
        out["lattice_gcd"] = 0
        out["lattice_offset"] = 0
    return out


def _residue_task(task):
    from stage9_build03 import _read_with_fallback
    out, errors = [], []
    for meta, channels in task:
        fp, member, sheet, ext, seed = meta
        want = [ch["column_name"] for ch in channels]
        try:
            df, _ = _read_with_fallback(fp, member, sheet, ext, seed, want)
        except Exception as e:
            for ch in channels:
                errors.append((fp, member, ch["column_name"], repr(e)[:300]))
            continue
        for ch in channels:
            col = ch["column_name"]
            if col not in df.columns:
                errors.append((fp, member, col, "column missing"))
                continue
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            try:
                ints, _ = lib.coerce_ints(series, ch["monetary"])
                row = residue_row(ints)
            except Exception as e:
                errors.append((fp, member, col, repr(e)[:300]))
                continue
            if row is None:
                continue
            row["dataset_id"] = ch["dataset_id"]
            out.append(row)
    return out, errors


def residues():
    frames = []
    for fname in ("observational_corpus_v2.csv", "synthetic_control_v2.csv"):
        frames.append(pd.read_csv(os.path.join(FROZEN, fname),
                                  low_memory=False))
    dd = pd.concat(frames, ignore_index=True)
    for c in ("archive_member", "sheet_or_table"):
        dd[c] = dd[c].fillna("")
    dd["monetary"] = (dd.channel_kind == "amount") | \
        dd.looks_monetary.astype(bool)
    tables = []
    for (fp, member, sheet), g in dd.groupby(
            ["file_path", "archive_member", "sheet_or_table"], sort=True):
        seed = int(hashlib.md5(f"{fp}|{member}".encode()).hexdigest()[:8],
                   16)
        channels = [dict(dataset_id=r.dataset_id,
                         column_name=r.column_name,
                         monetary=bool(r.monetary)) for r in g.itertuples()]
        tables.append(((fp, member, sheet, ext_of(member or fp), seed),
                       channels))
    tasks, cur = [], []
    for t in tables:
        cur.append(t)
        if len(cur) >= 40:
            tasks.append(cur)
            cur = []
    if cur:
        tasks.append(cur)
    print(f"residue pass: {len(dd):,} channels, {len(tasks)} tasks",
          flush=True)
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i, (rows, errs) in enumerate(pool.map(_residue_task, tasks)):
            results.extend(rows)
            errors.extend(errs)
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    res = pd.DataFrame(results)
    ids = dd[["dataset_id", "dataset_family"]]
    res = res.merge(ids, on="dataset_id", how="left")
    res.sort_values("dataset_id").round(6).to_csv(
        os.path.join(TABLES, "residue_structure.csv"), index=False,
        encoding="utf-8")
    print(f"residue pass done: {len(res):,} channels, {len(errors)} errors")


# ======================================================== convergence rate
def alpha_fit(values, bl):
    """Fit D_base(B) ~ C * B^-alpha over magnitude caps."""
    v = values[values > 1]
    if len(v) < 5000:
        return None
    lo, hi = math.log10(float(v.min())), math.log10(float(v.max()))
    if hi - lo < 2.0:
        return None
    caps = np.logspace(lo + 0.75, hi, 6)
    E = {c: bl[c].to_numpy() for c in ("E_L1", "E_L2", "E_Tail")}
    D_pts, B_pts = [], []
    for B in caps:
        sub = v[v <= B]
        if len(sub) < 500:
            continue
        uniq, counts = np.unique(sub, return_counts=True)
        sums = np.zeros(3)
        dh = np.zeros(len(bl))
        for u, c in zip(uniq.tolist(), counts.tolist()):
            l1, l2, l3, tail, h2, r = lib.lprofile(u)
            sums += np.array([l1, l2, tail]) * c
            dh[min(len(str(u)), len(bl)) - 1] += c
        m = sums / len(sub)
        w = dh / dh.sum()
        null = np.array([w @ E["E_L1"], w @ E["E_L2"], w @ E["E_Tail"]])
        D = float(np.linalg.norm(m - null))
        if D > 1e-6:
            D_pts.append(D)
            B_pts.append(B)
    if len(D_pts) < 4:
        return None
    x = np.log(np.array(B_pts))
    y = np.log(np.array(D_pts))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res_, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[0]
    yhat = A @ coef
    dof = max(1, len(x) - 2)
    se = math.sqrt(float(np.sum((y - yhat) ** 2)) / dof /
                   float(np.sum((x - x.mean()) ** 2)))
    alpha = -slope
    if alpha > 0.02 and se < 0.10:
        cls = "resonant" if alpha <= 0.15 else \
            ("generic" if 0.2 <= alpha <= 0.45 else "unresolved")
    else:
        cls = "unresolved"
    return dict(alpha=alpha, alpha_se=se, n_caps=len(D_pts),
                n_records=int(len(v)), log10_width=hi - lo,
                classification=cls)


def _converge_task(args):
    famname, values = args
    bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
    bl = bl[bl.stratum != "PD(0,1)"].copy()
    bl["d"] = bl.stratum.astype(int)
    bl = bl.sort_values("d")
    fit = alpha_fit(np.asarray(values, dtype=np.int64), bl)
    if fit is None:
        return None
    fit["dataset_family"] = famname
    return fit


def _collect_values_task(task):
    """Collect pooled per-family value samples (proportional per channel)."""
    from stage9_build03 import _read_with_fallback
    out = {}
    for meta, channels in task:
        fp, member, sheet, ext, seed = meta
        want = [ch["column_name"] for ch in channels]
        try:
            df, _ = _read_with_fallback(fp, member, sheet, ext, seed, want)
        except Exception:
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
            except Exception:
                continue
            if len(ints):
                fam = ch["family"]
                cur = out.setdefault(fam, [])
                if len(cur) < 200_000:
                    cur.extend(ints[:max(0, 200_000 - len(cur))].tolist())
    return out


def converge():
    frames = []
    for fname in ("observational_corpus_v2.csv", "synthetic_control_v2.csv"):
        frames.append(pd.read_csv(os.path.join(FROZEN, fname),
                                  low_memory=False))
    dd = pd.concat(frames, ignore_index=True)
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
                         monetary=bool(r.monetary), family=r.dataset_family)
                    for r in g.itertuples()]
        tables.append(((fp, member, sheet, ext_of(member or fp), seed),
                       channels))
    tasks, cur = [], []
    for t in tables:
        cur.append(t)
        if len(cur) >= 40:
            tasks.append(cur)
            cur = []
    if cur:
        tasks.append(cur)
    print(f"collecting family value pools ({len(tasks)} tasks)...",
          flush=True)
    pools = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for out in pool.map(_collect_values_task, tasks):
            for fam, vals in out.items():
                cur = pools.setdefault(fam, [])
                if len(cur) < 200_000:
                    cur.extend(vals[:200_000 - len(cur)])
    args = [(fam, vals) for fam, vals in sorted(pools.items())
            if len(vals) >= 5000]
    print(f"fitting alpha for {len(args)} candidate families "
          f"(of {len(pools)})", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=lib.init_worker) as pool:
        for r in pool.map(_converge_task, args):
            if r:
                rows.append(r)
    cv = pd.DataFrame(rows).sort_values("dataset_family")
    cv.round(5).to_csv(os.path.join(TABLES, "convergence_rate.csv"),
                       index=False, encoding="utf-8")
    print(f"convergence_rate.csv: {len(cv)} families fitted; "
          f"classes: {cv.classification.value_counts().to_dict()}")


# ================================================================ controls
def _control_task(args):
    name, kind, rep = args
    rng = np.random.default_rng([SEED, rep,
                                 int(hashlib.md5(name.encode())
                                     .hexdigest()[:6], 16)])
    N = 100_000
    x = 10 ** rng.uniform(2, 8, N)
    if kind == "generic":
        v = np.floor(x).astype(np.int64)
    elif kind == "residue_class":
        m, r = 7, 3
        v = (np.floor(x / m) * m + r).astype(np.int64)
    elif kind == "ap_grid":
        v = (np.floor(x / 30) * 30 + 7).astype(np.int64)
    elif kind == "multiples_of_3":
        v = (np.floor(x / 3) * 3).astype(np.int64)
    else:
        raise ValueError(kind)
    v = v[v > 1]
    row = residue_row(v)
    row["control"] = name
    bl = pd.read_csv(os.path.join(TABLES, "baseline.csv"))
    bl = bl[bl.stratum != "PD(0,1)"].copy()
    bl["d"] = bl.stratum.astype(int)
    bl = bl.sort_values("d")
    fit = alpha_fit(v, bl)
    if fit:
        row.update(alpha=fit["alpha"], alpha_se=fit["alpha_se"],
                   classification=fit["classification"])
    return row


def controls():
    specs = [("generic_loguniform", "generic"),
             ("resonant_mod7_r3", "residue_class"),
             ("resonant_ap30_b7", "ap_grid"),
             ("resonant_mult3", "multiples_of_3")]
    args = [(f"{name}_rep{rep}", kind, rep)
            for name, kind in specs for rep in range(3)]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=lib.init_worker) as pool:
        for r in pool.map(_control_task, args):
            rows.append(r)
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "resonance_controls.csv"), index=False,
        encoding="utf-8")
    print("controls scored -> resonance_controls.csv")


# ================================================================= relate
def relate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

    res = pd.read_csv(os.path.join(TABLES, "residue_structure.csv"),
                      low_memory=False)
    frd = pd.read_csv(os.path.join(TABLES,
                                   "family_residual_deviation.csv"))
    cv = pd.read_csv(os.path.join(TABLES, "convergence_rate.csv"))
    enc = pd.read_csv(os.path.join(FROZEN,
                                   "encoding_features_family.csv"))
    labels = pd.read_csv(os.path.join(FROZEN, "ml_labels.csv"),
                         low_memory=False)
    ctr = pd.read_csv(os.path.join(TABLES, "resonance_controls.csv"))
    fam_dom = labels.groupby("dataset_family").domain.first()

    fam_res = res.groupby("dataset_family").agg(
        resonance_score=("resonance_score", "mean"),
        lattice_gcd_max=("lattice_gcd", "max")).reset_index()
    f = frd[frd.role == "observational"].merge(fam_res, on="dataset_family",
                                               how="inner")
    f = f.merge(enc[["dataset_family", "mean_log10_digits",
                     "rounding_mass"]], on="dataset_family", how="left")
    f = f.merge(cv[["dataset_family", "alpha", "classification"]],
                on="dataset_family", how="left")
    f = f.dropna(subset=["residual_norm_d7", "resonance_score"])
    f["domain"] = f.dataset_family.map(fam_dom)

    rho, p = spearmanr(f.resonance_score, f.residual_norm_d7)

    def rank(x):
        return pd.Series(x).rank().to_numpy()

    def resid(y, x):
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)

    rr = rank(f.resonance_score)
    ry = rank(f.residual_norm_d7)
    rm = rank(f.mean_log10_digits)
    partial = float(np.corrcoef(resid(rr, rm), resid(ry, rm))[0, 1])

    both = f.dropna(subset=["alpha"])
    rho_a, p_a = (spearmanr(both.alpha, both.residual_norm_d7)
                  if len(both) >= 8 else (np.nan, np.nan))
    proxy_rho, proxy_p = (spearmanr(both.alpha, both.resonance_score)
                          if len(both) >= 8 else (np.nan, np.nan))

    # variance lift on the residual: base covariates -> + resonance
    def r2_of(cols):
        X = f[cols].to_numpy(dtype=float)
        X = np.column_stack([X, np.ones(len(X))])
        y = f.residual_norm_d7.to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

    r2_base = r2_of(["mean_log10_digits", "rounding_mass"])
    r2_full = r2_of(["mean_log10_digits", "rounding_mass",
                     "resonance_score"])
    lift = r2_full - r2_base

    # ---- figure
    fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    cls_col = {"resonant": "#e34948", "generic": "#2a78d6"}
    for cls, colr in [("resonant", "#e34948"), ("generic", "#2a78d6")]:
        sub = f[f.classification == cls]
        ax.scatter(sub.resonance_score, sub.residual_norm_d7, s=30,
                   color=colr, edgecolors=SURFACE, linewidths=0.5,
                   zorder=4, label=f"α reads {cls}")
    rest = f[~f.classification.isin(cls_col)]
    ax.scatter(rest.resonance_score, rest.residual_norm_d7, s=14,
               color=GRAY, edgecolors=SURFACE, linewidths=0.4, zorder=3,
               label="α unresolved / not measurable")
    ax.set_xlabel("family resonance_score (mean TV from uniform residues, "
                  "primes 2–47)", fontsize=9, color=MUTED)
    ax.set_ylabel("Build-06 residual deviation (deep-core norm, ≤7 strip)",
                  fontsize=9, color=MUTED)
    ax.set_title(f"Resonance vs the unexplained residual — Spearman "
                 f"ρ={rho:.3f} (p={p:.1e}), partial(log10)={partial:.3f}, "
                 f"ΔR²={lift:+.3f}", fontsize=10, color=INK, loc="left",
                 pad=12)
    ax.legend(fontsize=7.5, frameon=False, loc="best", labelcolor=INK)
    fig.text(0.01, 0.01, f"run {PARENT_RUN} · sub-run {SUBRUN} · residues "
             "from raw integers; labels only for description", fontsize=6.5,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig06A_resonance_vs_residual.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- read-out
    tv_cols = [c for c in res.columns if c.startswith("tv_mod")]
    mod_means = {c.replace("tv_mod", ""):
                 float(res[c].mean()) for c in tv_cols}
    top_mods = sorted(mod_means.items(), key=lambda kv: -kv[1])[:6]
    resn = f[f.classification == "resonant"]
    gen_ok = ctr[ctr.control.str.startswith("generic")]
    reso_ok = ctr[~ctr.control.str.startswith("generic")]
    ctrl_pass = (reso_ok.resonance_score.min() >
                 gen_ok.resonance_score.max() * 5)

    print("\n================ BUILD 06A READ-OUT ================")
    print(f"1. resonance_score over {len(f)} observational families: "
          f"median={f.resonance_score.median():.4f}, "
          f"p90={f.resonance_score.quantile(.9):.4f}, "
          f"max={f.resonance_score.max():.4f}")
    print("   moduli carrying most residue non-uniformity (mean TV): "
          + ", ".join(f"mod{m}={v:.3f}" for m, v in top_mods))
    print(f"2. alpha measurable for {len(cv)} families; classes: "
          f"{cv.classification.value_counts().to_dict()}")
    print(f"   alpha distribution: median={cv.alpha.median():.3f}, "
          f"IQR=[{cv.alpha.quantile(.25):.3f}, "
          f"{cv.alpha.quantile(.75):.3f}]")
    if len(resn):
        print("   resonant (alpha<=0.15) observational families:")
        for r in resn.itertuples():
            print(f"     {r.dataset_family:<52} {str(r.domain):<18} "
                  f"alpha={r.alpha:.3f}")
    else:
        print("   no observational family reads resonant (alpha<=0.15)")
    print(f"   proxy validation: Spearman(alpha, resonance_score) = "
          f"{proxy_rho if not np.isnan(proxy_rho) else float('nan'):.3f} "
          f"on {len(both)} families with both")
    print(f"3. resonance vs 06 residual: Spearman rho={rho:.3f} (p={p:.1e})"
          f"; partial controlling log10 = {partial:.3f}; "
          f"alpha vs residual rho={rho_a if not np.isnan(rho_a) else float('nan'):.3f}; "
          f"variance lift on residual dR2={lift:+.4f} "
          f"(base R2={r2_base:.3f} -> {r2_full:.3f})")
    print(f"4. controls: planted resonant scores "
          f"[{reso_ok.resonance_score.min():.3f}, "
          f"{reso_ok.resonance_score.max():.3f}], generic "
          f"[{gen_ok.resonance_score.min():.4f}, "
          f"{gen_ok.resonance_score.max():.4f}] -> "
          f"{'PASS' if ctrl_pass else 'FAIL'}; planted alphas: "
          + ", ".join(f"{r.control}:{getattr(r, 'alpha', float('nan')):.3f}"
                      for r in ctr.itertuples() if hasattr(r, "alpha")
                      and not pd.isna(r.alpha)))
    rsa = ctrCheck = None
    # constructed families: report from residue table
    con = res[res.dataset_family.str.contains("gen-rsa|gen-kprime|"
                                              "gen-bsmooth", na=False,
                                              regex=True)]
    if len(con):
        print(f"   RSA/kprime/bsmooth constructed families: "
              f"resonance_score median="
              f"{con.resonance_score.median():.4f} (n={len(con)}) — "
              f"constructed-but-not-residue-resonant contrast")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    with open(os.path.join(CONFIG, "run_config_06a.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN, "seed": SEED,
            "workers": WORKERS,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "resonance_score": "mean total-variation distance from uniform "
                               "of value mod p over primes 2..47, scored "
                               "only for moduli with 5m <= max(value); "
                               "chi^2 z recorded per modulus; powers/"
                               "composites 4,8,9,25,27,49 recorded",
            "alpha": "D_base(B) ~ C*B^-alpha, 6 log-spaced caps, OLS on "
                     "log-log; resonant <=0.15, generic 0.2-0.45, needs "
                     "se<0.1; families with log10 width >=2 and >=5k "
                     "records",
            "input_mapping": "family_residual_deviation.csv = Build-06 "
                             "deep-core (<=7) residual norm; "
                             "deviation_metrics/property_battery from the "
                             "prompt do not exist — encoding_features_"
                             "family.csv supplies magnitude/rounding "
                             "covariates",
            "geometry_hygiene": "residues from raw integers; labels only "
                                "for post-hoc description",
        }, f, indent=2)
    if stage in ("residues", "all"):
        residues()
    if stage in ("converge", "all"):
        converge()
    if stage in ("controls", "all"):
        controls()
    if stage in ("relate", "all"):
        relate()
