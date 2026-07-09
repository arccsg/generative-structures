"""Build 06: honest observational close-out.

Stages (python stage12_build06.py <stage>):
  freeze   — patch the `gen` quarantine leak, re-freeze corpus v2 (+manifest)
  pass     — deep-core corpus pass: strip all primes <=7 (and <=13 variant)
             per record, deep L-profiles (14 workers, memoized)
  baseline — matched deep baselines per ORIGINAL digit stratum (both variants)
  analyze  — accounting rescore on dL2', deep-strip survival test,
             observational-vs-constructed placement, corrected supervised
             number, figures + read-out

Shape coordinates are core-derived and scale-invariant; labels never enter
geometry (color/scoring only, post-hoc). Records whose deep core is 1
(p-smooth values) are excluded from deep-core means with the fraction logged;
null weighting renormalizes over strata with defined baselines.
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
SUBRUN = "06"
SEED = 20260706
BASELINE_N = 200_000
WORKERS = 14

GEN_RE = re.compile(r"(^|[_\-/ ])gen([_\-/ ]|$)", re.I)
STRIP7 = (2, 3, 5, 7)
STRIP13 = (2, 3, 5, 7, 11, 13)

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


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ freeze
def freeze():
    obs = pd.read_csv(os.path.join(FROZEN, "observational_corpus.csv"),
                      low_memory=False)
    syn = pd.read_csv(os.path.join(FROZEN, "synthetic_control.csv"),
                      low_memory=False)
    for d in (obs, syn):
        for c in ("archive_member", "sheet_or_table"):
            d[c] = d[c].fillna("")
    blob = (obs.file_path.astype(str) + "\x00" +
            obs.archive_member.astype(str) + "\x00" +
            obs.dataset_family.astype(str)).str.lower()
    hit = blob.str.contains(GEN_RE)
    moved = obs[hit].copy()
    kept = obs[~hit].copy()
    before_unres = int((obs.domain == "unresolved").sum())
    after_unres = int((kept.domain == "unresolved").sum())

    moved["corpus_role"] = "synthetic_control"
    kept["corpus_role"] = "observational"
    syn2 = pd.concat([syn, moved], ignore_index=True).sort_values(
        ["top_project", "file_path", "archive_member", "sheet_or_table",
         "column_name"], kind="mergesort")
    kept = kept.sort_values(
        ["top_project", "file_path", "archive_member", "sheet_or_table",
         "column_name"], kind="mergesort")

    p_obs = os.path.join(FROZEN, "observational_corpus_v2.csv")
    p_syn = os.path.join(FROZEN, "synthetic_control_v2.csv")
    kept.to_csv(p_obs, index=False, encoding="utf-8")
    syn2.to_csv(p_syn, index=False, encoding="utf-8")

    mv = moved.groupby("dataset_family").agg(
        channels=("dataset_id", "size"),
        example_path=("file_path", "first"),
        example_member=("archive_member", "first")).reset_index()
    mv.to_csv(os.path.join(DIAG, "quarantine_moved_06.csv"), index=False,
              encoding="utf-8")

    with open(os.path.join(FROZEN, "freeze_manifest_v2.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "patch": "quarantine token \\bgen\\b -> "
                     "(^|[_\\-/ ])gen([_\\-/ ]|$) (underscore leak)",
            "observational_corpus_v2": {
                "path": p_obs, "sha256": sha(p_obs), "rows": len(kept),
                "families": int(kept.dataset_family.nunique()),
                "domains": kept.domain.value_counts().to_dict()},
            "synthetic_control_v2": {
                "path": p_syn, "sha256": sha(p_syn), "rows": len(syn2)},
            "moved": {"families": int(moved.dataset_family.nunique()),
                      "channels": len(moved),
                      "unresolved_before": before_unres,
                      "unresolved_after": after_unres},
        }, f, indent=2)
    print(f"freeze v2: moved {moved.dataset_family.nunique()} families / "
          f"{len(moved):,} channels to synthetic_control; observational "
          f"now {len(kept):,} channels / {kept.dataset_family.nunique()} "
          f"families; unresolved {before_unres} -> {after_unres}")


# ================================================================== pass
def deep_stats(ints):
    n_used = len(ints)
    if n_used == 0:
        return None
    uniq, counts = np.unique(ints, return_counts=True)
    acc = {7: [np.zeros(4), 0, 0], 13: [np.zeros(4), 0, 0]}
    for v, c in zip(uniq.tolist(), counts.tolist()):
        pairs = lib.factor_pairs(v)
        ln_n = math.log(v)
        for variant, cut in ((7, STRIP7), (13, STRIP13)):
            contrib = [e * math.log(p) for p, e in pairs if p not in cut]
            ln_core = ln_n - sum(e * math.log(p) for p, e in pairs
                                 if p in cut)
            a = acc[variant]
            if not contrib or ln_core <= 1e-12:
                a[2] += c
                continue
            contrib.sort(reverse=True)
            L = [x / ln_core for x in contrib]
            h2 = sum(x * x for x in L)
            a[0] += np.array([L[0], L[1] if len(L) > 1 else 0.0,
                              1.0 - L[0] - (L[1] if len(L) > 1 else 0.0),
                              h2]) * c
            a[1] += c
    out = dict(n_used=int(n_used))
    for variant in (7, 13):
        sums, nc, nsm = acc[variant]
        sfx = f"_d{variant}"
        out[f"frac_smooth{sfx}"] = nsm / n_used
        if nc:
            out[f"L1{sfx}"] = sums[0] / nc
            out[f"L2{sfx}"] = sums[1] / nc
            out[f"Tail{sfx}"] = sums[2] / nc
            out[f"H2{sfx}"] = sums[3] / nc
        else:
            for k in ("L1", "L2", "Tail", "H2"):
                out[f"{k}{sfx}"] = np.nan
    return out


def _pass_task(task):
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
                prof = deep_stats(ints)
            except Exception as e:
                errors.append((fp, member, col, repr(e)[:300]))
                continue
            if prof is None:
                continue
            prof["dataset_id"] = ch["dataset_id"]
            out.append(prof)
    return out, errors


def corpus_pass():
    # process the full channel universe (v2 obs + v2 synthetic covers the
    # same set as v1); role split is applied at analyze time
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
    print(f"deep pass: {len(dd):,} channels, {len(tasks)} tasks, "
          f"{WORKERS} workers", flush=True)
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=lib.init_worker) as pool:
        for i, (rows, errs) in enumerate(pool.map(_pass_task, tasks)):
            results.extend(rows)
            errors.extend(errs)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    with open(os.path.join(INTER, "deep_v6.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"deep pass done: {len(results):,} channels, {len(errors)} errors",
          flush=True)


# =============================================================== baseline
def _baseline_task(args):
    d, chunk_idx, n = args
    lo = max(2, 10 ** (d - 1))
    rng = np.random.default_rng([SEED, d, chunk_idx])
    vals = rng.integers(lo, 10 ** d, size=n, dtype=np.int64)
    acc = {7: [np.zeros(4), np.zeros(4), 0, 0],
           13: [np.zeros(4), np.zeros(4), 0, 0]}
    for v in vals.tolist():
        pairs = lib.factor_pairs(v)
        ln_n = math.log(v)
        for variant, cut in ((7, STRIP7), (13, STRIP13)):
            contrib = [e * math.log(p) for p, e in pairs if p not in cut]
            ln_core = ln_n - sum(e * math.log(p) for p, e in pairs
                                 if p in cut)
            a = acc[variant]
            if not contrib or ln_core <= 1e-12:
                a[3] += 1
                continue
            contrib.sort(reverse=True)
            L = [x / ln_core for x in contrib]
            h2 = sum(x * x for x in L)
            x = np.array([L[0], L[1] if len(L) > 1 else 0.0,
                          1.0 - L[0] - (L[1] if len(L) > 1 else 0.0), h2])
            a[0] += x
            a[1] += x * x
            a[2] += 1
    return d, n, acc


def baseline_deep():
    D, chunks = 18, 10
    per = BASELINE_N // chunks
    tasks = [(d, c, per) for d in range(1, D + 1) for c in range(chunks)]
    acc = {d: {7: [np.zeros(4), np.zeros(4), 0, 0],
               13: [np.zeros(4), np.zeros(4), 0, 0]} for d in range(1, D + 1)}
    print(f"deep baseline: {D}×{BASELINE_N:,} seed={SEED}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=lib.init_worker) as pool:
        for d, n, a in pool.map(_baseline_task, tasks):
            for variant in (7, 13):
                for j in range(4):
                    if j < 2:
                        acc[d][variant][j] += a[variant][j]
                    else:
                        acc[d][variant][j] += a[variant][j]
    rows = []
    for d in range(1, D + 1):
        row = dict(stratum=d, n_draws=BASELINE_N)
        for variant in (7, 13):
            sums, sq, nc, nsm = acc[d][variant]
            sfx = f"_d{variant}"
            row[f"n_core_valid{sfx}"] = nc
            row[f"frac_smooth{sfx}"] = nsm / BASELINE_N
            if nc:
                mean = sums / nc
                sd = np.sqrt(np.maximum(sq / nc - mean ** 2, 0))
                row.update({f"E_L1{sfx}": mean[0], f"E_L2{sfx}": mean[1],
                            f"E_Tail{sfx}": mean[2], f"E_H2{sfx}": mean[3],
                            f"sd_L1{sfx}": sd[0], f"sd_Tail{sfx}": sd[2]})
            else:
                row.update({f"E_L1{sfx}": np.nan, f"E_L2{sfx}": np.nan,
                            f"E_Tail{sfx}": np.nan, f"E_H2{sfx}": np.nan,
                            f"sd_L1{sfx}": np.nan, f"sd_Tail{sfx}": np.nan})
        rows.append(row)
    pd.DataFrame(rows).round(6).to_csv(
        os.path.join(TABLES, "baseline_deepstrip.csv"), index=False,
        encoding="utf-8")
    print("baseline_deepstrip.csv written")


# ================================================================ analyze
def weighted_null(hists, bln, col):
    """Digit-weighted null with renormalization over strata whose baseline
    is defined (low strata can be fully p-smooth)."""
    D = len(bln)
    E = bln[col].to_numpy()
    valid = np.isfinite(E)
    out = np.full(len(hists), np.nan)
    for i, h in enumerate(hists):
        h = np.asarray(h, dtype=float)
        h = np.pad(h, (0, max(0, D - len(h))))[:D]
        w = h * valid
        s = w.sum()
        if s > 0:
            out[i] = (w / s) @ np.where(valid, E, 0.0)
    return out


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    FOOT = (f"run {PARENT_RUN} · sub-run {SUBRUN} · deep-core coordinates "
            "(primes ≤7 / ≤13 stripped), scale-invariant; labels post-hoc")

    obs2 = pd.read_csv(os.path.join(FROZEN, "observational_corpus_v2.csv"),
                       low_memory=False)
    syn2 = pd.read_csv(os.path.join(FROZEN, "synthetic_control_v2.csv"),
                       low_memory=False)
    with open(os.path.join(INTER, "deep_v6.pkl"), "rb") as f:
        deep = pd.DataFrame(pickle.load(f))
    with open(os.path.join(INTER, "obs_raw.pkl"), "rb") as f:
        dh = {r["dataset_id"]: r["digit_hist"]
              for r in pickle.load(f)["results"]}
    with open(os.path.join(INTER, "syn_raw.pkl"), "rb") as f:
        dh.update({r["dataset_id"]: r["digit_hist"]
                   for r in pickle.load(f)["results"]})
    bln = pd.read_csv(os.path.join(TABLES, "baseline_deepstrip.csv")) \
        .sort_values("stratum")
    shape = pd.read_csv(os.path.join(FROZEN, "shape_features.csv"),
                        low_memory=False)
    fam5 = pd.read_csv(os.path.join(FROZEN, "shape_features_family.csv"))

    ids = pd.concat([obs2[["dataset_id", "dataset_family", "domain",
                           "low_info"]].assign(role="observational"),
                     syn2[["dataset_id", "dataset_family", "domain",
                           "low_info"]].assign(role="synthetic_control")],
                    ignore_index=True)
    ch = ids.merge(deep, on="dataset_id", how="inner")
    hists = [dh.get(i, [1]) for i in ch.dataset_id]
    for variant in (7, 13):
        sfx = f"_d{variant}"
        for col in ("L1", "L2", "Tail", "H2"):
            ch[f"null_{col}{sfx}"] = weighted_null(hists, bln,
                                                   f"E_{col}{sfx}")
        ch[f"dL1{sfx}"] = ch[f"L1{sfx}"] - ch[f"null_L1{sfx}"]
        ch[f"dL2{sfx}"] = ch[f"L2{sfx}"] - ch[f"null_L2{sfx}"]
        ch[f"dTail{sfx}"] = ch[f"Tail{sfx}"] - ch[f"null_Tail{sfx}"]
        ch[f"keff{sfx}"] = ch[f"null_H2{sfx}"] / ch[f"H2{sfx}"]

    core = ch[~ch.low_info.astype(bool)]
    deep_cols = [c for c in ch.columns if c.startswith(("dL1_", "dL2_",
                 "dTail_", "keff_", "frac_smooth"))]
    g = core.groupby(["dataset_family", "role"])
    fam = g[deep_cols].mean()
    fam["domain"] = g.domain.first()
    fam = fam.reset_index()
    fam_obs = fam[fam.role == "observational"].merge(
        fam5[["dataset_family", "dTail_der", "dL1_der", "dL2_der"]],
        on="dataset_family", how="left")
    fam_syn = fam[fam.role == "synthetic_control"]

    # ---- family residual deviation (input for Build 06A): what remains
    # after magnitude (baseline), rounding (2·5) and composition (≤7 strip)
    frd = fam.copy()
    frd["residual_norm_d7"] = np.sqrt(
        frd.dL1_d7 ** 2 + frd.dL2_d7 ** 2 + frd.dTail_d7 ** 2)
    frd["residual_norm_d13"] = np.sqrt(
        frd.dL1_d13 ** 2 + frd.dL2_d13 ** 2 + frd.dTail_d13 ** 2)
    frd.sort_values(["role", "dataset_family"]).round(6).to_csv(
        os.path.join(TABLES, "family_residual_deviation.csv"), index=False,
        encoding="utf-8")

    # ---- Stage 2: accounting rescore on scale-clean dL2'
    labels = pd.read_csv(os.path.join(FROZEN, "ml_labels.csv"),
                         low_memory=False)
    fam_dom = labels.groupby("dataset_family").domain.first()
    f5 = fam5.copy()
    f5["domain"] = f5.dataset_family.map(fam_dom)
    rng = np.random.default_rng(SEED)

    def boot_ci(vals, n=1000):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        means = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                          for _ in range(n)])
        return float(vals.mean()), float(np.percentile(means, 5)), \
            float(np.percentile(means, 95))

    acc5 = f5[f5.domain == "accounting"]
    over5 = f5[f5.domain.isin(["equity_markets", "seismology",
                               "real_estate"])]
    rows = []
    for name, grp in [("accounting", acc5), ("overshoot_group", over5)]:
        for coord in ("dL2_der", "dTail_der", "dL1_der"):
            m, lo, hi = boot_ci(grp[coord])
            rows.append(dict(group=name, families=len(grp), coord=coord,
                             mean=m, ci5=lo, ci95=hi))
    acc_tab = pd.DataFrame(rows)
    acc_tab.round(5).to_csv(os.path.join(TABLES, "accounting_rescored.csv"),
                            index=False, encoding="utf-8")
    a_dl2 = acc_tab[(acc_tab.group == "accounting") &
                    (acc_tab.coord == "dL2_der")].iloc[0]
    o_dl2 = acc_tab[(acc_tab.group == "overshoot_group") &
                    (acc_tab.coord == "dL2_der")].iloc[0]
    acc_survives = (a_dl2.ci5 > 0) or (a_dl2.ci95 < 0)
    acc_distinct = (a_dl2.ci5 > o_dl2.ci95) or (a_dl2.ci95 < o_dl2.ci5)

    # ---- Stage 3: deep-strip survival
    eff_rows = []
    for dom, gg in fam_obs.groupby("domain"):
        eff_rows.append(dict(
            domain=dom, families=len(gg),
            mean_dTail_25strip=gg.dTail_der.mean(),
            mean_dTail_7strip=gg.dTail_d7.mean(),
            mean_dTail_13strip=gg.dTail_d13.mean(),
            mean_frac_smooth7=gg.frac_smooth_d7.mean(),
            mean_frac_smooth13=gg.frac_smooth_d13.mean()))
    eff = pd.DataFrame(eff_rows).sort_values("mean_dTail_25strip")
    eff.round(5).to_csv(os.path.join(TABLES, "deepstrip_effect.csv"),
                        index=False, encoding="utf-8")

    dom_top = fam_obs.domain.value_counts().index.tolist()[:7]
    cmap = {d: PALETTE[i] for i, d in enumerate(dom_top)}

    fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(0, color=BASELINE_C, linewidth=1.0)
    ax.axvline(0, color=BASELINE_C, linewidth=1.0)
    lim = np.nanpercentile(np.abs(fam_obs[["dTail_der",
                                           "dTail_d7"]].to_numpy()), 99)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.8,
            linestyle="--", zorder=2)
    for d in fam_obs.domain.unique():
        sub = fam_obs[fam_obs.domain == d]
        ax.scatter(sub.dTail_der, sub.dTail_d7, s=20,
                   color=cmap.get(d, GRAY), edgecolors=SURFACE,
                   linewidths=0.5, zorder=3)
    ax.set_xlabel("dTail' (2·5-stripped residual)", fontsize=9, color=MUTED)
    ax.set_ylabel("dTail'' (≤7-stripped residual)", fontsize=9, color=MUTED)
    ax.set_title("Does the overshoot survive the deep strip?", fontsize=11,
                 color=INK, loc="left", pad=12)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="",
                              markersize=6, markerfacecolor=v,
                              markeredgecolor=SURFACE, label=k)
                       for k, v in cmap.items()] +
              [Line2D([], [], marker="o", linestyle="", markersize=6,
                      markerfacecolor=GRAY, markeredgecolor=SURFACE,
                      label="other domains"),
               Line2D([], [], color=MUTED, linestyle="--",
                      label="y = x (no change)")],
              fontsize=7.5, frameon=False, loc="best", labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig06_overshoot_survival.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- Stage 4: observational vs constructed in deep-core space
    con = fam_syn.copy()
    con["is_gen"] = con.dataset_family.str.contains(
        r"gen-(rsa|kprime|bsmooth)", regex=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(0, color=BASELINE_C, linewidth=1.0)
    ax.axvline(0, color=BASELINE_C, linewidth=1.0)
    for d in fam_obs.domain.unique():
        sub = fam_obs[fam_obs.domain == d]
        ax.scatter(sub.dL1_d7, sub.dTail_d7, s=16, color=cmap.get(d, GRAY),
                   edgecolors=SURFACE, linewidths=0.4, zorder=3, alpha=0.9)
    other_syn = con[~con.is_gen]
    ax.scatter(other_syn.dL1_d7, other_syn.dTail_d7, s=30,
               facecolors="none", edgecolors=INK, linewidths=0.8, zorder=4,
               label="synthetic_control (other)")
    gen_syn = con[con.is_gen]
    ax.scatter(gen_syn.dL1_d7, gen_syn.dTail_d7, s=60, marker="X",
               color="#e34948", edgecolors=SURFACE, linewidths=0.6,
               zorder=5, label="constructed integers (RSA/kprime/bsmooth)")
    for r in gen_syn.itertuples():
        tag = re.search(r"gen-([a-z0-9\-]+?)-d\d", r.dataset_family)
        if tag and np.isfinite(r.dL1_d7):
            ax.annotate(tag.group(1), (r.dL1_d7, r.dTail_d7), fontsize=6,
                        color=INK, xytext=(4, 3), textcoords="offset points")
    # direction: obs overshoot-group centroid vs constructed centroid
    ovr = fam_obs[fam_obs.domain.isin(["equity_markets", "seismology",
                                       "real_estate"])]
    v_obs = np.array([np.nanmean(ovr.dL1_d7), np.nanmean(ovr.dTail_d7)])
    v_con = np.array([np.nanmean(gen_syn.dL1_d7),
                      np.nanmean(gen_syn.dTail_d7)])
    cosine = float(v_obs @ v_con /
                   (np.linalg.norm(v_obs) * np.linalg.norm(v_con))) \
        if np.linalg.norm(v_obs) > 0 and np.linalg.norm(v_con) > 0 else \
        float("nan")
    for v, lbl, c in [(v_obs, "obs overshoot centroid", INK),
                      (v_con, "constructed centroid", "#e34948")]:
        ax.annotate("", xy=tuple(v), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
    ax.set_xlabel("dL1'' (deep-core residual)", fontsize=9, color=MUTED)
    ax.set_ylabel("dTail''", fontsize=9, color=MUTED)
    ax.set_title(f"Observational families vs constructed integers, "
                 f"deep-core space (cosine of centroids = {cosine:.2f})",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    ax.legend(fontsize=7.5, frameon=False, loc="best", labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG,
                             "fig06_observational_vs_constructed.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- Stage 5: corrected supervised number
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import f1_score, balanced_accuracy_score

    sc_cols = ["dL1_der", "dL2_der", "dTail_der", "H2_der", "keff_der",
               "L1_der", "L2_der"]
    dsup = obs2[obs2.domain != "unresolved"].merge(
        shape.rename(columns={"channel_id": "dataset_id"}),
        on=["dataset_id", "dataset_family"], how="inner")
    fam_counts = dsup.groupby("domain").dataset_family.nunique()
    big = set(fam_counts[fam_counts >= 5].index)
    dsup["target"] = np.where(dsup.domain.isin(big), dsup.domain, "other")
    vc = dsup.target.value_counts()
    dsup["target"] = dsup.target.where(~dsup.target.isin(
        vc[vc < 20].index), "other_pooled")
    X = dsup[sc_cols].to_numpy(dtype=float)
    y = dsup.target.to_numpy()
    gvec = dsup.dataset_family.to_numpy()
    gkf = GroupKFold(n_splits=5)
    res_rows = []
    for mname, model in [
            ("hist_gb", HistGradientBoostingClassifier(
                class_weight="balanced", random_state=SEED,
                early_stopping=False)),
            ("chance", Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("m", DummyClassifier(
                                     strategy="stratified",
                                     random_state=SEED))]))]:
        yp = cross_val_predict(model, X, y, groups=gvec, cv=gkf,
                               n_jobs=WORKERS)
        res_rows.append(dict(model=mname,
                             macro_f1=f1_score(y, yp, average="macro"),
                             balanced_acc=balanced_accuracy_score(y, yp),
                             n=len(y), n_classes=len(set(y))))
    ml = pd.DataFrame(res_rows)
    ml.round(4).to_csv(os.path.join(TABLES, "ml_rescored_scaleclean.csv"),
                       index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    bars = [("Build 04\n(raw dL2, pre-patch)", 0.664, PALETTE[2]),
            ("Build 06\n(scale-clean, patched)",
             float(ml[ml.model == "hist_gb"].macro_f1.iloc[0]), PALETTE[0]),
            ("chance", float(ml[ml.model == "chance"].macro_f1.iloc[0]),
             GRAY)]
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars], width=0.55, edgecolor=SURFACE)
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.012, f"{b[1]:.3f}", ha="center", fontsize=9,
                color=INK)
    ax.set_ylabel("domain macro-F1 (grouped CV)", fontsize=9, color=MUTED)
    ax.set_title("Corrected supervised number", fontsize=11, color=INK,
                 loc="left", pad=12)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig06_ml_rescored.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- read-out
    with open(os.path.join(FROZEN, "freeze_manifest_v2.json")) as f:
        man = json.load(f)
    mf1 = float(ml[ml.model == "hist_gb"].macro_f1.iloc[0])
    mch = float(ml[ml.model == "chance"].macro_f1.iloc[0])
    print("\n================ BUILD 06 READ-OUT ================")
    print(f"1. QUARANTINE: moved {man['moved']['families']} families / "
          f"{man['moved']['channels']} channels; observational now "
          f"{man['observational_corpus_v2']['rows']:,} channels / "
          f"{man['observational_corpus_v2']['families']} families; "
          f"unresolved {man['moved']['unresolved_before']} -> "
          f"{man['moved']['unresolved_after']}")
    print(f"2. ACCOUNTING on dL2': mean={a_dl2['mean']:+.4f} 90% CI "
          f"[{a_dl2.ci5:+.4f}, {a_dl2.ci95:+.4f}] -> "
          f"{'SURVIVES (CI excludes 0)' if acc_survives else 'DISSOLVES'}"
          f"; distinct from overshoot group "
          f"({o_dl2['mean']:+.4f} [{o_dl2.ci5:+.4f},{o_dl2.ci95:+.4f}]): "
          f"{'yes' if acc_distinct else 'no'}")
    print("3. DEEP STRIP (mean dTail: 2·5 -> ≤7 -> ≤13):")
    for r in eff.itertuples():
        print(f"   {r.domain:<22} {r.mean_dTail_25strip:+.4f} -> "
              f"{r.mean_dTail_7strip:+.4f} -> {r.mean_dTail_13strip:+.4f}"
              f"   (7-smooth frac {r.mean_frac_smooth7:.2f})")
    print(f"4. OBS-vs-CONSTRUCTED direction cosine (deep-core plane): "
          f"{cosine:+.2f}")
    print(f"5. SUPERVISED corrected: macro-F1 {mf1:.3f} (was 0.664 with "
          f"raw dL2; chance {mch:.3f})")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    with open(os.path.join(CONFIG, "run_config_06.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN, "seed": SEED,
            "workers": WORKERS,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "quarantine_patch": r"(^|[_\-/ ])gen([_\-/ ]|$)",
            "deep_strip": "variants: primes <=7 and <=13 divided out; deep "
                          "baselines matched on ORIGINAL digit stratum; "
                          "p-smooth records excluded from core means "
                          "(fraction logged); null weights renormalized "
                          "over defined strata",
            "geometry_hygiene": "deep-core coordinates only; labels "
                                "post-hoc for color/scoring",
        }, f, indent=2)
    if stage in ("freeze", "all"):
        freeze()
    if stage in ("pass", "all"):
        corpus_pass()
    if stage in ("baseline", "all"):
        baseline_deep()
    if stage in ("analyze", "all"):
        analyze()
