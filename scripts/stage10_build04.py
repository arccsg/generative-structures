"""Build 04: does anything predict process beyond encoding?

Reads only frozen/ml_features.csv, frozen/ml_labels.csv,
tables/family_profiles_v3.csv, tables/channel_profiles_v3.csv. No
factorization, no file re-reads.

Guardrails baked in:
 - features never contain labels (separate files, joined only for CV targets)
 - GroupKFold by dataset_family — no dataset spans train/test
 - imbalance: >=5-family domain classes, rest pooled to "other";
   class-weighted models; macro-F1 / balanced accuracy headline
 - unresolved-domain channels EXCLUDED from the domain target (not a domain)
 - synthetic_control channels excluded from supervised runs
 - deviation (recorded): the prompt's "de-rounded H2'" was never computed in
   Build 03; process-only uses L2_der (de-rounded second component) instead.
 - small domains (taxi, crime, accounting, admin_codes, campaign_finance)
   are below the 5-family bar -> pooled as "other"; this build measures the
   big domains only.
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TABLES, CONFIG, OUT

DIAG = os.path.join(OUT, "diagnostics")
FROZEN = os.path.join(OUT, "frozen")
PARENT_RUN = "lpg01-0643fff9"
SUBRUN = "04"
SEED = 20260704
WORKERS = 14

FEATURE_SETS = {
    "encoding_only": ["rounding_mass", "v3_share", "mean_log10_digits",
                      "sd_log10_digits"],
    "process_only": ["dL1_der", "dL2", "dTail_der", "keff", "L1_der",
                     "L2_der"],
}
FEATURE_SETS["all"] = FEATURE_SETS["encoding_only"] + \
    FEATURE_SETS["process_only"]

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


def load():
    feats = pd.read_csv(os.path.join(FROZEN, "ml_features.csv"),
                        low_memory=False)
    labels = pd.read_csv(os.path.join(FROZEN, "ml_labels.csv"),
                         low_memory=False)
    # L2_der lives in channel_profiles_v3 (documented substitution for H2')
    l2d = pd.read_csv(os.path.join(TABLES, "channel_profiles_v3.csv"),
                      usecols=["dataset_id", "L2_der"], low_memory=False) \
        .rename(columns={"dataset_id": "channel_id"})
    df = feats.merge(l2d, on="channel_id", how="left") \
        .merge(labels, on=["channel_id", "dataset_family"], how="inner")
    fam = pd.read_csv(os.path.join(TABLES, "family_profiles_v3.csv"),
                      low_memory=False)
    return df, fam


# ---------------------------------------------------------------- stage 1
def stage1(fam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, mannwhitneyu
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

    f = fam.dropna(subset=["dTail_der", "v3_share"]).copy()
    rho, p = spearmanr(f.v3_share, f.dTail_der)

    # partial Spearman controlling mean_log10: residualize ranks
    def rank(x):
        return pd.Series(x).rank().to_numpy()

    rv, rt, rm = rank(f.v3_share), rank(f.dTail_der), rank(f.mean_log10)

    def resid(y, x):
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)

    pr = np.corrcoef(resid(rv, rm), resid(rt, rm))[0, 1]

    over = f[f.dTail_der < -0.01]
    rest = f[f.dTail_der >= -0.01]
    u, pu = mannwhitneyu(over.v3_share, rest.v3_share,
                         alternative="two-sided")

    # call: dozenal-encoding requires the overshoot group to carry MORE v3
    # and a clearly negative v3~dTail' association
    dozenal = (rho < -0.2 and p < 0.01 and
               over.v3_share.median() > rest.v3_share.median())
    call = "dozenal-encoding" if dozenal else "genuine-candidate"

    pd.DataFrame([dict(
        spearman_rho=rho, spearman_p=p,
        partial_spearman_controlling_log10=pr,
        overshoot_families=len(over), other_families=len(rest),
        overshoot_v3_median=over.v3_share.median(),
        other_v3_median=rest.v3_share.median(),
        mannwhitney_U=u, mannwhitney_p=pu, call=call,
    )]).round(6).to_csv(os.path.join(TABLES, "overshoot_v3_test.csv"),
                        index=False, encoding="utf-8")

    cmap_doms = f.domain.value_counts().index.tolist()[:7]
    cmap = {d: PALETTE[i] for i, d in enumerate(cmap_doms)}
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(0, color=BASELINE_C, linewidth=1.0)
    ax.axhline(-0.01, color=BASELINE_C, linewidth=0.8, linestyle="--")
    for d in f.domain.unique():
        sub = f[f.domain == d]
        ax.scatter(sub.v3_share, sub.dTail_der, s=20,
                   color=cmap.get(d, GRAY), edgecolors=SURFACE,
                   linewidths=0.5, zorder=3)
    ax.set_xlabel("family mean v3_share (share of ln n in 3s — recorded, "
                  "never stripped)", fontsize=9, color=MUTED)
    ax.set_ylabel("dTail' (de-rounded residual tail)", fontsize=9,
                  color=MUTED)
    ax.set_title(f"Is the overshoot dozenal? Spearman ρ={rho:.3f} "
                 f"(p={p:.1e}), partial(log10)={pr:.3f} → {call}",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="",
                              markersize=6, markerfacecolor=v,
                              markeredgecolor=SURFACE, label=k)
                       for k, v in cmap.items()] +
              [Line2D([], [], marker="o", linestyle="", markersize=6,
                      markerfacecolor=GRAY, markeredgecolor=SURFACE,
                      label="other domains")],
              fontsize=7.5, frameon=False, loc="best", labelcolor=INK)
    fig.text(0.01, 0.01, f"run {PARENT_RUN} · sub-run {SUBRUN}",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig04_overshoot_vs_v3.png"),
                facecolor=SURFACE)
    plt.close(fig)
    return call, rho, p, pr


# ---------------------------------------------------------------- stage 3
def build_targets(df, fam):
    fam_counts = fam.groupby("domain").size()
    big = set(fam_counts[fam_counts >= 5].index) - {"unresolved"}
    small = set(fam_counts.index) - big - {"unresolved"}
    d = df[(df.corpus_role == "observational")].copy()
    d = d[d.domain != "unresolved"]
    d["domain_target"] = np.where(d.domain.isin(big), d.domain, "other")
    return d, sorted(big), sorted(small)


def run_supervised(d, call_log):
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import f1_score, balanced_accuracy_score

    targets = {"domain": "domain_target",
               "generating_process": "generating_process",
               "channel_kind": "channel_kind"}
    rows, preds_store = [], {}
    groups = d.dataset_family.to_numpy()
    gkf = GroupKFold(n_splits=5)

    def models():
        return {
            "logreg_l2": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("m", LogisticRegression(max_iter=4000, C=1.0,
                                         class_weight="balanced",
                                         random_state=SEED))]),
            "hist_gb": HistGradientBoostingClassifier(
                class_weight="balanced", random_state=SEED,
                early_stopping=False),
            "chance": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", DummyClassifier(strategy="stratified",
                                      random_state=SEED))]),
        }

    for tname, tcol in targets.items():
        sub = d.dropna(subset=[tcol]).copy()
        sub[tcol] = sub[tcol].astype(str)
        vc = sub[tcol].value_counts()
        rare = set(vc[vc < 20].index)   # rare target classes -> pooled
        if rare:
            sub[tcol] = sub[tcol].where(~sub[tcol].isin(rare),
                                        "other_pooled")
            print(f"  [{tname}] pooled rare classes (<20 channels): "
                  f"{sorted(rare)}", flush=True)
        y = sub[tcol].to_numpy()
        g = sub.dataset_family.to_numpy()
        for fs_name, cols in FEATURE_SETS.items():
            X = sub[cols].to_numpy(dtype=float)
            for mname, model in models().items():
                yp = cross_val_predict(model, X, y, groups=g, cv=gkf,
                                       n_jobs=WORKERS)
                f1 = f1_score(y, yp, average="macro")
                ba = balanced_accuracy_score(y, yp)
                rows.append(dict(target=tname, feature_set=fs_name,
                                 model=mname, unit="channel",
                                 macro_f1=f1, balanced_acc=ba,
                                 n=len(y), n_classes=len(set(y))))
                if tname == "domain" and mname == "hist_gb":
                    preds_store[fs_name] = (y, yp)
                print(f"  [{tname}] {fs_name:<14} {mname:<10} "
                      f"macroF1={f1:.3f} balacc={ba:.3f}", flush=True)

    # family-level replicate (domain only)
    famX = d.groupby("dataset_family").agg(
        {**{c: "mean" for c in FEATURE_SETS["all"]},
         "domain_target": "first"}).reset_index()
    yf = famX.domain_target.to_numpy()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fs_name, cols in FEATURE_SETS.items():
        Xf = famX[cols].to_numpy(dtype=float)
        for mname, model in models().items():
            yp = cross_val_predict(model, Xf, yf, cv=kf, n_jobs=WORKERS)
            rows.append(dict(target="domain", feature_set=fs_name,
                             model=mname, unit="family",
                             macro_f1=f1_score(yf, yp, average="macro"),
                             balanced_acc=balanced_accuracy_score(yf, yp),
                             n=len(yf), n_classes=len(set(yf))))

    scores = pd.DataFrame(rows).sort_values(
        ["target", "unit", "feature_set", "model"])
    scores.round(4).to_csv(os.path.join(TABLES, "ml_featureset_scores.csv"),
                           index=False, encoding="utf-8")
    return scores, preds_store, d


def figures_supervised(scores, preds_store, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]
    FOOT = (f"run {PARENT_RUN} · sub-run {SUBRUN} · GroupKFold(5) by "
            "dataset_family · class-weighted · seed recorded")

    # ---- scores figure
    ch = scores[scores.unit == "channel"]
    targets = ["domain", "generating_process", "channel_kind"]
    fsets = ["encoding_only", "process_only", "all"]
    fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    w = 0.13
    xbase = np.arange(len(targets))
    colors = {"encoding_only": PALETTE[0], "process_only": PALETTE[1],
              "all": PALETTE[2]}
    for i, fs in enumerate(fsets):
        for j, m in enumerate(["logreg_l2", "hist_gb"]):
            vals = [ch[(ch.target == t) & (ch.feature_set == fs) &
                       (ch.model == m)].macro_f1.iloc[0] for t in targets]
            x = xbase + (i - 1) * 2.3 * w + (j - 0.5) * w
            ax.bar(x, vals, width=w * 0.92, color=colors[fs],
                   alpha=1.0 if m == "hist_gb" else 0.55,
                   edgecolor=SURFACE, linewidth=0.6,
                   label=f"{fs} ({m})" if True else None)
    for k, t in enumerate(targets):
        cvals = ch[(ch.target == t) & (ch.model == "chance")]
        cmean = cvals.macro_f1.mean()
        ax.hlines(cmean, xbase[k] - 0.42, xbase[k] + 0.42, color=INK,
                  linewidth=1.2, linestyle="--")
    ax.set_xticks(xbase, targets, fontsize=9, color=INK)
    ax.set_ylabel("macro-F1 (channel level, grouped CV)", fontsize=9,
                  color=MUTED)
    ax.set_title("Feature-set comparison — dashed line = stratified-chance "
                 "macro-F1", fontsize=11, color=INK, loc="left", pad=12)
    handles, labels_ = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen.add(l)
            hh.append(h)
            ll.append(l)
    ax.legend(hh, ll, fontsize=7, frameon=False, ncols=3, loc="upper left",
              labelcolor=INK)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig04_featureset_scores.png"),
                facecolor=SURFACE)
    plt.close(fig)

    # ---- confusion matrix, domain, process-only, hist_gb
    y, yp = preds_store["process_only"]
    labs = sorted(set(y))
    cm = confusion_matrix(y, yp, labels=labs, normalize="true")
    fig, ax = plt.subplots(figsize=(7.8, 6.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs)), labs, rotation=45, ha="right",
                  fontsize=7, color=INK)
    ax.set_yticks(range(len(labs)), labs, fontsize=7, color=INK)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xlabel("predicted", fontsize=9, color=MUTED)
    ax.set_ylabel("true (row-normalized recall)", fontsize=9, color=MUTED)
    ax.set_title("Domain from PROCESS-ONLY features (HistGB, grouped CV) — "
                 "the honest test", fontsize=10.5, color=INK, loc="left",
                 pad=12)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig04_confusion_process_only.png"),
                facecolor=SURFACE)
    plt.close(fig)


def importance(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

    cols = FEATURE_SETS["all"]
    X = d[cols].to_numpy(dtype=float)
    y = d.domain_target.to_numpy()
    g = d.dataset_family.to_numpy()
    tr, te = next(GroupKFold(n_splits=5).split(X, y, g))
    m = HistGradientBoostingClassifier(class_weight="balanced",
                                       random_state=SEED,
                                       early_stopping=False) \
        .fit(X[tr], y[tr])
    r = permutation_importance(m, X[te], y[te], n_repeats=20,
                               random_state=SEED, n_jobs=WORKERS,
                               scoring="balanced_accuracy")
    imp = pd.DataFrame({"feature": cols,
                        "importance_mean": r.importances_mean,
                        "importance_sd": r.importances_std}) \
        .sort_values("importance_mean", ascending=False)
    imp.round(5).to_csv(os.path.join(TABLES, "ml_feature_importance.csv"),
                        index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ii = imp.iloc[::-1]
    ax.barh(np.arange(len(ii)), ii.importance_mean, xerr=ii.importance_sd,
            height=0.6, color=PALETTE[0], edgecolor=SURFACE, linewidth=0.6,
            error_kw=dict(ecolor=MUTED, lw=0.8))
    ax.set_yticks(np.arange(len(ii)), ii.feature, fontsize=8, color=INK)
    ax.set_xlabel("permutation importance (balanced-accuracy drop, "
                  "held-out group fold)", fontsize=9, color=MUTED)
    ax.set_title("Domain model (all features): what carries the weight?",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, f"run {PARENT_RUN} · sub-run {SUBRUN}",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig04_feature_importance.png"),
                facecolor=SURFACE)
    plt.close(fig)
    return imp


# ---------------------------------------------------------------- stage 4
def unsupervised(fam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (adjusted_rand_score,
                                 adjusted_mutual_info_score,
                                 silhouette_score)
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]

    cols = ["dL1_der", "dL2", "dTail_der", "keff", "L1_der", "L2_der"]
    f = fam.dropna(subset=[c for c in cols if c in fam.columns]).copy()
    X = f[[c for c in cols if c in f.columns]].to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)
    named = f.domain != "unresolved"

    rows = []
    lab_h = HDBSCAN(min_cluster_size=10).fit_predict(X)
    rows.append(("hdbscan_mcs10",
                 len(set(lab_h)) - (1 if -1 in lab_h else 0), lab_h))
    best = None
    for k in range(2, 13):
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=SEED).fit_predict(X)
        s = silhouette_score(X, km)
        if best is None or s > best[1]:
            best = (k, s, km)
    rows.append((f"kmeans_k{best[0]}", best[0], best[2]))

    out = []
    for name, k, lab in rows:
        mask = named.to_numpy() & (lab != -1)
        out.append(dict(
            method=name, clusters=k,
            ari_vs_domain=adjusted_rand_score(f.domain[mask], lab[mask]),
            ami_vs_domain=adjusted_mutual_info_score(f.domain[mask],
                                                     lab[mask]),
            n_scored=int(mask.sum()),
            note="unresolved families and HDBSCAN noise excluded from "
                 "scoring; labels used for scoring only"))
    res = pd.DataFrame(out)
    res.round(4).to_csv(os.path.join(TABLES, "unsup_vs_domain.csv"),
                        index=False, encoding="utf-8")

    P = PCA(n_components=2, random_state=SEED).fit_transform(X)
    lab = rows[1][2]  # kmeans best
    shapes = ["o", "s", "^", "D", "v", "P", "X", "*"]
    top_dom = f.domain.value_counts().index.tolist()[:8]
    smap = {d: shapes[i] for i, d in enumerate(top_dom)}
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    for cl in sorted(set(lab)):
        sub = lab == cl
        c = PALETTE[cl % len(PALETTE)]
        for dname in f.domain.unique():
            m2 = sub & (f.domain == dname).to_numpy()
            if m2.any():
                ax.scatter(P[m2, 0], P[m2, 1], s=24, color=c,
                           marker=smap.get(dname, "o"),
                           edgecolors=SURFACE, linewidths=0.5, zorder=3)
    ax.set_xlabel("PC1 of process-only space", fontsize=9, color=MUTED)
    ax.set_ylabel("PC2", fontsize=9, color=MUTED)
    ari = res.iloc[1].ari_vs_domain
    ax.set_title(f"Unsupervised on process-only features (KMeans "
                 f"k={best[0]}) — ARI vs domain = {ari:.3f}",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    fig.text(0.01, 0.01, f"run {PARENT_RUN} · sub-run {SUBRUN} · color = "
             "cluster, marker = domain (scoring only)", fontsize=6.5,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DIAG, "fig04_unsup_process_only.png"),
                facecolor=SURFACE)
    plt.close(fig)
    return res


def main():
    with open(os.path.join(CONFIG, "run_config_04.json"), "w") as f:
        json.dump({
            "run_id": PARENT_RUN, "subrun": SUBRUN, "seed": SEED,
            "workers": WORKERS,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "feature_sets": FEATURE_SETS,
            "deviations": ["process_only uses L2_der in place of the "
                           "prompt's de-rounded H2' (never computed in "
                           "Build 03; recomputing would need a corpus "
                           "re-read)"],
            "exclusions": ["synthetic_control channels excluded from "
                           "supervised runs", "domain=unresolved excluded "
                           "from the domain target",
                           "domains with <5 families pooled as 'other'"],
            "cv": "GroupKFold(5), groups=dataset_family; family-level "
                  "replicate with KFold(5)",
        }, f, indent=2)

    df, fam = load()
    call, rho, p, pr = stage1(fam)
    d, big, small = build_targets(df, fam)
    print(f"\ndomain classes (>=5 families): {big}")
    print(f"pooled as 'other' (<5 families — cannot be evaluated here): "
          f"{small}", flush=True)

    scores, preds, d = run_supervised(d, call)
    figures_supervised(scores, preds, d)
    imp = importance(d)
    unsup = unsupervised(fam)

    # ------------------------------------------------------- read-out
    ch = scores[(scores.unit == "channel")]

    def get(t, fs, m, col="macro_f1"):
        r = ch[(ch.target == t) & (ch.feature_set == fs) & (ch.model == m)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    print("\n================ BUILD 04 READ-OUT ================")
    print(f"1. OVERSHOOT CALL: {call}  (Spearman v3~dTail' rho={rho:.3f} "
          f"p={p:.1e}; partial controlling log10 = {pr:.3f})")
    print("2. process-only vs chance (macro-F1, HistGB | chance):")
    for t in ("domain", "generating_process", "channel_kind"):
        print(f"   {t:<20} {get(t, 'process_only', 'hist_gb'):.3f} | "
              f"chance {get(t, 'process_only', 'chance'):.3f}")
    print("3. all vs encoding-only lift (macro-F1, HistGB):")
    for t in ("domain", "generating_process", "channel_kind"):
        lift = get(t, "all", "hist_gb") - get(t, "encoding_only", "hist_gb")
        print(f"   {t:<20} all={get(t, 'all', 'hist_gb'):.3f}  "
              f"encoding={get(t, 'encoding_only', 'hist_gb'):.3f}  "
              f"lift={lift:+.3f}")
    print("4. top-5 permutation-important features (all-feature domain "
          "model):")
    for r in imp.head(5).itertuples():
        print(f"   {r.feature:<18} {r.importance_mean:.4f} "
              f"± {r.importance_sd:.4f}")
    print("5. unsupervised process-only vs domain:")
    for r in unsup.itertuples():
        print(f"   {r.method:<14} k={r.clusters} ARI={r.ari_vs_domain:.3f} "
              f"AMI={r.ami_vs_domain:.3f}")
    dom_po = get("domain", "process_only", "hist_gb")
    dom_ch = get("domain", "process_only", "chance")
    verdict = ("process-only features beat chance — some process signal "
               "survives encoding control"
               if dom_po > dom_ch + 0.05 else
               "process-only is at/near chance — on this corpus the "
               "L-profile is an encoding fingerprint")
    print(f"6. SUMMARY: {verdict}. Small domains (accounting, taxi, crime, "
          f"genomics<bar…) were not evaluable; their hypotheses move to "
          f"the data-acquisition track.")


if __name__ == "__main__":
    main()
