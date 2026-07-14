"""TEST 2 (v10.3, EXPLORATORY): task-sensitivity positive control for the
family-grouped domain-transfer design, plus diagnostics of the real geo15
digit-residual stage. Plan frozen in config/run_config_taskpos24.json.

Outputs: tables/taskpos24_calibration.csv, tables/taskpos24_permutation.csv,
         tables/taskpos24_s1_diagnostics.csv
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import build02_lib as lib
import stage22_geo15 as g15

CFG = json.load(open(os.path.join(common.CONFIG, "run_config_taskpos24.json")))
SEED = CFG["seed"]
T = common.TABLES
N_FAM, CH_PER_FAM, N_REC = 150, 3, 600


def make_corpus(rng, grid, w, tick=None):
    """Synthetic pseudo-channels; class B gets the recording treatment."""
    base = g15.load_baseline()
    rows = []
    for fam in range(N_FAM):
        d = int(rng.integers(4, 9))          # family latent magnitude
        cls = fam % 2                        # alternate classes across families
        for ch in range(CH_PER_FAM):
            v = rng.integers(10 ** (d - 1), 10 ** d, size=N_REC)
            if cls == 1 and (grid or tick):
                m = rng.random(N_REC) < w
                if tick:
                    v[m] = (v[m] // tick) * tick
                else:
                    v[m] = (np.round(v[m] / grid) * grid).astype(np.int64)
            cm = g15.channel_means(v)
            if cm is None:
                continue
            w_d = cm.pop("digit_hist")
            eL1 = float(np.dot(w_d, base["E_L1"].reindex(range(1, 19)).to_numpy()))
            eL3 = float(np.dot(w_d, base["E_L3"].reindex(range(1, 19)).to_numpy()))
            # grid-mass of the channel (c2/c5 log-share, for the effect size)
            uniq, cnt = np.unique(v[v > 1], return_counts=True)
            c25 = 0.0
            for x, c in zip(uniq.tolist(), cnt.tolist()):
                pr = dict(lib.factor_pairs(int(x)))
                c25 += c * (pr.get(2, 0) * math.log(2) + pr.get(5, 0) * math.log(5)) / math.log(x)
            rows.append(dict(family=f"f{fam}", cls=cls, L1=cm["L1"], L3=cm["L3"],
                             dL1=cm["L1"] - eL1, dL3=cm["L3"] - eL3,
                             grid_mass=c25 / cnt.sum()))
    return pd.DataFrame(rows)


def auc_pipeline(df, cols, seed):
    return g15._stage_auc(df[cols].to_numpy(float), df.cls.to_numpy(),
                          df.family.to_numpy(), seed, n_boot=100)


def main():
    lib.init_worker()
    rng = np.random.default_rng(SEED)
    rows = []
    for name, grid, w in [tuple(x) for x in CFG["effect_levels"]]:
        tick = None
        if name.startswith("tick"):
            tick, grid = grid, None
        df = make_corpus(rng, grid, w, tick)
        shift = df[df.cls == 1].grid_mass.mean() - df[df.cls == 0].grid_mass.mean()
        for coords, label in ((["L1", "L3"], "S0_raw"), (["dL1", "dL3"], "S1_digit")):
            auc, lo, hi, n = auc_pipeline(df, coords, SEED)
            rows.append(dict(level=name, coords=label, grid_mass_shift=shift,
                             auc=auc, auc_lo=lo, auc_hi=hi, n_channels=n))
            print(f"{name:14s} {label:9s} shift={shift:+.4f} "
                  f"AUC={auc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(T, "taskpos24_calibration.csv"), index=False)

    # family-level label-permutation null on the largest-effect condition
    df = make_corpus(rng, 100, 1.0)
    perm_rows = []
    fams = df.family.unique()
    for i in range(20):
        prng = np.random.default_rng([SEED, 500 + i])
        lab = dict(zip(fams, prng.permutation([f % 2 for f in range(len(fams))])))
        dfp = df.assign(cls=df.family.map(lab))
        auc, lo, hi, n = auc_pipeline(dfp, ["L1", "L3"], SEED + i)
        perm_rows.append(dict(draw=i, auc=auc))
    pdf = pd.DataFrame(perm_rows)
    pdf.to_csv(os.path.join(T, "taskpos24_permutation.csv"), index=False)
    print(f"permutation null: mean {pdf.auc.mean():.3f} "
          f"range [{pdf.auc.min():.3f},{pdf.auc.max():.3f}]")

    # diagnostics of the real geo15 S1 stage (per-fold instability)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    ch = pd.read_csv(os.path.join(T, "geo15_controlled_channels.csv"))
    ch = ch.dropna(subset=["g15_dL1", "g15_dL3"])
    fam_per_dom = ch.groupby("domain")["dataset_family"].nunique()
    ch = ch[ch.domain.isin(fam_per_dom[fam_per_dom >= 5].index)]
    X = ch[["g15_dL1", "g15_dL3"]].to_numpy(float)
    y = ch.domain.to_numpy()
    grp = ch.dataset_family.to_numpy()
    diag = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, grp)):
        clf = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        cols = {c: i for i, c in enumerate(clf.classes_)}
        aucs = []
        for c in np.unique(y[te]):
            yy = (y[te] == c).astype(int)
            if 0 < yy.sum() < len(yy) and c in cols:
                aucs.append(roc_auc_score(yy, p[:, cols[c]]))
        diag.append(dict(fold=k, macro_auc=float(np.mean(aucs)),
                         n_test=len(te), n_test_families=len(np.unique(grp[te])),
                         n_test_classes=len(np.unique(y[te])),
                         dominant_class_share=float(
                             pd.Series(y[te]).value_counts(normalize=True).iloc[0])))
        print(f"S1 fold {k}: macro-AUC {diag[-1]['macro_auc']:.3f} "
              f"test fams {diag[-1]['n_test_families']} "
              f"classes {diag[-1]['n_test_classes']} "
              f"dom-share {diag[-1]['dominant_class_share']:.2f}", flush=True)
    pd.DataFrame(diag).to_csv(os.path.join(T, "taskpos24_s1_diagnostics.csv"), index=False)
    print("written taskpos24_*.csv")


if __name__ == "__main__":
    main()
