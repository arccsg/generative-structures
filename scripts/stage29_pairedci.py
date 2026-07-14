"""B2 (v10.4, EXPLORATORY): paired family-bootstrap CIs for the Table S1
ladder deltas. Plan frozen in config/run_config_pairedci29.json.
Output: tables/pairedci29_deltas.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import stage22_geo15 as g15

CFG = json.load(open(os.path.join(common.CONFIG, "run_config_pairedci29.json")))
SEED = CFG["seed"]
T = common.TABLES

feat = pd.read_csv(g15.CFG["inputs"]["features"])
lab = pd.read_csv(g15.CFG["inputs"]["labels"])
enc = pd.read_csv(g15.CFG["inputs"]["encoding"])[["channel_id", "decimals", "tick_log10"]]
res = pd.read_csv(os.path.join(T, "residue_structure.csv"),
                  usecols=["dataset_id", "resonance_score", "lattice_gcd"]
                  ).rename(columns={"dataset_id": "channel_id"})
df = (feat.merge(lab.drop(columns=["dataset_family"]), on="channel_id")
          .merge(enc, on="channel_id", how="left")
          .merge(res, on="channel_id", how="left"))
df = df[(df.domain != "unresolved") & (df.corpus_role != "synthetic_control")]
df = g15._restrict_min_families(df)
df["log10_lattice_gcd"] = np.log10(df.lattice_gcd.clip(lower=1))

R = ["mean_log10_digits", "sd_log10_digits", "rounding_mass", "v3_share",
     "decimals", "tick_log10", "resonance_score", "log10_lattice_gcd"]
LADDERS = [("R_recording", R), ("R_plus_L1", R + ["L1"]),
           ("R_plus_L1_L3", R + ["L1", "L3"]),
           ("R_full_profile", R + ["L1", "L2", "L3", "Tail", "H2", "keff"])]

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

y = df.domain.to_numpy()
groups = df.dataset_family.to_numpy()
classes = np.unique(y)
probas = {}
for name, cols in LADDERS:
    X = np.nan_to_num(df[cols].to_numpy(float), nan=-1.0)
    proba = np.full((len(y), len(classes)), 0.0)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        cmap = {c: i for i, c in enumerate(clf.classes_)}
        for j, c in enumerate(classes):
            if c in cmap:
                proba[te, j] = p[:, cmap[c]]
    probas[name] = proba
    print("OOF done:", name, flush=True)

def macro_auc(idx, proba):
    aucs = []
    for j, c in enumerate(classes):
        yy = (y[idx] == c).astype(int)
        if 0 < yy.sum() < len(yy):
            aucs.append(roc_auc_score(yy, proba[idx, j]))
    return float(np.mean(aucs)) if aucs else np.nan

fams = pd.unique(groups)
fidx = {f: np.where(groups == f)[0] for f in fams}
rng = np.random.default_rng(SEED)
draws = {name: [] for name, _ in LADDERS}
for b in range(CFG["n_boot"]):
    pick = rng.choice(fams, size=len(fams), replace=True)   # ONE resample
    idx = np.concatenate([fidx[f] for f in pick])           # shared by all
    for name, _ in LADDERS:
        draws[name].append(macro_auc(idx, probas[name]))
D = {k: np.array(v) for k, v in draws.items()}
rows = []
for name, _ in LADDERS:
    point = macro_auc(np.arange(len(y)), probas[name])
    rows.append(dict(ladder=name, auc=point))
for a, bse in [("R_plus_L1", "R_recording"), ("R_plus_L1_L3", "R_recording"),
               ("R_plus_L1_L3", "R_plus_L1"), ("R_full_profile", "R_recording"),
               ("R_full_profile", "R_plus_L1")]:
    d = D[a] - D[bse]
    pt = macro_auc(np.arange(len(y)), probas[a]) - macro_auc(np.arange(len(y)), probas[bse])
    rows.append(dict(ladder=f"delta {a} - {bse}", auc=pt,
                     lo=float(np.percentile(d, 5)), hi=float(np.percentile(d, 95))))
    print(f"delta {a}-{bse}: {pt:+.4f} [{np.percentile(d,5):+.4f},{np.percentile(d,95):+.4f}]", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(T, "pairedci29_deltas.csv"), index=False)
print("written pairedci29_deltas.csv")
