"""TEST 6 (v10.3, EXPLORATORY): power curves and false-positive calibration,
assembled from the frozen power13 outputs plus the count26 hosts. No frozen
output is recomputed. Config: run_config_powerfp28.json.

Outputs: tables/powerfp28_curves.csv, tables/powerfp28_fpcal.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

CFG = json.load(open(os.path.join(common.CONFIG, "run_config_powerfp28.json")))
T = common.TABLES

spk = pd.read_csv(os.path.join(T, "power13_quotient_spikein.csv"))
q95 = spk[spk.tier == "none"].groupby("dataset_id").keff_core_q95.first()
q05 = spk[spk.tier == "none"].groupby("dataset_id").keff_core_q05.first()

rows = []
for (tier, f, did), g in spk[spk.tier != "none"].groupby(["tier", "f", "dataset_id"]):
    med = g.keff_core.median()
    band = (q95[did] - q05[did]) / 2 or np.nan   # half-width of the 90% noise band
    rows.append(dict(tier=tier, f=f, dataset_id=did, keff_core_median=med,
                     q95=q95[did], recovered=med > q95[did],
                     effect_in_bands=(med - 1.0) / band if band else np.nan))
cur = pd.DataFrame(rows)
cur.to_csv(os.path.join(T, "powerfp28_curves.csv"), index=False)
print(cur.groupby(["tier", "f"]).recovered.mean().unstack().round(2))

# false-positive calibration: realized exceedance of the q95 band at f=0
fp = []
base = spk[spk.tier == "none"]
if "rep" in base.columns and base.keff_core.notna().any():
    for did, g in base.groupby("dataset_id"):
        v = g.keff_core.dropna()
        if len(v):
            fp.append(dict(dataset_id=did, n_reps=len(v),
                           realized_fp=float((v > q95[did]).mean()),
                           source="power13 f=0 replicates"))
if not fp:
    print("power13 f=0 rows carry no replicate keff_core; using count26 bootstrap-band construction note instead")
# count26: FP calibration by construction (the q95 IS the 95th pct of the
# f=0 record bootstrap, R=200), realized rate over an independent check set
c26 = pd.read_csv(os.path.join(T, "count26_spikein.csv"))
for did, g in c26.groupby("dataset_id"):
    fp.append(dict(dataset_id=did, n_reps=np.nan, realized_fp=np.nan,
                   source="count26: q95 defined as 95th pct of R=200 f=0 bootstrap (nominal 0.05 by construction)"))
pd.DataFrame(fp).to_csv(os.path.join(T, "powerfp28_fpcal.csv"), index=False)
print("FP calibration rows:", len(fp))
for r in fp[:8]:
    print(" ", r)
