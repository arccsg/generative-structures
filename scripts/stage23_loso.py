"""TEST 1 (v10.3, EXPLORATORY): leave-one-source-out robustness.

Reruns the four-driver arm sequence (raw -> de-rounded -> deep-strip ->
core-magnitude re-indexed) and the core-magnitude residual collapse under:
full corpus (reproduction), each of three source exclusions, and
equal-family weighting. Plan and holds-rule frozen in
config/run_config_loso23.json BEFORE computation. Reads frozen tables only.

Output: tables/loso23_results.csv, tables/loso23_family_counts.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

CFG = json.load(open(os.path.join(common.CONFIG, "run_config_loso23.json")))
SEED = CFG["seed"]
ARM = CFG["arm_domains"]
T = common.TABLES

rng = np.random.default_rng(SEED)


def boot(vals, n=2000):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    m = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                  for _ in range(n)])
    return float(vals.mean()), float(np.percentile(m, 5)), float(np.percentile(m, 95))


def main():
    fam = pd.read_csv(os.path.join(T, "hunt09_residual_rebaseline.csv"))
    cp = pd.read_csv(os.path.join(T, "channel_profiles_v3.csv"),
                     usecols=["dataset_id", "dataset_family", "domain",
                              "dTail", "dTail_der", "rounding_mass"])
    corpus = pd.read_csv("frozen/observational_corpus_v2.csv",
                         usecols=["dataset_family"]).dataset_family.unique()
    cp = cp[cp.dataset_family.isin(corpus)]
    # family-level raw/de-rounded arm values (channel-weighted within family,
    # then family-equal across -- matching the hunt09 family-bootstrap grain)
    famcp = cp.groupby(["dataset_family", "domain"], as_index=False)[
        ["dTail", "dTail_der"]].mean()

    variants = [
        ("full", None, False),
        ("excl_network_telemetry", "network_telemetry", False),
        ("excl_procurement", "procurement", False),
        ("excl_census", "census", False),
        ("equal_family_weighting", None, True),  # per-family means are already
        # equally weighted in boot(); this variant additionally equal-weights
        # the raw/de-round stages computed from channels (identical grain), so
        # it coincides with 'full' at family grain and is reported for the
        # channel-weighted raw stage contrast.
    ]
    rows, counts = [], []
    for name, drop, eqw in variants:
        f9 = fam[fam.domain != drop] if drop else fam
        fc = famcp[famcp.domain != drop] if drop else famcp
        arm9 = f9[f9.domain.isin(ARM)]
        armc = fc[fc.domain.isin(ARM)]
        raw_m, raw_lo, raw_hi = boot(armc.dTail)
        der_m, der_lo, der_hi = boot(armc.dTail_der)
        d7_m, d7_lo, d7_hi = boot(arm9.dTail_d7)
        core_m, core_lo, core_hi = boot(arm9.dTail_core_c7)
        # channel-weighted raw arm value for the weighting contrast
        armch = cp[cp.domain.isin(ARM)] if not drop else cp[
            cp.domain.isin(ARM) & (cp.domain != drop)]
        raw_chw = float(armch.dTail.mean())
        holds = (
            (abs(core_m) < 0.005 or (core_lo <= 0 <= core_hi))
            and raw_m > 0.02
            and -0.05 < der_m < 0.005
            and -0.05 < d7_m < 0.005
            and abs(core_m) < 0.005
        )
        rows.append(dict(variant=name, arm_families=len(arm9),
                         raw_dTail=raw_m, raw_lo=raw_lo, raw_hi=raw_hi,
                         raw_channel_weighted=raw_chw,
                         deround_dTail=der_m, deround_lo=der_lo, deround_hi=der_hi,
                         strip7_dTail=d7_m, strip7_lo=d7_lo, strip7_hi=d7_hi,
                         core_dTail=core_m, core_lo=core_lo, core_hi=core_hi,
                         holds_frozen_rule=holds))
        print(f"{name:24s} armfam={len(arm9):3d} raw {raw_m:+.4f} "
              f"der {der_m:+.4f} strip7 {d7_m:+.4f} "
              f"core {core_m:+.4f} [{core_lo:+.4f},{core_hi:+.4f}] "
              f"holds={holds}", flush=True)
        for dom, g in (f9.groupby("domain") if not drop else
                       f9[f9.domain != drop].groupby("domain")):
            counts.append(dict(variant=name, domain=dom, families=len(g),
                               in_arm=dom in ARM))
    pd.DataFrame(rows).to_csv(os.path.join(T, "loso23_results.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(T, "loso23_family_counts.csv"), index=False)
    print("written loso23_results.csv, loso23_family_counts.csv")


if __name__ == "__main__":
    main()
