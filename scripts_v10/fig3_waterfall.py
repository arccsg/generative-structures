"""Figure 3: the four-driver reduction, driver by driver.

(a) The rounded-arm Tail excess of the arm domains under successive removal
operators: raw -> de-round (2*5 strip) -> deep strip (primes <= 7) -> core-
magnitude re-indexing. The last step is the reference correction that removes
the surviving concentration excess (-0.0184 -> -0.0001).
(b) Residue quantization, the fourth driver: planted resonant controls vs
planted generic controls vs observed channels (resonance score), showing the
~9-30x separation of the negative-control validation.

Sources (frozen, hashed): tables/derounding_effect.csv, deepstrip_effect.csv,
hunt09_residual_rebaseline.csv + claim C09 (-0.0184 -> -0.0001, CI
[-0.0004,+0.0004], run hunt09), resonance_controls.csv, residue_structure.csv.
"""
import os

import numpy as np
import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt

der = pd.read_csv(os.path.join(fs.TAB, "derounding_effect.csv"))
dstr = pd.read_csv(os.path.join(fs.TAB, "deepstrip_effect.csv"))
rc = pd.read_csv(os.path.join(fs.TAB, "resonance_controls.csv"))
rs = pd.read_csv(os.path.join(fs.TAB, "residue_structure.csv"))

# arm domains: the FROZEN hunt09 arm set, at family-equal grain -- the same
# estimand as the supplement's LOSO analysis (loso23_results.csv, variant
# "full"), so figure and text print identical numbers.
ARM = ["equity_markets", "seismology", "real_estate", "procurement",
       "food_nutrition"]
cp = pd.read_csv(os.path.join(fs.TAB, "channel_profiles_v3.csv"),
                 usecols=["dataset_family", "domain", "dTail", "dTail_der"])
oc = pd.read_csv(os.path.join(fs.GEO, "frozen", "observational_corpus_v2.csv"),
                 usecols=["dataset_family"])
cp = cp[cp.dataset_family.isin(set(oc.dataset_family))]
famcp = cp.groupby(["dataset_family", "domain"], as_index=False)[
    ["dTail", "dTail_der"]].mean()
armc = famcp[famcp.domain.isin(ARM)]
h9 = pd.read_csv(os.path.join(fs.TAB, "hunt09_residual_rebaseline.csv"))
arm9 = h9[h9.domain.isin(ARM)]
raw = float(armc.dTail.mean())
derd = float(armc.dTail_der.mean())
strip7 = float(arm9.dTail_d7.mean())
CORE_BEFORE, CORE_AFTER, CORE_LO, CORE_HI = strip7, -0.0001, -0.0004, 0.0004

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.FULL, 3.15),
                               gridspec_kw=dict(width_ratios=[1.25, 1]))

stages = ["raw", "decimal\nrounding\nremoved", "primes $\\leq7$\nstripped",
          "core-magn.\nre-indexed"]
vals = [raw, derd, strip7, CORE_AFTER]
x = np.arange(4)
colors = [fs.BLUE, fs.BLUE, fs.BLUE, fs.GREEN]
bars = ax1.bar(x, vals, width=0.62, color=colors)
ax1.errorbar([3], [CORE_AFTER], yerr=[[CORE_AFTER - CORE_LO], [CORE_HI - CORE_AFTER]],
             fmt="none", ecolor="#333333", capsize=3, lw=1)
for xi, v in zip(x, vals):
    ax1.text(xi, v - 0.004 if v > 0.02 else v + (0.0025 if v >= 0 else -0.003),
             f"${v:+.4f}$" if abs(v) < 0.001 else f"${v:+.3f}$",
             ha="center",
             va="top" if v > 0.02 else ("bottom" if v >= 0 else "top"),
             fontsize=7.5, color="white" if v > 0.02 else "black")
ax1.set_ylim(-0.030, 0.104)
ax1.annotate("", xy=(2.95, -0.004), xytext=(2.45, 0.034),
             arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
ax1.text(1.30, 0.040, "the deep-core excess $-0.0184$\n(the strip-stage bar) collapses\nunder its own core-magnitude\nreference",
         fontsize=7, color="#444444")
ax1.axhline(0, color="#444444", lw=0.8)
ax1.set_xticks(x, stages)
ax1.set_ylabel("Tail excess of rounded-arm domains, $d\\mathrm{Tail}$")
ax1.set_title("(a) three removal operators and one reference correction")

rc["kind"] = rc.control.str.replace(r"_rep\d+$", "", regex=True)
means = rc.groupby("kind").resonance_score.mean().sort_values()
obs_scores = rs.resonance_score.dropna()

labels = ["planted generic\n(log-uniform)"]
data = [float(means[means.index.str.startswith("generic")].mean())]
cols = [fs.GRAY]
for nm, v in means[means.index.str.startswith("resonant")].groupby(
        means[means.index.str.startswith("resonant")].index).mean().items():
    labels.append("planted " + nm.replace("resonant_", "tick ").replace("_", " "))
    data.append(float(v))
    cols.append(fs.VERM)
labels.append("observed channels (median)")
data.append(float(obs_scores.median()))
cols.append(fs.BLUE)
labels.append("observed channels (95th pct)")
data.append(float(obs_scores.quantile(0.95)))
cols.append(fs.BLUE)

y = np.arange(len(labels))
ax2.barh(y, data, color=cols, height=0.6)
for yy, v in zip(y, data):
    ax2.text(v * 1.12, yy, f"{v:.4f}", va="center", fontsize=6.8)
ax2.set_yticks(y, labels, fontsize=6.8)
ax2.invert_yaxis()
ax2.set_xscale("log")
ax2.set_xlabel("residue-quantization score (log scale)")
ax2.set_title("(b) driver 4 validated by planted controls")
fs.save(fig, "fig3_driver_waterfall")
