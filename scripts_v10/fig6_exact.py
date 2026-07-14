"""Figure 6: the complementary exact-integer regime (positive contrast).

Effective factor count k_eff by class: additive constructions and recorded
fields sit at 1 (the generic integer); exact multiplicative constructions and
measured multiplicative fields depart in both directions.

Sources (frozen, hashed): tables/gate_rungs.csv (run gate07),
hunt09_lowkeff.csv, hunt09_highkeff.csv (run hunt09), vote14_profiles.csv,
emp11_profiles.csv; recorded-corpus k_eff from family_profiles_v3.csv.
"""
import os

import numpy as np
import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt

gate = pd.read_csv(os.path.join(fs.TAB, "gate_rungs.csv")).set_index("rung")
lowk = pd.read_csv(os.path.join(fs.TAB, "hunt09_lowkeff.csv")).set_index("source")
high = pd.read_csv(os.path.join(fs.TAB, "hunt09_highkeff.csv"))
fam = pd.read_csv(os.path.join(fs.TAB, "family_profiles_v3.csv"))
# restrict to the frozen post-quarantine corpus (367 families); the 41
# quarantined generator families in the pre-quarantine profile set are excluded
oc = pd.read_csv(os.path.join(fs.GEO, "frozen", "observational_corpus_v2.csv"),
                 usecols=["dataset_family"])
fam = fam[fam.dataset_family.isin(set(oc.dataset_family))]

items = [
    ("recorded corpus families\n(367, median and IQR)", None, fs.BLUE),
    ("additive: sums, tallies", float(gate.loc[["sum2", "sum3", "count"]].keff.mean()), fs.ORANGE),
    ("exact products, 2 factors", float(gate.loc["prod2"].keff), fs.VERM),
    ("exact products, 4 factors", float(gate.loc["prod4"].keff), fs.VERM),
    ("crystallographic multiplicities\n($n$=11,240)", float(lowk.iloc[0].keff_mean), fs.VERM),
    ("atomic degeneracy products\n($n$=505)", float(lowk.iloc[1].keff_mean), fs.VERM),
    ("graph automorphism orders\n(max)", float(high.keff.max()), fs.VERM),
]

fig, ax = plt.subplots(figsize=(fs.FULL * 0.78, 2.6))
y = np.arange(len(items))
med = fam.keff.median()
q1, q3 = fam.keff.quantile([0.25, 0.75])
ax.errorbar([med], [0], xerr=[[med - q1], [q3 - med]], fmt="o", ms=6,
            color=fs.BLUE, capsize=3.5)
for yy, (lbl, v, col) in zip(y[1:], items[1:]):
    ax.scatter([v], [yy], marker="D", s=34, color=col, zorder=5)
    ax.text(v, yy - 0.25, f"{v:.2f}", ha="center", fontsize=7.5, color=col)
ax.text(q3 + 0.06, 0, f"median {med:.2f}", va="center", ha="left",
        fontsize=7.5, color=fs.BLUE)
ax.axvline(1.0, color="#444444", lw=0.8, ls=":")
ax.text(0.965, 3.0, "generic integer\n($k_{\\mathrm{eff}}=1$)",
        fontsize=7, color="#444444", ha="right", va="center")
ax.set_yticks(y, [i[0] for i in items], fontsize=7.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
ax.set_xlabel("effective factor count $k_{\\mathrm{eff}}$ (log scale)")
ax.set_title("exact, unrounded multiplicative integers are the\n"
             "shape instrument's positive regime")
fs.save(fig, "fig6_exact_contrast")
