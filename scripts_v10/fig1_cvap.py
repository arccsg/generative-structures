"""Figure 1: the CVAP false alarm and its five-grid attribution.

Left: last-digit distribution (records >= 1000, the Beber-Scacco screen) for
the modeled CVAP series vs enumerated county returns, with the frozen chi^2
values. Right: residue-class excess over the magnitude-matched uniform null at
moduli 25 and 100 -- the direct signature of rounding to the nearest five.

Sources (frozen, hashed): tables/vote14_panel.csv, vote14_beber_scacco.csv,
vote14_grid_probe.csv (lprofile-geography, run vote14, config
run_config_vote14.json).
"""
import os

import numpy as np
import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt

panel = pd.read_csv(os.path.join(fs.TAB, "vote14_panel.csv"))
bs = pd.read_csv(os.path.join(fs.TAB, "vote14_beber_scacco.csv")).set_index("series")
gp = pd.read_csv(os.path.join(fs.TAB, "vote14_grid_probe.csv")).set_index("series")

SERIES = [
    ("cvap_county_modeled", dict(level="county", contest="cvap_2018_2022"),
     "CVAP (modeled, published)", fs.VERM),
    ("county_pres_total_2024", dict(level="county", contest="president_2024",
                                    candidate="total"),
     "county returns 2024 (enumerated)", fs.BLUE),
    ("county_pres_total_2020", dict(level="county", contest="president_2020",
                                    candidate="total"),
     "county returns 2020 (enumerated)", fs.GRAY),
]


def last_digit_freq(flt):
    m = pd.Series(True, index=panel.index)
    for k, v in flt.items():
        m &= panel[k] == v
    w = panel.loc[m, "count"].to_numpy(dtype=np.int64)
    w = w[w >= 1000]                      # the standard magnitude screen
    digits = w % 10
    return np.bincount(digits, minlength=10) / len(w), len(w)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.FULL, 2.7),
                               gridspec_kw=dict(width_ratios=[1.35, 1]))

x = np.arange(10)
markers = {"cvap_county_modeled": "o", "county_pres_total_2024": "s",
           "county_pres_total_2020": "^"}
for name, flt, label, color in SERIES:
    f, n = last_digit_freq(flt)
    chi2 = bs.loc[name, "bs_chi2"]
    lbl = (f"{label}\n$\\chi^2_9$ = {chi2:,.0f}" if chi2 > 100
           else f"{label}  ($\\chi^2_9$ = {chi2:.1f}, passes)")
    ax1.plot(x, f, marker=markers[name], ms=4.5, color=color, label=lbl,
             linestyle="-" if name.startswith("cvap") else "--",
             zorder=3 if name.startswith("cvap") else 2)
ax1.axhline(0.1, color="#444444", lw=0.8, ls=":", zorder=1)
ax1.text(9.35, 0.1, "uniform", va="center", ha="left", fontsize=7,
         color="#444444")
ax1.set_xticks(x)
ax1.set_xlabel("last digit of the published value (records $\\geq$ 1000)")
ax1.set_ylabel("relative frequency")
ax1.set_title("(a) a standard last-digit test fires only on the modeled series")
ax1.legend(loc="upper right", frameon=False, handlelength=1.6)
ax1.set_xlim(-0.5, 11.2)

series_order = ["county_pres_total_2020", "county_pres_total_2024",
                "cvap_county_modeled"]
labels = ["county 2020\n(enum.)", "county 2024\n(enum.)", "CVAP\n(modeled)"]
y = np.arange(len(series_order))
w25 = [gp.loc[s, "tv25_excess"] for s in series_order]
w100 = [gp.loc[s, "tv100_excess"] for s in series_order]
ax2.barh(y + 0.19, w25, height=0.34, color=fs.VERM, label="mod 25")
ax2.barh(y - 0.19, w100, height=0.34, color=fs.VERM, alpha=0.45,
         label="mod 100", hatch="///", edgecolor=fs.VERM, lw=0)
ax2.axvline(0, color="#444444", lw=0.8)
ax2.set_yticks(y, labels)
ax2.set_xlabel("residue excess vs magnitude-matched null")
ax2.set_title("(b) the five-grid signature attributes it")
ax2.legend(loc="lower right", frameon=False)
for yy, v in zip(y, w25):
    if v > 0.1:
        ax2.text(v - 0.02, yy + 0.19, f"+{v:.2f}", va="center", ha="right",
                 fontsize= 7.5, color="white", fontweight="bold")

fs.save(fig, "fig1_cvap_false_alarm")
