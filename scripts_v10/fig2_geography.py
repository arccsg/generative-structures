"""Figure 2: the apparent (L1, L3) geography, and what survives controls.

(a) Raw channel-level geography: recorded channels (density), the
magnitude-conditioned null (uniform-integer pseudo-channels), exact
multiplicative constructions, the same constructions after one decimal-grid
rounding, and additive controls. Channels whose records mostly have < 3
prime-power parts (L3 structurally zero) are marked separately.
(b) Out-of-family domain information in the coordinates at each control stage
(grouped 5-fold by dataset-family; macro one-vs-rest AUC; family-bootstrap
90% CI) -- the map's apparent structure never carried domain information.

Sources: tables/geo15_geography_channel.csv, geo15_controlled.csv
(exploratory Build geo15, seed 20260715, config run_config_geo15.json).
"""
import os

import numpy as np
import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt

geo = pd.read_csv(os.path.join(fs.TAB, "geo15_geography_channel.csv"))
ctl = pd.read_csv(os.path.join(fs.TAB, "geo15_controlled.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.FULL, 3.1),
                               gridspec_kw=dict(width_ratios=[1.45, 1]))

obs = geo[geo.group == "observational"].dropna(subset=["L1", "L3"])
deep = obs[obs.k_parts_mean >= 2.5]
shallow = obs[obs.k_parts_mean < 2.5]
ax1.hexbin(deep.L1, deep.L3, gridsize=48, cmap="Blues", mincnt=1,
           bins="log", extent=(0.2, 1.0, 0, 0.35), linewidths=0.1, zorder=1)
ax1.scatter(shallow.L1, shallow.L3, s=3, marker="x", color="#8AB8D8",
            lw=0.5, zorder=2,
            label="recorded, most records $<3$ parts ($L_3$ structural 0)")

nul = geo[geo.group == "uniform_null"]
nc = nul.groupby(nul.d_mean.astype(int))[["L1", "L3"]].mean()
ax1.plot(nc.L1, nc.L3, color=fs.GREEN, marker="o", ms=3.5, lw=1.4, zorder=4,
         label="uniform-integer null, 2--12 digits")
ax1.annotate("2 digits", (nc.L1.iloc[0], nc.L3.iloc[0]),
             textcoords="offset points", xytext=(4, -9), fontsize=7,
             color=fs.GREEN)
ax1.annotate("12", (nc.L1.iloc[-1], nc.L3.iloc[-1]),
             textcoords="offset points", xytext=(-2, 7), fontsize=7,
             color=fs.GREEN)

ex = geo[geo.domain == "exact"].set_index("group")
labels = {"exact_product_2": "2-factor product", "exact_product_3": "3-factor",
          "exact_product_4": "4-factor"}
for g, lbl in labels.items():
    r = ex.loc[g]
    ax1.scatter(r.L1, r.L3, marker="D", s=26, color=fs.VERM, zorder=5)
    off = {"exact_product_2": (6, -3), "exact_product_3": (-46, 1),
           "exact_product_4": (6, 3)}[g]
    ax1.annotate(lbl, (r.L1, r.L3), textcoords="offset points", xytext=off,
                 fontsize=7, color=fs.VERM)
gr = geo[geo.domain == "grid_demo"]
ax1.scatter(gr.L1, gr.L3, marker="s", s=18, facecolor="none",
            edgecolor=fs.PINK, lw=1.2, zorder=5,
            label="same products after one grid rounding")
ad = geo[geo.domain == "additive"]
ax1.scatter(ad.L1, ad.L3, marker="^", s=26, color=fs.ORANGE, zorder=5,
            label="additive controls (sums, tallies)")
ax1.annotate("$n!$, $\\binom{2n}{n}$: off-scale left\n($L_1\\leq0.15$, Tail 0.73--0.95)",
             xy=(0.205, 0.115), xytext=(0.24, 0.055), fontsize=7, color=fs.VERM,
             arrowprops=dict(arrowstyle="->", color=fs.VERM, lw=0.8))
ax1.set_xlabel("$L_1$ (largest log-share)")
ax1.set_ylabel("$L_3$ (third-largest log-share)")
ax1.set_title("(a) raw geography: recorded data hugs magnitude + grid")
ax1.legend(loc="upper right", frameon=False, fontsize=6.8, handlelength=1.2)
ax1.set_xlim(0.2, 1.0)
ax1.set_ylim(-0.012, 0.35)

order = ["S0_raw", "S1_digit", "S2_deround", "S4_residue"]
chn = pd.read_csv(os.path.join(fs.TAB, "geo15_controlled_channels.csv"))
stage_cols = {"S0_raw": ["L1", "L3"], "S1_digit": ["g15_dL1", "g15_dL3"],
              "S2_deround": ["dL1_der", "dTail_der"],
              "S4_residue": ["g15_dL1_der_resid", "g15_dTail_der_resid"]}
nfam = {k: chn.dropna(subset=v).dataset_family.nunique()
        for k, v in stage_cols.items()}
base_names = ["raw\n$(L_1,L_3)$", "digit-length\nresiduals", "+ $2\\cdot5$\nstrip",
              "+ residue\nconditioning"]
names = [f"{b}\n({nfam[o]} fam.)" for b, o in zip(base_names, order)]
c = ctl.set_index("stage").loc[order]
x = np.arange(len(order))
ax2.errorbar(x, c.macro_auc, yerr=[c.macro_auc - c.auc_lo,
                                   c.auc_hi - c.macro_auc],
             color=fs.BLUE, marker="o", ms=5, capsize=3, lw=1.4)
ax2.axhline(0.5, color="#444444", lw=0.8, ls=":")
ax2.text(len(order) - 0.55, 0.502, "chance", fontsize=7, color="#444444")
ax2.set_xticks(x, names)
ax2.set_ylim(0.28, 0.72)
ax2.set_ylabel("out-of-family domain AUC (macro OVR)")
ax2.set_title("(b) essentially no domain information at any stage")
fs.save(fig, "fig2_geography_raw_controlled")
