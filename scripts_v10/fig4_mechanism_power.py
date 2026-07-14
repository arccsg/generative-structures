"""Figure 4: recording destroys the shape window, and the null is powered.

(a) Detection AUC for exact composite products against perturbation:
grid coarsening (grid 2 -> 1000) and the minimal additive perturbation
n -> n+1 (one additive unit collapses detection to below chance).
(b) Quotient-level spike-in: fraction of the six host channels recovered
(replicate-median core k_eff above the channel's own f=0 bootstrap 95th
percentile) vs mixture fraction, by injection tier. Host-matched tiers are
recovered 6/6 at a 5% mixture -- the negative result is not underpowered.

Sources (frozen, hashed): tables/hard10_degradation.csv (run hard10),
power13_quotient_spikein.csv (run power13).
"""
import os

import numpy as np
import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt

deg = pd.read_csv(os.path.join(fs.TAB, "hard10_degradation.csv"))
spk = pd.read_csv(os.path.join(fs.TAB, "power13_quotient_spikein.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.FULL, 2.9))

g = deg[deg.family == "grid"]
none = deg[deg.family == "none"].auc_same_perturbed.iloc[0]
ax1.plot([1] + g.param.tolist(), [none] + g.auc_same_perturbed.tolist(),
         marker="o", ms=4.5, color=fs.BLUE, label="decimal grid of width $g$")
add1 = deg[(deg.family == "additive") & (deg.param == 1)].auc_same_perturbed.iloc[0]
ax1.scatter([1], [add1], marker="D", s=42, color=fs.VERM, zorder=5,
            label="$n \\to n{+}1$ (one additive unit)")
ax1.annotate(f"AUC {add1:.2f}", (1, add1), textcoords="offset points",
             xytext=(8, -3), fontsize=7.5, color=fs.VERM)
ax1.annotate(f"no grid: {none:.2f}", (1, none), textcoords="offset points",
             xytext=(6, 3), fontsize=7.5, color=fs.BLUE)
ax1.axhline(0.5, color="#444444", lw=0.8, ls=":")
ax1.text(600, 0.507, "chance", fontsize=7, color="#444444")
ax1.set_xscale("log")
ax1.set_xlabel("perturbation (grid width $g$; diamond: $+1$)")
ax1.set_ylabel("detection AUC (products vs matched null)")
ax1.set_ylim(0.38, 0.92)
ax1.legend(loc="upper right", frameon=False)
ax1.set_title("(a) one additive or quantizing step empties the window")

tiers = [("b_magmatch", "host-matched (grid, magnitude)", fs.BLUE, "o", "-"),
         ("c_basematch", "adversarial (+ small-prime joint)", fs.VERM, "s", "--"),
         ("a_naive", "naive (off-grid; confounded)", fs.GRAY, "^", ":")]
# the f=0 bootstrap threshold lives on the tier="none" rows, per channel
q95 = spk[spk.tier == "none"].groupby("dataset_id").keff_core_q95.first()
for tier, lbl, col, mk, ls in tiers:
    sub = spk[spk.tier == tier]
    med = sub.groupby(["f", "dataset_id"]).keff_core.median().reset_index()
    med["rec"] = med.keff_core > med.dataset_id.map(q95)
    rec = med.groupby("f").rec.mean()
    rec = rec[rec.index > 0]
    ax2.plot(rec.index * 100, rec.values, marker=mk, ms=4.5, color=col,
             ls=ls, label=lbl)
    print(tier, rec.round(2).to_dict())
ax2.set_xscale("log")
ax2.set_xticks([1, 5, 10, 25], ["1%", "5%", "10%", "25%"])
ax2.set_xlabel("mixture fraction $f$ (share of records replaced)")
ax2.set_ylabel("channels recovered (of 6)")
ax2.set_yticks([0, 1 / 3, 2 / 3, 1.0], ["0/6", "2/6", "4/6", "6/6"])
ax2.set_ylim(-0.06, 1.09)
ax2.legend(loc="center right", frameon=False, fontsize=7)
ax2.set_title("(b) the surviving alternative is recovered at 5%")
fs.save(fig, "fig4_destruction_power")
