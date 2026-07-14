"""Figure 5: the ACS margin-of-error localization.

Grid mass (log-share carried by the primes 2 and 5) with 90% bootstrap CIs for
the three county-level cross-sections: the decennial census count, the ACS
5-year point estimate (pre-registered to be grid-marked -- it is not), and the
ACS published margin of error (where the grid mark actually lives).

Source (frozen, hashed): tables/whisp12_acs.csv (run whisp12,
config run_config_whisp12.json).
"""
import os

import numpy as np
import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt

acs = pd.read_csv(os.path.join(fs.TAB, "whisp12_acs.csv")).set_index("cross_section")

rows = [
    ("census2020_count", "decennial census count\n(enumerated, $n$=3144 counties)", fs.BLUE),
    ("acs5yr_estimate", "ACS 5-yr point estimate\n(modeled, $n$=3144)", fs.GRAY),
    ("acs5yr_MOE", "ACS margin of error\n(published, $n$=125)", fs.VERM),
]

fig, ax = plt.subplots(figsize=(fs.FULL * 0.72, 2.3))
y = np.arange(len(rows))
for yy, (key, lbl, col) in zip(y, rows):
    r = acs.loc[key]
    ax.errorbar(r.rounding_mass, yy,
                xerr=[[r.rounding_mass - r.rounding_mass_lo],
                      [r.rounding_mass_hi - r.rounding_mass]],
                fmt="o", ms=6, color=col, capsize=3.5, lw=1.4)
    ax.text(r.rounding_mass_hi + 0.004, yy, f"{r.rounding_mass:.3f}",
            va="center", ha="left", fontsize=7.5, color=col)
ax.axvline(acs.loc["census2020_count", "rounding_mass"], color=fs.BLUE,
           lw=0.8, ls=":", alpha=0.7)
ax.set_yticks(y, [r[1] for r in rows], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("grid mass $\\rho$ (log-share of primes 2 and 5), 90% CI")
ax.set_title("the grid mark is on the margin of error, not the estimate")
ax.set_xlim(0.095, 0.235)
fs.save(fig, "fig5_acs_moe")
