#!/usr/bin/env python3
"""
Analyze emittance and momentum spread vs Q874 current.

Outputs:
  3_results/csv/emittance_vs_Q874_current.csv
  3_results/figures/emittance_vs_Q874_current.png
"""

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(BASE, "..", "3_results", "csv")
FIG  = os.path.join(BASE, "..", "3_results", "figures")
os.makedirs(CSV, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# -------------------------------
# Q874 CURRENT SETTINGS
# -------------------------------
order = ["-9", "-6", "-3", "0", "+3", "+6", "+9"]
xlabels = {k: f"Q874 {k}A" if k!="0" else "Q874 Baseline" for k in order}
monitors = ["MW873","MW875","MW876"]

# -------------------------------
# ⚠️ INSERT YOUR MEASURED Q874 σ TABLE HERE
# -------------------------------
meas_data = {
    # "Q874 -9A": [873H,873V,875H,875V,876H,876V],
}

# -------------------------------
# TFS FILE PICKER
# -------------------------------
FILES = {k: f"twiss_Q874_{k}A.tfs" if k!="0" else "twiss_Q874_default.tfs" for k in order}

# -------------------------------
# FIT FUNCTION
# -------------------------------
def fit_eps_sdel(sig_mm, beta, disp):
    y = np.array([(s*1e-3)**2 for s in sig_mm])
    X = np.column_stack([beta, np.array(disp)**2])
    (eps, sdel2), *_ = np.linalg.lstsq(X, y, rcond=None)
    return max(eps,0), max(sdel2,0)

# -------------------------------
# MAIN LOOP
# -------------------------------
results = []

for k in order:
    df = pd.read_csv(FILES[k], sep=r"\s+", comment="#")
    meas = meas_data[xlabels[k]]

    betx = df.loc[df.NAME.str.contains("MW"), "BETX"].values
    bety = df.loc[df.NAME.str.contains("MW"), "BETY"].values
    dx   = df.loc[df.NAME.str.contains("MW"), "DX"].values
    dy   = df.loc[df.NAME.str.contains("MW"), "DY"].values

    epsx, sdx2 = fit_eps_sdel(meas[0::2], betx, dx)
    epsy, sdy2 = fit_eps_sdel(meas[1::2], bety, dy)

    results.append({
        "Label": xlabels[k],
        "eps_x": epsx,
        "eps_y": epsy,
        "sigma_delta_x": math.sqrt(sdx2),
        "sigma_delta_y": math.sqrt(sdy2),
    })

res = pd.DataFrame(results).set_index("Label")
res.to_csv(os.path.join(CSV, "emittance_vs_Q874_current.csv"))

res[["eps_x","eps_y"]].mul(1e6).plot(marker="o")
plt.ylabel("Emittance [µm·rad]")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "emittance_vs_Q874_current.png"), dpi=150)
plt.close()

