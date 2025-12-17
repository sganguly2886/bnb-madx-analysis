#!/usr/bin/env python3
"""
Reproduce optics vs Q873 current plots exactly as in the notebook.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tfs_utils import read_tfs

# ------------------------------------------------------------
# Configuration (matches notebook)
# ------------------------------------------------------------
TWISS_DIR = "../1_twiss_outputs"
OUTDIR = "../3_results/figures"
os.makedirs(OUTDIR, exist_ok=True)

order = ["-30","-24","-18","-12","-6","0","+6","+12","+18","+24","+30"]

xlabels = {
    "-30":"Q873 −30A","-24":"Q873 −24A","-18":"Q873 −18A","-12":"Q873 −12A",
    "-6":"Q873 −6A","0":"Q873 Baseline","+6":"Q873 +6A","+12":"Q873 +12A",
    "+18":"Q873 +18A","+24":"Q873 +24A","+30":"Q873 +30A"
}

monitors = ["MW873","MW875","MW876"]

OLD_FILES = {
    "-30": "twiss_Q873_minus30A.tfs",
    "-24": "twiss_Q873_minus24A.tfs",
    "-18": "twiss_Q873_minus18A.tfs",
    "-12": "twiss_Q873_minus12A.tfs",
    "-6" : "twiss_Q873_minus6A.tfs",
    "0"  : "twiss_Q873_default.tfs",
    "+6" : "twiss_Q873_plus6A.tfs",
    "+12": "twiss_Q873_plus12A.tfs",
    "+18": "twiss_Q873_plus18A.tfs",
    "+24": "twiss_Q873_plus24A.tfs",
    "+30": "twiss_Q873_plus30A.tfs",
}

def pick_file(key):
    if key == "0":
        return "twiss_Q873_default.tfs"
    flip = f"twiss_Q873_{key}A_flipPol.tfs"
    if os.path.exists(os.path.join(TWISS_DIR, flip)):
        return flip
    return OLD_FILES[key]

def get_mw_rows(df):
    out = {}
    for w in monitors:
        m = df[df["NAME"] == w]
        if m.empty:
            m = df[df["NAME"].str.contains(w, na=False)]
        out[w] = m.iloc[0]
    return out

# ------------------------------------------------------------
# Load optics
# ------------------------------------------------------------
betx = {w: [] for w in monitors}
dx   = {w: [] for w in monitors}
dy   = {w: [] for w in monitors}

labels_used = []

for key in order:
    fn = pick_file(key)
    path = os.path.join(TWISS_DIR, fn)
    if not os.path.exists(path):
        print("Skipping missing:", fn)
        continue

    df = read_tfs(path)
    rows = get_mw_rows(df)
    labels_used.append(key)

    for w in monitors:
        betx[w].append(float(rows[w]["BETX"]))
        dx[w].append(float(rows[w]["DX"]))
        dy[w].append(float(rows[w]["DY"]))

xs = np.array([int(k) for k in labels_used])
xticks = [xlabels[k] for k in labels_used]

# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def plot_quantity(data, ylabel, title, fname):
    plt.figure(figsize=(10,6))
    for w in monitors:
        plt.plot(xs, data[w], marker="o", label=w)
    plt.xticks(xs, xticks, rotation=30, ha="right")
    plt.xlabel("Q873 Current Setting")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.legend(title="Wire")
    plt.tight_layout()
    out = os.path.join(OUTDIR, fname)
    plt.savefig(out, dpi=150)
    print("Saved:", out)
    plt.show()

# ------------------------------------------------------------
# Make plots (matches notebook)
# ------------------------------------------------------------
plot_quantity(
    betx,
    "BETX [m]",
    "BETX at MW873 / MW875 / MW876 vs Q873 Current",
    "Q873_BETX_vs_current.png"
)

plot_quantity(
    dx,
    "Dispersion Dₓ [m]",
    "Horizontal dispersion Dₓ vs Q873 Current",
    "Q873_DX_vs_current.png"
)

plot_quantity(
    dy,
    "Dispersion Dᵧ [m]",
    "Vertical dispersion Dᵧ vs Q873 Current",
    "Q873_DY_vs_current.png"
)
