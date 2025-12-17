#!/usr/bin/env python3
"""
Exact conversion of Q873 Jupyter notebook → Python script.
Reproduces ALL sigma plots (measured vs predicted) exactly.
"""

from io import StringIO
import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
TWISS_DIR = "../1_twiss_outputs"

order = ["-30","-24","-18","-12","-6","0","+6","+12","+18","+24","+30"]

xlabels = {
    "-30":"Q873 −30A","-24":"Q873 −24A","-18":"Q873 −18A","-12":"Q873 −12A","-6":"Q873 −6A",
    "0":"Q873 Baseline","+6":"Q873 +6A","+12":"Q873 +12A","+18":"Q873 +18A","+24":"Q873 +24A","+30":"Q873 +30A"
}

monitors = ["MW873","MW875","MW876"]

# ============================================================
# Measured σ table (authoritative, from notebook)
# ============================================================
meas_data = {
    "Q873 − 30A":    [2.30,5.05,3.56,4.50,2.29,4.40],
    "Q873 − 24A":    [2.20,4.95,3.40,3.84,2.44,3.60],
    "Q873 − 18A":    [2.29,5.01,3.81,3.20,2.78,2.89],
    "Q873 − 12A":    [2.19,4.97,3.76,2.58,2.84,2.21],
    "Q873 − 6A":     [2.18,5.00,3.92,1.95,0.745,1.56],
    "Q873 Baseline": [2.293,4.959,4.957,1.371,3.771,0.9685],
    "Q873 + 6A":     [2.22,5.01,3.92,4.22,3.60,0.538],
    "Q873 + 12A":    [2.30,5.01,4.62,0.487,4.04,0.950],
    "Q873 + 18A":    [2.32,5.03,4.81,0.918,4.32,1.64],
    "Q873 + 24A":    [2.33,5.05,5.10,1.54,4.37,2.37],
    "Q873 + 30A":    [2.30,5.02,5.14,2.23,4.67,3.07],
}

meas_df = pd.DataFrame.from_dict(
    meas_data, orient="index",
    columns=["873 H","873 V","875 H","875 V","876 H","876 V"]
)

def find_meas(label):
    tgt = label.replace(" ", "")
    for idx in meas_df.index:
        if idx.replace(" ", "") == tgt:
            return meas_df.loc[idx]
    raise KeyError(label)

# ============================================================
# Robust TFS reader
# ============================================================
def read_tfs(path):
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    h = next(i for i,l in enumerate(lines) if l.lstrip().startswith("*"))
    cols = lines[h].replace("*","").split()

    start = h + 1
    if start < len(lines) and lines[start].lstrip().startswith("$"):
        start += 1

    data = [
        l for l in lines[start:]
        if l.strip() and not l.lstrip().startswith(("@","#","$","*"))
    ]

    df = pd.read_csv(
        StringIO("".join(data)),
        sep=r"\s+",
        names=cols,
        engine="python"
    )

    df.columns = [c.upper() for c in df.columns]
    df["NAME"] = (
        df["NAME"].astype(str)
        .str.upper()
        .str.strip('"').str.strip("'")
        .str.replace(r"[:\.].*$","",regex=True)
    )
    return df

def get_rows(df):
    out = {}
    for w in monitors:
        r = df[df["NAME"].str.contains(w)]
        out[w] = r.iloc[0]
    return out

# ============================================================
# Fit ε and σδ at baseline
# ============================================================
df0 = read_tfs(os.path.join(TWISS_DIR, "twiss_Q873_default.tfs"))
rows0 = get_rows(df0)
base = find_meas("Q873 Baseline")

def fit_plane(plane):
    beta = "BETX" if plane=="H" else "BETY"
    disp = "DX" if plane=="H" else "DY"
    y, X = [], []
    for w,s in zip(monitors,["873","875","876"]):
        sig = base[f"{s} {plane}"] * 1e-3
        y.append(sig**2)
        X.append([rows0[w][beta], rows0[w][disp]**2])
    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    return p[0], max(p[1], 0.0)

eps_x, sdel2_x = fit_plane("H")
eps_y, sdel2_y = fit_plane("V")

print("Baseline fit:")
print(f"  H: eps={eps_x:.3e}, sigma_delta={math.sqrt(sdel2_x):.3e}")
print(f"  V: eps={eps_y:.3e}, sigma_delta={math.sqrt(sdel2_y):.3e} (expect ~0)")

# ============================================================
# Sweep currents
# ============================================================
sigma_pred = {(w,p):[] for w in monitors for p in "HV"}
sigma_meas = {(w,p):[] for w in monitors for p in "HV"}
xs = []

for k in order:
    fn = f"twiss_Q873_{k}A_flipPol.tfs" if k!="0" else "twiss_Q873_default.tfs"
    path = os.path.join(TWISS_DIR, fn)
    if not os.path.exists(path):
        continue

    df = read_tfs(path)
    rows = get_rows(df)
    xs.append(int(k))

    m = find_meas(xlabels[k])
    for w,s in zip(monitors,["873","875","876"]):
        bx, dx = rows[w]["BETX"], rows[w]["DX"]
        by, dy = rows[w]["BETY"], rows[w]["DY"]

        sigma_pred[(w,"H")].append(math.sqrt(eps_x*bx + sdel2_x*dx*dx)*1e3)
        sigma_pred[(w,"V")].append(math.sqrt(eps_y*by + sdel2_y*dy*dy)*1e3)

        sigma_meas[(w,"H")].append(m[f"{s} H"])
        sigma_meas[(w,"V")].append(m[f"{s} V"])

# ============================================================
# Plot helpers
# ============================================================
def rmse(a,b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a-b)**2)))

def tick_labels(xs):
    return [xlabels[f"{x:+d}" if x!=0 else "0"] for x in xs]

xticks = tick_labels(xs)

# ============================================================
# Plot 1: measured vs predicted (one panel per wire)
# ============================================================
fig, axs = plt.subplots(3,1,figsize=(11,10),sharex=True)

for ax,w in zip(axs,monitors):
    ax.plot(xs, sigma_meas[(w,"H")], "-o", label="H meas")
    ax.plot(xs, sigma_pred[(w,"H")], "--o", label="H pred")
    ax.plot(xs, sigma_meas[(w,"V")], "-^", label="V meas")
    ax.plot(xs, sigma_pred[(w,"V")], "--^", label="V pred")
    ax.set_title(
        f"{w} (RMSE H={rmse(sigma_meas[(w,'H')],sigma_pred[(w,'H')]):.2f}, "
        f"V={rmse(sigma_meas[(w,'V')],sigma_pred[(w,'V')]):.2f})"
    )
    ax.grid(True, ls=":")
    ax.set_ylabel("σ [mm]")

axs[-1].set_xticks(xs)
axs[-1].set_xticklabels(xticks, rotation=30, ha="right")
axs[-1].set_xlabel("Q873 Current Setting")

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# ============================================================
# Plot 2: overlay panels with shared y-limits
# ============================================================
fig, axes = plt.subplots(3,1,figsize=(11,10),sharex=True)
lo, hi = 1e9, -1e9

for ax,w in zip(axes,monitors):
    y = (
        sigma_meas[(w,"H")] + sigma_pred[(w,"H")] +
        sigma_meas[(w,"V")] + sigma_pred[(w,"V")]
    )
    lo, hi = min(lo,min(y)), max(hi,max(y))

    ax.plot(xs, sigma_meas[(w,"H")], "-o", label="H meas")
    ax.plot(xs, sigma_pred[(w,"H")], "--o", label="H pred")
    ax.plot(xs, sigma_meas[(w,"V")], "-^", label="V meas")
    ax.plot(xs, sigma_pred[(w,"V")], "--^", label="V pred")
    ax.grid(True, ls=":")
    ax.set_ylabel("σ [mm]")
    ax.set_title(w)

pad = 0.1*(hi-lo)
for ax in axes:
    ax.set_ylim(lo-pad, hi+pad)

axes[-1].set_xticks(xs)
axes[-1].set_xticklabels(xticks, rotation=30, ha="right")
axes[-1].set_xlabel("Q873 Current Setting")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
fig.suptitle("Beam-spot σ vs Q873 Current — Measured (solid) vs Predicted (dashed)")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()
