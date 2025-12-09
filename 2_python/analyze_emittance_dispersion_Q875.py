#!/usr/bin/env python3
"""
Analyze horizontal/vertical emittance and momentum spread vs Q875 current.

Inputs:
  - 1_twiss_outputs/twiss_Q875_*.tfs
  - measured σ at MW873/875/876 for each Q875 setting

Outputs:
  - 3_results/csv/emittance_vs_Q875_current.csv
  - 3_results/figures/emittance_vs_Q875_current.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE    = os.path.dirname(os.path.abspath(__file__))
TFS_DIR = os.path.join(BASE, "..", "1_twiss_outputs")
CSV_DIR = os.path.join(BASE, "..", "3_results", "csv")
FIG_DIR = os.path.join(BASE, "..", "3_results", "figures")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Q875 scan definition
# ----------------------------------------------------------------------
# ### TODO: adjust if your Q875 scan uses different currents
ORDER = ["0", "+7", "+14", "+21", "+28"]
XLABELS = {
    "0":   "Q875 Baseline",
    "+7":  "Q875 +7A",
    "+14": "Q875 +14A",
    "+21": "Q875 +21A",
    "+28": "Q875 +28A",
}
MONITORS = ["MW873", "MW875", "MW876"]

# ### TODO: update these TFS filenames as needed
TFS_FILES = {
    "0":   "twiss_Q875_default.tfs",
    "+7":  "twiss_Q875_plus7A.tfs",
    "+14": "twiss_Q875_plus14A.tfs",
    "+21": "twiss_Q875_plus21A.tfs",
    "+28": "twiss_Q875_plus28A.tfs",
}

# ----------------------------------------------------------------------
# Measured σ at MW873/875/876 for each Q875 setting (mm)
# ----------------------------------------------------------------------
# Columns: [873 H, 873 V, 875 H, 875 V, 876 H, 876 V]
# ### TODO: fill this with your Q875 scan measurements
MEAS_DATA_Q875 = {
    # "Q875 Baseline": [...],
    # "Q875 +7A":      [...],
    # ...
}

if not MEAS_DATA_Q875:
    raise RuntimeError("Fill MEAS_DATA_Q875 with your measured sigma values before running.")

MEAS_DF = pd.DataFrame.from_dict(
    MEAS_DATA_Q875,
    orient="index",
    columns=["873 H", "873 V", "875 H", "875 V", "876 H", "876 V"]
)

# ----------------------------------------------------------------------
# TFS helpers
# ----------------------------------------------------------------------
def read_tfs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("*"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"No '*' header line in {path}")

    import re
    cols = re.sub(r"^\*", "", lines[header_idx]).strip().split()

    start = header_idx + 1
    if start < len(lines) and lines[start].lstrip().startswith("$"):
        start += 1

    data_rows = []
    for ln in lines[start:]:
        if ln.strip() == "" or ln.lstrip().startswith(("@", "#", "*", "$")):
            continue
        data_rows.append(ln)

    from io import StringIO
    df = pd.read_csv(
        StringIO("".join(data_rows)),
        sep=r"\s+",
        header=None,
        names=cols,
        engine="python"
    )

    df.columns = [c.upper() for c in df.columns]
    if "NAME" in df.columns:
        df["NAME"] = (
            df["NAME"].astype(str)
            .str.upper()
            .str.strip()
            .str.strip('"')
            .str.strip("'")
        )
    return df

def get_mw_rows(df):
    out = {}
    for w in MONITORS:
        m = df[df["NAME"] == w]
        if m.empty:
            m = df[df["NAME"].str.contains(w, na=False)]
        if m.empty:
            raise RuntimeError(f"{w} not found in TFS.")
        out[w] = m.iloc[0]
    return out

def fit_eps_sdel(sig_mm, beta, disp):
    y = np.array([(s * 1e-3) ** 2 for s in sig_mm], float)
    X = np.column_stack([np.array(beta, float), np.array(disp, float) ** 2])
    (eps, sdel2), *_ = np.linalg.lstsq(X, y, rcond=None)
    eps = max(float(eps), 0.0)
    sdel2 = max(float(sdel2), 0.0)
    return eps, sdel2

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
rows_out = []

for key in ORDER:
    label = XLABELS[key]
    if label not in MEAS_DF.index:
        raise KeyError(f"Measured σ row '{label}' missing in MEAS_DATA_Q875")

    tfname = TFS_FILES[key]
    tpath = os.path.join(TFS_DIR, tfname)
    df = read_tfs(tpath)
    rows = get_mw_rows(df)

    betx = [float(rows[w]["BETX"]) for w in MONITORS]
    bety = [float(rows[w]["BETY"]) for w in MONITORS]
    dx   = [float(rows[w]["DX"])   for w in MONITORS]
    dy   = [float(rows[w]["DY"])   for w in MONITORS]

    meas = MEAS_DF.loc[label]

    sigx_mm = [meas["873 H"], meas["875 H"], meas["876 H"]]
    sigy_mm = [meas["873 V"], meas["875 V"], meas["876 V"]]

    epsx, sdx2 = fit_eps_sdel(sigx_mm, betx, dx)
    epsy, sdy2 = fit_eps_sdel(sigy_mm, bety, dy)

    rows_out.append({
        "Label": label,
        "eps_x [m rad]": epsx,
        "eps_y [m rad]": epsy,
        "sigma_delta_x": math.sqrt(sdx2),
        "sigma_delta_y": math.sqrt(sdy2),
    })

res = pd.DataFrame(rows_out).set_index("Label")
out_csv = os.path.join(CSV_DIR, "emittance_vs_Q875_current.csv")
res.to_csv(out_csv)
print(f"Wrote {out_csv}")

# Plot
labels = [XLABELS[k] for k in ORDER]

plt.figure(figsize=(8, 5))
plt.plot(labels, res["eps_x [m rad]"] * 1e6, marker="o", label="εx")
plt.plot(labels, res["eps_y [m rad]"] * 1e6, marker="s", label="εy")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Emittance [µm·rad]")
plt.xlabel("Q875 current setting")
plt.title("Fitted Emittance vs Q875 Current")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "emittance_vs_Q875_current.png")
plt.savefig(out_fig, dpi=150)
plt.close()
print(f"Wrote {out_fig}")
