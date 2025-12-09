#!/usr/bin/env python3
"""
Predict beam spot size at the BNB target vs Q875 current.

Inputs:
  - 1_twiss_outputs/twiss_Q875_*.tfs
  - 3_results/csv/emittance_vs_Q875_current.csv
  - 3_results/csv/momentum_spread_vs_Q873_current_clean.csv
  - measured σ at MW873/875/876 for each Q875 setting

Outputs:
  - 3_results/csv/target_sigma_prediction_scaled_D_Q875.csv
"""

import os
import math
import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
TFS_DIR = os.path.join(BASE, "..", "1_twiss_outputs")
CSV_DIR = os.path.join(BASE, "..", "3_results", "csv")
os.makedirs(CSV_DIR, exist_ok=True)

EMI_CSV = os.path.join(CSV_DIR, "emittance_vs_Q875_current.csv")
MOM_CSV = os.path.join(CSV_DIR, "momentum_spread_vs_Q873_current_clean.csv")
OUT_CSV = os.path.join(CSV_DIR, "target_sigma_prediction_scaled_D_Q875.csv")

# Q875 scan definition (must match analyze script)
ORDER = ["0", "+7", "+14", "+21", "+28"]
XLABELS = {
    "0":   "Q875 Baseline",
    "+7":  "Q875 +7A",
    "+14": "Q875 +14A",
    "+21": "Q875 +21A",
    "+28": "Q875 +28A",
}
MONITORS = ["MW873", "MW875", "MW876"]

# ### TODO: update TFS filenames if needed
TFS_FILES = {
    "0":   "twiss_Q875_default.tfs",
    "+7":  "twiss_Q875_plus7A.tfs",
    "+14": "twiss_Q875_plus14A.tfs",
    "+21": "twiss_Q875_plus21A.tfs",
    "+28": "twiss_Q875_plus28A.tfs",
}

# Measured σ for Q875 (same content as in analyze script)
MEAS_DATA_Q875 = {
    # "Q875 Baseline": [...],
    # ...
}

if not MEAS_DATA_Q875:
    raise RuntimeError("Fill MEAS_DATA_Q875 with your measured sigma values before running.")

MEAS_DF = pd.DataFrame.from_dict(
    MEAS_DATA_Q875,
    orient="index",
    columns=["873 H", "873 V", "875 H", "875 V", "876 H", "876 V"]
)

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

def pick_target_row(df, s_nominal=207.0):
    for name in ["MTGT", "MWTGT", "TARGET", "TGT", "MTGT1"]:
        m = df[df["NAME"] == name]
        if not m.empty:
            return m.iloc[0]
    if "S" in df.columns:
        idx = (df["S"] - s_nominal).abs().idxmin()
        return df.loc[idx]
    raise RuntimeError("Could not pick target row from TFS.")

def median_scale(est_list, mod_list):
    ratios = [
        e / m for e, m in zip(est_list, mod_list)
        if (m is not None and abs(m) > 1e-12)
    ]
    return float(np.median(ratios)) if ratios else 1.0

# σδ from Q873 baseline
mom = pd.read_csv(MOM_CSV, index_col=0)
if "Q873 Baseline" not in mom.index:
    raise RuntimeError("Q873 Baseline row not found in momentum_spread_vs_Q873_current_clean.csv")
if "sigma_delta_x [rms]" in mom.columns:
    SIGMA_DELTA = float(mom.loc["Q873 Baseline", "sigma_delta_x [rms]"])
else:
    col = next(c for c in mom.columns if "sigma_delta_x" in c.lower())
    SIGMA_DELTA = float(mom.loc["Q873 Baseline", col])
print(f"Using σδ (from Q873 baseline) = {SIGMA_DELTA:.3e}")

emi = pd.read_csv(EMI_CSV, index_col=0)

rows_out = []

for key in ORDER:
    label = XLABELS[key]
    if label not in emi.index:
        raise KeyError(f"Label '{label}' not found in {EMI_CSV}")
    if label not in MEAS_DF.index:
        raise KeyError(f"Measured σ row '{label}' missing in MEAS_DATA_Q875")

    eps_x = float(emi.loc[label, "eps_x [m rad]"])
    eps_y = float(emi.loc[label, "eps_y [m rad]"])

    tfname = TFS_FILES[key]
    tpath = os.path.join(TFS_DIR, tfname)
    df = read_tfs(tpath)
    mw_rows = get_mw_rows(df)

    meas = MEAS_DF.loc[label]

    est_dx = []
    mod_dx = []

    for w, (colH, _) in zip(MONITORS, [("873 H", "873 V"), ("875 H", "875 V"), ("876 H", "876 V")]):
        row = mw_rows[w]
        betx = float(row["BETX"])
        Dx_model = float(row["DX"])

        sx_m = float(meas[colH]) * 1e-3
        base = max(sx_m * sx_m - eps_x * betx, 0.0)

        if SIGMA_DELTA > 0.0:
            Dx_est = math.sqrt(base) / SIGMA_DELTA if base > 0 else 0.0
        else:
            Dx_est = 0.0

        if abs(Dx_model) > 1e-12:
            Dx_est = math.copysign(Dx_est, Dx_model)

        est_dx.append(Dx_est)
        mod_dx.append(Dx_model)

    kx = median_scale(est_dx, mod_dx)

    tgt = pick_target_row(df)
    betx_tgt = float(tgt["BETX"])
    bety_tgt = float(tgt["BETY"])
    Dx_tgt_model = float(tgt["DX"])
    Dy_tgt_model = float(tgt["DY"]) if "DY" in tgt.index else 0.0

    Dx_tgt_scaled = kx * Dx_tgt_model
    Dy_tgt_scaled = Dy_tgt_model

    sigx_tgt_mm = math.sqrt(max(eps_x * betx_tgt + (SIGMA_DELTA * Dx_tgt_scaled) ** 2, 0.0)) * 1e3
    sigy_tgt_mm = math.sqrt(max(eps_y * bety_tgt + (SIGMA_DELTA * Dy_tgt_scaled) ** 2, 0.0)) * 1e3
    area_mm2 = sigx_tgt_mm * sigy_tgt_mm

    rows_out.append({
        "Label": label,
        "sigx_mm": sigx_tgt_mm,
        "sigy_mm": sigy_tgt_mm,
        "area_mm2": area_mm2,
        "kx": kx,
        "BETX_tgt": betx_tgt,
        "BETY_tgt": bety_tgt,
        "DX_tgt_model": Dx_tgt_model,
        "DX_tgt_scaled": Dx_tgt_scaled,
        "DY_tgt_model": Dy_tgt_model,
        "DY_tgt_scaled": Dy_tgt_scaled,
        "S_tgt": float(tgt["S"]) if "S" in tgt.index else math.nan,
    })

out = pd.DataFrame(rows_out).set_index("Label")
out.to_csv(OUT_CSV)
print(f"Wrote {OUT_CSV}")
