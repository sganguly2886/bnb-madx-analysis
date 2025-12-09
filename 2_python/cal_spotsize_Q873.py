#!/usr/bin/env python3
"""
Author: <Your Name>

Target beam–spot prediction for Q873 scan.

This script:
  - Uses 95% normalized emittance vs current from
        emittance95_normalized_pi-mm-mrad.csv
  - Uses measured MW873/875/876 σ vs current
  - Uses MAD-X Twiss files vs Q873 current
  - Fits σδ at baseline and infers effective dispersion at wires
  - Scales model Dx/Dy at the target (~207 m)
  - Predicts σx, σy at the target and writes:
        ../3_results/csv/target_sigma_prediction_scaled_D_Q873.csv
  - Also writes Dx/Dy estimate CSVs and plots.
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "..", "3_results", "csv")
FIG_DIR = os.path.join(BASE_DIR, "..", "3_results", "figures")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ORDER = ["-30","-24","-18","-12","-6","0","+6","+12","+18","+24","+30"]
XLABELS = {
    "-30":"Q873 −30A","-24":"Q873 −24A","-18":"Q873 −18A","-12":"Q873 −12A","-6":"Q873 −6A",
    "0":"Q873 Baseline","+6":"Q873 +6A","+12":"Q873 +12A",
    "+18":"Q873 +18A","+24":"Q873 +24A","+30":"Q873 +30A"
}
MONITORS = ["MW873","MW875","MW876"]

FILES_FLIP = {
    "-30": "twiss_Q873_-30A_flipPol.tfs",
    "-24": "twiss_Q873_-24A_flipPol.tfs",
    "-18": "twiss_Q873_-18A_flipPol.tfs",
    "-12": "twiss_Q873_-12A_flipPol.tfs",
    "-6" : "twiss_Q873_-6A_flipPol.tfs",
    "0"  : "twiss_Q873_default.tfs",
    "+6" : "twiss_Q873_+6A_flipPol.tfs",
    "+12": "twiss_Q873_+12A_flipPol.tfs",
    "+18": "twiss_Q873_+18A_flipPol.tfs",
    "+24": "twiss_Q873_+24A_flipPol.tfs",
    "+30": "twiss_Q873_+30A_flipPol.tfs",
}
FILES_ORIG = {
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

def choose_files():
    chosen = {}
    for k in ORDER:
        f = FILES_FLIP.get(k)
        if f and os.path.exists(f):
            chosen[k] = f
        elif FILES_ORIG.get(k) and os.path.exists(FILES_ORIG[k]):
            chosen[k] = FILES_ORIG[k]
    return chosen

FILES = choose_files()
if "0" not in FILES:
    raise RuntimeError("Baseline 0A TFS not found.")

# Measured σ table (same as in analyze script)
MEAS_DATA = {
    "Q873 − 30A":    [2.30,  5.05,  3.56,  4.50,  2.29,  4.40],
    "Q873 − 24A":    [2.20,  4.95,  3.40,  3.84,  2.44,  3.60],
    "Q873 − 18A":    [2.29,  5.01,  3.81,  3.20,  2.78,  2.89],
    "Q873 − 12A":    [2.19,  4.97,  3.76,  2.58,  2.84,  2.21],
    "Q873 − 6A":     [2.18,  5.00,  3.92,  1.95,  0.745, 1.56],
    "Q873 Baseline": [2.293, 4.959, 4.957, 1.371, 3.771, 0.9685],
    "Q873 + 6A":     [2.22,  5.01,  3.92,  4.22,  3.60,  0.538],
    "Q873 + 12A":    [2.30,  5.01,  4.62,  0.487, 4.04,  0.950],
    "Q873 + 18A":    [2.32,  5.03,  4.81,  0.918, 4.32,  1.64],
    "Q873 + 24A":    [2.33,  5.05,  5.10,  1.54,  4.37,  2.37],
    "Q873 + 30A":    [2.30,  5.02,  5.14,  2.23,  4.67,  3.07],
}
MEAS_DF = pd.DataFrame.from_dict(
    MEAS_DATA, orient="index",
    columns=["873 H","873 V","875 H","875 V","876 H","876 V"]
)
MEAS_DF.index.name = "Setup"

def find_meas_row(label: str) -> pd.Series:
    target = label.replace(" ", "")
    for idx in MEAS_DF.index:
        if idx.replace(" ", "") == target:
            return MEAS_DF.loc[idx]
    raise KeyError(f"Measurement row not found for '{label}'")

# ---- TFS helpers ----
def read_tfs(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("*"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"No '*' header in {path}")
    cols = re.sub(r"^\*", "", lines[header_idx]).strip().split()
    start = header_idx + 1
    if start < len(lines) and lines[start].lstrip().startswith("$"):
        start += 1
    rows = []
    for ln in lines[start:]:
        if ln.strip()=="" or ln.lstrip().startswith(("@","#","*","$")):
            continue
        rows.append(ln)
    from io import StringIO
    df = pd.read_csv(StringIO("".join(rows)),
                     sep=r"\s+", header=None, names=cols, engine="python")
    df.columns = [c.upper() for c in df.columns]
    if "NAME" in df.columns:
        df["NAME"] = (df["NAME"].astype(str).str.upper()
                      .str.strip().str.strip('"').str.strip("'")
                      .str.replace(r"[:\.].*$", "", regex=True))
    return df

def get_rows(df, name_list):
    out = {}
    for n in name_list:
        m = df[df["NAME"] == n]
        if m.empty:
            m = df[df["NAME"].str.startswith(n)]
        out[n] = None if m.empty else m.iloc[0]
    return out

def pick_target_row(df: pd.DataFrame) -> pd.Series:
    for n in ["MTGT","MWTGT","TARGET","TGT","MTGT1"]:
        m = df[df["NAME"] == n]
        if not m.empty:
            return m.iloc[0]
    if "S" in df.columns:
        window = df[(df["S"]>=206.0) & (df["S"]<=208.5)]
        if not window.empty:
            i = (window["S"]-207.0).abs().idxmin()
            return df.loc[i]
        i = (df["S"]-207.0).abs().idxmin()
        return df.loc[i]
    raise RuntimeError("No target row by name or S-range")

# ---- beam βγ and eps conversion ----
p_GeV = 8.83490
m0_GeV = 0.9382720813
gamma = math.sqrt(1.0 + (p_GeV/m0_GeV)**2)
beta = math.sqrt(1.0 - 1.0/gamma**2)
betagamma = beta * gamma

def eps95norm_to_epsrms(eps95_norm_pi_mm_mrad: float) -> float:
    """
    Convert 95% normalized [π·mm·mrad] to ε_rms geom [m·rad]
    via ε_95,norm ≈ 6 (βγ ε_rms).
    """
    return (eps95_norm_pi_mm_mrad / 6.0) * 1e-6 / betagamma

def median_scale(est_list, mod_list):
    ratios = [e/m for e, m in zip(est_list, mod_list)
              if m is not None and abs(m) > 1e-12]
    return float(np.median(ratios)) if ratios else 1.0

def main():
    # --- 1) load 95% normalized emittance vs current (from previous script) ---
    eps95_csv = os.path.join(CSV_DIR, "emittance95_normalized_pi-mm-mrad.csv")
    if not os.path.exists(eps95_csv):
        raise FileNotFoundError(
            f"{eps95_csv} not found. Run analyze_emittance_dispersion_Q873.py first."
        )

    eps_tab = pd.read_csv(eps95_csv, index_col=0)
    # try to find columns for epsx and epsy
    col_ex = [c for c in eps_tab.columns if "x_95_norm" in c.lower()]
    col_ey = [c for c in eps_tab.columns if "y_95_norm" in c.lower()]
    if not col_ex or not col_ey:
        # fallback: first two columns
        col_ex = [eps_tab.columns[0]]
        col_ey = [eps_tab.columns[1]]

    eps95_norm_map_x = {idx: float(eps_tab.loc[idx, col_ex[0]]) for idx in eps_tab.index}
    eps95_norm_map_y = {idx: float(eps_tab.loc[idx, col_ey[0]]) for idx in eps_tab.index}

    epsx_rms_map = {}
    epsy_rms_map = {}
    for k in ORDER:
        lbl = XLABELS[k]
        if lbl in eps95_norm_map_x and lbl in eps95_norm_map_y:
            epsx_rms_map[k] = eps95norm_to_epsrms(eps95_norm_map_x[lbl])
            epsy_rms_map[k] = eps95norm_to_epsrms(eps95_norm_map_y[lbl])

    # --- 2) fit σδ at baseline from horizontal MW σ ---
    df0 = read_tfs(FILES["0"])
    rows0 = get_rows(df0, MONITORS)
    meas0 = find_meas_row("Q873 Baseline")

    Y = []
    X = []
    for w, colH in [("MW873","873 H"),("MW875","875 H"),("MW876","876 H")]:
        r = rows0[w]
        if r is None:
            continue
        sx = float(meas0[colH]) * 1e-3
        bx = float(r["BETX"])
        Dx = float(r["DX"])
        epsx0 = epsx_rms_map["0"]
        Y.append(sx*sx - epsx0*bx)
        X.append([Dx*Dx])

    s2, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
    sigma_delta = math.sqrt(max(float(s2[0]), 0.0))
    sigma_delta_y = sigma_delta

    print(f"βγ = {betagamma:.6f} | fitted σδ = {sigma_delta:.3e}")

    # --- 3) loop currents, infer D at wires, scale model, propagate to target ---
    Dx_est = {w: [] for w in MONITORS}
    Dy_est = {w: [] for w in MONITORS}
    Dx_mod = {w: [] for w in MONITORS}
    Dy_mod = {w: [] for w in MONITORS}
    labels_avail = []
    target_sigmas = []

    for k in ORDER:
        if k not in FILES:
            continue
        df = read_tfs(FILES[k])
        rows = get_rows(df, MONITORS)
        labels_avail.append(k)

        meas = find_meas_row(XLABELS[k])
        ex = epsx_rms_map[k]
        ey = epsy_rms_map[k]

        est_dx, mod_dx = [], []
        est_dy, mod_dy = [], []

        for w, (colH, colV) in zip(MONITORS, [("873 H","873 V"),
                                              ("875 H","875 V"),
                                              ("876 H","876 V")]):
            r = rows[w]
            if r is None:
                Dx_est[w].append(np.nan); Dy_est[w].append(np.nan)
                Dx_mod[w].append(np.nan); Dy_mod[w].append(np.nan)
                continue

            bx = float(r["BETX"]); by = float(r["BETY"])
            Dx_m = float(r["DX"]);  Dy_m = float(r["DY"])
            sx_m = float(meas[colH]) * 1e-3
            sy_m = float(meas[colV]) * 1e-3

            base_x = max(sx_m*sx_m - ex*bx, 0.0)
            base_y = max(sy_m*sx_m - ey*by, 0.0)  # typo-resistant; but Dy~0 anyway

            Dx_e = 0.0 if sigma_delta <= 0 else math.sqrt(base_x)/sigma_delta
            Dy_e = 0.0 if sigma_delta_y <= 0 else math.sqrt(max(
                sy_m*sy_m - ey*by, 0.0)) / sigma_delta_y

            if abs(Dx_m) > 1e-12:
                Dx_e = math.copysign(Dx_e, Dx_m)
            if abs(Dy_m) > 1e-12:
                Dy_e = math.copysign(Dy_e, Dy_m)

            Dx_est[w].append(Dx_e); Dy_est[w].append(Dy_e)
            Dx_mod[w].append(Dx_m); Dy_mod[w].append(Dy_m)
            est_dx.append(Dx_e); mod_dx.append(Dx_m)
            est_dy.append(Dy_e); mod_dy.append(Dy_m)

        kx = median_scale(est_dx, mod_dx)
        ky = median_scale(est_dy, mod_dy)

        tgt = pick_target_row(df)
        Dx_tgt_mod = float(tgt["DX"])
        Dy_tgt_mod = float(tgt["DY"])
        Dx_tgt = Dx_tgt_mod * kx
        Dy_tgt = Dy_tgt_mod * ky

        betx_tgt = float(tgt["BETX"])
        bety_tgt = float(tgt["BETY"])

        sigx_tgt_mm = math.sqrt(max(ex*betx_tgt + (sigma_delta*Dx_tgt)**2, 0.0))*1e3
        sigy_tgt_mm = math.sqrt(max(ey*bety_tgt + (sigma_delta*Dy_tgt)**2, 0.0))*1e3

        target_sigmas.append({
            "label": XLABELS[k],
            "sigx_mm": sigx_tgt_mm,
            "sigy_mm": sigy_tgt_mm,
            "area_mm2": sigx_tgt_mm*sigy_tgt_mm,
            "kx": kx, "ky": ky,
            "Dx_tgt_scaled": Dx_tgt, "Dy_tgt_scaled": Dy_tgt,
            "Dx_tgt_model": Dx_tgt_mod, "Dy_tgt_model": Dy_tgt_mod,
            "BETX_tgt": betx_tgt, "BETY_tgt": bety_tgt,
            "S_tgt": float(tgt["S"]) if "S" in tgt.index else np.nan
        })

    labels = [XLABELS[k] for k in labels_avail]
    xi = np.arange(len(labels))

    dx_table = pd.DataFrame({
        "Dx_est_MW873": Dx_est["MW873"],
        "Dx_est_MW875": Dx_est["MW875"],
        "Dx_est_MW876": Dx_est["MW876"],
        "Dx_model_MW873": Dx_mod["MW873"],
        "Dx_model_MW875": Dx_mod["MW875"],
        "Dx_model_MW876": Dx_mod["MW876"],
    }, index=labels)

    dy_table = pd.DataFrame({
        "Dy_est_MW873": Dy_est["MW873"],
        "Dy_est_MW875": Dy_est["MW875"],
        "Dy_est_MW876": Dy_est["MW876"],
        "Dy_model_MW873": Dy_mod["MW873"],
        "Dy_model_MW875": Dy_mod["MW875"],
        "Dy_model_MW876": Dy_mod["MW876"],
    }, index=labels)

    tgt_df = pd.DataFrame(target_sigmas).set_index("label").loc[labels]

    dx_csv = os.path.join(CSV_DIR, "Dx_estimates_vs_current_Q873.csv")
    dy_csv = os.path.join(CSV_DIR, "Dy_estimates_vs_current_Q873.csv")
    tgt_csv = os.path.join(CSV_DIR, "target_sigma_prediction_scaled_D_Q873.csv")

    dx_table.to_csv(dx_csv); print("Wrote", dx_csv)
    dy_table.to_csv(dy_csv); print("Wrote", dy_csv)
    tgt_df.to_csv(tgt_csv); print("Wrote", tgt_csv)

    # ---- plots ----
    plt.figure(figsize=(11,6))
    plt.plot(xi, dx_table["Dx_est_MW873"], marker="o", label="MW873 Dx (est)")
    plt.plot(xi, dx_table["Dx_est_MW875"], marker="o", label="MW875 Dx (est)")
    plt.plot(xi, dx_table["Dx_est_MW876"], marker="o", label="MW876 Dx (est)")
    plt.plot(xi, dx_table["Dx_model_MW873"], linestyle="--", label="MW873 Dx (model)")
    plt.plot(xi, dx_table["Dx_model_MW875"], linestyle="--", label="MW875 Dx (model)")
    plt.plot(xi, dx_table["Dx_model_MW876"], linestyle="--", label="MW876 Dx (model)")
    plt.xticks(xi, labels, rotation=30, ha="right")
    plt.ylabel("D_x [m]")
    plt.xlabel("Q873 Current Setting")
    plt.title("Estimated D_x at MW873/875/876 vs Q873 Current")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "Dx_est_vs_Q873.png")
    plt.savefig(fig_path, dpi=150)
    print("Wrote", fig_path)
    plt.close()

    plt.figure(figsize=(11,6))
    plt.plot(xi, dy_table["Dy_est_MW873"], marker="o", label="MW873 Dy (est)")
    plt.plot(xi, dy_table["Dy_est_MW875"], marker="o", label="MW875 Dy (est)")
    plt.plot(xi, dy_table["Dy_est_MW876"], marker="o", label="MW876 Dy (est)")
    plt.plot(xi, dy_table["Dy_model_MW873"], linestyle="--", label="MW873 Dy (model)")
    plt.plot(xi, dy_table["Dy_model_MW875"], linestyle="--", label="MW875 Dy (model)")
    plt.plot(xi, dy_table["Dy_model_MW876"], linestyle="--", label="MW876 Dy (model)")
    plt.xticks(xi, labels, rotation=30, ha="right")
    plt.ylabel("D_y [m]")
    plt.xlabel("Q873 Current Setting")
    plt.title("Estimated D_y at MW873/875/876 vs Q873 Current")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "Dy_est_vs_Q873.png")
    plt.savefig(fig_path, dpi=150)
    print("Wrote", fig_path)
    plt.close()

    plt.figure(figsize=(11,6))
    plt.plot(xi, tgt_df["sigx_mm"], marker="o", label="σx @ target")
    plt.plot(xi, tgt_df["sigy_mm"], marker="o", label="σy @ target")
    plt.xticks(xi, labels, rotation=30, ha="right")
    plt.ylabel("σ [mm]")
    plt.xlabel("Q873 Current Setting")
    plt.title("Predicted σ at target (~207 m) using scaled dispersion")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "target_sigma_vs_Q873.png")
    plt.savefig(fig_path, dpi=150)
    print("Wrote", fig_path)
    plt.close()


if __name__ == "__main__":
    main()
