#!/usr/bin/env python3
"""
Author: <Your Name>

Emittance & momentum–spread analysis for Q873 scan.

This script:
  - Reads Q873 Twiss TFS files (prefers *_flipPol)
  - Uses measured MW873/875/876 beam sizes vs current
  - Fits εx, εy, σδ with uncertainties at each current
  - Writes CSVs to:   ../3_results/csv/
  - Writes plots to:  ../3_results/figures/
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Output directories
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "..", "3_results", "csv")
FIG_DIR = os.path.join(BASE_DIR, "..", "3_results", "figures")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------
# Currents, labels, monitors
# -----------------------------
ORDER = ["-30","-24","-18","-12","-6","0","+6","+12","+18","+24","+30"]
XLABELS = {
    "-30":"Q873 −30A","-24":"Q873 −24A","-18":"Q873 −18A","-12":"Q873 −12A","-6":"Q873 −6A",
    "0":"Q873 Baseline","+6":"Q873 +6A","+12":"Q873 +12A",
    "+18":"Q873 +18A","+24":"Q873 +24A","+30":"Q873 +30A"
}
MONITORS = ["MW873","MW875","MW876"]

# -----------------------------
# Prefer flipPol TFS files, fallback to original names
# -----------------------------
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

def pick_file_for_current(key: str) -> str:
    """
    Prefer flipped-polarity TFS if present, else original.
    """
    if key == "0":
        candidates = ["twiss_Q873_default.tfs"]
    else:
        tag = f"{key}A".replace("+", "+")
        candidates = [f"twiss_Q873_{tag}_flipPol.tfs",
                      OLD_FILES.get(key, "")]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    # if nothing exists, return first candidate and let caller fail
    return candidates[0]

FILES = {k: pick_file_for_current(k) for k in ORDER}

# -----------------------------
# Monitor resolution (mm) – set if needed
# -----------------------------
RES_H_MM = 0.00   # horizontal resolution (mm)
RES_V_MM = 0.00   # vertical resolution (mm)

def get_res_mm(plane: str, wire: str) -> float:
    """
    Currently a simple scalar; can be upgraded to per-wire dict.
    """
    return float(RES_H_MM if plane == "H" else RES_V_MM)

# -----------------------------
# Measured σ table (mm) that you provided
# -----------------------------
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
    """Robust lookup ignoring spaces."""
    target = label.replace(" ", "")
    for idx in MEAS_DF.index:
        if idx.replace(" ", "") == target:
            return MEAS_DF.loc[idx]
    raise KeyError(f"Measurement row not found for label '{label}'")

# -----------------------------
# TFS reader & MW selector
# -----------------------------
def read_tfs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # header line starting with '*'
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("*"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"No '*' header line in {path}")

    cols = re.sub(r"^\*", "", lines[header_idx]).strip().split()
    start = header_idx + 1
    if start < len(lines) and lines[start].lstrip().startswith("$"):
        start += 1

    data_rows = []
    for ln in lines[start:]:
        if ln.strip() == "" or ln.lstrip().startswith(("@","#","*","$")):
            continue
        data_rows.append(ln)

    from io import StringIO
    df = pd.read_csv(StringIO("".join(data_rows)),
                     sep=r"\s+", header=None, names=cols, engine="python")

    df.columns = [c.upper() for c in df.columns]
    if "NAME" in df.columns:
        df["NAME"] = (df["NAME"].astype(str).str.upper()
                      .str.strip().str.strip('"').str.strip("'")
                      .str.replace(r"[:\.].*$", "", regex=True))
    return df

def get_mw_rows(df: pd.DataFrame) -> dict:
    out = {}
    for w in MONITORS:
        m = df[df["NAME"] == w]
        if m.empty:
            m = df[df["NAME"].str.contains(w, na=False)]
        if m.empty:
            raise RuntimeError(f"{w} not found in TFS")
        out[w] = m.iloc[0]
    return out

# -----------------------------
# Fitting utilities
# Model: y = [β, D²]·[ε, σδ²]
# -----------------------------
def subtract_resolution_mm(sig_mm: float, res_mm: float) -> float:
    s2 = sig_mm**2 - res_mm**2
    return math.sqrt(s2) if s2 > 0 else 0.0

def fit_two_param(sig_mm_list, beta_list, disp_list):
    """
    Fit σ² = ε β + (σδ²) D² with LSQ and return:
      (ε, σδ², ε_std, σδ²_std)
    All in SI units (m, m²).
    """
    y = np.array([(s*1e-3)**2 for s in sig_mm_list], float)
    X = np.column_stack([np.array(beta_list, float),
                         np.array(disp_list, float)**2])

    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    eps, sdel2 = float(p[0]), float(p[1])

    r = y - X @ p
    N, P = X.shape
    dof = max(N - P, 1)
    rss = float(r.T @ r)
    s2 = rss / dof

    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        std = np.sqrt(np.diag(cov))
        eps_std, sdel2_std = float(std[0]), float(std[1])
    except np.linalg.LinAlgError:
        eps_std, sdel2_std = float("nan"), float("nan")

    return max(eps, 0.0), max(sdel2, 0.0), eps_std, sdel2_std

def fit_one_param(sig_mm_list, beta_list):
    """
    Fit σ² = ε β (Dy≈0) with analytic LSQ and return (ε, ε_std).
    """
    y = np.array([(s*1e-3)**2 for s in sig_mm_list], float)
    X = np.array(beta_list, float).reshape(-1, 1)

    denom = float((X.T @ X)[0, 0])
    if denom <= 0:
        return 0.0, float("nan")

    eps = float((X.T @ y)[0]) / denom
    r = y - X.flatten()*eps
    dof = max(len(y) - 1, 1)
    rss = float(r.T @ r)
    s2 = rss / dof
    eps_std = math.sqrt(s2 / denom)

    return max(eps, 0.0), eps_std

COND_THRESH = 1e10
DY_THRESH   = 1e-3  # m

# -----------------------------
# Main analysis
# -----------------------------
def main():
    rows_out = []
    xs, xtick_labels = [], []
    epsx, epsx_std, epsy, epsy_std = [], [], [], []
    sdx, sdx_std, sdy, sdy_std     = [], [], [], []

    for key in ORDER:
        path = FILES[key]
        if not os.path.exists(path):
            print(f"Skip {key}: {path} not found")
            continue

        df = read_tfs(path)
        rows = get_mw_rows(df)
        label = XLABELS[key]
        meas = find_meas_row(label)

        betx = [float(rows[w]["BETX"]) for w in MONITORS]
        bety = [float(rows[w]["BETY"]) for w in MONITORS]
        dx   = [float(rows[w]["DX"])   for w in MONITORS]
        dy   = [float(rows[w]["DY"])   for w in MONITORS]

        # resolution–subtracted sigmas
        sigx_mm = []
        sigy_mm = []
        for w, short in zip(MONITORS, ["873","875","876"]):
            sx = float(meas[f"{short} H"])
            sy = float(meas[f"{short} V"])
            sigx_mm.append(subtract_resolution_mm(sx, get_res_mm("H", w)))
            sigy_mm.append(subtract_resolution_mm(sy, get_res_mm("V", w)))

        # horizontal: 2-param
        ex, sdx2, ex_std, sdx2_std = fit_two_param(sigx_mm, betx, dx)
        sdx_val   = math.sqrt(sdx2)
        sdx_stdev = (0.5*sdx2_std/math.sqrt(sdx2)
                     if sdx2 > 0 and np.isfinite(sdx2_std) else float("nan"))

        # vertical: 2-param when possible, else 1-param (Dy≈0)
        use_one_param_v = (max(abs(v) for v in dy) < DY_THRESH)
        if not use_one_param_v:
            Xv = np.column_stack([np.array(bety, float),
                                  np.array(dy, float)**2])
            try:
                cond = np.linalg.cond(Xv)
                use_one_param_v = cond > COND_THRESH
            except np.linalg.LinAlgError:
                use_one_param_v = True

        if use_one_param_v:
            ey, ey_std = fit_one_param(sigy_mm, bety)
            sdy_val, sdy_stdev = 0.0, 0.0
        else:
            ey, sdy2, ey_std, sdy2_std = fit_two_param(sigy_mm, bety, dy)
            sdy_val   = math.sqrt(sdy2)
            sdy_stdev = (0.5*sdy2_std/math.sqrt(sdy2)
                         if sdy2 > 0 and np.isfinite(sdy2_std)
                         else float("nan"))

        xs.append(int(key))
        xtick_labels.append(label)
        epsx.append(ex); epsx_std.append(ex_std)
        epsy.append(ey); epsy_std.append(ey_std)
        sdx.append(sdx_val); sdx_std.append(sdx_stdev)
        sdy.append(sdy_val); sdy_std.append(sdy_stdev)

        rows_out.append({
            "Current": key,
            "Label": label,
            "eps_x [m rad]": ex,  "eps_x_std [m rad]": ex_std,
            "eps_y [m rad]": ey,  "eps_y_std [m rad]": ey_std,
            "sigma_delta_x": sdx_val, "sigma_delta_x_std": sdx_stdev,
            "sigma_delta_y": sdy_val, "sigma_delta_y_std": sdy_stdev,
            "fit_mode_y": "1-param (Dy≈0)" if use_one_param_v else "2-param",
        })

    # -----------------------------
    # Save main emittance CSV
    # -----------------------------
    res_df = pd.DataFrame(rows_out).set_index("Label")
    out_csv = os.path.join(CSV_DIR, "emittance_vs_Q873_current_with_errors.csv")
    res_df.to_csv(out_csv)
    print("Wrote", out_csv)

    xs = np.array(xs)

    # -----------------------------
    # Emittance plots/CSV in π·mm·mrad
    # -----------------------------
    scale_pi_mm_mrad = 1e6 / math.pi
    epsx_pi   = np.array(epsx)     * scale_pi_mm_mrad
    epsy_pi   = np.array(epsy)     * scale_pi_mm_mrad
    epsx_pi_e = np.array(epsx_std) * scale_pi_mm_mrad
    epsy_pi_e = np.array(epsy_std) * scale_pi_mm_mrad

    plt.figure(figsize=(9,5))
    plt.errorbar(xs, epsx_pi, yerr=epsx_pi_e, marker="o", linestyle="-",
                 capsize=3, label="εx")
    plt.errorbar(xs, epsy_pi, yerr=epsy_pi_e, marker="s", linestyle="-",
                 capsize=3, label="εy")
    plt.xticks(xs, xtick_labels, rotation=30, ha="right")
    plt.ylabel("Emittance [π·mm·mrad]")
    plt.xlabel("Q873 Current Setting")
    plt.title("Fitted Emittance vs Q873 Current (π·mm·mrad, 1σ errors)")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "emittance_vs_Q873_current_pi-mm-mrad.png")
    plt.savefig(fig_path, dpi=150)
    print("Wrote", fig_path)
    plt.close()

    emi_out = pd.DataFrame({
        "eps_x [π·mm·mrad]": epsx_pi,
        "eps_x_std [π·mm·mrad]": epsx_pi_e,
        "eps_y [π·mm·mrad]": epsy_pi,
        "eps_y_std [π·mm·mrad]": epsy_pi_e,
    }, index=xtick_labels)
    emi_csv = os.path.join(CSV_DIR, "emittance_vs_Q873_current_pi-mm-mrad.csv")
    emi_out.to_csv(emi_csv)
    print("Wrote", emi_csv)

    # -----------------------------
    # 95% emittance (geom & normalized)
    # -----------------------------
    F95 = 6.0
    epsx95_pi   = F95 * epsx_pi
    epsy95_pi   = F95 * epsy_pi
    epsx95_pi_e = F95 * epsx_pi_e
    epsy95_pi_e = F95 * epsy_pi_e

    p_GeV = 8.83490
    m_GeV = 0.9382720813
    bg = p_GeV / m_GeV

    epsx95n_pi   = epsx95_pi   * bg
    epsy95n_pi   = epsy95_pi   * bg
    epsx95n_pi_e = epsx95_pi_e * bg
    epsy95n_pi_e = epsy95_pi_e * bg

    # 95% geometric plot
    plt.figure(figsize=(9,5))
    plt.errorbar(xs, epsx95_pi, yerr=epsx95_pi_e, marker="o", linestyle="-",
                 capsize=3, label="εx, 95% geom")
    plt.errorbar(xs, epsy95_pi, yerr=epsy95_pi_e, marker="s", linestyle="-",
                 capsize=3, label="εy, 95% geom")
    plt.xticks(xs, xtick_labels, rotation=30, ha="right")
    plt.ylabel("Emittance [π·mm·mrad]")
    plt.xlabel("Q873 Current Setting")
    plt.title("95% Emittance vs Q873 Current (geometric)")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "emittance95_geometric_vs_Q873.png")
    plt.savefig(fig_path, dpi=150)
    print("Wrote", fig_path)
    plt.close()

    # 95% normalized plot
    plt.figure(figsize=(9,5))
    plt.errorbar(xs, epsx95n_pi, yerr=epsx95n_pi_e, marker="o", linestyle="-",
                 capsize=3, label="εx, 95% norm")
    plt.errorbar(xs, epsy95n_pi, yerr=epsy95n_pi_e, marker="s", linestyle="-",
                 capsize=3, label="εy, 95% norm")
    plt.xticks(xs, xtick_labels, rotation=30, ha="right")
    plt.ylabel("Normalized Emittance [π·mm·mrad]")
    plt.xlabel("Q873 Current Setting")
    plt.title("95% Normalized Emittance vs Q873 Current")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "emittance95_normalized_vs_Q873.png")
    plt.savefig(fig_path, dpi=150)
    print("Wrote", fig_path)
    plt.close()

    # CSVs
    geom_csv = os.path.join(CSV_DIR, "emittance95_geometric_pi-mm-mrad.csv")
    pd.DataFrame({
        "epsx_95_geom [π·mm·mrad]": epsx95_pi,
        "epsx_95_geom_std [π·mm·mrad]": epsx95_pi_e,
        "epsy_95_geom [π·mm·mrad]": epsy95_pi,
        "epsy_95_geom_std [π·mm·mrad]": epsy95_pi_e,
    }, index=xtick_labels).to_csv(geom_csv)
    print("Wrote", geom_csv)

    norm_csv = os.path.join(CSV_DIR, "emittance95_normalized_pi-mm-mrad.csv")
    pd.DataFrame({
        "epsx_95_norm [π·mm·mrad]": epsx95n_pi,
        "epsx_95_norm_std [π·mm·mrad]": epsx95n_pi_e,
        "epsy_95_norm [π·mm·mrad]": epsy95n_pi,
        "epsy_95_norm_std [π·mm·mrad]": epsy95n_pi_e,
    }, index=xtick_labels).to_csv(norm_csv)
    print("Wrote", norm_csv)

    # -----------------------------
    # Momentum–spread vs current (σδ and Δp)
    # -----------------------------
    sdx_arr = np.array(sdx)
    sdx_std_arr = np.array(sdx_std)
    sdy_arr = np.array(sdy)
    sdy_std_arr = np.array(sdy_std)

    # zero out absurd vertical values
    typical_sdx = np.nanmedian(sdx_arr)
    mask_bad = sdy_arr > 5 * typical_sdx
    sdy_arr[mask_bad] = np.nan
    sdy_std_arr[mask_bad] = np.nan

    plt.figure(figsize=(9,5))
    plt.errorbar(xs, sdx_arr*1e3, yerr=sdx_std_arr*1e3,
                 marker="o", linestyle="-", capsize=3, label="σδ (H)")
    plt.xticks(xs, xtick_labels, rotation=30, ha="right")
    plt.xlabel("Q873 Current Setting")
    plt.ylabel("Momentum Spread σδ ×10⁻³ (rms)")
    plt.title("Momentum Spread vs Q873 Current")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "momentum_spread_sigma_delta_vs_Q873.png")
    plt.savefig(fig_path, dpi=150)
    print("Wrote", fig_path)
    plt.close()

    sdel_df = pd.DataFrame({
        "sigma_delta_x [rms]": sdx_arr,
        "sigma_delta_x_std": sdx_std_arr,
        "sigma_delta_y [rms]": sdy_arr,
        "sigma_delta_y_std": sdy_std_arr,
    }, index=xtick_labels)
    sdel_csv = os.path.join(CSV_DIR, "momentum_spread_vs_Q873_current_clean.csv")
    sdel_df.to_csv(sdel_csv)
    print("Wrote", sdel_csv)

    # Δp
    p0 = 8000.0  # MeV/c
    dp_x = sdx_arr * p0
    dp_x_std = sdx_std_arr * p0

    plt.figure(figsize=(9,5))
    plt.errorbar(xs, dp_x, yerr=dp_x_std,
                 marker="o", linestyle="-", capsize=3, label="Δp (H)")
    plt.xticks(xs, xtick_labels, rotation=30, ha="right")
    plt.xlabel("Q873 Current Setting")
    plt.ylabel("Δp [MeV/c]")
    plt.title("Momentum Spread Δp (Horizontal) vs Q873 Current")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "momentum_spread_Dp_vs_Q873.png")
    plt.savefig(fig_path, dpi=150)
    print("Wrote", fig_path)
    plt.close()

    dp_df = pd.DataFrame({
        "sigma_delta_x [rms]": sdx_arr,
        "sigma_delta_x_std": sdx_std_arr,
        "Δp_x [MeV/c]": dp_x,
        "Δp_x_std [MeV/c]": dp_x_std,
    }, index=xtick_labels)
    dp_csv = os.path.join(CSV_DIR, "momentum_spread_Dp_horizontal_vs_Q873_current.csv")
    dp_df.to_csv(dp_csv)
    print("Wrote", dp_csv)


if __name__ == "__main__":
    main()
