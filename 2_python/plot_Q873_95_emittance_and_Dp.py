#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Constants
# ============================
F95 = 6.0                 # FNAL convention: 95% ≈ 6× rms
p0_MeVc = 8000.0          # reference momentum
p_GeV = 8.83490
m_GeV = 0.9382720813
beta_gamma = p_GeV / m_GeV

# ============================
# Load emittance results
# ============================
df = pd.read_csv(
    "emittance_vs_Q873_current_with_errors.csv",
    index_col=0
)

# Extract columns
epsx = df["eps_x [m rad]"].values
epsy = df["eps_y [m rad]"].values
epsx_e = df["eps_x_std [m rad]"].values
epsy_e = df["eps_y_std [m rad]"].values

sdx = df["sigma_delta_x"].values
sdx_e = df["sigma_delta_x_std"].values

labels = df.index.tolist()
xs = np.arange(len(labels))

# ============================
# 95% emittances (π·mm·mrad)
# ============================
scale = 1e6 / math.pi

epsx95 = F95 * epsx * scale
epsy95 = F95 * epsy * scale
epsx95_e = F95 * epsx_e * scale
epsy95_e = F95 * epsy_e * scale

epsx95n = epsx95 * beta_gamma
epsy95n = epsy95 * beta_gamma
epsx95n_e = epsx95_e * beta_gamma
epsy95n_e = epsy95_e * beta_gamma

# ============================
# Δp (horizontal)
# ============================
Dp = sdx * p0_MeVc
Dp_e = sdx_e * p0_MeVc

# ============================
# PRINT SUMMARY TABLE
# ============================
summary = pd.DataFrame({
    "εx_95_geom [π·mm·mrad]": epsx95,
    "εy_95_geom [π·mm·mrad]": epsy95,
    "εx_95_norm [π·mm·mrad]": epsx95n,
    "εy_95_norm [π·mm·mrad]": epsy95n,
    "σδx (rms)": sdx,
    "Δp_x [MeV/c]": Dp,
}, index=labels)

print("\n=== Q873 OPTICS SUMMARY ===")
print(summary.round(3))

# ============================
# PLOTS
# ============================

# --- 95% geometric ---
plt.figure(figsize=(9,5))
plt.errorbar(xs, epsx95, yerr=epsx95_e, fmt="o-", capsize=3, label="εx 95% (geom)")
plt.errorbar(xs, epsy95, yerr=epsy95_e, fmt="s-", capsize=3, label="εy 95% (geom)")
plt.xticks(xs, labels, rotation=30, ha="right")
plt.ylabel("Emittance [π·mm·mrad]")
plt.title("95% Geometric Emittance vs Q873 Current")
plt.grid(True, ls=":")
plt.legend()
plt.tight_layout()
plt.show()

# --- 95% normalized ---
plt.figure(figsize=(9,5))
plt.errorbar(xs, epsx95n, yerr=epsx95n_e, fmt="o-", capsize=3, label="εx 95% (norm)")
plt.errorbar(xs, epsy95n, yerr=epsy95n_e, fmt="s-", capsize=3, label="εy 95% (norm)")
plt.xticks(xs, labels, rotation=30, ha="right")
plt.ylabel("Normalized Emittance [π·mm·mrad]")
plt.title("95% Normalized Emittance vs Q873 Current")
plt.grid(True, ls=":")
plt.legend()
plt.tight_layout()
plt.show()

# --- Δp ---
plt.figure(figsize=(9,5))
plt.errorbar(xs, Dp, yerr=Dp_e, fmt="o-", capsize=3, label="Δpₓ")
plt.xticks(xs, labels, rotation=30, ha="right")
plt.ylabel("Δp [MeV/c]")
plt.title("Momentum Spread Δp (Horizontal) vs Q873 Current")
plt.grid(True, ls=":")
plt.legend()
plt.tight_layout()
plt.show()
