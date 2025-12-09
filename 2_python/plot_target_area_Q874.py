#!/usr/bin/env python3
"""
Plot predicted σx, σy and area at target vs Q874 current.

Input:
  - 3_results/csv/target_sigma_prediction_scaled_D_Q874.csv

Outputs:
  - 3_results/figures/target_sigma_Q874.png
  - 3_results/figures/target_area_Q874.png
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE, "..", "3_results", "csv")
FIG_DIR = os.path.join(BASE, "..", "3_results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

IN_CSV  = os.path.join(CSV_DIR, "target_sigma_prediction_scaled_D_Q874.csv")

df = pd.read_csv(IN_CSV, index_col=0)

labels = df.index.tolist()

# σx, σy
plt.figure(figsize=(8, 5))
plt.plot(labels, df["sigx_mm"], marker="o", label="σx")
plt.plot(labels, df["sigy_mm"], marker="s", label="σy")
plt.xticks(rotation=30, ha="right")
plt.ylabel("σ at target [mm]")
plt.xlabel("Q874 current setting")
plt.title("Predicted Beam Spot Size at Target vs Q874")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
out1 = os.path.join(FIG_DIR, "target_sigma_Q874.png")
plt.savefig(out1, dpi=150)
plt.close()
print(f"Wrote {out1}")

# area
plt.figure(figsize=(8, 5))
plt.plot(labels, df["area_mm2"], marker="o")
plt.xticks(rotation=30, ha="right")
plt.ylabel("σx · σy [mm²]")
plt.xlabel("Q874 current setting")
plt.title("Effective Beam Spot Area at Target vs Q874")
plt.grid(True, linestyle=":")
plt.tight_layout()
out2 = os.path.join(FIG_DIR, "target_area_Q874.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"Wrote {out2}")
