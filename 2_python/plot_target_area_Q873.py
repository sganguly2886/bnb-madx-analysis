#!/usr/bin/env python3
"""
Author: <Your Name>

Plot target spot area vs Q873 current.

Inputs:
    ../3_results/csv/target_sigma_prediction_scaled_D_Q873.csv
    
Outputs:
    ../3_results/figures/target_area_vs_Q873.png
    ../3_results/csv/target_area_vs_Q873.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR  = os.path.join(BASE_DIR, "..", "3_results", "csv")
FIG_DIR  = os.path.join(BASE_DIR, "..", "3_results", "figures")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def main():

    fn = os.path.join(CSV_DIR, "target_sigma_prediction_scaled_D_Q873.csv")
    if not os.path.exists(fn):
        raise FileNotFoundError(
            f"{fn} not found. Run cal_spotsize_Q873.py first."
        )

    df = pd.read_csv(fn, index_col=0)

    labels = df.index.tolist()
    xi = np.arange(len(labels))

    sigx = df["sigx_mm"].values
    sigy = df["sigy_mm"].values
    area = sigx * sigy  

    # Save area to CSV
    out_csv = os.path.join(CSV_DIR, "target_area_vs_Q873.csv")
    pd.DataFrame({
        "label": labels,
        "sigma_x_mm": sigx,
        "sigma_y_mm": sigy,
        "area_mm2": area
    }).to_csv(out_csv, index=False)
    print("Wrote", out_csv)

    # ---- Plot area ----
    plt.figure(figsize=(10,5))
    plt.plot(xi, area, marker="o", linestyle="-", color="purple")
    plt.xticks(xi, labels, rotation=30, ha="right")
    plt.ylabel("Spot Area  σx·σy  [mm²]")
    plt.xlabel("Q873 Current Setting")
    plt.title("BNB Target Spot Area vs Q873 Current")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    out_fig = os.path.join(FIG_DIR, "target_area_vs_Q873.png")
    plt.savefig(out_fig, dpi=150)
    print("Wrote", out_fig)
    plt.close()

    # ---- Also plot σx & σy ----
    plt.figure(figsize=(10,5))
    plt.plot(xi, sigx, marker="o", label="σx [mm]")
    plt.plot(xi, sigy, marker="o", label="σy [mm]")
    plt.xticks(xi, labels, rotation=30, ha="right")
    plt.ylabel("Spot Size [mm]")
    plt.xlabel("Q873 Current Setting")
    plt.title("Target σx, σy vs Q873")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()

    out_fig2 = os.path.join(FIG_DIR, "target_sigma_xy_vs_Q873.png")
    plt.savefig(out_fig2, dpi=150)
    print("Wrote", out_fig2)
    plt.close()


if __name__ == "__main__":
    main()
