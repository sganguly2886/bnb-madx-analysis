"""
Plot measured beam-spot sigmas from wire scanners (MW873, MW875, MW876)
as a function of quadrupole settings.

This script produces four figures:
  1. σ vs Q873 current
  2. σ vs Q874 current
  3. σ vs Q875 current
  4. σ vs combined Q873 + Q874 settings

All values are measurement-derived (not MAD-X predictions).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Shared plotting style
# ------------------------------------------------------------------------------
COLORS = {
    "873 H": "#1f77b4",
    "873 V": "#ff7f0e",
    "875 H": "#2ca02c",
    "875 V": "#d62728",
    "876 H": "#9467bd",
    "876 V": "#8c564b",
}


def plot_sigma_dataframe(df, title, xlabel, rotation=45):
    """Generic helper for plotting σ vs quad setting."""
    setups = df.index.tolist()
    x = np.arange(len(setups))

    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(
            x, df[col],
            marker="o",
            linestyle="-",
            color=COLORS[col],
            label=col
        )

    plt.xticks(x, setups, rotation=rotation, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel("σ [mm]")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.legend(title="Wire & Plane", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 1. Q873 scan (measurement)
# ==============================================================================
q873_data = {
    "Q873 −30A":  [2.30, 5.05, 3.56, 4.50, 2.29, 4.40],
    "Q873 −24A":  [2.20, 4.95, 3.40, 3.84, 2.44, 3.60],
    "Q873 −18A":  [2.29, 5.01, 3.81, 3.20, 2.78, 2.89],
    "Q873 −12A":  [2.19, 4.97, 3.76, 2.58, 2.84, 2.21],
    "Q873 −6A":   [2.18, 5.00, 3.92, 1.95, 0.745, 1.56],
    "Q873 Baseline": [2.293, 4.959, 4.957, 1.371, 3.771, 0.9685],
    "Q873 +6A":   [2.22, 5.01, 3.92, 4.22, 3.60, 0.538],
    "Q873 +12A":  [2.30, 5.01, 4.62, 0.487, 4.04, 0.950],
    "Q873 +18A":  [2.32, 5.03, 4.81, 0.918, 4.32, 1.64],
    "Q873 +24A":  [2.33, 5.05, 5.10, 1.54, 4.37, 2.37],
    "Q873 +30A":  [2.30, 5.02, 5.14, 2.23, 4.67, 3.07],
}

df873 = pd.DataFrame.from_dict(
    q873_data,
    orient="index",
    columns=["873 H", "873 V", "875 H", "875 V", "876 H", "876 V"]
)

plot_sigma_dataframe(
    df873,
    title="Beam-spot σ vs. Q873 Current (Measurements)",
    xlabel="Q873 Current Setting"
)


# ==============================================================================
# 2. Q874 scan (measurement)
# ==============================================================================
q874_data = {
    "Q874 −45A":   [2.18, 4.94, 5.90, 0.494, 6.02, 0.655],
    "Q874 −36A":   [2.23, 4.99, 5.62, 0.611, 5.77, 0.548],
    "Q874 −27A":   [2.22, 5.00, 5.24, 0.785, 5.27, 0.508],
    "Q874 −18A":   [2.25, 5.00, 4.96, 0.965, 5.02, 0.573],
    "Q874 −9A":    [2.27, 5.01, 4.60, 1.17,  4.17, 0.739],
    "Q874 Baseline":[2.25, 4.99, 4.11, 1.37,  3.44, 0.963],
    "Q874 +9A":    [2.15, 4.93, 3.52, 1.57,  2.42, 1.18],
    "Q874 +18A":   [2.27, 4.99, 3.16, 1.80,  1.75, 1.45],
    "Q874 +23.8A": [2.30, 5.00, 2.85, 1.94,  1.30, 1.61],
}

df874 = pd.DataFrame.from_dict(
    q874_data,
    orient="index",
    columns=["873 H", "873 V", "875 H", "875 V", "876 H", "876 V"]
)

plot_sigma_dataframe(
    df874,
    title="Beam-spot σ vs. Q874 Current (Measurements)",
    xlabel="Q874 Current Setting"
)


# ==============================================================================
# 3. Q875 scan (measurement)
# ==============================================================================
q875_data = {
    "Q875 −35A": [2.18, 4.95, 3.91, 1.43, 2.88, 1.08],
    "Q875 −28A": [2.17, 4.98, 3.96, 1.42, 2.94, 1.05],
    "Q875 −21A": [2.33, 5.01, 4.22, 1.41, 3.32, 1.03],
    "Q875 −14A": [2.20, 5.00, 4.06, 1.39, 3.15, 1.01],
    "Q875 −7A":  [2.24, 5.00, 4.11, 1.37, 3.29, 0.984],
    "Q875 Baseline": [2.17, 4.96, 3.98, 1.37, 3.24, 0.939],
    "Q875 +7A":  [2.21, 4.98, 4.14, 1.35, 3.66, 0.925],
    "Q875 +14A": [2.32, 5.01, 4.34, 1.33, 3.71, 0.90],
    "Q875 +21A": [2.22, 4.99, 4.22, 1.32, 3.61, 0.877],
    "Q875 +28A": [2.29, 5.01, 4.35, 1.31, 3.86, 0.850],
    "Q875 +35A": [2.23, 4.96, 4.30, 1.29, 3.80, 0.825],
}

df875 = pd.DataFrame.from_dict(
    q875_data,
    orient="index",
    columns=["873 H", "873 V", "875 H", "875 V", "876 H", "876 V"]
)

plot_sigma_dataframe(
    df875,
    title="Beam-spot σ vs. Q875 Current (Measurements)",
    xlabel="Q875 Current Setting"
)


# ==============================================================================
# 4. Combined Q873 + Q874 settings (measurement)
# ==============================================================================
combined_data = {
    "Q873 −12A + Q874 −18A": [2.27, 5.00, 4.64, 2.29, 4.41, 1.85],
    "Q873 −6A + Q874 −9A":  [2.18, 4.98, 4.25, 1.83, 3.77, 1.40],
    "Baseline Q873 + Q874": [2.23, 4.99, 4.16, 1.36, 3.53, 0.948],
    "Q873 +6A + Q874 +9A": [2.28, 5.03, 3.98, 0.922, 2.91, 0.603],
    "Q873 +12A + Q874 +18A":[2.33, 4.99, 3.50, 0.647, 2.14, 0.595],
    "Baseline (all off)":   [2.17, 4.96, 4.01, 1.37, 3.26, 0.952],
}

df_combined = pd.DataFrame.from_dict(
    combined_data,
    orient="index",
    columns=["873 H", "873 V", "875 H", "875 V", "876 H", "876 V"]
)

plot_sigma_dataframe(
    df_combined,
    title="Beam-spot σ vs. Combined Q873 + Q874 Settings (Measurements)",
    xlabel="Combined Quad Setting",
    rotation=30
)
