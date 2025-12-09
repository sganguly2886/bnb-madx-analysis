#!/usr/bin/env python3
"""
Plot raw horizontal/vertical MW873/875/876 profiles
for any list of DAQ runs.

Reproduces:
    - All multi–panel MW plots in the talk
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from wire_utils import parse_blob, load_wire_from_df


def plot_full_profiles(csv_dir, csv_files, wire_nums):
    for fn in csv_files:
        path = os.path.join(csv_dir, fn)
        df = pd.read_csv(path, converters={"value": parse_blob})

        fig, axes = plt.subplots(
            nrows=len(wire_nums), ncols=2,
            figsize=(8, 4 * len(wire_nums)),
            squeeze=False
        )

        fig.suptitle(fn, fontsize=14, y=0.98)

        for i, wn in enumerate(wire_nums):
            wire = load_wire_from_df(df, wn)

            # horizontal
            ax_h = axes[i, 0]
            ax_h.bar(range(len(wire.horizontal_values)),
                     wire.horizontal_values, width=1)
            ax_h.set_title(f"MW{wn} Horizontal")
            ax_h.grid(True, linestyle=":")

            # vertical
            ax_v = axes[i, 1]
            ax_v.bar(range(len(wire.vertical_values)),
                     wire.vertical_values, width=1, color="C1")
            ax_v.set_title(f"MW{wn} Vertical")
            ax_v.grid(True, linestyle=":")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out = os.path.join(csv_dir, fn.replace(".csv", ".png"))
        plt.savefig(out, dpi=150)
        print("Saved:", out)
        plt.close(fig)


def main():
    data_dir = "../0_raw_data"

    scans = [
        "bnb_br_study_baseline_2025-06-16T152208Z.csv",
        "bnb_br_study_br_narrow_2025-06-16T170402Z.csv",
        "bnb_br_study_br_sf_scan_2025-06-16T182825Z.csv",
        "bnb_br_study_Q875+28A_2025-06-16T163530Z.csv",
        "bnb_br_study_Q873_Q874+5p_2025-06-16T165401Z.csv",
    ]

    plot_full_profiles(data_dir, scans, [873, 875, 876])


if __name__ == "__main__":
    main()
