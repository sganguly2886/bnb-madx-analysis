#!/usr/bin/env python3
"""
Plot raw horizontal and vertical multiwire profiles (MW873, MW875, MW876)
from BNB beam study CSV files.

Reproduces all multi-panel wire profile plots used in the BNB narrow beam study.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from wire_utils import parse_blob, load_wire_from_df


def plot_full_profiles(data_dir, csv_files, wire_nums, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for fn in csv_files:
        path = os.path.join(data_dir, fn)
        print(f"Reading: {path}")

        df = pd.read_csv(path, converters={"value": parse_blob})

        fig, axes = plt.subplots(
            nrows=len(wire_nums),
            ncols=2,
            figsize=(8, 4 * len(wire_nums)),
            squeeze=False
        )

        fig.suptitle(fn, fontsize=14, y=0.97)

        for i, wn in enumerate(wire_nums):
            wire = load_wire_from_df(df, wn)

            # Horizontal profile
            ax_h = axes[i, 0]
            ax_h.bar(
                range(len(wire.horizontal_values)),
                wire.horizontal_values,
                width=1
            )
            ax_h.set_title(f"MW{wn} Horizontal")
            ax_h.set_ylabel("Counts")
            ax_h.grid(True, linestyle=":")

            # Vertical profile
            ax_v = axes[i, 1]
            ax_v.bar(
                range(len(wire.vertical_values)),
                wire.vertical_values,
                width=1,
                color="C1"
            )
            ax_v.set_title(f"MW{wn} Vertical")
            ax_v.grid(True, linestyle=":")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_png = os.path.join(
            out_dir,
            fn.replace(".csv", "_wire_profiles.png")
        )
        plt.savefig(out_png, dpi=150)
        print(f"Saved: {out_png}")

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw multiwire profiles from BNB beam study CSV files"
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing raw BNB beam study CSV files"
    )

    parser.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="List of CSV filenames to process"
    )

    parser.add_argument(
        "--wires",
        nargs="+",
        type=int,
        default=[873, 875, 876],
        help="Wire numbers to plot (default: 873 875 876)"
    )

    parser.add_argument(
        "--out-dir",
        default="../3_results/figures",
        help="Output directory for plots (default: ../3_results/figures)"
    )

    args = parser.parse_args()

    plot_full_profiles(
        data_dir=args.data_dir,
        csv_files=args.csv_files,
        wire_nums=args.wires,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
