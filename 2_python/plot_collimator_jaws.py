#!/usr/bin/env python3
"""
Plot collimator jaw positions vs time from BNB beamline CSV data.

This script reproduces the "Collimator jaw position vs time" plot
used in the BNB MAD-X analysis.

Raw data is NOT stored in the repository and must be supplied
explicitly at runtime.

Example
-------
python plot_collimator_jaws.py \
    --data-dir /path/to/bnb_br_study_2025-06-16 \
    --file bnb_br_study_br_collimation_2025-06-16T175101Z.csv
"""

import os
import ast
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_blob(x):
    """
    Convert literal byte strings stored in CSVs back into bytes.
    """
    if isinstance(x, (bytes, bytearray)):
        return x
    if isinstance(x, str) and x.startswith("b'"):
        return ast.literal_eval(x)
    return x


def load_collimator_data(csv_path):
    """
    Load and filter collimator jaw channels from raw CSV.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        converters={"value": parse_blob},
    )

    # Collimator jaw devices:
    # I:C836AH, I:C836AV, I:C838BH, I:C838BV, etc.
    mask = df.device.str.match(r"I:C83[68][AB][HV]$")
    coll = df.loc[mask].copy()

    coll["mills"] = coll["value"].astype(float)
    return coll


def plot_collimator_jaws(coll, outdir):
    """
    Generate and save collimator jaw position plot.
    """
    wide = coll.pivot(
        index="timestamp",
        columns="device",
        values="mills",
    )

    # mils → inches → mm
    wide_mm = wide * 0.001 * 25.4

    plt.figure(figsize=(10, 4))

    for col in wide_mm.columns:
        plt.plot(wide_mm.index, wide_mm[col], label=col)

    # ±0.250 inch threshold
    thr_mm = 0.250 * 25.4
    plt.axhline(thr_mm, color="k", ls="--")
    plt.axhline(-thr_mm, color="k", ls="--")

    plt.title("Collimator Jaw Positions vs Time")
    plt.xlabel("Time")
    plt.ylabel("Jaw Position [mm]")
    plt.legend(ncol=2)
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "collimator_jaws_vs_time.png")
    plt.savefig(outfile, dpi=150)
    plt.show()

    print(f"Saved figure: {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot collimator jaw positions vs time"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing raw bnb_br_study CSV files",
    )
    parser.add_argument(
        "--file",
        default="bnb_br_study_br_collimation_2025-06-16T175101Z.csv",
        help="Collimator CSV filename",
    )
    parser.add_argument(
        "--outdir",
        default="../3_results/figures",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, args.file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\nERROR: Collimator CSV not found:\n  {csv_path}\n"
            "Provide the correct --data-dir and --file."
        )

    coll = load_collimator_data(csv_path)
    plot_collimator_jaws(coll, args.outdir)


if __name__ == "__main__":
    main()
