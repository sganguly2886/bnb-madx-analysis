#!/usr/bin/env python3
"""
Fit Gaussian sigma_x and sigma_y for MW873, MW875, MW876
from raw BNB multiwire CSV files.

Outputs beam size tables used for emittance fitting.
"""

import os
import argparse
import pandas as pd

from wire_utils import parse_blob, load_wire_from_df, fit_sigma

# Wire pitch (mm per channel)
PITCH_MM = {
    "MW873": 1.0,
    "MW875": 0.5,
    "MW876": 0.5,
}


def fit_sigmas(data_dir, scans, wires, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for tag, fn in scans.items():
        path = os.path.join(data_dir, fn)
        print(f"Processing scan '{tag}': {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nCSV file not found:\n  {path}\n"
                f"Check --data-dir or --scans arguments."
            )

        df = pd.read_csv(path, converters={"value": parse_blob})

        for wn in wires:
            name = f"MW{wn}"
            pitch = PITCH_MM[name]
            wire = load_wire_from_df(df, wn)

            # Horizontal fit
            sigmaH_ch, poptH, perrH = fit_sigma(wire.horizontal_values)
            sigmaH_mm = sigmaH_ch * pitch

            # Vertical fit
            sigmaV_ch, poptV, perrV = fit_sigma(wire.vertical_values)
            sigmaV_mm = sigmaV_ch * pitch

            rows.append({
                "scan": tag,
                "csv_file": fn,
                "wire": name,
                "sigmaH_mm": sigmaH_mm,
                "sigmaV_mm": sigmaV_mm,
            })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "wire_sigma_by_scan.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian beam sizes from multiwire CSV data"
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing raw BNB beam study CSV files"
    )

    parser.add_argument(
        "--scans",
        nargs="+",
        required=True,
        help=(
            "List of scan definitions in the form tag=filename.csv "
            "(e.g. baseline=bnb_br_study_baseline_2025-06-16T152208Z.csv)"
        )
    )

    parser.add_argument(
        "--wires",
        nargs="+",
        type=int,
        default=[873, 875, 876],
        help="Wire numbers to analyze (default: 873 875 876)"
    )

    parser.add_argument(
        "--out-dir",
        default="../3_results/csv",
        help="Output directory for CSV files (default: ../3_results/csv)"
    )

    args = parser.parse_args()

    # Parse scan arguments into a dict
    scans = {}
    for s in args.scans:
        if "=" not in s:
            raise ValueError(
                f"Invalid scan format '{s}'. Expected tag=filename.csv"
            )
        tag, fn = s.split("=", 1)
        scans[tag] = fn

    fit_sigmas(
        data_dir=args.data_dir,
        scans=scans,
        wires=args.wires,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
