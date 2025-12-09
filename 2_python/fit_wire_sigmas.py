#!/usr/bin/env python3
"""
Fit Gaussian σx and σy for MW873/875/876.

Reproduces:
    - All σx, σy used in emittance fits
    - Beam size tables in the talk
"""

import os
import pandas as pd
from wire_utils import parse_blob, load_wire_from_df, fit_sigma

PITCH_MM = {"MW873": 1.0, "MW875": 0.5, "MW876": 0.5}


SCANS = {
    "baseline":   "bnb_br_study_baseline_2025-06-16T152208Z.csv",
    "narrow":     "bnb_br_study_br_narrow_2025-06-16T170402Z.csv",
    "collimation": "bnb_br_study_br_collimation_2025-06-16T175101Z.csv",
}


def main():
    data_dir = "../0_raw_data"
    rows = []

    for tag, fn in SCANS.items():
        df = pd.read_csv(os.path.join(data_dir, fn),
                         converters={"value": parse_blob})

        for wn in ["873", "875", "876"]:
            wire = load_wire_from_df(df, wn)
            name = f"MW{wn}"
            pitch = PITCH_MM[name]

            # horizontal
            sigmaH_ch, poptH, perrH = fit_sigma(wire.horizontal_values)
            sigmaH_mm = sigmaH_ch * pitch

            # vertical
            sigmaV_ch, poptV, perrV = fit_sigma(wire.vertical_values)
            sigmaV_mm = sigmaV_ch * pitch

            rows.append({
                "scan": tag,
                "wire": name,
                "sigmaH_mm": sigmaH_mm,
                "sigmaV_mm": sigmaV_mm,
            })

    out = pd.DataFrame(rows)
    out.to_csv("3_results/csv/wire_sigma_by_scan.csv", index=False)
    print("Saved 3_results/csv/wire_sigma_by_scan.csv")


if __name__ == "__main__":
    main()
