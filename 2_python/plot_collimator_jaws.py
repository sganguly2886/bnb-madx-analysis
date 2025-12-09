#!/usr/bin/env python3
"""
Reproduces the 'Collimator jaw position vs time' plot.
"""

import os
import ast
import pandas as pd
import matplotlib.pyplot as plt


def parse_blob(x):
    if isinstance(x, (bytes, bytearray)):
        return x
    return ast.literal_eval(x)


def main():
    data_dir = "../0_raw_data"
    fn = "bnb_br_study_br_collimation_2025-06-16T175101Z.csv"
    df = pd.read_csv(os.path.join(data_dir, fn),
                     parse_dates=["timestamp"],
                     converters={"value": parse_blob})

    mask = df.device.str.match(r"I:C83[68][AB][HV]$")
    coll = df[mask].copy()

    coll["mills"] = coll["value"].astype(float)
    wide = coll.pivot(index="timestamp", columns="device", values="mills")
    wide_mm = wide * 0.001 * 25.4

    plt.figure(figsize=(10, 4))
    for col in wide_mm.columns:
        plt.plot(wide_mm.index, wide_mm[col], label=col)

    thr = 0.250 * 25.4
    plt.axhline(thr, color="k", ls="--")
    plt.axhline(-thr, color="k", ls="--")

    plt.title("Collimator Jaw Positions vs Time")
    plt.ylabel("Jaw Position [mm]")
    plt.xlabel("Time")
    plt.legend(ncol=2)
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    out = "3_results/figures/collimator_jaws_vs_time.png"
    plt.savefig(out, dpi=150)
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()
