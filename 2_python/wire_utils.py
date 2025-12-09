#!/usr/bin/env python3
"""
Utilities for decoding multi–wire scanner blobs, Gaussian fitting,
and preparing MW873/875/876 profiles.

Used by:
    - plot_wire_profiles.py
    - fit_wire_sigmas.py
    - plot_collimator_jaws.py
"""

import ast
import numpy as np
from scipy.optimize import curve_fit
from wires.wire import Wire


# ------------------------------------------------------------
# Parse raw blob from CSV (string "b'...'" → bytes)
# ------------------------------------------------------------
def parse_blob(x):
    if isinstance(x, (bytes, bytearray)):
        return x
    return ast.literal_eval(x)


# ------------------------------------------------------------
# Gaussian model
# ------------------------------------------------------------
def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ------------------------------------------------------------
# Fit Gaussian to histogram counts
# ------------------------------------------------------------
def fit_sigma(counts):
    """
    Fit Gaussian to wire profile.
    Returns (sigma_ch, popt, perr).
    Falls back to RMS if fit fails.
    """
    y = np.asarray(counts, float)
    x = np.arange(len(y))
    tot = y.sum()

    if tot <= 0:
        return np.nan, None, None

    # RMS initial guesses
    mu0 = (y * x).sum() / tot
    sigma0 = np.sqrt(((x - mu0) ** 2 * y).sum() / tot)
    A0 = y.max()

    p0 = [A0, mu0, sigma0]
    bounds = ([0, 0, 0], [np.inf, len(y), len(y)])

    try:
        popt, pcov = curve_fit(gauss, x, y, p0=p0,
                               bounds=bounds, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return popt[2], popt, perr
    except:
        return sigma0, None, None


# ------------------------------------------------------------
# Load a single wire (MW873/875/876) from CSV
# ------------------------------------------------------------
def load_wire_from_df(df, wire_number):
    wn = str(wire_number)
    pos_blob = parse_blob(df.loc[df.device == f"E:M{wn}PO", "value"].iloc[0])
    scn_blob = parse_blob(df.loc[df.device == f"E:M{wn}DS", "value"].iloc[0])

    wire = Wire.deserialize(
        name=f"MW{wn}",
        sample_event=1,
        sample_date=None,
        position_data=pos_blob,
        scanner_data=scn_blob
    )
    return wire
