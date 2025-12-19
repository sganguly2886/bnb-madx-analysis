# Twiss Output Files (MAD-X)

This directory contains the complete set of **MAD-X Twiss table outputs (`.tfs`)**
used in the BNB narrow beam optics studies.

These files are the **primary numerical inputs** to all downstream Python analysis,
including emittance extraction, dispersion fitting, momentum spread determination,
and target beam spot size prediction.

All results in `2_python/` and `3_results/` ultimately derive from the optics
described in these Twiss files.

---

## What These Files Contain

Each `.tfs` file includes lattice-wide optics information such as:

- Horizontal and vertical beta functions: βx(s), βy(s)
- Horizontal and vertical dispersion: Dx(s), Dy(s)
- Phase advance and optical functions
- Element-by-element beamline description
- Response to quadrupole current variations

These quantities are evaluated at:
- MW873, MW875, MW876 (wire chambers)
- The BNB target region (~207 m)

---

## Naming Convention

All files follow a consistent pattern:

twiss_<QUAD>_<SETTING>[_flipPol].tfs


### Examples
- `twiss_Q873_default.tfs`
- `twiss_Q873_-12A_flipPol.tfs`
- `twiss_Q874_baseline_-27a_flipPol.tfs`
- `twiss_Q875_baseline_default_flipPol.tfs`

### Polarity Convention
- Files containing `_flipPol` correspond to the **final polarity convention**
  used throughout the published analysis.
- Legacy non-`flipPol` files are retained **only for reference and validation**.
- All Python scripts preferentially use `flipPol` files when available.

---

## Q873 Scan Outputs (Primary Horizontal Study)

Files prefixed with:

twiss_Q873_*.tfs


These represent systematic current scans of the Q873 quadrupole and are used to:

- Extract εx and εy emittance versus current
- Fit horizontal momentum spread (σδ) using dispersion
- Study βx and Dx sensitivity at MW873/875/876
- Predict σx and σy at the BNB target

Both baseline and ±A current steps are included.

---

## Q874 Scan Outputs (Dispersion-Dominated Studies)

Files prefixed with:


twiss_Q874_*.tfs


These scans explore:

- Dispersion-driven beam size growth
- Local optics minima (e.g. near −27 A)
- Coupling effects between Q873 and Q874
- Differences between polarity conventions

Both `flipPol` and `posPol` baselines are archived for completeness.

---

## Q875 Scan Outputs (Vertical Focusing)

Files prefixed with:

twiss_Q875_*.tfs



These are used to study:

- Vertical focusing response
- σy stability at the target
- Limited leverage of Q875 on σx

Only fixed-K1 configurations are used in the final analysis.

---

## How These Files Are Used

The Python scripts in `2_python/` read these Twiss files to compute:

1. **Optical functions** at MW873, MW875, MW876  
2. **Emittance extraction** using measured wire chamber beam sizes  
3. **Momentum spread (σδ)** via dispersion fits  
4. **Scaled dispersion models** versus quadrupole current  
5. **Predicted target beam spot sizes**

The core beam size model used throughout is:

\[
\sigma^2 = \varepsilon \beta + (\sigma_\delta D)^2
\]
---

## Data Policy

- These files are physics model outputs, not experimental raw data.
- They are included to ensure full transparency and reproducibility.
- Large exploratory or deprecated scan sets have been intentionally excluded.
