BNB MAD-X / Q873 Optics & Emittance Analysis
===========================================

This repository contains Python scripts and supporting utilities for analyzing
BNB beam optics, wire scanner data, emittance, momentum spread, and target beam
spot size as a function of Q873 quadrupole current.

The Q873 workflow has been migrated from exploratory Jupyter notebooks to
standalone Python scripts to ensure reproducibility, clarity, and
version-controlled physics results.


Requirements
------------

Python ≥ 3.9

Required packages:
- numpy
- pandas
- matplotlib

Install with:
pip install numpy pandas matplotlib


External Data
-------------

Raw beam data and wire scan inputs are typically not stored in this repository.

If you have local raw inputs, create a symbolic link:
ln -s /path/to/bnb_br_study external_data

Scripts resolve paths relative to this directory.


Quick Start (Reproduce All Q873 Results)
---------------------------------------

cd 2_python

1. Optics (β, D) vs Q873 current
python plot_Q873_optics_vs_current.py

2. Measured vs predicted wire sigmas
python plot_Q873_sigmas.py

3. Emittance & momentum spread extraction
python plot_Q873_95_emittance_and_Dp.py

4. Scaled dispersion & target beam spot prediction
python plot_Q873_scaled_dispersion_and_target_sigma.py

Scripts will display plots interactively and/or write outputs.
When saved, figures go to:
3_results/figures/

and tables/CSVs go to:
3_results/csv/


Optics vs Q873 Current
---------------------

Script:
plot_Q873_optics_vs_current.py

Description:
- Loads MAD-X Twiss files for each Q873 current
- Extracts βx, βy, Dx, Dy at MW873, MW875, MW876
- Plots optics functions vs Q873 current


Wire Sigma Comparison
---------------------

Script:
plot_Q873_sigmas.py

Description:
- Compares measured wire scanner σ to MAD-X predictions
- Validates optics trends prior to emittance fitting

### Measured Beam Spot Size Plots

Script:
plot_measured_sigmas_vs_quads.py

Description:
- Plots measured wire-scanner beam sizes (σx, σy)
- Covers scans of Q873, Q874, Q875, and combined Q873+Q874 settings
- Data shown are measurement-derived, not MAD-X predictions
- Used to validate optics trends and locate empirical minima

Outputs:
- Four figures showing σ vs quadrupole settings
- MW873, MW875, MW876 in both planes


Emittance & Momentum Spread Fitting
-----------------------------------

Script:
plot_Q873_95_emittance_and_Dp.py

Beam size model:
σ² = ε·β + (σδ·D)²

Outputs:
- RMS geometric emittance εx, εy
- 95% geometric emittance
- 95% normalized emittance
- RMS momentum spread σδ
- Absolute momentum spread Δp

Output CSV files:
- emittance_vs_Q873_current_with_errors.csv
- emittance_vs_Q873_current_pi-mm-mrad.csv


Scaled Dispersion & Target Beam Spot Prediction
-----------------------------------------------

Script:
plot_Q873_scaled_dispersion_and_target_sigma.py

Description:
- Inverts measured σ at MW873/875/876 to estimate dispersion
- Scales MAD-X dispersion using median ratios
- Propagates fitted emittance and momentum spread to the BNB target (~207 m)
- Predicts σx(target) and σy(target) vs Q873 current

Output CSV files:
- Dx_estimates_vs_current.csv
- Dy_estimates_vs_current.csv
- target_sigma_prediction_scaled_D.csv


Utility Modules
---------------

tfs_utils.py
Robust MAD-X TFS file reader and element selection helpers

wire_utils.py
Shared wire scanner fitting and plotting utilities


Jupyter Notebooks (Archive Only)
--------------------------------

The following notebooks are retained for historical reference only:

- Q873_betax_vs_current.ipynb
- Q874_betax_vs_current.ipynb
- Q875_betax_vs_current.ipynb

All authoritative physics logic for Q873 is in the standalone Python scripts.


Notes & Conventions
-------------------

- Emittances use the FNAL convention:
  95% ≈ 6 × RMS

- Normalized emittance:
  εn = βγ · ε

- Beam momentum is assumed to be 8.8349 GeV/c unless otherwise stated


Maintainer Notes
----------------

- Do not commit backup files (*.py~) or __pycache__/
- External raw inputs should remain outside version control (use symlinks)
- CSV outputs represent derived physics results and should be committed


Contact
-------

Sudeshna Ganguly @ sganguly@fnal.gov
