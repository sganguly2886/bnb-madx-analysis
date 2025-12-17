# Python Analysis Scripts

This folder contains all Python and Jupyter scripts used to process MAD-X Twiss outputs, analyze measured multiwire beam data, extract emittance and momentum spread, predict the target beam spot size, and generate all final figures used in the BNB narrow beam optics study.

This directory represents the full numerical and experimental reconstruction chain.

---

## Analysis Pipeline Overview

The full physics workflow implemented by these scripts is:

MAD-X lattices  
→ Twiss outputs (`1_twiss_outputs/`)  
→ Optics extraction  
→ Wire profile fitting  
→ Emittance & momentum spread fitting  
→ Target beam spot prediction  
→ Final physics plots (`3_results/figures/`)

All numerical conclusions and plots in this repository are derived from this pipeline.

---

# 1. Twiss Processing and Optics Extraction

These scripts read MAD-X Twiss output files and extract optical functions at the wire chambers (MW873, MW875, MW876).

### Scripts:
- (From earlier pipeline, Twiss inputs already exist in `1_twiss_outputs/`)

### Output CSV files written to:3_results/csv/


### Example CSV outputs:
- `betax_vs_Q873_current.csv`
- `dispersion_vs_Q873_current.csv`
- Corresponding HQ874 and HQ875 CSV files

These CSVs are used by all downstream fitting and plotting scripts.

---

# 2. Wire Profile Extraction and Gaussian Fitting

These scripts operate directly on **measured multiwire CSV beam data**.

### Core Utility
- `wire_utils.py`  
  Shared functions for:
  - Parsing binary multiwire blobs  
  - Gaussian fitting  
  - RMS fallback fitting  

### Wire Profile Plotting
- `plot_wire_profiles.py`  
  Generates full horizontal and vertical profiles for:
  - MW873
  - MW875
  - MW876

### Wire Sigma Extraction
- `fit_wire_sigmas.py`  
  Fits Gaussian widths (σ) for:
  - Horizontal and vertical beam sizes  
  - All wires  
  - All scan files  

### Output CSV files:3_results/csv/wire_sigmas_*.csv


These σ values are direct experimental inputs for emittance fitting.

---

# 3. Emittance and Momentum Spread Fitting

These scripts solve the beam size model:

\[
\sigma^2 = \varepsilon \cdot \beta + (\sigma_\delta \cdot D)^2
\]

Where:
- σ = RMS beam size from wire fits  
- ε = geometric RMS emittance  
- β = beta function from MAD-X  
- σδ = RMS fractional momentum spread  
- D = dispersion  

### Scripts:
- `analyze_emittance_dispersion_Q873.py`
- `analyze_emittance_dispersion_Q874.py`
- `analyze_emittance_dispersion_Q875.py`

### Extracted Parameters:
- εx, εy  
- σδ  

### Output CSV files:3_results/csv/


Example:
- `emittance_vs_Q873_current.csv`
- `momentum_spread_vs_Q873_current.csv`
- Corresponding HQ874 and HQ875 files

---

# 4. Target Beam Spot Size Prediction

These scripts propagate fitted beam parameters to the BNB target (~207 m downstream).

### Target Spot Calculation Scripts:
- `cal_spotsize_Q873.py`
- `cal_spotsize_Q874.py`
- `cal_spotsize_Q875.py`

### These compute:
- σx(target)
- σy(target)
- Effective 2D beam spot area:
  
\[
A = \sigma_x \cdot \sigma_y
\]

### Output CSV files:
3_results/csv/target_sigma_prediction_scaled_D_Q873.csv
3_results/csv/target_sigma_prediction_scaled_D_Q874.csv
3_results/csv/target_sigma_prediction_scaled_D_Q875.csv


---

# 5. Plotting and Final Visualization Scripts

These scripts generate all physics plots used in the talk.

### Target Spot Area vs Current
- `plot_target_area_Q873.py`
- `plot_target_area_Q874.py`
- `plot_target_area_Q875.py`

### Collimator Jaw Motion
- `plot_collimator_jaws.py`

### Wire Profile Visualization
- `plot_wire_profiles.py`

### All figures are saved to:3_results/figures/


These include:
- Beta-function scan plots  
- Dispersion scan plots  
- Emittance and momentum spread plots  
- Wire chamber beam size plots  
- Target beam spot optimization plots  

---
### Collimator Jaw Motion (External Raw Data Required)

- `plot_collimator_jaws.py`

This script plots the time evolution of BNB collimator jaw positions during
beamline operation.

⚠️ **Important:**  
This script requires **external raw BNB beamline CSV data** that is **not**
stored in this repository.

The required input is a CSV file produced by the BNB BR study data acquisition,
for example:

- `bnb_br_study_br_collimation_2025-06-16T175101Z.csv`

#### Usage Example

```bash
cd 2_python

python plot_collimator_jaws.py \
  --data-dir /path/to/bnb_br_study_2025-06-16
The script will:

Read the raw collimator jaw channels (I:C836*, I:C838*)

Convert jaw positions to mm

Plot jaw motion vs time

Save the figure to:
3_results/figures/collimator_jaws_vs_time.png




# 6. Original Jupyter Notebooks (Archive / Reference)

These notebooks contain the **original interactive analysis used during the study** and serve as archival references.

- `Q873_betax_vs_current.ipynb`
- `Q874_betax_vs_current.ipynb`
- `Q875_betax_vs_current.ipynb`

All physics logic from these notebooks has now been fully migrated into the standalone `.py` scripts above for full reproducibility.

---

# Output Directory Structure

All numerical outputs:3_results/csv/

All plots:3_results/figures/


---

#  Software Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy

---

#  Important Reproducibility Notes

- These scripts assume MAD-X Twiss files already exist in:1_twiss_outputs/


- All final scientific conclusions in this repository are derived solely from:
  - `2_python/`
  - `3_results/`

- The Jupyter notebooks are retained for traceability only.
- The `.py` scripts define the **official reproducible pipeline**.

---










