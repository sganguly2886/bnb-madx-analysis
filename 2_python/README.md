# Python Analysis Scripts

This folder contains all Python scripts used to analyze the MAD-X Twiss outputs and extract physical beam parameters for the BNB narrow beam optics studies.

These scripts perform:
- β-function and dispersion extraction
- Emittance and momentum spread fitting
- Predicted beam size calculations at the wire chambers
- Target beam spot size predictions
- Generation of all figures and CSV result files

All scripts in this folder operate on the Twiss outputs stored in:

1_twiss_outputs/

and write numerical results and plots into:

3_results/

---

## Core Physics Model

All beam size predictions are based on the standard linear optics model:

\[
\sigma^2 = \varepsilon \beta + (\sigma_\delta D)^2
\]

where:

- σ = RMS beam size at the measurement location  
- ε = geometric RMS emittance  
- β = beta function from MAD-X  
- σδ = RMS fractional momentum spread  
- D = dispersion from MAD-X  

The vertical plane typically assumes \( D_y \approx 0 \).

---

## Main Analysis Scripts

### **1. `analyze_emittance_dispersion.py`**
**Purpose:**
- Fits horizontal and vertical emittance (εₓ, εᵧ)
- Fits momentum spread (σδ)
- Uses measured σ at MW873, MW875, MW876
- Uses β and D from Twiss files

**Outputs:**
- `emittance_vs_Q873_current.csv`
- `momentum_spread_vs_Q873_current.csv`
- Emittance and σδ trends vs quadrupole current

---

### **2. `cal_spotsize.py`**
**Purpose:**
- Uses fitted ε and σδ
- Propagates beam size to the BNB target (~207 m)
- Computes:
  \[
  \sigma_{x,t}, \sigma_{y,t}, A = \sigma_{x,t} \sigma_{y,t}
  \]

**Outputs:**
- `target_sigma_prediction_scaled_D_Q873.csv`
- `target_sigma_prediction_scaled_D_Q874.csv`
- `target_sigma_prediction_scaled_D_Q875.csv`

---

### **3. `plot_twiss_updated.py`**
**Purpose:**
- Extracts βx, βy, Dx, Dy from Twiss files
- Evaluates optical response to quadrupole scans

**Outputs:**
- `betax_vs_Q873_current.csv`
- `dispersion_vs_Q873_current.csv`
- Corresponding optics plots

---

### **4. `plot_betax_dx.py`**
**Purpose:**
- Produces joint βx and Dx visualization
- Highlights optics sensitivity regions
- Used to identify focusing-dispersion tradeoffs

**Outputs:**
- Combined βx–Dx scan figures

---

### **5. `run_HQ873.py`, `run_HQ874.py`, `run_HQ875.py`**
**Purpose:**
- Automate MAD-X execution for each quadrupole scan
- Loop over current or K1 strength variations
- Generate full Twiss output datasets

---

## Script Execution Order (Recommended)

To fully reproduce the analysis:

```bash
python run_HQ873.py
python run_HQ874.py
python run_HQ875.py

python plot_twiss_updated.py
python analyze_emittance_dispersion.py
python cal_spotsize.py


Notes

All scripts assume a working MAD-X installation.

Python dependencies typically include:

numpy

pandas

matplotlib

No machine-specific data acquisition is required to rerun the optical analysis.

Scripts are written for physics transparency rather than execution speed.
