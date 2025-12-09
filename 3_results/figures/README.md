# Figures and Plots

This folder contains all visual outputs generated from the MAD-X + Python analysis pipeline. These figures represent the final scientific interpretation of the BNB narrow beam optics studies.

---

## Categories of Figures

### **1. Beta Function Scan Plots**
Examples:
- `betx_HQ874_readable.png`
- `betx_HQ875_readable.png`
- `betx_K1_scan_150m_onward.png`

**Show:**
- βx(s) evolution along the beamline
- Sensitivity to quadrupole current variations

**Purpose:**
- Diagnose focusing behavior
- Locate optics compression and expansion regions

---

### **2. Dispersion Scan Plots**
Examples:
- `betx_dx_zoom_around_200m.png`
- `betx_dx_HQ873_874_joint_variation.png`

**Show:**
- Joint evolution of βx and Dx
- Chromatic sensitivity of the optics

---

### **3. Emittance and Momentum Spread Plots**
Examples:
- `Q874_emittance95_geometric.png`
- `Q874_emittance95_normalized.png`

**Show:**
- Transverse beam quality versus quadrupole setting
- Stability and variation of ε and σδ

---

### **4. Beam Size at Wire Chambers**
Examples:
- `MAD-X_predicted___at_wires_vs_Q873_current.csv` (plotted)
- σx, σy trends at MW873, MW875, MW876

**Show:**
- Model vs measurement agreement
- Sensitivity of beam envelopes to optics changes

---

### **5. Target Beam Spot Prediction**
Examples:
- `target_sigma_prediction_scaled_D_Q873.csv` (plotted)
- Combined σx–σy beam spot area plots

**Show:**
- Direct optimization metric for BNB performance
- Identification of globally optimal optics settings

---

## Scientific Meaning of These Figures

These plots are used to conclude that:

- Q873 baseline provides the global σx minimum
- Q874 shows a local minimum near −27 A only
- Q875 has limited leverage on σx but affects σy
- Dispersion dominates σx behavior in several scan regions
- Target spot optimization must consider σx·σy, not βx alone

---

## Citation and Reuse

These figures may be reused for:
- Internal accelerator notes
- Run plan optimization reviews
- Neutrino flux reconstruction studies

Any reuse should cite this repository.

