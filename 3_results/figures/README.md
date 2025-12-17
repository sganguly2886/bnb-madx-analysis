# Final Figures — Q873 Optics Scan

This directory contains the **final, curated figures** produced from the
MAD-X + Python analysis of the **Q873 quadrupole current scan** in the BNB
narrow beam optics study.

All figures in this folder correspond to results that are:
- quantitatively validated,
- consistent with measured wire data,
- suitable for internal notes, reviews, and archival reference.

---

## Contents

### **1. Horizontal Beta Function vs Q873 Current**
**File:** `Q873_BETX_vs_current.png`

**Shows:**
- βx evaluated at MW873, MW875, MW876
- Dependence on Q873 current setting

**Purpose:**
- Identify focusing trends
- Demonstrate that βx alone does *not* fully explain beam size behavior

---

### **2. Horizontal Dispersion vs Q873 Current**
**File:** `Q873_DX_vs_current.png`

**Shows:**
- Dx at MW873, MW875, MW876
- Strong current-dependent dispersion modulation

**Purpose:**
- Explain σx variation observed in wire data
- Establish dispersion as the dominant driver of beam size changes

---

### **3. Vertical Dispersion vs Q873 Current**
**File:** `Q873_DY_vs_current.png`

**Shows:**
- Dy at MW873, MW875, MW876

**Purpose:**
- Confirm weak vertical chromatic effects
- Support assumption that σy is largely emittance-dominated

---

## How These Figures Are Generated

All figures are produced by:

2_python/plot_Q873_optics_vs_current.py


which reads MAD-X TFS files from `1_twiss_outputs/` and extracts optics
functions at the three downstream wire chambers.

No manual editing is performed on the plots.

---

## Relationship to Other Results

These figures are used in conjunction with:

- **Emittance extraction**
  - `3_results/csv/emittance_vs_Q873_current*.csv`
- **Momentum spread estimation**
  - `momentum_spread_vs_Q873_current.csv`
- **Scaled dispersion & target beam size prediction**
  - `target_sigma_prediction_scaled_D.csv`

Together, these establish a complete and self-consistent interpretation of
the Q873 scan.

---

## Scientific Conclusions Supported

From these figures and associated analysis:

- Q873 baseline is near the global σx minimum
- Dispersion, not βx alone, drives σx variation
- Vertical optics are comparatively stable
- Target optimization requires σx–σy consideration, not βx minimization

---

## Reuse and Citation

These figures may be reused for:
- BNB optics notes
- Run plan reviews
- Internal accelerator documentation

Please cite this repository when reusing the plots.
