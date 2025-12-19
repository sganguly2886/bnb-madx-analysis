# MAD-X Lattice Files

This folder contains the complete set of MAD-X input lattice files used for the
BNB narrow beam optics studies. These files define the accelerator optics model
used throughout the full analysis chain.

These lattice files define:

- The full BNB beamline geometry  
- Quadrupole strengths and scan configurations  
- Momentum and optics variations  
- Fixed and scanned K1 configurations for HQ873, HQ874, and HQ875  

All beam size predictions, emittance extraction, and target spot size optimization
ultimately depend on these optics definitions.

---

## File Naming Convention

### Baseline Lattices

These represent the nominal operating configurations used in all studies:

- bnbnew_HQ873_default.madx, bnbnew_HQ873_874_default.madx, bnbnew_HQ875K1_default.madx 
  Default HQ873 horizontal focusing configuration. This is the reference optics. 

---

## Quadrupole Scan Files

These files apply systematic current or K1 variations to study optics sensitivity.

---

### HQ873 Scans (Horizontal Focusing)

Files starting with:

- bnbnew_HQ873_*  
- bnbnew_HQ873K1_*  

These are used to study how Q873 affects:

- Horizontal beta function (beta_x) at MW873, MW875, and MW876  
- Horizontal dispersion (D_x)  
- Horizontal beam spot size at the target  

---

### HQ874 Scans (Dispersion-Dominated Studies)

Files included in this repository:

- bnbnew_IQ874_default.madx  
- bnbnew_IQ874_10percentup.madx  
- bnbnew_IQ874_10percentdown.madx  
- bnbnew_IQ874_20percentup.madx  
- bnbnew_IQ874_20percentdown.madx  

These explore:

- Sensitivity of horizontal dispersion to HQ874 current  
- Dispersion-driven beam size growth  
- Coupled behavior with HQ873 through emittance extraction  

---

## How These Files Are Used

These lattice files are executed using:

./madx < filename.madx

They generate Twiss output files (.tfs, .out) that are then analyzed using the
Python scripts in:

2_python/

---

## Important Notes

- These lattice files are not raw data; they represent the physics model used
  throughout the analysis.
- All beam size predictions at the target ultimately depend on these optics
  definitions.
- Any change in these files directly affects:

  - Beta functions  
  - Dispersion  
  - Emittance extraction  
  - Target spot size predictions  

