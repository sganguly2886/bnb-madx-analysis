# MAD-X Lattice Files

This folder contains all MAD-X input files used for the BNB narrow beam optics studies.

These lattice files define:
- The full BNB beamline geometry
- Quadrupole strengths and scan configurations
- Momentum and optics variations
- Fixed and scanned K1 configurations for HQ873, HQ874, and HQ875

---

## File Naming Convention

### Baseline Lattices
These represent the nominal operating configurations:

- `bnbnew_default.madx`
- `bnbnew_fixed.madx`
- `bnbnew_HQ873_default.madx`
- `bnbnew_HQ874_default.madx`
- `bnbnew_HQ875_default.madx`

---

### Quadrupole Scan Files

These files apply systematic current or K1 variations:

#### HQ873 Scans
Files starting with:
bnbnew_HQ873_*
bnbnew_HQ873K1_*

These are used to study how Q873 affects:
- βx at MW873, MW875, MW876
- Dispersion Dx
- Beam spot size at the target

---

#### HQ874 Scans
Files starting with:
bnbnew_HQ874_*
bnbnew_IQ874_*
bnbnew_HQ873_874_*

These explore:
- Local vs global optics minima
- Coupled behavior between Q873 and Q874
- Sensitivity of dispersion to HQ874 current

---

#### HQ875 Scans
Files starting with:
bnbnew_HQ875_*
bnbnew_IQ875_*

These probe:
- Vertical focusing behavior
- Stability of σy at the target
- Fixed-BETX optimization studies

---

## Percentage Variation Files

Files with names like:
bnbnew_10percentup_*
bnbnew_20percentdown_*
bnbnew_30percentup_*

represent systematic strength scaling studies to explore:
- Optics robustness
- Sensitivity to power supply drift
- Global lattice stability

---

## How These Files Are Used

These lattice files are executed using:

```bash
madx < filename.madx
They generate Twiss output files (.tfs, .out) that are then analyzed using the Python scripts in:
2_python/
Important Notes

These lattice files are not raw data; they represent the physics model used throughout the analysis.

All beam size predictions at the target ultimately depend on these optics definitions.

Any change in these files directly affects:

β-functions

Dispersion

Emittance extraction

Target spot size predictions

