# Changelog

## ✅ General Improvements

- Improved and expanded documentation across the entire codebase.

---

## 📦 Module-Specific Updates

### `utils`
- Added `methpax_delta` function.
- Added `analyze_distribution` function.

### `grep`
- New `dyn_file` function for parsing a QE `.dyn` file and extracting vibrational data such as q-point (in 2π/Å), lattice vectors (in Å), phonon frequencies, etc.
- New `dyn_q` function for locating and reading a `.dyn*` file for a specific q-point. Returns the full 3N×3N dynamical matrix.

### `spectrum`
- Added `DOS` class.
- Now supports computation of the density of states with both **Gaussian** and **MP smearing** (any order).

### `cell`
- Added `write_espresso_in` method for generating QE input files from a `Cell` object.

### `phonon`
- Added `Dyn` class for handling:
    - Reading and constructing dynamical matrices,
    - Diagonalization,
    - Extraction of eigenvectors, displacement patterns, and phonon frequencies.
- Added `CDW` class for building charge-density wave (CDW) distorted supercells from multiple q-point phonon modes.
- Added `BOES` class to construct the **Born-Oppenheimer Energy Surface (BOES)** associated with CDW distortions.
