# Changelog

## ✅ General Improvements

- Improved and expanded documentation across the entire code.
- New `phonon` module with phonon handling utilities.
- Added `tests`.

---

## 📦 Module-Specific Updates

### `utils`
- Added `methpax_delta` function.
- Added `analyze_distribution` function.
- Added `amplitude2order_parameter` to convert displacement amplitudes into proper order parameters with [length × sqrt(mass)] units.
- Added `cumulative_integral`, which computes the cumulative integral of a function defined by discrete x and y values.
- Added `kernel_density_on_grid`, that computes a kernel-broadened density on a grid.

### `grep`
- New `dyn_file` function for parsing a QE `.dyn` file and extracting vibrational data such as q-point, lattice vectors, phonon frequencies, etc.
- New `dyn_q` function for locating and reading a `.dyn*` file for a specific q-point. Returns the full 3N×3N dynamical matrix.
- Now `kpointsEnergies` also greps the **orbital resolved projections** of Bloch states if available (`PROCAR` files supported).
- Added support for QuantumEspresso `xml` files.

### `spectrum`
- Added `Density` class.
    - Supports computation of the density with both Gaussian and MP smearing (any order).
    - Supports density integration and upper integral limit calculation from the desired total integral.

### `plot`
- Added the option to highlight grid in the k-path for phonon bands.

### `cell`
- Added `write_espresso_in` method for generating QE input files from a `Cell` object.

### `phonon`
- Added `Dyn` class for handling:
    - Reading and constructing dynamical matrices,
    - Diagonalization,
    - Extraction of eigenvectors, displacement patterns, and phonon frequencies.
- Added `CDW` class for building charge-density wave (CDW) distorted supercells from multiple q-point phonon modes.
- Added `BOES` class to construct the Born-Oppenheimer energy surface associated with CDW distortions.
