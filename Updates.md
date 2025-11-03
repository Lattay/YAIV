# Changelog

## ✅ General Improvements

OLD - AS TEMPLATE:
- Improved and expanded documentation across the entire code.
- New `phonon` module with phonon handling utilities.
- Added `tests`.

---

## 📦 Module-Specific Updates

### `grep`
- Added `grep.symmetires`, currently only supporting QE xml files.
- Expanded `grep.kpointsEnergies` to grep projection over orbitals from Quantum Espresso's `projwfc.x`

### `utils`
- Added `utils.symmetry_orbit_kpoints`, for applying all symmetry rotations to a set of k-points and returning a unique set.
- Added `find_little_group` for finding the little group of a given set of points.
- Now `voigt2cartesian` and `cartesian2voigt` can transform full arrays.
- Added `kernel_density` for  building a callable density(X) that returns the kernel-broadened density evaluated at arbitrary positions X.
- Added `auto_kgrid`, which allows to comput a k-grid from a target k-point spacing or target kpoints-per-reciprocal-atom (KPPRA).
- Added `kernel_regresion` to build callables that perform 1D kernel regression (Nadaraya–Watson estimator) from samples.

### `cell`
- The `write_espresso_in` method inside the `Cell` class now allows for overwritting the kpoints.
- `Cell.print()` to write the structure in human-readable format.

### `phonons`
- The `BOES.save_jobs_pwi` now allows for overwritting the kpoints. Usefull as different BOES arising from different phonons will usually share all computational parameters except for the kpoints.
- Added automatic kgrid scaling to `BOES.save_jobs_pwi`.
