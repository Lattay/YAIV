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

### `utils`
- Added `utils.symmetry_orbit_kpoints`, for applying all symmetry rotations to a set of k-points and returning a unique set.
- Added `find_little_group` for finding the little group of a given set of points.
- Now `voigt2cartesian` and `cartesian2voigt` can transform full arrays.
