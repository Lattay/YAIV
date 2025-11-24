# Changelog

## ✅ General Improvements

- Fermi-Dirac included as a possible smearing for all methods (along with Gaussian and Methfessel-Paxton).
- New `convergence` module with tools to collect, organize, and visualize convergence data from ab initio calculations.
---

## 📦 Module-Specific Updates

### `grep`
- Energy decomposition of `grep.total_energy` now also works for fixed occuaption calcualtions.
- Added `grep.cutoff` to grep the cutoff energy used in calculations .
- Added `grep.smearing` to get smearings used in calculations.
- Added `grep.runtime` to get computational runtimes.
- Added `grep.ram` to get required RAM memory in calculations.
- Added `grep.k_grid` to get the k-grid used in calculations.
- Added `grep.atomic_forces` to grep atomic forces.

### `utils`
- Include and implement a `utils.fermidirac_kernel` accross different utilities in order to get Fermi-Dirac statistics (along with Gaussian and Methfessel-Paxton).

### `spectrum`
- Implement Fermi-Dirac smearing for the calculation of the Densities.
