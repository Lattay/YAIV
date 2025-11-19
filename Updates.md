# Changelog

## ✅ General Improvements

- Fermi-Dirac smearing is now included as a possible smearing for all methods (along with Gaussian and Methfessel-Paxton).

---

## 📦 Module-Specific Updates

### `grep`
- Energy decomposition of `grep.total_energy` now also works for fixed occuaption calcualtions.

### `utils`
- Include and implement a `utils.fermidirac_kernel` accross different utilities in order to get Fermi-Dirac statistics (along with Gaussian and Methfessel-Paxton).

### `spectrum`
- Implement Fermi-Dirac smearing for the calculation of the Densities.

