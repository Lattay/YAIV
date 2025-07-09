# List of updates to add to the new release:

- Improved documentation

Utils:
    - New `methpax_delta` function.
    - New `analyze_distribution` function.

Grep:
    - New `dyn_file` for parsing a QE `.dyn` file and getting vibrational data (q-point (2π/Å), lattice (Å), frequencies, ...).
    - New `dyn_q` for locating and reading a `.dyn*` file for a given q-point, returning the full dynamical matrix (3Nx3N).

Spectrum:
    - New `DOS` class.
    - DOS can be computed with both Gaussian and MP smearing of any order.

Cell:
    - New `write_espresso_in` method.

Phonon:
    - New `Dyn` class for handling dynamical matrices, diagnoalizing, eigenvectors, displacement vectors, frequencies...
    - New `CDW` class for constructing and manipulating charge-density wave (CDW) distorted supercells from multiple q-point phonon modes.
    - New `BOES` class for constructing a Born-Oppenheimer energy surface (BOES) associated with charge-density wave (CDW) distortions.
