import numpy as np

from yaiv.defaults.config import ureg
from yaiv import utils as ut
from yaiv import grep


__all__ = ["distort_crystal"]


def _QEdyn2Realdyn(
    dyn_mat: np.ndarray | ureg.Quantity, masses: np.ndarray | ureg.Quantity
) -> np.ndarray | ureg.Quantity:
    """
    Convert Quantum ESPRESSO dynamical matrix to the real dynamical matrix in physical units.

    In Quantum ESPRESSO, the dynamical matrix is written in units of [energy/h]^2 * [mass].
    To obtain the true dynamical matrix, whose eigenvalues are squared phonon frequencies (ν²) in
    units of [energy/h]^2, one must divide by √(m_i * m_j) for each atomic block.

    Parameters
    ----------
    dyn_mat : np.ndarray | ureg.Quantity
        The 3N×3N dynamical matrix as written by Quantum ESPRESSO.

    masses : np.ndarray | ureg.Quantity
        Array of atomic masses (length N), one per atom. Units should be consistent
        with those implied in `dyn_mat`. In QE, masses are typically given in units of 2 m_e.

    Returns
    -------
    np.ndarray or Quantity
        The real dynamical matrix with proper normalization, whose eigenvalues
        have units of squared frequencies, e.g., [Ry²/h²] or [s⁻²], depending on units.

    Raises
    ------
    ValueError
        If input shapes are inconsistent or mass array is not one-dimensional.
    """
    # Check unit consistency
    ut._check_unit_consistency([dyn_mat, masses], ["Dynamical matrix", "Atomic masses"])

    # Strip units if present
    if isinstance(dyn_mat, ureg.Quantity):
        units = dyn_mat.units / masses.units
        dyn_mat = dyn_mat.magnitude
        masses = masses.magnitude
    else:
        units = 1

    # Sanity checks
    masses = np.asarray(masses)
    dyn_mat = np.asarray(dyn_mat)

    if masses.ndim != 1:
        raise ValueError("Masses must be a 1D array (one mass per atom).")
    if dyn_mat.shape[0] != dyn_mat.shape[1]:
        raise ValueError("Dynamical matrix must be square.")
    if dyn_mat.shape[0] != 3 * len(masses):
        raise ValueError("Dynamical matrix shape is incompatible with number of atoms.")

    # Transform matrix
    N = len(masses)
    dyn = np.copy(dyn_mat)
    for n in range(N):
        for m in range(N):
            i = 3 * n
            j = 3 * m
            factor = np.sqrt(masses[n] * masses[m])
            dyn[i : i + 3, j : j + 3] /= factor

    return dyn * units


def distort_crystal(q_points, results_ph, order_parameter, modes, amplitude):
    print("TODO")
    # 1. Read crystal structure
    # 2. Get dynamical matrices
    # 3. Diagonalize matrices
    # 4. Get displacement vectors
    # 5. Find conmmensurate supercell
    # 6. Get proper phase factors for each cell
    # 7. Make Supercell displacement vectors combining phase factors and displacement vectors
    # 8. Add the displacements to create a supercell final displacement
    # 9. Apply displacement
    # 10. Return final cell

    pass
