"""
Functions
---------

read_dyn_file(file)
    Parses a QE `.dyn` file and returns vibrational data (q-point (2π/Å), lattice (Å), frequencies, ...)

read_dyn_q(q_cryst, results_ph_path, qe_format=True)
    Locates and reads `.dyn*` file for a given q-point, returning the full dynamical matrix (3Nx3N).

_find_dyn_file(q_cryst, results_ph_path)
    Internal helper that searches `.dyn*` files matching the specified q-point.
"""

import re
from types import SimpleNamespace

import numpy as np
import glob

from yaiv.defaults.config import ureg
from yaiv import grep
from yaiv import utils as ut
from yaiv.dev import phonon as ph


__all__ = ["read_dyn_file", "read_dyn_q"]


def read_dyn_file(file: str) -> SimpleNamespace:
    """
    Parse a dynamical matrix file and extract phonon mode information.

    This function extracts:
    - the lattice vectors,
    - the atomic species and their masses,
    - the atom types and positions,
    - the q-point at which the phonon modes are computed,
    - the phonon frequencies (in cm⁻¹),
    - and the polarization vectors (phonon eigenvectors).

    Parameters
    ----------
    file : str
        Path to the dynamical matrix file (e.g. QE `.dyn` or `.dynmat` file).

    Returns
    -------
    SimpleNamespace
        A container with the following fields:
        - q : ureg.Quantity, shape (3,)
            The q-point in where the calculation was performed.
        - lattice : ureg.Quantity, shape (3, 3)
            The lattice vectors of the unit cell.
        - freqs : ureg.Quantity, shape (n_modes,)
            Array of vibrational frequencies (in cm⁻¹).
        - displacements : np.ndarray, shape(n_modes,n_atoms,3)
            An (n_modes, n_atoms, 3) array of complex normalized displacement vectors for each mode.
        - positions : ureg.Quantity, shape (n_atoms, 3)
            Atomic positions.
        - elements : list of str
            Chemical symbol for each atom.
        - masses : ureg.Quantity, shape (n_atoms,)
            Atomic mass in atomic units for each atom.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """
    filetype = grep._filetype(file)
    if filetype != "qe_dyn":
        raise NotImplementedError("Unsupported filetype")
    lattice = grep.lattice(file)
    n_atoms = n_types = freqs = vec = alat = None
    species = []
    atoms = []
    displacements = []
    read_modes = False
    with open(file, "r") as lines:
        for line in lines:
            l = line.split()
            if n_atoms is None and len(l) == 9:
                # Get number of species and atoms
                l = line.split()
                n_types, n_atoms = int(l[0]), int(l[1])
                alat = float(l[3]) * ureg("bohr")
            elif n_types != 0 and len(l) == 4:
                # Get species
                new = [int(l[0]), l[1][1:], float(l[-1])]
                species.append(new)
                n_types -= 1
            elif n_atoms != 0 and len(l) == 5:
                # Get atomic positions
                new = [int(l[1])] + [float(x) for x in l[2:]]
                atoms.append(new)
                n_atoms -= 1
            elif "Diagonalizing" in line:
                read_modes = True
            elif read_modes:
                # Read modes
                if "q = (" in line:
                    q_point = np.array([float(x) for x in l[3:6]])
                elif "freq" in line:
                    new = float(l[-2])
                    freqs = (
                        np.array([new]) if freqs is None else np.hstack([freqs, new])
                    )
                elif len(l) == 8:
                    nums = l[1:-1]
                    new = [
                        complex(float(nums[i]), float(nums[i + 1]))
                        for i in range(0, 6, 2)
                    ]
                    vec = np.array([new]) if vec is None else np.vstack([vec, new])
                    if len(vec) == len(atoms):
                        displacements.append(vec)
                        vec = None

    # Attach units and get positions, elmenets and masses.
    positions = (np.array(atoms)[:, 1:] * alat).to("ang")
    indices = np.array(atoms)[:, 0].astype(int) - 1
    elements = [species[x][1] for x in indices]
    masses = np.array([species[x][2] for x in indices]) * ureg._2m_e
    displacements = np.array(displacements)
    q_point = (q_point * ureg._2pi / alat).to("_2pi/ang")
    freqs = freqs * ureg("c/cm")

    return SimpleNamespace(
        q=q_point,
        lattice=lattice,
        freqs=freqs,
        displacements=displacements,
        positions=positions,
        elements=elements,
        masses=masses,
    )


def _find_dyn_file(q_cryst: np.ndarray | ureg.Quantity, results_ph_path: str) -> str:
    """
    Search for the Quantum ESPRESSO `.dyn` file containing a specified q-point
    in crystalline coordinates.

    This function compares the requested q-point with those found in each
    `dyn*` file, accounting for equivalence under reciprocal lattice translations.

    Parameters
    ----------
    q_cryst : np.ndarray | ureg.Quantity
        The q-point to locate, expressed in crystalline (reduced) coordinates.
    results_ph_path : str
        Path to the folder where the phonon (`ph.x`) output `.dyn*` files are stored.

    Returns
    -------
    str
        The full path to the `.dyn` file that contains the matching q-point.

    Raises
    ------
    FileNotFoundError
        If no `.dyn*` file or no matching q-point is found in any of the `.dyn` files
    """
    # Locate a reference .dyn file to extract lattice
    dyn1 = glob.glob(results_ph_path + "/*dyn1") + glob.glob(results_ph_path + "/*dyn")
    if not dyn1:
        raise FileNotFoundError(
            "No 'dyn1' or 'dyn' file found in the specified folder."
        )

    # Read lattice and convert to alat units
    lattice = read_dyn_file(dyn1[0]).lattice
    lattice = lattice / np.linalg.norm(lattice[0]) * ureg.alat
    k_basis = ut.reciprocal_basis(lattice)

    # Scan all .dyn* files (excluding matdyn if present)
    dyn_files = glob.glob(results_ph_path + "*.dyn*")
    dyn_files = [f for f in dyn_files if "results_matdyn" not in f]

    for file in dyn_files:
        with open(file, "r") as f:
            for line in f:
                if "q = (" in line:
                    q_point = np.array([float(x) for x in line.split()[3:6]]) * ureg(
                        "_2pi/alat"
                    )
                    q_crys_from_file = ut.cartesian2cryst(q_point, k_basis)

                    # Account for periodic images using symmetry
                    for q_equiv in ut._expand_zone_border(q_crys_from_file):
                        if np.allclose(q_cryst, q_equiv, atol=1e-4):
                            return file

    raise FileNotFoundError(
        f"No `.dyn*` file found containing q = {q_cryst} in crystalline coordinates."
    )


def read_dyn_q(
    q_cryst: np.ndarray | ureg.Quantity, results_ph_path: str, qe_format: bool = True
) -> SimpleNamespace:
    """
    Reads the Quantum ESPRESSO `.dyn*` file corresponding to a given q-point.

    This function locates the `.dyn*` file generated by `ph.x` that corresponds to a
    desired q-point (in reduced crystalline coordinates), extracts the corresponding dynamical
    matrix, and optionally converts it to the real physical dynamical matrix in units
    of 1 / [time]².

    Parameters
    ----------
    q_cryst : np.ndarray | ureg.Quantity
        The q-point of interest, expressed in reduced crystalline coordinates (fractions of reciprocal lattice vectors).
        If not a `Quantity`, it is assumed to be in `_2pi/crystal` units.

    results_ph_path : str
        Path to the directory containing the Quantum ESPRESSO `ph.x` output `.dyn*` files.

    qe_format : bool, optional
        If True (default), returns the raw QE dynamical matrix (includes sqrt(m_i m_j) mass factors).
        If False, converts the dynamical matrix to true physical form in 1 / [time]² units (Ry/h)^2.

    Returns
    -------
    system : SimpleNamespace
        A container with the following fields:
        - q : ureg.Quantity, shape (3,)
            The q-point in where the calculation was performed.
        - lattice : ureg.Quantity, shape (3, 3)
            The lattice vectors of the unit cell.
        - freqs : ureg.Quantity, shape (n_modes,)
            Array of vibrational frequencies (in cm⁻¹).
        - positions : ureg.Quantity, shape (n_atoms, 3)
            Atomic positions.
        - elements : list of str
            Chemical symbol for each atom.
        - masses : ureg.Quantity, shape (n_atoms,)
            Atomic mass in atomic units for each atom.
        - dyn: ureg.Quantity
            The (3N × 3N) complex dynamical matrix (units depend on `qe_format`).

    Notes
    -----
    - The dynamical matrix read from QE includes a sqrt(m_i m_j) prefactor for each (3×3) subblock.
      This must be removed to obtain the physical matrix for diagonalization (ω² in Ry²/ħ²).
    - The units of the returned matrix are `_2m_e * Ry^2 / planck_constant^2` in QE format.
    """
    # Normalize units
    if isinstance(q_cryst, ureg.Quantity):
        q_cryst = q_cryst.to("_2pi/crystal")
    else:
        q_cryst = q_cryst * ureg("_2pi/crystal")

    file = _find_dyn_file(q_cryst, results_ph_path)

    system = read_dyn_file(file)
    dim = 3 * len(system.elements)
    dyn_mat = np.zeros((dim, dim), dtype=complex)

    # Lattice and reciprocal basis
    lattice = system.lattice / np.linalg.norm(system.lattice[0]) * ureg.alat
    k_basis = ut.reciprocal_basis(lattice)

    READ_dynmat = False
    with open(file, "r") as lines:
        for line in lines:
            if "q = (" in line:
                q_point = np.array([float(x) for x in line.split()[3:6]]) * ureg(
                    "_2pi/alat"
                )
                q_crys_from_file = ut.cartesian2cryst(q_point, k_basis)

                for q_equiv in ut._expand_zone_border(q_crys_from_file):
                    if np.allclose(q_cryst.magnitude, q_equiv.magnitude, atol=1e-4):
                        READ_dynmat = True
                        break

            if READ_dynmat:
                l = line.split()
                if len(l) == 2:  # matrix block index line
                    n, m = int(l[0]), int(l[1])
                    num = 0
                elif len(l) == 6:  # submatrix row
                    row = np.array(
                        [
                            complex(float(l[0]), float(l[1])),
                            complex(float(l[2]), float(l[3])),
                            complex(float(l[4]), float(l[5])),
                        ]
                    )
                    sub_mat = row if num == 0 else np.vstack([sub_mat, row])
                    num += 1
                    if num == 3:
                        i, j = 3 * (n - 1), 3 * (m - 1)
                        dyn_mat[i : i + 3, j : j + 3] = sub_mat
                if re.search("Dynamical", line) or re.search("Diagonalizing", line):
                    break

    # Clean output
    system.q = q_cryst
    delattr(system, "displacements")
    system.dyn = dyn_mat * ureg("_2m_e * Ry^2 / planck_constant^2")
    if not qe_format:
        system.dyn = ph._QEdyn2Realdyn(system.dyn, system.masses)

    return system
