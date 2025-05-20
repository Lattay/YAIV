import re
from types import SimpleNamespace

import numpy as np

from yaiv.defaults.config import ureg
from yaiv import grep


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
            - displacement_vec : np.ndarray, shape(n_modes,n_atoms,3)
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
    filetype = _filetype(file)
    if filetype != "qe_dyn":
        raise NotImplementedError("Unsupported filetype")
    lattice = grep.lattice(file)
    n_atoms = n_types = freqs = vec = alat = None
    species = []
    atoms = []
    displacement_vec = []
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
                        displacement_vec.append(vec)
                        vec = None

    # Attach units and get positions, elmenets and masses.
    positions = (np.array(atoms)[:, 1:] * alat).to("ang")
    indices = np.array(atoms)[:, 0].astype(int) - 1
    elements = [species[x][1] for x in indices]
    masses = np.array([species[x][2] for x in indices]) * 2 * ureg.electron_mass
    masses = masses.to("amu")
    displacement_vec = np.array(displacement_vec)
    q_point = (q_point * ureg._2pi / alat).to("_2pi/ang")
    freqs = freqs * ureg("c/cm")

    return SimpleNamespace(
        q=q_point,
        lattice=lattice,
        freqs=freqs,
        displacement_vec=displacement_vec,
        positions=positions,
        elements=elements,
        masses=masses,
    )


def distort_crystal(q_points, results_ph, order_parameter, modes, amplitude):
    # TODO
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
