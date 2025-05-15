import numpy as np

from ase.io import read, write
from ase import Atoms

__all__ = ["read", "write", "read_spg", "ase2spglib", "spglib2ase", "Cell"]


def ase2spglib(crystal_ase: Atoms) -> tuple:
    """
    Convert an ASE Atoms object into the tuple format required by spglib.

    This function extracts the lattice, scaled atomic positions, and atomic numbers
    from an ASE Atoms object and returns them in the standard spglib tuple format:
    (lattice, positions, numbers).

    Parameters
    ----------
    crystal_ase : Atoms
        ASE Atoms object representing the crystal structure.

    Returns
    -------
    spglib_crystal : tuple
        A 3-tuple (lattice, positions, numbers) where:
        - lattice : (3, 3) array of lattice vectors
        - positions : (N, 3) array of scaled atomic positions (fractional)
        - numbers : (N,) array of atomic numbers
    """
    lattice = np.array(crystal_ase.get_cell())
    positions = crystal_ase.get_scaled_positions()
    numbers = crystal_ase.get_atomic_numbers()
    spglib_crystal = (lattice, positions, numbers)
    return spglib_crystal


def spglib2ase(spglib_crystal: tuple) -> Atoms:
    """
    Convert a spglib-format crystal tuple into an ASE Atoms object.

    This function takes a tuple of the form (lattice, positions, numbers),
    as used by spglib, and creates a corresponding ASE Atoms object.

    Parameters
    ----------
    spglib_crystal : tuple
        A 3-tuple (lattice, positions, numbers) as returned by spglib,
        where:
        - lattice : (3, 3) array of lattice vectors
        - positions : (N, 3) array of scaled (fractional) positions
        - numbers : (N,) array of atomic numbers

    Returns
    -------
    ase_crystal : Atoms
        ASE Atoms object representing the crystal structure.
    """
    lattice = spglib_crystal[0]
    positions = spglib_crystal[1]
    numbers = spglib_crystal[2]
    ase_crystal = Atoms(scaled_positions=positions, numbers=numbers, cell=lattice)
    return ase_crystal


def read_spg(file: str) -> tuple:
    """
    Read a crystal structure file and convert it directly to spglib format.

    This function uses ASE's `read()` to load a crystal structure file
    and returns the structure as a tuple in the format required by spglib:
    (lattice, positions, atomic_numbers).

    Parameters
    ----------
    file : str
        Path to the structure file (e.g., CIF, POSCAR, XYZ, etc.) supported by ASE.

    Returns
    -------
    spglib_crystal : tuple
        A 3-tuple (lattice, positions, numbers) where:
        - lattice : (3, 3) array of lattice vectors
        - positions : (N, 3) array of scaled atomic positions (fractional)
        - numbers : (N,) array of atomic numbers
    """
    cryst = read(file)
    spglib_cryst = ase2spglib(cryst)
    return spglib_cryst


class Cell:
    """
    A wrapper that stores both an ASE Atoms object and the corresponding
    spglib-format tuple (lattice, positions, numbers).

    This class allows use in spglib (via tuple interface) and in ASE (via .atoms),
    while keeping both views synchronized.

    Attributes
    ----------
    atoms : ase.Atoms
        Full ASE Atoms object with chemical info.
    spglib : tuple
        Tuple (lattice, positions, numbers) derived from the Atoms object,
        used for spglib symmetry operations.
    """

    def __init__(
        self,
        lattice: np.ndarray = None,
        positions: np.ndarray = None,
        numbers: np.ndarray = None,
        atoms: Atoms = None,
    ):
        """
        Initialize a Cell object.

        Parameters
        ----------
        lattice : array-like, optional
            3x3 lattice matrix.
        positions : array-like, optional
            Nx3 array of fractional coordinates.
        numbers : array-like, optional
            Length-N array of atomic numbers.
        atoms : Atoms, optional
            Alternative way to initialize with an Atoms object.

        Raises
        ------
        ValueError
            If neither individual arguments nor `ase.Atoms` object is provided correctly.
        """
        if atoms is not None:
            if not isinstance(atoms, Atoms):
                raise ValueError("`atoms` must be an ase.Atoms object.")
            self.atoms = atoms
            self.spglib = ase2spglib(atoms)
        elif lattice is None or positions is None or numbers is None:
            raise ValueError(
                "Must provide either individual components or an `ase.Atoms` object."
            )
        else:
            self.spglib = (np.array(lattice), np.array(positions), np.array(numbers))
            self.atoms = spglib2ase(atoms)

    @classmethod
    def from_file(cls, file: str):
        """
        Read a structure file using ASE and return a Cell instance.

        Parameters
        ----------
        file : str
            Path to structure file (e.g. CIF, POSCAR).

        Returns
        -------
        Cell
            A new Cell instance with Atoms and spglib data.
        """
        atoms = read(file)
        return cls(atoms=atoms)

    @classmethod
    def from_spglib_tuple(cls, tup):
        """
        Initialize from a (lattice, positions, numbers) spglib tuple.

        Parameters
        ----------
        tup : tuple
            A 3-tuple (lattice, positions, numbers)

        Returns
        -------
        Cell
            A new Cell instance with Atoms and spglib data.
        """
        lattice, positions, numbers = map(np.array, tup)
        return cls(lattice, positions, numbers)

    def __iter__(self):
        return iter(self.spglib)

    def __getitem__(self, key):
        return self.spglib[key]

    def __len__(self):
        return len(self.spglib)

    def __repr__(self):
        return f"<Cell with {len(self.spglib[2])} atoms>\n\n{self.spglib}"
        return f"\n{self.spglib}"
