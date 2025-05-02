# PYTHON module with the electron classes for electronic spectrum

import warnings
import numpy as np
import ase.io.formats
import matplotlib.axes._axes
import matplotlib.pyplot as plt
from .units import ureg
from yaiv import grep as grep
from yaiv import utils as ut


class data_with_units:
    def __init__(self, data, units="arb."):
        self.data = data
        self.units = units

    def __array__(self, dtype=None):
        return self.data.astype(dtype) if dtype else self.data

    def __repr__(self):
        return f"<EigenArray shape={self.data.shape} units={self.units}>"


class _has_lattice:
    """
    Mixin class that provides lattice-related functionality:
    loading a lattice, computing its reciprocal basis, and transforming k-points.

    Attributes
    ----------
    lattice : np.ndarray
        3x3 matrix of direct lattice vectors in Å.
    k_lattice : np.ndarray
        3x3 matrix of reciprocal lattice vectors in 2πÅ⁻¹.
    """

    def update_lattice(self, lattice: np.ndarray = None):
        """
        Refreshes the lattice attribute and updates reciprocal lattice and kpath accordingly.

        Parameters
        ----------
        lattice : np.ndarray, optional
            3x3 matrix with the lattice in Å. If not provided, the lattice is loaded
            from the original file if possible.
        """
        try:
            if lattice is None:
                self.lattice = grep.lattice(self.filepath)
            else:
                self.lattice = lattice

            self.k_lattice = ut.K_basis(self.lattice)

            if hasattr(self, "spectrum") and hasattr(self.spectrum, "k_cryst"):
                self.spectrum.k_cart = ut.cryst2cartesian(
                    self.spectrum.k_cryst, self.k_lattice, list_of_vec=True
                )
            elif hasattr(self, "spectrum") and hasattr(self.spectrum, "k_alat"):
                alat_k_lattice = ut.K_basis(self.lattice, alat=True)
                self.spectrum.k_cryst = ut.cartesian2cryst(
                    self.spectrum.k_alat, alat_k_lattice, list_of_vec=True
                )

        except ase.io.formats.UnknownFileTypeError:
            warnings.warn(
                f"Could not read lattice from {self.filepath}. Lattice set to None."
            )
            self.lattice = None
            self.k_lattice = None


class spectrum:
    """
    General class for storing the eigenvalues of a periodic operator over k-points.

    This can represent band structures, phonon spectra, or eigenvalues of other operators.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Array of shape (nkpts, neigs), e.g., energy or frequency values.
    kpoints : np.ndarray
        Array of shape (nkpts, 3), in crystal or Cartesian coordinates.
    weights : np.ndarray, optional
        Optional weights for each k-point.
    """

    def __init__(self, eigenvalues, kpoints, weights=None):
        self.eigenvalues = eigenvalues
        self.kpoints = kpoints
        self.weights = weights if weights is not None else None

    def plot(
        self, ax: matplotlib.axes._axes.Axes = None, shift: float = 0.0, **kwargs
    ) -> matplotlib.axes._axes.Axes:
        """
        Plot the spectrum over a cumulative k-path.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift : float, optional
            A constant shift applied to the eigenvalues (e.g., Fermi level).
        **kwargs : dict
            Additional matplotlib arguments passed to `plot()`.

        Returns
        -------
        ax : matplotlib.axes._axes.Axes
            The axes with the spectrum plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Compute 1D k-path
        delta_k = np.diff(self.kpoints, axis=0)
        segment_lengths = np.linalg.norm(delta_k, axis=1)
        kpath = np.concatenate([[0]*self.kpoints.units, np.cumsum(segment_lengths)])

        # Apply shift to eigenvalues
        bands = self.eigenvalues - shift

        for band in bands.T:
            ax.plot(kpath, band, **kwargs)

        ax.set_xlabel(f"k-path ({self.kpoints.units})")
        ax.set_ylabel(f"Eigenvalues ({self.eigenvalues.units})")

        return ax


class electronBands(_has_lattice):
    """
    Class for handling electronic bandstructures and spectrums.

    Attributes
    ----------
    filepath : str
        Path to the file containing electronic structure output.
    filetype : str
        Type of the file (e.g., 'qe_scf_out', 'eigenval', etc.).
    electron_num : int
        Total number of electrons in the system.
    spectrum : SimpleNamespace
        Object containing `energies` (eV), `kpoints` (crystal units), and `weights`.
    lattice : np.ndarray
        3x3 matrix of lattice vectors in Å.
    k_lattice : np.ndarray
        3x3 matrix of reciprocal lattice vectors in 2πÅ⁻¹.
    fermi : float
        Fermi energy in eV (0 if not found).
    """

    def __init__(self, file):
        self.filepath = file
        self.filetype = grep._filetype(self.filepath)
        self.electron_num = grep.electron_num(self.filepath)
        self.spectrum = grep.kpointsEnergies(self.filepath)
        self.update_lattice()
        try:
            self.fermi = grep.fermi(self.filepath)
        except (NameError, NotImplementedError):
            self.fermi = 0


class phononBands(_has_lattice):
    """
    Class for handling phonon bandstructures and spectrums.

    Attributes
    ----------
    filepath : str
        Path to the file containing phonon frequencies output.
    filetype : str
        Type of the file (e.g., 'qe_freq_out', 'eigenval', etc.).
    spectrum : SimpleNamespace
        Object containing `frequencies` (cm-1) and `kpoints` (alat units).
    lattice : np.ndarray
        3x3 matrix of lattice vectors in Å.
    k_lattice : np.ndarray
        3x3 matrix of reciprocal lattice vectors in 2πÅ⁻¹.
    """

    def __init__(self, file):
        self.filepath = file
        self.filetype = grep._filetype(self.filepath)
        self.spectrum = grep.kpointsFrequencies(self.filepath)
        self.update_lattice()
