# PYTHON module with the electron classes for electronic spectrum

import warnings
import numpy as np
import ase.io.formats
import matplotlib.axes._axes
import matplotlib.pyplot as plt
from yaiv import grep as grep
from yaiv import utils as ut


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


class _has_spectrum:
    """
    Mixin class that provides spectrum-related functionality:

    Attributes
    ----------
    spectrum : SimpleNamespace
        Object containing `energies/frequencies` (eV/cm-1), `kpoints`, and `weights`.
        - energies/freqs : np.ndarray
            List of energies/freqs, each row corresponds to a particular k-point.
        - kpoints : np.ndarray
            List of k-points.
        - weights : np.ndarray
            List of kpoint-weights.
    """

    def plot(
        self, ax: matplotlib.axes._axes.Axes = None, shift_fermi: bool = True, **kwargs
    ) -> matplotlib.axes._axes.Axes:
        """
        Plot the electronic band structure.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift_fermi : bool, optional
            Whether to shift the energies so that the Fermi level is at 0. Default is True.
        **kwargs : dict
            Additional matplotlib arguments passed to `plot()`.

        Returns
        -------
        ax : matplotlib.axes._axes.Axes
            The axes with the band structure plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # electronic bands
        if hasattr(self.spectrum, "energies"):
            bands = self.spectrum.energies  # shape: (nkpts, nbands)
            if shift_fermi:
                bands = bands - self.fermi
        # phonon bands
        elif hasattr(self.spectrum, "freqs"):
            bands = self.spectrum.freqs  # shape: (nkpts, nbands)

        # Get x-coord for the plot
        if hasattr(self.spectrum, "k_cart"):
            delta_k = np.diff(self.spectrum.k_cart, axis=0)  # shape (N-1, 3)
        elif hasattr(self.spectrum, "k_alat"):
            delta_k = np.diff(self.spectrum.k_alat, axis=0)  # shape (N-1, 3)
        else:
            warnings.warn("K-path generated from crystal units!", UserWarning)
            delta_k = np.diff(self.spectrum.k_cryst, axis=0)  # shape (N-1, 3)
        segment_lengths = np.linalg.norm(delta_k, axis=1)  # shape (N-1,)
        x = np.concatenate([[0], np.cumsum(segment_lengths)])  # len (N)
        ax.plot(x, bands, **kwargs)
        return ax


class electronBands(_has_lattice, _has_spectrum):
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


class phononBands(_has_lattice, _has_spectrum):
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
