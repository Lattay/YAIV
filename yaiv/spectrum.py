"""
YAIV | yaiv.spectrum
====================

This module defines core classes for representing and plotting the eigenvalue spectrum
of periodic operators, such as electronic bands or phonon frequencies, across a set of
k-points. It also supports reciprocal lattice handling and coordinate transformations.

The classes in this module can be used independently or as output containers from
grepping functions.

Examples
--------
>>> from yaiv.spectrum import electronBands
>>> bands = electronBands("data/qe/Si.bands.pwo")
>>> bands.eigenvalues.shape
(100, 32)
>>> bands.plot()
(Figure)

See Also
--------
yaiv.grep     : File parsing functions that uses these utilities.
yaiv.utils    : Basis universal utilities
"""

# PYTHON module with the electron classes for electronic spectrum

import warnings
from types import SimpleNamespace

import numpy as np
import matplotlib.axes._axes
import matplotlib.pyplot as plt

from yaiv.defaults.config import ureg
import yaiv.utils as ut
from yaiv import grep as grep


class _has_lattice:
    """
    Mixin that provides lattice-related functionality:
    loading a lattice, computing its reciprocal basis, and transforming k-points.

    Parameters
    ----------
    lattice : np.ndarray, optional
        3x3 matrix of direct lattice vectors in [length] units.

    Attributes
    ----------
    lattice : np.ndarray
        3x3 matrix of direct lattice vectors in [length] units.
    k_lattice : np.ndarray
        3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
    """

    def __init__(self, lattice: np.ndarray = None):
        self._lattice = self._k_lattice = None
        if lattice is not None:
            self._lattice = lattice
            self._k_lattice = ut.reciprocal_basis(self._lattice)

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, value):
        self._lattice = value
        self._k_lattice = ut.reciprocal_basis(value)

    @property
    def k_lattice(self):
        return self._k_lattice

    @k_lattice.setter
    def k_lattice(self, value):
        self._k_lattice = value
        self._lattice = ut.reciprocal_basis(value)


class _has_kpath:
    """
    Mixin that provides lattice-related functionality:

    Attributes
    ----------
    kpath : SimpleNamespace | np.ndarray
        A namespace with attributes `path`(ndarray) and `labels`(list)
        or just a ndarray.
    """

    def __init__(self, kpath: SimpleNamespace | np.ndarray = None):
        self.kpath = kpath


class spectrum(_has_lattice, _has_kpath):
    """
    General class for storing the eigenvalues of a periodic operator over k-points.

    This can represent band structures, phonon spectra, or eigenvalues of other operators.


    Attributes
    ----------
    eigenvalues : np.ndarray, optional
        Array of shape (nkpts, neigs), e.g., energy or frequency values.
    kpoints : np.ndarray, optional
        Array of shape (nkpts, 3) with k-points.
    weights : np.ndarray, optional
        Optional weights for each k-point.
    lattice : np.ndarray, optional
        3x3 matrix of direct lattice vectors in [length] units.
    k_lattice : np.ndarray
        3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
    kpath : SimpleNamespace | np.ndarray
        A namespace with attributes `path`(ndarray) and `labels`(list)
        or just a ndarray.
    """

    def __init__(
        self,
        eigenvalues: np.ndarray = None,
        kpoints: np.ndarray = None,
        weights: list | np.ndarray = None,
        lattice: np.ndarray = None,
        kpath: SimpleNamespace | np.ndarray = None,
    ):
        self.eigenvalues = eigenvalues
        self.kpoints = kpoints
        self.weights = weights
        _has_lattice.__init__(self, lattice)
        _has_kpath.__init__(self, kpath)

    def get_1Dkpath(self, patched=True) -> np.ndarray:
        """
        Computes the 1D cumulative k-path from the k-point coordinates.

        Parameters
        ----------
        patched : bool, optional
            If True, attempts to patch discontinuities in the k-path.
            This prevents artificially connected lines in band structure
            plots (e.g., flat bands across high-symmetry points).

        Returns
        ----------
        kpath : np.ndarray
            The 1D cumulative k-path from the k-point coordinates.
        """
        if self.kpoints is None:
            raise ValueError("kpoints are not defined.")

        # Strip units for math, retain them for reapplication later
        if hasattr(self.kpoints, "units"):
            kpoints = self.kpoints
            if "crystal" in kpoints.units._units and self.k_lattice is not None:
                kpoints = ut.cryst2cartesian(self.kpoints, self.k_lattice)
            units = kpoints.units
            kpts_val = kpoints.magnitude
        else:
            units = None
            kpts_val = self.kpoints

        # Compute segment lengths
        delta_k = np.diff(kpts_val, axis=0)
        segment_lengths = np.linalg.norm(delta_k, axis=1)
        if patched:
            # Define discontinuities as large jumps relative to minimum segment
            threshold = np.min(segment_lengths[segment_lengths >= 1e-5]) * 10
            segment_lengths = np.where(segment_lengths > threshold, 0, segment_lengths)
        kpath = np.concatenate([[0], np.cumsum(segment_lengths)])
        if units is not None:
            kpath = kpath * units  # reattach units
        return kpath

    def plot(
        self,
        ax: matplotlib.axes._axes.Axes = None,
        shift: float = None,
        patched: bool = True,
        bands: list[int] = None,
        **kwargs,
    ) -> matplotlib.axes._axes.Axes:
        """
        Plot the spectrum over a cumulative k-path.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift : float, optional
            A constant shift applied to the eigenvalues (e.g., Fermi level).
            Fermi level shift is the default for electronic spectra.
        patched : bool, optional
            If True, attempts to patch discontinuities in the k-path.
            This prevents artificially connected lines in band structure
            plots (e.g., flat bands across high-symmetry points).
        bands : list of int, optional
            Indices of the bands to plot. If None, all bands are plotted.
        **kwargs : dict
            Additional matplotlib arguments passed to `plot()`.

        Returns
        ----------
        ax : matplotlib.axes._axes.Axes
            The axes with the spectrum plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Apply shift to eigenvalues
        if shift is None:
            if hasattr(self, "fermi"):
                shift = self.fermi if self.fermi is not None else 0
            else:
                shift = 0

        eigen = self.eigenvalues - shift
        kpath = self.get_1Dkpath(patched)
        x = kpath / kpath[-1]
        if isinstance(eigen, ureg.Quantity):
            eigen = eigen.magnitude
        if isinstance(x, ureg.Quantity):
            x = x.magnitude

        band_indices = bands if bands is not None else range(eigen.shape[1])

        label = kwargs.pop("label", None)  # remove label from kwargs
        for j, i in enumerate(band_indices):
            if j == 0:
                ax.plot(x, eigen[:, i], label=label, **kwargs)
            else:
                ax.plot(x, eigen[:, i], **kwargs)

        ax.set_xlim(0, 1)
        ax.set_xlabel(f"k-path ({kpath.units})")
        ax.set_ylabel(f"Eigenvalues ({self.eigenvalues.units})")

        return ax


class electronBands(spectrum):
    """
    Class for handling electronic bandstructures and spectrums.

    Parameters
    ----------
    file : str
        File from which to extract the bands.

    Attributes
    ----------
    filepath : str
        Path to the file containing electronic structure output.
    electron_num : int
        Total number of electrons in the system.
    eigenvalues : np.ndarray
        Array of shape (nkpts, neigs) with energy values.
    kpoints : np.ndarray
        Array of shape (nkpts, 3) with k-points.
    weights : np.ndarray
        Optional weights for each k-point.
    lattice : np.ndarray
        3x3 matrix of lattice vectors in [length] units.
    k_lattice : np.ndarray
        3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
    fermi : float
        Fermi energy (0 if not found).
    """

    def __init__(self, file: str = None):
        if file is not None:
            self.filepath = file
            self.electron_num = grep.electron_num(self.filepath)
            try:
                self.fermi = grep.fermi(self.filepath)
            except (NameError, NotImplementedError):
                self.fermi = None
            try:
                lattice = grep.lattice(self.filepath)
            except NotImplementedError:
                lattice = None
            spec = grep.kpointsEnergies(self.filepath)
            spectrum.__init__(
                self,
                eigenvalues=spec.eigenvalues,
                kpoints=spec.kpoints,
                weights=spec.weights,
                lattice=lattice,
            )
        else:
            self.electron_num = self.fermi = None
            spectrum.__init__(self)


class phononBands(spectrum):
    """
    Class for handling phonon bandstructures and spectrums.

    Parameters
    ----------
    file : str
        File from which to extract the spectrum.

    Attributes
    ----------
    filepath : str
        Path to the file containing phonon frequencies output.
    eigenvalues : np.ndarray
        Array of shape (nkpts, neigs) with frequency values.
    kpoints : np.ndarray
        Array of shape (nkpts, 3) with k-points.
    weights : np.ndarray
        Optional weights for each k-point.
    lattice : np.ndarray
        3x3 matrix of lattice vectors in [length] units.
    k_lattice : np.ndarray
        3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
    """

    def __init__(self, file: str = None):
        if file is not None:
            self.filepath = file
            try:
                lattice = grep.lattice(self.filepath)
            except NotImplementedError:
                lattice = None
            spec = grep.kpointsFrequencies(self.filepath)
            spectrum.__init__(
                self,
                eigenvalues=spec.eigenvalues,
                kpoints=spec.kpoints,
                weights=spec.weights,
                lattice=lattice,
            )
        else:
            spectrum.__init__(self)
