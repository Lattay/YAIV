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
>>> from yaiv.spectrum import ElectronBands
>>> bands = ElectronBands("data/qe/Si.bands.pwo")
>>> bands.eigenvalues.shape
(100, 32)
>>> bands.plot()
(Figure)

See Also
--------
yaiv.grep     : Low-level data extractors used to populate spectrum objects
yaiv.utils    : Basis universal utilities and vector transformations
yaiv.defaults : Configuration and default plotting values
"""

import warnings
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from yaiv.defaults.config import ureg
from yaiv.defaults.config import plot_defaults as pdft
from yaiv import utils as ut
from yaiv import grep as grep


__all__ = ["Spectrum" "ElectronBands", "PhononBands"]


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


class Spectrum(_has_lattice, _has_kpath):
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
    k_lattice : np.ndarray, optional
        3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
    kpath : SimpleNamespace | np.ndarray, optional
        A namespace with attributes `path`(ndarray) and `labels`(list)
        or just a ndarray.
    DOS : SimpleNamespace, optional
        - vgrid : np.ndarray | ureg.Quantity
            Array of shape (steps,) with the eigenvalue units.
        - DOS : np.ndarray
            Array of shape (steps,) with the corresponding DOS values.
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
        self.DOS = None

    def get_DOS(
        self,
        center: float | ureg.Quantity = None,
        window: float | list[float] | ureg.Quantity = None,
        smearing: float | ureg.Quantity = None,
        steps: int = None,
        precision: float = 3.0,
    ):
        """
        Compute a density of states (DOS) using Gaussian smearing.

        This implementation uses a normal distribution to smear each eigenvalue
        and returns the total DOS over an eigenvalue grid.

        Parameters
        ----------
        center : float | ureg.Quantity, optional
            Center for the energy window (e.g., Fermi energy). Default is zero.
        window : float | list[float] | ureg.Quantity, optional
            Value window for the DOS. If float, interpreted as symmetric [-window, window].
            If list, used as [Vmin, Vmax]. If None, the eigenvalue range is used.
        smearing : float | ureg.Quantity, optional
            Gaussian smearing width in the same units as eigenvalues. Default is (window_size/200).
        steps : int, optional
            Number of grid points for DOS sampling. Default is 4 * (window_size/smearing).
        precision : float, optional
            Number of smearing widths to use for truncation (e.g., 3 means ±3σ).

        Returns
        -------
        self.DOS : SimpleNamespace
            - vgrid : np.ndarray | ureg.Quantity
                Array of shape (steps,) with the eigenvalue units.
            - DOS : np.ndarray | ureg.Quantity
                Array of shape (steps,) with the computed DOS values.

        Raises
        ------
        ValueError
            If eigenvalues shape is incorrect or weights do not match.
        """
        # Handle units
        eigenvalues = self.eigenvalues
        quantities = [eigenvalues, center, window, smearing]
        names = ["eigenvalues", "center", "window", "smearing"]
        ut._check_unit_consistency(quantities, names)
        # If unitful, convert all to common unit
        if isinstance(eigenvalues, ureg.Quantity):
            units = eigenvalues.units
            eigenvalues, center, window, smearing = [
                x.to(units).magnitude if isinstance(x, ureg.Quantity) else x
                for x in quantities
            ]
        else:
            units = 1

        if eigenvalues.ndim != 2:
            raise ValueError(
                "Eigenvalues must be a 2D array of shape (n_kpts, n_bands)"
            )
        n_kpts, n_bands = eigenvalues.shape
        if self.weights is None:
            self.weights = weights = (
                np.ones(n_kpts) / n_kpts
            )  # Weights that sum one (one state per band).
        else:
            weights = np.asarray(self.weights)
        if weights.shape[0] != n_kpts:
            raise ValueError("Weights must match the number of k-points")

        # Determine computing center, window, smearing and steps
        center = 0 if center is None else center
        if window is None:
            V_min, V_max = eigenvalues.min(), eigenvalues.max()
        elif isinstance(window, (float, int)):
            V_min, V_max = np.array([-window, window]) + center
        else:
            V_min, V_max = np.asarray(window) + center
        window_size = V_max - V_min
        if smearing is None:
            smearing = window_size / 200
        if steps is None:
            steps = int(4 * (window_size / smearing))
        if window is None:
            V_min = V_min - smearing * precision
            V_max = V_max + smearing * precision
        V_grid = np.linspace(V_min, V_max, steps)

        # Flatten eigenvalues and weights
        flattened_eigs = eigenvalues.flatten()
        flattened_weights = np.repeat(weights, n_bands)
        # Order energies and weights
        sort = np.argsort(flattened_eigs)
        flattened_eigs = flattened_eigs[sort]
        flattened_weights = flattened_weights[sort]

        DOS = np.zeros_like(V_grid)

        # DOS calculation (using the fact that eigenvalues are sorted)
        for i, V in enumerate(V_grid):
            for j, e in enumerate(flattened_eigs):
                if e >= (V - precision * smearing):
                    if DOS[i] == 0:
                        truncated_eigs = flattened_eigs[j:]
                        truncated_weights = flattened_weights[j:]
                    DOS[i] = (
                        DOS[i] + ut._normal_dist(e, V, smearing) * flattened_weights[j]
                    )
                if e >= (V + precision * smearing):
                    flattened_eigs = truncated_eigs
                    flattened_weights = truncated_weights
                    break
        self.DOS = SimpleNamespace(vgrid=V_grid * units, DOS=DOS * 1 / units)

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
        if isinstance(self.kpoints, ureg.Quantity):
            kpoints = self.kpoints
            if "crystal" in kpoints.units._units and self.k_lattice is not None:
                kpoints = ut.cryst2cartesian(self.kpoints, self.k_lattice)
            units = kpoints.units
            kpts_val = kpoints.magnitude
        else:
            units = 1
            kpts_val = self.kpoints

        # Compute segment lengths
        delta_k = np.diff(kpts_val, axis=0)
        segment_lengths = np.linalg.norm(delta_k, axis=1)
        if patched:
            # Define discontinuities as large jumps relative to minimum segment
            threshold = np.min(segment_lengths[segment_lengths >= 1e-5]) * 10
            segment_lengths = np.where(segment_lengths > threshold, 0, segment_lengths)
        kpath = np.concatenate([[0], np.cumsum(segment_lengths)])
        return kpath * units

    def plot(
        self,
        ax: Axes = None,
        shift: float | ureg.Quantity = None,
        patched: bool = True,
        bands: list[int] = None,
        **kwargs,
    ) -> Axes:
        """
        Plot the spectrum over a cumulative k-path.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift : float | ureg.Quantity, optional
            A constant shift applied to the eigenvalues (e.g., Fermi level).
            Default is zero.
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
        ax : Axes
            The axes with the spectrum plot.
        """
        # Handle units
        quantities = [self.eigenvalues, shift]
        names = ["eigenvalues", "shift"]
        ut._check_unit_consistency(quantities, names)

        if ax is None:
            fig, ax = plt.subplots()

        # Apply shift to eigenvalues
        eigen = self.eigenvalues - shift if shift is not None else self.eigenvalues
        kpath = self.get_1Dkpath(patched)
        x = kpath / kpath[-1].magnitude

        band_indices = bands if bands is not None else range(eigen.shape[1])

        label = kwargs.pop("label", None)  # remove label from kwargs
        ax.plot(x, eigen[:, band_indices[0]], label=label, **kwargs)
        ax.plot(x, eigen[:, band_indices[1:]], **kwargs)

        ax.set_xlim(0, 1)
        return ax

    def plot_DOS(
        self,
        ax: Axes = None,
        shift: float | ureg.Quantity = None,
        switchXY: bool = False,
        fill: bool = True,
        alpha: float = pdft.alpha,
        **kwargs,
    ) -> Axes:
        """
        Plot the DOS over an eigenvalue-window.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift : float | ureg.Quantity, optional
            A constant shift applied to the DOS (e.g., Fermi level).
            Default is zero.
        switchXY : bool, optional
            Whether to plot the DOS along the x-axis (horizontal plot). Default is False.
        fill : bool, optional
            Whether to fill the area under the curve. Default is True.
        alpha : float, optional
            Opacity of the fill (0 = transparent, 1 = solid).
        **kwargs : dict
            Additional matplotlib arguments passed to `plot()`.

        Returns
        ----------
        ax : Axes
            The axes with the spectrum plot.
        """
        # Handle units
        if self.DOS is None:
            quantities = [self.eigenvalues, shift]
            names = ["self.eigenvalues", "shift"]
        else:
            quantities = [self.DOS.vgrid, shift]
            names = ["self.DOS.vgrid", "shift"]
        ut._check_unit_consistency(quantities, names)

        if ax is None:
            fig, ax = plt.subplots()

        if self.DOS is None:
            self.get_DOS()
        x = self.DOS.vgrid if shift is None else self.DOS.vgrid - shift
        y = self.DOS.DOS

        z_line = kwargs.pop("zorder", 2)  # allow overriding via kwargs
        z_fill = z_line - 1  # ensure fill is below the line

        if switchXY:
            # DOS on x-axis, energy on y-axis
            (line,) = ax.plot(y, x, zorder=z_line, **kwargs)
            if fill:
                ax.fill_betweenx(
                    x, 0, y, alpha=alpha, color=line.get_color(), zorder=z_fill
                )
            ax.set_xlabel(f"DOS({y.units})")
            ax.set_xlim(left=0)
            ax.set_ylim(np.min(x), np.max(x))
        else:
            # Energy on x-axis, DOS on y-axis
            (line,) = ax.plot(x, y, zorder=z_line, **kwargs)
            if fill:
                ax.fill_between(
                    x, y, alpha=alpha, color=line.get_color(), zorder=z_fill
                )
            ax.set_ylabel(f"DOS({y.units})")
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(bottom=0)

        return ax


class ElectronBands(Spectrum):
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
            Spectrum.__init__(
                self,
                eigenvalues=spec.eigenvalues,
                kpoints=spec.kpoints,
                weights=spec.weights,
                lattice=lattice,
            )
        else:
            self.electron_num = self.fermi = None
            Spectrum.__init__(self)


class PhononBands(Spectrum):
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
            Spectrum.__init__(
                self,
                eigenvalues=spec.eigenvalues,
                kpoints=spec.kpoints,
                weights=spec.weights,
                lattice=lattice,
            )
        else:
            Spectrum.__init__(self)
