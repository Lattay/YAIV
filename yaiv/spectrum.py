"""
YAIV | yaiv.spectrum
====================

This module defines core classes for representing and plotting the eigenvalue spectrum
of periodic operators, such as electronic bands or phonon frequencies, across a set of
k-points. It also supports reciprocal lattice handling and coordinate transformations.

The classes in this module can be used independently or as output containers from
grepping functions.

Classes
-------
Spectrum
    General container for k-resolved eigenvalue spectra (e.g., bands, phonons).
    Supports plotting, DOS calculation, and band visualizations.
    Provides:
    - get_DOS(...): Computes the density of states via Gaussian or Methfessel–Paxton smearing.
    - plot(...): Plots the band structure along a cumulative k-path.
    - plot_fat(...): Fat-band style scatter plot for visualizing weights/projections over bands.
    - plot_color(...): Color-gradient line plot for weights/projections over bands.

ElectronBands
    Specialized `Spectrum` subclass for electronic band structures extracted from files.
    Adds Fermi level, number of electrons, and automatic parsing.

PhononBands
    Specialized `Spectrum` subclass for phonon spectra extracted from files.

DOS
    Container for density of states data. Supports integration and plotting.
    Provides:
    - integrate(): Computes the integral of the DOS or finds the eigenvalue corresponding to a filled state count.
    - plot(): Plots the DOS curve with optional fill and orientation options.

Private Utilities
-----------------
_Has_lattice
    Mixin that adds lattice handling capabilities.

_Has_kpath
    Mixin that adds support for k-path functionalities.
    Provides:
    - get_1Dkpath(self, patched=True): Provides a one dimensional Kpath

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
from matplotlib.collections import PathCollection, LineCollection
from scipy import interpolate, integrate

from yaiv.defaults.config import ureg
from yaiv.defaults.config import plot_defaults as pdft
from yaiv import utils as ut
from yaiv import grep as grep


__all__ = ["Spectrum" "ElectronBands", "PhononBands", "DOS"]


class _Has_lattice:
    """
    Mixin that provides lattice-related functionality:
    loading a lattice, computing its reciprocal basis, and keeping them syncd.

    Attributes
    ----------
    lattice : np.ndarray
        3x3 matrix of direct lattice vectors in [length] units.
    k_lattice : np.ndarray
        3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
    alat : ureg.Quantity
        `alat` factor for conversions, defined as the norm of the first
        vector of the lattice.
    """

    def __init__(
        self,
        lattice: np.ndarray | ureg.Quantity = None,
        k_lattice: np.ndarray | ureg.Quantity = None,
    ):
        """
        Initialize the _Has_lattice object from either the real or reciprocal space lattice.

        Parameters
        ----------
        lattice : np.ndarray | ureg.Quantity
            3x3 matrix of direct lattice vectors in [length] units.
        k_lattice : np.ndarray | ureg.Quantity
            3x3 matrix of reciprocal lattice vectors in [length]⁻¹ units.
        """
        self._lattice = self._k_lattice = None
        if lattice is not None:
            self._lattice = lattice
            self._k_lattice = ut.reciprocal_basis(self._lattice)
        elif k_lattice is not None:
            self._k_lattice = k_lattice
            self._lattice = ut.reciprocal_basis(self._k_lattice)

    @property
    def lattice(self):
        return self._lattice

    @property
    def k_lattice(self):
        return self._k_lattice

    @property
    def alat(self):
        return np.linalg.norm(self.lattice[0]) / ureg.alat

    @lattice.setter
    def lattice(self, value):
        self._lattice = value
        self._k_lattice = ut.reciprocal_basis(value)

    @k_lattice.setter
    def k_lattice(self, value):
        self._k_lattice = value
        self._lattice = ut.reciprocal_basis(value)


class _Has_kpath:
    """
    Mixin that provides kpath-related functionality:

    Attributes
    ----------
    kpath : SimpleNamespace | np.ndarray
        A namespace with attributes `path`(ndarray) and `labels`(list)
        or just a ndarray.

    Methods
    -------
    get_1Dkpath()
        Computes the 1D cumulative k-path from the k-point coordinates.
    """

    def __init__(self, kpath: SimpleNamespace | np.ndarray = None):
        """
        Initialize the _Has_kpath object from a SimpleNamespace as given by
        `yaiv.grep.kpath`.

        Parameters
        ----------
        kpath : SimpleNamespace | np.ndarray
            A namespace with attributes `path`(ndarray) and `labels`(list)
            or just a ndarray.
        """
        self.kpath = kpath

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
        -------
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
                kpoints = kpoints.to('_2pi/ang')
            if "alat" in kpoints.units._units and self.k_lattice is not None:
                kpoints = kpoints / self.alat
                kpoints = kpoints.to('_2pi/ang')
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


class Spectrum(_Has_lattice, _Has_kpath):
    """
    General class for storing the eigenvalues of a periodic operator over k-points.

    This can represent band structures, phonon spectra, or eigenvalues of other operators.
    It is a subclass of `_Has_lattice` and `_Has_kpath` mixing classes.

    Attributes
    ----------
    eigenvalues : np.ndarray | ureg.Quantity, optional
        Array of shape (nkpts, neigs), e.g., energy or frequency values.
    kpoints : np.ndarray | ureg.Quantity, optional
        Array of shape (nkpts, 3) with k-points.
    weights : np.ndarray, optional
        Optional weights for each k-point.
    DOS : DOS, optional
        - vgrid : np.ndarray | pint.Quantity
            Array of shape (steps,) with the eigenvalue units.
        - DOS : np.ndarray
            Array of shape (steps,) with the corresponding DOS values.

    Methods
    -------
    get_DOS(...)
        Compute a density of states (DOS) for the set of eigenvalues.
    def plot(...)
        Plot the spectrum over a cumulative k-path.
    def plot_fat(...)
        Fat-band style plotting for weights over a cumulative k-path.
    def plot_color(...)
        Color gradient line-style for weights over a cumulative k-path.
    """

    def __init__(
        self,
        eigenvalues: np.ndarray | ureg.Quantity = None,
        kpoints: np.ndarray | ureg.Quantity = None,
        weights: list | np.ndarray = None,
        lattice: np.ndarray | ureg.Quantity = None,
        k_lattice: np.ndarray | ureg.Quantity = None,
        kpath: SimpleNamespace | np.ndarray = None,
    ):
        """
        Initialize Spectrum object.

        Parameters
        ----------
        eigenvalues : np.ndarray | ureg.Quantity, optional
            Array of shape (nkpts, neigs), e.g., energy or frequency values.
        kpoints : np.ndarray | ureg.Quantity, optional
            Array of shape (nkpts, 3) with k-points.
        weights : np.ndarray, optional
            Optional weights for each k-point.
        lattice : np.ndarray | ureg.Quantity, optional
            3x3 matrix of direct lattice vectors in [length] units.
        k_lattice : np.ndarray | ureg.Quantity, optional
            3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
            Will be ignored when defining the spectrum if lattice is given.
        kpath : SimpleNamespace | np.ndarray, optional
            A namespace with attributes `path`(ndarray) and `labels`(list)
            or just a ndarray.
        """
        self.eigenvalues = eigenvalues
        self.kpoints = kpoints
        self.weights = weights
        _Has_lattice.__init__(self, lattice, k_lattice)
        _Has_kpath.__init__(self, kpath)
        self.DOS = DOS(parent=self)

    def get_DOS(
        self,
        center: float | ureg.Quantity = None,
        window: float | list[float] | ureg.Quantity = None,
        smearing: float | ureg.Quantity = None,
        steps: int = None,
        order: int = 0,
        precision: float = 3.0,
    ):
        """
        Compute a density of states (DOS) using Gaussian or Methfessel-Paxton (MP)
        smearing of any order.

        This implementation uses a MP delta function to smear each eigenvalue and
        returns the total DOS over an eigenvalue grid. Since the default order is zero,
        it defaults to using a Gaussian distribution.

        Parameters
        ----------
        center : float | pint.Quantity, optional
            Center for the energy window (e.g., Fermi energy). Default is zero.
        window : float | list[float] | pint.Quantity, optional
            Value window for the DOS. If float, interpreted as symmetric [-window, window].
            If list, used as [Vmin, Vmax]. If None, the eigenvalue range (± smearing * precision) is used.
        smearing : float | pint.Quantity, optional
            Smearing width in the same unit dimension as eigenvalues. Default is (window_size/200).
        steps : int, optional
            Number of grid points for DOS sampling. Default is 4 * (window_size/smearing).
        order : int, optional
            Order of the Methfessel-Paxton expansion. Default is 0, which recovers a Gaussian smearing.
        precision : float, optional
            Number of smearing widths to use for truncation (e.g., 3 means ±3σ).

        Returns
        -------
        self.DOS : DOS
            - vgrid : np.ndarray | pint.Quantity
                Array of shape (steps,) with the eigenvalue units.
            - DOS : np.ndarray | pint.Quantity
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
            V_min = V_min - smearing * (precision + 1)
            V_max = V_max + smearing * (precision + 1)
        V_grid = np.linspace(V_min, V_max, steps)

        # Flatten eigenvalues and weights
        flattened_eigs = eigenvalues.flatten()
        flattened_weights = np.repeat(weights, n_bands)
        # Order energies and weights
        sort = np.argsort(flattened_eigs)
        flattened_eigs = flattened_eigs[sort]
        flattened_weights = flattened_weights[sort]

        dos = np.zeros_like(V_grid)

        # DOS calculation (using the fact that eigenvalues are sorted)
        if order == 0:
            for i, V in enumerate(V_grid):
                start = np.searchsorted(
                    flattened_eigs, V - precision * smearing, side="left"
                )
                stop = np.searchsorted(
                    flattened_eigs, V + precision * smearing, side="right"
                )
                dos[i] = np.sum(
                    ut._normal_dist(flattened_eigs[start:stop], V, smearing)
                    * flattened_weights[start:stop]
                )
                # Truncate data
                flattened_eigs = flattened_eigs[start:]
                flattened_weights = flattened_weights[start:]
        else:
            for i, V in enumerate(V_grid):
                start = np.searchsorted(
                    flattened_eigs, V - precision * smearing, side="left"
                )
                stop = np.searchsorted(
                    flattened_eigs, V + precision * smearing, side="right"
                )
                dos[i] = np.sum(
                    ut.methpax_delta(flattened_eigs[start:stop], V, smearing, order)
                    * flattened_weights[start:stop]
                )
                # Truncate data
                flattened_eigs = flattened_eigs[start:]
                flattened_weights = flattened_weights[start:]
        self.DOS = DOS(vgrid=V_grid * units, DOS=dos * 1 / units, parent=self)

    def _pre_plot(
        self=None,
        ax=None,
        shift=None,
        bands=None,
        patched=True,
        weights=None,
        window=None,
    ):
        """
        Pre plotting tool to avoid code duplication.
        """
        # Handle units
        if shift is not None:
            quantities = [self.eigenvalues, shift]
            names = ["eigenvalues", "shift"]
            ut._check_unit_consistency(quantities, names)

        # Create fig if necessary
        if ax is None:
            fig, ax = plt.subplots()

        # Apply shift to eigenvalues
        eigen = self.eigenvalues - shift if shift is not None else self.eigenvalues
        kpath = self.get_1Dkpath(patched)
        lenght = kpath[-1].magnitude if isinstance(kpath, ureg.Quantity) else kpath[-1]
        x = kpath / lenght

        band_indices = bands if bands is not None else range(eigen.shape[1])

        # Handle weights if present
        if weights is not None:
            W = weights.magnitude if isinstance(weights, ureg.Quantity) else weights
            window = (
                window.to(weights.units).magnitude
                if isinstance(window, ureg.Quantity)
                else window
            )
            if window is None:
                vmin = np.min(W[:, band_indices])
                vmax = np.max(W[:, band_indices])
            else:
                vmin, vmax = window
            return SimpleNamespace(
                ax=ax,
                x=x,
                eigen=eigen,
                band_indices=band_indices,
                weights=W,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            return SimpleNamespace(
                ax=ax,
                x=x,
                eigen=eigen,
                band_indices=band_indices,
            )

    def plot(
        self,
        ax: Axes = None,
        shift: float | ureg.Quantity = None,
        patched: bool = True,
        bands: list[int] = None,
        **kwargs,
    ) -> (Axes, Axes):
        """
        Plot the spectrum over a cumulative k-path.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift : float | pint.Quantity, optional
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
        P = self._pre_plot(ax, shift, bands, patched)
        label = kwargs.pop("label", None)  # remove label from kwargs
        P.ax.plot(P.x, P.eigen[:, P.band_indices[0]], label=label, **kwargs)
        P.ax.plot(P.x, P.eigen[:, P.band_indices[1:]], **kwargs)

        P.ax.set_xlim(0, 1)
        return P.ax

    def plot_fat(
        self,
        weights: np.ndarray,
        window: list[float, float] | ureg.Quantity = None,
        ax: Axes = None,
        size_change: bool = False,
        alpha_change: bool = False,
        shift: float | ureg.Quantity = None,
        patched: bool = True,
        bands: list[int] = None,
        **kwargs,
    ) -> (Axes, PathCollection):
        """
        Fat-band style plotting for weights over a cumulative k-path.

        These weights can represent projections over orbitals or other similar attributes.
        A point will be scattered at coordinates (k,E) with color, size, transparency related to the weights input.

        Parameters
        ----------
        weights : np.ndarray, pint.Quantity
            Array of shape (nkpts, neigs).
        window : list[float,float], optional
            Minimal and maximum values for the colormap of the weights.
            Default is minimal of maximal values for the set of weights.
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        size_change : bool, optional
            Whether the size of the dots should also change (linked to the window).
        alpha_change : bool, optional
            Whether the transparency (alpha) of the dots should also change (linked to the window).
        shift : float | pint.Quantity, optional
            A constant shift applied to the eigenvalues (e.g., Fermi level).
            Default is zero.
        patched : bool, optional
            If True, attempts to patch discontinuities in the k-path.
            This prevents artificially connected lines in band structure
            plots (e.g., flat bands across high-symmetry points).
        bands : list of int, optional
            Indices of the bands to plot. If None, all bands are plotted.
        **kwargs : dict
            Additional matplotlib arguments passed to `scatter()`.

        Returns
        ----------
        ax : Axes
            The axes with the spectrum plot.
        scatter : matplotlib.collections.PathCollection
            The PathCollection for generating the colorbar.
        """
        P = self._pre_plot(ax, shift, bands, patched, weights, window)

        # Remove some labels from kwargs
        label = kwargs.pop("label", None)
        s = kwargs.pop("s", pdft.weights_s)
        alpha = kwargs.pop("alpha", 1)
        if alpha_change:
            alpha = np.clip((P.weights - P.vmin) / (P.vmax - P.vmin), 0, 1)
        else:
            alpha = np.ones(P.weights.shape)
        if size_change:
            s = np.clip((P.weights - P.vmin) / (P.vmax - P.vmin), 0, 1) * s
        else:
            s = np.ones(P.weights.shape) * s

        scatter = P.ax.scatter(
            P.x,
            P.eigen[:, P.band_indices[0]],
            c=P.weights[:, P.band_indices[0]],
            s=s[:, P.band_indices[0]],
            alpha=alpha[:, P.band_indices[0]],
            vmin=P.vmin,
            vmax=P.vmax,
            label=label,
            edgecolors="none",
            **kwargs,
        )
        for i in P.band_indices[1:]:
            P.ax.scatter(
                P.x,
                P.eigen[:, i],
                c=P.weights[:, i],
                s=s[:, i],
                alpha=alpha[:, i],
                vmin=P.vmin,
                vmax=P.vmax,
                edgecolors="none",
                **kwargs,
            )

        P.ax.set_xlim(0, 1)
        return P.ax, scatter

    def plot_color(
        self,
        weights: np.ndarray,
        window: list[float, float] | ureg.Quantity = None,
        ax: Axes = None,
        shift: float | ureg.Quantity = None,
        patched: bool = True,
        bands: list[int] = None,
        **kwargs,
    ) -> (Axes, LineCollection):
        """
        Color gradient line-style for weights over a cumulative k-path.

        These weights can represent projections over orbitals or other similar attributes.
        A LineCollection will be plotted with color related to the weights input.

        Parameters
        ----------
        weights : np.ndarray, pint.Quantity
            Array of shape (nkpts, neigs).
        window : list[float,float], optional
            Minimal and maximum values for the colormap of the weights.
            Default is minimal of maximal values for the set of weights.
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift : float | pint.Quantity, optional
            A constant shift applied to the eigenvalues (e.g., Fermi level).
            Default is zero.
        patched : bool, optional
            If True, attempts to patch discontinuities in the k-path.
            This prevents artificially connected lines in band structure
            plots (e.g., flat bands across high-symmetry points).
        bands : list of int, optional
            Indices of the bands to plot. If None, all bands are plotted.
        **kwargs : dict
            Additional matplotlib arguments passed to `LineCollection()`.

        Returns
        ----------
        ax : Axes
            The axes with the spectrum plot.
        line : matplotlib.collections.LineCollection
            The LineCollection for generating the colorbar.
        """
        P = self._pre_plot(ax, shift, bands, patched, weights, window)

        # Remove some labels from kwargs
        label = kwargs.pop("label", None)
        linewidth = kwargs.pop("linewidth", pdft.gradcolor_w)

        norm = plt.Normalize(P.vmin, P.vmax)
        # Plotting band by band
        points = np.array([P.x.magnitude, P.eigen.magnitude[:, P.band_indices[0]]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            norm=norm,
            label=label,
            **kwargs,
        )
        lc.set_array(P.weights[:, P.band_indices[0]])
        lc.set_linewidth(linewidth)
        line = P.ax.add_collection(lc)
        for i in P.band_indices[1:]:
            points = np.array([P.x.magnitude, P.eigen.magnitude[:, P.band_indices[i]]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                norm=norm,
                **kwargs,
            )
            lc.set_array(P.weights[:, P.band_indices[i]])
            lc.set_linewidth(linewidth)
            P.ax.add_collection(lc)

        P.ax.autoscale_view()
        P.ax.set_xlim(0, 1)
        return P.ax, line


class ElectronBands(Spectrum):
    """
    Dressed `Spectrum` subclass for handling electronic bandstructures and spectrums.

    Attributes
    ----------
    filepath : str
        Path to the file containing electronic structure output.
    electron_num : int
        Total number of electrons in the system.
    fermi : float
        Fermi energy (0 if not found).
    """

    def __init__(self, file: str = None):
        """
        Initialize ElectronBands object.

        Parameters
        ----------
        file : str
            File from which to extract the bands.
        """
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
                eigenvalues=spec.energies,
                kpoints=spec.kpoints,
                weights=spec.weights,
                lattice=lattice,
            )
        else:
            self.electron_num = self.fermi = None
            Spectrum.__init__(self)


class PhononBands(Spectrum):
    """
    Dressed `Spectrum` subclass for handling phonon bandstructures and spectrums.

    Attributes
    ----------
    filepath : str
        Path to the file containing phonon frequencies output.
    """

    def __init__(self, file: str = None):
        """
        Initialize PhononBands object.

        Parameters
        ----------
        file : str
            Path to the file containing phonon frequencies output.
        """
        if file is not None:
            self.filepath = file
            try:
                lattice = grep.lattice(self.filepath)
            except NotImplementedError:
                lattice = None
            spec = grep.kpointsFrequencies(self.filepath)
            Spectrum.__init__(
                self,
                eigenvalues=spec.frequencies,
                kpoints=spec.kpoints,
                lattice=lattice,
            )
        else:
            Spectrum.__init__(self)


class DOS:
    """
    General class for storing density of states (DOS) values.

    Attributes
    ----------
    DOS : np.ndarray | pint.Quantity
        Array of shape (N,) with the corresponding DOS values (1/eigenvalue) units.
    vgrid : np.ndarray | pint.Quantity
        Array of shape (N,) with the eigenvalue units.
    parent : Spectrum | ElectronBands | PhononBands
        Parent class so that DOS can access parent attributes.

    Methods
    -------
    integrate(...)
        Integrate the density of states (DOS) up to a given energy or to determine
        the energy at which a certain number of states are filled.
    plot(...)
        Plot the DOS over an eigenvalue-window.
    """

    def __init__(
        self,
        DOS: np.ndarray | ureg.Quantity = None,
        vgrid: np.ndarray | ureg.Quantity = None,
        parent: Spectrum | ElectronBands | PhononBands = None,
    ):
        """
        Initialize DOS object.

        Parameters
        ----------
        DOS : np.ndarray | pint.Quantity
            Array of shape (N,) with the corresponding DOS values (1/eigenvalue) units.
        vgrid : np.ndarray | pint.Quantity
            Array of shape (N,) with the eigenvalue units.
        parent : Spectrum | ElectronBands | PhononBands, optional
            Parent class so that DOS can access parent attributes.
        """
        self.DOS = DOS
        self.vgrid = vgrid
        self._parent = parent

    def integrate(
        self, limit: float | ureg.Quantity = None, occ_states: float = None
    ) -> (float, float):
        """
        Integrate the density of states (DOS) up to a given energy or to determine
        the energy at which a certain number of states are filled.

        This method supports two use cases:
        1. **Plain integration**: returns the total number of states up to `limit`.
        2. **Inverse filling**: finds the energy at which the number of filled states equals `occ_states`.

        Parameters
        ----------
        limit : float or pint.Quantity, optional
            Upper energy bound for the integration. If None, integrates up to the
            maximum of `self.vgrid`.

        occ_states : float, optional
            Target number of filled states. If provided, the method finds the energy
            at which this number of states is reached by integration of the DOS.

        Returns
        -------
        tuple(float, float) : value, error
            - If `occ_states` is None: value is the integrated number of states (and estimated error).
            - If `occ_states` is set: value is the energy at which that number of states is filled (and error).

        Notes
        -----
        The method uses cubic interpolation and `scipy.integrate.quad` for accurate integration.
        A binary search strategy is used when `occ_states` is specified. Which can fail when the DOS has negative values.

        Raises
        ------
        RuntimeError:
           When convergence for the energy to get the filled states is not achieved in 100 iterations.
        """
        # Determine integration limit
        if limit is None:
            limit = self.vgrid[-1]

        # Unit consistency check
        quantities = [self.DOS, self.vgrid, limit]
        names = ["DOS", "vgrid", "limit"]
        ut._check_unit_consistency(quantities, names)

        # Handle Pint quantities
        if isinstance(self.DOS, ureg.Quantity):
            units = self.vgrid.units
            X = self.vgrid.magnitude
            Y = self.DOS.to(1 / units).magnitude
            X_max = limit.to(units).magnitude
        else:
            units = 1
            X = self.vgrid
            Y = self.DOS
            X_max = limit

        # Create interpolation function
        f_interp = interpolate.interp1d(
            X, Y, kind="cubic", fill_value=0.0, bounds_error=False
        )

        if occ_states is None:
            # Case 1: Plain integration
            integral, error = integrate.quad(f_interp, X[0], X_max, limit=100)
            return integral, error
        else:
            # Case 2: Inverse problem — find X_occ such that integral = occ_states
            X_low = X[0]
            X_high = X[-1]
            max_iter = 100

            for _ in range(max_iter):
                X_occ = 0.5 * (X_low + X_high)
                integral, error = integrate.quad(f_interp, X[0], X_occ, limit=100)
                if abs(integral - occ_states) < error:
                    break
                if integral > occ_states:
                    X_high = X_occ
                else:
                    X_low = X_occ
            else:
                raise RuntimeError(
                    "Did not converge to target occ_states within error tolerance in 100 iterations."
                )
            return X_occ * units, error * units

    def plot(
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
        shift : float | pint.Quantity, optional
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
            quantities = [self._parent.eigenvalues, shift]
            names = ["self.eigenvalues", "shift"]
        else:
            quantities = [self.DOS, self.vgrid, shift]
            names = ["self.DOS", "self.vgrid", "shift"]
        ut._check_unit_consistency(quantities, names)

        if ax is None:
            fig, ax = plt.subplots()

        if self.DOS is None:
            self._parent.get_DOS()
        x = self.vgrid if shift is None else self.vgrid - shift
        y = self.DOS

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
            ax.set_xlim(left=np.min(y))
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
            ax.set_ylim(bottom=np.min(y))

        return ax
