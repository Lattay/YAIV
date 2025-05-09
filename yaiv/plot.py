"""
YAIV | yaiv.plot
================

This module provides plotting utilities for visualizing eigenvalue spectra from periodic
systems. It supports electronic and vibrational spectra obtained from common ab initio
codes such as Quantum ESPRESSO and VASP.

Functions in this module are designed to work seamlessly with spectrum-like objects
(e.g., `spectrum`, `electronBands`, `phononBands`) and accept units-aware data.

The visualizations are based on `matplotlib`, and include options for:

- Plotting band structures and phonon spectra
- Automatically shifting eigenvalues (e.g., Fermi level)
- Detecting and patching discontinuities in the k-path
- Annotating high-symmetry points from KPOINTS or bands.in

Examples
--------
>>> from yaiv.plot import plot_spectrum
>>> from yaiv.grep import kpointsEnergies
>>> spectrum = kpointsEnergies("OUTCAR")
>>> plot_spectrum(spectrum)

>>> from yaiv import electron
>>> bands = electron.electronBands("qe.out")
>>> bands.plot()  # internally uses yaiv.plot utilities

See Also
--------
yaiv.spectrum : Base class for storing and plotting eigenvalue spectra
yaiv.grep     : Low-level data extractors used to populate spectrum objects
"""

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes

from yaiv.defaults.config import ureg
from yaiv.defaults.config import plot_defaults as pdef
from yaiv import utils as ut
from yaiv import spectrum as spec


def get_HSP_ticks(
    kpath: SimpleNamespace | np.ndarray, k_lattice: np.ndarray = None
) -> SimpleNamespace:
    """
    Compute tick positions and labels for high-symmetry points (HSPs) along a k-path.

    Parameters
    ----------
    kpath : SimpleNamespace or np.ndarray
        A k-path object as given by yaiv.grep.kpath()
    k_lattice : np.ndarray, optional
        3x3 matrix of reciprocal lattice vectors in rows (optional).
        If provided, the high-symmetry points are converted from crystal to Cartesian coordinates.

    Returns
    -------
    ticks : SimpleNamespace
        Object with the following attributes:
        - x_coord : np.ndarray
            Normalized cumulative distance for each high-symmetry point.
        - labels : list of str or None
            Corresponding labels for the ticks, or None if not available.
    """
    if isinstance(kpath, SimpleNamespace):
        path_array = kpath.path
        label_list = kpath.labels
    else:
        path_array = kpath
        label_list = None

    if isinstance(path_array, ureg.Quantity):
        path_array = path_array.magnitude

    segment_counts = [int(n) for n in path_array[:, -1]]
    hsp_coords = path_array[:, :3]

    # Convert to Cartesian coordinates if lattice is provided
    if k_lattice is not None:
        hsp_coords = ut.cryst2cartesian(hsp_coords, k_lattice).magnitude

    delta_k = np.diff(hsp_coords, axis=0)
    segment_lengths = np.linalg.norm(delta_k, axis=1)

    x_coord = [0.0]
    for i, length in enumerate(segment_lengths):
        if segment_counts[i] != 1:
            x_coord.append(x_coord[-1] + length)

    x_coord = np.array(x_coord)
    x_coord /= x_coord[-1]  # Normalize to [0, 1]

    # Merge labels at discontinuities (where N=1)
    if label_list is not None:
        merged_labels = []
        for i, label in enumerate(label_list):
            label = label.strip()
            latex_label = r"$\Gamma$" if label.lower() == "gamma" else rf"${label}$"
            if i != 0 and segment_counts[i - 1] == 1:
                merged_labels[-1] = merged_labels[-1][:-1] + "|" + latex_label[1:]
            else:
                merged_labels.append(latex_label)
    else:
        merged_labels = None
    ticks = SimpleNamespace(ticks=x_coord, labels=merged_labels)
    return ticks


def kpath(
    ax: matplotlib.axes._axes.Axes,
    kpath: SimpleNamespace | np.ndarray,
    k_lattice: np.ndarray = None,
):
    """
    Plots the high-symmetry points (HSPs) along a k-path in a given ax.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes to plot on. If None, a new figure and axes are created.
    kpath : SimpleNamespace or np.ndarray
        A k-path object as given by yaiv.grep.kpath()
    k_lattice : np.ndarray, optional
        3x3 matrix of reciprocal lattice vectors in rows (optional).
        If provided, the high-symmetry points are converted from crystal to Cartesian coordinates.
    """
    ticks = get_HSP_ticks(kpath, k_lattice)
    for tick in ticks.ticks:
        ax.axvline(
            tick,
            color=pdef.vline_c,
            linewidth=pdef.vline_w,
            linestyle=pdef.vline_s,
        )
    if ticks.labels is not None:
        ax.set_xticks(ticks.ticks, ticks.labels)
    else:
        ax.set_xticks(ticks.ticks)
    ax.xaxis.label.set_visible(False)


def _compare_spectra(
    spectra: list[spec.spectrum],
    ax: matplotlib.axes._axes.Axes,
    patched: bool = True,
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes:

    user_color = kwargs.pop("color", None)  # user-defined color overrides all
    user_label = kwargs.pop("label", None)  # user-defined label

    cycle_iter = iter(pdef.color_cycle)
    for i, S in enumerate(spectra):
        color = user_color or (
            colors[i] if colors is not None and i < len(colors) else next(cycle_iter)
        )
        label = user_label or (
            labels[i] if labels is not None and i < len(labels) else f"Band {i+1}"
        )
        S.plot(
            ax,
            S.fermi,
            patched,
            color=color,
            label=label,
            **kwargs,
        )
    ax.legend()
    return ax


def bands(
    electronBands: spec.electronBands | list[spec.electronBands],
    ax: matplotlib.axes._axes.Axes = None,
    patched: bool = True,
    window: list[float] | float = [-1, 1],
    colors: list[str] = None,
    labels: list[str] = None,
    deg: bool = False,
    **kwargs,
) -> matplotlib.axes._axes.Axes:
    """
    Plot electronic band structures for one or multiple systems.

    Parameters
    ----------
    electronBands : electronBands or list of electronBands
        Band structure objects to plot.
    ax : matplotlib.axes._axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : list[float] or float, optional
        Energy window to be shown.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        Axes containing the plot, if one was provided as input.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if type(electronBands) is not list:
        user_color = kwargs.pop("color", None)  # user-defined color overrides all
        user_label = kwargs.pop("label", None)  # user-defined label
        band = electronBands
        indices = list(range(band.eigenvalues.shape[1]))
        # plot valence bands
        band.plot(
            ax,
            band.fermi,
            patched,
            bands=indices[: band.electron_num],
            color=user_color or pdef.valence_c,
            label=user_label,
            **kwargs,
        )
        # plot conduction bands
        band.plot(
            ax,
            band.fermi,
            patched,
            bands=indices[band.electron_num :],
            color=user_color or pdef.conduction_c,
            **kwargs,
        )
    else:
        _compare_spectra(electronBands, ax, patched, colors, labels, **kwargs)
        band = electronBands[0]

    if band.kpath is not None:
        kpath(ax, band.kpath, band.k_lattice)

    if band.fermi is not None:
        ax.axhline(y=0, color=pdef.fermi_c, linewidth=pdef.fermi_w)

    if type(window) is int or type(window) is float:
        window = [-window, window]
    ax.set_ylim(window[0], window[1])

    plt.tight_layout()
    return ax


def phonons(
    phononBands: spec.phononBands | list[spec.phononBands],
    ax: matplotlib.axes._axes.Axes = None,
    patched: bool = True,
    window: list[float] | float = [-1, 1],
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes:
    """
    Plot electronic band structures for one or multiple systems.

    Parameters
    ----------
    phononBands : phononBands or list of phononBands
        Phonon band objects to plot.
    ax : matplotlib.axes._axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : list[float] or float, optional
        Energy window to be shown.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        Axes containing the plot, if one was provided as input.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if type(phononBands) is not list:
        user_color = kwargs.pop("color", None)  # user-defined color overrides all
        user_label = kwargs.pop("label", None)  # user-defined label
        band = phononBands
        band.plot(
            ax,
            patched=patched,
            color=user_color or pdef.valence_c,
            label=user_label,
            **kwargs,
        )
    else:
        _compare_spectra(phononBands, ax, patched, colors, labels, **kwargs)
        band = phononBands[0]

    if band.kpath is not None:
        kpath(ax, band.kpath, band.k_lattice)

    ax.axhline(y=0, color=pdef.fermi_c, linewidth=pdef.fermi_w)

    plt.tight_layout()
    return ax
