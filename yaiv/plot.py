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
>>> from yaiv.spectrum import electronBands
>>> from yaiv import plot
>>> bands = kpointsEnergies("OUTCAR")
>>> plot.bands(bands)

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
        ax = S.plot(
            ax=ax,
            patched=patched,
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
    window: list[float] | float | ureg.Quantity = [-1, 1] * ureg("eV"),
    colors: list[str] = None,
    labels: list[str] = None,
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

    if type(electronBands) is not list:
        user_color = kwargs.pop("color", None)  # user-defined color overrides all
        user_label = kwargs.pop("label", None)  # user-defined label
        band = electronBands
        indices = list(range(band.eigenvalues.shape[1]))
        # plot valence bands
        ax = band.plot(
            ax,
            band.fermi,
            patched,
            bands=indices[: band.electron_num],
            color=user_color or pdef.valence_c,
            label=user_label,
            **kwargs,
        )
        # plot conduction bands
        ax = band.plot(
            ax,
            band.fermi,
            patched,
            bands=indices[band.electron_num :],
            color=user_color or pdef.conduction_c,
            **kwargs,
        )
    else:
        ax = _compare_spectra(electronBands, ax, patched, colors, labels, **kwargs)
        band = electronBands[0]

    if band.kpath is not None:
        kpath(ax, band.kpath, band.k_lattice)

    if band.fermi is not None:
        ax.axhline(y=0, color=pdef.fermi_c, linewidth=pdef.fermi_w)

    # Handle units and setup window
    window = (
        window.to(band.eigenvalues.units).magnitude
        if isinstance(window, ureg.Quantity)
        else window
    )
    if type(window) is int or type(window) is float:
        window = [-window, window]
    ax.set_ylim(window[0], window[1])

    plt.tight_layout()
    return ax


def phonons(
    phononBands: spec.phononBands | list[spec.phononBands],
    ax: matplotlib.axes._axes.Axes = None,
    patched: bool = True,
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

    if type(phononBands) is not list:
        user_color = kwargs.pop("color", None)  # user-defined color overrides all
        user_label = kwargs.pop("label", None)  # user-defined label
        band = phononBands
        ax = band.plot(
            ax,
            patched=patched,
            color=user_color or pdef.valence_c,
            label=user_label,
            **kwargs,
        )
    else:
        ax = _compare_spectra(phononBands, ax, patched, colors, labels, **kwargs)
        band = phononBands[0]

    if band.kpath is not None:
        kpath(ax, band.kpath, band.k_lattice)

    ax.axhline(y=0, color=pdef.fermi_c, linewidth=pdef.fermi_w)

    plt.tight_layout()
    return ax


def DOS(
    spectra: spec.electronBands | spec.phononBands | spec.spectrum,
    ax: matplotlib.axes._axes.Axes = None,
    window: float | list[float] = None,
    smearing: float | ureg.Quantity = None,
    steps: int = None,
    precision: float = 3.0,
    fill: bool = True,
    switchXY: bool = False,
    alpha: float = pdef.alpha,
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes:
    """
    [TODO:summary]

    [TODO:description]

    Parameters
    ----------
    spectra : spec.electronBands | spec.phononBands | spec.spectrum
        Spectra from which to plot the DOS.
    ax : matplotlib.axes._axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    window : float | list[float] | ureg.Quantity, optional
        Value window for the DOS. If float, interpreted as symmetric [-window, window].
        If list, used as [Vmin, Vmax]. If None, the eigenvalue range is used.
    smearing : float | ureg.Quantity, optional
        Gaussian smearing width in the same units as eigenvalues. Default is (window_size/200).
    steps : int
        [TODO:description]
    precision : float
        [TODO:description]
    fill : bool
        [TODO:description]
    switchXY : bool
        [TODO:description]
    alpha : float
        [TODO:description]
    colors : list[str]
        [TODO:description]
    labels : list[str]
        [TODO:description]

    Returns
    -------
    matplotlib.axes._axes.Axes
        [TODO:description]
    """

    user_color = kwargs.pop("color", None)  # user-defined color overrides all
    user_label = kwargs.pop("label", None)  # user-defined label

    if ax is None:
        fig, ax = plt.subplots()

    if type(spectra) is not list:
        spectra.get_DOS(
            window=window, smearing=smearing, steps=steps, precision=precision
        )
        ax = spectra.plot_DOS(
            ax,
            switchXY=switchXY,
            fill=fill,
            alpha=alpha,
            color=user_color or pdef.DOS_c,
            label=user_label,
            **kwargs,
        )
    else:
        cycle_iter = iter(pdef.color_cycle)
        zorder = 2
        for i, S in enumerate(spectra):
            color = user_color or (
                colors[i]
                if colors is not None and i < len(colors)
                else next(cycle_iter)
            )
            label = user_label or (
                labels[i] if labels is not None and i < len(labels) else f"DOS {i+1}"
            )
            S.get_DOS(
                window=window, smearing=smearing, steps=steps, precision=precision
            )
            ax = S.plot_DOS(
                ax,
                switchXY=switchXY,
                fill=fill,
                alpha=alpha,
                color=color,
                label=label,
                zorder=zorder,
                **kwargs,
            )
            zorder = zorder + 2
        ax.legend()
        spectra = spectra[0]
    if hasattr(spectra, "fermi"):
        if switchXY == True:
            ax.axhline(y=0, color=pdef.fermi_c, linewidth=pdef.fermi_w)
        else:
            ax.axvline(x=0, color=pdef.fermi_c, linewidth=pdef.fermi_w)
    plt.tight_layout()
    return ax
