"""
YAIV | yaiv.plot
================

This module provides plotting utilities for visualizing eigenvalue spectra from periodic
systems. It supports electronic and vibrational spectra obtained from common ab initio
codes such as Quantum ESPRESSO and VASP.

Functions in this module are designed to work seamlessly with spectrum-like objects
(e.g., `Spectrum`, `ElectronBands`, `PhononBands`) and accept units-aware data.

The visualizations are based on `matplotlib`, and include options for:

- Plotting band structures and phonon spectra
- Automatically shifting eigenvalues (e.g., Fermi level)
- Detecting and patching discontinuities in the k-path
- Annotating high-symmetry points from KPOINTS or bands.in

Functions
---------
get_HSP_ticks(kpath, k_lattice=None)
    Computes tick positions and LaTeX labels for high-symmetry points along a k-path.

kpath(ax, kpath, k_lattice=None)
    Plots vertical lines and labels at high-symmetry points in a matplotlib Axes.

bands(electronBands, ax=None, ...)
    Plots the electronic band structure for one or multiple systems.

phonons(phononBands, ax=None, ...)
    Plots the phonon band structure for one or multiple systems.

DOS(spectra, ax=None, ...)
    Plots the density of states (DOS) for a single or multiple eigenvalue spectra.

bandsDOS(electronBands, fig=None, axes=None, ...)
    Plots a band structure and its corresponding DOS side-by-side.

phononDOS(phononBands, fig=None, axes=None, ...)
    Plots a phonon band structure and its corresponding DOS side-by-side.

Private Utilities
-----------------
_compare_spectra(spectra, ax, ...)
    Internal utility to overlay multiple spectra on the same Axes with legends and formatting.

_spectra_DOS(spectra, plot_func, ...)
    Internal helper to produce spectrum + DOS panels for electronic or vibrational bands.

Examples
--------
>>> from yaiv.spectrum import ElectronBands
>>> from yaiv import plot
>>> bands = ElectronBands("OUTCAR")
>>> plot.bands(bands)

See Also
--------
yaiv.spectrum : Base class for storing and plotting eigenvalue spectra
yaiv.grep     : Low-level data extractors used to populate spectrum objects
yaiv.defaults : Configuration and default plotting values
"""

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from yaiv.defaults.config import ureg
from yaiv.defaults.config import plot_defaults as pdft
from yaiv import utils as ut
from yaiv import spectrum as spec


__all__ = [
    "get_HSP_ticks",
    "kpath",
    "bands",
    "phonons",
    "DOS",
    "bandsDOS",
    "phononDOS",
]


def get_HSP_ticks(
    kpath: SimpleNamespace | np.ndarray,
    k_lattice: np.ndarray = None,
    grid: list[int] = None,
) -> SimpleNamespace:
    """
    Compute tick positions and labels for high-symmetry points (HSPs) along a k-path.
    And optionally also the ticks for the grid points that lie in the path.

    Parameters
    ----------
    kpath : SimpleNamespace or np.ndarray
        A k-path object as given by yaiv.grep.kpath()
    k_lattice : np.ndarray, optional
        3x3 matrix of reciprocal lattice vectors in rows (optional).
        If provided, the high-symmetry points are converted from crystal to Cartesian coordinates.
    grid : list[int], optional
        Γ centred grid to show in the path.

    Returns
    -------
    ticks : SimpleNamespace
        Object with the following attributes:
        - ticks : np.ndarray
            Normalized cumulative distance for each high-symmetry point.
        - labels : list of str or None
            Corresponding labels for the ticks, or None if not available.
        - grid : np.ndarray
            Normalized cumulative distance for each grid point in the k-path.
    """
    if isinstance(kpath, SimpleNamespace):
        path_array = kpath.path
        label_list = kpath.labels
    else:
        path_array = kpath
        label_list = None
    if grid is not None:
        grid = ut.grid_generator(grid, periodic=True) * ureg('_2pi/crystal')
        grid = ut._expand_zone_border(grid)

    # Handle units
    quantities, names = [path_array, k_lattice], ["kpath", "k_lattice"]
    ut._check_unit_consistency(quantities, names)
    if isinstance(path_array, ureg.Quantity):
        units = path_array.units
        path_array = path_array.magnitude
    else:
        units = 1

    segment_counts = [int(n) for n in path_array[:, -1]]
    hsp_coords = path_array[:, :3] * units

    # Convert to Cartesian coordinates if lattice is provided
    if k_lattice is not None:
        hsp_coords = ut.cryst2cartesian(hsp_coords, k_lattice).magnitude
        if grid is not None:
            grid = ut.cryst2cartesian(grid, k_lattice).magnitude
    else:
        hsp_coords = hsp_coords.magnitude

    # Ticks positions
    x_coord, grid_coord = [0.0], []
    for i, s in enumerate(segment_counts):
        if s != 1:
            length = np.linalg.norm(hsp_coords[i + 1] - hsp_coords[i])
            x_coord.append(x_coord[-1] + length)
            if grid is not None:
                for g in grid:
                    seg_distance = ut._point_to_segment_distance(
                        g, hsp_coords[i], hsp_coords[i + 1]
                    )
                    if np.around(seg_distance, decimals=3) == 0:
                        if (
                            np.around(np.linalg.norm(g - hsp_coords[i]), decimals=3)
                            == 0
                        ):
                            grid_coord.append(x_coord[-2])
                        elif (
                            np.around(
                                np.linalg.norm(g - [hsp_coords[i + 1]]), decimals=3
                            )
                            == 0
                        ):
                            grid_coord.append(x_coord[-1])
                        else:
                            lenght = np.linalg.norm(g - hsp_coords[i])
                            grid_coord.append(x_coord[-2] + lenght)
    x_coord = np.array(x_coord)
    # Normalize to [0, 1]
    grid_coord /= x_coord[-1]
    x_coord /= x_coord[-1]

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
    ticks = SimpleNamespace(ticks=x_coord, labels=merged_labels, grid=grid_coord)
    return ticks


def kpath(
    ax: Axes,
    kpath: SimpleNamespace | np.ndarray,
    k_lattice: np.ndarray = None,
    grid: list[int] = None,
):
    """
    Plots the high-symmetry points (HSPs) along a k-path in a given ax. And optionally,
    also the ticks for the grid points that lie in the path.

    Parameters
    ----------
    ax : Axes
        Axes to plot on. If None, a new figure and axes are created.
    kpath : SimpleNamespace or np.ndarray
        A k-path object as given by yaiv.grep.kpath()
    k_lattice : np.ndarray, optional
        3x3 matrix of reciprocal lattice vectors in rows (optional).
        If provided, the high-symmetry points are converted from crystal to Cartesian coordinates.
    grid : list[int], optional
        Γ centred grid to show in the path.
    """
    ticks = get_HSP_ticks(kpath, k_lattice, grid)
    for tick in ticks.ticks:
        ax.axvline(
            tick,
            color=pdft.vline_c,
            linewidth=pdft.vline_w,
            linestyle=pdft.vline_s,
        )
    for tick in ticks.grid:
        ax.axvline(
            tick,
            color=pdft.grid_c,
            linewidth=pdft.grid_w,
            linestyle=pdft.vline_s,
        )
    if ticks.labels is not None:
        ax.set_xticks(ticks.ticks, ticks.labels)
    else:
        ax.set_xticks(ticks.ticks)
    ax.xaxis.label.set_visible(False)


def _compare_spectra(
    spectra: list[spec.Spectrum],
    ax: Axes,
    patched: bool = True,
    colors: list[str] = None,
    labels: list[str] = None,
    grid: list[list[int]] = None,
    **kwargs,
) -> Axes:
    """
    Plot and compare multiple spectra on a shared axes object.

    Parameters
    ----------
    spectra : list[spec.Spectrum]
    A list of spectrum objects to be plotted. Each spectrum must implement
        a `.plot()` method compatible with the plotting interface.
    ax : Axes
        Axes to plot on.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    colors : list[str], optional
        Colors to use when plotting multiple bands.
    labels : list[str], optional
        Labels to assign to each band in multi-plot case.
    grid : list[list[int]], optional
        Γ centred grids to show in the path.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        Axes containing the plot.
    """

    cycle_iter = iter(pdft.color_cycle)
    if len(np.shape(grid)) == 2 and np.shape(grid)[0] == len(spectra):
        GRID = True
    else:
        GRID = False
    for i, S in enumerate(spectra):
        color = (
            colors[i] if colors is not None and i < len(colors) else next(cycle_iter)
        )

        label = (
            labels[i] if labels is not None and i < len(labels) else f"Spectrum {i+1}"
        )
        ax = S.plot(
            ax=ax,
            shift=getattr(S, "fermi", None),
            patched=patched,
            color=color,
            label=label,
            **kwargs,
        )
        if GRID:
            ticks = get_HSP_ticks(spectra[-1].kpath, spectra[-1].k_lattice, grid[i])
            for tick in ticks.grid:
                ax.axvline(
                    tick,
                    color=color,
                    linewidth=pdft.grid_w,
                    linestyle=pdft.vline_s,
                )
    ax.legend()
    return ax


def bands(
    electronBands: spec.ElectronBands | list[spec.ElectronBands],
    ax: Axes = None,
    patched: bool = True,
    window: list[float] | float | ureg.Quantity = [-1, 1] * ureg("eV"),
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> Axes:
    """
    Plot electronic band structures for one or multiple systems.

    Parameters
    ----------
    electronBands : ElectronBands or list of ElectronBands
        Band structure objects to plot.
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : list[float] | float | ureg.Quantity, optional
        Energy window to be shown, default is [-1,1] eV.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        Axes containing the plot.
    """

    # Pop user-level styling
    user_color = kwargs.pop("color", None)  # user-defined color overrides all
    user_label = kwargs.pop("label", None)  # user-defined label

    if type(electronBands) is not list:
        band = electronBands
        indices = list(range(band.eigenvalues.shape[1]))
        # For non-spin orbit weights might add up to 2.
        valence_bands = round(band.electron_num / np.sum(band.weights))
        # plot valence bands
        ax = band.plot(
            ax=ax,
            shift=band.fermi,
            patched=patched,
            bands=indices[:valence_bands],
            color=user_color or pdft.valence_c,
            label=user_label,
            **kwargs,
        )
        # plot conduction bands
        ax = band.plot(
            ax=ax,
            shift=band.fermi,
            patched=patched,
            bands=indices[valence_bands:],
            color=user_color or pdft.conduction_c,
            **kwargs,
        )
    else:
        ax = _compare_spectra(electronBands, ax, patched, colors, labels, **kwargs)
        band = electronBands[0]

    if band.kpath is not None:
        kpath(ax, band.kpath, band.k_lattice)

    if band.fermi is not None:
        ax.axhline(y=0, color=pdft.fermi_c, linewidth=pdft.fermi_w)

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
    phononBands: spec.PhononBands | list[spec.PhononBands],
    ax: Axes = None,
    patched: bool = True,
    window: list[float] | float | ureg.Quantity = None,
    colors: list[str] = None,
    labels: list[str] = None,
    grid: list[int] | list[list[int]] = None,
    **kwargs,
) -> Axes:
    """
    Plot electronic band structures for one or multiple systems.

    Parameters
    ----------
    phononBands : spec.PhononBands | list[spec.PhononBands]
        Phonon band objects to plot.
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : list[float] | float | ureg.Quantity, optional
        Frequency window to be shown, default is the whole spectra.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    grid : list[int] | list[list[int]], optional
        Γ centred grid (or grids) to show in the path.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        Axes containing the plot.
    """

    # Pop user-level styling
    user_color = kwargs.pop("color", None)  # user-defined color overrides all
    user_label = kwargs.pop("label", None)  # user-defined label

    if type(phononBands) is not list:
        band = phononBands
        ax = band.plot(
            ax,
            patched=patched,
            color=user_color or pdft.valence_c,
            label=user_label,
            **kwargs,
        )
    else:
        ax = _compare_spectra(
            phononBands, ax, patched, colors, labels, grid=grid, **kwargs
        )
        band = phononBands[0]

    if band.kpath is not None:
        if grid is None or len(np.shape(grid)) == 1:
            kpath(ax, band.kpath, band.k_lattice, grid)
        else:
            kpath(ax, band.kpath, band.k_lattice)

    # Handle units and setup window
    if window is not None:
        window = (
            window.to(band.eigenvalues.units).magnitude
            if isinstance(window, ureg.Quantity)
            else window
        )
        if type(window) is int or type(window) is float:
            window = [-window, window]
        ax.set_ylim(window[0], window[1])

    ax.axhline(y=0, color=pdft.fermi_c, linewidth=pdft.fermi_w)

    plt.tight_layout()
    return ax


def DOS(
    spectra: spec.ElectronBands | spec.PhononBands | spec.Spectrum,
    ax: Axes = None,
    window: float | list[float] | ureg.Quantity = None,
    smearing: float | ureg.Quantity = None,
    steps: int = None,
    order: int = 0,
    cutoff_sigmas: float = 3.0,
    switchXY: bool = False,
    fill: bool = True,
    alpha: float = pdft.alpha,
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> Axes:
    """
    Plot the density of states (DOS) for a single or list of spectra.

    Parameters
    ----------
    spectra : spec.ElectronBands | spec.PhononBands | spec.Spectrum
        Spectra from which to plot the DOS.
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    window : float | list[float] | ureg.Quantity, optional
        Value window for the DOS. If float, interpreted as symmetric [-window, window].
        As a default and if a Fermi level is present it will compute the DOS in a [-1,1] eV window
        centered at E_f, otherwise it defaults to the whole eigenvalue range.
    smearing : float | ureg.Quantity, optional
        Gaussian smearing width in the same units as eigenvalues. Default is (window_size/200).
    steps : int, optional
        Number of grid points for DOS sampling. Default is 4 * (window_size/smearing).
    order : int, optional
        Order of the Methfessel-Paxton expansion. Default is 0, which recovers a Gaussian smearing.
    cutoff_sigmas : float, optional
        Number of smearing widths to use for truncation (e.g., 3 means ±3σ).
    switchXY : bool, optional
        Whether to plot the DOS along the x-axis (horizontal plot). Default is False.
    fill : bool, optional
        Whether to fill the area under the curve. Default is True.
    alpha : float, optional
        Opacity of the fill (0 = transparent, 1 = solid).
    colors : list[str], optional
        Colors to use when plotting multiple DOS.
    labels : list[str], optional
        Labels to assign to each DOS in multi-plot case.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        Axes containing the plot.
    """
    # Extract first spectrum for defaults
    S = spectra[0] if isinstance(spectra, list) else spectra

    # Default window based on presence of Fermi level
    if window is None:
        window = [-1, 1] * ureg.eV if hasattr(S, "fermi") else None

    # Pop user-level styling
    user_color = kwargs.pop("color", None)  # user-defined color overrides all
    user_label = kwargs.pop("label", None)  # user-defined label

    if not isinstance(spectra, list):
        # Single plot
        S.get_DOS(
            center=getattr(S, "fermi", None),
            window=window,
            smearing=smearing,
            steps=steps,
            order=order,
            cutoff_sigmas=cutoff_sigmas,
        )
        ax = S.DOS.plot(
            ax,
            shift=getattr(S, "fermi", None),
            switchXY=switchXY,
            fill=fill,
            alpha=alpha,
            color=user_color or (pdft.DOS_c if hasattr(S, "fermi") else None),
            label=user_label,
            **kwargs,
        )
    else:
        # Multi plot
        cycle_iter = iter(pdft.color_cycle)
        zorder = 2
        for i, S in enumerate(spectra):
            color = (
                colors[i]
                if colors is not None and i < len(colors)
                else next(cycle_iter)
            )
            label = (
                labels[i] if labels is not None and i < len(labels) else f"DOS {i+1}"
            )
            S.get_DOS(
                center=getattr(S, "fermi", None),
                window=window,
                smearing=smearing,
                steps=steps,
                order=order,
                cutoff_sigmas=cutoff_sigmas,
            )
            ax = S.DOS.plot(
                ax,
                shift=getattr(S, "fermi", None),
                switchXY=switchXY,
                fill=fill,
                alpha=alpha,
                color=color,
                label=label,
                zorder=zorder,
                **kwargs,
            )
            zorder += 2
        ax.legend()

    if hasattr(S, "fermi"):
        if switchXY == True:
            ax.axhline(y=0, color=pdft.fermi_c, linewidth=pdft.fermi_w)
        else:
            ax.axvline(x=0, color=pdft.fermi_c, linewidth=pdft.fermi_w)
    # Labels
    if switchXY:
        if isinstance(S.DOS.density, ureg.Quantity):
            ax.set_xlabel(f"DOS ({S.DOS.density.units})")
        else:
            ax.set_xlabel("DOS")
    else:
        if isinstance(S.DOS.density, ureg.Quantity):
            ax.set_ylabel(f"DOS ({S.DOS.density.units})")
        else:
            ax.set_ylabel("DOS")

    plt.tight_layout()
    return ax


def _spectra_DOS(
    spectra: spec.ElectronBands | spec.PhononBands | list,
    plot_func: callable,
    fig: Figure = None,
    axes: list[Axes] = None,
    patched: bool = True,
    window: float | list[float] | ureg.Quantity = None,
    colors: list[str] = None,
    labels: list[str] = None,
    grid: list[int] | list[list[int]] = None,
    **kwargs,
) -> tuple[Axes, Axes]:
    """
    Internal helper to plot a spectrum and its corresponding DOS.

    Parameters
    ----------
    spectra : spec.ElectronBands | spec.PhononBands | list,
        List of spectrum objects (e.g., ElectronBands or PhononBands).
    plot_func : callable
        Function to plot the band structure (e.g., `bands()` or `phonons()`).
    fig : Figure, optional
        Optional figure object to plot into. If not provided, a new one is created.
    axes : list of Axes, optional
        Optional list or tuple of two axes [ax_band, ax_DOS] to use instead of creating new ones.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : float | list[float] | Quantity, optional
        Energy/frequency window to show. Interpreted symmetrically if float.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    grid : list[int] | list[list[int]], optional
        Γ centred grid (or grids) to show in the path.
    **kwargs
        Additional keyword arguments passed to `plot_func()` and `DOS()`.

    Returns
    -------
    ax : Axes
        Axis with the band or phonon structure.
    ax_DOS : Axes
        Axis with the horizontal DOS plot.
    """
    if axes is not None:
        ax, ax_DOS = axes
        fig = ax.figure
    else:
        fig = fig or plt.figure()
        gs = fig.add_gridspec(
            1,
            2,
            hspace=0,
            wspace=0,
            width_ratios=[1 - pdft.bandsDOS_ratio, pdft.bandsDOS_ratio],
        )
        ax, ax_DOS = gs.subplots(sharex="col", sharey="row")

    user_color = kwargs.pop("color", None)
    user_label = kwargs.pop("label", None)

    if grid is None:
        plot_func(
            spectra,
            ax=ax,
            patched=patched,
            window=window,
            colors=colors,
            labels=labels,
            color=user_color,
            label=user_label,
            **kwargs,
        )
    else:
        plot_func(
            spectra,
            ax=ax,
            patched=patched,
            window=window,
            colors=colors,
            labels=labels,
            color=user_color,
            label=user_label,
            grid=grid,
            **kwargs,
        )

    DOS(
        spectra,
        ax=ax_DOS,
        switchXY=True,
        window=window,
        colors=colors,
        labels=labels,
        color=user_color,
        label=user_label,
        **kwargs,
    )

    # Clean up DOS axis
    ax_DOS.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    legend = ax_DOS.get_legend()
    if legend is not None:
        legend.remove()
    for name, spine in ax_DOS.spines.items():
        if name not in ["bottom", "left"]:
            spine.set_visible(False)
    ax_DOS.set_xlabel("DOS")
    plt.tight_layout()
    return ax, ax_DOS


def bandsDOS(
    electronBands: spec.ElectronBands | list[spec.ElectronBands],
    fig: Figure = None,
    axes: list[Axes] = None,
    patched: bool = True,
    window: list[float] | float | ureg.Quantity = [-1, 1] * ureg("eV"),
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> tuple[Axes, Axes]:
    """
    Plot a band structure and its corresponding density of states (DOS) side-by-side.

    Parameters
    ----------
    electronBands : spec.ElectronBands | list[spec.ElectronBands]
        A spectrum or list of spectra representing electronic band structures.
    fig : Figure, optional
        Optional figure object to plot into. If not provided, a new figure is created.
    axes : list[Axes], optional
        Optional list or tuple of two axes [ax_band, ax_DOS] to use instead of creating new ones.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : float | list[float] | Quantity, optional
        Energy window to be shown, default is [-1,1] eV.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    **kwargs
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        The axis containing the band structure plot.
    ax_DOS : Axes
        The axis containing the DOS plot.
    """

    return _spectra_DOS(
        spectra=electronBands,
        plot_func=bands,
        fig=fig,
        axes=axes,
        patched=patched,
        window=window,
        colors=colors,
        labels=labels,
        **kwargs,
    )


def phononsDOS(
    phononBands: spec.PhononBands | list[spec.PhononBands],
    fig: Figure = None,
    axes: list[Axes] = None,
    patched: bool = True,
    window: list[float] | float | ureg.Quantity = None,
    colors: list[str] = None,
    labels: list[str] = None,
    grid: list[int] | list[list[int]] = None,
    **kwargs,
) -> tuple[Axes, Axes]:
    """
    Plot a phonon band structure and its corresponding density of states (DOS) side-by-side.

    Parameters
    ----------
    phononBands : spec.PhononBands | list[spec.PhononBands]
        A spectrum or list of spectra representing phonon band structures.
    fig : Figure, optional
        Optional figure object to plot into. If not provided, a new figure is created.
    axes : list[Axes], optional
        Optional list or tuple of two axes [ax_band, ax_DOS] to use instead of creating new ones.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : float | list[float] | Quantity, optional
        Energy window to be shown, default is [-1,1] eV.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    grid : list[int] | list[list[int]], optional
        Γ centred grid (or grids) to show in the path.
    **kwargs
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        The axis containing the band structure plot.
    ax_DOS : Axes
        The axis containing the DOS plot.
    """

    ax, ax_DOS = _spectra_DOS(
        spectra=phononBands,
        plot_func=phonons,
        fig=fig,
        axes=axes,
        patched=patched,
        window=window,
        colors=colors,
        labels=labels,
        grid=grid,
        **kwargs,
    )
    ax.autoscale(), ax.set_xlim([0, 1])

    plt.tight_layout()
    return ax, ax_DOS
