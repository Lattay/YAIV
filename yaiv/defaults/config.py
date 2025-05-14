"""
YAIV | yaiv.defaults.config
===========================

This module defines global configuration options and physical unit handling
for the YAIV (Yet Another Ab Initio Visualizer) library.

It initializes a `pint.UnitRegistry` used across the codebase for handling
quantities with units (e.g., energies in eV, lattice vectors in Å). It also
provides default plotting settings for consistent visual style across all
YAIV plots.

The plotting defaults are defined in a `SimpleNamespace` object `plot_defaults`,
which centralizes control over line widths, marker styles, color palettes, font sizes, etc.
These settings are also used to update matplotlib's global `rcParams`.
"""

from types import SimpleNamespace
from importlib.resources import files

import numpy as np
import pint
import matplotlib
import matplotlib.pyplot as plt

# === Units ===
ureg = pint.UnitRegistry()
ureg.setup_matplotlib(True)
ureg.load_definitions(files("yaiv") / "defaults/extra_units.txt")


def inv_quantity(matrix: np.ndarray | ureg.Quantity) -> np.ndarray | ureg.Quantity:
    """
    Inverts a matrix with (or without) units of 1/[input_units].

    Parameters
    ----------
    matrix : np.ndarray | ureg.Quantity
        Square matrix, with or without units.

    Returns
    -------
    inverse : np.ndarray | ureg.Quantity
        Square matrix, with (1/[input]) or without units (depending on the input).
    """
    if isinstance(matrix, pint.Quantity):
        return np.linalg.inv(matrix.magnitude) * (1 / matrix.units)
    else:
        return np.linalg.inv(matrix)


# === Plotting defaults ===

plot_defaults = SimpleNamespace(
    color_cycle=plt.get_cmap("tab10").colors,  # tuple of 10 RGB colors
    #    linewidth=1.5,
    vline_w=0.4,
    vline_c="gray",
    vline_s="--",
    fermi_c="black",
    fermi_w=0.4,
    valence_c="tab:blue",
    conduction_c="tab:red",
    alpha=0.5,
    DOS_c='black',
    #    linestyle='-',
    #    marker='o',
    #    markersize=4,
    #    cmap='viridis',
    #    font_size=12,
    #    label_size=10,
    #    tick_size=10,
    #    axis_linewidth=1.2,
    #    dpi=100,
)

# Optional: override matplotlib rcParams directly if desired
matplotlib.rcParams["axes.prop_cycle"] = plt.cycler(color=plot_defaults.color_cycle)
# matplotlib.rcParams["lines.linewidth"] = plot_defaults.linewidth
# matplotlib.rcParams["font.size"] = plot_defaults.font_size
# matplotlib.rcParams["axes.labelsize"] = plot_defaults.label_size
# matplotlib.rcParams["xtick.labelsize"] = plot_defaults.tick_size
# matplotlib.rcParams["ytick.labelsize"] = plot_defaults.tick_size
# matplotlib.rcParams["axes.linewidth"] = plot_defaults.axis_linewidth
# matplotlib.rcParams["figure.dpi"] = plot_defaults.dpi
