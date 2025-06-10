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

import pint
import matplotlib
import matplotlib.pyplot as plt

# === Units ===
ureg = pint.UnitRegistry()
ureg.setup_matplotlib(True)
ureg.load_definitions(files("yaiv") / "defaults/extra_units.txt")

# === Misc defaults ===
defaults = SimpleNamespace(symprec=1e-5)  # Default symmetry precision

# === Plotting defaults ===

plot_defaults = SimpleNamespace(
    color_cycle=plt.get_cmap(
        "tab10"
    ).colors,  # Default color cycle for matplotlib plots
    vline_w=0.4,  # Vertical high-symmetry lines width
    vline_c="gray",  # Vertical high-symmetry lines color
    vline_s="--",  # Vertical high-symmetry lines style
    fermi_c="black",  # Fermi line color
    fermi_w=0.4,  # Fermi line width
    valence_c="tab:blue",  # Valence bands color
    conduction_c="tab:red",  # Conduction bands color
    alpha=0.2,  # Transparency
    DOS_c="black",  # Default DOS color for band calculations
    bandsDOS_ratio=0.2,  # Ratio between DOS and bands in spectraDOS plots
    weights_s=40,  # Default size for weight scatter plots
    gradcolor_w=2,  # Default size for color_gradient plot
    #    linewidth=1.5,
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
matplotlib.rcParams['image.cmap'] = 'plasma'
# matplotlib.rcParams["lines.linewidth"] = plot_defaults.linewidth
# matplotlib.rcParams["font.size"] = plot_defaults.font_size
# matplotlib.rcParams["axes.labelsize"] = plot_defaults.label_size
# matplotlib.rcParams["xtick.labelsize"] = plot_defaults.tick_size
# matplotlib.rcParams["ytick.labelsize"] = plot_defaults.tick_size
# matplotlib.rcParams["axes.linewidth"] = plot_defaults.axis_linewidth
# matplotlib.rcParams["figure.dpi"] = plot_defaults.dpi
