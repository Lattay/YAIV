"""
YAIV | yaiv.convergence
=======================

Tools to collect, organize, and visualize convergence data from ab initio
calculations. Typical use cases include scanning cutoffs, smearing, and k-point grids,
then plotting how total energy, forces, Fermi level, runtime, and memory usage
evolve with these parameters.

Classes
-------
Self_consistent
    Container and utilities for self‑consistent convergence analysis.
    Provides:
    - read_data(): Recursively scan a folder for outputs and populate convergence data.
    - plot(): Plot quantities agains computational parameters for checking convergence.

Examples
--------
>>> from yaiv.convergence import Self_consistent
>>> analysis = Self_consistent()
>>> analysis.read_data(folder)
>>> analysis.plot("energy", "kgrid", "smearing")
(Figure or the energy evolution respect to the kgrid for different smearings)
"""

import glob
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from yaiv.defaults.config import ureg
from yaiv import grep

from yaiv.dev import grep as grepx


class Self_consistent:
    """
    Container and utilities for self‑consistent convergence analysis.

    This class scans a folder of output files (e.g., QE .pwo files), extracts
    the relevant quantities (cutoff, smearing, k‑grid, time, Fermi level,
    memory usage, forces, total energy), stores them in a SimpleNamespace
    with consistent units, and provides simple plotting utilities.

    Attributes
    ----------
    data : SimpleNamespace | None
        Namespace holding arrays for each collected attribute (cutoff, smearing,
        kgrid, time, fermi, ram, forces, energy). Each attribute is either a
        NumPy array or a pint.Quantity array with a common unit.

    Methods
    -------
    read_data(folder)
        Recursively scan `folder` for outputs and populate `self.data`.
    plot(x, y, group=None)
        Plot y(x); optionally group by a third attribute `group`.
    """

    def __init__(self):
        """
        Initialize an empty Self_consistent object for convergence analysis.

        Use `read_data(folder)` to populate `self.data` from output files.
        """
        self.data = None

    def read_data(self, folder: str):
        """
        Recursively read convergence data from a folder of outputs.

        This scans for files matching `**/*pwo` and extracts from each:
          - cutoff (energy cutoff)
          - smearing
          - kgrid (Monkhorst–Pack grid as (Nk1,Nk2,Nk3))
          - energy (total energy)
          - forces (total atomic force norm)
          - fermi (Fermi energy)
          - ram (peak memory usage if available)
          - time (runtime)

        The collected values are converted to consistent units per attribute
        and stored in `self.data` as arrays (or pint.Quantity arrays).

        Parameters
        ----------
        folder : str
            Root directory to search (recursively). For example: "runs/" or "./".

        Notes
        -----
        - This relies on helper parsers in yaiv.grep.
        - Attributes with no data remain None.
        """
        files = glob.glob(folder + "**/*pwo", recursive=True)
        data = SimpleNamespace(
            cutoff=[],
            smearing=[],
            kgrid=[],
            time=[],
            fermi=[],
            ram=[],
            forces=[],
            energy=[],
        )
        for file in files:
            data.cutoff.append(grepx.cutoff(file))
            data.smearing.append(grepx.smearing(file))
            data.kgrid.append(grepx.k_grid(file))
            data.time.append(grepx.time(file))
            data.fermi.append(grep.fermi(file))
            data.ram.append(grepx.ram(file))
            data.forces.append(grepx.atomic_forces(file).total)
            data.energy.append(grep.total_energy(file))
        # Unify units
        for atr in data.__dict__:
            d = data.__getattribute__(atr)
            if len(d) != 0:
                if isinstance(d[0], ureg.Quantity):
                    units = d[0].units
                    new = np.asarray([x.to(units).magnitude for x in d]) * units
                else:
                    new = np.asarray(d)
                data.__setattr__(atr, new)
            else:
                data.__setattr__(atr, None)
        self.data = data

    def plot(
        self,
        x: str,
        y: str,
        group: str = None,
        ax: Axes = None,
        **kwargs,
    ) -> (Axes, Axes):
        """
        Plot y vs x with optional grouping.

        Parameters
        ----------
        x : str
            Name of the attribute in `self.data` for the x‑axis
            If x is "kgrid", the product Nk1*Nk2*Nk3 is plotted.
        y : str
            Name of the attribute in `self.data` for the y‑axis.
        group : str, optional
            Name of an attribute in `self.data` used to group the data into curves
            (e.g., group="smearing"). Each unique group value produces a separate line.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        **kwargs : dict
            Additional keyword arguments passed to `ax.plot()`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plotted data.

        Raises
        ------
        NameError
            If `x`, `y`, or `group` (when provided) do not exist in `self.data`.
        """
        # Get X, Y
        attributes = self.data.__dict__.keys()
        if x not in attributes:
            raise NameError(f"{x} attribute not present at self.data.\n {attributes}")
        else:
            X = self.data.__getattribute__(x)
        if y not in attributes:
            raise NameError(f"{y} attribute not present at self.data.\n {attributes}")
        else:
            Y = self.data.__getattribute__(y)

        # Special case: kgrid on x-axis -> use total number of k-points (product)
        if isinstance(X[0], np.ndarray):
            X = np.asarray([np.prod(grid) for grid in X])

        # Prepare groups
        if group is not None:
            if group not in attributes:
                raise NameError(
                    f"{group} attribute not present at self.data.\n {attributes}"
                )
            else:
                Z = self.data.__getattribute__(group)
            if isinstance(Z, ureg.Quantity):
                groups = np.unique(Z.magnitude, axis=0) * Z.units
            else:
                groups = np.unique(Z, axis=0)

        # Create fig if necessary
        if ax is None:
            fig, ax = plt.subplots()

        # Plot grouped or ungrouped
        if group is not None:
            for g in groups:
                indices = []
                for i, z in enumerate(Z):
                    if np.all(z == g):
                        indices.append(i)
                Ysort = Y[indices][np.argsort(X[indices])]
                Xsort = X[indices][np.argsort(X[indices])]
                ax.plot(Xsort, Ysort, ".-", label=str(g), **kwargs)
        else:
            Ysort = Y[np.argsort(X)]
            Xsort = X[np.argsort(X)]
            ax.plot(X, Y, ".-", **kwargs)

        # Decorations
        if isinstance(X, ureg.Quantity):
            ax.set_xlabel(f"{x} ({X.units})")
        else:
            ax.set_xlabel(f"{x}")
        if isinstance(Y, ureg.Quantity):
            ax.set_ylabel(f"{y} ({Y.units})")
        else:
            ax.set_xlabel(f"{y}")
        if group is not None:
            ax.legend()

        plt.tight_layout()
        return ax


class Phonons:
    """
    TODO...
    """

    def __init__(self):
        self.data = None
