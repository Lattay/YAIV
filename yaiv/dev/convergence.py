# PYTHON module for cutoff convergence analysis
import glob
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from yaiv.defaults.config import ureg
from yaiv import grep

from yaiv.dev import grep as grepx


class Self_consistent:
    def __init__(self):
        self.data = None

    def read_data(self, folder: str):
        files = glob.glob(folder + "**/*pwo", recursive=True)
        cutoff, smearing, kgrid, time, fermi, ram, forces, energy = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for file in files:
            cutoff.append(grepx.cutoff(file))
            smearing.append(grepx.smearing(file))
            kgrid.append(grepx.k_grid(file))
            time.append(grepx.time(file))
            fermi.append(grep.fermi(file))
            ram.append(grepx.ram(file))
            forces.append(grepx.atomic_forces(file).total)
            energy.append(grep.total_energy(file))
        data = SimpleNamespace(
            cutoff=cutoff,
            smearing=smearing,
            kgrid=kgrid,
            time=time,
            fermi=fermi,
            ram=ram,
            forces=forces,
            energy=energy,
        )
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
        Plot ...

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        x : str
            Attribute to be plotted in the x axis.
            Attribute must be present in `self.data`.
        y : str
            Attribute to be plotted in the y axis.
            Attribute must be present in `self.data`.
        group : str, optional
            Attributed from which your dataset should be grouped.
            Attribute must be present in `self.data`.
        **kwargs : dict
            Additional matplotlib arguments passed to `plot()`.

        Returns
        ----------
        ax : Axes
            The axes with the spectrum plot.
        """
        attributes = self.data.__dict__.keys()
        if x not in attributes:
            raise NameError(f"{x} attribute not present at self.data.\n {attributes}")
        else:
            X = self.data.__getattribute__(x)
            # Case for kgrid in X-axis
            if isinstance(X[0], np.ndarray):
                X = np.asarray([np.prod(grid) for grid in X])
        if y not in attributes:
            raise NameError(f"{y} attribute not present at self.data.\n {attributes}")
        else:
            Y = self.data.__getattribute__(y)
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
    def __init__(self):
        self.data = None
