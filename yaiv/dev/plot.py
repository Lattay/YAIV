from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from yaiv.defaults.config import ureg
from yaiv.defaults.config import plot_defaults as pdft
from yaiv import utils as ut
from yaiv import spectrum as spec


def miller_plane(miller, lattice, axis, label=None, size=0.15, alpha=0.4):
    """
    Displayes the desired Miller Plane.

    miller = Miller plane indices.
    lattice = Real space lattice.
    axis = 3D axis in which you want your Miller plane to be displayed
    label = Desired label for the Miller plane
    size = size of the miller_plane
    alpha = Transparency of the miller plane
    """
    ax = axis
    lim1 = -size
    lim2 = size
    reciprocal = ut.K_basis(lattice)
    miller = ut.cryst2cartesian(miller, reciprocal)
    if miller[0] != 0:
        y = np.linspace(lim1, lim2, 10)
        z = np.linspace(lim1, lim2, 10)
        y, z = np.meshgrid(y, z)
        x = -(miller[1] * y + miller[2] * z) / miller[0]
    elif miller[1] != 0:
        x = np.linspace(lim1, lim2, 10)
        z = np.linspace(lim1, lim2, 10)
        x, z = np.meshgrid(x, z)
        y = -(miller[0] * x + miller[2] * z) / miller[1]
    elif miller[2] != 0:
        x = np.linspace(lim1, lim2, 10)
        y = np.linspace(lim1, lim2, 10)
        x, y = np.meshgrid(x, y)
        z = -(miller[0] * x + miller[1] * y) / miller[2]
    ax.plot_surface(x, y, z, alpha=alpha, color="pink")
    if label != None:
        ax.plot([0, 0], [0, 0], [0, 0], label=label, color="pink")
