# PYTHON module for analyzing charge density waves and Bohr-Oppenheimer energy surfaces

import numpy as np
import matplotlib.pyplot as plt
import glob
from ase import Atoms
import re
import os
import sys
from math import gcd
import spglib as spg

# import yaiv.transformations as trs
import yaiv.utils as ut

# import yaiv.cell_analyzer as cell
# import yaiv.constants as cons

def pp_CDW_sym_analysis(OPs, SGs):
    """It process the output of CDW_sym_analysis returning a reduce list of the distinct possible SpaceGroups, as well
    as a corresponding list with the indices for such items in the OrderParameter list.

    example:
    diff_SGs[i] appears in all OPs[indices[i]]

    return diff_SGs,indices
    """
    diff_SGs = list(set(SGs))
    SGs = np.array(SGs)
    indices = []
    for SG in diff_SGs:
        ind = np.where(SGs == SG)[0]
        indices = indices + [ind]
    return diff_SGs, indices

def poli(x, coef):
    """Generates the y value at the x point for a polinomy defined by certain coeficients
    x = point to evaluate
    coef = coeficients from the highest degree to the lowest (weird)
    """
    y = 0
    for deg in range(coef.shape[0]):
        y = y + coef[deg] * x ** (coef.shape[0] - 1 - deg)
    return y


def plot_energy_landscape(
    data,
    title=None,
    relative=True,
    grid=True,
    color=None,
    prim_axis="ang",
    sec_axis=True,
    axis=None,
    label=None,
    markersize=None,
    save_as=None,
):
    """Plots the energy landscape data

    data = Either the folder containing the energy landscape calculations or the already read data by read_energy_surf_data
    title = 'Your nice and original title for the plot'
    relative = If relative is true then the relative energy respect the undistorted is plotted
    color = string with the color
    prim_axis = 'ang' or 'd' or 'q', depending you want it in angstrom or d 'order parameter' or real q order parameter (with units)
    sec_axis = Whether to add a secondary axis
    grid = (Bolean) It can display an automatic grid in the plot
    axis = Matplotlib axis in which to plot, if no axis is present new figure is created
    label = Label for your plot.
    markersize = Desired markersize
    save_as = 'name.png' or whatever format
    """

    if type(data) == str:
        data = read_energy_surf_data(data, relative=relative)
    (
        lattice,
        atoms,
        positions,
        masses,
        alat,
        boundary,
        supercell,
        OPs,
        energies,
        SGs,
        displacements,
    ) = data
    direction = OPs[0] / boundary[0]
    if prim_axis == "q":
        norm2 = (
            __qe_norm_factor2(displacements[0], masses)
            * (2 * cons.me / cons.u2Kg)
            * (alat**2)
        )  # To Ang * amu^2 units
        N = np.sqrt(norm2)

    if axis == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axis

    # Necessary functions to create a maximum displacement (in Ang) axis
    v = 0
    for i, d in enumerate(displacements):
        v = v + d * direction[i]
    norms = [np.linalg.norm(x) for x in v]
    largest = np.max(norms)
    i = np.where(norms == largest)[0][0]
    species = atoms[i]
    largest = largest * alat * cons.au2ang

    def d2ang(x):
        return x * largest

    def ang2d(x):
        return x / largest

    def q2ang(x):
        return x * largest / N

    def ang2q(x):
        return x * N / largest

    X = np.linspace(boundary[0], boundary[1], num=len(OPs))
    if prim_axis == "ang":
        X = X * largest
    elif prim_axis == "q":
        X = X * N
    else:
        X_plot = X

    ax.plot(X, energies, ".", label=label, color=color, markersize=markersize)

    ax.set_ylabel("energy difference (meV/cell)")
    if prim_axis == "ang":
        ax.set_xlabel(species + " displacement ($\mathrm{\AA}$)")
    elif prim_axis == "q":
        ax.set_xlabel("$q\ \mathrm{(\AA\sqrt{amu})}$")
    else:
        ax.set_xlabel("Order parameter (d)")

    if sec_axis == True:
        if prim_axis == "ang":
            secax = ax.secondary_xaxis("top", functions=(ang2d, d2ang))
            secax.set_xlabel("Order parameter (d)")
        elif prim_axis == "q":
            secax = ax.secondary_xaxis("top", functions=(q2ang, ang2q))
            secax.set_xlabel(species + " displacement ($\mathrm{\AA}$)")
        else:
            secax = ax.secondary_xaxis("top", functions=(d2ang, ang2d))
            secax.set_xlabel(species + " displacement ($\mathrm{\AA}$)")
    if grid == True:
        ax.grid()
    if title != None:  # Title option
        ax.set_title(title)
    plt.tight_layout()
    if save_as != None:  # Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()


def energy_landscape_fit(
    data,
    title=None,
    trim_points=None,
    poli_order="automatic",
    relative=True,
    prim_axis="ang",
    sec_axis=True,
    grid=True,
    axis=None,
    save_as=None,
):
    """Fit a polinomial to your energy landscape

    data = Either the folder containing the energy landscape calculations or the already read data by read_energy_surf_data
    title = 'Your nice and original title for the plot'
    trim_points = Amount of points to be trimmed from the energy landscape data at each of the sides (left,right)
    poli_order = Order of the polinomy (automatically it will select the highest possible up to 20)
    relative = If relative is true then the relative energy respect the undistorted is plotted
    prim_axis = 'ang' or 'd', depending you want it in angstrom or d 'order parameter'
    sec_axis = Whether to add a secondary axis
    grid = (Bolean) It can display an automatic grid in the plot
    axis = Matplotlib axis in which to plot, if no axis is present new figure is created
    save_as = 'name.png' or whatever format
    """

    if type(data) == str:
        data = read_energy_surf_data(data, relative=relative)
    (
        lattice,
        atoms,
        positions,
        masses,
        alat,
        boundary,
        supercell,
        OPs,
        energies,
        SGs,
        displacements,
    ) = data
    direction = OPs[0] / boundary[0]

    if axis == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axis

    # create a maximum displacement (in Ang) axis defining the necesary functions
    v = 0
    for i, d in enumerate(displacements):
        v = v + d * direction[i]
    norms = [np.linalg.norm(x) for x in v]
    largest = np.max(norms)
    i = np.where(norms == largest)[0][0]
    species = atoms[i]
    largest = largest * alat * cons.au2ang

    def d2ang(x):
        return x * largest

    def ang2d(x):
        return x / largest

    X = np.linspace(boundary[0], boundary[1], num=len(OPs))
    if prim_axis == "ang":
        X = X * largest
    # Trim data (remove points that you don't want)
    if trim_points != None:
        s = trim_points[0]
        f = len(X) - trim_points[1]
        X = X[s:f]
        energies = energies[s:f]

    # generate the polinomial (using the points between the fit_lim)
    if poli_order == "automatic":
        poli_order = len(X) - 1
        if poli_order > 20:
            poli_order = 20
    coef = np.polyfit(X, energies, poli_order)
    Y = np.linspace(X[0], X[-1], 1001)
    poli_fit = poli(Y, coef)

    # PLOTTING
    ax.plot(X, energies, ".", label="DFT points")  # scf data
    ax.plot(Y, poli_fit, linewidth=1, label="Poly fit")  # polinomial fit data
    ax.set_ylabel("Energy difference (meV/cell)")
    if prim_axis == "ang":
        ax.set_xlabel(species + " displacement ($\mathrm{\AA}$)")
    else:
        ax.set_xlabel("Order parameter (d)")

    if sec_axis == True:
        if prim_axis == "ang":
            secax = ax.secondary_xaxis("top", functions=(ang2d, d2ang))
            secax.set_xlabel("Order parameter (d)")
        else:
            secax = ax.secondary_xaxis("top", functions=(d2ang, ang2d))
            secax.set_xlabel(species + " displacement ($\mathrm{\AA}$)")

    if len(displacements) == 1:
        frequency = frozen_phonon_freq(
            data, trim_points=trim_points, poli_order=poli_order
        )
        ax.text(
            0.15,
            0.93,
            "Freq = " + str(np.around(frequency, decimals=2)) + " $cm^{-1}$",
            size=9,
            ha="center",
            va="center",
            transform=ax.transAxes,
            horizontalalignment="left",
            bbox=dict(boxstyle="round", ec=(1.0, 0.5, 0.5), fc=(1.0, 0.8, 0.8)),
        )
    ax.legend()

    if grid == True:
        ax.grid()
    if title != None:  # Title option
        ax.set_title(title)
    plt.tight_layout()
    if save_as != None:  # Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()


def frozen_phonon_freq(data, trim_points=None, poli_order="automatic"):
    """Return the frozen phonon frequency in (cm-1)

    data = Either the folder containing the energy landscape calculations or the already read data by read_energy_surf_data
    trim_points = Amount of points to be trimmed from the energy landscape data at each of the sides (left,right)
    poli_order = Order of the polinomy (automatically it will select the highest possible up to 20)
    """

    if type(data) == str:
        data = read_energy_surf_data(data, relative=True)
    (
        lattice,
        atoms,
        positions,
        masses,
        alat,
        boundary,
        supercell,
        OPs,
        energies,
        SGs,
        displacements,
    ) = data
    direction = OPs[0] / boundary[0]
    if len(displacements) > 1:
        print("More than one eigenvector... Thus not a well defined eigenvector")
    norm2 = __qe_norm_factor2(displacements[0], masses) * (
        2 * cons.me / cons.u2Kg
    )  # To alat^2 * amu2 units

    X = np.linspace(boundary[0], boundary[1], num=len(OPs))
    # Trim data (remove points that you don't want)
    if trim_points != None:
        s = trim_points[0]
        f = len(X) - trim_points[1]
        X = X[s:f]
        energies = energies[s:f]

    if poli_order == "automatic":
        poli_order = len(X) - 1
        if poli_order > 20:
            poli_order = 20

    Norm = np.sqrt(norm2)
    X = X * Norm * alat
    energies = energies / (cons.Ry2eV * 1000)  # Pass to Ry
    coef = np.polyfit(X, energies, poli_order)
    quadra = coef[coef.shape[0] - 3]
    if quadra < 0:
        sign = -1
        quadra = -quadra
    else:
        sign = 1
    freq = np.sqrt((quadra * 2 * cons.Ry2jul) / (cons.u2Kg * cons.bohr2metre**2))
    freq = sign * freq * cons.hz2cm / (2 * np.pi)
    return freq
