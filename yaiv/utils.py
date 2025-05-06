"""
YAIV | yaiv.utils
=================

This module provides general-purpose utility functions that are used across various classes
and methods in the codebase. They are also intended to be reusable by the user for custom
workflows, especially when combined with the data extraction tools.

See Also
--------
yaiv.grep             : File parsing functions that uses these utilities.
yaiv.spectrum         : Core spectral class storing eigenvalues and k-points.
"""

import numpy as np

from yaiv.defaults.config import ureg


def reciprocal_basis(lattice: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal lattice vectors (rows) from a direct lattice basis.

    Parameters
    ----------
    lattice : np.ndarray
        Direct lattice vectors in rows, optionally with units as pint.Quantity.

    Returns
    -------
    K_vec : np.ndarray
        Reciprocal lattice vectors in rows, with units of 2π / [input_units].
    """
    if isinstance(lattice, ureg.Quantity):
        lat = lattice.magnitude
        units = lattice.units
    else:
        lat = lattice
        units = None

    K_vec = np.linalg.inv(lat).transpose()  # reciprocal vectors in rows
    if units is not None:
        K_vec = K_vec * (ureg._2pi / units)

    return K_vec


def _lenght2crystal(unit) -> ureg.Unit:
    """
    Replace all units of length in a compound Pint unit with 'crystal'.

    Parameters
    ----------
    unit : pint.Unit
        Compound unit expression.

    Returns
    -------
    pint.Unit
        Modified unit with all [length] units replaced by 'crystal'.
    """
    result = 1 * ureg.dimensionless
    for name, exp in unit._units.items():
        base = ureg(name)
        if base.dimensionality == ureg.meter.dimensionality:
            result *= ureg.crystal**exp
        else:
            result *= base**exp
    return result.units


def cartesian2cryst(
    cartesian_coord: np.ndarray, cryst_basis: np.ndarray, list_of_vec=False
) -> np.ndarray:
    """
    Convert from Cartesian to crystal coordinates.

    Parameters
    ----------
    cartesian_coord : np.ndarray
        Vector or matrix in Cartesian coordinates.
    cryst_basis : np.ndarray
        Basis vectors written as rows. May include units.
    list_of_vec : bool, optional
        If True, treat input as a list of vectors.

    Returns
    -------
    crystal_coord : Quantity
        Result in crystal coordinates, possibly with modified units.
    """
    if isinstance(cartesian_coord, ureg.Quantity):
        cartesian = cartesian_coord.magnitude

    if isinstance(cryst_basis, ureg.Quantity):
        basis = cryst_basis.magnitude
        units = cryst_basis.units
        final_units = _lenght2crystal(units)

    inv = np.linalg.inv(basis)

    if np.ndim(cartesian) == 1 or list_of_vec:
        crystal_coord = np.matmul(cartesian, inv)
    else:
        crystal_coord = inv.T @ cartesian @ basis.T

    return crystal_coord * final_units


def cryst2cartesian(
    crystal_coord: np.ndarray, cryst_basis: np.ndarray, list_of_vec: bool = False
) -> np.ndarray:
    """
    Convert from crystal to Cartesian coordinates.

    Parameters
    ----------
        crystal_coord : np.ndarray
            Coordinates or matrix in crystal units.
        cryst_basis : np.ndarray
            Basis vectors written as rows.
        list_of_vec : bool, optional
            If True, treat input as a list of vectors.

    Returns
    -------
        cartesian_coord : ndarray or Quantity
            Cartesian coordinates. Unit preserved if crystal_coord had one.
    """
    if isinstance(crystal_coord, ureg.Quantity):
        cryst = crystal_coord.magnitude
    else:
        cryst = crystal_coord

    if np.ndim(cryst) == 1 or list_of_vec:
        cartesian_coord = np.matmul(cryst, cryst_basis)
    else:
        inv = np.linalg.inv(cryst_basis)
        cartesian_coord = cryst_basis.T @ cryst @ inv.T

    return cartesian_coord
