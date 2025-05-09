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


def cartesian2cryst(
    cartesian_coord: np.ndarray | ureg.Quantity, cryst_basis: np.ndarray | ureg.Quantity
) -> np.ndarray | ureg.Quantity:
    """
    Convert from Cartesian to crystal coordinates.

    Parameters
    ----------
    cartesian_coord : np.ndarray | ureg.Quantity
        Vector or matrix in Cartesian coordinates. May include units.
    cryst_basis : np.ndarray | ureg.Quantity
        Basis vectors written as rows. May include units.

    Returns
    -------
    crystal_coord : np.ndarray | ureg.Quantity
        Result in crystal coordinates, with modified units if possible.

    Raises
    ------
    TypeError
        If the input units are not compatible with the basis units (i.e., their ratio is not dimensionless).
    """

    if isinstance(cartesian_coord, ureg.Quantity) and not isinstance(
        cryst_basis, ureg.Quantity
    ):
        raise TypeError(
            "Input and basis units are not compatible. Provide both with or without units."
        )
    elif isinstance(cartesian_coord, ureg.Quantity) and isinstance(
        cryst_basis, ureg.Quantity
    ):
        in_units = cartesian_coord.units
        basis_units = cryst_basis.units
        cartesian_coord = cartesian_coord.magnitude
        cryst_basis = cryst_basis.magnitude
        if not (in_units / basis_units).dimensionless:
            raise TypeError(
                "Input and basis units are not compatible for coordinate transformation"
            )

        if in_units.dimensionality == ureg.meter.dimensionality:
            out_units = in_units / basis_units * (ureg.crystal)
        elif in_units.dimensionality == 1 / ureg.meter.dimensionality:
            out_units = in_units / basis_units * (ureg._2pi / ureg.crystal)
        else:
            raise TypeError(
                "Input units must have dimensionality of [length] or [1/length]"
            )
    else:
        out_units = 1

    inv = np.linalg.inv(cryst_basis)
    crystal_coord = cartesian_coord @ inv

    return crystal_coord * out_units


def cryst2cartesian(
    crystal_coord: np.ndarray | ureg.Quantity, cryst_basis: np.ndarray | ureg.Quantity
) -> np.ndarray | ureg.Quantity:
    """
    Convert from crystal to Cartesian coordinates.

    Parameters
    ----------
        crystal_coord : np.ndarray | ureg.Quantity
            Coordinates or matrix in crystal units.
        cryst_basis : np.ndarray | ureg.Quantity
            Basis vectors written as rows.

    Returns
    -------
        cartesian_coord : np.ndarray | ureg.Quantity
            Result in cartesian coordinates, with modified units if possible.

    Raises
    ------
    TypeError
        If the input units are not correct (i.e., not providing crystal units).
    """
    if isinstance(crystal_coord, ureg.Quantity) and not isinstance(
        cryst_basis, ureg.Quantity
    ):
        raise TypeError(
            "Input and basis units are not compatible. Provide both with or without units."
        )
    elif isinstance(crystal_coord, ureg.Quantity) and isinstance(
        cryst_basis, ureg.Quantity
    ):
        in_units = crystal_coord.units
        basis_units = cryst_basis.units
        crystal_coord = crystal_coord.magnitude
        cryst_basis = cryst_basis.magnitude
        if in_units.dimensionality == ureg.crystal.dimensionality:
            out_units = basis_units * in_units * (1 / ureg.crystal)
        elif in_units.dimensionality == 1 / ureg.crystal.dimensionality:
            out_units = basis_units * in_units * (ureg.crystal / ureg._2pi)
        else:
            raise TypeError("Input units are not crystal units.")
    else:
        out_units = 1

    cartesian_coord = crystal_coord @ cryst_basis

    return cartesian_coord * out_units


def cartesian2voigt(xyz: np.ndarray | ureg.Quantity) -> np.ndarray | ureg.Quantity:
    """
    Convert a symmetric 3x3 tensor from Cartesian (matrix) to Voigt notation.

    This is commonly used for stress and strain tensors, where the 3x3 symmetric
    tensor is flattened into a 6-element vector:
        [xx, yy, zz, yz, xz, xy]

    Parameters
    ----------
    xyz : np.ndarray | ureg.Quantity
        A 3x3 symmetric tensor in Cartesian notation. Can optionally carry physical units.

    Returns
    -------
    np.ndarray | ureg.Quantity
        A 1D array of length 6 in Voigt notation. If the input had units, they are preserved.
    """
    voigt = np.array([xyz[0, 0], xyz[1, 1], xyz[2, 2], xyz[1, 2], xyz[0, 2], xyz[0, 1]])
    if isinstance(xyz, ureg.Quantity):
        voigt = voigt * xyz.units
    return voigt


def voigt2cartesian(voigt: np.ndarray | ureg.Quantity) -> np.ndarray | ureg.Quantity:
    """
    Convert a symmetric tensor from Voigt to Cartesian (3x3 matrix) notation.

    This reverses the `cartesian2voigt` operation, converting a 6-element vector into
    a symmetric 3x3 matrix:
        [xx, yy, zz, yz, xz, xy] → [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]

    Parameters
    ----------
    voigt : np.ndarray | ureg.Quantity
        A 1D array of length 6 in Voigt notation. Can optionally carry physical units.

    Returns
    -------
    np.ndarray | ureg.Quantity
        A 3x3 symmetric tensor in Cartesian matrix notation. If the input had units, they are preserved.
    """
    xyz = np.array(
        [
            [voigt[0], voigt[5], voigt[4]],
            [voigt[5], voigt[1], voigt[3]],
            [voigt[4], voigt[3], voigt[2]],
        ]
    )
    if isinstance(voigt, ureg.Quantity):
        xyz = xyz * voigt.units
    return xyz


def grid_generator(grid: list[int], periodic: bool = False) -> np.ndarray:
    """
    Generate a uniform real-space grid of points within [-1, 1]^D or [0, 1)^D,
    where D is the grid dimensionality.

    This function constructs a D-dimensional mesh by specifying the number of
    points along each axis. The resulting points are returned as a (N, D) array,
    where N is the total number of grid points.

    Parameters
    ----------
    grid : list[int]
        List of integers specifying the number of points along each dimension.
        For example, [10, 10, 10] creates a 10×10×10 grid.
    periodic : bool, optional
        If True, the grid will in periodic boundary style. Centered at 0(Γ) with
        values (-0.5,0.5] avoiding duplicate zone borders.
        If False (default), the grid spans from -1 to 1 (inclusive).

    Returns
    -------
    np.ndarray
        Array of shape (N, D), where each row is a point in the D-dimensional grid.
    """
    # Generate the GRID
    DIM = len(grid)
    temp = []
    for g in grid:
        if periodic:
            s = 0
            temp = temp + [np.linspace(s, 1, g, endpoint=False)]
        elif g == 1:
            s = 1
            temp = temp + [np.linspace(s, 1, g)]
        else:
            s = -1
            temp = temp + [np.linspace(s, 1, g)]
    res_to_unpack = np.meshgrid(*temp)
    assert len(res_to_unpack) == DIM

    # Unpack the grid as points
    for x in res_to_unpack:
        c = x.reshape(np.prod(np.shape(x)), 1)
        try:
            coords = np.hstack((coords, c))
        except NameError:
            coords = c
    if periodic == True:
        for c in coords:
            c[c > 0.5] -= 1  # remove 1 to all values above 0.5
    return coords
