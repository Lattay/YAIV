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

from types import SimpleNamespace
from typing import Sequence, Any

import numpy as np

from yaiv.defaults.config import ureg

__all__ = [
    "invQ",
    "reciprocal_basis",
    "cartesian2cryst",
    "cryst2cartesian",
    "cartesian2voigt",
    "voigt2cartesian",
    "grid_generator",
]


def _check_unit_consistency(quantities: Sequence[Any], names: Sequence[str] = None):
    """
    Ensure that all (non-None) inputs are either unitful (pint.Quantity) or all unitless.

    Parameters
    ----------
    quantities : list | tuple
        List of values to check (e.g., eigenvalues, shift, etc.).
    names : list[str], optional
        Names of variables (for debugging messages).

    Raises
    ------
    TypeError
        If the list contains a mix of unitful and unitless variables.
    """
    has_units = [
        isinstance(x, ureg.Quantity) if x is not None else x for x in quantities
    ]
    S = set(has_units)
    S.discard(None)
    if len(S) != 1:
        if names is not None:
            print("Units check failed for:", names)
        print("Units status:", has_units)
        raise TypeError("Either all or none of the variables must have units.")


def invQ(matrix: np.ndarray | ureg.Quantity) -> np.ndarray | ureg.Quantity:
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
    if isinstance(matrix, ureg.Quantity):
        return np.linalg.inv(matrix.magnitude) * (1 / matrix.units)
    else:
        return np.linalg.inv(matrix)


def reciprocal_basis(lattice: np.ndarray | ureg.Quantity) -> np.ndarray:
    """
    Compute reciprocal lattice vectors (rows) from a direct lattice basis.

    Parameters
    ----------
    lattice : np.ndarray
        Direct lattice vectors in rows, optionally with units as ureg.Quantity.

    Returns
    -------
    K_vec : np.ndarray
        Reciprocal lattice vectors in rows, with units of 2π / [input_units].
    """
    K_vec = (invQ(lattice) * ureg._2pi).transpose()  # reciprocal vectors in rows
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
    else:
        inv = invQ(cryst_basis)
        crystal_coord = cartesian_coord @ inv
        if isinstance(cartesian_coord, ureg.Quantity) and isinstance(
            cryst_basis, ureg.Quantity
        ):
            if not crystal_coord.dimensionless:
                raise TypeError(
                    "Input and basis units are not compatible for coordinate transformation"
                )
            in_units = cartesian_coord.units
            if in_units.dimensionality in [
                ureg.meter.dimensionality,
                ureg.alat.dimensionality,
            ]:
                crystal_coord = crystal_coord * (ureg.crystal)

            elif in_units.dimensionality in [
                1 / ureg.meter.dimensionality,
                1 / ureg.alat.dimensionality,
            ]:
                crystal_coord = crystal_coord * (ureg._2pi / ureg.crystal)
            else:
                raise TypeError(
                    "Input units must have dimensionality of [length] or [1/length]"
                )

    return crystal_coord


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
    else:
        if isinstance(crystal_coord, ureg.Quantity) and isinstance(
            cryst_basis, ureg.Quantity
        ):
            if crystal_coord.dimensionality == ureg.crystal.dimensionality:
                crystal_coord = crystal_coord * (1 / ureg.crystal)
            elif crystal_coord.dimensionality == 1 / ureg.crystal.dimensionality:
                crystal_coord = crystal_coord * (ureg.crystal / ureg._2pi)
            else:
                raise TypeError("Input units are not crystal units.")
        cartesian_coord = crystal_coord @ cryst_basis

    return cartesian_coord


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


def _normal_dist(x, mean, sd, A=1):
    """
    Evaluate a normalized Gaussian (normal) distribution.

    Parameters
    ----------
    x : float or np.ndarray
        Point(s) at which to evaluate the distribution.
    mean : float
        Center (mean) of the Gaussian.
    sd : float
        Standard deviation (width) of the Gaussian.
    A : float, optional
        Amplitude factor. If A=1, the distribution integrates to unity. Default is 1.

    Returns
    -------
    y : float or np.ndarray
        Value(s) of the normalized Gaussian distribution at `x`.

    Notes
    -----
    The Gaussian is defined as:
        A / (σ√(2π)) * exp(-0.5 * ((x - μ) / σ)^2)
    where μ is the mean and σ is the standard deviation.
    """
    y = A / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return y


def _expand_zone_border(
    q_point: ureg.Quantity | np.ndarray,
) -> ureg.Quantity | np.ndarray:
    """
    Expand a q-point by adding periodic equivalents related by reciprocal lattice translations.

    This function generates a set of symmetry-related points lying at the borders of the Brillouin zone,
    by adding and subtracting ±1 in each reciprocal direction. This is useful in phonon or electron band
    structure calculations when points like (0.5, 0, 0) and (-0.5, 0, 0) are physically equivalent
    but not explicitly included in the star of q-points.

    Parameters
    ----------
    q_point : pint.Quantity | np.ndarray
        A 3-element q-point in fractional (crystal) coordinates.

    Returns
    -------
    q_points : np.ndarray | pint.Quantity
        A (3N+1, 3)-shaped array containing the original q-point and its ±1-shifted images
        along the three reciprocal directions. Units are preserved if input had units.

    Raises
    ------
    TypeError
        If `q_point` is a Quantity but not in crystal units (i.e., dimensionless reciprocal).
    """
    # Validate units if pint.Quantity
    if isinstance(q_point, ureg.Quantity):
        if (
            q_point.dimensionality != ureg.crystal.dimensionality
            and q_point.dimensionality != (1 / ureg.crystal).dimensionality
        ):
            raise TypeError(
                "If q_point has units, they must have crystal or 1/crystal dimensionality."
            )
        units = q_point.units
        q_point = q_point.magnitude
    else:
        q_point = np.array(q_point)
        units = 1

    output = [q_point]
    for i in range(3):
        new_points = []
        for point in output:
            point = np.array(point)
            for delta in [-1, 1]:
                shifted = point.copy()
                shifted[i] += delta
                new_points.append(shifted)
        output.extend(new_points)

    output = np.array(output)

    return output * units
