"""
YAIV | yaiv.utils
=================

This module provides general-purpose utility functions that are used across various classes
and methods in the codebase. They are also intended to be reusable by the user for custom
workflows, especially when combined with the data extraction tools.

Functions
---------
_check_unit_consistency(quantities, names=None)
    Verifies that a list of variables are either all unitful or all unitless. Raises `TypeError` if mixed.

invQ(matrix)
    Returns the inverse of a matrix, preserving units if present.

reciprocal_basis(lattice)
    Computes the reciprocal lattice vectors from a real-space lattice basis.

cartesian2cryst(cartesian_coord, cryst_basis)
    Transforms coordinates from Cartesian to crystal basis, with unit handling.

cryst2cartesian(crystal_coord, cryst_basis)
    Transforms coordinates from crystal to Cartesian basis, with unit handling.

cartesian2voigt(xyz)
    Converts a 3×3 symmetric tensor to 6-element Voigt notation.

voigt2cartesian(voigt)
    Converts a 6-element Voigt vector to a 3×3 symmetric tensor.

grid_generator(grid, periodic=False)
    Generates a uniform D-dimensional grid in either periodic or bounded mode.

methpax_delta(x, mean=0.0, smearing=0.1, order=1, A=1.0)
    Evaluates the Methfessel–Paxton delta approximation up to a given order.

analyze_distribution(x, y)
    Computes the mean, std, skewness, kurtosis and normalization of a distribution defined over `X`.

def kernel_density_on_grid(x,values)
    Compute a kernel-broadened density on a grid from samples located at `x`.

amplitude2order_parameter(amplitudes, masses, displacements)
    Convert displacement amplitudes into proper order parameters with [length × sqrt(mass)] units.

cumulative_integral(X, Y)
    Compute the cumulative integral of a function defined by discrete x and y values.

find_little_group(kpoints,symmetries)
    Compute the little group for each input k-point.

symmetry_orbit_kpoints(kpoints,symmetries)
    Apply all symmetry rotations to a set of k-points (row vectors) and return the unique set as well
    as the little group of the original points.

Private Utilities
-----------------
_normal_dist(x, mean=0, sd=0.1, A=1)
    Computes the value of a normalized Gaussian distribution.

_expand_zone_border(q_point)
    Returns a q-point and its ±1-shifted equivalents along each reciprocal direction.

_point_to_segment_distance(point, endpoint_a, endpoint_b)
    Compute the Euclidean distance from a point to a line segment in 3D.

See Also
--------
yaiv.grep             : File parsing functions that uses these utilities.
yaiv.spectrum         : Core spectral class storing eigenvalues and k-points.
"""

from types import SimpleNamespace
from typing import Sequence, Any

import numpy as np
from scipy.special import factorial, hermite
from scipy import integrate

from yaiv.defaults.config import ureg
from yaiv.defaults.config import defaults

__all__ = [
    "invQ",
    "reciprocal_basis",
    "cartesian2cryst",
    "cryst2cartesian",
    "cartesian2voigt",
    "voigt2cartesian",
    "grid_generator",
    "methpax_delta",
    "analyze_distribution",
    "kernel_density_on_grid",
    "amplitude2order_parameter",
    "cumulative_integral",
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


def reciprocal_basis(lattice: np.ndarray | ureg.Quantity) -> ureg.Quantity:
    """
    Compute reciprocal lattice vectors (rows) from a direct lattice basis.

    Parameters
    ----------
    lattice : np.ndarray
        Direct lattice vectors in rows, optionally with units as ureg.Quantity.

    Returns
    -------
    K_vec : ureg.Quantity
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
    _check_unit_consistency([cartesian_coord, cryst_basis], ["coordinates", "basis"])

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
    _check_unit_consistency([crystal_coord, cryst_basis], ["coordinates", "basis"])

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
        A numpy array with last two indices being a 3x3 symmetric tensor
        in Cartesian notation (a,b,...,3,3). Can optionally carry physical units.

    Returns
    -------
    np.ndarray | ureg.Quantity
        A (a,b,...,6) array in Voigt notation. If the input had units, they are preserved.
    """
    if isinstance(xyz, ureg.Quantity):
        units = xyz.units
        xyz = xyz.magnitude
    else:
        units = 1
    voigt = np.array(
        [
            xyz[..., 0, 0],
            xyz[..., 1, 1],
            xyz[..., 2, 2],
            xyz[..., 1, 2],
            xyz[..., 0, 2],
            xyz[..., 0, 1],
        ]
    )
    # reshape: (a, b, c, d, e) -> result shape: (b, c, d, e, a)
    voigt = np.moveaxis(voigt, (0), (-1))
    return voigt * units


def voigt2cartesian(voigt: np.ndarray | ureg.Quantity) -> np.ndarray | ureg.Quantity:
    """
    Convert a symmetric tensor from Voigt to Cartesian (3x3 matrix) notation.

    This reverses the `cartesian2voigt` operation, converting a 6-element vector into
    a symmetric 3x3 matrix:
        [xx, yy, zz, yz, xz, xy] → [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]

    Parameters
    ----------
    voigt : np.ndarray | ureg.Quantity
        An array with last index being length 6 in Voigt notation (a,b,...,6).
        If the input had units, they are preserved.

    Returns
    -------
    np.ndarray | ureg.Quantity
        A (a,b,...,3,3) symmetric tensor in Cartesian matrix notation.
        If the input had units, they are preserved.
    """
    units = 1
    if isinstance(voigt, ureg.Quantity):
        units = voigt.units
        voigt = voigt.magnitude
    else:
        units = 1
    xyz = np.array(
        [
            [voigt[..., 0], voigt[..., 5], voigt[..., 4]],
            [voigt[..., 5], voigt[..., 1], voigt[..., 3]],
            [voigt[..., 4], voigt[..., 3], voigt[..., 2]],
        ]
    )
    # reshape: (a, b, c, d, e) -> result shape: (c, d, e, a, b)
    xyz = np.moveaxis(xyz, (0, 1), (-2, -1))
    return xyz * units


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


def _normal_dist(x, mean=0, sd=0.1, A=1):
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


def methpax_delta(
    x: float, mean: float = 0.0, smearing: float = 0.1, order: int = 1, A: float = 1.0
) -> float:
    """
    Methfessel-Paxton (MP) approximation to the delta fuction δ(x).

    The MP impulse function generalizes a Gaussian by adding Hermite polynomial
    corrections to improve convergence in electronic structure integrations.

    For order = 0, the function reduces to a normalized Gaussian:
        A / (σ√(2π)) * exp(-0.5 * ((x - μ) / σ)^2)

    Parameters
    ----------
    x : float
        Point(s) at which to evaluate the smearing function.
    mean : float, optional
        Center (mean) of the distribution. Default is 0.
    smearing : float, optional
        Smearing width, directly related to the standard deviation (σ) in the Gaussian case,
        being [smearing = (σ√2)]. Default is 0.1.
    order : int, optional
        Order of the Methfessel-Paxton expansion. Order 0 recovers a Gaussian.
    A : float, optional
        Amplitude factor. If A=1, the result integrates to unity. Default is 1.

    Returns
    -------
    y : float
        Value(s) of the MP-smearing function evaluated at x.
    """

    def A_n(n):
        return (-1) ** n / (factorial(n) * (4**n) * np.sqrt(np.pi))

    x_scaled = (x - mean) / (smearing)

    y = 0
    for n in range(order + 1):
        coeff = A_n(n)
        H = hermite(2 * n)(x_scaled)  # Evaluate the Hermite polynomial
        y += coeff * H * np.exp(-(x_scaled**2))

    normalization = A / smearing  # Restores scale in physical units
    return normalization * y


def analyze_distribution(X, Y):
    """
    Analyze the statistical properties of a function defined over a domain.

    Parameters
    ----------
    X : array_like
        1D array representing the domain (e.g., energy, position).
    Y : array_like
        1D array representing the function values over X (e.g., DOS, intensity).
        Does not need to be normalized; normalization is handled internally.

    Returns
    -------
    stats : SimpleNamespace
        Object containing:
            - mean : float
                First moment (average) of the distribution.
            - variance : float
                Second central moment (spread squared).
            - std : float
                Standard deviation (spread of the function).
            - skewness : float
                Third standardized moment (asymmetry).
                Skewness > 0: tail on the right; < 0: tail on the left.
            - kurtosis : float
                Fourth standardized moment (peakedness, excess).
                Kurtosis > 3: sharper than Gaussian; < 3: flatter.
            - norm : float
                Area under the curve (integral of Y over X).

    Raises
    ------
    ValueError
        If the integral (normalization) of Y is zero. This means the function has
        no area under the curve, and statistical quantities become ill-defined.
    """
    # Ensure arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Compute differential element
    dx = np.gradient(X)

    # Normalization factor
    norm = np.sum(Y * dx)

    # Avoid division by zero
    if norm == 0:
        raise ValueError("Integral (normalization) of Y is zero.")

    # Mean
    mean = np.sum(X * Y * dx) / norm

    # Central moments
    x_shifted = X - mean
    variance = np.sum((x_shifted) ** 2 * Y * dx) / norm
    std = np.sqrt(variance)

    # Higher moments
    skewness = np.sum((x_shifted) ** 3 * Y * dx) / (norm * std**3)
    kurtosis = np.sum((x_shifted) ** 4 * Y * dx) / (norm * std**4)

    return SimpleNamespace(
        mean=mean,
        variance=variance,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        norm=norm,
    )


def kernel_density_on_grid(
    x: np.ndarray | ureg.Quantity,
    values: np.ndarray | ureg.Quantity = None,
    weights: np.ndarray | None = None,
    center: float | ureg.Quantity | None = None,
    x_window: float | list[float] | ureg.Quantity | None = None,
    sigma: float | ureg.Quantity | None = None,
    steps: int | None = None,
    order: int = 0,
    cutoff_sigmas: float = defaults.cutoff_sigmas,
) -> SimpleNamespace:
    """
    Compute a kernel-broadened density on a grid from samples located at `x`.

    This implements a DOS-like convolution:
        density(X) = sum_i values_i * K_sigma(X - x_i) * w_k(i)
    where K is either a Gaussian (order=0) or a Methfessel–Paxton kernel (order>=0).

    Parameters
    ----------
    x : np.ndarray | ureg.Quantity
        Sample locations (e.g., energies). If unitful, all other related inputs
        (center, x_window, sigma) must be compatible with `x` units. Shape (nkpts, nbnds)
    values : np.ndarray | ureg.Quantity, optional
        Amplitudes per sample (e.g., projections). Defaults to ones, producing a DOS.
        Shape (nkpts, nbnds)
    weights : np.ndarray, optional
        k-point weights that sum to 1. If None, uniform weights are used: 1/nkpts.
    center : float | ureg.Quantity, optional
        Center of the window (e.g., Fermi level). Defaults to 0.
    x_window : float | list[float] | ureg.Quantity, optional
        Window for the output grid. If a float, interpreted as symmetric [center - w, center + w].
        If a list/array, interpreted as [xmin, xmax] around `center`.
        If None, inferred from min/max(x) expanded by `sigma` and `cutoff_sigmas`.
    sigma : float | ureg.Quantity, optional
        Kernel width. Defaults to (window_size / 200). (smearing)
    steps : int, optional
        Number of grid points. Defaults to int(4 * (window_size / sigma)), with a minimum of 128.
    order : int, optional
        Order of the Methfessel-Paxton kernel. Default is 0, which recovers a Gaussian kernel.
    cutoff_sigmas : float, optional
        Truncate kernel support to [-cutoff_sigmas * sigma, +cutoff_sigmas * sigma]
        when summing contributions. Default yaiv.defaults.config.defaults.cutoff_sigmas.

    Returns
    -------
    SimpleNamespace
        grid : np.ndarray | ureg.Quantity
            The output grid in the units of `x`.
        density : np.ndarray | ureg.Quantity
            The computed density at each grid point. Units are (values units) / (x units).

    Raises
    ------
    ValueError
        If shapes are inconsistent (e.g., weights length not equal to nk, or values shape
        does not match `x`).
    TypeError
        If unit consistency is violated among unitful inputs.
    """
    # Default values: DOS
    if values is None:
        values = np.ones_like(x)

    # Shapes
    if x.ndim != 2:
        raise ValueError("`x` must be a 2D array of shape (nk, nb).")
    nk, nb = x.shape
    if values.shape != x.shape:
        raise ValueError("`values` must have the same shape as `x` (nk, nb).")
    if weights is None:
        w = np.ones(nk, dtype=float) / nk
    else:
        w = np.asarray(weights, dtype=float)
    if w.shape[0] != nk:
        raise ValueError(
            "`weights` must have shape (nk,) matching the number of k-points."
        )

    # Handle units
    quantities = [x, center, x_window, sigma]
    names = ["x", "center", "x_window", "sigma"]
    _check_unit_consistency(quantities, names)

    # If unitful, convert all to common unit
    if isinstance(x, ureg.Quantity):
        x_units = x.units
        x, center, x_window, sigma = [
            x.to(x_units).magnitude if isinstance(x, ureg.Quantity) else x
            for x in quantities
        ]
    else:
        x_units = 1
    if isinstance(values, ureg.Quantity):
        val_units = values.units
        values = np.asarray(values.magnitude)
    else:
        val_units = 1

    # Determine computing center, window, sigma and steps
    x_min = x.min()
    x_max = x.max()
    center = 0 if center is None else center
    if x_window is None:
        x_min = x.min()
        x_max = x.max()
    elif isinstance(x_window, (float, int)):
        x_min, x_max = np.array([-x_window, x_window]) + center
    else:
        x_min, x_max = np.asarray(x_window) + center
    window_size = x_max - x_min
    if sigma is None:
        sigma = window_size / 200.0
    if steps is None:
        # Ensure at least some reasonable resolution
        steps = max(int(4.0 * (window_size / sigma)), 128)
    # If window was None, expand by cutoff beyond data range
    if x_window is None:
        x_min = x_min - sigma * (cutoff_sigmas + 1)
        x_max = x_max + sigma * (cutoff_sigmas + 1)

    grid = np.linspace(x_min, x_max, steps)

    # Flatten samples and align weights across bands
    x_flat = x.flatten()
    v_flat = values.flatten()
    w_flat = np.repeat(w, nb)

    # Sort by x for efficient windowing with searchsorted
    sort_idx = np.argsort(x_flat)
    x_flat = x_flat[sort_idx]
    v_flat = v_flat[sort_idx]
    w_flat = w_flat[sort_idx]

    density = np.zeros_like(grid, dtype=float)

    # Loop over grid points and accumulate contributions within cutoff window
    for i, X in enumerate(grid):
        left = X - cutoff_sigmas * sigma
        right = X + cutoff_sigmas * sigma
        start = np.searchsorted(x_flat, left, side="left")
        stop = np.searchsorted(x_flat, right, side="right")

        x_loc = x_flat[start:stop]
        v_loc = v_flat[start:stop]
        w_loc = w_flat[start:stop]

        if order == 0:
            K = _normal_dist(x_loc, mean=X, sd=sigma)
        else:
            K = methpax_delta(x_loc, mean=X, smearing=sigma, order=order)

        density[i] = np.sum(K * v_loc * w_loc)

    # Units of the density: (values units) / (x units)
    grid_out = grid * x_units
    density_out = density * val_units / x_units

    return SimpleNamespace(grid=grid_out, density=density_out)


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
        A 3-element q-point in fractional (crystal) coordinates. Or a np.ndarray of shape(N,3) with N
        q-points.

    Returns
    -------
    q_points : np.ndarray | pint.Quantity
        A (27*N, 3)-shaped array containing the original q-point and its ±1-shifted images
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

    if len(q_point.shape) == 1:
        output = np.array([q_point])
    else:
        output = np.array(q_point)
    for i in range(3):
        new_points = []
        for point in output:
            for delta in [-1, 1]:
                shifted = point.copy()
                shifted[i] += delta
                new_points.append(shifted)
        output = np.vstack([output, new_points])

    return output * units


def amplitude2order_parameter(
    amplitudes: ureg.Quantity | list[complex],
    masses: ureg.Quantity | list[float],
    displacements: list[np.ndarray],
) -> ureg.Quantity:
    """
    Convert amplitudes of normalized displacements into proper order parameters with mass scaling.

    Given phonon mode amplitudes applied to mass-normalized displacement vectors, this function
    returns the associated order parameters with physical units of length × sqrt(mass). The relation used is:

        q_s = A × sqrt(Σ_i M_i |ε_i^s|²)

    Parameters
    ----------
    amplitudes : ureg.Quantity or list[complex]
        Scalar amplitudes for each phonon distortion mode (with units of length or dimensionless).

    masses : ureg.Quantity or list[float]
        Atomic masses (usually in 2m_e units), one per atom in the unit cell.

    displacements : list[np.ndarray]
        Normalized displacement vectors for each mode. Each element has shape (N_atoms, 3)
        and is assumed to be mass-normalized (as in QE output).

    Returns
    -------
    ureg.Quantity
        Order parameters with units of length × sqrt(mass), one per mode.
    """
    # Strip units
    units = 1
    if isinstance(amplitudes, ureg.Quantity):
        units *= amplitudes.units
        amplitudes = amplitudes.magnitude
    if isinstance(masses, ureg.Quantity):
        units *= np.sqrt(1 * masses.units)
        masses = masses.magnitude

    # Formats amplitudes and displacements as np.ndarrays
    if isinstance(amplitudes, int | float | complex):
        amplitudes = np.array([amplitudes])
    else:
        amplitudes = np.array(amplitudes)
    if not isinstance(displacements, list):
        displacements = [displacements]

    # QE normalization constant: sum_i M_i * |ε_i|^2
    if len(amplitudes.shape) == 2:
        NN = [0] * amplitudes.shape[1]
    else:
        NN = [0] * len(amplitudes)

    for i in range(len(NN)):
        for j in range(len(masses)):
            NN[i] += masses[j] * (np.linalg.norm(displacements[i][j])) ** 2

    order_parameter = amplitudes * np.sqrt(NN)
    return order_parameter * units


def cumulative_integral(x: np.ndarray, y: np.ndarray):
    """
    Compute the cumulative integral of a function defined by discrete x and y values
    (handles pint units).

    Parameters
    ----------
    x : np.ndarray | ureg.Quantity
        1D array of x-values (must be increasing).
    y : np.ndarray | ureg.Quantity
        1D array of y-values evaluated at the x-points.

    Returns
    -------
    I : ndarray
        1D array of cumulative integral values at each x, starting from zero.

    Notes
    -----
    It uses scipy's cumulative_trapezoid method.
    """
    units = 1
    if isinstance(x, ureg.Quantity):
        units *= x.units
        x = x.magnitude
    if isinstance(y, ureg.Quantity):
        units *= y.units
        y = y.magnitude

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing.")

    Y = np.hstack([0, integrate.cumulative_trapezoid(y, x)])
    return Y * units


def _point_to_segment_distance(
    point: np.ndarray, endpoint_a: np.ndarray, endpoint_b: np.ndarray
) -> float:
    """
    Compute the Euclidean distance from a point to a line segment in 3D.

    The function returns the shortest distance from `point` to the line segment
    defined by the endpoints `endpoint_a` and `endpoint_b`. All inputs must be
    3-element numpy arrays representing Cartesian coordinates.

    Parameters
    ----------
    point : np.ndarray
        Cartesian coordinates of the point (shape: (3,)).
    endpoint_a : np.ndarray
        Cartesian coordinates of the first segment endpoint (shape: (3,)).
    endpoint_b : np.ndarray
        Cartesian coordinates of the second segment endpoint (shape: (3,)).

    Returns
    -------
    float
        The shortest distance from the point to the segment.
    """
    # Tangent vector of the segment, normalized
    segment_vec = endpoint_b - endpoint_a
    segment_dir = segment_vec / np.linalg.norm(segment_vec)

    # Signed projections onto the line extension
    dist_a = np.dot(endpoint_a - point, segment_dir)
    dist_b = np.dot(point - endpoint_b, segment_dir)

    # Clamp to segment endpoints if projection is outside
    parallel_dist = max(dist_a, dist_b, 0)

    # Orthogonal (perpendicular) component
    perpendicular_vec = np.cross(point - endpoint_a, segment_dir)

    return np.hypot(parallel_dist, np.linalg.norm(perpendicular_vec))


def find_little_group(
    kpoints: np.ndarray | ureg.Quantity,
    symmetries: list,
    tol: float = 1e-8,
    mod_G: bool = False,
) -> list[np.ndarray]:
    """
    Compute the little group for each input k-point.

    For each k (row vector), returns the indices i of symmetry operations R_i
    that leave k invariant, using the row-vector convention k' = k @ R.

    Parameters
    ----------
    kpoints : np.ndarray | ureg.Quantity, shape (N, 3)
        Input k-points as row vectors. If a Quantity, units are preserved internally.
        If `mod_G=True`, kpoints must be in crystal units (2π/crystal).
    symmetries : list
        List of symmetry objects, each with attribute `R` (3×3 rotation matrix).
        Translations (if present) are ignored for k.
    tol : float, optional
        Numerical tolerance for invariance checks. Default 1e-8.
    mod_G : bool, optional
        If True, test invariance modulo reciprocal lattice vectors:
        (k @ R) = k (mod 1). If False, test direct equality
        (k @ R − k) = 0 within `tol`. Default is False.

    Returns
    -------
    little_group : list[np.ndarray]
        A list of length N. Each entry is an array of symmetry indices that
        leave the corresponding k-point invariant (under the chosen check).

    Raises
    ------
    ValueError
        If `kpoints` are not shape (N, 3), or if `mod_G=True` and units are not 2π/crystal.

    Notes
    -----
    - Row-vector convention: k' = k @ R.
    - If `mod_G=True`, invariance is tested modulo 1 per component (crystal units).
    """
    # Units handling
    if isinstance(kpoints, ureg.Quantity):
        units = kpoints.units
        kpts = np.asarray(kpoints.magnitude, dtype=float)
    else:
        units = 1
        kpts = np.asarray(kpoints, dtype=float)

    if kpts.ndim == 1:
        kpts = np.asarray([kpoints], dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError("kpoints must be of shape (N, 3)")

    little_group = []

    for k in kpts:
        inv_ops = []
        # Pre-wrap reference if using mod_G
        if mod_G:
            if units != 1 and units != ureg("_2pi/crystal"):
                raise ValueError(
                    "mod_G=True requires kpoints in crystal units (2π/crystal). "
                    "Convert your k-points before calling or set mod_G=False."
                )
            k_wr = k - np.floor(k + 0.5)

        for i, sym in enumerate(symmetries):
            R = np.asarray(sym.R, dtype=float)
            kR = k @ R
            if mod_G:
                kR_wr = kR - np.floor(kR + 0.5)
                d = kR_wr - k_wr
                # modulo-1 closeness: subtract nearest integers
                d = d - np.round(d)
            else:
                d = kR - k
            ok = np.all(np.abs(d) <= tol)

            if ok:
                inv_ops.append(i)
        little_group.append(np.asarray(inv_ops, dtype=int))
    return little_group


def symmetry_orbit_kpoints(
    kpoints: np.ndarray | ureg.Quantity,
    symmetries: list,
    tol: float = 1e-8,
    mod_G: bool = True,
) -> SimpleNamespace:
    """
    Apply all symmetry rotations to a set of k-points (row vectors) and return
    the unique set.

    This treats k-points as row vectors and multiplies on the right by the
    rotation matrices (k' = k @ R). It preserves first occurrence so identity-
    generated points are kept when duplicates occur.

    Parameters
    ----------
    kpoints : np.ndarray | ureg.Quantity, shape (N, 3)
        Input k-points (rows). If a Quantity, units are preserved. For mod_G=True,
        points are assumed in crystal units (2π/crystal).
    symmetries : list
        List of symmetry objects, each with attribute R (3×3 rotation matrix).
        The first element must be the identity symmetry.
    tol : float, optional
        Numerical tolerance used for detecting duplicates (after rounding). Default 1e-8.
    mod_G : bool, optional
        If True (default), identify k ≡ k + G via wrapping to (-0.5, 0.5]:
        k -> k - floor(k + 0.5). This maps -0.5 to +0.5 so boundary points are
        handled consistently.

    Returns
    -------
    orbit : SimpleNamespace
        - kpoints : np.ndarray | pint.Quantity, shape (M, 3)
            Unique symmetry-expanded k-points (within `tol`), same units as input.
        - sym : np.ndarray, shape (M,)
            Symmetry index (into `symmetries`) of the representative kept.
        - origin : np.ndarray, shape (M,)
            Original input k-point index from which the representative was generated.
        - weights : np.ndarray, shape (N,)
            Normalized weights per original k-point, computed as the fraction of orbit
            points whose representative originated from each input k-point (sums to 1).

    Raises
    ------
    ValueError
        If mod_G=True and units are not 2π/crystal.

    Notes
    -----
    - Row-vector convention: k' = k @ R.
    - First occurrence order is preserved when removing duplicates (identity wins).
    - Little group checks use full equivalence: (k @ R) ≈ k, not k + G.
    """

    # Units handling
    if isinstance(kpoints, ureg.Quantity):
        units = kpoints.units
        kpts = np.asarray(kpoints.magnitude, dtype=float)
    else:
        units = 1
        kpts = np.asarray(kpoints, dtype=float)

    if kpts.ndim == 1:
        kpts = np.asarray([kpoints], dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError("kpoints must be of shape (N, 3)")

    # Expand via symmetries (identity first)
    expanded = []
    idx_pairs = []
    for i, sym in enumerate(symmetries):
        for j, k in enumerate(kpts):
            expanded.append(k @ sym.R)  # row-vector convention
            idx_pairs.append([i, j])
    expanded = np.asarray(expanded)  # shape (S*N, 3)

    # Order-preserving uniqueness via rounding to tolerance
    rounded = np.round(expanded / tol) * tol

    # Wrap modulo reciprocal lattice vectors: (-0.5, 0.5]
    if mod_G:
        if units != 1 and units != ureg("_2pi/crystal"):
            raise ValueError(
                "mod_G=True requires kpoints in crystal units (2π/crystal). "
                "Convert your k-points before calling or set mod_G=False."
            )
        rounded = rounded - np.floor(rounded + 0.5)

    seen = set()
    keep_idx = []
    for idx, row in enumerate(rounded):
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            keep_idx.append(idx)

    unique_kpts = expanded[keep_idx]
    idx_map = np.array([idx_pairs[i] for i in keep_idx], dtype=int)

    # Compute origin weights: count how many orbit points came from each original k-point
    N_orig = kpts.shape[0]
    origin_counts = np.bincount(idx_map[:, 1], minlength=N_orig)
    origin_weights = origin_counts / origin_counts.sum()

    return SimpleNamespace(
        kpoints=unique_kpts * units,
        sym=idx_map[:, 0],
        origin=idx_map[:, 1],
        weights=origin_weights,
    )
