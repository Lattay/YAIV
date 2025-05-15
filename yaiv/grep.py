"""
YAIV | yaiv.grep
================

This module provides text-scraping utilities for extracting (grepping) structural and spectral
information from first-principles calculation outputs. It supports common DFT packages
such as Quantum ESPRESSO and VASP.

The functions in this module perform low-level parsing (i.e., grepping) of data such as:

- Electronic eigenvalues and k-points
- Phonon frequencies and paths
- Lattice vectors and stress tensors
- Number of electrons and total energies
- Fermi level and reciprocal space paths

Supported formats include:
- Quantum ESPRESSO output/input: `pw.x`, `ph.x`, `bands.in`, `projwfc.x`, `matdyn.x`
- VASP output: `OUTCAR`, `EIGENVAL`, `KPOINTS`, `PROCAR`

This module is intended to feed high-level classes like `electronBands` and `phononBands`
by providing clean NumPy arrays or `spectrum` objects with physical units.

Examples
--------
>>> from yaiv.grep import kpointsEnergies
>>> spectrum = kpointsEnergies("OUTCAR")
>>> spectrum.eigenvalues.shape
(100, 32)

>>> from yaiv.grep import lattice
>>> a = lattice("qe.out")
>>> a.shape
(3, 3)

See Also
--------
yaiv.spectrum : Data container and plotter for eigenvalue spectra
yaiv.utils    : Basis universal utilities
"""

from __future__ import annotations
import re
import warnings
from types import SimpleNamespace

import numpy as np
from ase import io

from yaiv.defaults.config import ureg
import yaiv.utils as ut


def _filetype(file: str) -> str:
    """
    Detects the filetype of the provided file.

    Parameters
    ----------
    file : str
        Filepath for the file to analyze.

    Returns
    -------
    filetype : str
        Detected filetype (None if not filetype is detected).
    """
    filetype = None
    with open(file, "r") as lines:
        for line in lines:
            line = line.strip().lower()

            if re.search(r"calculation.*scf|calculation.*nscf", line):
                filetype = "qe_scf_in"
                break
            elif "program pwscf" in line:
                filetype = "qe_scf_out"
                break
            elif "program phonon" in line:
                filetype = "qe_ph_out"
                break
            elif "calculation" in line and "bands" in line:
                filetype = "qe_bands_in"
                break
            elif "flfrc" in line:
                filetype = "matdyn_in"
                break
            elif "&plot nbnd=" in line:
                filetype = "qe_freq_out"
                break
            elif "projwave" in line:
                filetype = "qe_proj_out"
                break
            elif "procar" in line:
                filetype = "procar"
                break
            elif "vasp" in line:
                filetype = "outcar"
                break
            elif len(line.split()) == 4 and all(x.isdigit() for x in line.split()):
                filetype = "eigenval"
                break
            elif "line-mode" in line:
                filetype = "kpath"
                break
            elif ("direct" in line and "directory" not in line) or "cartesian" in line:
                filetype = "poscar"
                break
    return filetype


def electron_num(file: str) -> int:
    """
    Greps the number of electrons.

    It supports different filetypes as Quantum Espresso or VASP outputs.

    Parameters
    ----------
    file : str
        File from which to extract the electron number, it currently supports:
        - QuantumEspresso pw.x output.
        - VASP OUTCAR.

    Returns
    -------
    num_elec : int
        Number of electrons.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The number of electrons was not found in the provided file.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in lines:
                if "number of electrons" in line:
                    num_elec = int(float(line.split()[4]))
                    break
        elif filetype == "outcar":
            for line in lines:
                if "NELECT" in line:
                    num_elec = int(float(line.split()[2]))
                    break
        elif filetype == "eigenval":
            for line in lines:
                if len(line.split()) == 3:
                    num_elec = int(line.split()[0])
                    break
        else:
            raise NotImplementedError("Unsupported filetype")
        if "num_elec" not in locals():
            raise NameError("Number of electrons not found.")
    return num_elec


def lattice(file: str, alat: bool = False) -> np.ndarray:
    """
    Greps the lattice vectors from various outputs.

    Parameters
    ----------
    file : str
        File from which to extract the lattice.
    alat : bool, optional
        Whether to return lattice in internal units (alat). Default is False (return in Å).

    Returns
    -------
    lattice : np.ndarray
        3x3 array of lattice vectors with attached units (ureg.Quantity).
        Units are either Å or 'alat', depending on the `alat` flag.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """
    filetype = _filetype(file)

    if filetype == "qe_ph_out":
        READ = False
        with open(file, "r") as lines:
            for line in lines:
                if "lattice parameter" in line:
                    line = line.split()
                    ALAT = float(line[4]) * ureg.bohr  # lattice parameter in Bohr
                elif re.search("crystal axes", line, flags=re.IGNORECASE):
                    READ = True
                    lattice = []
                elif READ:
                    values = line.split()
                    vec = np.array(
                        [float(values[3]), float(values[4]), float(values[5])]
                    )
                    lattice.append(vec)
                    if len(lattice) == 3:
                        break
        if alat:
            return lattice * ureg("alat")  # lattice in lattice units
        else:
            # Convert alat to Å
            lattice = np.array(lattice) * ALAT.to("angstrom")
            return lattice

    else:
        # Fallback to ASE
        try:
            lattice = io.read(file).cell  # (3, 3) in Å
        except io.formats.UnknownFileTypeError:
            raise NotImplementedError(
                "Unsupported filetype: ASE is not handling it correctly"
            )
        if alat:
            # Normalize to lattice units
            a_norm = np.linalg.norm(lattice[0])
            return (lattice / a_norm) * ureg("alat")
        else:
            return lattice * ureg.angstrom


def fermi(file: str) -> float:
    """
    Greps the Fermi energy from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the Fermi energy.

    Returns
    -------
    E_f : float
        Fermi energy.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The Fermi energy was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in reversed(list(lines)):
                # If smearing is used
                if "Fermi energy is" in line:
                    E_f = float(line.split()[4])
                    break
                # If fixed occupations is used
                if "highest occupied" in line:
                    if "unoccupied" in line:
                        split = line.split()
                        E1, E2 = float(split()[6]), float(split()[7])
                        # Fermi level between the unoccupied and occupied bands
                        E_f = E1 + (E2 - E1) / 2
                    else:
                        E_f = float(line.split()[4])
                    break
        elif filetype == "outcar":
            for line in reversed(list(lines)):
                if "E-fermi" in line:
                    E_f = float(line.split()[2])
                    break
        else:
            raise NotImplementedError("Unsupported filetype")
    if "E_f" not in locals():
        raise NameError("Fermi energy not found.")
    return E_f * ureg("eV")


def total_energy(file: str, decomposition: bool = False) -> float | SimpleNamespace:
    """
    Greps the total free energy or it's decomposition.

    Parameters
    ----------
    file : str
        File from which to extract the energy.
    decomposition : bool, optional
        If True an energy decomposition is returned instead. Default is False.

    Returns
    -------
    energy : float | SimpleNamespace
        If decomposition is False a single float with the free energy is returned.
        If decomposition is True a namespace with the following attributes is returned:
            -  F            -> Total Free energy
            - -TS           -> Smearing contribution
            -  U (= F+TS)   -> Internal energy
                -  U_one_electron
                -  U_hartree
                -  U_exchange-correlational
                -  U_ewald

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The energy was not found in the provided file.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in reversed(list(lines)):
                if "!" in line:
                    F = float(line.split()[4]) * ureg("Ry")
                    break
                elif "smearing contrib" in line:
                    TS = float(line.split()[4]) * ureg("Ry")
                elif "internal energy" in line:
                    U = float(line.split()[4]) * ureg("Ry")
                elif "one-electron" in line:
                    U_one_electron = float(line.split()[3]) * ureg("Ry")
                elif "hartree contribution" in line:
                    U_h = float(line.split()[3]) * ureg("Ry")
                elif "xc contribution" in line:
                    U_xc = float(line.split()[3]) * ureg("Ry")
                elif "ewald" in line:
                    U_ewald = float(line.split()[3]) * ureg("Ry")
            if decomposition and "TS" in locals():
                energy = SimpleNamespace(
                    F=F,
                    TS=TS,
                    U=U,
                    U_one_electron=U_one_electron,
                    U_hartree=U_h,
                    U_xc=U_xc,
                    U_ewald=U_ewald,
                )
            else:
                energy = F
        elif filetype == "outcar":
            for line in reversed(list(lines)):
                if "sigma->" in line:
                    l = line.split()
                    energy = float(l[-1])
                    break
            energy = energy * ureg("eV").to("Ry")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "energy" not in locals():
        raise NameError("Total energy not found.")
    return energy


def stress_tensor(file: str) -> np.ndarray:
    """
    Greps the total stress tensor.

    Parameters
    ----------
    file : str
        File from which to extract the stress tensor.

    Returns
    -------
    stress : np.ndarray
        Stress tensor.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The energy was not found in the provided file.
    """
    filetype = _filetype(file)
    READ = False
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in lines:
                if READ == True:
                    vec = np.array([float(x) for x in line.split()[:3]])
                    stress = np.vstack([stress, vec]) if "stress" in locals() else vec
                    if stress.shape == (3, 3):
                        break
                elif re.search("total.*stress", line):
                    READ = True
            stress = stress * (ureg("Ry") / ureg("bohr") ** 3).to("kbar")
        elif filetype == "outcar":
            for line in lines:
                if "in kB" in line:
                    l = [float(x) for x in line.split()[2:]]
                    voigt = np.array([l[0], l[1], l[2], l[4], l[5], l[3]])
                    stress = ut.voigt2cartesian(voigt) * ureg("kbar")
                    warnings.warn(
                        "According to VASP this is kB units, but when comparing to QE it appears to be GPa.",
                        UserWarning,
                    )
        else:
            raise NotImplementedError("Unsupported filetype")
        lines.close()
    if "stress" not in locals():
        raise NameError("Stress tensor not found.")
    return stress


def kpath(file: str, labels: bool = True) -> SimpleNamespace | np.ndarray:
    """
    Greps the coordinates, labels and number of poiints from the path in reciprocal space.

    Currently supports:
    - QuantumEspresso: qe_bands_in, matdyn_in.
    - VASP: KPATH (KPOINTS in line mode).

    The code expects the labels to be after the high-symmetry points commented with a `!` as:
    ...
    0   0   0   ! Gamma
    0   0.5 0   ! X
    ...

    Parameters
    ----------
    file : str
        File from which to extract the kpath.
    labels : bool, optional
        Whether labels for the high-symmetry points are extracted. Default is True.

    Returns
    -------
    kpath : SimpleNamespace | np.ndarray
        If labels is True, a namespace with attributes `path` and `labels` is returned.
        Otherwise, the kpath is returned as an ndarray.


    Raises
    ------
    NameError:
        If label or kpath is not found.
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """

    def read_qe_path(line_iter):
        kpath = k_names = N = None
        for line in line_iter:
            if N is None:
                N = int(line.split()[0])
            else:
                if labels:
                    try:
                        kpoint, label = line.split("!")
                    except ValueError:
                        raise NameError("Label not found, try using labels=False.")
                else:
                    kpoint = line
                # Grep K point
                kpoint = [float(x) for x in kpoint.split()]
                kpath = np.vstack([kpath, kpoint]) if kpath is not None else [kpoint]
                # Grep K point label
                if labels:
                    new_name = label.split()[0]
                    k_names = (
                        k_names + [new_name] if k_names is not None else [new_name]
                    )
                # Check if path is complete
                if len(kpath) == N:
                    break
        return kpath, k_names

    filetype = _filetype(file)
    READ = EVEN = False
    kpath = k_names = N = None

    with open(file, "r") as lines:
        # QE input format
        if filetype in ["qe_bands_in", "matdyn_in"]:
            line_iter = iter(lines)
            for line in line_iter:
                if re.search("K_POINTS.*crystal_b", line, flags=re.IGNORECASE) or (
                    filetype == "matdyn_in" and re.search("/", line.split()[0])
                ):
                    kpath, k_names = read_qe_path(line_iter)
                    break
        # VASP KPATH format
        elif filetype == "kpath":
            for line in lines:
                # Grep number of points for each subpath
                if N is None:
                    try:
                        N = int(line.split()[0])
                    except ValueError:
                        pass
                elif re.search("Reciprocal", line, flags=re.IGNORECASE):
                    READ = True
                # Read path and labels
                elif READ:
                    if labels:
                        try:
                            kpoint, label = line.split("!")
                        except ValueError:
                            raise NameError("Label not found, try using labels=False.")
                    else:
                        kpoint = line
                    kpoint = [float(x) for x in kpoint.split()]
                    if kpath is None:
                        kpath = np.array([kpoint + [N]])
                        if labels:
                            k_names = [label.split()[0]]
                    # If points is different from previous in the KPATH file
                    elif (kpoint[:3] != kpath[-1][:3]).any():
                        if EVEN:
                            kpath = np.vstack([kpath, kpoint + [1]])
                        else:
                            kpath = np.vstack([kpath, kpoint + [N]])
                        if labels:
                            k_names = k_names + [label.split()[0]]
                    # If point is repeated and odd
                    elif not EVEN:
                        kpath[-1, -1] = N
                    EVEN = EVEN is False
        else:
            raise NotImplementedError("Unsupported filetype")
    if "kpath" is None:
        raise NameError("Kpath not found.")
    kpath = kpath * ureg._2pi / ureg.crystal
    if labels:
        # Post-process labels
        [l.replace("Gamma", r"\Gamma") for l in k_names]
        return SimpleNamespace(path=kpath, labels=k_names)
    else:
        return kpath


def kpointsEnergies(file: str) -> spectrum:
    """
    Grep the kpoints, energies and kpoint-weights for different file kinds.

    Energies are given in eV and kpoints in reciprocal crystal units.
    Currently supports:
    - QuantumEspresso: qe_scf_out.
    - VASP: OUTCAR, EIGENVAL.

    Parameters
    ----------
    file : str
        File from which to extract the spectrum.

    Returns
    -------
    spectrum : spectrum
        Spectrum class with the following attributes:
        - eigenvalues : np.ndarray
            List of energies, each row corresponds to a particular k-point.
        - kpoints : np.ndarray
            List of k-points.  - weights : np.ndarray
        - weights : np.ndarray
            List of kpoint-weights.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """

    from yaiv.spectrum import spectrum

    filetype = _filetype(file)
    READ_energies = READ_kpoints = RELAX_calc = RELAXED = OCCUPATIONS = False
    KPOINTS = ENERGIES = WEIGHTS = E = None
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in lines:
                # Grep number of bands
                if "number of Kohn-Sham" in line:
                    num_bands = int(line.split("=")[1])
                elif "number of k points" in line:
                    num_points = int(line.split()[4])
                elif " cryst. coord." in line:
                    READ_kpoints = True
                elif "force convergence" in line:
                    RELAX_calc = True
                elif "Final scf calculation at the relaxed" in line:
                    RELAXED = True
                elif re.search("End of .* calculation", line):
                    if (RELAX_calc == False) or (
                        RELAX_calc == True and RELAXED == True
                    ):
                        READ_energies = True
                elif READ_kpoints:
                    k = [float(x) for x in line.split("(")[2].split(")")[0].split()]
                    w = float(line.split()[-1])
                    KPOINTS = (
                        np.vstack([KPOINTS, k])
                        if KPOINTS is not None
                        else np.array([k])
                    )
                    WEIGHTS = (
                        np.hstack([WEIGHTS, w])
                        if WEIGHTS is not None
                        else np.array([w])
                    )
                    if len(WEIGHTS) == num_points:
                        READ_kpoints = False
                elif READ_energies:
                    if line.lstrip().startswith("k"):
                        OCCUPATIONS = False
                    elif OCCUPATIONS:
                        pass
                    elif line.strip() != "":
                        e = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                        E = np.hstack([E, e]) if E is not None else e
                        if len(E) == num_bands:
                            ENERGIES = (
                                np.vstack([ENERGIES, E]) if ENERGIES is not None else E
                            )
                            E = None
                            OCCUPATIONS = True
        elif filetype == "eigenval":
            for i, line in enumerate(lines):
                l = line.split()
                if i == 5:
                    num_points, num_bands = int(l[1]), int(l[2])
                    READ_kpoints = READ_energies = True
                elif READ_kpoints:
                    # Kpoint line
                    if len(l) == 4:
                        k = [float(x) for x in l[:3]]
                        w = float(l[-1])
                        KPOINTS = (
                            np.vstack([KPOINTS, k])
                            if KPOINTS is not None
                            else np.array([k])
                        )
                        WEIGHTS = (
                            np.hstack([WEIGHTS, w])
                            if WEIGHTS is not None
                            else np.array([w])
                        )
                    # Energy line
                    elif len(l) == 3:
                        e = float(l[1])
                        E = np.hstack([E, e]) if E is not None else [e]
                        if len(E) == num_bands:
                            ENERGIES = (
                                np.vstack([ENERGIES, E]) if ENERGIES is not None else E
                            )
                            E = None
        elif filetype == "outcar":
            for line in lines:
                if "NBANDS" in line:
                    num_bands = int(line.split()[-1])
                elif "Coordinates" in line and KPOINTS is None:
                    READ_kpoints = True
                elif "band No." in line:
                    READ_energies = True
                elif READ_kpoints:
                    l = line.split()
                    if len(l) != 0:
                        k = [float(x) for x in l[:3]]
                        w = float(l[-1])
                        KPOINTS = (
                            np.vstack([KPOINTS, k])
                            if KPOINTS is not None
                            else np.array([k])
                        )
                        WEIGHTS = (
                            np.hstack([WEIGHTS, w])
                            if WEIGHTS is not None
                            else np.array([w])
                        )
                    else:
                        num_points = len(KPOINTS)
                        READ_kpoints = False
                elif READ_energies:
                    l = line.split()
                    if len(l) == 3:
                        e = float(l[1])
                        E = np.hstack([E, e]) if E is not None else [e]
                        if len(E) == num_bands:
                            ENERGIES = (
                                np.vstack([ENERGIES, E]) if ENERGIES is not None else E
                            )
                            if len(ENERGIES) == num_points:
                                break
                            E = None
        else:
            raise NotImplementedError("Unsupported filetype")
    return spectrum(
        ENERGIES * ureg("eV"), KPOINTS * (ureg._2pi / ureg.crystal), WEIGHTS
    )


def kpointsFrequencies(file: str) -> spectrum:
    """
    Grep the kpoints and frequencies from phonon ouputs.

    Frequencies are given in cm-1 and kpoints in reciprocal alat.
    Currently supports:
    - QuantumEspresso: qe_freq_out.

    Parameters
    ----------
    file : str
        File from which to extract the spectrum.

    Returns
    -------
    spectrum : spectrum
        Spectrum class with the following attributes:
        - eigenvalues : np.ndarray
            List of frequencies, each row corresponds to a particular k-point.
        - kpoints : np.ndarray
            List of k-points.
        - weights : np.ndarray
            List of kpoint-weights.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """

    from yaiv.spectrum import spectrum

    filetype = _filetype(file)
    KPOINTS = FREQS = F = None
    READ_freqs = False
    with open(file, "r") as lines:
        if filetype == "qe_freq_out":
            for line in lines:
                l = line.split()
                if "nbnd" in line:
                    num_bands, num_points = int(l[2][:-1]), int(l[-2])
                elif len(l) == 3:
                    k = [float(x) for x in l]
                    KPOINTS = (
                        np.vstack([KPOINTS, k])
                        if KPOINTS is not None
                        else np.array([k])
                    )
                    READ_freqs = True
                elif READ_freqs:
                    f = [float(x) for x in l]
                    F = np.hstack([F, f]) if F is not None else f
                    if len(F) == num_bands:
                        FREQS = np.vstack([FREQS, F]) if FREQS is not None else F
                        F = None
        else:
            raise NotImplementedError("Unsupported filetype")
    # Give proper units
    FREQS = FREQS * ureg("c") / ureg("cm")
    KPOINTS = KPOINTS * (ureg("_2pi") / ureg("alat"))
    return spectrum(FREQS, KPOINTS, None)
