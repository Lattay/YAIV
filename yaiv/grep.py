# PYTHON module focussed on text scraping

import numpy as np
import re
from ase import io
from types import SimpleNamespace
import warnings
import yaiv.constants as const
import yaiv.utils as ut


def __filetype(file: str) -> str:
    """
    Detects the filetype of the provided file.

    Currently it supports:
    - QuantumEspresso: qe_scf_in, qe_scf_out, qe_bands_in, qe_ph_out, matdyn_in
    - VASP: POSCAR, OUTCAR, KPATH (KPOINTS in line mode), EIGENVAL

    Parameters
    ----------
    file : str
        Filepath for the file to analyze.

    Returns
    -------
    filetype : str
        Detected filetype (None if not filetype is detected).
    """
    lines = open(file)
    counter = 0
    for line in lines:
        if re.search("calculation.*scf.*", line, re.IGNORECASE) or re.search(
            "calculation.*nscf.*", line, re.IGNORECASE
        ):
            filetype = "qe_scf_in"
            break
        elif re.search("Program PWSCF", line, re.IGNORECASE):
            filetype = "qe_scf_out"
            break
        elif re.search("Program PHONON", line, re.IGNORECASE):
            filetype = "qe_ph_out"
            break
        elif re.search("calculation.*bands.*", line, re.IGNORECASE):
            filetype = "qe_bands_in"
            break
        elif re.search("flfrc", line, re.IGNORECASE):
            filetype = "matdyn_in"
            break
        elif re.search("projwave", line, re.IGNORECASE):
            filetype = "qe_proj_out"
            break
        elif re.search("PROCAR", line, re.IGNORECASE):
            filetype = "procar"
            break
        elif re.search("vasp", line, re.IGNORECASE):
            filetype = "outcar"
            break
        elif len(line.split()) == 4 and all([x.isdigit() for x in line.split()]):
            filetype = "eigenval"
            break
        elif re.search("line.mode", line, re.IGNORECASE):
            filetype = "kpath"
            break
        elif (
            re.search("direct", line, re.IGNORECASE)
            and not re.search("directory", line, re.IGNORECASE)
            or re.search("cartesian", line, re.IGNORECASE)
        ):
            filetype = "poscar"
            break
        else:
            filetype = None
    lines.close()
    return filetype


def electrons(file: str) -> int:
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
    filetype = __filetype(file)
    lines = open(file)
    if filetype == "qe_scf_out":
        for line in lines:
            if re.search("number of electrons", line):
                num_elec = int(float(line.split()[4]))
                break
    elif filetype == "outcar":
        for line in lines:
            if re.search("NELECT", line):
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
    lines.close()
    return num_elec


def lattice(file: str, alat: bool = False) -> np.ndarray:
    """
    Greps the lattice vectors (in Angstroms) from a variety of outputs.

    When possible it uses ASE internally

    Parameters
    ----------
    file : str
        File from which to extract the lattice.
    alat : bool, optional
        Whether the lattice is given in alat units. Default is False.

    Returns
    -------
    lattice : np.ndarray
        np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]])
    """
    filetype = __filetype(file)
    if filetype == "qe_ph_out":
        READ = False
        lines = open(file, "r")
        for line in lines:
            if re.search("lattice parameter", line):
                line = line.split()
                alat_au = float(line[4])
            elif read_vectors:
                values = line.split()
                vec = np.array([float(values[3]), float(values[4]), float(values[5])])
                lattice = np.vstack([lattice, vec]) if "lattice" in locals() else vec
                if lattice.shape == (3, 3):
                    break
            elif re.search("crystal axes", line, flags=re.IGNORECASE):
                READ = True
        lines.close()
        if alat == True:
            return lattice
        else:
            return lattice * alat_au * const.au2ang
    else:
        # Get with ASE
        lattice = io.read(file).cell.T
        if alat == True:
            lattice = lattice / np.linalg.norm(lattice[0])
    return lattice


def fermi(file: str) -> float:
    """
    Greps the Fermi energy from a variety of filetypes and returns it in eV

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
    filetype = __filetype(file)
    lines = open(file)
    if filetype == "qe_scf_out":
        for line in lines:
            # If smearing is used
            if re.search("Fermi energy is", line):
                E_f = float(line.split()[4])
                break
            # If fixed occupations is used
            if re.search("highest occupied", line):
                if re.search("unoccupied", line):
                    split = line.split()
                    E1, E2 = float(split()[6]), float(split()[7])
                    # Fermi level between the unoccupied and occupied bands
                    E_f = E1 + (E2 - E1) / 2
                else:
                    E_f = float(line.split()[4])
                break
    elif filetype == "outcar":
        for line in lines:
            if re.search("E-fermi", line):
                E_f = float(line.split()[2])
                break
    else:
        raise NotImplementedError("Unsupported filetype")
    lines.close()
    if "E_f" not in locals():
        raise NameError("Fermi energy not found.")
    return E_f


def total_energy(
    file: str, meV: bool = False, decomposition: bool = False
) -> float | SimpleNamespace:
    """
    Greps the total free energy or it's decomposition and returns the value in Ry.

    Parameters
    ----------
    file : str
        File from which to extract the energy.
    meV : bool, optional
        Whether the energy is given in meV units. Default is False.
    decomposition : bool, optional
        If True an energy decomposition is returned instead. Default is False.

    Returns
    -------
    energy : float | SimpleNamespace
        If decomposition is False a single float with the free energy is returned.
        If decomposition is True a class with the following attributes is returned:
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
    filetype = __filetype(file)
    lines = open(file)
    if filetype == "qe_scf_out":
        for line in reversed(list(lines)):
            if re.search("!", line):
                l = line.split()
                F = float(l[4])
                break
            elif re.search("smearing contrib", line):
                l = line.split()
                TS = float(l[4])
            elif re.search("internal energy", line):
                l = line.split()
                U = float(l[4])
            elif re.search("one-electron", line):
                l = line.split()
                U_one_electron = float(l[3])
            elif re.search("hartree contribution", line):
                l = line.split()
                U_h = float(l[3])
            elif re.search("xc contribution", line):
                l = line.split()
                U_xc = float(l[3])
            elif re.search("ewald", line):
                l = line.split()
                U_ewald = float(l[3])
        if decomposition and "TS" in locals():
            energy = SimpleNamespace(
                F=F,
                TS=TS,
                U=U,
                U_one_electron=U_one_electron,
                U_h=U_h,
                U_xc=U_xc,
                U_ewald=U_ewald,
            )
        else:
            energy = F
    elif filetype == "outcar":
        for line in reversed(list(lines)):
            if re.search("sigma->", line):
                l = line.split()
                energy = float(l[-1])
                break
        energy = energy * const.eV2Ry
    else:
        raise NotImplementedError("Unsupported filetype")
    lines.close()
    if "energy" not in locals():
        raise NameError("Total energy not found.")
    if meV and isinstance(energy, SimpleNamespace):
        for attr in vars(energy):
            setattr(energy, attr, getattr(energy, attr) * const.Ry2meV)
    elif meV:
        energy *= const.Rym2eV
    return energy


def stress_tensor(file: str) -> np.ndarray:
    """
    Greps the total stress tensor in kbar.

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
    filetype = __filetype(file)
    lines = open(file, "r")
    READ = False
    if filetype == "qe_scf_out":
        for line in lines:
            if READ == True:
                vec = np.array([float(x) for x in line.split()[:3]])
                stress = np.vstack([stress, vec]) if "stress" in locals() else vec
                if stress.shape == (3, 3):
                    break
            elif re.search("total.*stress", line):
                READ = True
        stress = stress * (const.Ry2jul / (const.bohr2metre**3)) * const.pas2bar / 1000
    elif filetype == "outcar":
        for line in lines:
            if re.search("in kB", line):
                l = [float(x) for x in line.split()[2:]]
                voigt = np.array([l[0], l[1], l[2], l[4], l[5], l[3]])
                stress = ut.voigt2cartesian(voigt)
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
