import re
import warnings
import glob
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from ase import io

from yaiv.defaults.config import ureg
from yaiv import utils as ut
from yaiv import grep
from yaiv.grep import _filetype
from yaiv import phonon as ph


class _Qe_xml(grep._Qe_xml):
    def __init__(self, file):
        grep._Qe_xml.__init__(self, file)

    def cutoff(self) -> ureg.Quantity:
        """
        Greps the cutoff energy.

        Returns
        -------
        cutoff : ureg.Quantity
            Cutoff energy with attached units (ureg.Quantity).
        """
        cutoff = self.root.find(".//ecutwfc").text * ureg.Ry
        return cutoff

    def smearing(self) -> ureg.Quantity:
        """
        Greps the smearing.

        Returns
        -------
        smearing : ureg.Quantity
            Smearing with attached units (ureg.Quantity).
        """
        smearing = float(self.root.find(".//smearing").attrib['degauss']) * ureg.Ry
        return smearing

    def time(self) -> ureg.Quantity:
        """
        Greps the computational time.

        Returns
        -------
        time : ureg.Quantity
            Computational time with attached units (ureg.Quantity).
        """
        lines = self.root.find(".//timing_info")
        time = float(lines.find(".//total").find(".//cpu").text)
        return time * ureg.second

    def k_grid(self) -> ureg.Quantity:
        """
        Greps the k-grid from a variety of filetypes.

        Returns
        -------
        k_grid : list(int)
            K-grid used in the computation.
        """
        lines = self.root.find(".//k_points_IBZ").find(".//monkhorst_pack")
        kgrid = [
            int(lines.attrib["nk1"]),
            int(lines.attrib["nk2"]),
            int(lines.attrib["nk3"]),
        ]
        return kgrid


def cutoff(file: str) -> ureg.Quantity:
    """
    Greps the cutoff energy from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the cutoff energy.

    Returns
    -------
    cutoff : ureg.Quantity
        Cutoff energy with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The cutoff energy was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            cutoff = _Qe_xml(file).cutoff()
        elif filetype == "qe_scf_out":
            for line in lines:
                # If smearing is used
                if "kinetic-energy cutoff" in line:
                    cutoff = float(line.split()[-2])
                    break
            cutoff *= ureg("Ry")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "cutoff" not in locals():
        raise NameError("Cutoff energy not found.")
    return cutoff


def smearing(file: str) -> ureg.Quantity:
    """
    Greps the smearing from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the smearing.

    Returns
    -------
    smearing : ureg.Quantity
        Smearing with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The smearing was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            smearing = _Qe_xml(file).smearing()
        elif filetype == "qe_scf_out":
            for line in lines:
                if "smearing, width" in line:
                    smearing = float(line.split()[-1])
                    break
            smearing *= ureg("Ry")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "smearing" not in locals():
        raise NameError("Smearing energy not found.")
    return smearing


def time(file: str) -> ureg.Quantity:
    """
    Greps the computational time from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the computational time.

    Returns
    -------
    time : ureg.Quantity
        Computational time with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The computational time was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            time = _Qe_xml(file).time()
        elif filetype == "qe_scf_out":
            for line in reversed(list(lines)):
                # If smearing is used
                if "PWSCF" in line:
                    h, m, s = 0, 0, 0
                    time = line.split()[2]
                    if "h" in time:
                        h = int(time.split("h")[0])
                        time = time.split("h")[1]
                    if "m" in time:
                        m = int(time.split("m")[0])
                        time = time.split("m")[1]
                    if "s" in time:
                        s = float(time.split("s")[0])
                    time = s * ureg.second + m * ureg.minute + h * ureg.hour
                    break
        else:
            raise NotImplementedError("Unsupported filetype")
    if "time" not in locals():
        raise NameError("Computation time not found.")
    return time


def ram(file: str) -> ureg.Quantity:
    """
    Greps the RAM needed in the computation from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the RAM.

    Returns
    -------
    RAM : ureg.Quantity
        RAM with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The RAM was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in lines:
                if "total dynamical RAM" in line:
                    RAM = float(line.split()[5])
                    units = ureg(line.split()[6])
                    break
            RAM *= units
        else:
            raise NotImplementedError("Unsupported filetype")
    if "RAM" not in locals():
        raise NameError("RAM not found.")
    return RAM


def k_grid(file: str) -> list[int]:
    """
    Greps the k-grid from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the k-grid.

    Returns
    -------
    k_grid : list(int)
        K-grid used in the computation.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The k-grid was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            kgrid = _Qe_xml(file).k_grid()
        elif filetype == "qe_scf_in":
            READ = False
            for line in lines:
                if "K_POINTS" in line:
                    READ = True
                elif READ:
                    l = line.split()
                    if len(l) >= 3:
                        kgrid = [int(x) for x in l[:3]]
                        break
        elif filetype == "qe_scf_out":
            kgrid = k_grid(file[:-1] + "i")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "kgrid" not in locals():
        raise NameError("K-grid not found.")
    return kgrid


def forces(file: str) -> SimpleNamespace:
    """
    Greps the atomic forces from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the atomic forces.

    Returns
    -------
    forces : SimpleNamespace
        SimpleNamespace class with the following attributes:
        SimpleNamespace with atomic forces with attached units (ureg.Quantity).
        - per_atom : np.ndarray | ureg.Quantity
            List of foces per atom with shape (N,3) where N is the number of atoms.
        - total : np.ndarray | ureg.Quantity
            Total average force.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The atomic forces were not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            READ, WRITE = False, False
            forces = []
            for line in lines:
                if "Forces acting on atoms" in line:
                    READ, WRITE = True, True
                elif READ:
                    l = line.split()
                    if "atom" in line and WRITE:
                        f = [float(x) for x in l[6 : 6 + 3]]
                        forces.append(f)
                    elif "Total force" in line:
                        total_force = float(l[3])
                        break
                    elif len(forces) != 0:
                        WRITE = False
            forces = np.asarray(forces) * ureg("Ry/bohr")
            total_force *= ureg("Ry/bohr")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "forces" not in locals():
        raise NameError("Atomic forces not found.")
    return SimpleNamespace(per_atom=forces, total=total_force)
