import numpy as np
import matplotlib.pyplot as plt
import spglib as spg
import re
import os
from copy import deepcopy

from ase.io import read, write
from ase.visualize import view
from ase import Atoms


def write_struc(crystal, file, primitive=True, conventional=False, silent=True):
    """Output a structure in a sensible readable way
    crystal = Either QE/VASP file, spglib or ase object.
    file = File name for your output primitive = bolean (whether you want the primitive cell)
    conventional = bolean (whether you want the conventional cell)
    silent = bolean (whether you want some sort of printed output)
    """
    print(type(crystal))
    if type(crystal) == str:  # We are loading a file
        ASE = read(crystal)
        SPG = ase2spglib(ASE)
    elif type(crystal) == Atoms:  # We have an ase structure
        SPG = ase2spglib(crystal)
    elif type(crystal) == tuple:  # spglib structure
        SPG = crystal
    else:
        print("Cannot print! Don't understand format")
    if primitive == True:
        SPG = spg.find_primitive(SPG)
    elif conventional == True:
        SPG = spg.standardize_cell(SPG)
    ASE = spglib2ase(SPG)
    CELL = np.array(ASE.get_cell())
    POS = np.array(ASE.get_scaled_positions())
    SYM = ASE.get_chemical_symbols()
    # create(positions)
    for i, s in enumerate(SYM):
        new = np.hstack([s, np.array(POS[i], dtype=object)])
        try:
            positions = np.vstack((new, positions))
        except NameError: positions = new
    # print(positions)
    if silent == True:
        np.savetxt(file, CELL, fmt="%14.9f", header="CELL (Anstrom)")
        fmt = "%-2s %14.9f %14.9f %14.9f"
        with open(file, "ab") as f:
            np.savetxt(f, positions, header="\n Atomic Positions (crystal)", fmt=fmt)
    else:
        print(CELL)
        positions[:, 1:] = np.around(
            np.array(positions[:, 1:], dtype=float), decimals=8
        )
        print(positions)


def enantio_test(c1, c2, precision=0.001):
    """
    Fast test to check if the structures are enantiomers, it just inverts the structure and checks whether it can be superimposed over the other...
    c1 = Cryst struct 1 (either ase or spglib)
    c2 = Cryst struct 2 (either ase or spglib)
    precision = maximum distance between alowed between atoms(in crystal units).
    """
    if type(c1) != tuple:
        c1 = cell.ase2spglib(c1)
    if type(c2) != tuple:
        c2 = cell.ase2spglib(c2)
    c1 = spg.find_primitive(c1)
    c2 = spg.find_primitive(c2)
    c1_enant = deepcopy(c1)
    for i, vec in enumerate(c1_enant[1]):
        c1_enant[1][i] = -c1_enant[1][i] + np.array([1, 1, 1])
    # TEST
    count = 0
    for elem in c1_enant[1]:
        for i in c2[1]:
            # print(np.linalg.norm(elem-i))
            if np.linalg.norm(elem - i) < precision:
                # print(elem)
                count = count + 1
    if count == len(c1[1]):
        print("They are enantiomers!!!")
    else:
        print("They are not!")
