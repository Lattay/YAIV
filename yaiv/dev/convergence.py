"""
YAIV | yaiv.convergence
=======================

Tools to collect, organize, and visualize convergence data from ab initio
calculations. Typical use cases include scanning cutoffs, smearing, and k-point grids,
then plotting how total energy, forces, Fermi level, runtime, and memory usage
evolve with these parameters.

Classes
-------
Self_consistent
    Container and utilities for self‑consistent convergence analysis.
    Provides:
    - read_data(): Recursively scan a folder for outputs and populate convergence data.
    - save_as(): Save the entire Self_consistent object in a .pkl file.
    - from_pkl(): Load a Self_consistent object from a .pkl file.
    - plot(): Plot quantities agains computational parameters for checking convergence.
    - _Analyze:
        Helper for generating multi-panel convergence figures.
        Provides:
        - cutoff(): Generate a multi-panel figure summarizing convergence vs. cutoff.
        - kgrid(): Generate a multi-panel figure summarizing convergence vs. k-grid (and smearing).

Examples
--------
>>> from yaiv.convergence import Self_consistent
>>> analysis = Self_consistent()
>>> analysis.read_data(folder)
>>> analysis.plot("energy", "kgrid", "smearing")
(Figure or the energy evolution respect to the kgrid for different smearings)
>>> analysis.analyze.cutoff
(Multi-panel figure for analyzis of convergence vs cutoff)
>>> analysis.analyze.kgrid
(Multi-panel figure for analyzis of convergence vs kgrid)
"""

import glob
from types import SimpleNamespace
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from yaiv.defaults.config import ureg
from yaiv import grep

from yaiv.dev import grep as grepx

class Phonons:
    """
    TODO...
    """

    def __init__(self):
        self.data = None
