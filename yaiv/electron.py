# PYTHON module with the electron classes for electronic spectrum

import yaiv.utils as ut


class electronBands:
    """Class for handling electronic bandstructures"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filetype = ut.grep_filetype(filepath)
    def grep_total_energy(self, meV=False):
        """Returns the total energy in (Ry). Check grep_total_energy"""
        out = ut.grep_total_energy(self.file, meV=meV, filetype=self.filetype)
        self.total_energy = out
        return out
