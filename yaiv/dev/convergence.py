#PYTHON module for cutoff convergence analysis

import glob

class cutoff:
    def __init__(self):
        self.data=None
    def read_data(self, folder:str):
        files = glob.glob(folder+'**/*pwo',recursive=True)
        return files
