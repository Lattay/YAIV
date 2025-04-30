# PYTHON module for storing usefull constants:

# Others
Boltz = 1.380649e-23
me = 9.1093837e-31
pas2bar = 1e-5
Kbar2Gpa = 1 / 10

# Energy
Ry2eV = 13.6056980659
eV2Ry = 1 / Ry2eV
Ry2meV = Ry2eV * 1000
meV2Ry = 1 / Ry2meV

GHz2eV = 4.13566553853598e-06
ev2GHz = 1 / GHz2eV
GHz2meV = 4.13566553853598e-03
meV2GHz = 1 / GHz2meV

hartree2eV = 27.2114
eV2hartree = 1 / hartree2eV
hartree2meV = 27.2114 * 1000
meV2hartree = 1 / hartree2meV

Ry2jul = 2.179872e-18
jul2Ry = 1 / Ry2jul

Ry2cm_QE = 132.064879 / 1.20346372e-03  # fitted to match freqs

hz2cm = 3.33565e-11  # checked
cm2hz = 1 / hz2cm

Ry2K = Ry2jul / Boltz
K2Ry = 1 / Ry2K

eV2cm = 8065.544
cm2eV = 1 / eV2cm
cm2meV = 1000 / eV2cm
meV2cm = 1 / cm2meV

Ry2cm = Ry2eV * eV2cm
cm2Ry = 1 / Ry2cm

jul2eV = jul2Ry * Ry2eV
eV2jul = 1 / jul2eV
jul2meV = jul2Ry * Ry2meV
meV2jul = 1 / jul2meV

# Mass
u2Kg = 1.66054e-27

# Lenght
au2ang = 0.52917721067121
bohr2ang = au2ang
ang2au = 1 / au2ang
ang2bohr = 1 / au2ang

ang2metre = 1e-10
metre2ang = 1 / ang2metre

bohr2metre = bohr2ang * ang2metre
metre2bohr = 1 / bohr2metre
au2metre = bohr2metre
metre2au = metre2bohr


# Functions
def smear2temp(smear):
    """Smearing is expected in Ry"""
    temp = Ry2K * smear
    return temp


def temp2smear(temp):
    """Smearing is returned in Ry"""
    smear = K2Ry * temp
    return smear


def F_Ry2meV(e):
    return e * Ry2eV * 1000


def F_meV2Ry(e):
    return e / (Ry2eV * 1000)
