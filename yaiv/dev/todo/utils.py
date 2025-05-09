# PYTHON module with nice tu have utilitary tools, mostly focussed on text scraping

import warnings
import numpy as np
import re
from ase import io
from types import SimpleNamespace
from .units import ureg
from yaiv import utils as ut
from yaiv import constants as const
from yaiv import plot as plot
from yaiv import cell_analyzer as cell

# GREPPING utilities----------------------------------------------------------------


def grep_ph_grid_points(file, expanded=False, decimals=3):
    """Greps the grid points from a ph.pwo file, it reads the points and the star of those
    points given in the QE ouput and expresses them in reciprocal space lattice vectors. This grid
    can be further expanded to equivalent points line (0,0,0.5) and (0,0,-0.5).

    file='material.ph.pwo'
    """
    vectors = grep_lattice(file, alat=True, filetype="qe_ph_out")
    text = open(file)
    read_text = False
    for line in text:
        if read_text == True and line.split()[0] == str(i):
            split = line.split()
            point = np.array([float(split[1]), float(split[2]), float(split[3])])
            point = np.matmul(point, vectors.transpose())
            try:
                grid_points = np.vstack([grid_points, point])
            except NameError:
                grid_points = point
            i = i + 1
            if i > num_star:
                read_text = False
        elif re.search("Number of q in the star", line):
            num_star = int(line.split()[7])
            i = 1
            read_text = True
        elif re.search("In addition there is the", line):
            i = 1
            read_text = True
    grid_points = np.around(
        grid_points, decimals=decimals
    )  # Fix for detecting the grid in the paths

    if expanded == True:
        initial_grid = grid_points
        for point in initial_grid:
            expanded_star = __expand_star(point)
            try:
                grid = np.vstack([grid, expanded_star])
            except NameError:
                grid = expanded_star
    else:
        grid = grid_points
    return grid


def __expand_star(q_point):
    """Expands the "star" of each point to equivalent points inside (111) (related by lattice) in the border of the BZ.
    This is usefull when a High Sym point is the (0.5,0,0), this generates (-0.5,0,0), which is not in the star
    since is it the same point.
    """
    output = [q_point]
    for i in range(3):
        for point in output:
            related1 = np.array(point)
            related2 = np.array(point)
            related1[i] = point[i] + 1
            related2[i] = point[i] - 1
            output = np.vstack([output, related1, related2])
    return output


def grep_number_of_bands(file, window=None, fermi=None, filetype=None, silent=True):
    """Counts the number of bands in an energy window for all file types supported by grep_kpoints_energies and .gnu files from QE. (It counts them in the first k-point)

    file = File from in which you want to count the bands.
    window = Window of energies where you want to count the number of bands
    fermi = Fermi energy for applying such shift to energies
    silent = No text output
    filetype =Should be detected automatically, but it supports all file types supported by grep_kpoints_energies and .gnu files from QE

    return bands
    """
    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype = filetype.lower()
    if filetype == "data":
        data = np.loadtxt(fname=file)
        data = data[
            :, :2
        ]  # select the first two columns to process (for wannier tools)
        rows = 1
        position = data[rows, 0]
        while (
            position != 0
        ):  # counter will tell me how many points are in the x axis, number of rows
            rows = rows + 1
            position = data[rows, 0]
        columns = int(2 * data.shape[0] / rows)
        data = np.reshape(data, (rows, columns), order="F")
        final_columns = int(columns / 2 - 1)
        data = np.delete(data, np.s_[0:final_columns], 1)
    else:
        data = grep_kpoints_energies(file, filetype=filetype)[0][:, 2:]

    if fermi == None:
        fermi = grep_fermi(file)
        if fermi == None:
            fermi = 0

    # The actual calculation
    if window == None:
        bands = data.shape[1] - 1
        if silent == False:
            print("the total number of bands is", bands)
    else:
        bands = 0
        first_q = data[0, 1:] - fermi
        for item in first_q:
            if item >= window[0] and item <= window[1]:
                bands = bands + 1
        if silent == False:
            print(
                "The number of bands between",
                str(window[0]) + "eV and",
                str(window[1]) + "eV is",
                bands,
            )
    return bands


def grep_gap(file, filetype=None):
    """Get the direct and indirect gaps
    file = File with the bands
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo, relax.pwo)
               vaps (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)

    return direct_gap, indirect_gap
    """
    if filetype == None:
        filetype = grep_filetype(file)
    F = ut.file(file, filetype)
    KE, W = F.grep_kpoints_energies()
    K, E = KE[:, :3], KE[:, 3:]
    valence = E[:, F.electrons - 1]
    conduction = E[:, F.electrons]
    IND_GAP = np.min(conduction) - np.max(valence)
    DIR_GAP = np.min(conduction - valence)
    if IND_GAP < 0:
        IND_GAP = 0
    if DIR_GAP < 0:
        DIR_GAP = 0
    return DIR_GAP, IND_GAP


def grep_DOS(
    file, fermi=0, smearing=0.02, window=None, steps=500, precision=3, filetype=None
):
    """
    Grep the density of states from a scf or nscf file.

    => returns energies, DOS

    file = File from which to extract the DOS
    fermi = Fermi level to shift accordingly
    smearing = Smearing of your normal distribution around each energy
    window = energy window in which to compute the DOS
    steps = Number of values for which to compute the DOS
    precision = Truncation of your normal distrib (truncated from precision*smearing)
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               vaps (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
    """
    if filetype == None:
        filetype = grep_filetype(file)
    data, weights = grep_kpoints_energies(file, filetype=filetype)
    data = data[:, 3:]  # Remove the Kpoint info
    n_bands = len(data[0])
    for w in weights:
        try:
            W = np.vstack((W, np.ones(n_bands) * w))
        except NameError:
            W = np.ones(n_bands) * w
    s = np.shape(W)[0] * np.shape(W)[1]
    # SORT Energies and Weights
    D = data.reshape((s))
    W = W.reshape((s))
    sort = np.argsort(D)
    D = D[sort] - fermi
    W = W[sort]
    # Compute DOS
    if window == None:
        energies = np.linspace(D[0], D[-1], steps)
    elif type(window) is int or type(window) is float:
        energies = np.linspace(-window, window, steps)
    else:
        energies = np.linspace(window[0], window[1], steps)
    DOS = []
    for E in energies:
        dos = 0
        m = np.argmax(D >= E - precision * smearing)
        M = np.argmin(D <= E + precision * smearing)
        for i, e in enumerate(D[m:M]):
            dos = dos + normal_dist(e, E, smearing) * W[i + m]
        DOS = DOS + [dos]
    return energies, DOS


def grep_DOS_projected(
    file,
    proj_file,
    fermi=0,
    smearing=0.02,
    window=None,
    steps=500,
    precision=3,
    filetype=None,
    proj_filetype=None,
    species=None,
    atoms=None,
    l=None,
    j=None,
    mj=None,
    symprec=1e-5,
    silent=False,
):
    """
    Grep the projected density of states from a scf or nscf file, together with a proj.pwo or PROCAR file.
    If you do not ask for specific projections, it will decompose the DOS into different orbitals of different Wyckoff positions.

    => returns energies, proj_DOS, labels

    file = File from which to extract the DOS
    proj_file = File with the projected bands
    fermi = Fermi level to shift accordingly
    smearing = Smearing of your normal distribution around each energy
    window = energy window in which to compute the DOS
    steps = Number of values for which to compute the DOS
    precision = Truncation of your normal distrib (truncated from precision*smearing)
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               vaps (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
    proj_filetype = qe_proj_out (quantum espresso proj.pwo)
                    procar (VASP PROCAR file)
    species = list of atomic species ['Bi','Se'...]
    atoms = list with atoms index [1,2...]
    l = list of orbital atomic numbers:
        qe: [0, 1, 2]
        vasp: ['s','px','py','dxz']  (as written in POSCAR)
    j = total angular mometum. (qe only)
    mj = m_j state. (qe only)
    symprec = symprec for spglib detection of wyckoff positions
    silent = Booleand controling whether you want text output
    """
    if filetype == None:
        filetype = grep_filetype(file)
    if proj_filetype == None:
        proj_filetype = grep_filetype(proj_file)
    data, weights = grep_kpoints_energies(file, filetype=filetype)
    energies = data[:, 3:]  # Remove the Kpoint info
    STATES, KPOINTS, ENERGIES, PROJS = grep_kpoints_energies_projections(
        proj_file, proj_filetype
    )
    if np.shape(energies) != np.shape(ENERGIES):
        print("Files not compatible")
        quit()
    n_bands = len(ENERGIES[0])
    for w in weights:
        try:
            WEIGHTS = np.vstack((WEIGHTS, np.ones(n_bands) * w))
        except NameError:
            WEIGHTS = np.ones(n_bands) * w
    size = np.shape(WEIGHTS)[0] * np.shape(WEIGHTS)[1]
    # SORT Energies
    D = ENERGIES.reshape((size))
    sort = np.argsort(D)
    D = D[sort] - fermi
    dic = {0: "s", 1: "p", 2: "d", 3: "f"}

    if species == None and atoms == None and l == None and j == None and mj == None:
        if silent == False:
            print("Decomposing per Wyckoff positions")
        projections = []
        species, indep_WP, positions, indices = cell.wyckoff_positions(file, symprec)
        indices = [
            [x + 1 for x in y] for y in indices
        ]  # Add one to all indices (atoms starts in 1)
        if "qe" in proj_filetype:
            for i, wp in enumerate(indep_WP):
                L = []
                for S in STATES:
                    if S[0] in indices[i] and S[3] not in L:
                        L = L + [S[3]]
                        label = species[i] + "(" + wp + "," + dic[S[3]] + ")"
                        projections = projections + [
                            [None, indices[i], S[3], None, None, label]
                        ]
            projections = [[None, None, None, None, None, "Total"]] + projections
        elif proj_filetype == "procar":
            for i, wp in enumerate(indep_WP):
                for L in ["s", "p", "d"]:
                    label = species[i] + "(" + wp + "," + L + ")"
                    projections = projections + [
                        [None, indices[i], L, None, None, label]
                    ]
            projections = [[None, None, None, None, None, "Total"]] + projections
    else:
        projections = [[species, atoms, l, j, mj, "Custom_sum"]]
    DOSs, LABELS = [], []
    for P in projections:
        PROJ_SUM, number = sum_projections(
            STATES,
            PROJS,
            proj_filetype,
            species=P[0],
            atoms=P[1],
            l=P[2],
            j=P[3],
            mj=P[4],
        )  # SUM[K,E] energy level E in kpoint K
        if silent == False:
            print(number, "satates summed")
        W = (
            WEIGHTS * PROJ_SUM
        )  # Kpoints weights * projection for each energy at each k point
        # SORT Weights
        W = W.reshape((size))
        W = W[sort]
        # Compute DOS
        if window == None:
            energies = np.linspace(D[0], D[-1], steps)
        elif type(window) is int or type(window) is float:
            energies = np.linspace(-window, window, steps)
        else:
            energies = np.linspace(window[0], window[1], steps)
        proj_DOS = []
        for E in energies:
            dos = 0
            m = np.argmax(D >= E - precision * smearing)
            M = np.argmin(D <= E + precision * smearing)
            for i, e in enumerate(D[m:M]):
                dos = dos + normal_dist(e, E, smearing) * W[i + m]
            proj_DOS = proj_DOS + [dos]
        DOSs = DOSs + [proj_DOS]
        LABELS = LABELS + [P[5]]
    if len(DOSs) == 1:
        DOSs = DOSs[0]
        LABELS = LABELS[0]
    return energies, np.array(DOSs), LABELS


def grep_electron_phonon_nesting(file, return_star=True, filetype=None):
    """
    Greps the nesting, frequencies (in cm-1),lamdas (e-ph coupling), gamma-linewidths (GHz) and q-points (QE alat units) from a qe.ph.out file
    file = File to read from
    return_star = Boolean controling wether to return only que q point, or the whole star (if possible).
    The filetype should be detected automatically, but it supports:
    qe_ph_out

    return POINTS, NESTING, FREQS, LAMBDAS, GAMMAS
    """

    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype = filetype.lower()
    READ, READING, STAR = False, False, False
    POINTS, NESTING = [], []
    if filetype == "qe_ph_out":
        lines = open(file, "r")
        freqs, lambdas, gammas = [], [], []
        for line in lines:
            if re.search("Diagonalizing", line):
                READ = True
            if STAR == True and READ == False:
                point = np.array([float(x) for x in line.split()[1:]])
                try:
                    if len(point) == 3:
                        star = np.vstack((star, point))
                except NameError:
                    star = point
            if re.search("List of q in the star", line) and return_star == True:
                STAR = True
            if READ == True and re.search("q = \(", line):
                if STAR == False:
                    star = np.array([float(x) for x in line.split()[3:6]])
                STAR = False
            if READ == True and re.search("freq", line):
                freqs = freqs + [float(line.split()[-2])]
            if READ == True and re.search("double delta at Ef", line):
                nest = float(line.split()[-1])
            if READ == True and re.search("lambda", line):
                READING = True
                lam = float(line.split("=")[1].split()[0])
                lambdas = lambdas + [lam]
                gam = float(line.split("=")[2].split()[0])
                gammas = gammas + [gam]
            if READ == True and READING == True and not re.search("lambda", line):
                READ, READING = False, False
                POINTS = POINTS + [star]
                NESTING = NESTING + [nest]
                try:
                    FREQS = np.vstack((FREQS, freqs))
                    LAMBDAS = np.vstack((LAMBDAS, lambdas))
                    GAMMAS = np.vstack((GAMMAS, gammas))
                except NameError:
                    FREQS = np.array(freqs)
                    LAMBDAS = np.array(lambdas)
                    GAMMAS = np.array(gammas)
                del star
                freqs, lambdas, gammas = [], [], []
    else:
        print("FILE NOT SOPPORTED")
    return POINTS, NESTING, FREQS, LAMBDAS, GAMMAS


def grep_kpoints_energies_projections(filename, filetype=None, IgnoreWeight=True):
    """Grep the kpoints and energies and projections, it outputs per rows:

    filename = File with the projected bands
    filetype = qe_proj_out (quantum espresso proj.pwo)
               procar (VASP PROCAR file)
    IgnoreWeight = Boolean controlling whether points with non-zero weight would be ignored

    returns STATES, KPOINTS, ENERGIES, PROJECTIONS

    **********************
    STATES = state over which it projects:
        qe: [# ion, species, wfc, l, j, m_j ]
        vasp: [# ion, species, orbital, σ_j]
            - with orbital being [s,py,pz,px,dxy,dyz,dz2,dxz,dx2-y2]
            - with j being 0(total),1(x),2(y),3(z)
    KPOINTS = list of K points as given by the code.
        KPOINTS[k] = the "k" K-point
    ENERGIES = list of energies for each K point as given by the code.
        ENERGIES[k,e] = the energy level "e" of the kpoint "k"
    PROJECTIONS = List of projections over the different states for each of the energies.
        PROJECTIONS[k,e,s] = Projection of energy level "e" of kpoint "k" over state number "s"
    """
    lines = open(filename, "r")
    if filetype == None:
        filetype = grep_filetype(filename)
    READ_PROJ = False
    READ_STATES = False
    if filetype == "qe_proj_out":
        for i, line in enumerate(lines):
            # BASIC LIMITS OF THE DATA
            if re.search("natomwfc", line):
                num_states = int(line.split()[-1])
            if re.search("nbnd", line):
                num_bands = int(line.split()[-1])
            if re.search("nkstot", line):
                num_points = int(line.split()[-1])
                KPOINTS = np.zeros([num_points, 3])  # KPOINTS
                ENERGIES = np.zeros([num_points, num_bands])  # ENERGIES
                PROJS = np.zeros([num_points, num_bands, num_states])  # PROJECTIONS
                STATES = []  # STATES
                k, e, p = -1, -1, 0  # Counters
            # Scrape STATES info
            if re.search("state #", line):
                READ_STATES = True
            elif READ_STATES == True:
                READ_STATES = False
            if READ_STATES == True:
                l = line.split()
                # atom, species, wfc, l, j, mj
                state = [
                    int(l[4]),
                    l[5][1:],
                    int(l[8]),
                    int(l[9][3:]),
                    float(l[10][2:]),
                    float(l[-1].split("=")[-1][:-1]),
                ]
                STATES = STATES + [state]
            # DATA SCRAPING
            if re.search("k =", line):
                k_point = [float(x) for x in line.split()[2:]]
                k = k + 1
                e = -1
                KPOINTS[k] = k_point
            if re.search("===", line):
                energy = float(line.split()[-3])
                e = e + 1
                ENERGIES[k, e] = energy
            if re.search("\|psi\|", line):
                READ_PROJ = False
            if READ_PROJ == True:
                l = line.split("]")[:-1]
                for x in l:
                    split = x.split("*[#")
                    proj = float(split[0])
                    index = int(split[1]) - 1
                    PROJS[k, e, index] = proj
            if re.search("psi =", line):
                READ_PROJ = True
                l = line.split("=")[1].split("]")[:-1]
                for x in l:
                    split = x.split("*[#")
                    proj = float(split[0])
                    index = int(split[1]) - 1
                    PROJS[k, e, index] = proj
    elif filetype == "procar":
        for i, line in enumerate(lines):
            # BASIC LIMITS OF THE DATA
            if re.search("k-points", line):
                l = line.split()
                num_points = int(l[3])
                num_bands = int(l[7])
                num_ions = int(l[11])
                num_states = (
                    num_ions * 9 * 4
                )  # ions * (s,pz,px,pz...) * (total,σx,σy,σz)
                KPOINTS = np.zeros([num_points, 3])  # KPOINTS
                WEIGHTS = np.zeros(num_points)  # KPOINTS
                ENERGIES = np.zeros([num_points, num_bands])  # ENERGIES
                PROJS = np.zeros([num_points, num_bands, num_states])  # PROJECTIONS
                STATES = []  # STATES
                # Scrape STATES info
                POSCAR = "/".join(filename.split("/")[:-1]) + "/POSCAR"
                species = cell.read(POSCAR).get_chemical_symbols()
                for j in range(4):
                    for num_I, ion in enumerate(species):
                        for orbital in range(9):
                            state = [num_I + 1, ion, orbital, j]
                            STATES = STATES + [state]
                k, e, p = -1, -1, 0  # Counters
                READ_PROJ = True
            # DATA SCRAPING
            elif re.search("k-point ", line):
                line = plot.__insert_space_before_minus(line)
                k_point = [float(x) for x in line.split(":")[1].split()[0:3]]
                weight = float(line.split("=")[1])
                k = k + 1
                e = -1
                KPOINTS[k] = k_point
                WEIGHTS[k] = weight
            elif re.search("energy", line):
                energy = float(line.split()[4])
                e = e + 1
                ENERGIES[k, e] = energy
                p = 0
            elif READ_PROJ == True and not re.search("tot", line) and line.strip():
                projs = [float(x) for x in line.split()[1:-1]]
                n = len(projs)
                PROJS[k, e, p : p + n] = projs
                p = p + n
    else:
        print("File format not suported, check grep_filetype output")
    if IgnoreWeight == False:  # Remove points with non-zero weight
        remove = np.where(WEIGHTS != 0)
        KPOINTS = np.delete(KPOINTS, remove, axis=0)
        ENERGIES = np.delete(ENERGIES, remove, axis=0)
        PROJS = np.delete(PROJS, remove, axis=0)
    return STATES, KPOINTS, ENERGIES, PROJS


def cartesian2spherical(xyz, degrees=False):
    """From cartesian to spherical coord
    mod, theta(z^x), phi(x^y)"""
    ptsnew = np.zeros(3)
    xy = xyz[0] ** 2 + xyz[1] ** 2
    ptsnew[0] = np.sqrt(xy + xyz[2] ** 2)
    ptsnew[1] = np.arctan2(
        np.sqrt(xy), xyz[2]
    )  # for elevation angle defined from Z-axis down
    # ptsnew[1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[2] = np.arctan2(xyz[1], xyz[0])
    if degrees == True:
        ptsnew[1:] = ptsnew[1:] * 180 / np.pi
    return ptsnew


def cartesian2voigt(xyz):
    """From Cartesian to Voigt notation"""
    voigt = np.array([xyz[0, 0], xyz[1, 1], xyz[2, 2], xyz[1, 2], xyz[0, 2], xyz[0, 1]])
    return voigt


def voigt2cartesian(voigt):
    """From Voigt to Cartesian notation"""
    xyz = np.array(
        [
            [voigt[0], voigt[5], voigt[4]],
            [voigt[5], voigt[1], voigt[3]],
            [voigt[4], voigt[3], voigt[2]],
        ]
    )
    return xyz


def spherical2cartesian(coord, degrees=False):
    """From spherical to cartesian coord
    coord = mod, theta(z^x), phi(x^y)"""
    if degrees == True:
        coord[1:] = coord[1:] * 2 * np.pi / 360
    r, t, p = coord
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return np.array([x, y, z])


def cryst2spherical(cryst, cryst_basis, degrees=False):
    """From crystal coord to spherical (usefull for SKEAF)
    mod, theta(z^x), phi(x^y)"""
    xyz = cryst2cartesian(cryst, cryst_basis)
    spherical = cartesian2spherical(xyz, degrees)
    return spherical


def spherical2cryst(coord, cryst_basis, degrees=False):
    """From spherical coord to crystal (usefull for SKEAF)
    coord = mod, theta(z^x), phi(x^y)"""
    xyz = spherical2cartesian(coord, degrees=degrees)
    cryst = cartesian2cryst(xyz, cryst_basis)
    return cryst


def rotation(phi, u, radians=False):
    """
    Rotation matrix for an angle phi in a direction u
    """
    if radians == False:
        phi = (phi / 360) * 2 * np.pi
    u = u / np.linalg.norm(u)
    x, y, z = u
    sin = np.sin(phi)
    cos = np.cos(phi)
    R = np.array(
        [
            [
                cos + (x**2) * (1 - cos),
                x * y * (1 - cos) - z * sin,
                x * z * (1 - cos) + y * sin,
            ],
            [
                y * x * (1 - cos) + z * sin,
                cos + (y**2) * (1 - cos),
                y * z * (1 - cos) - x * sin,
            ],
            [
                z * x * (1 - cos) - y * sin,
                z * y * (1 - cos) + x * sin,
                cos + (z**2) * (1 - cos),
            ],
        ]
    )
    return R


def transform_path(KPATH, lattice1, lattice2, decimals=5, save_as=None):
    """Transforms the KPATH in crystal units to another set of coordinates in order to match paths for structures with different lattices.
    KPATH = File from which to read the Kpath of as given by utils.file(file).path (QE format):
            [kx ky kz num_points
             ... ... .. ... ...]
    lattice1 = original lattice.
    lattice2 = lattice after change of coordinates.
    decimals = Amount of decimals you want.
    save_as = path in which you may want to save the new path.
    """
    # reciprocal vectors in columns
    rec1 = ut.K_basis(lattice1)
    rec2 = ut.K_basis(lattice2)
    path = ut.file(KPATH).path
    kpoints = path[:, :3]
    cart = ut.cryst2cartesian(kpoints, rec1, list_of_vec=True)
    cryst = ut.cartesian2cryst(cart, rec2, list_of_vec=True)
    new_path = np.c_[np.around(cryst, decimals), path[:, 3]]
    if save_as != None:
        np.savetxt(save_as, new_path, fmt="%-8.5f %-8.5f% -8.5f% -8.0f ")
    return new_path


# & Usefull functions----------------------------------------------------------------


def normal_dist(x, mean, sd, A=1):
    """Just a regular normal (gaussian) distribution generator. It integrates to one.
    A = An amplitud term, if A=1 it integrates to unity
    """
    prob_density = A / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


def lorentzian_dist(x, center, hwhm, A=1):
    """Just a regular Lorentzian distribution generator.
    center = Point in which the lorentzian is centered
    hwhm = The half-widh half-maximum
    A = An amplitud term, if A=1 it integrates to unity

    """
    OUT = A * (1 / np.pi) * hwhm / ((x - center) ** 2 + hwhm**2)
    return OUT




# & Postprocessing functions----------------------------------------------------------------


def integrate_DOS(
    data, fermi, filetype=None, shift=None, doping=None, force_positive=False
):
    """Integrates the density of states for different purposes
    data = file from which to read the DOS or DOS itself as given by grep_DOS.
    fermi = Energy up to which the integration is done.
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               outcar (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
               data file (as generated when running the plot.pdos QE routine)
    force_positive = The DOS may be negative with certain smearings, do you want to force it to be positive?

    This method serves a couple of porpuses and depending on the input gives different output. The integration is done with the trapezoidal method.
    JUST INTEGRATION
        returns the integrated DOS (aka number of electrons)
    Looking to shift the fermi level?
        shift = Float with the desired shift in the Fermi level
        returns the necessary doping per cell (negative being electron doping)
    Looking to achieve certain doping?
        doping = Doping per cell (in electron units) (negative being electron doping)
        returns = Stimated Fermi level for that doping
    """
    #    if type(data)==str:
    data = np.loadtxt(data)
    # Force possitive DOS
    if force_positive == True:
        for i, num in enumerate(data[:, 1]):
            if num < 0:
                data[i, 1] = 0

    # Compute number of electrons up to charge neutrality
    for i, num in enumerate(data[:, 0]):
        if num > fermi:
            limit = i
            break
    num_elec = np.trapz(data[:limit, 1], data[:limit, 0])

    # If you are looking for doping in order to achieve certain shift in the Fermi level
    if shift != None:
        for i, num in enumerate(data[:, 0]):
            if num > fermi + shift:
                limit = i
                break
        doping = num_elec - np.trapz(data[:limit, 1], data[:limit, 0])
        return doping
    # If you want to predict the chemical potential shift from certain doping
    if doping != None:
        tota_elec = num_elec - doping
        low = 0
        for limit, dum in enumerate(data[:, 0]):
            elec = np.trapz(data[:limit, 1], data[:limit, 0])
            high = data[limit, 0]
            if elec >= tota_elec:
                break
            else:
                low = high
        return low + (high - low) / 2
    return num_elec


def sum_projections(
    STATES, PROJECTIONS, filetype, species=None, atoms=None, l=None, j=None, mj=None
):
    """
    Sums projections obtained by grep_energies_kpoints_projections:

    STATES = output from grep_energies_kpoints_projections.
        qe: [# ion, species, wfc, l, j, m_j ]
        vasp: [# ion, species, orbital, σ_j]
            - with orbital being [s,py,pz,px,dxy,dyz,dz2,dxz,dx2-y2]
            - with j being 0(total),1(x),2(y),3(z)
    PROJECTIONS = output from grep_energies_kpoints_projections.
        PROJECTIONS[k,e,s] = Projection of energy level "e" of kpoint "k" over state number "s"
    filetype = qe_proj_out (quantum espresso proj.pwo)
               procar (VASP PROCAR file)
    species = list of atomic species ['Bi','Se'...]
    atoms = list with atoms index [1,2...]
    l = list of orbital atomic numbers:
        qe: [0, 1, 2]
        vasp: ['s','px','py','dxz']  (as written in POSCAR)
    j = total angular mometum. (qe only)
    m_j = m_j state. (qe only)

    return SUM, number

    *******************************

    SUM = total sum of the projections
        SUM[k,e] = Sum for energy level "e" of kpoint "k".
    number = number of states that were summed
    """
    indices = []
    if type(species) != list and type(species).__module__ != np.__name__:
        species = [species]
    if type(atoms) != list and type(atoms).__module__ != np.__name__:
        atoms = [atoms]
    if type(l) != list and type(l).__module__ != np.__name__:
        l = [l]
    if type(j) != list and type(j).__module__ != np.__name__:
        j = [j]
    if type(mj) != list and type(mj).__module__ != np.__name__:
        mj = [mj]
    if filetype == "qe_proj_out":
        for i, s in enumerate(STATES):
            if s[1] in species or species[0] == None:
                if s[0] in atoms or atoms[0] == None:
                    if s[3] in l or l[0] == None:
                        if s[4] in j or j[0] == None:
                            if s[5] in mj or mj[0] == None:
                                indices = indices + [i]
    elif filetype == "procar":
        if "p" in l or "d" in l:
            FLATLIST = True
        else:
            FLATLIST = False
        vasp2l = {
            None: None,
            "s": 0,
            "py": 1,
            "pz": 2,
            "px": 3,
            "dxy": 4,
            "dyz": 5,
            "dz2": 6,
            "dxz": 7,
            "dx2-y2": 8,
            "x2-y2": 8,
            "p": [1, 2, 3],
            "d": [4, 5, 6, 7, 8],
        }
        l = [vasp2l[x] for x in l]
        if FLATLIST == True:
            l = [x for xs in l for x in xs]
        for i, s in enumerate(STATES):
            if s[1] in species or species[0] == None:
                if s[0] in atoms or atoms[0] == None:
                    if s[2] in l or l[0] == None:
                        if s[3] == 0:  # not sum for σ_j=(1,2,3)
                            indices = indices + [i]
    for i in indices:
        try:
            OUT = OUT + PROJECTIONS[:, :, i]
        except NameError:
            OUT = PROJECTIONS[:, :, i]
    number = len(indices)

    return OUT, number
