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


class file:
    """
    TODO: REMOVE
    A class for file scraping, depending on the filetype a different set of attributes will initialize.

    - QuantumEspresso: qe_scf_in, qe_scf_out, qe_bands_in, qe_ph_out, matdyn_in
    - VASP: POSCAR, OUTCAR, KPATH (KPOINTS in line mode), EIGENVAL
    """
    def __init__(self,file):
        self.file = file
        #Define file type
        self.filetype = grep_filetype(file)
        #Read attributes:
        if self.filetype in ['qe_scf_out','qe_scf_in','qe_bands_in','qe_ph_out','outcar','poscar']:
            self.lattice = grep_lattice(self.file,filetype=self.filetype)
        if self.filetype in ['qe_scf_out','outcar','eigenval']:
            self.electrons = grep_electrons(file,filetype=self.filetype)
            self.fermi = grep_fermi(file,filetype=self.filetype,silent=True)
        if self.filetype == 'kpath':
            self.path,self.labels = grep_ticks_labels_KPATH(file)
        if self.filetype in ['qe_bands_in','matdyn_in']:
            self.path = grep_ticks_QE(self.file,self.filetype)
    def __str__(self):
        return str(self.filetype) + ':\n' + self.file
    def grep_lattice(self,alat=False):
        """Check grep_lattice function"""
        self.lattice = grep_lattice(self.file,filetype=self.filetype,alat=alat)
        return self.lattice
    def reciprocal_lattice(self,alat=False):
        """Check K_basis function"""
        if hasattr(self, 'lattice'):
            return K_basis(self.lattice,alat=alat)
        else:
            print('No lattice data in order to compute reciprocal lattice')
    def grep_ph_grid_points(self,expanded=False,decimals=3):
        """Check grep_ph_grid_points function"""
        if self.filetype != 'qe_ph_out':
            print('This method if for ph.x outputs, which this is not...')
            print('Check the documentation for grep_ph_grid_points function')
        else:
            grid = grep_ph_grid_points(self.file,expanded=expanded,decimals=decimals)
            self.ph_grid_points = grid
            return grid
    def grep_total_energy(self,meV=False):
        """Returns the total energy in (Ry). Check grep_total_energy"""
        out= grep_total_energy(self.file,meV=meV,filetype=self.filetype)
        self.total_energy = out
        return out
    def grep_energy_decomposition(self,meV=False):
        """Greps the total energy decomposition with it's contributions. Check grep_energy_decomposition"""
        F, TS, U, U_one_electron, U_h, U_xc, U_ewald = grep_energy_decomposition(self.file,meV=meV,filetype=self.filetype)
        self.total_energy = F
        self.F = F
        self.TS = TS
        self.U = U
        self.U_one_electron = U_one_electron
        self.U_h = U_h
        self.U_xc = U_xc
        self.U_ewald = U_ewald
    def grep_stress_tensor(self,kbar=True):
        """Returns the total stress tensor in (kbar) or default unit (Ry/bohr**3 for QE and X for VASP)"""
        out=grep_stress_tensor(self.file,kbar=kbar,filetype=self.filetype)
        self.stress=out
        return out
    def grep_kpoints_energies(self):
        """ Greps the Kpoints, energies and weights...
        For more info check grep_kpoints_energies function"""
        out=grep_kpoints_energies(self.file,filetype=self.filetype,vectors=self.grep_lattice())
        self.kpoints_energies=out[0]
        self.kpoints_weights=out[1]
        return out
    def grep_gap(self):
        """Get the direct and indirect gaps
        For more info check grep_gap
        return direct_gap, indirect_gap"""
        out=grep_gap(self.file,filetype=self.filetype)
        self.direct_gap=out[0]
        self.indirect_gap=out[1]
        return out
    def grep_kpoints_energies_projections(filename,filetype,IgnoreWeight=True):
        """
        Grep the kpoints and energies and projections

        returns STATES, KPOINTS, ENERGIES, PROJECTIONS
        For more info check the grep_kpoints_energies_projections function
        """
        out=grep_kpoints_energies_projections(filename,filetype)
        self.states=out[0]
        self.kpoints=out[1]
        self.energies=out[2]
        self.projections=out[3]
        return out
    def grep_DOS(self,fermi='auto',smearing=0.02,window=None,steps=500,precision=3):
        """
        Grep the density of states from a scf or nscf file. 
        For more info check grep_DOS function
        """
        if fermi == 'auto':
            fermi=grep_fermi(self.file,silent=True)
            if fermi==None:
                fermi=0
        out=grep_DOS(self.file,fermi=fermi,smearing=smearing,window=window,
                     steps=steps,precision=precision)
        self.DOS=out
        return out

    def grep_DOS_projected(self,aux_file,fermi='auto',smearing=0.02,window=None,steps=500,precision=3,species=None,atoms=None,l=None,j=None,mj=None,symprec=1e-5,silent=False):
        """
        Grep the projected density of states from a scf or nscf file, together with a proj.pwo or PROCAR file. 
        For more info check grep_DOS_projected
        """
        if self.filetype in ['procar','qe_proj_out']:
            proj_file=self.file
            file = aux_file
        else:
            proj_file=aux_file
            file=self.file
        filetype,proj_filetype=None,None
        if fermi == 'auto':
            fermi=grep_fermi(aux_file,silent=True)
            if fermi==None:
                fermi=0
        out = grep_DOS_projected(file,proj_file,fermi,smearing,window,steps,precision,filetype,
                                 proj_filetype,species,atoms,l,j,mj,symprec,silent)
        self.DOS_projected=out
        return out
    def grep_number_of_bands(self,window=None,fermi=None,filetype=None,silent=True):
        """
        Counts the number of bands in an energy window
        For more info check grep_number_of_bands function
        """
        if fermi==None:
            fermi=self.fermi
        out=grep_number_of_bands(self.file,window,fermi,self.filetype,silent)
        return out
    def grep_frequencies(self,return_star=True,filetype=None):
        """
        Greps the frequencies (in cm-1)  and q-points (QE alat units) from a qe.ph.out file.
        For more info check grep_frequencies function
        """
        out=grep_frequencies(self.file,return_star,self.filetype)
        self.frequencies=out[1]
        self.frequencies_points=out[0]
        return out
    def grep_electron_phonon_nesting(self,return_star=True,filetype=None):
        """
        Greps the nesting, frequencies (in cm-1),lamdas (e-ph coupling), gamma-linewidths (GHz) and q-points (QE alat units) from a qe.ph.out file
        For more info check grep_electron_phonon_nesting function
        """
        out=grep_electron_phonon_nesting(self.file,return_star,self.filetype)
        self.frequencies_points=out[0]
        self.nestings=out[1]
        self.frequencies=out[2]
        self.lambdas=out[3]
        self.gammas=out[4]
        return out

