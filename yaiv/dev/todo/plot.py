#PYTHON module for ploting 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import glob
import spglib as spg

import yaiv.utils as ut
import yaiv.constants as cons
import yaiv.cell_analyzer as cell
import yaiv.defaults as defaults

# PLOTTING BANDS----------------------------------------------------------------

def __ticks_generator(vectors,ticks,grid=None):
    """From the real vectors, the High Sym points for the path in crystal reciprocal space units and the 
    grid, it generates the positions for the hight sym points (tick_pos) and grid (grid_pos):

    vectors=[[a1,a2,a3],...,[c1,c2,c3]]
    ticks=[[tick1x,tick1y,tick1z,100],...,[ticknx,tickny,ticknz,1]]
    grid=[[grid1x,grid1y,grid1z],...,[gridnx,gridny,gridnz]]

    returns either tick_pos or tick_pos, grid_pos
    """
    K_vec=ut.K_basis(vectors)
    path=0
    tick0=ticks[0][:3]
    ticks_pos=np.array(0)
    for i in range(1,ticks.shape[0]):
        tick1=ticks[i][:3]
        if  ticks[i-1][3]==1:
            tick0=tick1
        else:
            if np.any(grid!= None):
                for point in grid:
                    dist=__lineseg_dist(point,tick0,tick1)
                    if np.around(dist,decimals=3)==0:
                        if np.around(np.linalg.norm(point-tick0),decimals=3)==0:
                            delta=0
                        elif np.around(np.linalg.norm(point-tick1),decimals=3)==0:
                            vector=(tick1-tick0)
                            delta=np.linalg.norm(ut.cryst2cartesian(vector,K_vec))
                        else:
                            vector=(point-tick0)
                            delta=np.linalg.norm(ut.cryst2cartesian(vector,K_vec))
                        try:
                            if np.all(grid_ticks!=(path+delta)):  #grid_ticks are not degenerate
                                grid_ticks=np.append(grid_ticks,path+delta)
                        except NameError:
                            grid_ticks=np.array(path+delta)
            vector=(tick1-tick0)
            delta=ut.cryst2cartesian(vector,K_vec)
            path=path+np.linalg.norm(delta)
            ticks_pos=np.append(ticks_pos,path)
            tick0=tick1
    if np.any(grid!= None):
        return ticks_pos, grid_ticks
    else:
        return ticks_pos



def DOS_projected(file,proj_file,fermi='auto',smearing=0.02,window=[-5,5],steps=500,precision=3,filetype=None,proj_filetype=None,
                  species=None,atoms=None,l=None,j=None,mj=None,title=None,figsize=None,reverse=False,legend=None,color='black',
                  save_as=None,axis=None,silent=False,fill=True,alpha=0.5,linewidth=1.0,symprec=1e-5):
    """
    Plots the projected Density Of States

    file = File from which to extract the DOS (scf, nscf, bands)
    proj_file = File with the projected bands or output from grep_DOS_projected:
                    qe_proj_out (quantum espresso proj.pwo)
                    procar (VASP PROCAR file)
    fermi = Fermi level to shift accordingly
    smearing = Smearing of your normal distribution around each energy
    window = energy window in which to compute the DOS
    steps = Number of values for which to compute the DOS
    precision = Truncation of your normal distrib (truncated from precision*smearing)
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               vaps (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
    species = list of atomic species ['Bi','Se'...]
    atoms = list with atoms index [1,2...]
    l = list of orbital atomic numbers:
        qe: [0, 1, 2]
        vasp: ['s','px','py','dxz']  (as written in POSCAR)
    j = total angular mometum. (qe only)
    mj = m_j state. (qe only)
    title = 'Your nice and original title for the plot'
    figsize = (int,int) => Size and shape of the figure
    reverse = Bolean switching the DOS and energies axis
    legend = label for the plot
    color = matplotlib color for the line, or list of colors for different lines
    save_as = 'wathever.format'
    axis = ax in which to plot, if no axis is present new figure is created
    silent = Boolean controling whether you want text output
    fill = Boolean controling whether you want to fill the DOS
    alpha = Opaciety of the fill
    linewidht = linewidth of the lines
    symprec = symprec for spglib detection of wyckoff positions
    """
    if filetype == None:
        file = ut.file(file)
    else:
        file = ut.file(file,filetype)
    if fermi == 'auto':
        fermi=ut.grep_fermi(file.file,silent=True)
        if fermi == None:
            fermi=0
    if type(window) is int or type(window) is float:
        window=[-window,window]
    if type(proj_file)!=str:
        E,DOSs,LABELS = proj_file
    else:
        E,DOSs,LABELS = file.grep_DOS_projected(proj_file,fermi=fermi,smearing=smearing,window=window,steps=steps,
                                            precision=precision,species=species,atoms=atoms,l=l,j=j,mj=mj,symprec=symprec,silent=silent)
    if type(LABELS)!= list:
        DOSs=[DOSs]
        LABELS=[legend]
    if type(color)!= list:
        color = [color]+10*list(mcolors.TABLEAU_COLORS.values())
    
    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis
    if reverse==False:
        if len(LABELS)==1 and fill==True:
            ax.plot(E,DOSs[0],'-',label=LABELS[0],color=color[0],linewidth=linewidth)
            ax.fill_between(E,DOSs[0],color=color[0],alpha=alpha)
        else:
            ax.plot(E,DOSs[0],'-',label=LABELS[0],color=color[0],linewidth=linewidth)
        for i,L in enumerate(LABELS[1:]):
            if fill==True:
                ax.plot(E,DOSs[i+1],'-',color=color[i+1],linewidth=linewidth)
                ax.fill_between(E,DOSs[i+1],'-',color=color[i+1],label=L,alpha=alpha)
            else:
                ax.plot(E,DOSs[i+1],'-',color=color[i+1],label=L)
        ax.set_xlim(E[0],E[-1])
        ax.set_yticks([])
        ax.set_xlabel('energy (eV)')
        ax.set_ylabel('DOS (a.u)')
        ax.set_ylim(0,np.max(DOSs[0])*1.1)
        if fermi!=None:                       #Fermi energy
            ax.axvline(x=0,color='black',linewidth=0.4)
    else:
        if len(LABELS)==1 and fill==True:
            ax.plot(DOSs[0],E,'-',label=LABELS[0],color=color[0],linewidth=linewidth)
            ax.fill_betweenx(DOSs[0],E,color=color[0],alpha=alpha)
        else:
            ax.plot(DOSs[0],E,'-',label=LABELS[0],color=color[0],linewidth=linewidth)
        for i,L in enumerate(LABELS[1:]):
            if fill==True:
                ax.plot(DOSs[i+1],E,'-',color=color[i+1],linewidth=linewidth)
                ax.fill_betweenx(E,DOSs[i+1],color=color[i+1],label=L,alpha=alpha)
            else:
                ax.plot(DOSs[i+1],E,'-',label=L,color=color[i+1],linewidth=linewidth)
        ax.set_ylim(E[0],E[-1])
        ax.set_ylabel('energy (eV)')
        ax.set_xlabel('DOS (a.u)')
        ax.set_xticks([])
        MAX=np.max(DOSs[:,np.where((E>=window[0]) & (E<=window[1]))[0]])
        ax.set_xlim(0,MAX*1.05)
        ax.set_ylim(window[0],window[1])
        
        if fermi!=None:                       #Fermi energy
            ax.axhline(y=0,color='black',linewidth=0.4)
    ax.legend(loc='upper right',fontsize='small')
    if title!=None:                             #Title option
        ax.set_title(title)
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()



def bands_fat(file,proj_file,KPATH=None,aux_file=None,species=None,atoms=None,l=None,j=None,mj=None,
          title=None,color='Reds',colormap=True,vmin=0,vmax=1,shift=0,size=50,legend=None,only_fat=False,
          vectors=None,ticks=None,labels=None,fermi=None,window=None,
          back_color='gray',style=None,linewidth=0.5,filetype=None,proj_filetype=None,
          figsize=(8,4),plot_ticks=True,
          IgnoreWeight=True,save_as=None,axis=None):
    """Plots fat bands over:
        bands.pwo file of a band calculation in Quantum Espresso
        EIGENVALUES file of a VASP calculation
        bands.dat.gnu file of bands postprocessing (Quantum Espresso)
        band.dat file in Wannier90
        bulkek.dat in Wanniertools

    Minimal plots can be done with just:
        file = Path to the file with bandstructure
        proj_file = File with the projected bands, or the output from grep_kpoints_energies_projections
                   qe_proj_out (quantum espresso out for projwfc.x)
                   PROCAR (VASP projections file)

    Two aditional files can be provide to autocomplete almost everything:
        KPATH = File with PATH and legends for the HSP in the VASP format as provided by topologicalquantumchemistry.fr
        aux_file = A file from which read the Fermi level, number of electrons and structure.
                   In the case of QE this would be an scf.pwo of nscf.pwo
                   In the case of VASP this is the OUTCAR
    
    species = list of atomic species ['Bi','Se'...] to project over.
    atoms = list with atoms index [1,2...] to project over.
    l = list of orbital atomic numbers to project over:
        qe: [0, 1, 2]
        vasp: ['s','px','py','dxz']  (as written in POSCAR)
    j = total angular mometum of projected states. (qe only)
    mj = m_j state of projected states. (qe only)

    However everything may be introduced manually:
    
    title = 'Your nice and original title for the plot'
    color = Color for your fat bands
    colormap = Boolean whether you are inputing a color or a colormap.
    vmin = Minimum value of a projection for the colormap
    vmax = Maximum value of a projection for the colormap
    shift = For plotting multiple projections it is handy to sligtly shift them.
    size = factor for which the size of projections is mutiplied.
    legend = Legend for flat bands.
    only_fat = Boolean controlling whether you want just the fat bands (to overlap over other plot).
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks = np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
            Ticks in your bandstructure (the usual HSP)
    labels = ["$\Gamma$","$X$","$M$","$\Gamma$"]
    fermi = Fermi energy in order to shift the band structure accordingly
    window = Window of energies to plot around Fermi, it can be a single number or 2
            either window=0.5 or window=[-0.5,0.5] => Same result
    back_color = Color for your "non-projected" bands
    style = desired line style (solid, dashed, dotted...)
    linewidth = desired line width
    proj_filetype = qe_proj_out (quantum espresso proj.pwo)
                    procar (VASP PROCAR file)
    filetype = Filetype of yoru bandstructure
                   qe (quantum espresso bands.pwo)
                   EIGENVAL (VASP EIGENVAL file)
                   data (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)   
    figsize = (int,int) => Size and shape of the figure
    plot_ticks = Boolean describing wether you want your ticks and labels
    IgnoreWeight = Boolean controlling whether points with non-zero weight would be ignored
    save_as = 'wathever.format'
    axis = ax in which to plot, if no axis is present new figure is created
    """
    #READ input
    if filetype == None:
        file = ut.file(file)
    else:
        file = ut.file(file,filetype)
    if KPATH != None:
        KPATH=ut.file(KPATH)
    if aux_file != None:
        aux_file=ut.file(aux_file)
    if type(proj_file)!=str:
        STATES, KPOINTS, ENERGIES, PROJECTIONS = proj_file
    else:
        proj_file=ut.file(proj_file)
        STATES, KPOINTS, ENERGIES, PROJECTIONS = proj_file.grep_kpoints_energies_projections(proj_file.filetype,IgnoreWeight)

    #Select parameters
    if KPATH!=None:
        ticks,labels=KPATH.path,KPATH.labels
    if file.filetype[:2]=='qe' and aux_file==None:
        vectors=file.lattice

    if aux_file!=None:
        v=aux_file.lattice
        f=aux_file.fermi
        n=aux_file.electrons
        if fermi==None:
            fermi=f
        if vectors is None:
            vectors=v

    if fermi!=None:
        if window==None:
            window=1
    else:
        fermi=0

    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis

    data=__process_electron_bands(file.file,file.filetype,vectors,IgnoreWeight)
    limits=__plot_electrons(file.file,file.filetype,vectors,ticks,fermi,color=back_color,linewidth=linewidth,
                                 style=style,ax=ax,IgnoreWeight=IgnoreWeight,plot=not only_fat)

    K_len=data[:,0]*limits[1]/data[:,0].max()
    ENERGIES=ENERGIES-fermi

    proj,n = ut.sum_projections(STATES,PROJECTIONS,proj_filetype,species,atoms,l,j,mj)
    print('(',species, atoms,l,j,mj,') ',n, 'states summed')

    proj=proj.transpose()
    for i,E in enumerate(ENERGIES.transpose()):
        if colormap==True:
            scatter=ax.scatter(K_len,E+shift,s=proj[i]*size,c=proj[i],cmap=color,alpha=proj[i],vmin=vmin,vmax=vmax,edgecolors='none')
        else:
            ax.scatter(K_len,E+shift,s=proj[i]*size,c=color,alpha=proj[i],edgecolors='none')
    if legend!=None:
        if colormap==True:
            ax.scatter(-1,0,c=0.7,cmap=color,s=20,label=legend,vmin=0,vmax=1)                #Dummy point (outside the plot) for the legend
        else:
            ax.scatter(-1,0,c=color,s=20,label=legend)

    if fermi!=None and only_fat==False:                       #Fermi energy
        ax.axhline(y=0,color='black',linewidth=0.4)
        if window!=None:                   #Limits y axis
            if type(window) is int or type(window) is float:
                window=[-window,window]
            ax.set_ylim(window[0],window[1])
        else:
            delta_y=limits[3]-limits[2]
            ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1)

    if vectors is not None and ticks is not None and plot_ticks==True and only_fat==False:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        if labels != None :
            ax.set_xticks(ticks,labels)
        else:
            ax.set_xticks(ticks)
        for i in range(1,ticks.shape[0]-1):
            ax.axvline(ticks[i],color='gray',linestyle='--',linewidth=0.4)
    if title!=None:                             #Title option
        ax.set_title(title)

    ax.set_ylabel('energy (eV)',labelpad=-1)
    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    plt.tight_layout()

    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()

# PLOTTING PHONONS----------------------------------------------------------------

def __lineseg_dist(p, a, b):
    """Function lineseg_dist returns the distance the distance from point p to line segment [a,b]. p, a and b are np.arrays."""
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))
    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])
    # perpendicular distance component
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c))



# PLOTTING MISCELLANY----------------------------------------------------------------
 
def lattice_comparison(folder,title=None,control=None,percentile=True,axis=None,markersize=8,save_as=None,output=False):
    """
    Plots the lattice comparison between different relax procedures, it is usefull to find the best pseudo/interaction matching your system.
    CAUTION: Be aware that it works in the STANDARDICE CELL convention!!! It will convert the files to compare in such setting.
    
    folder = Parent folder from where your relaxation kinds span. (the expected structure is explained below)
    title = Title for your plot
    control = File containing your control structure (for example the experimental one)
    percentile = If control structure is provided, then a percentile error plot is done.
    axis = ax in which to plot, if no axis is present new figure is created
    markersize = Size of the points
    save_as = Path and file type for your plot to be saved
    output = if true then the procedure returns two lists containing the interactions and the lattice parameters for each kind.
    
    ---
    The folowing folder structure is expected:
    Parent folder => Interaction1 => relax1 => output.pwo
                                  => relax2 => output.pwo
                  => Interaction2 => relax1 => output.pwo
                                  => relax2 => output.pwo
                                  => relax3 => output.pwo
                  ...
    The code will automatically select the last relaxation iteration's output.
    """
    #Grep the kinds of interactions from the respective folders
    interactions=[]
    folders=glob.glob(folder+'*')
    for file in folders:
        inter=file.split('/')[-1]
        interactions=interactions+[inter]   
    
    #For each interaction take the last relax output (assuming iterative relaxes named as relax<#num>)
    relax=[]
    for inter in interactions:
        relaxes=glob.glob(folder+'/'+inter+'/*/*pwo')
        relax_iter=0
        for r in relaxes:
            new_iter=int(r.split('/')[-2].split('relax')[1])
            if new_iter>relax_iter:
                last_relax=r
        relax=relax+[last_relax]
    
    #Read the data (in standardize cell)
    lattices=[]
    for file in relax:
        c=cell.read_spg(file)
        lattices=lattices+[spg.standardize_cell(c)[0]]
    
    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    #Plot if None experimental (then just plotting the results, not the comparison)
    if control==None or percentile==False:
        if control!=None:
            c_lattice=spg.standardize_cell(cell.read_spg(control))[0]
            ax.plot(0,np.linalg.norm(c_lattice[0]),'o',color='tab:red',label='a',markersize=markersize)
            ax.plot(0,np.linalg.norm(c_lattice[1]),'o',color='tab:green',label='b',markersize=markersize)
            ax.plot(0,np.linalg.norm(c_lattice[2]),'o',color='tab:blue',label='c',markersize=markersize)
            n=0
        else:
            ax.plot(0,np.linalg.norm(lattices[0][0]),'o',color='tab:red',label='a',markersize=markersize)
            ax.plot(0,np.linalg.norm(lattices[0][1]),'o',color='tab:green',label='b',markersize=markersize)
            ax.plot(0,np.linalg.norm(lattices[0][2]),'o',color='tab:blue',label='c',markersize=markersize)
            n=1
        for i,d in enumerate(lattices[n:]):
            ax.plot(i+1,np.linalg.norm(d[0]),'o',color='tab:red',markersize=markersize)
            ax.plot(i+1,np.linalg.norm(d[1]),'o',color='tab:green',markersize=markersize)
            ax.plot(i+1,np.linalg.norm(d[2]),'o',color='tab:blue',markersize=markersize)
        
        ax.set_ylabel('angstrom ($\mathrm{\AA}$)')
        if control==None:
            ax.set_xticks(range(len(interactions)),labels=interactions,rotation=50)
        else:
            ax.set_xticks(range(len(interactions)+1),labels=['EXP']+interactions,rotation=50)

    #Plot with experimental structure as control (show percentile error)
    if control!=None and percentile==True:
        c_lattice=spg.standardize_cell(cell.read_spg(control))[0]
        c0=np.linalg.norm(c_lattice[0])
        c1=np.linalg.norm(c_lattice[1])
        c2=np.linalg.norm(c_lattice[2])
        ax.axhline(y=0,color='tab:red',linestyle='-',linewidth=0.5)
        ax.plot(0,(np.linalg.norm(lattices[0][0])-c0)*100/c0,'o',color='tab:red',label='a',markersize=markersize)
        ax.plot(0,(np.linalg.norm(lattices[0][1])-c1)*100/c1,'o',color='tab:green',label='b',markersize=markersize)
        ax.plot(0,(np.linalg.norm(lattices[0][2])-c2)*100/c2,'o',color='tab:blue',label='c',markersize=markersize)
        for i,d in enumerate(lattices[1:]):
            ax.plot(i+1,(np.linalg.norm(d[0])-c0)*100/c0,'o',color='tab:red',markersize=markersize)
            ax.plot(i+1,(np.linalg.norm(d[1])-c1)*100/c1,'o',color='tab:green',markersize=markersize)
            ax.plot(i+1,(np.linalg.norm(d[2])-c2)*100/c2,'o',color='tab:blue',markersize=markersize)
        ax.set_ylabel('percentile error (%)')
        ax.set_xticks(range(len(interactions)),labels=interactions,rotation=50)
        
    if title!=None:
        ax.set_title(title)
    ax.legend()
    ax.grid()
    if axis == None:
        plt.tight_layout()
        plt.show()
        if save_as!=None:
            plt.savefig(save_as,dpi=300)
    if output==True:
        return interactions,lattices
