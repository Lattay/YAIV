#!/bin/bash

#Load calculation settings
source ../calcSettings.sh

mkdir RESULTS
cp * RESULTS
cd RESULTS
cp $PSEUDOS/Si_POTCAR ./POTCAR

# Actual JOBS
#########################################################################
# Self-consistent calculation
cp ./KPOINTS_SCC ./KPOINTS
cp ./INCAR_SCC ./INCAR
echo "scf calculation..."
$VASP_PATH/vasp_ncl >&SCC.log
cp ./OUTCAR ./OUTCAR_SCC
cp ./CHG ./CHG_SCC
cp vasprun.xml vasprun_SCC.xml

# Bands calculation
cp ./INCAR_BS ./INCAR
cp ./KPATH ./KPOINTS
echo "BS calculation..."
$VASP_PATH/vasp_ncl >&BS.log
cp ./OUTCAR ./OUTCAR_BS
cp ./EIGENVAL ./EIGENVAL_BS
cp vasprun.xml vasprun_BS.xml
#########################################################################
#
# Cleaning
rm CHG* CONTCAR DOSCAR IBZKPT INCAR* KPOINTS* KPATH OSZICAR OUTCAR PCDAT POSCAR REPORT vasprun.xml WAVECAR XDATCAR EIGENVAL POTCAR
echo "DONE"
