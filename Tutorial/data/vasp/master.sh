#!/bin/bash

mkdir RESULTS
cp * RESULTS
cd RESULTS

# Actual JOBS
#########################################################################
# Self-consistent calculation
cp ./KPOINTS_SCC ./KPOINTS
cp ./INCAR_SCC ./INCAR
echo "scf calculation..."
vasp_ncl >&SCC.log
cp ./OUTCAR ./OUTCAR_SCC
cp ./CHG ./CHG_SCC
cp vasprun.xml vasprun_SCC.xml

# Bands calculation
cp ./INCAR_BS ./INCAR
cp ./KPATH ./KPOINTS
echo "BS calculation..."
vasp_ncl >&BS.log
cp ./OUTCAR ./OUTCAR_BS
cp ./EIGENVAL ./EIGENVAL_BS
cp vasprun.xml vasprun_BS.xml
#########################################################################
echo "DONE"
