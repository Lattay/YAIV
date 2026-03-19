#!/bin/bash

#Load calculation settings
source ../calcSettings.sh
#Load system
source SYSTEM.INFO

#Actual JOBS
##########################################################################
# Electronic spectrum
bash scf.sh
bash bands.sh
bash project_bands.sh

# Phonon spectrum
bash ph.sh
bash matdyn.sh
#########################################################################
rm -r tmp
mkdir results
cp results_scf/Si.scf.pw* results_ph/Si.ph.pwo results_bands/Si.bands.pwi results_matdyn/matdyn.in results_ph/Si.dyn1 results_matdyn/Si.freq results_proj/Si.proj.pwo results_bands/bands.xml results
echo "DONE"
