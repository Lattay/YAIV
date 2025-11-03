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
rm -r tmp
#########################################################################

echo "DONE"
