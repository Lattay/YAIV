#!/bin/bash
# Usefull script to create test_enviroments and check installation
ENV=qgt-dev
J_ENV=QGT-dev
DIR=`pwd`
cd ~/Software/enviroments
#jupyter kernelspec uninstall test_env
rm -r $ENV
/usr/bin/python3.10 -m venv $ENV
source $ENV/bin/activate
pip install --upgrade pip
#pip install ipykernel==6.17.1
pip install ipykernel
python -m ipykernel install --user --name=$J_ENV
#jupyter kernelspec list
cd $DIR
#Autoinstall YAIV
#pip install -e ./ | tee install.log
#pip install .
