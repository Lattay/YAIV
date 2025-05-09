#!/bin/bash
# Usefull script to create test_enviroments and check installation
ENV=yaiv-test
DIR=`pwd`
cd ~/Software/enviroments
#jupyter kernelspec uninstall test_env
rm -r $ENV
/usr/bin/python3.10 -m venv $ENV
source $ENV/bin/activate
pip install --upgrade pip
pip install ipykernel
#python -m ipykernel install --user --name=$ENV
#jupyter kernelspec list
cd $DIR
#Autoinstall YAIV
#pip install .
#pip install -e ./ | tee install.log
#pip install . --config-settings editable_mode=strict
pip install -e . --config-settings editable_mode=strict
