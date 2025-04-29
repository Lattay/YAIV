#!/bin/bash
SOFTWARE=${HOME}/Software
export PSEUDOS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"/pseudos
export NPROCS=2
export QE_PATH=$SOFTWARE/qe-7.3/bin
export VASP_PATH=$SOFTWARE/VASP/vasp.6.4.1/bin
