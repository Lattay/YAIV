#!/bin/bash
#
python -X importtime -c "import yaiv.defaults.config" 2> "importtimes/config"
python -X importtime -c "import yaiv.utils" 2> "importtimes/utils"
python -X importtime -c "import yaiv.grep" 2> "importtimes/grep"
python -X importtime -c "import yaiv.spectrum" 2> "importtimes/spectrum"
python -X importtime -c "import yaiv.plot" 2> "importtimes/plot"
python -X importtime -c "import yaiv.cell" 2> "importtimes/cell"
python -X importtime -c "import yaiv.phonon" 2> "importtimes/phonon"
python -X importtime -c "import yaiv.convergence" 2> "importtimes/convergence"
