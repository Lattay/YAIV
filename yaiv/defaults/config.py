#PYTHON module for storing yaiv defaults
from pint import UnitRegistry
from importlib.resources import files
import matplotlib

# First we create the registry.
ureg = UnitRegistry()
#ureg = UnitRegistry(system='atomic')
# Load unit definitions
ureg.load_definitions(files("yaiv") / "defaults/extra_units.txt")

colors='tab10'
