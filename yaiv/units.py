from pint import UnitRegistry
from importlib.resources import files

# First we create the registry.
ureg = UnitRegistry()
#ureg = UnitRegistry(system='atomic')
# Load unit definitions
ureg.load_definitions(files("yaiv") / "extra_units.txt")
