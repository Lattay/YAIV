from types import SimpleNamespace
from typing import Sequence, Any

import numpy as np
from scipy.special import factorial, hermite
from scipy import integrate

from yaiv.defaults.config import ureg
from yaiv import utils as ut
from yaiv import spectrum as spec
