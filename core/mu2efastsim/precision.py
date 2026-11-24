"""mu2efastsim.precision: hep-simkernels' hepsimkernels.precision (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.precision as _module
sys.modules[__name__] = _module
