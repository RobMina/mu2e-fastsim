"""mu2efastsim.eslab: hep-simkernels' hepsimkernels.eslab (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.eslab as _module
sys.modules[__name__] = _module
