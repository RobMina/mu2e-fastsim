"""mu2efastsim.eloss: hep-simkernels' hepsimkernels.eloss (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.eloss as _module
sys.modules[__name__] = _module
