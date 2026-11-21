"""mu2efastsim.crossings: hep-simkernels' hepsimkernels.crossings (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.crossings as _module
sys.modules[__name__] = _module
