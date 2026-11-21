"""mu2efastsim.eannihil: hep-simkernels' hepsimkernels.eannihil (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.eannihil as _module
sys.modules[__name__] = _module
