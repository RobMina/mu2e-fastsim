"""mu2efastsim.world: hep-simkernels' hepsimkernels.world (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.world as _module
sys.modules[__name__] = _module
