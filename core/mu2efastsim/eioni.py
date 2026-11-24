"""mu2efastsim.eioni: hep-simkernels' hepsimkernels.eioni (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.eioni as _module
sys.modules[__name__] = _module
