"""mu2efastsim.transport: hep-simkernels' hepsimkernels.transport (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.transport as _module
sys.modules[__name__] = _module
