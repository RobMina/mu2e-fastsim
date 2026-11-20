"""mu2efastsim.determinism: hep-simkernels' hepsimkernels.determinism (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.determinism as _module
sys.modules[__name__] = _module
