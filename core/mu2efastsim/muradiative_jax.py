"""mu2efastsim.muradiative_jax: hep-simkernels' hepsimkernels.muradiative_jax (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.muradiative_jax as _module
sys.modules[__name__] = _module
