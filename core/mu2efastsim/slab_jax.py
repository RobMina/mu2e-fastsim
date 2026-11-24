"""mu2efastsim.slab_jax: hep-simkernels' hepsimkernels.slab_jax (the code moved there; this name stays for every existing import)."""
import sys

import hepsimkernels.slab_jax as _module
sys.modules[__name__] = _module
