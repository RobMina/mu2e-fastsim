"""mu2efastsim.eslab_jax: hep-simkernels' hepsimkernels.eslab_jax (the code moved there; this name stays for every existing import and
``python3 -m mu2efastsim.eslab_jax``)."""
import sys

import hepsimkernels.eslab_jax as _module

if __name__ == "__main__":
    import runpy
    runpy.run_module("hepsimkernels.eslab_jax", run_name="__main__", alter_sys=True)
    raise SystemExit(0)
sys.modules[__name__] = _module
