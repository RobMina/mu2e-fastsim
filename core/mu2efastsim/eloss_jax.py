"""mu2efastsim.eloss_jax: hep-simkernels' hepsimkernels.eloss_jax (the code moved there; this name stays for every existing import and
``python3 -m mu2efastsim.eloss_jax``)."""
import sys

import hepsimkernels.eloss_jax as _module

if __name__ == "__main__":
    import runpy
    runpy.run_module("hepsimkernels.eloss_jax", run_name="__main__", alter_sys=True)
    raise SystemExit(0)
sys.modules[__name__] = _module
