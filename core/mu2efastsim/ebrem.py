"""mu2efastsim.ebrem: hep-simkernels' hepsimkernels.ebrem (the code moved there; this name stays for every existing import and
``python3 -m mu2efastsim.ebrem``)."""
import sys

import hepsimkernels.ebrem as _module

if __name__ == "__main__":
    import runpy
    runpy.run_module("hepsimkernels.ebrem", run_name="__main__", alter_sys=True)
    raise SystemExit(0)
sys.modules[__name__] = _module
