"""mu2efastsim.ehelix: hep-simkernels' hepsimkernels.ehelix (the code moved there; this name stays for every existing import and
``python3 -m mu2efastsim.ehelix``)."""
import sys

import hepsimkernels.ehelix as _module

if __name__ == "__main__":
    import runpy
    runpy.run_module("hepsimkernels.ehelix", run_name="__main__", alter_sys=True)
    raise SystemExit(0)
sys.modules[__name__] = _module
