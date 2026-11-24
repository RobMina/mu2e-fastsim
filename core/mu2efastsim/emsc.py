"""mu2efastsim.emsc: hep-simkernels' hepsimkernels.emsc (the code moved there; this name stays for every existing import and
``python3 -m mu2efastsim.emsc``)."""
import sys

import hepsimkernels.emsc as _module

if __name__ == "__main__":
    import runpy
    runpy.run_module("hepsimkernels.emsc", run_name="__main__", alter_sys=True)
    raise SystemExit(0)
sys.modules[__name__] = _module
