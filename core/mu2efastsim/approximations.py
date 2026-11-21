"""mu2efastsim.approximations: hep-simbench's hepsimbench.approximations (the code moved there; this name stays for every existing import and
``python3 -m mu2efastsim.approximations``)."""
import sys

import hepsimbench.approximations as _module

if __name__ == "__main__":
    raise SystemExit(_module.main())
sys.modules[__name__] = _module
