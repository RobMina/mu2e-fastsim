"""benchmarks.fastsim_score.from_root: hep-simbench's hepsimbench.from_root (the code moved there; this name stays for every existing import and
``python3 -m benchmarks.fastsim_score.from_root``)."""
import sys

import hepsimbench.from_root as _module

if __name__ == "__main__":
    raise SystemExit(_module.main())
sys.modules[__name__] = _module
