"""benchmarks.fastsim_score.score: hep-simbench's hepsimbench.score (the code moved there; this name stays for every existing import and
``python3 -m benchmarks.fastsim_score.score``)."""
import sys

import hepsimbench.score as _module

if __name__ == "__main__":
    raise SystemExit(_module.main())
sys.modules[__name__] = _module
