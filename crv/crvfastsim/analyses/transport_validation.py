"""Alias of crvfastsim.validation.transport_validation (moved there; this alias is kept for one release)."""
import sys

from ..validation import transport_validation as _moved

if __name__ == "__main__":
    _moved.main()
else:
    sys.modules[__name__] = _moved
