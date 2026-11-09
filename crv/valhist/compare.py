"""Alias of crvfastsim.validation.compare (moved there; this alias is kept for one release)."""
import sys

from crvfastsim.validation import compare as _moved

if __name__ == "__main__":
    _moved.main()
else:
    sys.modules[__name__] = _moved
