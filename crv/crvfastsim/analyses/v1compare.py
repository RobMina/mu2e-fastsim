"""Alias of crvfastsim.validation.v1compare (moved there; this alias is kept for one release)."""
import sys

from ..validation import v1compare as _moved

sys.modules[__name__] = _moved
