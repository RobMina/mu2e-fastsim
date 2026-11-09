"""Alias of crvfastsim.validation.v2compare (moved there; this alias is kept for one release)."""
import sys

from ..validation import v2compare as _moved

sys.modules[__name__] = _moved
