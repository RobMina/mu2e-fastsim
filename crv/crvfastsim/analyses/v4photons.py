"""Alias of crvfastsim.validation.v4photons (moved there; this alias is kept for one release)."""
import sys

from ..validation import v4photons as _moved

sys.modules[__name__] = _moved
