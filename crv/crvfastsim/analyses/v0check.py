"""Alias of crvfastsim.validation.v0check (moved there; this alias is kept for one release)."""
import sys

from ..validation import v0check as _moved

sys.modules[__name__] = _moved
