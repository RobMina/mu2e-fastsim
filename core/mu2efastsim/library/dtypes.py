"""The dtype policy of a library (docs/libraries.md, "Contract").

- Storage: floating arrays are stored as built (float64 unless the manifest declares a smaller storage dtype for
  size, e.g. float32 pools); integer keys and indices as int64 or the smallest integer type that holds them; the
  manifest lists every array's dtype.
- Host lookups (NumPy) compute in float64 and return int64 indices.
- Device lookups compute in the dtype the caller asks for, float32 or float64, both supported by every library
  (``manifest.dtype.compute``); floating arrays are cast to it once, when the device copy is made, and integer index
  arrays become int32. ``compute_dtype(None)`` is the device float of the run (MU2EFS_DEVICE_FLOAT, as in
  :mod:`mu2efastsim.precision`).
"""
import os

import numpy as np

COMPUTE = ("float32", "float64")


def compute_dtype(dtype=None):
    """np.dtype of a compute dtype: float32 or float64; None gives the run's device float (MU2EFS_DEVICE_FLOAT)."""
    if dtype is None:
        dtype = os.environ.get("MU2EFS_DEVICE_FLOAT") or "float32"
    d = np.dtype(dtype)
    if d.name not in COMPUTE:
        raise ValueError(f"compute dtype must be float32 or float64, not {d.name}")
    return d


def device_arrays(arrays, dtype, *, index_dtype=np.int32, keep=(), xp=None):
    """Device copies of a library's arrays: floating arrays in ``dtype``, integer arrays in ``index_dtype``, arrays in
    ``keep`` (and booleans, strings) unchanged. ``xp`` is jax.numpy (default, imported here) or numpy (tests)."""
    if xp is None:
        import jax.numpy as xp
    d = compute_dtype(dtype)
    out = {}
    for k, v in arrays.items():
        a = np.asarray(v)
        if k in keep or a.dtype.kind in "bUS":
            out[k] = a if a.dtype.kind in "US" else xp.asarray(a)
        elif a.dtype.kind == "f":
            out[k] = xp.asarray(a, d)
        elif a.dtype.kind in "iu":
            if a.size and (a.max() > np.iinfo(index_dtype).max or a.min() < np.iinfo(index_dtype).min):
                raise OverflowError(f"array {k!r} does not fit {np.dtype(index_dtype).name}")
            out[k] = xp.asarray(a, index_dtype)
        else:
            raise TypeError(f"array {k!r}: dtype {a.dtype} has no device policy")
    return out
