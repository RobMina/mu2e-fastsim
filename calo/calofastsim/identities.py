"""Vectorized persistent identity keys for one source artifact at a time."""
from __future__ import annotations

import numpy as np


EVENT_FIELDS = ("run", "subrun", "event")


def keys(table, fields):
    arrays = [np.asarray(table[field]) for field in fields]
    if any(array.ndim != 1 for array in arrays) or len({array.size for array in arrays}) != 1:
        raise ValueError("key columns must be one-dimensional with equal lengths")
    if any(not np.issubdtype(array.dtype, np.integer) or np.any(array < 0)
           or np.any(array > np.iinfo(np.int64).max) for array in arrays):
        raise ValueError("keys must be nonnegative integers in the supported int64 range")
    return np.rec.fromarrays([array.astype(np.int64) for array in arrays], names=fields)


def contains(sorted_unique, rows):
    if len(sorted_unique) == 0:
        return np.zeros(len(rows), dtype=bool)
    indices = np.searchsorted(sorted_unique, rows)
    valid = indices < len(sorted_unique)
    return valid & (sorted_unique[np.minimum(indices, len(sorted_unique) - 1)] == rows)
