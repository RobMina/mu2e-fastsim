"""Complete donor histories expanded in fixed device blocks, without per-history caps."""
from __future__ import annotations

import numpy as np


def prepare(keys, donor_count, keep=None):
    """Build immutable row-CSR from sorted library identities, including empty donors."""
    from .tier1b_jax import _jax
    _, j = _jax()
    keys = np.asarray(keys, np.int64)
    if np.any(keys < 0) or np.any(keys >= donor_count) or np.any(keys[1:] < keys[:-1]):
        raise ValueError("CSR requires sorted in-range donor identities")
    rows = np.arange(len(keys), dtype=np.int64)
    if keep is not None:
        rows = rows[np.asarray(keep, bool)]
    counts = np.bincount(keys[rows], minlength=donor_count)
    offsets = np.r_[0, np.cumsum(counts)]
    return dict(offsets=j.asarray(offsets), rows=j.asarray(rows if len(rows) else [0], dtype=j.int64))


def layout(csr, picks, valid):
    """Prefix sums retain complete selected histories and zero-history donors."""
    from .tier1b_jax import _jax
    _, j = _jax()
    safe = j.clip(picks, 0, len(csr["offsets"]) - 2)
    starts = csr["offsets"][safe]
    counts = j.where(valid & (picks >= 0), csr["offsets"][safe + 1] - starts, 0)
    ends = j.cumsum(counts)
    return dict(starts=starts, ends=ends, total=ends[-1])


def block(csr, plan, offset, lanes):
    """Expand a bounded row range; overflow is drained by more blocks, never truncated."""
    from .tier1b_jax import _jax
    _, j = _jax()
    pos = offset + lanes
    valid = pos < plan["total"]
    ix = j.minimum(j.searchsorted(plan["ends"], pos, side="right"), len(plan["ends"]) - 1)
    previous = j.where(ix > 0, plan["ends"][j.maximum(ix - 1, 0)], 0)
    loc = plan["starts"][ix] + pos - previous
    rows = csr["rows"][j.clip(loc, 0, len(csr["rows"]) - 1)]
    return ix, rows, valid
