"""Coverage and out-of-domain accounting (docs/libraries.md, "Domain").

A library declares, per key, what a value outside its tabulated domain does (the axis ``policy``) and, per cell, any
fallback it uses where the training sample is sparse (a coarser cell, a neighbouring bin). Neither may be silent:

- ``refuse``: the lookup raises :class:`OutOfDomain` naming the rows, before any random number is drawn;
- ``clip`` / ``nan`` / a named fallback: the lookup proceeds as declared, and a :class:`DomainCounter` counts it.

A counter is plain host data (NumPy integers), so it can be filled from device results: a jitted lookup returns its
per-row codes (``axes.domain_codes_jax``) or fallback levels next to its output, and the caller adds them with
:meth:`DomainCounter.add_codes` / :meth:`DomainCounter.add_levels` after the device sync. ``record()`` is the JSON
entry an output or a validation record carries: counts and the coverage and out-of-domain fractions.
"""
from dataclasses import dataclass, field

import numpy as np

from .axes import ABOVE, BELOW, CODE_NAMES, IN, NONFINITE, domain_codes


class OutOfDomain(ValueError):
    """Rows outside a library's declared domain under the ``refuse`` policy (never answered with a made-up value)."""

    def __init__(self, library, axis, rows, codes=None):
        self.library, self.axis = library, axis
        self.rows = np.asarray(rows, dtype=np.int64)
        self.codes = None if codes is None else np.asarray(codes, dtype=np.int8)
        shown = self.rows[:10].tolist()
        more = "" if self.rows.size <= 10 else f" (+{self.rows.size - 10} more)"
        super().__init__(f"{library}: {self.rows.size} row(s) outside the domain of {axis!r}: rows {shown}{more}")


def check(library, axis, codes):
    """Raise OutOfDomain for any row whose code is not IN (the ``refuse`` policy)."""
    codes = np.asarray(codes)
    bad = np.flatnonzero(codes != IN)
    if bad.size:
        raise OutOfDomain(library, axis, bad, codes[bad])


@dataclass
class DomainCounter:
    """Counts of lookups, per axis out-of-domain outcomes and per fallback level, for one library."""
    library: str
    lookups: int = 0
    axes: dict = field(default_factory=dict)        # axis -> {"below": n, "above": n, "nonfinite": n}
    levels: dict = field(default_factory=dict)      # fallback name -> {level: n}
    refused: int = 0

    def add_lookups(self, n):
        self.lookups += int(n)

    def add_codes(self, axis, codes, mask=None):
        """Count the out-of-domain codes of one axis (``mask``: only these rows were looked up)."""
        codes = np.asarray(codes)
        if mask is not None:
            codes = codes[np.asarray(mask, bool)]
        counts = np.bincount(codes.astype(np.int64).ravel(), minlength=4)
        d = self.axes.setdefault(axis, {CODE_NAMES[k]: 0 for k in (BELOW, ABOVE, NONFINITE)})
        for k in (BELOW, ABOVE, NONFINITE):
            d[CODE_NAMES[k]] += int(counts[k])

    def add_values(self, axis, edges, values, mask=None):
        """Code ``values`` against ``edges`` (NumPy) and count them."""
        self.add_codes(axis, domain_codes(edges, values), mask)

    def add_levels(self, fallback, levels, mask=None):
        """Count the fallback level used per row (0 = the declared cell itself; >0 = a declared fallback)."""
        levels = np.asarray(levels)
        if mask is not None:
            levels = levels[np.asarray(mask, bool)]
        values, counts = np.unique(levels.astype(np.int64).ravel(), return_counts=True)
        d = self.levels.setdefault(fallback, {})
        for v, c in zip(values.tolist(), counts.tolist()):
            d[str(v)] = d.get(str(v), 0) + int(c)

    def add_refused(self, n):
        self.refused += int(n)

    def merge(self, other):
        if other.library != self.library:
            raise ValueError(f"cannot merge counters of {other.library!r} into {self.library!r}")
        self.lookups += other.lookups
        self.refused += other.refused
        for axis, d in other.axes.items():
            mine = self.axes.setdefault(axis, {k: 0 for k in d})
            for k, v in d.items():
                mine[k] = mine.get(k, 0) + v
        for name, d in other.levels.items():
            mine = self.levels.setdefault(name, {})
            for k, v in d.items():
                mine[k] = mine.get(k, 0) + v
        return self

    def out_of_domain(self, axis=None):
        """Rows outside the domain of one axis (or summed over axes: an upper bound on the rows, a row can count
        on several axes)."""
        axes = [axis] if axis is not None else list(self.axes)
        return int(sum(sum(self.axes.get(a, {}).values()) for a in axes))

    def fallback(self, name):
        """Rows that used a level above 0 of the named fallback."""
        return int(sum(v for k, v in self.levels.get(name, {}).items() if k != "0"))

    def record(self):
        """JSON entry: counts and fractions (coverage = 1 - out-of-domain fraction, per axis)."""
        n = max(self.lookups, 1)
        per_axis = {a: dict(d, fraction=sum(d.values()) / n) for a, d in self.axes.items()}
        fallbacks = {name: dict(levels=dict(d), fraction=self.fallback(name) / n) for name, d in self.levels.items()}
        return dict(library=self.library, lookups=self.lookups, refused=self.refused, axes=per_axis,
                    fallbacks=fallbacks,
                    coverage={a: 1.0 - v["fraction"] for a, v in per_axis.items()},
                    out_of_domain_fraction=max((v["fraction"] for v in per_axis.values()), default=0.0))
