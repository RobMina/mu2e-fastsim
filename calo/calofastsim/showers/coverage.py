"""Descriptive whole-event CSS coverage, without draws, cuts or library changes.

``census(*export.read(...)[...])`` takes entries, steps, the original stored
census and the stored-event Birks totals (not geometry). ``frozen(mapping)``
takes the numeric mapping returned by library.load. Both return ``(arrays,
report)``. Arrays contain ONLY 96 flat cell rows, ordered disk, energy, cosine,
radius (radius fastest). All count columns are int64; birks_sum is float64.
``positive_birks`` counts entries with positive whole-event Birks, not CSS rows.
Out-of-bin entries are summarized separately and never fill a boundary cell.
Reports and ``compare`` results are JSON-compatible dictionaries.
"""
from __future__ import annotations

import copy

import numpy as np

from .library import Binning, cells, frame


_COORDS = ("disk", "energy_bin", "cosine_bin", "radius_bin")
_COUNTS = ("counts", "empty", "zero_energy", "weak", "positive_birks", "css_rows")
_SHAPE = (2, 4, 3, 4)
_SIZE = 96
_CAVEAT = (
    "Descriptive stored-population coverage only; thresholds are not effective N, "
    "precision guarantees or measured-query closure. No response veto, tuning, "
    "sampling or donor merging. Empty means no CSS rows; zero_energy means "
    "whole-event both-disk Birks == 0; weak means Birks <= 10 MeV including zeros. "
    "Non-front events and failed/unavailable transport are not front-entry zeros."
)


def _numeric(value, shape, name, *, integer=False, nonnegative=False):
    a = np.asarray(value)
    kinds = "iu" if integer else "iuf"
    if a.shape != shape or a.dtype.kind not in kinds or not np.all(np.isfinite(a)):
        raise ValueError("invalid " + name)
    if nonnegative and np.any(a < 0):
        raise ValueError("negative " + name)
    return a


def _check(entries, steps):
    """Equivalent shape/physics checks to library._check, with strict ID dtypes.

    Zero entry energy is accepted for the fixed lower bin boundary. Cosine must
    remain physical [0,1]; the terminal 1.000001 edge is a binning guard only.
    """
    e = {k: np.asarray(v) for k, v in entries.items()}
    s = {k: np.asarray(v) for k, v in steps.items()}
    if e["energy"].ndim != 1 or s["entry"].ndim != 1:
        raise ValueError("entry energy and step entry must be vectors")
    n, m = len(e["energy"]), len(s["entry"])
    for name in ("energy", "cosine", "radius", "time", "disk", "event_row"):
        _numeric(e[name], (n,), "entry " + name,
                 integer=name in ("disk", "event_row"), nonnegative=name != "time")
    for name in ("position", "direction"):
        _numeric(e[name], (n, 3), "entry " + name)
    if np.any(e["disk"] > 1) or np.any(e["cosine"] > 1):
        raise ValueError("invalid entry disk or cosine")
    frame(e["direction"])
    if len(np.unique(e["event_row"])) != n:
        raise ValueError("one entry per source event is required")
    for name in ("entry", "energy_g4", "energy_birks", "time", "crystal", "disk"):
        _numeric(s[name], (m,), "step " + name,
                 integer=name in ("entry", "crystal", "disk"), nonnegative=name != "time")
    _numeric(s["position"], (m, 3), "step position")
    if np.any(s["entry"] >= n) or np.any(s["disk"] > 1):
        raise ValueError("invalid step entry or disk")
    # Normalize checked integer indices to avoid unsigned indexing surprises.
    for table, names in ((e, ("disk",)), (s, ("entry", "disk"))):
        for name in names:
            table[name] = table[name].astype(np.int64, copy=False)
    return e, s


def _thresholds(counts):
    return dict(cells=len(counts), occupied=int(np.sum(counts > 0)),
                gt8=int(np.sum(counts > 8)), ge100=int(np.sum(counts >= 100)),
                ge200=int(np.sum(counts >= 200)), ge400=int(np.sum(counts >= 400)))


def _summary(rows, energy, mask):
    r, b = rows[mask], energy[mask]
    total = float(b.sum())
    if not np.isfinite(total):
        raise ValueError("nonfinite accumulated Birks energy")
    return dict(counts=len(b), empty=int(np.sum(r == 0)),
                zero_energy=int(np.sum(b == 0)), weak=int(np.sum(b <= 10)),
                positive_birks=int(np.sum(b > 0)), css_rows=int(r.sum()), birks_sum=total)


def _response(rows, energy, disk_rows, disk_energy, mask, entry_disk):
    result = _summary(rows, energy, mask)
    result["actual_css_disk"] = {
        str(d): _summary(disk_rows[:, d], disk_energy[:, d], mask) for d in (0, 1)
    }
    other = 1 - entry_disk
    index = np.arange(len(rows))
    result["other_disk"] = _summary(disk_rows[index, other], disk_energy[index, other], mask)
    result["actual_css_disk_denominator"] = "all entries in this group, including entries with no rows in that CSS disk"
    return result


def _calculate(entries, steps):
    e, s = _check(entries, steps)
    n = len(e["energy"])
    row = s["entry"]
    rows = np.bincount(row, minlength=n)
    energy = np.bincount(row, weights=s["energy_birks"], minlength=n)
    disk_rows = np.bincount(2 * row + s["disk"], minlength=2 * n).reshape(n, 2)
    disk_energy = np.bincount(2 * row + s["disk"], weights=s["energy_birks"], minlength=2 * n).reshape(n, 2)
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(disk_energy)):
        raise ValueError("nonfinite accumulated Birks energy")
    index, outside = cells(e["energy"], e["cosine"], e["radius"], Binning())
    keep = ~outside
    label = np.ravel_multi_index((e["disk"][keep], *index[keep].T), _SHAPE)
    arrays = dict(zip(_COORDS, np.indices(_SHAPE, dtype=np.int64).reshape(4, -1)))
    arrays["counts"] = np.bincount(label, minlength=_SIZE).astype(np.int64)
    values = dict(empty=rows == 0, zero_energy=energy == 0, weak=energy <= 10,
                  positive_birks=energy > 0, css_rows=rows, birks_sum=energy)
    for name, value in values.items():
        a = np.bincount(label, weights=value[keep], minlength=_SIZE)
        arrays[name] = a.astype(np.float64 if name == "birks_sum" else np.int64, copy=False)
    all_entries = np.ones(n, bool)
    response = lambda mask: _response(rows, energy, disk_rows, disk_energy, mask, e["disk"])
    stratum = (e["disk"] == 1) & (e["energy"] >= 95) & (e["radius"] < 450) & (e["cosine"] >= .7)
    report = dict(
        bins={name: edge.tolist() for name, edge in zip(("energy", "cosine", "radius"), Binning().arrays())},
        cell_order=list(_COORDS), cell_count=_SIZE,
        bin_convention="left closed, right open except exact terminal edges included; outside excluded",
        front=response(all_entries), in_bins=response(keep), out_of_bins=response(outside),
        entry_disk={str(d): response(e["disk"] == d) for d in (0, 1)},
        out_of_bins_entry_disk={str(d): response(outside & (e["disk"] == d)) for d in (0, 1)},
        highEinnerdisk1=dict(definition="disk == 1, energy >= 95 MeV, radius < 450 mm, cosine >= 0.7; pre-existing diagnostic failure region, not a selection",
                            **response(stratum)),
        thresholds=_thresholds(arrays["counts"]),
        thresholds_entry_disk={str(d): _thresholds(arrays["counts"][arrays["disk"] == d]) for d in (0, 1)},
        caveat=_CAVEAT,
    )
    return arrays, report, e, energy


def _count(value, name):
    a = _numeric(value, (), name, integer=True, nonnegative=True)
    return int(a)


def _json_value(value):
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value.item() if isinstance(value, np.generic) else copy.deepcopy(value)


def census(entries, steps, stored_census, total):
    """Census the first forward direct-CE entries from export.read(primary).

    ``total`` is the *stored-event* both-disk Birks vector, indexed by event_row.
    The unmodified stored census (including its primary audit) is retained in
    report['stored_census']. No non-front event becomes an empty front shower.
    """
    arrays, report, e, energy = _calculate(entries, steps)
    required = ("events", "front_entry_events", "non_front_entry_events", "empty_css_events",
                "non_front_entry_positive_events", "css_rows", "retained_css_rows")
    counts = {k: _count(stored_census[k], k) for k in required}
    stored = counts["events"]
    total = _numeric(total, (stored,), "stored-event Birks total", nonnegative=True)
    if (counts["front_entry_events"] != len(energy) or
            counts["non_front_entry_events"] != stored - len(energy) or
            counts["retained_css_rows"] != len(steps["entry"]) or
            counts["css_rows"] < counts["retained_css_rows"] or
            counts["empty_css_events"] > stored or np.any(e["event_row"] >= stored)):
        raise ValueError("stored census does not match front entries / CSS")
    event_row = e["event_row"].astype(np.int64, copy=False)
    if not np.allclose(total[event_row], energy, rtol=1e-12, atol=1e-12):
        raise ValueError("stored-event totals disagree with whole-event front CSS")
    nonfront = np.ones(stored, bool)
    nonfront[event_row] = False
    if counts["non_front_entry_positive_events"] != int(np.sum(total[nonfront] > 0)):
        raise ValueError("stored non-front positive census mismatch")
    if (counts["empty_css_events"] < report["front"]["empty"] or
            counts["empty_css_events"] > int(np.sum(total == 0))):
        raise ValueError("stored empty CSS census mismatch")
    report.update(population="stored first-forward direct-CE front entries; not unconditional generated acceptance",
                  stored_census=_json_value(stored_census),
                  stored_events=stored, front_entry_events=len(energy),
                  non_front_entry_events=counts["non_front_entry_events"])
    return arrays, report


def frozen(library):
    """Read a packed frozen library without build/sample or changing its arrays.

    The saved bin edges, cell labels and outside flags must agree exactly with
    the fixed 96-cell grid. Different frozen binning is rejected, never rebinned.
    Relative positions/times are validated in their saved frame; no coordinate
    reconstruction is needed for this scalar whole-event census. Stored/no-front
    counts are unknown for this selected population and are explicitly null.
    """
    for name, edge in zip(("energy", "cosine", "radius"), Binning().arrays()):
        saved = _numeric(library["bins_" + name], edge.shape, "bins " + name)
        if not np.array_equal(saved, edge):
            raise ValueError("frozen bin edges differ from fixed coverage grid")
    entries = {k: library["entry_" + k] for k in
               ("event_row", "disk", "position", "direction", "energy", "cosine", "radius", "time")}
    if np.ndim(entries["energy"]) != 1 or np.ndim(library["step_energy_birks"]) != 1:
        raise ValueError("invalid packed energy shapes")
    n, m = len(entries["energy"]), len(library["step_energy_birks"])
    offsets = _numeric(library["offsets"], (n + 1,), "library offsets", integer=True, nonnegative=True)
    if offsets[0] != 0 or offsets[-1] != m or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid library offsets")
    row = np.repeat(np.arange(n), np.diff(offsets.astype(np.int64)))
    steps = {k: library["step_" + k] for k in ("energy_g4", "energy_birks", "disk", "crystal")}
    steps.update(entry=row, position=library["step_relative_position"], time=library["step_relative_time"])
    arrays, report, e, _ = _calculate(entries, steps)
    index, outside = cells(e["energy"], e["cosine"], e["radius"], Binning())
    saved_cell = _numeric(library["entry_cell"], (n, 3), "frozen entry cell", integer=True)
    saved_outside = np.asarray(library["entry_outside_bins"])
    if (not np.array_equal(saved_cell, index) or saved_outside.dtype.kind != "b" or
            saved_outside.shape != (n,) or not np.array_equal(saved_outside, outside)):
        raise ValueError("frozen cell labels or outside flags disagree with features")
    report.update(population="selected frozen library donors only; source stored-event census unavailable",
                  stored_census=None, stored_events=None, front_entry_events=n, non_front_entry_events=None)
    return arrays, report


def _cell_arrays(arrays):
    expected = np.indices(_SHAPE, dtype=np.int64).reshape(4, -1)
    for name, value in zip(_COORDS, expected):
        a = _numeric(arrays[name], (_SIZE,), name, integer=True)
        if not np.array_equal(a, value):
            raise ValueError("cell coordinates must include the ordered fixed 96 cells")
    result = {name: _numeric(arrays[name], (_SIZE,), name, integer=True, nonnegative=True)
              for name in _COUNTS}
    result["birks_sum"] = _numeric(arrays["birks_sum"], (_SIZE,), "birks_sum", nonnegative=True)
    c = result["counts"]
    if (np.any(result["empty"] > result["zero_energy"]) or
            np.any(result["zero_energy"] > result["weak"]) or np.any(result["weak"] > c) or
            np.any(result["positive_birks"] != c - result["zero_energy"]) or
            np.any(result["css_rows"] < c - result["empty"]) or
            np.any((result["birks_sum"] > 0) != (result["positive_birks"] > 0))):
        raise ValueError("inconsistent cell outcomes")
    return result


def compare(training_arrays, legacy_arrays):
    """Compare all fixed cells; never combine donors or claim query closure.

    Newly populated/lost are training relative to legacy. Differences reflect
    selected-population and production/runtime differences, not causal gains.
    Out-of-bin entries are absent from these arrays; consult each census report.
    """
    train, legacy = _cell_arrays(training_arrays), _cell_arrays(legacy_arrays)
    t, old = train["counts"], legacy["counts"]
    labels = np.indices(_SHAPE, dtype=np.int64).reshape(4, -1).T
    new, lost = (t > 0) & (old == 0), (t == 0) & (old > 0)
    records = []
    # A bounded 96-row serialization loop, not an event/CSS hot-path loop.
    for i, label in enumerate(labels):
        record = dict(zip(_COORDS, map(int, label)))
        record.update(training_counts=int(t[i]), legacy_counts=int(old[i]),
                      count_delta=int(t[i]) - int(old[i]))
        record["training"] = {k: v[i].item() for k, v in train.items()}
        record["legacy"] = {k: v[i].item() for k, v in legacy.items()}
        records.append(record)
    return dict(cells=records, cell_count=_SIZE,
                training_thresholds=_thresholds(t), legacy_thresholds=_thresholds(old),
                training_entries_in_bins=int(t.sum()), legacy_entries_in_bins=int(old.sum()),
                newly_populated_cells=labels[new].tolist(), newly_populated_count=int(new.sum()),
                lost_cells=labels[lost].tolist(), lost_count=int(lost.sum()),
                caveat="Separate populations, no legacy donors added to training. Distribution differences are confounded by selected population and production/runtime. All 96 in-bin cells compared; outside counts remain in the individual reports. Descriptive occupancy is not effective N or measured-query closure.")
