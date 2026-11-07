"""Synthetic scalar references for the descriptive, no-draw coverage census."""
from copy import deepcopy
import json

import numpy as np
import pytest

from calofastsim.showers import coverage
from calofastsim.showers import library as shower_library


@pytest.fixture(autouse=True)
def no_build_or_sample(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("coverage must not build or sample")
    monkeypatch.setattr(shower_library, "build", forbidden)
    monkeypatch.setattr(shower_library, "sample", forbidden)


def make_case(n=10, records=None):
    e = dict(event_row=np.arange(n, dtype=np.int64), disk=np.arange(n, dtype=np.int64) % 2,
             energy=np.full(n, 100.), cosine=np.full(n, .8), radius=np.full(n, 300.),
             time=np.zeros(n), position=np.zeros((n, 3)), direction=np.tile([0., 0., 1.], (n, 1)))
    records = [] if records is None else records
    row = np.asarray([r[0] for r in records], np.int64)
    energy = np.asarray([r[2] for r in records], float)
    m = len(records)
    s = dict(entry=row, disk=np.asarray([r[1] for r in records], np.int64),
             energy_birks=energy, energy_g4=energy.copy(), time=np.zeros(m),
             crystal=np.arange(m, dtype=np.int64), position=np.zeros((m, 3)))
    return e, s


def stored(e, s, extra_totals=()):
    """Construct complete stored-event census; extra zero totals have no CSS."""
    n = len(e["energy"])
    total = np.zeros(n + len(extra_totals))
    row_counts = np.zeros(n, int)
    for entry, value in zip(s["entry"], s["energy_birks"]):
        total[e["event_row"][entry]] += value
        row_counts[entry] += 1
    total[n:] = extra_totals
    original = dict(events=len(total), front_entry_events=n, non_front_entry_events=len(extra_totals),
                    empty_css_events=int(np.sum(row_counts == 0) + np.sum(np.asarray(extra_totals) == 0)),
                    non_front_entry_positive_events=int(np.sum(np.asarray(extra_totals) > 0)),
                    css_rows=len(s["entry"]) + int(np.sum(np.asarray(extra_totals) > 0)),
                    retained_css_rows=len(s["entry"]), direct_ce_particles=np.int64(n))
    return original, total


def run(e, s, extra_totals=()):
    original, total = stored(e, s, extra_totals)
    return coverage.census(e, s, original, total)


def packed(e, s):
    """Synthetic packed mapping, deliberately without library.build/sample."""
    n = len(e["energy"])
    order = np.argsort(s["entry"], kind="stable")
    lib = {"entry_" + k: v.copy() for k, v in e.items()}
    lib.update({"step_" + k: v[order].copy() for k, v in s.items() if k != "entry"})
    # Relative values need only be finite for scalar coverage.
    lib["step_relative_position"] = lib.pop("step_position")
    lib["step_relative_time"] = lib.pop("step_time")
    lib["offsets"] = np.r_[0, np.cumsum(np.bincount(s["entry"], minlength=n))]
    edges = shower_library.Binning().arrays()
    lib.update({"bins_" + k: v.copy() for k, v in zip(("energy", "cosine", "radius"), edges)})
    coordinate, outside = [], []
    for values in zip(e["energy"], e["cosine"], e["radius"]):
        coordinate.append([min(max(sum(x >= edge for edge in bins) - 1, 0), len(bins) - 2)
                           for x, bins in zip(values, edges)])
        outside.append(any(x < bins[0] or x > bins[-1] for x, bins in zip(values, edges)))
    lib["entry_cell"] = np.asarray(coordinate, np.int64).reshape(n, 3)
    lib["entry_outside_bins"] = np.asarray(outside, bool)
    return lib


def scalar(e, s):
    """Independent event/step loop reference, confined to tests."""
    labels = [(d, a, b, c) for d in (0, 1) for a in range(4) for b in range(3) for c in range(4)]
    names = ("counts", "empty", "zero_energy", "weak", "positive_birks", "css_rows", "birks_sum")
    result = {k: np.zeros(96, float if k == "birks_sum" else np.int64) for k in names}
    outside = 0
    edges = ((0, 50, 80, 95, 110), (0, .4, .7, 1.000001), (0, 450, 550, 750, 1000))
    for i in range(len(e["energy"])):
        values = [e[k][i] for k in ("energy", "cosine", "radius")]
        if any(x < bins[0] or x > bins[-1] for x, bins in zip(values, edges)):
            outside += 1
            continue
        coordinate = [min(sum(x >= edge for edge in bins) - 1, len(bins) - 2)
                      for x, bins in zip(values, edges)]
        cell = labels.index((int(e["disk"][i]), *coordinate))
        deposits = [value for row, value in zip(s["entry"], s["energy_birks"]) if row == i]
        energy = sum(deposits)
        values = (1, not deposits, energy == 0, energy <= 10, energy > 0, len(deposits), energy)
        for name, value in zip(names, values):
            result[name][cell] += value
    result.update(dict(zip(("disk", "energy_bin", "cosine_bin", "radius_bin"), np.asarray(labels).T)))
    return result, outside


@pytest.fixture
def mixed():
    e, s = make_case(records=[(1, 1, 0), (2, 0, 10), (3, 1, 10.001), (4, 1, 12),
                             (5, 0, 5), (5, 1, 6), (6, 1, 0), (8, 0, 8)])
    e["energy"][:] = [0, 49, 50, 80, 95, 110, 111, 100, 105, 20]
    e["cosine"][:] = [0, .399, .4, .699, .7, 1, .8, .8, .8, .8]
    e["radius"][:] = [0, 449, 450, 549, 550, 1000, 300, 1000.001, 300, 300]
    return e, s


def test_scalar_all_cells_edges_outside_and_original_census(mixed):
    e, s = mixed
    original, total = stored(e, s, [0, 17])
    original["audit"] = dict(saved_primary=True, count=np.int64(12))
    arrays, report = coverage.census(e, s, original, total)
    expected, outside = scalar(e, s)
    assert set(arrays) == set(expected)
    for name in arrays:
        assert arrays[name].shape == (96,)
        np.testing.assert_allclose(arrays[name], expected[name], rtol=0, atol=1e-12)
        assert arrays[name].dtype == (np.float64 if name == "birks_sum" else np.int64)
    assert arrays["counts"].sum() == 8
    assert report["out_of_bins"]["counts"] == outside == 2
    assert report["front"]["counts"] == 10
    assert report["front"]["empty"] == 3
    assert report["front"]["zero_energy"] == 5
    assert report["front"]["weak"] == 7  # exactly 10 is weak, 10.001 is not
    assert report["front"]["css_rows"] == 8
    assert report["stored_events"] == 12
    assert report["non_front_entry_events"] == 2
    assert report["stored_census"] == original
    assert report["stored_census"] is not original
    assert report["stored_census"]["audit"] is not original["audit"]
    assert json.loads(json.dumps(report, allow_nan=False))["stored_census"]["audit"]["count"] == 12
    # Exact upper edges enter the last bin; out-of-range energy/radius do not.
    last = (arrays["disk"] == 1) & (arrays["energy_bin"] == 3) & (arrays["cosine_bin"] == 2) & (arrays["radius_bin"] == 3)
    assert arrays["counts"][last].item() == 1
    assert arrays["birks_sum"][last].item() == 11


def test_other_disk_is_not_masked_and_fixed_stratum():
    e, s = make_case(5, [(0, 0, 15), (1, 1, 4), (1, 0, 7), (2, 0, 2), (3, 1, 1)])
    e["disk"][:] = 1
    e["energy"][:] = [95, 120, 94.999, 100, 100]
    e["radius"][:] = [449.999, 100, 200, 450, 200]
    e["cosine"][:] = [.7, .8, .8, .8, .6999]
    _, report = run(e, s)
    group = report["entry_disk"]["1"]
    assert group["birks_sum"] == 29
    assert group["weak"] == 3
    assert group["other_disk"]["birks_sum"] == 24
    assert group["other_disk"]["css_rows"] == 3
    assert group["actual_css_disk"]["1"]["birks_sum"] == 5
    assert group["actual_css_disk"]["1"]["empty"] == 3
    assert group["empty"] == 1
    stratum = report["highEinnerdisk1"]
    assert stratum["counts"] == 2  # diagnostic includes outside bins, no extra cut
    assert stratum["birks_sum"] == 26
    assert stratum["weak"] == 0
    assert report["out_of_bins"]["counts"] == 1


def test_every_cell_and_descriptive_thresholds():
    occupancy = np.array([0, 1, 8, 9, 99, 100, 199, 200, 399, 400, 505] + [1] * 85)
    labels = np.indices((2, 4, 3, 4)).reshape(4, -1).T
    expanded = np.repeat(labels, occupancy, axis=0)
    e, s = make_case(len(expanded))
    e["disk"] = expanded[:, 0]
    e["energy"] = np.array([20., 60., 90., 100.])[expanded[:, 1]]
    e["cosine"] = np.array([.2, .5, .8])[expanded[:, 2]]
    e["radius"] = np.array([200., 500., 600., 900.])[expanded[:, 3]]
    arrays, report = run(e, s)
    np.testing.assert_array_equal(arrays["counts"], occupancy)
    np.testing.assert_array_equal(arrays["empty"], occupancy)
    assert report["thresholds"] == dict(cells=96, occupied=95, gt8=8, ge100=6, ge200=4, ge400=2)
    assert report["thresholds_entry_disk"]["1"]["occupied"] == 48


@pytest.mark.parametrize("n", [0, 1, 6])
def test_empty_and_no_step_inputs(n):
    e, s = make_case(n)
    arrays, report = run(e, s)
    assert len(arrays["counts"]) == 96
    assert arrays["birks_sum"].dtype == np.float64
    assert arrays["counts"].sum() == n
    assert arrays["empty"].sum() == arrays["zero_energy"].sum() == arrays["weak"].sum() == n
    assert arrays["positive_birks"].sum() == arrays["css_rows"].sum() == arrays["birks_sum"].sum() == 0
    assert report["front"]["counts"] == n
    fa, fr = coverage.frozen(packed(e, s))
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], fa[name])
    assert fr["front"] == report["front"]
    assert fr["stored_events"] is None and fr["non_front_entry_events"] is None
    json.dumps(fr, allow_nan=False)


def test_frozen_readonly_mapping_preserved(mixed):
    e, s = mixed
    lib = packed(e, s)
    before = deepcopy(lib)
    for value in lib.values():
        value.flags.writeable = False
    arrays, report = coverage.frozen(lib)
    expected, raw_report = run(e, s)
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], expected[name])
    assert report["front"] == raw_report["front"]
    for name in lib:
        np.testing.assert_array_equal(lib[name], before[name])
        assert not lib[name].flags.writeable


@pytest.mark.parametrize("offsets", [np.array([1, 1, 1]), np.array([0, 2, 1]), np.array([0, 0, 2]),
                                     np.array([0, 1]), np.array([0., 0., 1.]), np.array([0, -1, 1]),
                                     np.array([0, 2, 1], dtype=np.uint64)])
def test_invalid_frozen_offsets(offsets):
    e, s = make_case(2, [(1, 0, 1)])
    lib = packed(e, s)
    lib["offsets"] = offsets
    with pytest.raises(ValueError, match="offsets"):
        coverage.frozen(lib)


@pytest.mark.parametrize("field,value", [("bins_energy", np.array([0., 50., 80., 95., 111.])),
                                         ("entry_cell", np.zeros((2, 3), int)),
                                         ("entry_outside_bins", np.ones(2, bool)),
                                         ("entry_outside_bins", np.zeros(2, int)),
                                         ("step_relative_position", np.zeros((1, 2))),
                                         ("step_relative_time", np.array([np.nan]))])
def test_frozen_saved_schema_validation(field, value):
    e, s = make_case(2, [(1, 0, 1)])
    lib = packed(e, s)
    lib[field] = value
    with pytest.raises(ValueError):
        coverage.frozen(lib)


@pytest.mark.parametrize("table,field,value", [
    ("entry", "event_row", np.array([0, 0])), ("entry", "event_row", np.array([0., 1.])),
    ("entry", "event_row", np.array([-1, 1])), ("entry", "event_row", np.array([0, 2])),
    ("entry", "disk", np.array([0., 1.])), ("entry", "disk", np.array([0, 2])),
    ("entry", "disk", np.array([0, -1])), ("entry", "energy", np.array([100., np.nan])),
    ("entry", "energy", np.array([-1., 100.])), ("entry", "radius", np.array([-1., 100.])),
    ("entry", "cosine", np.array([.8, 1.000001])), ("entry", "cosine", np.array([-.1, .8])),
    ("entry", "position", np.zeros((2, 2))), ("entry", "direction", np.zeros((2, 3))),
    ("entry", "time", np.array([np.inf, 0])), ("entry", "energy", np.ones((2, 1))),
    ("step", "entry", np.array([2])), ("step", "entry", np.array([-1])),
    ("step", "entry", np.array([0.])), ("step", "disk", np.array([2])),
    ("step", "disk", np.array([0.])), ("step", "energy_birks", np.array([-1.])),
    ("step", "energy_birks", np.array([np.nan])), ("step", "energy_g4", np.array([-1.])),
    ("step", "energy_g4", np.array([np.inf])), ("step", "crystal", np.array([1.5])),
    ("step", "position", np.full((1, 3), np.nan)), ("step", "energy_birks", np.array(["1"])),
])
def test_input_validation(table, field, value):
    e, s = make_case(2, [(0, 0, 1)])
    original, total = stored(e, s)
    (e if table == "entry" else s)[field] = value
    with pytest.raises(ValueError):
        coverage.census(e, s, original, total)


@pytest.mark.parametrize("field,value", [("events", 3), ("front_entry_events", 1),
                                         ("non_front_entry_events", 1), ("retained_css_rows", 0),
                                         ("css_rows", 0), ("empty_css_events", 3),
                                         ("non_front_entry_positive_events", 1), ("events", 2.)])
def test_original_census_consistency(field, value):
    e, s = make_case(2, [(0, 0, 1)])
    original, total = stored(e, s)
    original[field] = value
    with pytest.raises(ValueError):
        coverage.census(e, s, original, total)


@pytest.mark.parametrize("value", [np.array([np.nan, 0]), np.array([-1., 0]), np.array([2., 0]), np.zeros(3)])
def test_stored_total_validation(value):
    e, s = make_case(2, [(0, 0, 1)])
    original, _ = stored(e, s)
    with pytest.raises(ValueError):
        coverage.census(e, s, original, value)


def test_noncontiguous_reordered_stored_event_rows():
    e, s = make_case(2, [(0, 1, 12)])
    e["event_row"] = np.array([3, 1], np.uint32)
    original = dict(events=5, front_entry_events=2, non_front_entry_events=3,
                    empty_css_events=3, non_front_entry_positive_events=1,
                    css_rows=2, retained_css_rows=1)
    arrays, report = coverage.census(e, s, original, np.array([0., 0., 8., 12., 0.]))
    assert arrays["birks_sum"].sum() == 12
    assert report["front"]["empty"] == 1
    assert report["non_front_entry_events"] == 3


def test_compare_all_cells_new_lost_without_merging(mixed):
    e, s = mixed
    training, _ = run(e, s)
    old_e, old_s = make_case(3, [(0, 0, 1)])
    old_e["energy"][:] = 60
    legacy, _ = run(old_e, old_s)
    before = deepcopy(training)
    report = coverage.compare(training, legacy)
    assert report["cell_count"] == len(report["cells"]) == 96
    assert report["training_entries_in_bins"] == 8
    assert report["legacy_entries_in_bins"] == 3
    assert report["lost_count"] == 2
    assert report["newly_populated_count"] == int(np.sum(training["counts"] > 0))
    for i, cell in enumerate(report["cells"]):
        assert cell["training_counts"] == training["counts"][i]
        assert cell["legacy_counts"] == legacy["counts"][i]
        assert cell["count_delta"] == training["counts"][i] - legacy["counts"][i]
    for name in training:
        np.testing.assert_array_equal(training[name], before[name])
    assert "confounded" in report["caveat"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field", ["counts", "energy_bin", "weak", "birks_sum"])
def test_compare_validation(field):
    e, s = make_case(2)
    a, _ = run(e, s)
    b = deepcopy(a)
    if field == "energy_bin":
        b[field] = b[field][::-1]
    elif field == "weak":
        b[field][:] = 3
    elif field == "birks_sum":
        b[field][0] = np.nan
    else:
        b[field] = b[field].astype(float)
    with pytest.raises(ValueError):
        coverage.compare(a, b)
