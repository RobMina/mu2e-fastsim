"""Synthetic fixed-K8 support checks; no data files, sampling or reconstruction."""
from copy import deepcopy
import json

import numpy as np
import pytest

from calofastsim.showers import library as sampler
from calofastsim.showers import pool_census as legacy
from calofastsim.showers import support_audit as sa
from calofastsim.showers.compare import paired_mean


@pytest.fixture(autouse=True)
def no_random_or_sample(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic support audit attempted sampling or RNG")
    monkeypatch.setattr(sampler, "sample", forbidden)
    for name in ("default_rng", "RandomState", "Generator", "random", "random_sample", "rand", "randn", "choice", "seed", "uniform"):
        monkeypatch.setattr(np.random, name, forbidden)


def entries(radius, disk=None, energy=None, cosine=None):
    radius = np.asarray(radius, float)
    n = len(radius)
    return dict(radius=radius, disk=np.zeros(n, np.int64) if disk is None else np.asarray(disk, np.int64),
                energy=np.full(n, 100.) if energy is None else np.asarray(energy, float),
                cosine=np.full(n, .8) if cosine is None else np.asarray(cosine, float),
                time=np.arange(n, dtype=float), event_row=101 * np.arange(n),
                position=np.column_stack((radius, np.zeros(n), np.zeros(n))),
                direction=np.tile([.6, 0., .8], (n, 1)))


def library(radius, disk=None, energy=None, cosine=None, lengths=None, response=None):
    e = entries(radius, disk, energy, cosine)
    n = len(radius)
    lengths = np.ones(n, np.int64) if lengths is None else np.asarray(lengths, np.int64)
    row = np.repeat(np.arange(n), lengths)
    m = len(row)
    lib = {"entry_" + k: v.copy() for k, v in e.items()}
    lib.update(offsets=np.r_[0, lengths.cumsum()], step_energy_birks=np.full(m, 20.) if response is None else np.asarray(response, float),
               step_energy_g4=np.full(m, 30.), step_relative_position=np.zeros((m, 3)), step_relative_time=np.zeros(m),
               step_disk=np.arange(m) % 2, step_crystal=np.arange(m), geometry_disk_disk=np.array([0, 1]))
    for name, edge in zip(("energy", "cosine", "radius"), sampler.Binning().arrays()):
        lib["bins_" + name] = edge
    lib["entry_cell"], lib["entry_outside_bins"] = sampler.cells(e["energy"], e["cosine"], e["radius"], sampler.Binning())
    for a in "xyz":
        lib["geometry_disk_origin_" + a] = np.zeros(2)
        lib["geometry_disk_axis_z_" + a] = np.full(2, float(a == "z"))
    return lib


def original(n):
    return dict(entry=np.arange(n), energy_birks=np.arange(n, dtype=float) * 15.)


def oracle(lib, query):
    edges = [lib["bins_" + k] for k in ("energy", "cosine", "radius")]
    def coords(table, prefix=""):
        values = zip(*(table[prefix + k] for k in ("energy", "cosine", "radius")))
        return [tuple(min(max(sum(x >= edge for edge in e) - 1, 0), len(e) - 2)
                      for x, e in zip(row, edges)) for row in values]
    qcell, dcell = coords(query), coords(lib, "entry_")
    labels = [(int(d), *c) for d, c in zip(lib["entry_disk"], dcell)]
    occupied = sorted(set(labels))
    inverse = np.array([occupied.index(label) for label in labels], np.int64)
    n = len(qcell)
    eligible = np.full((n, 8), -1, np.int64)
    chosen = np.full(n, -1, np.int64)
    size, full, exact = (np.zeros(n, np.int64) for _ in range(3))
    outside = np.array([any(x < e[0] or x > e[-1] for x, e in zip(row, edges))
                        for row in zip(*(query[k] for k in ("energy", "cosine", "radius")))], bool)
    fallback = outside.copy()
    for i, (d, cell, radius) in enumerate(zip(query["disk"], qcell, query["radius"])):
        candidates = [j for j, label in enumerate(occupied) if label[0] == d]
        if not candidates:
            fallback[i] = True
            continue
        j = min(candidates, key=lambda j: (sum((a - b) ** 2 for a, b in zip(cell, occupied[j][1:])), occupied[j]))
        chosen[i] = j
        rows = [r for r, label in enumerate(labels) if label == occupied[j]]
        full[i], size[i] = len(rows), min(8, len(rows))
        exact[i] = len(rows) if tuple(cell) == occupied[j][1:] else 0
        fallback[i] |= exact[i] == 0
        near = sorted(rows, key=lambda r: (abs(lib["entry_radius"][r] - radius), r))[:8]
        eligible[i, :len(near)] = sorted(near)
    return dict(eligible=eligible, query_cell=np.column_stack((query["disk"], np.array(qcell).reshape(-1, 3))).astype(np.int64),
                chosen_cell=chosen, occupied=np.array(occupied, np.int64).reshape(-1, 4), library_cell=inverse,
                counts=np.bincount(inverse, minlength=len(occupied)), size=size, full_size=full, exact_size=exact,
                supported=chosen >= 0, fallback=fallback, outside=outside, substituted=(chosen >= 0) & (exact == 0))


def assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 17, 101])
def test_oracle_pool_sizes_ties_missing_disk_and_legacy_parity(n):
    lib = library(100. + (np.arange(n) % 5) * 10.)
    query = entries([0., 100., 105., 120., 449., 450., 1000., 1100., 100.], disk=[0] * 8 + [1],
                    energy=[100., 100., 100., 100., 94., 100., 110., 120., 100.])
    pools = sa.enumerate_pools(lib, query)
    assert_same(pools, oracle(lib, query))
    saved = dict(supported=pools["supported"], fallback=pools["fallback"], radius_pool_size=pools["size"],
                 bin_pool_size=pools["full_size"], donor=pools["eligible"][:, 0])
    assert_same(pools, legacy.enumerate_pools(lib, query, saved))


def test_locality_matches_existing_saved_metric_without_renormalization():
    lib = library([100., 110.])
    query = entries([120.])
    lib["entry_direction"] *= 1. - 2e-10
    pools = sa.enumerate_pools(lib, query)
    expected = legacy.coverage(lib, query, pools)
    _, arrays = sa.audit(lib, query, original(1))
    for name, values in expected.items():
        np.testing.assert_array_equal(arrays["pool_" + name], values)


def test_lexicographic_cell_tie_and_library_row_radius_ties():
    lib = library([100.] * 10 + [100.] * 10, energy=[75.] * 10 + [100.] * 10)
    query = entries([100.], energy=[85.])
    pools = sa.enumerate_pools(lib, query)
    assert_same(pools, oracle(lib, query))
    assert pools["substituted"].tolist() == [True]
    np.testing.assert_array_equal(pools["eligible"][0], np.arange(8))
    np.testing.assert_array_equal(pools["occupied"][pools["chosen_cell"][0]], [0, 1, 2, 0])
    lib = library([100.] * 10 + [120.] * 10)
    pools = sa.enumerate_pools(lib, entries([110., 120.]))
    np.testing.assert_array_equal(pools["eligible"][0], np.arange(8))
    np.testing.assert_array_equal(pools["eligible"][1], np.arange(10, 18))


def test_exact_probabilities_exposure_outcomes_and_zero_step_preservation():
    lib = library([100., 110., 120., 130.], lengths=[0, 1, 2, 1], response=[0., 4., 6., 30.])
    query = entries([100., 120., 130., 100.], disk=[0, 0, 0, 1])
    css = dict(entry=np.array([1, 2, 3]), energy_birks=np.array([5., 20., 999.]))
    report, arrays = sa.audit(lib, query, css)
    np.testing.assert_array_equal(arrays["outcome_empty"], [True, False, False, False])
    np.testing.assert_array_equal(arrays["outcome_weak"], [True, True, True, False])
    np.testing.assert_allclose(arrays["prob_empty"], [.25, .25, .25, np.nan])
    np.testing.assert_allclose(arrays["prob_weak"], [.75, .75, .75, np.nan])
    np.testing.assert_allclose(arrays["prob_energy"], [10., 10., 10., np.nan])
    np.testing.assert_array_equal(arrays["original_empty"], [True, False, False, False])
    region = report["regions"]["all"]
    assert (region["queries"], region["supported"], region["unsupported"]) == (4, 3, 1)
    empty = region["outcomes"]["empty"]
    assert empty["expected_count"] == .75
    assert empty["original"] == 1
    assert empty["conditional_draw_sd"] == .75
    assert region["outcomes"]["energy"]["original_mean"] == pytest.approx(25 / 3)
    assert region["outcomes"]["energy"]["expected_minus_original"] == paired_mean(np.full(3, 10.), [0, 5, 20], query["event_row"][:3] // 100)
    np.testing.assert_allclose(arrays["exposure_all"], [.75] * 4)
    assert region["expected_unique_drawn"] == pytest.approx(4 * (1 - .75 ** 3))
    assert region["unique_eligible_donors"] == 4
    assert region["eligible_empty"] == 1 and region["eligible_weak"] == 3
    assert report["regions"]["disk1"]["outcomes"]["energy"]["expected_minus_original"]["difference"] is None
    assert "inner_highE_disk1_coshigh" in report["regions"]
    json.dumps(report, allow_nan=False)


def test_all_96_cells_outside_clipped_and_terminal_edge_included():
    lib = library([1100., 1000.], disk=[1, 1], energy=[120., 110.])
    query = entries([1100., 1000., 200.], disk=[1, 1, 0], energy=[120., 110., 100.])
    report, arrays = sa.audit(lib, query, original(3))
    assert len(report["cells"]) == 96
    assert sum(c["queries"] for c in report["cells"]) == 3
    assert sum(c["donors"] for c in report["cells"]) == 2
    assert sum(c["outside_queries"] for c in report["cells"]) == 1
    assert sum(c["outside_donors"] for c in report["cells"]) == 1
    assert sum(c["unsupported_queries"] for c in report["cells"]) == 1
    np.testing.assert_array_equal(arrays["outside"], [True, False, False])
    np.testing.assert_array_equal(arrays["fallback"], [True, False, True])
    np.testing.assert_array_equal(arrays["substituted"], [False, False, False])
    assert arrays["exact_size"][0] == 2
    assert "clipped" in report["bin_convention"] and "NOT excluded" in report["caveat"]


@pytest.mark.parametrize("nquery", [0, 3])
@pytest.mark.parametrize("ndonor", [0, 1])
def test_empty_queries_or_library(nquery, ndonor):
    lib, query = library([100.] * ndonor), entries([100.] * nquery)
    report, arrays = sa.audit(lib, query, original(nquery))
    assert arrays["eligible"].shape == (nquery, 8)
    assert sum(c["queries"] for c in report["cells"]) == nquery
    assert report["regions"]["all"]["expected_unique_drawn"] == (1. if nquery and ndonor else 0.)
    if not ndonor:
        assert np.isnan(arrays["prob_empty"]).all()
        assert np.isnan(arrays["pool_angle_deg"]).all()
        assert report["regions"]["all"]["unsupported"] == nquery
        assert report["regions"]["all"]["outcomes"]["empty"]["original"] == 0
    json.dumps(report, allow_nan=False)


def test_direction_rotation_signed_deltas_and_readonly_inputs():
    lib = library([100., 110.], energy=[100., 101.])
    query = entries([120.], energy=[105.], cosine=[.9])
    query["position"][0] = [0., 120., 0.]
    query["direction"][0] = [0., .6, .8]
    css = original(1)
    before = deepcopy((lib, query, css))
    for table in (lib, query, css):
        for value in table.values():
            value.flags.writeable = False
    report, arrays = sa.audit(lib, query, css)
    for actual, expected in zip((lib, query, css), before):
        assert_same(actual, expected)
        assert all(not v.flags.writeable for v in actual.values())
    np.testing.assert_allclose(arrays["pool_angle_deg"][0, :2], 0., atol=1e-6)
    np.testing.assert_allclose(arrays["pool_radius_delta"][0, :2], [-20., -10.])
    np.testing.assert_allclose(arrays["pool_energy_delta"][0, :2], [-5., -4.])
    np.testing.assert_allclose(arrays["pool_cosine_delta"][0, :2], [-.1, -.1])
    assert np.isnan(arrays["pool_radius_delta"][0, 2:]).all()
    coverage = report["regions"]["all"]["coverage"]["radius_delta"]
    assert [coverage[k]["mean"] for k in ("best", "worst", "mean")] == [10., 20., 15.]


@pytest.mark.parametrize("field,bad", [
    ("entry_energy", np.array([[100.], [100.]])), ("entry_energy", np.array([np.nan, 100.])),
    ("entry_disk", np.array([0., 0.])), ("entry_disk", np.array([0, 2])),
    ("entry_event_row", np.array([0, 0])), ("entry_event_row", np.array([0., 1.])),
    ("entry_direction", np.zeros((2, 3))), ("entry_direction", np.ones((2, 3))),
    ("entry_cosine", np.array([1.1, .8])), ("entry_radius", np.array([-1., 100.])),
    ("entry_cell", np.zeros((2, 3))), ("entry_outside_bins", np.array([True, False])),
    ("entry_outside_bins", np.array([0, 0])), ("offsets", np.array([0., 1., 2.])),
    ("offsets", np.array([1, 1, 2])), ("offsets", np.array([0, 2, 1])),
    ("offsets", np.array([0, 1, 3])), ("step_energy_birks", np.array([-1., 20.])),
    ("step_energy_g4", np.array([np.inf, 20.])), ("step_disk", np.array([0., 1.])),
    ("step_crystal", np.array([0., 1.])), ("step_relative_position", np.zeros((2, 2))),
    ("step_relative_time", np.array([np.nan, 0.])), ("bins_radius", np.array([0., 1000.])),
    ("geometry_disk_disk", np.array([1, 0])), ("geometry_disk_disk", np.array([0])),
    ("geometry_disk_origin_x", np.array([0., np.nan])), ("geometry_disk_axis_z_z", np.array([2., 1.])),
    ("geometry_disk_axis_z_x", np.array([1., 0.])),
])
def test_malformed_library_rejected(field, bad):
    lib = library([100., 110.])
    lib[field] = bad
    with pytest.raises(ValueError):
        sa.audit(lib, entries([100.]), original(1))


@pytest.mark.parametrize("field,bad", [("event_row", np.array([1, 1])), ("event_row", np.array([1., 2.])),
    ("disk", np.array([0, 2])), ("direction", np.full((2, 3), np.nan)), ("time", np.zeros(1)),
    ("position", np.zeros((2, 2))), ("energy", np.array([np.inf, 100.])),
    ("cosine", np.array([-.1, .8])), ("radius", np.array([-1., 100.]))])
def test_malformed_queries_rejected(field, bad):
    query = entries([100., 110.])
    query[field] = bad
    with pytest.raises(ValueError):
        sa.audit(library([100.]), query, original(2))


@pytest.mark.parametrize("css", [dict(entry=np.array([1]), energy_birks=np.array([10.])),
    dict(entry=np.array([0.]), energy_birks=np.array([10.])),
    dict(entry=np.array([-1]), energy_birks=np.array([10.])),
    dict(entry=np.array([0]), energy_birks=np.array([np.nan])),
    dict(entry=np.array([0]), energy_birks=np.array([-1.])),
    dict(entry=np.array([0]), energy_birks=np.array([1.]), disk=np.array([2])),
    dict(entry=np.array([0]), energy_birks=np.array([1.]), offsets=np.array([0., 1.])),
    dict(entry=np.array([0]), energy_birks=np.array([1.]), offsets=np.array([0, 0]))])
def test_malformed_original_css_rejected(css):
    with pytest.raises(ValueError):
        sa.audit(library([100.]), entries([100.]), css)


def test_k8_local_expectations_not_full_cell_and_single_group_error():
    lib = library(100. + 10 * np.arange(10), lengths=[1] * 8 + [0, 0])
    query = entries([100., 190.])
    query["event_row"] = np.array([1, 2])
    report, arrays = sa.audit(lib, query, original(2))
    np.testing.assert_allclose(arrays["prob_empty"], [0., .25])
    assert report["regions"]["all"]["outcomes"]["empty"]["expected_count"] == .25
    assert report["regions"]["all"]["outcomes"]["energy"]["expected_minus_original"]["error"] is None
    assert report["regions"]["all"]["count_distributions"]["size"] == {"8": 2}


def test_integer_valued_features_and_unsigned_ids():
    lib = library([100.] * 10)
    query = entries([100.])
    for table, prefix in ((lib, "entry_"), (query, "")):
        for name in ("radius", "energy", "position", "time"):
            table[prefix + name] = table[prefix + name].astype(np.int64)
        table[prefix + "direction"] = np.tile([0, 0, 1], (len(table[prefix + "energy"]), 1))
        table[prefix + "disk"] = table[prefix + "disk"].astype(np.uint64)
        table[prefix + "event_row"] = table[prefix + "event_row"].astype(np.uint64)
    lib["entry_cell"] = lib["entry_cell"].astype(np.uint64)
    report, arrays = sa.audit(lib, query, original(1))
    assert report["regions"]["all"]["supported"] == 1
    np.testing.assert_array_equal(arrays["eligible"][0], np.arange(8))
    np.testing.assert_allclose(arrays["pool_angle_deg"], 0.)


def test_original_packed_offsets_validate_order_and_both_disks():
    lib = library([100., 110.], lengths=[0, 2], response=[4., 9.])
    css = dict(entry=np.array([0, 0]), energy_birks=np.array([4., 9.]),
               offsets=np.array([0, 2, 2]), disk=np.array([0, 1]))
    _, arrays = sa.audit(lib, entries([100., 110.]), css)
    np.testing.assert_array_equal(arrays["original_energy"], [13., 0.])
    np.testing.assert_array_equal(arrays["original_empty"], [False, True])
    css["entry"] = np.array([1, 0])
    with pytest.raises(ValueError, match="offsets disagree"):
        sa.audit(lib, entries([100., 110.]), css)


def test_every_fixed_cell_queries_and_outside_original_accounting():
    coordinates = np.indices((2, 4, 3, 4)).reshape(4, -1)
    disk, energy, cosine, radius = coordinates
    query = entries(np.array([100., 500., 600., 800.])[radius], disk=disk,
                    energy=np.array([25., 65., 85., 100.])[energy], cosine=np.array([.2, .5, .8])[cosine])
    report, _ = sa.audit(library([100.]), query, original(96))
    assert all(cell["queries"] == 1 for cell in report["cells"])
    assert sum(cell["supported_queries"] for cell in report["cells"]) == 48
    assert sum(cell["unsupported_queries"] for cell in report["cells"]) == 48
    assert report["regions"]["all"]["queries"] == 96
    assert report["regions"]["disk1"]["queries"] == 48


def test_malformed_empty_library_payload_not_silently_unsupported():
    lib = library([])
    lib["step_energy_birks"] = np.array([1.])
    with pytest.raises(ValueError):
        sa.audit(lib, entries([100.]), original(1))
