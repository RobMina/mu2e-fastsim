"""Synthetic-only scalar pool/payload and RNG-schedule checks of the opt-in sampler."""
from copy import deepcopy
import inspect
import math

import numpy as np
import pytest
import uproot

from calofastsim.showers import direction_sampler as candidate
from calofastsim.showers import library as original
from calofastsim.showers import support_audit as sa
from test_showers_support_audit import assert_same, entries, library, oracle as baseline_oracle


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic tests must not read data or use real random generators")
    for module, names in ((np, ("load",)), (uproot, ("open", "iterate", "concatenate")),
                          (np.random, ("default_rng", "RandomState", "Generator", "random", "random_sample",
                                       "rand", "randn", "choice", "seed", "uniform"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(original, "build", forbidden)


class SyntheticRNG:
    """An explicit finite uniform tape, not a random generator or real-data draw."""

    def __init__(self, values=None):
        self.values = np.asarray(values if values is not None else (np.arange(1000) * 37 % 101) / 101., float)
        self.calls = []
        self.cursor = 0

    def random(self, size):
        self.calls.append(size)
        end = self.cursor + size
        assert end <= len(self.values)
        value = self.values[self.cursor:end].copy()
        self.cursor = end
        return value


def scalar_local(lib, table, prefix=""):
    result = []
    for row, disk in enumerate(table[prefix + "disk"]):
        x = float(table[prefix + "position"][row, 0] - lib["geometry_disk_origin_x"][disk])
        y = float(table[prefix + "position"][row, 1] - lib["geometry_disk_origin_y"][disk])
        norm = math.hypot(x, y)
        rx, ry = (x / norm, y / norm) if norm else (1., 0.)
        ux, uy, uz = map(float, table[prefix + "direction"][row])
        result.append((rx * ux + ry * uy, rx * uy - ry * ux, uz))
    return result


def scalar_pools(lib, query):
    pools = baseline_oracle(lib, query)
    q, d = scalar_local(lib, query), scalar_local(lib, lib, "entry_")
    count = np.zeros(len(q), np.int64)
    used = np.zeros(len(q), bool)
    for i, direction in enumerate(q):
        if not pools["supported"][i]:
            continue
        accepted = []
        for row, local in enumerate(d):
            if pools["library_cell"][row] != pools["chosen_cell"][i]:
                continue
            distance = abs(float(lib["entry_radius"][row]) - float(query["radius"][i]))
            same_sign = all((direction[a] >= 0) == (local[a] >= 0) for a in (0, 1))
            angle = math.degrees(math.acos(min(1., max(-1., sum(a * b for a, b in zip(direction, local))))))
            if same_sign and angle <= 10. and distance <= 25.:
                accepted.append(row)
        count[i] = len(accepted)
        if len(accepted) >= 8:
            used[i] = True
            nearest = sorted(accepted, key=lambda row: (abs(float(lib["entry_radius"][row]) - float(query["radius"][i])), row))[:8]
            pools["eligible"][i] = sorted(nearest)
            pools["size"][i] = 8
    pools.update(direction_count=count, direction_used=used, direction_fallback=pools["supported"] & ~used)
    return pools


def check_pools(lib, query):
    pools = candidate.enumerate_pools(lib, query)
    assert_same(pools, scalar_pools(lib, query))
    base = sa.enumerate_pools(lib, query)
    assert set(pools) == set(base) | {"direction_count", "direction_used", "direction_fallback"}
    for key in base.keys() - {"eligible", "size"}:
        np.testing.assert_array_equal(pools[key], base[key], err_msg=key)
    return pools


def scalar_payload(lib, query, draw):
    source, target, positions, times = [], [], [], []
    for i, donor in enumerate(draw["donor"]):
        if donor < 0:
            continue
        z = np.asarray(lib["entry_direction"][donor], float)
        z = z / math.sqrt(sum(float(v) ** 2 for v in z))
        ref = np.array([1., 0., 0.]) if abs(z[2]) > .9 else np.array([0., 0., 1.])
        x = np.cross(ref, z)
        x /= math.sqrt(sum(float(v) ** 2 for v in x))
        y = np.cross(z, x)
        disk = query["disk"][i]
        origin = np.array([lib["geometry_disk_origin_" + a][disk] for a in "xy"])
        sx, sy = lib["entry_position"][donor, :2] - origin
        tx, ty = query["position"][i, :2] - origin
        angle = math.atan2(ty, tx) - math.atan2(sy, sx)
        for step in range(lib["offsets"][donor], lib["offsets"][donor + 1]):
            a, b, c = lib["step_relative_position"][step]
            delta = a * x + b * y + c * z
            dx, dy, dz = delta
            rotated = [math.cos(angle) * dx - math.sin(angle) * dy,
                       math.sin(angle) * dx + math.cos(angle) * dy, dz]
            source.append(step)
            target.append(i)
            positions.append(query["position"][i] + rotated)
            times.append(query["time"][i] + lib["step_relative_time"][step])
    source = np.asarray(source, np.int64)
    np.testing.assert_array_equal(draw["entry"], target)
    np.testing.assert_allclose(draw["position"], np.asarray(positions).reshape(-1, 3), atol=1e-13, rtol=1e-14)
    np.testing.assert_array_equal(draw["time"], times)
    for key in ("energy_birks", "energy_g4", "disk", "crystal", "tag"):
        if "step_" + key in lib:
            np.testing.assert_array_equal(draw[key], lib["step_" + key][source])
    length = [0 if d < 0 else lib["offsets"][d + 1] - lib["offsets"][d] for d in draw["donor"]]
    np.testing.assert_array_equal(draw["offsets"], np.r_[0, np.cumsum(length)])


def test_fixed_policy_ties_signs_cap_sparse_and_same_chosen_cell():
    radii = [100.] * 8 + [125., 75., 110., 90., 120., 80., 105., 95., 125., 125.000001]
    lib = library(radii + [100.] * 8 + [100.] * 3,
                  disk=[0] * 26 + [1] * 3, energy=[100.] * 18 + [75.] * 8 + [25.] * 3)
    lib["entry_direction"][:8] = [-.6, 0., .8]
    query = entries([100., 100., 100., 500., 100., 100., 1100.],
                    disk=[0, 0, 0, 0, 1, 0, 0], energy=[100., 100., 100., 100., 25., 85., 120.])
    query["direction"][1] = [-.6, 0., .8]
    query["direction"][2] = [.6, -1e-12, .8]
    pools = check_pools(lib, query)
    assert pools["direction_count"].tolist() == [9, 8, 0, 0, 3, 8, 0]
    np.testing.assert_array_equal(pools["eligible"][0], np.arange(8, 16))
    assert pools["direction_used"].tolist() == [True, True, False, False, False, True, False]
    assert pools["fallback"][3] and pools["fallback"][5] and pools["outside"][6]
    assert not pools["direction_fallback"][5]
    for i in (2, 3, 4, 6):
        np.testing.assert_array_equal(pools["eligible"][i], baseline_oracle(lib, query)["eligible"][i])


@pytest.mark.parametrize("radius,used", [(75., True), (125., True), (74.999999, False), (125.000001, False)])
@pytest.mark.parametrize("theta,angular", [(0., True), (9.999999, True), (10.000001, False)])
def test_inclusive_radius_and_angular_boundary(radius, used, theta, angular):
    lib, query = library([radius] * 8), entries([100.])
    query["direction"][:] = [0., 0., 1.]
    lib["entry_direction"][:] = [math.sin(math.radians(theta)), 0., math.cos(math.radians(theta))]
    pools = check_pools(lib, query)
    assert bool(pools["direction_used"][0]) == (used and angular)
    assert candidate.K == 8 and candidate.ANGLE_DEG == 10. and candidate.RADIUS_MM == 25.


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("component,accepted", [(-0., True), (0., True), (1e-12, True), (-1e-12, False)])
def test_zero_signs_and_axis_convention(axis, component, accepted):
    lib, query = library([0.] * 8), entries([0.])
    query["direction"][:] = [0., 0., 1.]
    lib["entry_direction"][:] = [0., 0., 1.]
    lib["entry_direction"][:, axis] = component
    pools = check_pools(lib, query)
    assert bool(pools["direction_used"][0]) is accepted


@pytest.mark.parametrize("donors", [0, 1, 7, 8, 9, 31])
@pytest.mark.parametrize("queries", [0, 5])
def test_empty_sparse_missing_disk_and_base_payload_parity(donors, queries):
    lib = library([100.] * donors, lengths=[i % 4 for i in range(donors)])
    query = entries([100.] * queries, disk=np.arange(queries) % 2)
    pools = check_pools(lib, query)
    a, b = SyntheticRNG(), SyntheticRNG()
    draw = candidate.sample(lib, query, a)
    base = original.sample(lib, query, b, radius_neighbors=8, orientation="disk")
    assert a.calls == b.calls and a.cursor == b.cursor
    assert_same({k: draw[k] for k in base}, base)
    for key in ("direction_count", "direction_used", "direction_fallback"):
        np.testing.assert_array_equal(draw[key], pools[key])
    assert not draw["direction_fallback"][~draw["supported"]].any()
    scalar_payload(lib, query, draw)


def test_rng_schedule_changed_direction_pools_and_unchanged_fallback_payload():
    lib = library([100.] * 16 + [100.] * 16, disk=[0] * 16 + [1] * 16,
                  lengths=[0, 1, 3, 2] * 8)
    lib["entry_direction"][np.r_[0:8, 16:24]] = [-.6, 0., .8]
    m = len(lib["step_energy_birks"])
    lib["step_relative_position"][:] = (np.arange(m * 3).reshape(m, 3) - 30.) / 17.
    lib["step_relative_time"][:] = np.arange(m) / 13. - 5.
    lib["step_energy_birks"][:] = np.arange(m) / 20.
    lib["step_energy_g4"][:] = np.arange(m) / 7.
    lib["step_tag"] = np.arange(m) + 1234
    query = entries([100.] * 8, disk=[1, 0, 1, 0, 1, 0, 1, 0])
    query["direction"][[1, 4]] = [.6, -1e-12, .8]
    query["position"][:, :2] = [0., 100.]
    for i in range(8):
        radial, tangent, z = query["direction"][i]
        query["direction"][i] = [-tangent, radial, z]
    pools = check_pools(lib, query)
    a, b = SyntheticRNG([0., .15, .4, .999, .01, .3, .5, .999]), SyntheticRNG([0., .15, .4, .999, .01, .3, .5, .999])
    draw = candidate.sample(lib, query, a)
    base = original.sample(lib, query, b, radius_neighbors=8, orientation="disk")
    assert a.calls == b.calls == [4, 4] and a.cursor == b.cursor == 8
    expected = np.full(8, -1, np.int64)
    cursor = 0
    for disk in (0, 1):
        for i in np.flatnonzero(query["disk"] == disk):
            expected[i] = pools["eligible"][i, int(a.values[cursor] * pools["size"][i])]
            cursor += 1
    np.testing.assert_array_equal(draw["donor"], expected)
    assert np.all(draw["donor"][pools["direction_used"]] != base["donor"][pools["direction_used"]])
    for i in np.flatnonzero(pools["direction_fallback"]):
        assert draw["donor"][i] == base["donor"][i]
        left = slice(draw["offsets"][i], draw["offsets"][i + 1])
        right = slice(base["offsets"][i], base["offsets"][i + 1])
        for key in ("position", "time", "energy_birks", "energy_g4", "disk", "crystal", "tag"):
            np.testing.assert_array_equal(draw[key][left], base[key][right])
    scalar_payload(lib, query, draw)
    assert set(draw) == set(base) | {"direction_count", "direction_used", "direction_fallback"}


def test_empty_weak_both_disk_donors_retained_without_rescale_or_outcome_filter():
    lib = library([100.] * 8, lengths=[0, 2, 2, 1, 1, 1, 1, 1], response=[0., 0., 4., 6., 1., 20., 30., 40., 50.])
    query = entries([100.] * 8, energy=[109.] * 8)
    draw = candidate.sample(lib, query, SyntheticRNG((np.arange(8) + .5) / 8))
    np.testing.assert_array_equal(draw["donor"], np.arange(8))
    np.testing.assert_array_equal(draw["offsets"], lib["offsets"])
    assert draw["supported"].all() and draw["direction_used"].all()
    assert draw["offsets"][0] == draw["offsets"][1]
    assert set(draw["disk"][draw["entry"] == 2]) == {0, 1}
    scalar_payload(lib, query, draw)


def test_readonly_translation_azimuth_and_default_invariance(monkeypatch):
    lib, query = library([100.] * 8), entries([105., 110.])
    lib["geometry_disk_origin_x"][:] = [30., -70.]
    lib["geometry_disk_origin_y"][:] = [-15., 45.]
    lib["entry_position"][:, :2] = [130., -15.]
    query["position"][:, :2] = [[30., 90.], [-80., -15.]]
    query["direction"][:] = [[0., .5, math.sqrt(.75)], [-.5, 0., math.sqrt(.75)]]
    lib["step_relative_position"][:] = [2., 3., 4.]
    lib["step_relative_time"][:] = -2.
    before = deepcopy((lib, query))
    signature = inspect.signature(original.sample)
    assert signature.parameters["orientation"].default == "incident"
    assert signature.parameters["radius_neighbors"].default == 0
    for table in (lib, query):
        for value in table.values():
            value.flags.writeable = False
    a, b = SyntheticRNG(), SyntheticRNG()
    base = original.sample(lib, query, b, radius_neighbors=8, orientation="disk")
    def forbidden(*args, **kwargs):
        raise AssertionError("opt-in sampler must enumerate and draw once, not call the original sampler")
    monkeypatch.setattr(original, "sample", forbidden)
    draw = candidate.sample(lib, query, a)
    assert_same({key: draw[key] for key in base}, base)
    scalar_payload(lib, query, draw)
    for actual, expected in zip((lib, query), before):
        assert_same(actual, expected)
        assert all(not value.flags.writeable for value in actual.values())


@pytest.mark.parametrize("table,field", [("query", k) for k in ("energy", "cosine", "radius", "time", "position", "direction")] +
                         [("library", k) for k in ("entry_direction", "entry_radius", "entry_position", "step_relative_position",
                                                    "step_relative_time", "step_energy_birks", "step_energy_g4", "geometry_disk_origin_x")])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_before_rng(table, field, bad):
    lib, query = library([100.] * 8), entries([100.])
    (lib if table == "library" else query)[field].flat[0] = bad
    rng = SyntheticRNG()
    with pytest.raises(ValueError):
        candidate.enumerate_pools(lib, query)
    with pytest.raises(ValueError):
        candidate.sample(lib, query, rng)
    assert rng.calls == []


def test_no_direction_driven_cell_search_even_when_neighbor_cell_has_eight():
    lib = library([100.] * 8 + [100.] * 8, energy=[75.] * 8 + [100.] * 8)
    lib["entry_direction"][:8] = [-.6, 0., .8]
    query = entries([100.], energy=[85.])
    pools = check_pools(lib, query)
    assert pools["direction_count"].tolist() == [0]
    assert pools["direction_fallback"].tolist() == [True]
    assert pools["fallback"].tolist() == [True]
    np.testing.assert_array_equal(pools["eligible"][0], np.arange(8))


def test_nearest_radius_then_row_ties_then_final_library_order():
    lib = library([100.] * 8 + [120., 80., 115., 85., 110., 90., 105., 95., 120., 80., 100.])
    lib["entry_direction"][:8] = [-.6, 0., .8]
    pools = check_pools(lib, entries([100.]))
    assert pools["direction_count"].tolist() == [11]
    np.testing.assert_array_equal(pools["eligible"][0], [8, 10, 11, 12, 13, 14, 15, 18])


def test_smallest_representable_steps_straddling_ten_degree_threshold():
    boundary = math.cos(math.radians(10.))
    lib, query = library([100.] * 16), entries([100.])
    query["direction"][:] = [0., 0., 1.]
    for rows, cosine in ((slice(0, 8), np.nextafter(boundary, np.inf)),
                         (slice(8, 16), np.nextafter(boundary, -np.inf))):
        lib["entry_direction"][rows] = [math.sqrt(1. - cosine ** 2), 0., cosine]
    pools = check_pools(lib, query)
    assert pools["direction_count"].tolist() == [8]
    np.testing.assert_array_equal(pools["eligible"][0], np.arange(8))


def test_enumerates_once_and_leaves_default_sampler_unchanged(monkeypatch):
    lib, query = library([100.] * 16), entries([100., 101., 100.], disk=[0, 0, 1])
    default_before = original.sample(lib, query, SyntheticRNG())
    actual_enumerate = candidate.enumerate_pools
    calls = []
    def enumerate_once(library, entries):
        calls.append(1)
        return actual_enumerate(library, entries)
    monkeypatch.setattr(candidate, "enumerate_pools", enumerate_once)
    rng = SyntheticRNG()
    candidate.sample(lib, query, rng)
    assert calls == [1] and rng.calls == [2]
    assert_same(original.sample(lib, query, SyntheticRNG()), default_before)
    assert "radius_pool_size" not in default_before
    assert "direction_used" not in default_before


def test_float32_payload_arithmetic_is_bit_identical_to_original():
    lib, query = library([100.] * 8), entries([100., 105.])
    lib["entry_direction"][:] = [-.6, 0., .8]
    lib["step_relative_position"][:] = [1. / 3., -2. / 7., 9. / 13.]
    lib["step_relative_time"][:] = -1. / 7.
    query["position"][:, :2] = [[0., 100.], [-105., 0.]]
    query["direction"][:] = [[0., .6, .8], [-.6, 0., .8]]
    lib = {k: v.astype(np.float32) if v.dtype.kind == "f" and not k.startswith("bins_") else v for k, v in lib.items()}
    query = {k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in query.items()}
    a, b = SyntheticRNG(), SyntheticRNG()
    draw = candidate.sample(lib, query, a)
    base = original.sample(lib, query, b, radius_neighbors=8, orientation="disk")
    assert draw["direction_fallback"].all()
    assert_same({key: draw[key] for key in base}, base)
    assert a.calls == b.calls
