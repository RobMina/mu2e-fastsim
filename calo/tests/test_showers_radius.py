"""Radius-local whole-shower draws against an independent dense reference."""
from copy import deepcopy

import numpy as np
import pytest

from calofastsim.showers import Binning, build, frame, sample
from calofastsim.showers.library import _radius_pool


BINNING = Binning(energy=(0., 50., 100., 150.), cosine=(0., .5, 1.000001),
                  radius=(0., 500., 1000.))


def make_entries(radius, disk=None, energy=None):
    radius = np.asarray(radius, dtype=float)
    n = len(radius)
    return dict(radius=radius, disk=np.zeros(n, dtype=int) if disk is None else np.asarray(disk),
                energy=np.full(n, 75.) if energy is None else np.asarray(energy, dtype=float),
                cosine=np.full(n, .8), time=np.arange(n, dtype=float) * 8,
                position=np.column_stack((radius, np.zeros(n), np.full(n, 20.))),
                direction=np.tile([0., 0., 1.], (n, 1)), event_row=np.arange(n))


def make_library(radius, disk=None, energy=None, empty=()):
    entries = make_entries(radius, disk, energy)
    n = len(radius)
    lengths = np.where(np.isin(np.arange(n), empty), 0, 2)
    row = np.repeat(np.arange(n), lengths)
    slot = np.tile([0, 1], np.count_nonzero(lengths))
    steps = dict(entry=row, energy_g4=row.astype(float) + slot + 1.,
                 energy_birks=row.astype(float) + slot + .5,
                 time=entries["time"][row] + slot + .25,
                 position=entries["position"][row] + np.column_stack((slot + 1., slot * 0., slot + 2.)),
                 crystal=row * 2 + slot, disk=(entries["disk"][row] + slot) % 2)
    library = build(entries, steps, development=np.ones(n, bool), binning=BINNING)
    library.update(geometry_disk_disk=np.array([0, 1]),
                   geometry_disk_origin_x=np.zeros(2), geometry_disk_origin_y=np.zeros(2),
                   geometry_disk_origin_z=np.zeros(2), geometry_disk_axis_z_x=np.zeros(2),
                   geometry_disk_axis_z_y=np.zeros(2), geometry_disk_axis_z_z=np.ones(2))
    return library


def dense_reference(library, entries, rng, neighbors=0):
    """Full query-by-library ranking, independent of the bounded candidate search."""
    features = np.column_stack([entries[k] for k in ("energy", "cosine", "radius")])
    edges = [library["bins_" + k] for k in ("energy", "cosine", "radius")]
    query = np.column_stack([np.clip(np.digitize(features[:, k], e) - 1, 0, len(e) - 2)
                             for k, e in enumerate(edges)])
    outside = np.any(np.column_stack([(features[:, k] < e[0]) | (features[:, k] > e[-1])
                                      for k, e in enumerate(edges)]), axis=1)
    labels = np.column_stack((library["entry_disk"], library["entry_cell"]))
    occupied = np.unique(labels, axis=0)
    distance = np.sum((query[:, None, :] - occupied[None, :, 1:]) ** 2, axis=2).astype(float)
    same_disk = entries["disk"][:, None] == occupied[None, :, 0]
    distance[~same_disk] = np.inf
    chosen = np.argmin(distance, axis=1)
    supported = np.any(same_disk, axis=1)
    fallback = outside | (distance[np.arange(len(query)), chosen] != 0)
    eligible = np.all(labels[None, :, :] == occupied[chosen, None, :], axis=2) & supported[:, None]
    bin_size = eligible.sum(axis=1)
    size = np.minimum(neighbors, bin_size) if neighbors else bin_size
    rows = np.broadcast_to(np.arange(len(labels)), eligible.shape)
    if neighbors:
        delta = np.abs(entries["radius"][:, None] - library["entry_radius"][None, :])
        delta[~eligible] = np.inf
        rank = np.argsort(np.argsort(delta, kind="stable", axis=1), axis=1)
        eligible &= rank < size[:, None]
    pool = np.sort(np.where(eligible, rows, len(labels)), axis=1)
    donors = np.full(len(query), -1, dtype=np.int64)
    for disk in np.unique(entries["disk"]):
        where = np.flatnonzero((entries["disk"] == disk) & supported)
        if len(where):
            pick = (rng.random(len(where)) * size[where]).astype(int)
            donors[where] = pool[where, pick]
    length = np.where(supported, np.diff(library["offsets"])[donors], 0)
    slots = np.arange(length.max(initial=0))
    mask = slots[None, :] < length[:, None]
    source = (library["offsets"][donors, None] + slots[None, :])[mask]
    row = np.nonzero(mask)[0]
    result = {k[5:]: value[source] for k, value in library.items() if k.startswith("step_")}
    result["position"] = entries["position"][row] + np.einsum(
        "nij,nj->ni", frame(entries["direction"])[row], result.pop("relative_position"))
    result["time"] = entries["time"][row] + result.pop("relative_time")
    result.update(entry=row, offsets=np.r_[0, length.cumsum()], donor=donors,
                  supported=supported, fallback=fallback)
    if neighbors:
        result.update(radius_pool_size=size, bin_pool_size=bin_size)
    return result


def assert_arrays(actual, expected):
    assert list(actual) == list(expected)
    for key in actual:
        assert actual[key].dtype == expected[key].dtype, key
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.fixture
def mixed():
    radii = [480., 120., 450., 400., 130., 400., 150., 300., 490., 250.,
             100., 200., 350., 400., 440., 470., 650., 750., 200., 220.]
    disk = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    energy = [75.] * 18 + [125., 125.]
    library = make_library(radii, disk, energy, empty=(5, 10, 17))
    query = make_entries([399., 200., 600., 100., 470., 1200., 250., 400., 650., 210.],
                         [0, 1, 0, 2, 0, 0, 1, 0, 1, 0],
                         [75., 75., 75., 75., 75., 200., 20., 75., 75., 125.])
    query["direction"][::2] = [1., 0., 0.]
    return library, query


@pytest.mark.parametrize("seed", [0, 19, 1234])
@pytest.mark.parametrize("explicit", [False, True])
def test_default_exact_independent_reference(mixed, seed, explicit):
    library, query = mixed
    actual_rng, reference_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    kwargs = {"radius_neighbors": 0} if explicit else {}
    actual = sample(library, query, actual_rng, **kwargs)
    assert_arrays(actual, dense_reference(library, query, reference_rng))
    assert "radius_pool_size" not in actual and "bin_pool_size" not in actual
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state


@pytest.mark.parametrize("neighbors", [1, 2, 8, np.int64(8), 100, 10**100])
def test_candidate_dense_reference_and_rng(mixed, neighbors):
    library, query = mixed
    actual_rng, reference_rng = np.random.default_rng(21), np.random.default_rng(21)
    actual = sample(library, query, actual_rng, radius_neighbors=neighbors)
    assert_arrays(actual, dense_reference(library, query, reference_rng, min(int(neighbors), 1000)))
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state
    baseline_rng = np.random.default_rng(21)
    baseline = sample(library, query, baseline_rng)
    np.testing.assert_array_equal(actual["fallback"], baseline["fallback"])
    np.testing.assert_array_equal(actual["supported"], baseline["supported"])
    assert actual_rng.bit_generator.state == baseline_rng.bit_generator.state
    small = actual["bin_pool_size"] <= neighbors
    np.testing.assert_array_equal(actual["donor"][small], baseline["donor"][small])


class PrescribedRNG:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.calls = []
        self.used = 0

    def random(self, size):
        self.calls.append(size)
        result = self.values[self.used:self.used + size]
        self.used += size
        assert len(result) == size
        return result


def test_known_draws_nearest_ties_original_row_order_and_empty():
    library = make_library([490., 300., 100., 300., 400., 200., 300., 300., 300., 450.], empty=(1,))
    query = make_entries([300.] * 8)
    rng = PrescribedRNG((np.arange(8) + .5) / 8)
    actual = sample(library, query, rng, radius_neighbors=8)
    np.testing.assert_array_equal(actual["donor"], [1, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(actual["radius_pool_size"], 8)
    np.testing.assert_array_equal(actual["bin_pool_size"], 10)
    assert actual["supported"].all() and not actual["fallback"].any()
    assert actual["offsets"][1] == 0
    assert rng.calls == [8]
    tie = sample(library, make_entries([350., 350.]), PrescribedRNG([0., .999]), radius_neighbors=2)
    np.testing.assert_array_equal(tie["donor"], [1, 3])


def test_known_legacy_disk_group_draws():
    library = make_library([100., 200., 300., 400.], [1, 0, 1, 0])
    query = make_entries([250.] * 5, [1, 0, 2, 1, 0])
    rng = PrescribedRNG([0., .999, .75, .25])
    actual = sample(library, query, rng)
    np.testing.assert_array_equal(actual["donor"], [2, 1, -1, 0, 3])
    assert rng.calls == [2, 2]
    candidate_rng = PrescribedRNG([0., .999, .75, .25])
    candidate = sample(library, query, candidate_rng, radius_neighbors=8)
    np.testing.assert_array_equal(candidate["donor"], actual["donor"])
    np.testing.assert_array_equal(candidate["bin_pool_size"], [2, 2, 0, 2, 2])
    assert candidate_rng.calls == rng.calls


@pytest.mark.parametrize("neighbors", [1, 2, 8, 17])
def test_bounded_pool_matches_dense_with_duplicate_runs(neighbors):
    rng = np.random.default_rng(71)
    radius = rng.integers(0, 12, 137).astype(float)
    rows = np.sort(rng.choice(len(radius), size=91, replace=False))
    query = np.r_[np.arange(-1., 14., .5), rng.uniform(-5, 20, 100)]
    distance = np.abs(query[:, None] - radius[rows])
    expected = np.sort(rows[np.argsort(distance, axis=1, kind="stable")[:, :neighbors]], axis=1)
    np.testing.assert_array_equal(_radius_pool(rows, radius, query, neighbors), expected)


def test_sparse_fallback_stays_in_chosen_cell_not_globally_nearest_radius():
    library = make_library([100., 490., 510., 520.], energy=[75., 125., 75., 75.])
    query = make_entries([499., 1200., 499.], disk=[0, 0, 9], energy=[20., 75., 20.])
    actual = sample(library, query, PrescribedRNG([0., .99]), radius_neighbors=1)
    np.testing.assert_array_equal(actual["donor"], [0, 3, -1])
    np.testing.assert_array_equal(actual["fallback"], [True, True, True])
    np.testing.assert_array_equal(actual["supported"], [True, True, False])
    np.testing.assert_array_equal(actual["bin_pool_size"], [1, 2, 0])
    np.testing.assert_array_equal(actual["radius_pool_size"], [1, 1, 0])


@pytest.mark.parametrize("orientation", ["incident", "disk"])
def test_whole_shower_energy_time_conservation_and_no_mutation(mixed, orientation):
    library, query = mixed
    original_library, original_query = deepcopy(library), deepcopy(query)
    actual = sample(library, query, np.random.default_rng(5), orientation=orientation, radius_neighbors=8)
    assert_arrays(library, original_library)
    assert_arrays(query, original_query)
    row = actual["entry"]
    donor = actual["donor"][row]
    local = np.arange(len(row)) - actual["offsets"][row]
    source = library["offsets"][donor] + local
    for key in ("energy_g4", "energy_birks", "crystal", "disk"):
        np.testing.assert_array_equal(actual[key], library["step_" + key][source])
    np.testing.assert_array_equal(actual["time"], query["time"][row] + library["step_relative_time"][source])
    for key in ("energy_g4", "energy_birks"):
        prefix = np.r_[0., library["step_" + key].cumsum()]
        sums = np.diff(prefix[library["offsets"]])
        expected = np.where(actual["supported"], sums[actual["donor"]], 0.)
        np.testing.assert_array_equal(np.bincount(row, weights=actual[key], minlength=len(query["energy"])), expected)
    for array in actual.values():
        array[...] = 0
    assert_arrays(library, original_library)
    assert_arrays(query, original_query)


def test_no_continuous_energy_or_direction_conditioning(mixed):
    library, query = mixed
    changed = deepcopy(query)
    changed["energy"] = np.where(query["energy"] == 75., 99., query["energy"])
    changed["cosine"][:] = .99
    changed["direction"][:] = [0., 1., 1.]
    before = sample(library, query, np.random.default_rng(9), radius_neighbors=8)
    after = sample(library, changed, np.random.default_rng(9), radius_neighbors=8)
    for key in ("donor", "energy_g4", "energy_birks", "time", "offsets", "fallback"):
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("value", [-1, np.int64(-2), True, False, np.bool_(True), 0., 8., np.float64(8),
                                   "8", None, np.array(8)])
def test_invalid_neighbors_before_rng(mixed, value):
    library, query = mixed
    rng = PrescribedRNG([])
    with pytest.raises(ValueError, match="radius_neighbors must be a nonnegative integer"):
        sample(library, query, rng, radius_neighbors=value)
    assert not rng.calls


@pytest.mark.parametrize("neighbors", [0, 8])
@pytest.mark.parametrize("orientation", ["incident", "disk"])
def test_empty_queries(mixed, neighbors, orientation):
    library, query = mixed
    query = {key: value[:0] for key, value in query.items()}
    rng = PrescribedRNG([])
    actual = sample(library, query, rng, radius_neighbors=neighbors, orientation=orientation)
    assert not rng.calls
    assert actual["position"].shape == (0, 3)
    np.testing.assert_array_equal(actual["offsets"], [0])
    assert ("radius_pool_size" in actual) == bool(neighbors)
    assert ("bin_pool_size" in actual) == bool(neighbors)
    assert all(len(value) == 0 for key, value in actual.items() if key != "offsets")


def test_all_empty_donors_and_all_unsupported():
    library = make_library(np.arange(100., 200., 10.), empty=np.arange(10))
    query = make_entries([150., 155.])
    actual = sample(library, query, PrescribedRNG([0., .99]), radius_neighbors=8)
    assert actual["supported"].all()
    np.testing.assert_array_equal(actual["offsets"], [0, 0, 0])
    np.testing.assert_array_equal(actual["radius_pool_size"], [8, 8])
    query["disk"][:] = 9
    rng = PrescribedRNG([])
    unsupported = sample(library, query, rng, radius_neighbors=8)
    assert not rng.calls
    assert unsupported["fallback"].all() and not unsupported["supported"].any()
    np.testing.assert_array_equal(unsupported["donor"], [-1, -1])
    np.testing.assert_array_equal(unsupported["radius_pool_size"], [0, 0])
    np.testing.assert_array_equal(unsupported["bin_pool_size"], [0, 0])
