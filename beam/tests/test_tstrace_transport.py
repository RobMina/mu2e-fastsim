"""Selected-lane observer regressions; no collector, saved inputs or replay needed."""
import numpy as np
import pytest

from beamfastsim import tstransport as ts, tsphysics as phys
from mu2efastsim import materials
from test_tstransport import Field, Geometry, source


class Observer:
    """Keep callback snapshots, optionally vandalizing every supplied array."""

    def __init__(self, rows, mutate=False):
        self.rows = np.asarray(rows)
        self.mutate = mutate
        self.events = []

    def _record(self, kind, arrays):
        self.events.append((kind, {k: v.copy() if isinstance(v, np.ndarray) else v
                                   for k, v in arrays.items()}))
        if self.mutate:
            for value in arrays.values():
                if isinstance(value, np.ndarray):
                    value[...] = ~value if value.dtype.kind == "b" else -99

    def begin(self, **arrays):
        self._record("begin", arrays)

    def material(self, **arrays):
        self._record("material", arrays)

    def end(self, **arrays):
        self._record("end", arrays)


def assert_bit_identical(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if key != "seconds":
            a, b = np.asarray(left[key]), np.asarray(right[key])
            assert a.shape == b.shape, key
            assert a.dtype == b.dtype, key
            assert a.tobytes() == b.tobytes(), key


def assert_chronology(observer, initial, result):
    """Replay bookkeeping in batched rows; loop over callbacks, not particles."""
    n = len(initial["pdg"])
    counts = np.zeros(n, dtype=np.int32)
    previous = {key: initial[key].copy() for key in ("x", "p", "t", "pdg")}
    previous["path"] = np.zeros(n)
    final_fate = np.full(n, ts.LIVE)
    begin = None
    material_p = None
    for kind, data in observer.events:
        if kind == "begin":
            assert begin is None, "begin must follow end"
            begin = data
            rows = begin["row"]
            assert np.isin(rows, observer.rows).all()
            assert len(np.unique(rows)) == len(rows)
            np.testing.assert_array_equal(begin["step"], counts[rows])
            for key in previous:
                np.testing.assert_array_equal(begin[key + "0"], previous[key][rows])
            assert begin["proposed_x"].shape == (len(rows), 3)
            assert begin["h"].shape == begin["valid"].shape == (len(rows),)
            assert np.all(begin["h"] >= 0)
            material_p = np.full((n, 3), np.nan)
        elif kind == "material":
            assert begin is not None, "material must occur between begin and end"
            rows = data["row"]
            assert len(rows) and np.isin(rows, begin["row"]).all()
            assert len(np.unique(rows)) == len(rows)
            assert isinstance(data["name"], str)
            assert data["requested_mm"].shape == data["fraction"].shape == data["stopped"].shape == (len(rows),)
            assert data["p_before"].shape == data["p_after"].shape == (len(rows), 3)
            assert np.all(data["requested_mm"] > 0)
            assert np.all((data["fraction"] >= 0) & (data["fraction"] <= 1))
            np.testing.assert_array_equal(data["p_after"][data["stopped"]], 0)
            material_p[rows] = data["p_after"]
        else:
            assert kind == "end" and begin is not None, "end must match begin"
            rows = begin["row"]
            delta = data["path1"] - begin["path0"]
            assert np.all(delta >= 0)
            assert np.all(delta <= begin["h"] + 1e-12)
            assert np.all(data["t1"] >= begin["t0"])
            # These fixtures contain no pion decay after the material callback.
            touched = np.isfinite(material_p[rows]).all(axis=1)
            np.testing.assert_array_equal(data["p1"][touched], material_p[rows][touched])
            for key in previous:
                previous[key][rows] = data[key + "1"]
            final_fate[rows] = data["fate"]
            counts[rows] += 1
            begin = None
    assert begin is None
    rows = observer.rows
    np.testing.assert_array_equal(counts[rows], result["steps"][rows])
    for key in previous:
        np.testing.assert_array_equal(previous[key][rows], result[key][rows])
    np.testing.assert_array_equal(final_fate[rows], result["fate"][rows])
    assert not counts[~np.isin(np.arange(n), rows)].any()


@pytest.mark.parametrize("method", ["rk4", "helix"])
@pytest.mark.parametrize("with_material", [False, True])
@pytest.mark.parametrize("rows", [np.array([], dtype=np.int64), np.array([5, 0, 3])],
                         ids=["empty", "several-unsorted"])
def test_observer_does_not_change_any_returned_bits(method, with_material, rows):
    initial = source(n=6, p=8 if with_material else 1000)
    geom = Geometry(materials.ALUMINIUM if with_material else None)
    if not with_material:
        initial["x"][:, 2] = [0, 96, 98, 99, 94, 97]
    kwargs = dict(seed=42, step_mm=2, max_path_mm=7, method=method)
    baseline = ts.transport(initial, Field(), geom, trace=None, **kwargs)
    observer = Observer(rows)
    traced = ts.transport(initial, Field(), geom, trace=observer, **kwargs)
    assert_bit_identical(baseline, traced)
    assert_chronology(observer, initial, traced)
    assert observer.events
    if not len(rows):
        assert all(kind != "material" for kind, _ in observer.events)
        assert all(not len(value) for _, data in observer.events for value in data.values())
    elif with_material:
        assert any(kind == "material" for kind, _ in observer.events)
        np.testing.assert_array_equal(traced["fate"], ts.RANGEOUT)
    else:
        # Selected source row IDs persist through active-lane compaction.
        np.testing.assert_array_equal(traced["fate"][rows], [ts.DS, ts.PATH_LIMIT, ts.DS])
        assert len(np.unique(traced["steps"][rows])) == 3


@pytest.mark.parametrize("rows", [
    pytest.param([1, 1], id="duplicate"),
    pytest.param([-1], id="negative"),
    pytest.param([4], id="past-end"),
    pytest.param(np.array([2**64 - 1], dtype=np.uint64), id="unsigned-overflow"),
    pytest.param([0.5], id="fractional"),
    pytest.param([1.0], id="integral-float"),
    pytest.param([True], id="boolean"),
    pytest.param(["1"], id="string"),
    pytest.param([np.nan], id="nan"),
    pytest.param(1, id="scalar"),
    pytest.param([[0, 1]], id="matrix"),
])
def test_invalid_trace_rows_are_rejected_before_callbacks(rows):
    observer = Observer(rows)
    with pytest.raises(ValueError, match="trace rows"):
        ts.transport(source(n=4), Field(), Geometry(), trace=observer)
    assert observer.events == []


def test_material_stop_callback_precedes_final_state():
    initial = source(n=4, p=8)
    observer = Observer([3, 1])
    out = ts.transport(initial, Field(), Geometry(materials.ALUMINIUM), seed=42,
                       trace=observer)
    assert_chronology(observer, initial, out)
    stops = []
    for index, (kind, data) in enumerate(observer.events):
        if kind != "material" or not data["stopped"].any():
            continue
        assert data["name"] == materials.ALUMINIUM.name
        begin_kind, begin = observer.events[index - 1]
        end_kind, end = observer.events[index + 1]
        assert (begin_kind, end_kind) == ("begin", "end")
        row = data["row"][data["stopped"]]
        begin_index = np.searchsorted(begin["row"], row)
        fraction = data["fraction"][data["stopped"]]
        assert np.all((fraction > 0) & (fraction < 1))
        np.testing.assert_array_equal(end["fate"][begin_index], ts.RANGEOUT)
        np.testing.assert_array_equal(end["p1"][begin_index], 0)
        np.testing.assert_allclose(end["path1"][begin_index] - begin["path0"][begin_index],
                                   begin["h"][begin_index] * fraction, rtol=1e-12)
        expected_x = (begin["x0"][begin_index] + fraction[:, None]
                      * (begin["proposed_x"][begin_index] - begin["x0"][begin_index]))
        np.testing.assert_array_equal(end["x1"][begin_index], expected_x)
        stops.append(row)
    assert stops
    np.testing.assert_array_equal(np.sort(np.concatenate(stops)), np.sort(observer.rows))
    np.testing.assert_array_equal(out["fate"], ts.RANGEOUT)
    assert np.all(out["material_path"] > 0)


@pytest.mark.parametrize("method", ["rk4", "helix"])
def test_mutating_callback_arrays_cannot_change_physics(method):
    initial = source(n=6, p=8)
    initial_copy = {key: value.copy() for key, value in initial.items()}
    geom = Geometry(materials.ALUMINIUM)
    kwargs = dict(seed=42, method=method)
    baseline = ts.transport(initial, Field(), geom, trace=None, **kwargs)
    observer = Observer([5, 0, 3], mutate=True)
    mutated = ts.transport(initial, Field(), geom, trace=observer, **kwargs)
    assert_bit_identical(baseline, mutated)
    assert_bit_identical(initial_copy, initial)
    assert_chronology(observer, initial, mutated)
    assert {kind for kind, _ in observer.events} == {"begin", "material", "end"}


def test_observer_preserves_rng_state_and_freezes_selection(monkeypatch):
    factory = np.random.default_rng
    generators = []

    def rng(seed):
        generator = factory(seed)
        generators.append(generator)
        return generator

    monkeypatch.setattr(np.random, 'default_rng', rng)
    initial = source(n=4, p=8)
    geom = Geometry(materials.ALUMINIUM)
    plain = ts.transport(initial, Field(), geom, seed=71)

    class ChangingRows(Observer):
        def begin(self, **arrays):
            super().begin(**arrays)
            self.rows[:] = 0

    observer = ChangingRows([3, 1])
    traced = ts.transport(initial, Field(), geom, seed=71, trace=observer)
    assert_bit_identical(plain, traced)
    assert generators[0].bit_generator.state == generators[1].bit_generator.state
    for kind, arrays in observer.events:
        if kind != 'end':
            assert set(arrays['row']) <= {1, 3}


@pytest.mark.parametrize("all_cold", [False, True], ids=["mixed", "only-cold"])
def test_cold_at_start_has_final_fate_without_accepted_steps(all_cold):
    initial = source(n=3)
    cold = np.full(3, all_cold)
    cold[1] = True
    initial["p"][cold, 2] = np.sqrt(phys.TK_STOP_MEV * phys.MMU)
    observer = Observer([2, 1, 0])
    baseline = ts.transport(initial, Field(), Geometry(), max_path_mm=1, trace=None)
    out = ts.transport(initial, Field(), Geometry(), max_path_mm=1, trace=observer)
    assert_bit_identical(baseline, out)
    np.testing.assert_array_equal(out["fate"][cold], ts.RANGEOUT)
    np.testing.assert_array_equal(out["steps"][cold], 0)
    np.testing.assert_array_equal(out["path"][cold], 0)
    np.testing.assert_array_equal(out["material_path"][cold], 0)
    np.testing.assert_array_equal(out["x"][cold], initial["x"][cold])
    np.testing.assert_array_equal(out["t"][cold], initial["t"][cold])
    if all_cold:
        assert observer.events == []
    else:
        assert [kind for kind, _ in observer.events] == ["begin", "end"]
        np.testing.assert_array_equal(observer.events[0][1]["row"], np.flatnonzero(~cold))
        np.testing.assert_array_equal(observer.events[1][1]["fate"], ts.PATH_LIMIT)
