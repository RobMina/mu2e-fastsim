"""Saved donor-frame reuse tables (mixing.device_reuse_table): a loaded table equals the computed one and gives the
same digis; loading refuses another seed, pools, conditions, setup, tracker float, an uncovered shift or longer
events."""
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mu2efastsim.testing")
pytest.importorskip("trkfastsim")

from mixing import device_reuse, device_reuse_table as rt

from test_mixing_device_reuse import CAPS, _mixed, _pools, _run, conditions, helpers, x64  # noqa: F401

REUSE = {"compute": CAPS, "precompute": CAPS, "frames_per_batch": 8, "straws": 8, "shifts": "events"}
EVENTS = np.array([-2., -9., -2., -9.])


def _table(c, data, seed=0, events=EVENTS, extra=()):
    return rt.obtain(None, c, REUSE, data, 1700., seed, rows_capacity=8, event_shifts=events, extra_shifts=extra)


def _same(a, b):
    for name in rt.FIELDS + ("copy_shifts",):
        np.testing.assert_array_equal(np.asarray(a[name]), np.asarray(b[name]), err_msg=name)
        assert np.asarray(a[name]).dtype.kind == np.asarray(b[name]).dtype.kind, name


def test_obtain_without_a_path_is_the_precompute(conditions, helpers):
    """No table path: the precompute with the events' mean shift and margin, as device_physics did before."""
    c = conditions
    data, _, _ = _pools(c, helpers)
    mean = float(np.mean(EVENTS))
    direct = device_reuse.precompute(c, REUSE, data, 1700., 0, rows_capacity=8, shift=mean,
                                     margin=float(np.max(np.abs(EVENTS - mean))) + 5.0, event_shifts=EVENTS)
    _same(_table(c, data), direct)
    assert _table(c, data)["shift"] == direct["shift"]


def test_saved_table_round_trip_and_digis(conditions, helpers, tmp_path):
    """Save then load: the same table (frames remapped to the job's pool starts) and the same mixed digis."""
    c = conditions
    data, index, frames = _pools(c, helpers)
    table = _table(c, data)
    key = rt.table_key(c, REUSE, data, 0, 8)
    path = tmp_path / "reuse.npz"
    rt.save(path, table, key, data["starts"], 1700.)
    loaded = rt.load(path, key, data["starts"], EVENTS, 1700.)
    _same(loaded, table)
    moved = rt.load(path, key, data["starts"] + np.array([0, 3, 3, 3, 3]), EVENTS, 1690.)   # 3 more signal frames
    np.testing.assert_array_equal(moved["frame"], table["frame"] + 3)
    collection = _mixed(c, data, index, frames)
    times = {"event_length": np.array([1700., 1690.]), "mc_pbtime": np.array([-2., -9.]), "event_mask": np.ones(2, bool)}
    a, ea = _run(device_reuse.prepare(c, REUSE, 2, table, data["starts"]), collection, times)
    b, eb = _run(device_reuse.prepare(c, REUSE, 2, loaded, data["starts"]), collection, times)
    assert not any(np.any(v) for v in {**ea, **eb}.values())
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_load_refuses_another_job(conditions, helpers, tmp_path):
    c = conditions
    data, _, _ = _pools(c, helpers)
    key = rt.table_key(c, REUSE, data, 0, 8)
    path = tmp_path / "reuse.npz"
    rt.save(path, _table(c, data), key, data["starts"], 1700.)
    with pytest.raises(ValueError, match="tracker_seed"):
        rt.load(path, rt.table_key(c, REUSE, data, 1, 8), data["starts"], EVENTS, 1700.)
    with pytest.raises(ValueError, match="setup"):
        rt.load(path, rt.table_key(c, {**REUSE, "frames_per_batch": 4}, data, 0, 8), data["starts"], EVENTS, 1700.)
    with pytest.raises(ValueError, match="setup"):      # the row budget sets the setup batches, hence the draws
        rt.load(path, rt.table_key(c, REUSE, data, 0, 16), data["starts"], EVENTS, 1700.)
    other = {**data, "collections": {"tracker": {**data["collections"]["tracker"], "columns": dict(
        data["collections"]["tracker"]["columns"])}}}
    first = sorted(other["collections"]["tracker"]["columns"])[0]
    values = other["collections"]["tracker"]["columns"][first].copy()
    values[data["collections"]["tracker"]["offsets"][1]] += 1      # a step of the first donor frame
    other["collections"]["tracker"]["columns"][first] = values
    with pytest.raises(ValueError, match="pools"):
        rt.load(path, rt.table_key(c, REUSE, other, 0, 8), data["starts"], EVENTS, 1700.)
    from trkfastsim.digi.digicond import Conditions
    s = dict(c.s)
    s["deadTimeAnalog"] = float(s["deadTimeAnalog"]) + 1.0
    with pytest.raises(ValueError, match="conditions"):
        rt.load(path, rt.table_key(Conditions(s, c.a), REUSE, data, 0, 8), data["starts"], EVENTS, 1700.)
    with pytest.raises(ValueError, match="tracker_float"):
        rt.load(path, {**key, "tracker_float": "other"}, data["starts"], EVENTS, 1700.)
    with pytest.raises(ValueError, match="no copy at event shifts"):
        rt.load(path, key, data["starts"], np.r_[EVENTS, -7.], 1700.)
    with pytest.raises(ValueError, match="events up to"):
        rt.load(path, key, data["starts"], EVENTS, 1710.)
    with pytest.raises(ValueError, match="shifts"):
        rt.table_key(c, {k: v for k, v in REUSE.items() if k != "shifts"}, data, 0, 8)


def test_obtain_saves_then_loads_the_union(conditions, helpers, tmp_path, monkeypatch):
    """With a path: the first job computes the table for its shifts plus the extra ones and saves it; a job with
    other shifts of that union then loads it without a precompute."""
    c = conditions
    data, _, _ = _pools(c, helpers)
    path = tmp_path / "reuse.npz"
    made = rt.obtain(path, c, REUSE, data, 1700., 0, rows_capacity=8, event_shifts=EVENTS, extra_shifts=[-4.])
    np.testing.assert_array_equal(made["copy_shifts"], [-9., -4., -2.])
    assert json.loads(str(np.load(path)["key"]))["version"] == rt.VERSION

    def no_precompute(*a, **k):
        raise AssertionError("precompute called although the table exists")
    monkeypatch.setattr(device_reuse, "precompute", no_precompute)
    loaded = rt.obtain(path, c, REUSE, data, 1700., 0, rows_capacity=8, event_shifts=np.array([-4., -4., -9.]))
    _same(loaded, made)
