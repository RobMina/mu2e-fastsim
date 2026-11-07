"""Synthetic exact-column checks; mocked CLI source tests wiring, not art decoding."""
import json

import numpy as np
import pytest

from calofastsim.showers import production_check as check
from test_showers_export import write
from test_showers_production import production_tables


def fixture():
    ids = np.array([[1, 0, 7], [1, 1, 7], [2, 0, 7], [2, 1, 7], [3, 0, 7]])
    entry = np.array([0, 0, 2, 3])
    raw = dict(entry=entry, sim_product=np.array([100, 999, 100, 100], np.uint32),
               sim_key=np.array([20, 20, 20, 2**63 + 1], np.uint64),
               n_compress=np.array([1, 2, 3, 4], np.uint32), volume=np.array([10, 11, 10, 11], np.int32),
               time=np.array([1.0000000000000002, 2., 3., 4.]),
               energy_g4=np.array([1.1, 2.2, 3.3, 4.4], np.float32),
               energy_birks=np.array([.1, .2, .3, .4], np.float32),
               momentum_in=np.array([10.1, 20.2, 30.3, 40.4], np.float32),
               pos_x=np.array([.10000000000000002, .2, .3, .4]),
               pos_y=np.array([2., 3., 4., 5.]), pos_z=np.array([3., 4., 5., 6.]))
    events = dict(zip(check.export.EVENT, ids.T.copy()), n_showers=np.array([2, 0, 1, 1, 0]))
    showers = dict(zip(check.export.EVENT, ids[entry].T.copy()), index=np.array([0, 1, 0, 0]))
    for source, target in check.FIELDS.items():
        showers[target] = raw[source].astype(np.float64) if raw[source].dtype.kind == "f" else raw[source].copy()
    return raw, ids, events, showers


def test_exact_all_fields_empty_events_and_reused_keys():
    assert check.reconcile(*fixture()) == dict(events=5, css_rows=4, all_fields_exact=True)


@pytest.mark.parametrize("no_events", [False, True])
def test_all_empty(no_events):
    raw, ids, events, showers = fixture()
    raw = {k: v[:0] for k, v in raw.items()}
    showers = {k: v[:0] for k, v in showers.items()}
    events["n_showers"][:] = 0
    if no_events:
        ids, events = ids[:0], {k: v[:0] for k, v in events.items()}
    assert check.reconcile(raw, ids, events, showers) == dict(events=len(ids), css_rows=0, all_fields_exact=True)


@pytest.mark.parametrize("source,target", check.FIELDS.items())
def test_every_field_exact_including_one_bit_double_changes(source, target):
    raw, ids, events, showers = fixture()
    col = showers[target]
    col[0] = np.nextafter(col[0], np.inf) if col.dtype.kind == "f" else col[0] + 1
    with pytest.raises(ValueError, match=f"CSS {source} -> {target} mismatch"):
        check.reconcile(raw, ids, events, showers)


@pytest.mark.parametrize("field", ["time", "position_x"])
def test_precision_loss_rejected(field):
    raw, ids, events, showers = fixture()
    showers[field] = showers[field].astype(np.float32)
    with pytest.raises(ValueError, match=field):
        check.reconcile(raw, ids, events, showers)


@pytest.mark.parametrize("field", check.export.EVENT)
def test_full_ordered_event_identity(field):
    raw, ids, events, showers = fixture()
    events[field][0] += 1
    with pytest.raises(ValueError, match="ordered event IDs"):
        check.reconcile(raw, ids, events, showers)


@pytest.mark.parametrize("case,match", [
    ("event-order", "ordered event IDs"), ("duplicate-event", "duplicate source event"),
    ("shower-order", "shower local index"), ("duplicate-index", "shower local index"),
    ("wrong-run", "ordered shower run"), ("wrong-subrun", "ordered shower subrun"),
    ("missing-shower", "ordered shower run"), ("extra-shower", "ordered shower run"),
    ("empty-census", "source CSS counts"), ("nonempty-census", "source CSS counts"),
    ("source-entry", "source CSS entry order"), ("source-range", "source CSS entry out of range"),
    ("source-column-length", "source CSS time row count"),
])
def test_order_duplicates_and_census(case, match):
    raw, ids, events, showers = fixture()
    if case == "event-order":
        events = {k: v[::-1] for k, v in events.items()}
    elif case == "duplicate-event":
        ids[1] = ids[0]
    elif case == "shower-order":
        showers = {k: v[[1, 0, 2, 3]] for k, v in showers.items()}
    elif case == "duplicate-index":
        showers["index"][1] = 0
    elif case.startswith("wrong-"):
        showers[case[6:]][0] += 1
    elif case == "missing-shower":
        showers = {k: v[:-1] for k, v in showers.items()}
    elif case == "extra-shower":
        showers = {k: np.r_[v, v[-1:]] for k, v in showers.items()}
    elif case.endswith("census"):
        events["n_showers"][1 if case == "empty-census" else 0] += 1
    elif case == "source-entry":
        raw["entry"][:] = [0, 2, 0, 3]
    elif case == "source-range":
        raw["entry"][0] = len(ids)
    elif case == "source-column-length":
        raw["time"] = raw["time"][:-1]
    with pytest.raises(ValueError, match=match):
        check.reconcile(raw, ids, events, showers)


def test_cli_wiring_with_mocked_raw_source(tmp_path, monkeypatch, capsys):
    data = production_tables()
    showers = data["showers"]
    showers.update(index=np.array([0, 1, 0]), n_compress=np.array([2, 3, 1]), momentum_in=np.array([1., 2., 3.]))
    tables, art, out = tmp_path / "numeric.root", tmp_path / "mock-art.root", tmp_path / "report/check.json"
    write(tables, data)
    real_read = check.export.read
    calls = []

    def validated(path, *, input_mode):
        assert path == tables and input_mode == "primary"
        result = real_read(path, input_mode=input_mode)
        calls.append("validated")
        return result

    def raw_source(path, *, label):
        assert calls == ["validated"] and path == art and label == "compressDetStepMCs__Primary"
        calls.append("source")
        raw = {source: showers[target].copy() for source, target in check.FIELDS.items()}
        raw["entry"] = np.array([0, 0, 2])
        return raw, np.column_stack([data["events"][k] for k in check.export.EVENT])

    monkeypatch.setattr(check.export, "read", validated)
    monkeypatch.setattr(check.artsteps, "read_steps", raw_source)
    check.main(["--art", str(art), "--tables", str(tables), "--out", str(out)])
    report = json.loads(out.read_text())
    assert json.loads(capsys.readouterr().out) == report
    assert calls == ["validated", "source"]
    assert report["source_label"] == "compressDetStepMCs__Primary"
    assert report["reconciliation"] == dict(events=3, css_rows=3, all_fields_exact=True)
    assert report["census"]["empty_css_events"] == 1
    assert report["census"]["other_ancestry_css_rows"] == 1
    assert "stored/selected-population" in report["denominator"]
    assert "not unconditional" in report["denominator"]
    assert "not checked" in report["geometry"]
    json.dumps(report, allow_nan=False)


def test_cli_census_failure_before_source_read(tmp_path, monkeypatch):
    data = production_tables()
    data["events"]["n_showers"][1] = 1
    tables, out = tmp_path / "bad.root", tmp_path / "must-not-exist.json"
    write(tables, data)

    def forbidden(*args, **kwargs):
        pytest.fail("raw source must not be read after failed graph/census validation")

    monkeypatch.setattr(check.artsteps, "read_steps", forbidden)
    with pytest.raises(ValueError, match="shower row counts"):
        check.main(["--art", "unused.root", "--tables", str(tables), "--out", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("defect", [None, "crystal-frame", "disk-frame", "vd-frame", "crossing", "duplicate", "missing"])
def test_geometry_frames_and_crossings(tmp_path, defect):
    import uproot

    data = production_tables()
    data["disks"]["origin_z"] += 9.99
    data["virtualdetectors"]["half_length"] = np.full(4, .01)
    for axis in "xyz":
        data["crystals"]["local_position_" + axis] = np.full(4, -9.99 if axis == "z" else 0.)
    if defect == "crystal-frame":
        data["crystals"]["local_position_z"][0] += 1
    elif defect == "disk-frame":
        data["disks"]["axis_z_z"][0] = 2
    elif defect == "vd-frame":
        data["virtualdetectors"]["position_z"][0] += 1
    elif defect == "crossing":
        data["steps"]["position_z"][0] += 1
    elif defect == "duplicate":
        data["crystals"]["crystal"][0] = 11
    elif defect == "missing":
        data["virtualdetectors"] = {k: v[1:] for k, v in data["virtualdetectors"].items()}
    path = tmp_path / "geometry.root"
    write(path, data)
    with uproot.open(path) as source:
        if defect:
            with pytest.raises(ValueError):
                check.geometry_check(source)
        else:
            result = check.geometry_check(source)
            assert result["crystals"] == 4
            assert result["crystals_per_disk"] == [1, 1, 1, 1]
            assert result["checked_front_crossings"] == 6
            assert result["max_front_plane_distance_mm"] == 0
