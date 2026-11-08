"""Multi-muon identity audit (scripts/audit_native_primary_identity.py): tiny frames and schema-v2 donors."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multiprimary_fixtures import _frames, _tiny_counters, _write
from crvfastsim import genassoc, geometry

pytestmark = pytest.mark.slow   # campaign and validation tooling, not the simulation chain: runs with --runslow

ROOT = Path(__file__).resolve().parents[1]


def _audit():
    spec = importlib.util.spec_from_file_location("native_primary_identity_audit", ROOT / "scripts/audit_native_primary_identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def audit():
    return _audit()


@pytest.fixture
def geom():
    return SimpleNamespace(sector=np.zeros(4, dtype=np.int64), lib_group=np.array(["T1"]))


def _parts(rows):
    frame = pd.DataFrame(rows, columns=["event", "key", "parent", "pdg", "isprim"], dtype=np.int64)
    return frame.assign(run=np.int64(1400), subrun=np.int64(101))


def _steps(rows):
    frame = pd.DataFrame(rows, columns=["event", "sp", "bar", "t0", "x0", "edep"])
    for name in ("event", "sp", "bar"):
        frame[name] = frame[name].astype(np.int64)
    return frame.assign(run=np.int64(1400), subrun=np.int64(101), x1=frame.x0,
                        y0=0., y1=0., z0=0., z1=0., px0=1., py0=0., pz0=0., len=2.)


def test_multi_muon_samebar_merges_and_uses_earliest_actual_primary(audit, geom):
    parts = _parts([(1, 7, -1, 13, 1), (1, 9, -1, -13, 1),
                    (1, 8, 7, 11, 0), (1, 10, 9, 11, 0)])
    steps = _steps([(1, 7, 0, 10., 0., 2.), (1, 9, 0, 1., 100., 3.),
                    (1, 8, 0, 11., 0., .4), (1, 10, 0, 12., 100., .8)])
    before_s, before_p = steps.copy(deep=True), parts.copy(deep=True)
    result = audit.analyze(steps, parts, geom)
    assert result["primary_muons"]["histogram"] == {"2": 1}
    assert result["crossings"] == {"historical_event_bar": 1, "ancestor_key_bar_reference": 2,
        "merged_event_bar": 1, "merged_events": 1, "max_primaries_per_event_bar": 2,
        "primary_multiplicity_histogram": {"2": 1}, "affected_primary_steps": 2,
        "affected_primary_edep": 5., "affected_primary_length": 4.}
    d = result["descendants"]
    assert d["historical_wrong_ancestor_samebar"]["rows"] == 1
    assert d["historical_wrong_ancestor_samebar"]["edep"] == pytest.approx(.4)
    assert d["historical_wrong_ancestor_distance"]["rows"] == 0
    assert d["identity_safe_attached"]["rows"] == 2
    assert d["reference_changed_primary_or_bar"]["rows"] == 1
    pd.testing.assert_frame_equal(steps, before_s)
    pd.testing.assert_frame_equal(parts, before_p)


def test_swapped_closest_and_samebar_priority_are_not_truth_matches(audit, geom):
    parts = _parts([(1, 7, -1, 13, 1), (1, 9, -1, 13, 1),
                    (1, 8, 7, 11, 0), (1, 10, 9, 11, 0), (1, 11, 7, 11, 0)])
    steps = _steps([(1, 7, 0, 10., 0., 2.), (1, 9, 1, 1., 100., 3.),
                    (1, 8, 2, 11., 100., .2), (1, 10, 3, 12., 0., .3),
                    (1, 11, 1, 13., 0., .7)])
    d = audit.analyze(steps, parts, geom)["descendants"]
    assert d["historical_wrong_ancestor"]["rows"] == 3
    assert d["historical_wrong_ancestor"]["edep"] == pytest.approx(1.2)
    assert d["historical_wrong_ancestor_distance"]["edep"] == pytest.approx(.5)
    assert d["historical_wrong_ancestor_samebar"]["edep"] == pytest.approx(.7)
    assert d["reference_changed_primary_or_bar"]["rows"] == 3


@pytest.mark.parametrize("samebar", [False, True])
def test_historical_ties_follow_bar_then_time_not_nearest_primary(audit, geom, samebar):
    parts = _parts([(1, 7, -1, 13, 1), (1, 9, -1, 13, 1), (1, 10, 9, 11, 0)])
    steps = _steps([(1, 9, 1, 1., 100., 3.), (1, 7, 0, 10., 0., 2.),
                    (1, 10, 2, 12., 50., .3)])
    if samebar:
        steps.loc[steps.sp != 10, "bar"] = 0
        steps.loc[steps.sp == 10, "bar"] = 0
        steps.loc[steps.sp != 10, "t0"] = 1.
        steps = steps.iloc[[1, 0, 2]].reset_index(drop=True)
    result = audit.analyze(steps, parts, geom)
    assert result["descendants"]["historical_wrong_ancestor"]["rows"] == 1
    assert result["descendants"]["historical_wrong_ancestor"]["edep"] == pytest.approx(.3)


def test_noncrossing_ancestor_lost_from_event_only_nearmiss(audit, geom):
    parts = _parts([(1, 7, -1, 13, 1), (1, 9, -1, 13, 1), (1, 10, 9, 11, 0),
                    (2, 7, -1, 13, 1), (2, 8, 7, 11, 0)])
    steps = _steps([(1, 7, 0, 1., 0., 2.), (1, 10, 1, 2., 100., .3),
                    (2, 8, 1, 2., 100., .4)])
    result = audit.analyze(steps, parts, geom)
    d = result["descendants"]
    assert d["intended_no_ancestor_crossing"]["rows"] == 2
    assert d["intended_no_ancestor_crossing"]["edep"] == pytest.approx(.7)
    assert d["historical_unattached"]["rows"] == 1
    assert d["historical_unattached"]["edep"] == pytest.approx(.4)
    assert d["event_only_nearmiss_excluded"]["rows"] == 1
    assert d["event_only_nearmiss_excluded"]["edep"] == pytest.approx(.3)
    assert d["historical_wrong_ancestor"]["rows"] == 1
    assert d["identity_safe_attached"]["rows"] == 0
    assert result["donor_local_nearmiss"] == {"identity_no_crossing_muons": 2,
        "historical_event_only_no_crossing_muons": 1, "event_only_excluded_muons": 1,
        "excluded_muons_with_descendants": 1}


@pytest.mark.parametrize("empty", [False, True])
def test_single_muon_or_empty_null_control(audit, geom, empty):
    parts = _parts([(1, 7, -1, 13, 1), (1, 8, 7, 11, 0)])
    steps = _steps([(1, 7, 0, 1., 0., 2.), (1, 8, 1, 2., 100., .3)])
    if empty:
        parts, steps = parts.iloc[:0], steps.iloc[:0]
    result = audit.analyze(steps, parts, geom)
    assert result["crossings"]["merged_event_bar"] == 0
    for name in ("historical_wrong_ancestor", "reference_changed_primary_or_bar",
                 "event_only_nearmiss_excluded", "intended_no_ancestor_crossing"):
        assert result["descendants"][name]["rows"] == 0


def test_full_int64_keys_and_complete_event_tuples(audit, geom):
    base = 2**60
    parts = _parts([(1, base + 7, -1, 13, 1), (1, base + 8, -1, 13, 1),
                    (1, base + 9, base + 8, 11, 0),
                    (2, base + 7, -1, 13, 1), (2, base + 9, base + 7, 11, 0)])
    steps = _steps([(1, 0, 0, 1., 0., 2.), (1, 0, 1, 2., 100., .3),
                    (2, 0, 1, 2., 100., .4)])
    steps["sp"] = np.array([base + 7, base + 9, base + 9], dtype=np.int64)
    for frame in (parts, steps):
        frame.loc[frame.event == 1, "event"] = 2**20
        frame.loc[frame.event == 2, "subrun"] = 102
        frame.loc[frame.event == 2, "event"] = 0
    result = audit.analyze(steps, parts, geom)
    assert result["primary_muons"]["histogram"] == {"1": 1, "2": 1}
    assert result["descendants"]["historical_wrong_ancestor"]["rows"] == 1
    assert result["descendants"]["intended_no_ancestor_crossing"]["rows"] == 2
    assert result["descendants"]["historical_unattached"]["rows"] == 1
    assert result["donor_local_nearmiss"]["event_only_excluded_muons"] == 1


@pytest.fixture
def real_input(tmp_path):
    counters = _tiny_counters(tmp_path / "counters.txt")
    geom = geometry.from_offline_config(counters)
    donor, _ = _frames(geom)
    path = _write(tmp_path / "donor.root", donor)
    return {"donors": [str(path)], "run_config": "run1a_v01", "counters": str(counters)}, path, donor


def test_real_schema_v2_empty_census(audit, tmp_path, real_input):
    config, donor, _ = real_input
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = audit.run(config_path, tmp_path / "audit.json")
    assert json.loads((tmp_path / "audit.json").read_text()) == result
    assert result["files"][0]["validation"]["associations_checked"] is True
    assert result["files"][0]["measurements"]["primary_muons"]["histogram"] == {"0": 2, "1": 1}
    census = result["files"][0]["measurements"]["identity_census"]
    assert all(v.get("parentless", [0, 0.0])[0] == 0 for v in census.values())


def test_real_schema_v2_multi_muon_samebar(audit, tmp_path, real_input):
    config, path, frames = real_input
    frames = deepcopy(frames)
    extra = frames["particles"].iloc[[0]].copy().assign(key=np.int64(10))
    frames["particles"] = pd.concat([frames["particles"], extra], ignore_index=True)
    extra_step = frames["steps"].iloc[[0]].copy().assign(sp=np.int64(10), t0=0., edep=3.)
    frames["steps"] = pd.concat([frames["steps"], extra_step], ignore_index=True)
    frames["events"].loc[frames["events"].event == 1, ["n_particles", "n_steps"]] = 3
    _write(path, frames)
    assert genassoc.validate_donor(path)["schema_version"] == 2
    result = audit.run(config, tmp_path / "multi.json")["files"][0]["measurements"]
    assert result["crossings"]["merged_event_bar"] == 1
    assert result["primary_muons"]["multi_muon_events"] == 1
    assert result["descendants"]["historical_wrong_ancestor"]["rows"] == 1
    assert result["descendants"]["historical_wrong_ancestor"]["edep"] == pytest.approx(.2)


def test_schema_gate_before_diagnostic_payload_read(audit, tmp_path, real_input, monkeypatch):
    config, path, frames = real_input
    frames["events"]["schema_version"] = np.uint32(1)
    _write(path, frames)

    def forbidden(*args, **kwargs):
        pytest.fail("payload read before strict admission")

    monkeypatch.setattr(audit.tables, "read", forbidden)
    out = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="schema_version=2"):
        audit.run(config, out)
    assert not out.exists()


@pytest.mark.parametrize("scenario", ["swapped", "noncrossing", "int64"])
def test_real_schema_v2_identity_cases(audit, tmp_path, real_input, scenario):
    config, path, frames = real_input
    frames = deepcopy(frames)
    original = frames["particles"].iloc[[0]]
    prim = original.copy().assign(key=np.int64(10))
    frames["particles"] = pd.concat([frames["particles"], prim], ignore_index=True)
    event_one = frames["events"].event == 1
    frames["events"].loc[event_one, "n_particles"] = 3
    if scenario == "swapped":
        second_step = frames["steps"].iloc[[0]].copy().assign(sp=np.int64(10), bar=np.int64(1))
        second_step[["x0", "y0", "z0"]] = frames["steps"].loc[1, ["x0", "y0", "z0"]].to_numpy()
        frames["steps"] = pd.concat([frames["steps"], second_step], ignore_index=True)
        frames["steps"].loc[frames["steps"].sp == 8, "bar"] = 2
        frames["events"].loc[event_one, "n_steps"] = 3
    else:
        frames["particles"].loc[frames["particles"].key == 8, "parent"] = np.int64(10)
    if scenario == "int64":
        for column in ("key", "parent"):
            values = frames["particles"][column].to_numpy(dtype=np.int64)
            frames["particles"][column] = np.where(values >= 0, values + 2**60, values)
        frames["steps"]["sp"] = frames["steps"].sp.to_numpy(dtype=np.int64) + 2**60
    _write(path, frames)
    result = audit.run(config, tmp_path / f"{scenario}.json")["files"][0]["measurements"]
    assert result["descendants"]["historical_wrong_ancestor"]["rows"] == 1
    assert result["descendants"]["historical_wrong_ancestor"]["edep"] == pytest.approx(.2)
    if scenario == "swapped":
        assert result["descendants"]["historical_wrong_ancestor_distance"]["rows"] == 1
        assert result["descendants"]["identity_safe_attached"]["rows"] == 1
    else:
        assert result["descendants"]["intended_no_ancestor_crossing"]["rows"] == 1
        assert result["descendants"]["historical_unattached"]["rows"] == 0
        assert result["descendants"]["event_only_nearmiss_excluded"]["edep"] == pytest.approx(.2)
        assert result["donor_local_nearmiss"]["event_only_excluded_muons"] == 1


def test_full_generator_multiplicity_is_separate_not_a_particle_join(audit, tmp_path, real_input):
    config, _, frames = real_input
    _, full = _frames(geometry.from_offline_config(config['counters']))
    extra = full['gen'].iloc[:1].copy().assign(idx=np.int64(1))
    full['gen'] = pd.concat([full['gen'], extra], ignore_index=True)
    full['events'].loc[0, 'n_gen'] += 1
    path = _write(tmp_path / 'full.root', full)
    config['generators'] = [str(path)]
    result = audit.run(config, tmp_path / 'full-audit.json')
    assert result['full_generator_multiplicity'][0]['multi_muon_events'] == 1
    assert result['full_generator_multiplicity'][0]['events'] == 6
    assert result['files'][0]['measurements']['primary_muons']['multi_muon_events'] == 0
