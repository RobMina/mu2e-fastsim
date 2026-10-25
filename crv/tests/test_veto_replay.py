"""Raw-time replay tables and backward-compatible mixer output."""
import json

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto.mixframes import mix
from crvfastsim.veto.replay import STEP_TYPES, export, replay_tables, write_replay


def sample():
    ev = pd.DataFrame({"run": [1430, 1430], "subrun": [0, 0], "event": [1, 2],
                       "pbtime": [-224.46, -204.46], "intensity": [1.5e7, 2e7]})
    st = pd.DataFrame({k: np.zeros(2, dtype=v) for k, v in STEP_TYPES.items()})
    st["run"], st["event"], st["sp"], st["pdg"] = 1430, 1, 60000001, 13
    st["t0"], st["t1"] = [-10.0, 3400.0], [-9.0, 3401.0]
    return st, ev


def test_raw_times_particle_mapping_and_stepless_event(tmp_path):
    st, ev = sample()
    trees = replay_tables(st, ev)
    assert trees["steps"].sp.tolist() == [1, 1]
    assert trees["particles"].key.tolist() == [1]
    np.testing.assert_array_equal(trees["steps"].t0, st.t0)
    path = tmp_path / "replay.root"
    assert write_replay(path, trees) == {"steps": 2, "particles": 1, "events": 2}
    with uproot.open(path) as f:
        assert f["dump/steps"].classname == "TTree"
        assert f["dump/steps/run"].typename == "uint32_t"
        assert f["dump/steps/sp"].typename == "int64_t"
        assert f["dump/steps/edep"].typename == "double"
        assert f["dump/particles/pdg"].typename == "int32_t"
        assert f["dump/events"].num_entries == 2
    with pytest.raises(FileExistsError):
        write_replay(path, trees)


def test_entirely_stepless_sample(tmp_path):
    _, ev = sample()
    trees = replay_tables(pd.DataFrame(), ev)
    assert write_replay(tmp_path / "empty.root", trees) == {"steps": 0, "particles": 0, "events": 2}


def test_mixer_optional_events_does_not_change_draws():
    old = mix({}, 10, np.random.default_rng(17))
    new = mix({}, 10, np.random.default_rng(17), return_events=True)
    assert len(old) == 4 and len(new) == 5
    np.testing.assert_array_equal(old[1], new[1])
    np.testing.assert_array_equal(old[2], new[2])
    assert old[3] == new[3]
    np.testing.assert_array_equal(-new[4].pbtime, new[2])
    assert new[4].intensity.mean() == new[3]["mean_intensity"]


@pytest.mark.parametrize("case,match", [("missing", "absent"), ("pdg", "inconsistent"),
                                        ("negative", "known"), ("duplicate", "duplicate"),
                                        ("nan", "finite")])
def test_invalid_replay(case, match):
    st, ev = sample()
    if case == "missing":
        st["event"] = 3
    elif case == "pdg":
        st.loc[1, "pdg"] = 11
    elif case == "negative":
        st["sp"] = -1
    elif case == "duplicate":
        ev.loc[1, "event"] = 1
    elif case == "nan":
        st.loc[0, "edep"] = np.nan
    with pytest.raises(ValueError, match=match):
        replay_tables(st, ev)


def test_export_manifest_and_all_empty_pools(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from crvfastsim.veto.mixframes import MIXERS
    for name in MIXERS:
        folder = tmp_path / "tables" / name
        folder.mkdir(parents=True)
        table = pa.Table.from_pandas(pd.DataFrame({"subrun": pd.Series(dtype="int64"),
                                                 "event": pd.Series(dtype="int64")}))
        meta = {"dataset": name, "frames": 10, "stepped_frames": 0, "steps": 0}
        table = table.replace_schema_metadata({b"crvfastsim": json.dumps(meta).encode()})
        pq.write_table(table, folder / "steps.parquet")
    out = tmp_path / "sample.root"
    report = export(tmp_path / "tables", 5, 1234, out)
    assert report["entries"] == {"steps": 0, "particles": 0, "events": 5}
    assert report["raw_times"] is True
    assert set(report["sources"]) == set(MIXERS)
    assert len(report["output_sha256"]) == 64
    with pytest.raises(ValueError, match="new .root"):
        export(tmp_path / "tables", 5, 1234, out)


def test_nan_times_preserved_for_both_response_skip_rules():
    st, ev = sample()
    st.loc[0, "t1"] = np.nan
    trees = replay_tables(st, ev)
    assert len(trees["steps"]) == 2
    assert np.isnan(trees["steps"].t1.iloc[0])
    st.loc[0, "t1"] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        replay_tables(st, ev)
