"""Development-only campaign readers preserve empty events and unknown ancestry."""
import json
from pathlib import Path

import numpy as np
import pytest
import uproot

from beamfastsim import tscampaign as ts
from test_tssmoke import particles, hits


def fixture(tmp_path, index=0):
    root = tmp_path / "campaign"
    shard = root / "shards" / f"{index:03d}"
    shard.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "logs" / f"12345_{index}.log").write_text("Art has completed and will exit with status 0.\n")
    art = shard / f"sim.user.TSRun1ARaw.campaign_v1.001430_{920000 + index:08d}.art"
    with uproot.recreate(art) as f:
        f.mktree("Events", {"event": np.array([1, 2])})
    with uproot.recreate(shard / "tsrun1a_raw.root") as f:
        for name, a in (("ntpart", particles()), ("ntvd", hits([2], [-211], [10])),
                        ("nttvd", hits([3], [13], [50]))):
            f.mktree(name, a)
    return root, art


def test_reader_counts_finished_events_not_request(tmp_path):
    root, _ = fixture(tmp_path)
    report = ts.inspect_shard(root, 0, tmp_path / "result.json", generation_job=12345)
    assert report["finished_art_events"] == 2
    assert not report["checks_passed"]
    assert not report["status_g4"]["decoded"]
    assert report["events_without_forward_mu_pi_in_ntpart"] == 1
    assert json.loads((tmp_path / "result.json").read_text())["index"] == 0


@pytest.mark.parametrize("index", [-1, 100, 199])
def test_heldout_not_opened(tmp_path, index):
    with pytest.raises(ValueError, match="development indices"):
        ts.inspect_shard(tmp_path / "absent", index, tmp_path / "result.json", generation_job=12345)


def test_unfinished_not_opened(tmp_path):
    root, _ = fixture(tmp_path)
    (root / "logs" / "12345_0.log").write_text("Begin processing\n")
    with pytest.raises(ValueError, match="no successful art"):
        ts.inspect_shard(root, 0, tmp_path / "result.json", generation_job=12345)


def test_combine_keeps_unknowns_and_rejects_duplicates(tmp_path):
    root, _ = fixture(tmp_path)
    out = tmp_path / "result.json"
    report = ts.inspect_shard(root, 0, out, generation_job=12345)
    assert ts.combine([out])["finished_art_events"] == 2
    report["source_tracks_without_ds_mu_pi_descendant"] = None
    out.write_text(json.dumps(report))
    assert ts.combine([out])["source_tracks_without_ds_mu_pi_descendant"] is None
    with pytest.raises(ValueError, match="distinct"):
        ts.combine([out, out])
    report["index"] = 100
    out.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="development"):
        ts.combine([out])


def test_status_object_fields(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    path = tmp_path / "status.art"
    name = "mu2e::StatusG4_g4run__TSRun1ARaw.obj"
    a = {"_status": np.array([0, 1, 10], dtype="i4"), "_nG4Tracks": np.array([20, 30, 40], dtype="i4"),
         "_overflowSimParticles": np.array([False, False, True]),
         "_nKilledStepLimit": np.array([0, 1, 0], dtype="i4"),
         "_nKilledByFieldPropagator": np.array([0, 0, 0], dtype="i4"),
         "_cpuTime": np.array([1, 2, 3], dtype="f4"), "_realTime": np.array([2, 3, 4], dtype="f4")}
    import awkward as ak
    events = MagicMock()
    events.keys.return_value = [name]
    events.__getitem__.return_value.array.side_effect = lambda **kwargs: ak.zip(a)
    monkeypatch.setattr(ts, "_tree", lambda *args: (events, {"tree": "Events;2"}))
    monkeypatch.setattr(ts.uproot, "open", MagicMock())
    report = ts.status_g4(path)
    assert report["decoded"], report
    assert report["status_counts"] == {"0": 1, "1": 1, "10": 1}
    assert report["overflow_events"] == 1
    a["_realTime"][0] = np.nan
    assert not ts.status_g4(path)["decoded"]


def test_stock_status_histogram_entries_include_overflows(tmp_path):
    path = tmp_path / "status.root"
    with uproot.recreate(path) as f:
        f["status/numG4Tracks"] = np.histogram([10, 20, 30], bins=[0, 100])
        f["status/statusValue"] = np.histogram([1, 10], bins=np.arange(21))
        f["status/overflowSimParticles"] = np.histogram([0], bins=[-.5, .5])
        f["status/hNKilledStep"] = np.histogram([2], bins=np.arange(101))
    report = ts.status_histograms(path)
    assert report["events"] == 3
    assert report["zero_status_events"] == 1
    assert report["nonzero_status_events"] == 2
    assert report["overflow_events"] == report["step_limit_events"] == 1
    assert len(report["histograms"]["statusValue"]["counts_with_flow"]) == 22


def test_status_reader_has_no_simulation_and_bounds_development():
    import subprocess
    root = Path(ts.__file__).parent
    fcl = (root / "tsstatus.fcl").read_text()
    script = (root / "tsstatus.sbatch").read_text()
    assert 'module_type: RootInput' in fcl
    assert 'module_type: StatusG4Analyzer' in fcl
    assert 'services.SeedService.baseSeed: 1' in fcl
    assert 'geom_run1_a.txt' in fcl
    assert 'producers:' not in fcl and 'Mu2eG4' not in fcl.split('physics:')[1]
    assert 'for index in {0..99}' in script
    assert 'OFFLINE_EVENTS=500000' in script
    assert 'muse build' not in script
    subprocess.run(['bash', '-n', root / 'tsstatus.sbatch'], check=True)


@pytest.mark.parametrize("job", [0, -1, "../job", "abc"])
def test_invalid_generation_job_rejected(tmp_path, job):
    with pytest.raises(ValueError, match="positive integer job ID"):
        ts.inspect_shard(tmp_path / "absent", 0, tmp_path / "result.json", generation_job=job)


def test_configurable_generation_log(tmp_path):
    root, _ = fixture(tmp_path)
    (root / "logs/12345_0.log").rename(root / "logs/98765_0.log")
    report = ts.inspect_shard(root, 0, tmp_path / "result.json", generation_job=98765)
    assert report["finished_art_events"] == 2
    script = Path(ts.__file__).with_name("tsstatus.sbatch").read_text()
    assert "${TS_GENERATION_JOB}_$index.log" in script
