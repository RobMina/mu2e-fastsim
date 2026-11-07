"""Explicit-cohort campaign readers preserve empty events and unknown ancestry."""
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
    assert 'development) indices=({0..99})' in script
    assert 'heldout) indices=({100..199})' in script
    assert 'OFFLINE_WARM_REPEAT=0' in script
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


@pytest.mark.parametrize("index", [100, 199])
def test_explicit_heldout_reader_and_combine(tmp_path, index):
    root, _ = fixture(tmp_path, index=index)
    out = tmp_path / "heldout.json"
    report = ts.inspect_shard(root, index, out, generation_job=12345, cohort="heldout")
    assert report["cohort"] == "heldout"
    assert report["finished_art_events"] == 2
    summary = ts.combine([out], cohort="heldout")
    assert summary["cohort"] == "heldout"
    assert summary["indices"] == [index]
    assert summary["finished_art_events"] == 2
    with pytest.raises(ValueError, match="development"):
        ts.combine([out])


@pytest.mark.parametrize("index", [-1, 0, 99, 200])
def test_heldout_rejects_other_indices_before_reads(tmp_path, index):
    with pytest.raises(ValueError, match="heldout indices must be 100..199"):
        ts.inspect_shard(tmp_path / "absent", index, tmp_path / "out.json",
                         generation_job=12345, cohort="heldout")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("cohort", ["", "all", "Heldout"])
def test_invalid_cohort_rejected_before_reads(tmp_path, cohort):
    with pytest.raises(ValueError, match="cohort must be"):
        ts.inspect_shard(tmp_path / "absent", 100, tmp_path / "out.json",
                         generation_job=12345, cohort=cohort)
    with pytest.raises(ValueError, match="cohort must be"):
        ts.combine([tmp_path / "absent.json"], cohort=cohort)
    assert not (tmp_path / "out.json").exists()


def test_combine_rejects_mixed_or_mislabeled_cohorts(tmp_path):
    dev, heldout = tmp_path / "dev.json", tmp_path / "heldout.json"
    # Guard validation must occur before accessing any census payload.
    dev.write_text(json.dumps({"index": 0, "cohort": "development"}))
    heldout.write_text(json.dumps({"index": 100, "cohort": "heldout"}))
    for cohort in ("development", "heldout"):
        with pytest.raises(ValueError, match=f"{cohort} reports only"):
            ts.combine([dev, heldout], cohort=cohort)
    for index, label in ((99, "heldout"), (200, "heldout"), (100, "development"), (100, "other")):
        heldout.write_text(json.dumps({"index": index, "cohort": label}))
        with pytest.raises(ValueError, match="heldout reports only"):
            ts.combine([heldout], cohort="heldout")


def test_cli_passes_explicit_cohort(tmp_path, monkeypatch):
    import sys
    root, _ = fixture(tmp_path, index=100)
    out, summary = tmp_path / "heldout.json", tmp_path / "summary.json"
    for arguments, code in ((["--root", str(root), "--index", "100", "--generation-job", "12345",
                             "--out", str(out)], 1),
                            (["--combine", str(out), "--out", str(summary)], 1)):
        monkeypatch.setattr(sys, "argv", ["tscampaign", "--cohort", "heldout", *arguments])
        with pytest.raises(SystemExit) as exc:
            ts.main()
        assert exc.value.code == code  # Synthetic fixture has 2, not 5000, finished events.
    assert json.loads(summary.read_text())["cohort"] == "heldout"


@pytest.mark.parametrize("cohort,start", [(None, 0), ("development", 0), ("heldout", 100)])
def test_status_wrapper_selects_only_cohort_and_runs_once(tmp_path, cohort, start):
    import os
    import subprocess
    root = tmp_path / "campaign"
    (root / "config").mkdir(parents=True)
    (root / "logs").mkdir()
    for index in range(start, start + 100):
        (root / "logs" / f"12345_{index}.log").write_text("Art has completed and will exit with status 0.\n")
    # No raw data or real Offline process: the existing runner interface is captured.
    (root / "config/offline_uva.sh").write_text(
        'printf "%s\\n" "$OFFLINE_WARM_REPEAT" "$OFFLINE_EVENTS" "$@" >> "$CAPTURE"\n')
    bash_env = tmp_path / "bash_env"
    bash_env.write_text('module() { :; }\n')
    capture, out = tmp_path / "capture", tmp_path / "output"
    fcl = Path(ts.__file__).with_name("tsstatus.fcl")
    env = {**os.environ, "TS_CAMPAIGN_ROOT": str(root), "TS_GENERATION_JOB": "12345",
           "TS_STATUS_OUT": str(out), "TS_STATUS_FCL": str(fcl), "CVMFSEXEC_DIST": "unused",
           "BASH_ENV": str(bash_env), "CAPTURE": str(capture), "OFFLINE_WARM_REPEAT": "1"}
    env.pop("TS_STATUS_COHORT", None)
    if cohort is not None:
        env["TS_STATUS_COHORT"] = cohort
    subprocess.run(["bash", str(Path(ts.__file__).with_name("tsstatus.sbatch"))],
                   env=env, check=True, capture_output=True, text=True)
    lines = capture.read_text().splitlines()
    assert lines[:2] == ["0", "500000"]
    assert lines[2:4] == ["/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ac", str(out / "read.fcl")]
    assert lines[4:] == [str(root / "shards" / f"{index:03d}" /
                            f"sim.user.TSRun1ARaw.campaign_v1.001430_{920000 + index:08d}.art")
                         for index in range(start, start + 100)]
    assert (out / "read.fcl").read_bytes() == fcl.read_bytes()


@pytest.mark.parametrize("cohort", ["", "all", "Heldout"])
def test_status_wrapper_invalid_cohort_before_reads_or_output(tmp_path, cohort):
    import os
    import subprocess
    env = {**os.environ, "TS_STATUS_COHORT": cohort, "TS_CAMPAIGN_ROOT": str(tmp_path / "absent"),
           "TS_STATUS_OUT": str(tmp_path / "output"), "TS_GENERATION_JOB": "12345",
           "TS_STATUS_FCL": "absent.fcl", "CVMFSEXEC_DIST": "unused"}
    result = subprocess.run(["bash", str(Path(ts.__file__).with_name("tsstatus.sbatch"))],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stderr.strip() == "TS_STATUS_COHORT must be development or heldout"
    assert not (tmp_path / "output").exists()
