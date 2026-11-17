"""GPU requests: the stage table against the kit, sizing, and the guard (allow_gpu and the configured cap)."""
import pytest

import fastsim_tools
from fastsim_tools import backend
from fastsim_tools import design_point as dp
from fastsim_tools.tools import GPU_CAP_VAR, gpu_hours_cap

from test_tools import FakeKit

pytest.importorskip("mu2efastsim.endpoints.geometry")


@pytest.fixture
def kit(tmp_path, monkeypatch):
    monkeypatch.delenv(GPU_CAP_VAR, raising=False)
    k = FakeKit(tmp_path.resolve())
    backend.set_backend(backend.Backend(kit=k, state_dir=tmp_path))
    yield k
    backend.set_backend(None)


def test_stage_requests_match_the_kit():
    """gpus and wall limits mirror the kit's tasks and, where the beam tasks stream (Slurm), their delegates."""
    pytest.importorskip("hependpoints")
    from mu2efastsim.endpoints import kit
    from beamfastsim import kit_uva
    delegates = {"beam_point": kit_uva.BeamPointStream, "beam_sob": kit_uva.BeamSobStream,
                 "beam_edep": kit_uva.BeamEdepStream, "beam_edep_cpu": kit_uva.BeamEdepCpuStream}
    for (stage, device), st in dp.STAGES.items():
        task = kit.TASKS[st.task]
        assert (task.resources.gpus, task.resources.minutes) == (st.gpus, st.minutes), st.task
        assert ("dtype" in task.params) == st.has_dtype, st.task
        assert ("electrons" in task.params) == (st.electrons is not None), st.task
        if st.task in delegates:
            r = delegates[st.task].RESOURCES
            assert (r.gpus, r.minutes) == (st.gpus, st.minutes), st.task


def test_gpu_request_size():
    reqs = dp.requests("g", "edep", ["float64", "float32"], 200_000, 0, 1, "nominal", "gpu")
    assert [r["handle"] for r in reqs] == ["g.edep.gpu.float64", "g.edep.gpu.float32"]
    assert {r["params"]["task"] for r in reqs} == {"beam_edep"}
    ranges = [dp.expected_minutes_range("edep", "gpu", 200_000, 0, r["dtype"]) for r in reqs]
    size = dp.gpu_request(reqs, 1, 60, ranges)
    assert size["jobs"] == 2 and size["gpu_hours_requested"] == 2.0
    lo, hi = size["gpu_hours_expected"]
    assert 0 < lo < hi <= 2.0
    unknown = dp.gpu_request(reqs, 1, 90, [None, None])
    assert unknown["gpu_hours_requested"] == 3.0 and unknown["gpu_hours_expected"] is None


def test_stops_stage_is_one_beam_point_job():
    (r,) = dp.requests("s", "stops", ["float64", "float32"], 5000, 0, 3, "nominal", "gpu")
    assert r == {"handle": "s.stops.gpu", "dtype": None,
                 "params": {"task": "beam_point", "muons": dp.BEAM_TABLE_MUONS, "seed": 3, "repeat": 0,
                            "geom": "nominal"}}


def test_cap_setting(monkeypatch):
    monkeypatch.delenv(GPU_CAP_VAR, raising=False)
    assert gpu_hours_cap() == 0
    monkeypatch.setenv(GPU_CAP_VAR, "2.5")
    assert gpu_hours_cap() == 2.5
    monkeypatch.setenv(GPU_CAP_VAR, "-1")
    assert gpu_hours_cap() == 0
    monkeypatch.setenv(GPU_CAP_VAR, "lots")
    with pytest.raises(ValueError):
        gpu_hours_cap()


def test_gpu_off_without_a_cap(kit):
    r = fastsim_tools.run_design_point("g1", device="gpu", electrons=200_000, allow_gpu=True)
    assert r.status == "error" and "GPU jobs are off" in r.message
    assert r.metadata["gpu_request"]["gpu_hours_requested"] == 2.0
    assert not kit.submitted


def test_needs_approval_then_submit(kit, monkeypatch):
    monkeypatch.setenv(GPU_CAP_VAR, "4")
    r = fastsim_tools.run_design_point("g2", device="gpu", electrons=200_000)
    assert r.status == "needs_approval" and "allow_gpu" in r.message
    assert r.metadata["gpu_request"]["jobs"] == 2 and not kit.submitted
    assert fastsim_tools.job_status("g2").status == "error"            # nothing recorded
    ok = fastsim_tools.run_design_point("g2", device="gpu", electrons=200_000, allow_gpu=True)
    assert ok.status == "submitted", ok.message
    assert set(kit.submitted) == {"g2.edep.gpu.float64", "g2.edep.gpu.float32"}
    assert kit.submitted["g2.edep.gpu.float32"]["task"] == "beam_edep"
    assert ok.metadata["gpu_request"]["gpu_hours_requested"] == 2.0 and "GPU request" in ok.message
    st = fastsim_tools.job_status("g2")
    assert st.metadata["jobs"][0]["inputs_streamed"] is False          # beam_edep streams like beam_edep_cpu


def test_cap_refuses_a_larger_request(kit, monkeypatch):
    monkeypatch.setenv(GPU_CAP_VAR, "1.5")
    r = fastsim_tools.run_design_point("g3", device="gpu", electrons=200_000, allow_gpu=True)
    assert r.status == "error" and "exceeds the configured cap" in r.message and not kit.submitted
    one = fastsim_tools.run_design_point("g3", device="gpu", electrons=200_000, dtypes=["float64"], allow_gpu=True)
    assert one.status == "submitted" and list(kit.submitted) == ["g3.edep.gpu.float64"]


def test_gpu_dry_run_sizes_and_checks(kit, monkeypatch):
    monkeypatch.setenv(GPU_CAP_VAR, "4")
    r = fastsim_tools.run_design_point("g4", device="gpu", stage="stops", dry_run=True)
    assert r.status == "success" and r.metadata["gpu_request"]["gpu_hours_requested"] == 0.5
    assert set(r.metadata["checks"]) == {"g4.stops.gpu"} and not kit.submitted


def test_kit_refusal_stops_a_gpu_request(kit, monkeypatch):
    monkeypatch.setenv(GPU_CAP_VAR, "4")
    r = fastsim_tools.run_design_point("g5", device="gpu", stage="centuple", electrons=20_000, allow_gpu=True)
    assert r.status == "error" and "FNAL storage" in r.message and not kit.submitted


def test_stage_device_mismatch(kit, monkeypatch):
    monkeypatch.setenv(GPU_CAP_VAR, "4")
    assert "no gpu task" in fastsim_tools.run_design_point("g6", device="gpu", stage="sob").message
    assert "no cpu task" in fastsim_tools.run_design_point("g6", stage="stops").message
    assert "electrons" in fastsim_tools.run_design_point("g6", device="gpu", stage="centuple",
                                                         electrons=500_000, allow_gpu=True).message
    assert not kit.submitted


def test_stops_outputs(kit, monkeypatch):
    monkeypatch.setenv(GPU_CAP_VAR, "1")
    assert fastsim_tools.run_design_point("g7", device="gpu", stage="stops", allow_gpu=True).status == "submitted"
    kit.states["g7.stops.gpu"] = "completed"
    out = fastsim_tools.fetch_outputs("g7")
    assert out.status == "success" and "stopping-target" in out.metadata["outputs"][0]["note"]


def test_crv_chain_point(kit, monkeypatch):
    monkeypatch.setenv(GPU_CAP_VAR, "3")
    r = fastsim_tools.run_crv_chain("c1", shards=4, wall_minutes=40)
    assert r.status == "needs_approval"
    size = r.metadata["gpu_request"]
    assert size["jobs"] == 4 and size["sub_jobs"] == 4 and size["gpu_hours_requested"] == round(4 * 40 / 60, 2)
    lo, hi = size["gpu_hours_expected"]
    assert 0 < lo <= hi <= size["gpu_hours_requested"]
    ok = fastsim_tools.run_crv_chain("c1", shards=4, wall_minutes=40, allow_gpu=True)
    assert ok.status == "submitted", ok.message
    assert kit.submitted["c1.crv"]["task"] == "chain_point" and kit.submitted["c1.crv"]["shards"] == 4
    assert fastsim_tools.job_status("c1").metadata["jobs"][0]["inputs_streamed"] is None
    big = fastsim_tools.run_crv_chain("c2", shards=8, wall_minutes=60, allow_gpu=True)
    assert big.status == "error" and "exceeds" in big.message


def test_crv_chain_task_limits(kit, monkeypatch):
    monkeypatch.setenv(GPU_CAP_VAR, "3")
    assert "shards 1-10" in fastsim_tools.run_crv_chain("c3", task="chain", shards=11).message
    r = fastsim_tools.run_crv_chain("c3", task="chain", shards=2)
    assert r.status == "needs_approval" and r.metadata["gpu_request"]["gpu_hours_expected"] is None
    assert not kit.submitted


def test_cpu_path_unchanged(kit):
    r = fastsim_tools.run_design_point("p_cpu", electrons=3000)
    assert r.status == "submitted" and "gpu_request" not in r.metadata
    assert [j["handle"] for j in r.metadata["jobs"]] == ["p_cpu.edep.float64", "p_cpu.edep.float32"]
    assert "device" not in r.metadata["request"]
