"""The tool functions end to end with a stand-in kit (no endpoint, no network)."""
import json
from pathlib import Path

import pytest

import fastsim_tools
from fastsim_tools import backend
from fastsim_tools import design_point as dp

pytest.importorskip("mu2efastsim.endpoints.geometry")


class FakeKit:
    """The kit contract (submit / status / results / check / cancel) with jobs that complete on demand."""

    def __init__(self, root: Path):
        self.root = root
        self.submitted, self.states, self.files_seen = {}, {}, {}

    def check(self, name, params, files, inputs):
        return {"ok": params["task"] in ("beam_edep_cpu", "beam_sob"), "message": ""}

    def submit(self, name, params, files, inputs):
        if name in self.submitted and self.submitted[name] != params:
            raise ValueError("request differs")
        if files:   # the kit's own descriptor and digest checks
            from mu2efastsim.endpoints import geometry
            geometry.resolve(geometry.reference(files, inputs, params))
        self.submitted[name] = params
        self.files_seen[name] = files
        self.states.setdefault(name, "working")
        jobs = self.root / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        (jobs / f"{name}.json").write_text(json.dumps({"job_id": str(1000 + len(self.submitted))}))
        return {"handle": name}

    def status(self, name):
        return {"state": self.states[name], "message": "RUNNING" if self.states[name] == "working" else "COMPLETED"}

    def results(self, name):
        out = self.root / "out" / name / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / dp.EDEP_FILE).write_bytes(b"root")
        return {"metrics": {"stops_per_pot": 1.25e-3, "ce_acceptance": 0.52, "sensitivity": 1.1},
                "files": [{"name": dp.EDEP_FILE, "uri": (out / dp.EDEP_FILE).as_uri(), "kind": "root"}],
                "metadata": {"job_id": "1001"}}

    def cancel(self, name):
        self.states[name] = "cancelled"
        return {"state": "cancelled"}


@pytest.fixture
def kit(tmp_path):
    k = FakeKit(tmp_path.resolve())
    backend.set_backend(backend.Backend(kit=k, state_dir=tmp_path))
    yield k
    backend.set_backend(None)


def test_exports_are_the_tools():
    assert fastsim_tools.__all__ == ["list_workflows", "describe_approximations", "run_design_point",
                                     "job_status", "fetch_outputs", "cancel_design_point"]


def test_list_workflows():
    r = fastsim_tools.list_workflows()
    assert r.status == "success" and len(r.metadata["workflows"]) == 9


def test_describe_approximations_chain():
    r = fastsim_tools.describe_approximations("edepana_writer", "fidelity")
    assert r.status == "success" and set(r.metadata["workflows"]) == {"W1", "W2-W4", "W7"}
    assert r.metadata["chain"] == "edepana_writer"
    assert fastsim_tools.describe_approximations("W9").status == "error"


def test_point_lifecycle_nominal(kit):
    r = fastsim_tools.run_design_point("p1", electrons=3000, seed=5)
    assert r.status == "submitted", r.message
    assert [j["handle"] for j in r.metadata["jobs"]] == ["p1.edep.float64", "p1.edep.float32"]
    assert all(j["job_id"] for j in r.metadata["jobs"])
    assert kit.submitted["p1.edep.float32"]["dtype"] == "float32"

    again = fastsim_tools.run_design_point("p1", electrons=3000, seed=5)
    assert again.status == "submitted" and again.metadata["jobs"] == r.metadata["jobs"]
    other = fastsim_tools.run_design_point("p1", electrons=4000, seed=5)
    assert other.status == "error" and "different request" in other.message

    s = fastsim_tools.job_status("p1")
    assert s.status == "working" and s.metadata["state"] == "working"
    assert fastsim_tools.fetch_outputs("p1").status == "working"

    kit.states["p1.edep.float64"] = "completed"
    part = fastsim_tools.fetch_outputs("p1")
    assert part.status == "success" and len(part.metadata["outputs"]) == 1 and part.metadata["pending"]

    kit.states["p1.edep.float32"] = "completed"
    assert fastsim_tools.job_status("p1").metadata["state"] == "completed"
    full = fastsim_tools.fetch_outputs("p1")
    assert full.status == "success" and {o["dtype"] for o in full.metadata["outputs"]} == {"float64", "float32"}
    call = full.metadata["outputs"][0]["score_with"]["arguments"]
    assert call["analysis"] == "approx_ce_sensitivity" and Path(call["data_file"]).exists()
    one = fastsim_tools.fetch_outputs("p1", dtype="float32")
    assert [o["dtype"] for o in one.metadata["outputs"]] == ["float32"]


def test_point_with_target_overlay(kit, tmp_path):
    target = {"radii": [75.0, 70.0], "zVars": [0.0, 30.0], "deltaZ": 30.0}
    r = fastsim_tools.run_design_point("p2", target=target, dtypes=["float64"])
    assert r.status == "submitted", r.message
    files = kit.files_seen["p2.edep.float64"]
    assert len(files) == 1 and files[0]["kind"] == "geom"
    assert "geom" not in kit.submitted["p2.edep.float64"]
    assert r.metadata["request"]["geometry"]["kind"] == "overlay"
    assert Path(dp.file_path(files[0]["uri"])).read_text() == dp.render_target(target)


def test_bad_target_is_an_error_message(kit):
    r = fastsim_tools.run_design_point("p3", target={"radii": [-1.0]})
    assert r.status == "error" and "positive" in r.message and not kit.submitted


def test_dry_run_submits_nothing(kit):
    r = fastsim_tools.run_design_point("p4", dry_run=True)
    assert r.status == "success" and set(r.metadata["checks"]) == {"p4.edep.float64", "p4.edep.float32"}
    assert not kit.submitted and fastsim_tools.job_status("p4").status == "error"


def test_sob_stage_single_job(kit):
    r = fastsim_tools.run_design_point("p5", stage="sob", electrons=20_000)
    assert r.status == "submitted" and [j["handle"] for j in r.metadata["jobs"]] == ["p5.sob"]
    kit.states["p5.sob"] = "completed"
    out = fastsim_tools.fetch_outputs("p5")
    assert out.metadata["outputs"][0]["sensitivity"] == 1.1


def test_schema_limits_refuse_before_the_kit(kit):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        fastsim_tools.run_design_point("p6", electrons=10)
    with pytest.raises(ValidationError):
        fastsim_tools.run_design_point("p6", dtypes=["float16"])
    assert fastsim_tools.run_design_point("p6", electrons=60_000).status == "error"   # edep CPU cap
    assert not kit.submitted


def test_cancel_and_unknown(kit):
    fastsim_tools.run_design_point("p7", dtypes=["float64"])
    assert fastsim_tools.cancel_design_point("p7").metadata["states"] == {"p7.edep.float64": "cancelled"}
    assert fastsim_tools.job_status("p7").metadata["state"] == "cancelled"
    assert "no design point" in fastsim_tools.job_status("nope").message


def test_stream_state_markers(tmp_path):
    be = backend.Backend(kit=object(), state_dir=tmp_path)
    assert be.stream_state("beam_edep_cpu", "h") == {"inputs_streamed": False}
    (tmp_path / "beam_edep_cpu").mkdir()
    (tmp_path / "beam_edep_cpu" / "h.streamed").write_text("t")
    assert be.stream_state("beam_edep_cpu", "h") == {"inputs_streamed": True}
    (tmp_path / "beam_edep_cpu" / "h.streamerr").write_text("md5 mismatch")
    assert be.stream_state("beam_edep_cpu", "h")["stream_error"] == "md5 mismatch"
