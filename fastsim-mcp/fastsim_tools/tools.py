"""The MCP tool functions. Plain Python: the type hints, Field constraints and docstrings become the tool schema.

Discovery (list_workflows, describe_approximations) reads the repository and needs no compute. A design point
(run_design_point) is asynchronous: it submits one CPU job per dtype through the fastsim kit and returns at once;
job_status polls the jobs and fetch_outputs brings back the files and metrics once they complete.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validate_call

from . import approximations_doc, catalog
from . import backend as _backend
from . import design_point as dp

Dtype = Literal["float64", "float32"]


class ToolResult(BaseModel):
    """What every tool returns: a status, a one-paragraph message, output files, and structured metadata."""
    status: Literal["success", "submitted", "working", "error"]
    message: str
    files: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def repo_root() -> Path:
    """The mu2e-fastsim checkout this package sits in; $M2FS_CODE when the package is installed elsewhere."""
    here = Path(__file__).resolve().parents[2]
    if (here / "docs" / "approximations.md").exists() or not os.environ.get("M2FS_CODE"):
        return here
    return Path(os.environ["M2FS_CODE"]).expanduser()


def _error(exc: Exception) -> ToolResult:
    return ToolResult(status="error", message=f"{type(exc).__name__}: {exc}")


@validate_call
def list_workflows() -> ToolResult:
    """List the Mu2e fast-simulation workflows: what each starts from and hands on, and how to run it.

    Call this first. Each entry gives the workflow's boundary (input -> output), the repository packages and
    command lines that implement it, the approximation sets that apply (pass the key to describe_approximations),
    the documents holding its validation and rates, the fastsimkit tasks that run it on a compute endpoint, and
    whether run_design_point exposes it. Workflows: W1 beam (TS entrance -> stopped muons), W2-W4 tracker, W5 CRV,
    W6 pileup mixing, W7 calorimeter, W8 trigger, the conversion-electron chain, and the EdepAna and EventNtuple
    writers whose files the Mu2e analysis server reads unchanged.
    """
    entries = catalog.describe_all()
    runnable = [w["key"] for w in entries if w["run_design_point"]]
    return ToolResult(
        status="success",
        message=(f"{len(entries)} workflows: {', '.join(w['key'] for w in entries)}. run_design_point runs a "
                 f"stopping-target design point through {', '.join(runnable)} on CPU; rates and validity are in "
                 "the documents each entry names."),
        metadata={"workflows": entries, "repository": str(repo_root())})


@validate_call
def describe_approximations(
    workflow: Annotated[Optional[str], Field(description="W1, W2-W4 (or W2/W3/W4), W5, W6, W7, W8, or a chain "
                                                         "from list_workflows (ce_chain, edepana_writer, ...); "
                                                         "omit for all")] = None,
    preset: Annotated[Optional[Literal["default", "fast", "fidelity"]],
                      Field(description="also report the level each switchable approximation takes under this "
                                        "preset")] = None,
) -> ToolResult:
    """Describe the approximations a fast workflow makes in place of the stock simulation, with their switches.

    For each approximation: its levels (default first), the level the fast and fidelity presets select, what
    gives the stock result (an exact port, stock art inputs, or none yet), and the command-line switch. Fixed
    approximations (no switch yet) are listed with what stock does instead, and the per-workflow notes of
    docs/approximations.md carry the measured fidelity cost of the opt-in levels. A chain key (ce_chain,
    edepana_writer, eventntuple_writer) returns the approximations of every workflow it runs.
    """
    try:
        text = approximations_doc.read(repo_root())
        if workflow is None:
            d = approximations_doc.describe(text, None, preset)
        else:
            w = catalog.find(workflow)
            parts = [approximations_doc.describe(text, a, preset)["workflows"] for a in w.approximations]
            d = approximations_doc.describe(text, w.approximations[0], preset)
            d["workflows"] = {k: v for part in parts for k, v in part.items()}
            d["chain"] = w.key
            if w.notes:
                d["chain_notes"] = list(w.notes)
    except (OSError, ValueError) as exc:
        return _error(exc)
    counts = {k: (len(v["switchable"]), len(v["fixed"])) for k, v in d["workflows"].items()}
    summary = "; ".join(f"{k}: {s} switchable, {f} fixed" for k, (s, f) in counts.items())
    return ToolResult(status="success", message=f"Approximations ({summary}). Source: docs/approximations.md.",
                      files=[{"name": "approximations.md", "uri": (repo_root() / "docs" / "approximations.md").as_uri(),
                              "kind": "text"}],
                      metadata=d)


@validate_call
def run_design_point(
    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_-]*$",
                               description="a name for the point, used as the job handle prefix")],
    target: Annotated[Optional[Dict[str, Any]],
                      Field(description="a new stopping-target design: radii, halfThicknesses, zVars, holeRadii "
                                        "(lists in mm, one per foil); deltaZ, z0InMu2e, holeRadius, "
                                        "distFromTargetEnd (mm); hasTSdA false. Overrides the nominal geometry; "
                                        "omit to use `geometry`")] = None,
    geometry: Annotated[str, Field(description="'nominal' or a geometry file under beam/geom (ignored with "
                                               "target)")] = "nominal",
    stage: Annotated[Literal["edep", "sob"],
                     Field(description="edep: stops + CE chain + the EdepAna file for approx_ce_sensitivity (one "
                                       "CPU core per dtype); sob: stops + CE chain + the built-in approximate "
                                       "sensitivity (one 32-core CPU job, no dtype switch)")] = "edep",
    electrons: Annotated[int, Field(ge=1000, le=200_000, description="conversion electrons from the stops "
                                                                     "(edep: at most 50,000)")] = 5000,
    muons: Annotated[int, Field(ge=0, le=10 ** 7, description="beam muons to transport; 0 = the whole beam "
                                                              "table")] = 0,
    seed: Annotated[int, Field(ge=0, lt=2 ** 31)] = 1,
    dtypes: Annotated[List[Dtype], Field(min_length=1, max_length=2,
                                         description="device precision(s); each runs as its own job")]
    = ["float64", "float32"],
    dry_run: Annotated[bool, Field(description="check the request with the kit without submitting")] = False,
) -> ToolResult:
    """Run one stopping-target design point through the fast chain: stops -> CE chain -> S/sqrt(B)-ready outputs.

    Asynchronous: submits CPU-only jobs to the configured compute endpoint and returns their handles and job ids
    at once. A point takes minutes to hours on one core (about 20 minutes for 5,000 electrons in float64); poll
    job_status(name), then fetch_outputs(name). Keep this server running until job_status reports
    inputs_streamed: the job waits up to 15 minutes for the server to stream its Mu2e-internal inputs in.
    Submitting the same name and request again returns the same jobs; a different request under an existing
    name is refused. stage 'edep' returns, per dtype, the EdepAna file and the exact approx_ce_sensitivity call
    for the Mu2e analysis server.
    """
    try:
        dp.check_name(name)
        be = _backend.get()
        files: List[Dict[str, Any]] = []
        geom: Optional[Dict[str, Any]]
        if target is not None:
            text = dp.render_target(target)
            from mu2efastsim.endpoints import geometry as kit_geometry   # the kit's own admission rules
            kit_geometry.validate_text(text.encode("ascii"))
            ref = dp.file_ref("", text)
            path = be.write_geometry(text, ref["sha256"])
            ref["uri"] = path.as_uri()
            files = [ref]
            geom = {"kind": "overlay", "sha256": ref["sha256"], "text": text, "path": str(path)}
            reqs = dp.requests(name, stage, dtypes, electrons, muons, seed, None)
        else:
            geom = {"kind": "nominal" if geometry == "nominal" else "file", "name": dp.check_geometry(geometry)}
            reqs = dp.requests(name, stage, dtypes, electrons, muons, seed, geometry)
        request = {"stage": stage, "electrons": electrons, "muons": muons, "seed": seed,
                   "dtypes": [r["dtype"] for r in reqs], "geometry": {k: v for k, v in geom.items() if k != "path"}}
        old = be.load_point(name)
        if old is not None and old["request"] != request:
            raise ValueError(f"point {name!r} already exists with a different request; choose another name")
        if dry_run:
            checks = {r["handle"]: be.check(r["handle"], r["params"], files) for r in reqs}
            ok = all(c.get("ok") for c in checks.values())
            return ToolResult(status="success" if ok else "error",
                              message=("the kit admits " if ok else "the kit refuses ") + ", ".join(checks),
                              metadata={"checks": checks, "request": request})
        jobs = []
        for r in reqs:
            be.submit(r["handle"], r["params"], files)
            rec = be.job_record(r["handle"])
            jobs.append({"handle": r["handle"], "dtype": r["dtype"], "task": r["params"]["task"],
                         "params": r["params"], "job_id": rec.get("job_id"),
                         "expected_minutes": dp.expected_minutes(stage, electrons, r["dtype"])})
        record = old or {"name": name, "created": _backend.now(), "request": request, "jobs": jobs}
        for j, new in zip(record["jobs"], jobs):
            j["job_id"] = j.get("job_id") or new["job_id"]
        be.save_point(record)
    except Exception as exc:   # every failure goes back to the agent as a message
        return _error(exc)
    ids = ", ".join(f"{j['handle']} (job {j['job_id']})" for j in record["jobs"])
    return ToolResult(status="submitted",
                      message=(f"design point {name!r} submitted: {ids}. Expected run time once started: "
                               f"{', '.join(str(j['expected_minutes']) + ' min' for j in record['jobs'])}, plus "
                               "the queue. Poll job_status, then fetch_outputs."),
                      metadata=record)


@validate_call
def job_status(
    name: Annotated[str, Field(min_length=1, description="the design point name given to run_design_point")],
) -> ToolResult:
    """The state of a design point's jobs: working, completed, failed or cancelled, with the site's own state.

    Also reports, per job, whether the server has streamed the job's inputs in (inputs_streamed): until then the
    server must stay running. Poll at most once a minute; when the point is completed call fetch_outputs.
    """
    try:
        be = _backend.get()
        record = be.load_point(name)
        if record is None:
            known = be.list_points()
            raise ValueError(f"no design point {name!r}; known: {', '.join(known) or 'none'}")
        jobs = []
        for j in record["jobs"]:
            s = be.status(j["handle"])
            rec = be.job_record(j["handle"])
            jobs.append({"handle": j["handle"], "dtype": j["dtype"], "state": s.get("state"),
                         "message": s.get("message", ""), "job_id": rec.get("job_id") or j.get("job_id"),
                         **be.stream_state(j["task"], j["handle"])})
    except Exception as exc:
        return _error(exc)
    state = dp.overall_state(j["state"] for j in jobs)
    status = {"completed": "success", "working": "working"}.get(state, "error")
    text = "; ".join(f"{j['dtype'] or j['handle']}: {j['state']} ({j['message']})" for j in jobs)
    return ToolResult(status=status, message=f"{name}: {state}. {text}",
                      metadata={"name": name, "state": state, "jobs": jobs, "request": record["request"]})


@validate_call
def fetch_outputs(
    name: Annotated[str, Field(min_length=1, description="the design point name given to run_design_point")],
    dtype: Annotated[Optional[Dtype], Field(description="one dtype only; omit for every completed job")] = None,
) -> ToolResult:
    """Fetch a completed design point's outputs: metrics, the output files, and how to score them.

    For stage 'edep', per dtype: the EdepAna file (nts.mu2efastsim.edep.root, EDepAna/tree) with its .edep.json
    run record and edep_summary.txt, the point's metrics (stops_per_pot, CE acceptance, stored events, timings),
    and score_with: the run_analysis call of the Mu2e analysis server's approx_ce_sensitivity on that file. For
    stage 'sob': the built-in approximate sensitivity and its factors. Jobs that are not completed are listed with
    their state.
    """
    try:
        be = _backend.get()
        record = be.load_point(name)
        if record is None:
            raise ValueError(f"no design point {name!r}; known: {', '.join(be.list_points()) or 'none'}")
        stage = record["request"]["stage"]
        outputs, pending, files = [], [], []
        for j in record["jobs"]:
            if dtype is not None and j["dtype"] != dtype:
                continue
            s = be.status(j["handle"])
            if s.get("state") != "completed":
                pending.append({"handle": j["handle"], "dtype": j["dtype"], "state": s.get("state"),
                                "message": s.get("message", "")})
                continue
            summary = dp.summarise(stage, j["dtype"], be.results(j["handle"]))
            summary["handle"] = j["handle"]
            outputs.append(summary)
            files += [{**f, "dtype": j["dtype"]} for f in summary["files"]]
    except Exception as exc:
        return _error(exc)
    if not outputs:
        return ToolResult(status="working" if any(p["state"] == "working" for p in pending) else "error",
                          message=f"{name}: no completed job yet; " + "; ".join(
                              f"{p['dtype'] or p['handle']}: {p['state']}" for p in pending),
                          metadata={"name": name, "pending": pending})
    key = "sensitivity" if stage == "sob" else "ce_acceptance"
    text = ", ".join(f"{o['dtype'] or 'cpu'}: {key} {o['metrics'].get(key)}, stops_per_pot "
                     f"{o['metrics'].get('stops_per_pot')}" for o in outputs)
    return ToolResult(status="success",
                      message=f"{name}: {len(outputs)} job(s) fetched ({text})."
                              + (f" Still pending: {', '.join(p['handle'] for p in pending)}." if pending else ""),
                      files=files, metadata={"name": name, "stage": stage, "outputs": outputs, "pending": pending})


@validate_call
def cancel_design_point(
    name: Annotated[str, Field(min_length=1, description="the design point name given to run_design_point")],
) -> ToolResult:
    """Cancel a design point's jobs that are still queued or running (completed jobs are left as they are)."""
    try:
        be = _backend.get()
        record = be.load_point(name)
        if record is None:
            raise ValueError(f"no design point {name!r}")
        states = {j["handle"]: be.cancel(j["handle"]).get("state") for j in record["jobs"]}
    except Exception as exc:
        return _error(exc)
    return ToolResult(status="success", message=f"{name}: " + ", ".join(f"{h} {s}" for h, s in states.items()),
                      metadata={"name": name, "states": states})
