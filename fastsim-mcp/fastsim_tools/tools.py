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
    status: Literal["success", "submitted", "needs_approval", "working", "error"]
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
        message=(f"{len(entries)} workflows: {', '.join(w['key'] for w in entries)}. Runnable here: "
                 f"{', '.join(runnable)} (run_design_point on CPU by default or GPU, run_crv_chain on GPU; GPU "
                 "requests are sized and need allow_gpu); rates and validity are in the documents each entry "
                 "names."),
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


GPU_CAP_VAR = "FASTSIM_MCP_GPU_HOURS_CAP"


def gpu_hours_cap() -> float:
    """The most GPU-hours (at the jobs' wall-time limits) one tool call may request: ${GPU_CAP_VAR}, default 0 (GPU
    jobs off). A site setting, so an agent cannot raise it."""
    raw = os.environ.get(GPU_CAP_VAR, "0") or "0"
    try:
        cap = float(raw)
    except ValueError:
        raise ValueError(f"{GPU_CAP_VAR}={raw!r} is not a number") from None
    return cap if cap > 0 else 0.0


def _gpu_guard(size: Dict[str, Any], allow_gpu: bool, dry_run: bool) -> Optional[ToolResult]:
    """None when a GPU request may go ahead; otherwise the reply that stops it (with the request size)."""
    cap = gpu_hours_cap()
    meta = {"gpu_request": size, "gpu_hours_cap": cap}
    want = size["gpu_hours_requested"]
    text = (f"{size['jobs']} GPU job(s) of {size['gpus_per_job']} {size['gpu_type']}, "
            f"{want} GPU-h at the wall limits")
    if size["gpu_hours_expected"]:
        lo, hi = size["gpu_hours_expected"]
        text += f", {lo}-{hi} GPU-h expected"
    if cap == 0:
        return ToolResult(status="error", metadata=meta,
                          message=f"GPU jobs are off on this server ({GPU_CAP_VAR} unset or 0). Request: {text}.")
    if want > cap:
        return ToolResult(status="error", metadata=meta,
                          message=f"request exceeds the configured cap of {cap} GPU-h ({GPU_CAP_VAR}): {text}.")
    if not allow_gpu and not dry_run:
        return ToolResult(status="needs_approval", metadata=meta,
                          message=(f"not submitted: {text}. Get the user's approval for this request, then call "
                                   "again with allow_gpu=true."))
    return None


def _submit(name: str, request: Dict[str, Any], reqs: List[Dict[str, Any]], files: List[Dict[str, Any]],
            expected: Dict[str, Any], size: Optional[Dict[str, Any]], allow_gpu: bool, dry_run: bool) -> ToolResult:
    """Shared by the run tools: refuse a changed request, size and guard GPU work, check, submit, record."""
    be = _backend.get()
    old = be.load_point(name)
    if old is not None and old["request"] != request:
        raise ValueError(f"point {name!r} already exists with a different request; choose another name")
    if size is not None and old is None:
        stop = _gpu_guard(size, allow_gpu, dry_run)
        if stop is not None:
            stop.metadata["request"] = request
            return stop
    if dry_run or size is not None:   # GPU requests are always checked with the kit first
        checks = {r["handle"]: be.check(r["handle"], r["params"], files) for r in reqs}
        ok = all(c.get("ok") for c in checks.values())
        if dry_run or not ok:
            bad = {h: c.get("message", "") for h, c in checks.items() if not c.get("ok")}
            return ToolResult(status="success" if ok else "error",
                              message=(("the kit admits " + ", ".join(checks)) if ok else
                                       "the kit refuses: " + "; ".join(f"{h}: {m}" for h, m in bad.items())),
                              metadata={"checks": checks, "request": request, "gpu_request": size})
    jobs = []
    for r in reqs:
        be.submit(r["handle"], r["params"], files)
        rec = be.job_record(r["handle"])
        jobs.append({"handle": r["handle"], "dtype": r["dtype"], "task": r["params"]["task"],
                     "params": r["params"], "job_id": rec.get("job_id"),
                     "expected_minutes": expected.get(r["handle"])})
    record = old or {"name": name, "created": _backend.now(), "request": request, "jobs": jobs}
    for j, new in zip(record["jobs"], jobs):
        j["job_id"] = j.get("job_id") or new["job_id"]
    if size is not None:
        record.setdefault("gpu_request", size)
    be.save_point(record)
    ids = ", ".join(f"{j['handle']} (job {j['job_id']})" for j in record["jobs"])
    mins = [j.get("expected_minutes") for j in record["jobs"]]
    when = (", ".join(f"{m} min" for m in mins) if all(m is not None for m in mins) else "not measured")
    return ToolResult(status="submitted",
                      message=(f"{name!r} submitted: {ids}. Expected run time once started: {when}, plus the queue."
                               + (f" GPU request: {size['gpu_hours_requested']} GPU-h at the wall limits."
                                  if size else "") + " Poll job_status, then fetch_outputs."),
                      metadata=record)


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
    stage: Annotated[Literal["edep", "sob", "stops", "centuple"],
                     Field(description="edep: stops + CE chain + the EdepAna file for approx_ce_sensitivity (cpu: "
                                       "one core per dtype; gpu: one GPU per dtype); sob: stops + CE chain + the "
                                       "built-in approximate sensitivity (cpu only, one 32-core job, no dtype "
                                       "switch); stops: stopping-target metrics only (gpu only, no dtype switch); "
                                       "centuple: the fast EventNtuple file (gpu only; needs an endpoint with "
                                       "/cvmfs and FNAL storage)")] = "edep",
    device: Annotated[Literal["cpu", "gpu"],
                      Field(description="cpu (default) or gpu. A gpu request is sized first and goes ahead only "
                                        "with allow_gpu=true and within the server's configured GPU-hour cap")]
    = "cpu",
    electrons: Annotated[int, Field(ge=1000, le=2_000_000, description="conversion electrons from the stops "
                                                                       "(edep cpu: at most 50,000; edep gpu: "
                                                                       "2,000,000; sob, centuple: 200,000)")]
    = 5000,
    muons: Annotated[int, Field(ge=0, le=10 ** 7, description="beam muons to transport; 0 = the whole beam "
                                                              "table")] = 0,
    seed: Annotated[int, Field(ge=0, lt=2 ** 31)] = 1,
    dtypes: Annotated[List[Dtype], Field(min_length=1, max_length=2,
                                         description="device precision(s); each runs as its own job")]
    = ["float64", "float32"],
    allow_gpu: Annotated[bool, Field(description="submit GPU jobs. Set it only after the user approved the GPU "
                                                 "request this tool returned without it")] = False,
    dry_run: Annotated[bool, Field(description="size the request and check it with the kit without "
                                               "submitting")] = False,
) -> ToolResult:
    """Run one stopping-target design point through the fast chain: stops -> CE chain -> S/sqrt(B)-ready outputs.

    Asynchronous: submits jobs to the configured compute endpoint and returns their handles and job ids at once.
    CPU (default): about 20 minutes for 5,000 electrons per core in float64. GPU (device='gpu'): the reply first
    gives the request size (jobs, GPU-hours at the wall limits and expected) and status needs_approval; nothing is
    submitted until the call is repeated with allow_gpu=true, and never beyond the server's GPU-hour cap.
    Poll job_status(name), then fetch_outputs(name). Keep this server running until job_status reports
    inputs_streamed: the job waits up to 15 minutes for the server to stream its Mu2e-internal inputs in.
    Submitting the same name and request again returns the same jobs; a different request under an existing
    name is refused. stage 'edep' returns, per dtype, the EdepAna file and the exact approx_ce_sensitivity call
    for the Mu2e analysis server.
    """
    try:
        dp.check_name(name)
        st = dp.stage(stage, device)
        be = _backend.get()
        files: List[Dict[str, Any]] = []
        if target is not None:
            text = dp.render_target(target)
            from mu2efastsim.endpoints import geometry as kit_geometry   # the kit's own admission rules
            kit_geometry.validate_text(text.encode("ascii"))
            ref = dp.file_ref("", text)
            path = be.write_geometry(text, ref["sha256"])
            ref["uri"] = path.as_uri()
            files = [ref]
            geom = {"kind": "overlay", "sha256": ref["sha256"], "text": text}
            reqs = dp.requests(name, stage, dtypes, electrons, muons, seed, None, device)
        else:
            geom = {"kind": "nominal" if geometry == "nominal" else "file", "name": dp.check_geometry(geometry)}
            reqs = dp.requests(name, stage, dtypes, electrons, muons, seed, geometry, device)
        request = {"stage": stage, "electrons": electrons if st.electrons else None, "muons": muons, "seed": seed,
                   "dtypes": [r["dtype"] for r in reqs], "geometry": geom}
        if device == "gpu":
            request["device"] = "gpu"
        ranges = {r["handle"]: dp.expected_minutes_range(stage, device, electrons, muons, r["dtype"]) for r in reqs}
        expected = {h: (None if v is None else round(sum(v) / 2, 1)) for h, v in ranges.items()}
        size = (dp.gpu_request(reqs, st.gpus, st.minutes, ranges.values()) if device == "gpu" else None)
        return _submit(name, request, reqs, files, expected, size, allow_gpu, dry_run)
    except Exception as exc:   # every failure goes back to the agent as a message
        return _error(exc)


@validate_call
def run_crv_chain(
    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_-]*$",
                               description="a name for the point, used as the job handle prefix")],
    task: Annotated[Literal["chain_point", "chain"],
                    Field(description="chain_point: the whole CRV chain as `shards` one-GPU jobs with the inputs "
                                      "streamed in (Slurm endpoints); chain: one GPU job reading the inputs in "
                                      "place (endpoints that mount FNAL storage)")] = "chain_point",
    shards: Annotated[int, Field(ge=1, le=32, description="chain_point: sub-jobs; chain: gen shards (1-10)")] = 1,
    files_per_job: Annotated[int, Field(ge=1, le=25, description="chain_point: gen files per sub-job")] = 1,
    max_events: Annotated[int, Field(ge=0, le=10 ** 7, description="events per gen file, 0 = all")] = 0,
    seed: Annotated[int, Field(ge=0, lt=2 ** 31)] = 1,
    wall_minutes: Annotated[int, Field(ge=10, le=180, description="chain_point: wall limit per sub-job")] = 60,
    gen_set: Annotated[int, Field(ge=0, le=5, description="chain_point gen set (0 g4ref_v1 with a Geant4 "
                                                          "reference, 1 crygen_v1, ...)")] = 0,
    allow_gpu: Annotated[bool, Field(description="submit. Set it only after the user approved the GPU request "
                                                 "this tool returned without it")] = False,
    dry_run: Annotated[bool, Field(description="size and check without submitting")] = False,
) -> ToolResult:
    """Run the fast CRV chain (W5: cosmic generator events -> CRV pulses and coincidences) on GPUs.

    Always GPU: the first call returns the request size (jobs, GPU-hours at the wall limits and expected) with
    status needs_approval; repeat with allow_gpu=true after the user approved it; the server's GPU-hour cap
    applies. Device precision is the chain's default (float32): the kit task has no precision switch yet.
    Poll job_status(name), then fetch_outputs(name) for accepted events and clusters per generated event.
    """
    try:
        dp.check_name(name)
        if task == "chain_point":
            from mu2efastsim.endpoints import chainpoint as cp
            params = cp.validate({"task": "chain_point", "shards": shards, "files_per_job": files_per_job,
                                  "shard_offset": 0, "max_events": max_events, "seed": seed,
                                  "wall_minutes": wall_minutes, "gen_set": gen_set})
            low = cp.chain_estimate_s(params) / 60
            high = min(wall_minutes, (cp.wait_budget_s(params) + cp.chain_estimate_s(params)) / 60)
            per_job = [(round(low, 1), round(high, 1))] * shards
            reqs = [{"handle": f"{name}.crv", "dtype": None, "params": {"task": "chain_point", **params}}]
            size = dp.gpu_request(reqs * shards, 1, wall_minutes, per_job)
            size["sub_jobs"] = shards
        else:
            if not 1 <= shards <= 10:
                raise ValueError("chain: shards 1-10 (the held-out Geant4 shards 11-20)")
            reqs = [{"handle": f"{name}.crv", "dtype": None,
                     "params": {"task": "chain", "shards": shards, "max_events": max_events, "seed": seed}}]
            size = dp.gpu_request(reqs, 1, 60, [None])
        request = {"stage": "crv", "task": task, "params": reqs[0]["params"], "device": "gpu"}
        expected = {reqs[0]["handle"]: size["expected_minutes_per_job"][0][1]
                    if size["expected_minutes_per_job"] else None}
        return _submit(name, request, reqs, [], expected, size, allow_gpu, dry_run)
    except Exception as exc:
        return _error(exc)


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
    key = {"sob": "sensitivity", "stops": "stop_frac", "centuple": "efficiency", "crv": "accepted_events"}.get(
        stage, "ce_acceptance")
    text = ", ".join(f"{o['dtype'] or 'job'}: {key} {o['metrics'].get(key)}"
                     + (f", stops_per_pot {o['metrics'].get('stops_per_pot')}" if "stops_per_pot" in o["metrics"]
                        else "") for o in outputs)
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
