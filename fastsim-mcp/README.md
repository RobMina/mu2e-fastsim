# fastsim-mcp: the Mu2e fast simulation as agent tools

An MCP server that lets an agent (Claude Code, Claude desktop, Codex, Cursor, or a Python `mcp` client) ask which
fast workflows exist and what they approximate, then run a stopping-target design point through the fast chain and
get back the EdepAna file that the Mu2e analysis server scores with `approx_ce_sensitivity`.

It follows the pattern of the Mu2e analysis server: the science code stays plain Python, and a generic wrapper
publishes it.

- `fastsim_tools/` is an ordinary package that never imports MCP.
- `fastsim_mcp_server/` reads `[tool.mcp-server] tool_modules` from `pyproject.toml` and registers every function in
  the module's `__all__` as an MCP tool. Type hints, pydantic `Field`s and docstrings are the schema.

## Tools

| tool | what it does | compute |
|---|---|---|
| `list_workflows` | W1 beam, W2-W4 tracker, W5 CRV, W6 mixing, W7 calorimeter, W8 trigger, the CE chain, the EdepAna and EventNtuple writers: boundary (input -> output), packages, command lines, documents, kit tasks | none |
| `describe_approximations` | per workflow (or chain): switchable approximations with levels, presets, stock alternative and switch; fixed ones; the notes of `docs/approximations.md` | none |
| `run_design_point` | one target design (`target` overlay, or `geometry` nominal / a `beam/geom` file), `device` cpu (default) or gpu. Stages: `edep` stops -> CE chain -> EdepAna file (cpu `beam_edep_cpu`, one core per dtype; gpu `beam_edep`, one GPU per dtype); `sob` built-in sensitivity (cpu `beam_sob`, 32 cores); `stops` stopping metrics only (gpu `beam_point`); `centuple` the fast EventNtuple (gpu `beam_centuple`, endpoints with /cvmfs and FNAL storage). Returns job handles at once | submits jobs |
| `run_crv_chain` | the W5 CRV chain on GPUs: `chain_point` (N one-GPU sub-jobs, inputs streamed) or `chain` (one job, inputs read in place); float32, the chain's default | submits GPU jobs |
| `job_status` | per job: working / completed / failed / cancelled, site state, job id, whether the inputs were streamed in | ssh status |
| `fetch_outputs` | metrics, the EdepAna file with its `.edep.json` run record and `edep_summary.txt`, and `score_with`: the analysis server's `run_analysis` call for `approx_ce_sensitivity` | fetches `out/` |
| `cancel_design_point` | cancels a point's queued or running jobs | ssh cancel |

The compute tools are asynchronous: a CPU point takes about 20 minutes for 5,000 electrons on one core, compilation
included. Both dtypes run by default, each as its own job.

### GPU requests

GPU work never starts by accident:

1. **Sized first.** A call with `device="gpu"` (or any `run_crv_chain` call) returns `needs_approval` and submits
   nothing. The reply's `gpu_request` gives the jobs, GPUs per job and card, the GPU-hours at the jobs' wall-time
   limits (the most the request can use) and the expected GPU-hours from the measured rates (`null` where nothing
   was measured).
2. **Explicit flag.** The agent repeats the call with `allow_gpu=true` only after the user approved that request.
3. **Configured cap.** `FASTSIM_MCP_GPU_HOURS_CAP` (a site setting, not a tool argument) bounds the GPU-hours at the
   wall limits of one call. Unset or 0 turns GPU jobs off. The kit's own per-job budget guard applies on top.
4. **Kit admission.** Every GPU request is checked with the kit before anything is submitted. A task the endpoint
   cannot run, such as `beam_centuple` without FNAL storage and /cvmfs, comes back as an error.

`dry_run=true` sizes and checks a GPU request without the flag and without submitting.

`run_design_point` goes through the fastsim kit (`mu2efastsim.endpoints.kit`), so the kit's rules hold: the target
overlay must pass the kit's geometry admission (`core/mu2efastsim/endpoints/geometry.py`), the same handle and
request return the same job, and the Mu2e-internal inputs are streamed from the FNAL host into the running job, never
stored on the endpoint.

### Detached workers: points outlive the session

A job waits at most 15 minutes for its inputs, which the process holding the kit streams in, so a point must not
depend on the MCP client staying connected. `run_design_point` saves the point and starts a **detached worker**
(`python -m fastsim_tools.worker`, in its own session, logging to `<state>/mcp_workers/<name>.log`). The worker:

- submits the jobs through the kit (idempotent per handle);
- polls them, which keeps the kit's input streaming going;
- fetches each job's outputs when it completes;
- writes a heartbeat (`<state>/mcp_workers/<name>.json`) after every round.

It holds a lock per point, so there is at most one worker per point, and it exits when every job is done.

`job_status` and `fetch_outputs` never stream from the server: they read the heartbeat. From any later session
they respawn a worker that died before the point was done (the restarted worker gets the same jobs back from the
kit and resumes streaming), at most 5 times per point. The client, the server or the network can go away
mid-stream; the worker carries on, and the next `job_status` reattaches. Settings: `FASTSIM_MCP_WORKER_POLL_S`
(default 30), `FASTSIM_MCP_WORKER_MAX_H` (48), `FASTSIM_MCP_SUBMIT_WAIT_S` (how long `run_design_point` waits for
the job ids, 120).

## Running it

The server process is a kit host: it needs this checkout (core and beam), `mcp` and `pydantic`, and, for the compute
tools, `hep-endpoints` (`pip install '.[kit]'`) and ssh access to the compute endpoint and the FNAL input host. Site
settings (hosts, account, work areas, the image) come from the environment or a site file of `NAME=value` lines, as
for every kit user (`site.env.example`). Discovery tools need none of it.

```bash
cd fastsim-mcp
python -m venv venv && venv/bin/pip install 'mcp[cli]>=1.27' 'pydantic>=2,<3' '.[kit]'
venv/bin/python -m fastsim_mcp_server --site-env /path/to/site.env            # stdio
venv/bin/python examples/smoke_client.py transcript.json \
  --point '{"name": "demo1", "electrons": 5000}' -- --site-env /path/to/site.env
```

`examples/chain_client.py` drives two servers as an agent would: `fetch_outputs` here, then each returned
`score_with` call on the Mu2e analysis server (its command in `ANALYSIS_MCP_CMD`).

The site file also sets `M2FS_CODE` (this checkout) and `FASTSIMKIT_STATE_DIR` (job records and fetched outputs on
the kit host). `PYTHONPATH` in it is ignored: set it in the client's environment instead.

Claude Code (`.mcp.json` in a project; see `examples/mcp.json.example`):

```json
{"mcpServers": {"mu2e-fastsim": {
  "command": "/path/to/fastsim-mcp/venv/bin/python",
  "args": ["-m", "fastsim_mcp_server", "--site-env", "/path/to/site.env"],
  "cwd": "/path/to/mu2e-fastsim/fastsim-mcp"}}}
```

## Tests

```bash
cd fastsim-mcp && python -m pytest -q tests      # PYTHONPATH: this directory, core, beam, calo, crv, tracker, the root
```

The tool functions run against a stand-in kit (no endpoint, no network); the geometry overlay is checked with the
kit's own admission code, and the parsed approximation table against the packages' declarations. CI runs the suite as
`mcp` (`.github/workflows/tests.yml`).

## Gaps

- The worker runs on the kit host. If that machine loses its connection to the endpoint or to the FNAL host for
  longer than a job's input wait, that job fails; the worker records it, and the point is rerun under a new name.
- A stalled input stream ends the point: the kit marks it as a stream error and cancels the job, and
  `job_status` reports `stream_error`. Rerun the point under a new name (a failed handle is not resubmitted).
- The CRV chain runs at its default device precision (float32): the `chain_point` and `chain` kit tasks have no
  precision parameter yet, so float64 CRV needs a kit change.
- `beam_point` and `beam_sob` have no dtype switch: one job each.
- Expected GPU time of `beam_centuple` and `chain` is not measured here; their requests are sized at the wall
  limits only.
- Scoring is a second server: `fetch_outputs` returns the call, the Mu2e analysis server runs it.
