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
| `run_design_point` | one target design (`target` overlay, or `geometry` nominal / a `beam/geom` file) -> stops -> CE chain -> EdepAna file (stage `edep`, one CPU core per dtype) or the built-in sensitivity (stage `sob`, one 32-core CPU job); returns job handles at once | submits CPU jobs |
| `job_status` | per job: working / completed / failed / cancelled, site state, job id, whether the inputs were streamed in | ssh status |
| `fetch_outputs` | metrics, the EdepAna file with its `.edep.json` run record and `edep_summary.txt`, and `score_with`: the analysis server's `run_analysis` call for `approx_ce_sensitivity` | fetches `out/` |
| `cancel_design_point` | cancels a point's queued or running jobs | ssh cancel |

The compute tools are asynchronous: a point takes minutes to hours on one CPU core (about 9 minutes plus 53 ms per
electron in float64, `beam/docs/edepana.md`). Both dtypes run by default, each as its own job. The server only asks
for CPU jobs (`beam_edep_cpu`, `beam_sob`); GPU tasks are not exposed.

`run_design_point` goes through the fastsim kit (`mu2efastsim.endpoints.kit`), so the kit's rules hold: the target
overlay must pass the kit's geometry admission (`core/mu2efastsim/endpoints/geometry.py`), the same handle and
request return the same job, and the Mu2e-internal inputs are streamed from the FNAL host into the running job, never
stored on the endpoint. **Keep the server running until `job_status` reports `inputs_streamed`**: the job waits at
most 15 minutes for them.

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

- The inputs are streamed by the server process: a point submitted by a server that exits before the job starts
  fails after 15 minutes (`job_status` on a new server restarts streaming while the job is still waiting).
- CPU stages only; the GPU stages (`beam_edep`, `beam_point`) and the EventNtuple writer (`beam_centuple`, which
  also needs `/cvmfs`) are not exposed.
- Scoring is a second server: `fetch_outputs` returns the call, the Mu2e analysis server runs it.
