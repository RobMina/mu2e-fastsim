# Explicit-input response diagnostics

This extends the one-pass runner in [response-validation.md](response-validation.md), not the BO board or its
measurement definition. Response fields and units follow [response-configuration.md](response-configuration.md).
No scheduler, staging, build, transport change, or automatic physics acceptance is added.

## API and command line

```python
from crvfastsim.objective.validation import run

run(manifest_path, out, seed, knobs, jax=False, calibration=False,
    response_config=None, joint_capture=False, diagnostic_code=False)
```

Existing positional arguments and `revision(package)` are unchanged. `response_config` accepts a Python object,
a JSON file path, or inline JSON, as supported by `crvfastsim.response_config.parse`. The CLI accepts a path or
inline JSON:

```bash
python3 -m crvfastsim.objective.validation run \
  --manifest /data/heldout.json --out /data/new_diagnostic_run --seed 20261026 \
  --response-config '{"schema":1,"sipm":{"overvoltage":2.9}}' \
  --diagnostic-code --joint-capture
```

The destination must not already exist. Use an output directory **outside both source checkouts**. In particular,
a new untracked result directory inside a checkout changes the recorded Git status and invalidates the run.
Do not edit either checkout, source/config files, or frozen inputs while a pass is running. The runner never
edits source, checks out revisions, stages files, or commits work. Partial failed outputs are retained; rerun in
a new directory rather than overwriting them.

`--response-config` does **not** require diagnostic mode. `--diagnostic-code` may be used without joint capture.
Only `--joint-capture` requires `--diagnostic-code`; this runner always uses R3's objective path. The joint collector
is an observation sidecar, not a replacement response objective or a new calibration score.

## Configuration and frozen assets

Configuration is resolved with `RC.parse` and `RC.apply_knobs` before input verification or physics asset I/O.
Malformed/unsupported fields and invalid knobs fail early. Existing scalar knobs override config-file values.
The runner still forces `v05=True`: per-event FEB phase and zero DB offsets override requested timing modes.

**Config asset overrides are refused**, even if they point at a manifest asset. All `assets` fields must have
shipped default values, and `waveform.single_pe_file` must remain the shipped default. This prevents a config
from silently changing calibration/deviation, enabling a background/time shift in closure mode, changing
geometry/LUT/photon-map/template assumptions, or introducing an unrecorded reference. A `reference` config asset
is not in schema v1 and is rejected. This runner intentionally supports fewer asset overrides than standalone R3.

Explicit calibration/deviation/reference assets still come from the frozen inventory manifest. Calibration mode
requires its background and time-shift assets; closure mode keeps both disabled even when the manifest contains
them. Their bytes are nevertheless checked. Reference and excluded-input hashes are checked as well. Runtime
geometry, waveform, optical tables, and photon map retain the existing resolved-file hash checks. Changing asset
bytes requires a new inventory/validation contract, not a hidden config override.

`provenance.json.response_config` contains the full effective values, canonical JSON string, canonical SHA256,
and the schema/support information returned by `RC.record`. The effective values include knob precedence, v05
and the explicit manifest assets. For a file request, `requested_file` separately records its absolute path and
exact byte SHA256; inline/object/default requests have `requested_file: null`. The original file is rehashed at
the end. The canonical hash is an execution/configuration identity, not proof of physics validity or a BO hash.
R3 receives this resolved dictionary through `response_config=...` and retains its own `resolved_config` receipt.

## Clean revisions versus diagnostic snapshots

The default remains the original clean-revision gate for **both** `crvfastsim` and `mu2efastsim`. It calls the
unchanged `revision(package)` before and after running; any output from `git status --porcelain`, including
untracked files, is rejected. Default mode does not silently fall back to diagnostics.

`diagnostic_code=True` is an explicit opt-in to intentional dirty/untracked worktrees. It still requires valid
Git checkouts and HEADs; a missing/unavailable Git repository is not excused. It records:

* Git HEAD and repository-wide `status --porcelain=v1 --untracked-files=all` for each package checkout;
* individual SHA256s for **all `.py` and `.json` files under each imported package**, including ignored and
  untracked files (so the validation implementation and shipped preset are included);
* tracked/untracked repository `.py`/`.json` files outside the package, plus source/config/document files under
  `scripts/`, `tests/`, `docs/` and at the repository root (`.py`, `.json`, `.md`, `.rst`, `.sh`, `.toml`, `.cfg`,
  `.yaml`, `.yml`), including ignored support files;
* a canonical SHA256 of the relative-path-to-file-SHA256 map.

The source manifest is a **hash snapshot**, not a backup of file contents or an environment/container capture.
Bytecode caches and unrelated data directories are not source snapshots. Installed dependency versions and
runtime asset hashes remain separately recorded. HEAD/status are checked around snapshot collection, and both
full snapshots must match before/after execution. Added/deleted sources, changed bytes with unchanged dirty Git
status, HEAD changes, and Git-status changes fail validation. These are endpoint integrity checks, not a filesystem
lock: a transient edit restored between checks cannot be certified absent.

Before data are run, `provenance.json` contains `code`, `status: "running"`, and a `code_policy` of either
`"clean-revision"` or `"unpublished-diagnostic-snapshot"`. Diagnostic code entries have
`mode: "diagnostic-worktree"` and `published: false`; **their HEAD is not claimed to describe their source bytes**.
Dirty diagnostic results must not be presented as clean, committed, published, nominal-equivalent, or accepted
physics results. Preserve the exact worktree sources separately if they will need replaying. The existing `paired`
report rejects diagnostic-code and failed-execution provenance rather than issuing its ordinary clean-revision
PASS for such outputs; legacy completed clean receipts retain their comparison behavior.

After execution and scoring, the runner rehashes manifest/input/excluded/asset/config files and records
`code_after`, `integrity_errors`, and `status: "complete"` or `"failed"`. Integrity failure raises an error. The
same checks run if R3 or scoring raises; `execution_error` preserves that failure. If an after-snapshot cannot be
read, the receipt records the available entries and fails. Early admission failures need not create an output
directory; setup failures may leave partial output. A complete diagnostic receipt means the one-pass bookkeeping
and integrity checks passed, **not** that a physics compatibility/identifiability gate passed.

## Joint sidecar integration

With joint capture enabled, validation constructs `crvfastsim.objective.joint.JointCapture()` and passes the object
as `r3.run(..., objective=True, joint_capture=collector)`. R3's observation hook supplies
`collector.add(c, p, o, event_keys, source_index=0)`; collection must not modify the physics inputs/results or RNG
stream. The collector implementation defines the captured array schema and returned summary.

After the pass, validation calls `collector.write(destination / "joint.npz")` and saves its returned JSON-serializable
summary to `joint.json`. `provenance.json.joint_files` records absolute paths and SHA256s for both sidecars. Capture
is off by default (`joint_capture=None` in the R3 call); without it no sidecar is written. Use the response result,
comparison, provenance and joint summary together, and retain per-source event identities when comparing inputs.
The sidecars are diagnostic observations; they do not retune empirical background or frozen time calibration.

## Local checks

`tests/test_validation_config_diagnostic.py` uses mocked physics to cover configuration forwarding and canonical
receipts, knob/v05 precedence, forbidden assets and early rejection, closure/calibration background/time-shift
isolation, unchanged clean revision API/gate, explicit diagnostic admission, full source snapshots, source/Git and
input mutation rejection, failure receipts, CLI wiring, joint serialization/hashes, and nonoverwrite behavior.
These tests require no ROOT simulation, scheduler, build, or remote command; they do not establish CPU/GPU physics
equivalence for changed response settings.
