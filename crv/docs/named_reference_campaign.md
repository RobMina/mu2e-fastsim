# Frozen named-reference full-statistics comparison

`python -m crvfastsim.analyses.named_reference_campaign` is a bounded, serial
analysis driver for the **completed** Run1A/Run2 reference campaign: subruns
101–120, run 1400, 50,000 generated events/shard, **1,000,000/configuration**.
It does not submit, monitor, build, install, regenerate G4, tune physics, or resume.
The old 20k pilot subruns are not part of this comparison.

## Invocation

Use the frozen validation worktrees, not the umbrella `core`. The driver checks
both the imported core location and explicit `PYTHONPATH`, rejects another core
on that path, and compares loaded-module locations. Existing geometry text must
be locally resolvable through the usual `MU2E_SEARCH_PATH`/Offline roots; geometry
fetching is disabled. `--geometry-dir` is the directory containing the versioned
`mu2e_run1a_v01.gdml` and `mu2e_run2_v10.gdml`, **not** its parent data directory.
All input paths must be local, without symlink components or dCache (`$CRVFS_DCACHE`) paths.

Run on a reserved GPVM with valid worktree git metadata: the laptop mirror's `.git`
files point to cluster paths and cannot establish `repo_state` locally. Synthetic tests
below remain local. The production launch must independently pin core `2bcefde` and CRV
`3895aae` and retain the approved audit/path JSON hashes; consistency with arbitrary
supplied old evidence is not by itself approval of a physics revision.

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && pyenv ana
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/path/to/core:/path/to/crv
D=$CRVFS_DATA
python3 -m crvfastsim.analyses.named_reference_campaign \
  --certificates "$D/thread_a_certification_baseline_20261025.json" \
    "$D/thread_a_grid_continuation_20261025/pilot/certification.json" \
    "$D/thread_a_grid_completion_20261026/certification.json" \
  --retained-candidates "$D/thread_a_named70k_20261025" \
  --retained-audits "$D/thread_a_paired70k_v3_20261025" \
  --retained-paths "$D/thread_a_paths70k_final_20261025" \
  --geometry-dir "$D/geometry" --outdir "$D/NEW_full1M"
```

The default `--config all` processes Run1A then Run2, serially. `--config run1a`
or `--config run2` permits a host to handle only one configuration, with a
separate **new** output directory. Even then, the certificates must establish
completion of all 40 reference shards. Only the selected configuration's retained
candidates/diagnostics and geometry are required. There is no aggregate physics
claim about an unrequested configuration.

No launcher/CLI registration in other files is required; invoke the module
explicitly. `--help` describes all arguments.

## Fail-closed gates

Before creating the output directory:

1. Require `reference_certification` schema 1, `status: PASS`, run 1400, both
   configurations and consistent declared intervals. The union must cover each
   `(config, subrun)` exactly once for 101–120. The standard three certificates
   cover 101, 102–104 and 105–120; other nonoverlapping partitions of exactly that
   set are accepted. Reject missing/duplicate/extra shards and repeated generator
   hashes within a configuration.
2. Require generated count 50,000, positive accepted count equal to particle
   events and worker passed, passed+failed=50,000, consistent successful worker
   and dump-log summaries, and highest-cycle ROOT inventories. Rehash all certified
   generator/reference/log bytes and the certificate files themselves.
3. Compare current `transport_validation.repo_state` CRV/core **heads and
   `source_sha256`** against the retained `audit50k.json`. Compare counters/GDML
   hashes and diagnostic hash, retained path source hashes/settings, exact source
   identities/ranges, accepted-event evidence, and all-sector/group coverage.
   Worktree root and git status strings are recorded but are not equality gates:
   a relocated worktree or these new files must not invalidate identical physics.
   Git calls are read-only, with `GIT_OPTIONAL_LOCKS=0`.
4. Verify every reused ROOT input against its saved audit/path fingerprints and
   the certified source inventories. Verify all original audit artifact hashes,
   including the retained candidate `transport.log`. Record the retained audit
   and path JSON fingerprints in the new manifest.

For each configuration, reuse exactly:

- `--retained-candidates/<cfg>/tables/fastC.001400_00000101.root`
- `--retained-audits/<cfg>/audit50k.json`
- `--retained-paths/<cfg>/paths50k.json`

**Never run transport for subrun 101.** Its saved audit and path diagnostics are
used directly in the merges, without rewriting or refreshing their provenance.
A new V1 emit for 101 is produced from the retained ROOT file.

For each new subrun 102–120, call the established vectorized transport once with
one input: `transport.run([gen], tables_dir, seed=subrun-100, stage='C',
run_config='run1a_v01'|'run2_v10', gdml=versioned_path)`. This gives seeds 2–20,
with the transport input-file index always zero. No per-event driver loops or
alternate physics implementations are introduced. Check and save source hashes
immediately before/after each new generation. Recheck certified input hashes.
Run the existing transport audit with expected generated=50,000, externally
certified expected accepted, and worker/dump/certificate/transport-log artifacts;
run the existing path analysis; then use `v1compare.run(..., emit=...)`.

Only after exactly 20 disjoint shards are checked does each configuration merge
its audits, paths and V1 emits using the existing merge implementations. Audit
merging accepts the old 101 diagnostic only with matching source/geometry/hash
provenance. The driver additionally requires exact run/subrun coverage,
1,000,000 generated events and certified accepted totals. It rechecks frozen
sources and all input/result fingerprints before merges and at final completion.
The existing V1 thresholds, grouping and validity guards are unchanged.

## Outputs and failure semantics

- `manifest.json`: frozen source/code/geometry snapshot, all 40 certified shard
  records, certificate and input fingerprints, selected configs and seed rule.
- `progress.jsonl`: flushed UTC start/completion/failure records and stage elapsed
  seconds; concise progress is also printed. Each shard/stage has its own log.
- `<cfg>/<subrun>/`: new transport tables only for 102–120, `source_before.json`
  and `source_after.json`, audit/path JSON, V1 emit and `shard.json` provenance.
  Subrun 101 has no generated table directory; its manifest points to retained
  files. Its diagnostic checks and fresh V1 emit still have logs.
- `<cfg>/audit1M.json`, `paths1M.json`, `v1/v1compare_stats.json` and the comparator's
  ordinary PNGs. No additional counter geometry, PDFs or physics plots are made.
- `<cfg>/summary.json` and top-level `summary.json`: exact generated/accepted
  counts, V1 PASS/FAIL and numeric group results, all-sector paired diagnostics,
  and fingerprints pointing to merged path/audit/V1 results.
- `source_final.json`: final frozen source/geometry snapshot on successful final
  checking. `final_check.log` and progress record the final verification stage.

`execution_complete` and `v1_pass` are separate. A complete V1 **FAIL** is retained
explicitly and does **not** prevent analysis of the other configuration. The
campaign CLI returns **0 for a fully completed execution**, even with V1 FAIL;
consumers must inspect `v1_pass`/per-config `v1_status` for physics results. The
historical V1 `events_union` counts particle-record event union and is not relabeled
as the 1M generated-event denominator.

Operational exceptions, missing V1 stats, failed provenance/count checks and
source changes stop immediately with a **nonzero exit**. A post-preflight failure
leaves a top summary with `execution_complete: false`, `v1_pass: null` and the
error, plus any completed configuration summaries. A failed preflight creates no
output. All partial outputs remain; an existing output directory, even empty,
is refused. No overwrite, cleanup, retry, automatic resubmission or resume is
provided. Hard termination can leave no final summary; it is never evidence of
completion. Review partial evidence manually before authorizing another new run.

## Provenance limits

Certificates are trusted records from the existing certifier, not cryptographic
signatures. Their byte fingerprints are rechecked, but this driver does not
rerun the certifier's ROOT identity reconstruction. Fresh audits perform the
established persisted source-identity/value checks. A certificate proves
completeness relative to successful worker evidence, not an independent
reconstruction of the Geant4 acceptance decision.

The retained audit/path hashes describe analysis-time code; they cannot prove
which producer code created a historical ROOT file. Original artifact logs are
required and retained, but that limitation is not upgraded into a production
attestation. Audit/path JSONs themselves have no prior external signature: the
new manifest anchors their bytes from this campaign's preflight onward. Retained
ROOT files may be explicitly relocated if bytes match; embedded original audit
artifact log paths must still be readable and unchanged. The driver does not
search for substitutes or silently remap historical artifact paths.

Retained audits do not store the old V1 comparator hash. V1 emits are therefore
regenerated for all 20 shards using the **current** comparator, whose hash is
frozen and recorded for this execution; no historical V1 implementation identity
is claimed. Filesystem pre/post hashes detect persistent changes, not transient
changes reverted between checks. Source hashes cover the established validation
scope and path-analysis dependency set, not every possible dependency, runtime
library, or in-memory monkey patch. Avoid concurrent source/input modifications.
Paired and path results remain truth-conditioned diagnostics, not correction
models or invented PASS gates.

## Synthetic tests

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/path/to/core:/path/to/crv \
python3 -m pytest -p no:cacheprovider \
  tests/test_named_reference_campaign.py -q
```

Tests use synthetic certificate inventories/files and mocked numerical analyses;
no production ROOT tables, transport, G4, network calls, installation or build is
required. They exercise coverage/count/hash guards, retained source/geometry
checks, core selection, 20-shard merging, exact serial seeds, immutable reuse of
101, single-config execution, physics-FAIL continuation, source-change stopping,
operational failures and preservation/refusal of partial output.
