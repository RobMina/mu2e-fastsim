# Run 2 grid reference extension (on hold: production-reuse audit)

**Not to be submitted or retried pending a production-reuse decision.** The one test job failed
namespace preflight before Geant4 (empty-directory listing returned 404), with no physics outputs. The attempt
marker is preserved. Existing `sim.mu2e.CosmicDSStopsCRYAll.MDC2030af.art` contains 1.903B retained S1 events
in the same Run2 `nominal_v10` geometry and must be evaluated first; see [reference sizing](run2-reference-sizing.md).
The commands below document the prepared design; they are not evidence that new production is needed.

Extend the retained 1M G4 transport reference to **20M generated events** with **380 × 50k** grid jobs.
All dts/gen art, compressed worker logs, resolved input FCL, runtime identity, byte hashes and prospective completion
receipts are written **directly to pnfs scratch**. No new build, muse tarball, or bulk exp storage is required.
This extends the frozen transport reference, not a corrected-stock response V2 population.

## Frozen configuration and identities

- Musing **SimJob MDC2025ay**, Offline **v13_39_00**, **al9-prof-e29-p107**, unchanged Run 2 FCL/geometry.
- Run 1400, subruns **121–500**, seeds **20265220–20265599** (`20265100 + subrun - 1`).
- Retained Run 2 subruns 101–120 / seeds 20265200–20265219 are never rerun.
- One CPU, **3000 MB RAM**, **4000 MB worker disk**, **4h expected lifetime**, automatic release.
- Established 50k Run 2 worker: **1:08:52 wall**, 2,033,344 kB peak RSS. No runtime projection from that one shard.
  Retain the proven chunk for the first test; revisit sizing using its actual measurements rather than assume 30 min.
- No new physics: same CRY config, field-off world, 1 mm range cut and original accepted/gen output selection.

The opt-in `run2_grid_plan` module freezes the three FCL/geometry files and worker into a local payload with hashes.
The `campaign.json` hash selects an immutable pnfs config directory; a submit command never removes or overwrites
another campaign's FCL. Historical `submit_ref_sim.sh` is unchanged and must not be used for this prepared extension.

## Prepare (no network or submission)

```bash
PYTHONPATH="$CORE:$CRV" python3 -m crvfastsim.analyses.run2_grid_plan \
  --state "$STATE" --outdir "$CRVFS_PNFS/g4ref_run2_ext_v1_20m" --owner "$CRVFS_OWNER"
bash "$CRV/grid/submit_run2_extension.sh" "$STATE" test
```

Preparation writes `STATE/payload`, `campaign.sha256`, `commands.sh` and a summary. Output paths must be new.
Dry-run is the submission script's default: it prints an exact `jobsub_submit` command with no pnfs writes,
credentials, setup or queue access. Grid submission must run on a Mu2e interactive host after setting up ops and
valid grid credentials, from a local state/data directory, not a synced source tree or pnfs. Python3 on the submit
host is sufficient; the preparer itself is run in the project Python environment.

## Pilot-gated submission (explicit approval required)

```bash
# After setupmu2e-art.sh and muse setup ops, user approval, live queue/output review:
bash "$CRV/grid/submit_run2_extension.sh" "$STATE" test --submit
# Only after persisted-count/runtime/copyout certification of subrun 121:
bash "$CRV/grid/submit_run2_extension.sh" "$STATE" pilot --submit
# Only after certification of 121, 122 and 123:
bash "$CRV/grid/submit_run2_extension.sh" "$STATE" remainder --submit
```

| phase | jobs | subruns | seeds | generated |
|---|---:|---|---|---:|
| test | 1 | 121 | 20265220 | 50k |
| pilot | 2 | 122–123 | 20265221–20265222 | 100k |
| remainder | 377 | 124–500 | 20265223–20265599 | 18.85M |

Every phase obtains a submit lock, checks frozen payload hashes, captures the current `jobsub_q` output and pnfs
inventories, and refuses colliding output names or another reference job in the queue. A new absent namespace is
allowed only with `RUN2_NEW_NAMESPACE_APPROVED` exactly equal to the output directory after reviewing `ifdh` errors;
a failed remote lookup is not automatically treated as empty. Initial creation and all copy-out use the explicit
logical `--need-storage-modify=/mu2e/scratch/...` scope. The small config files are staged once by content hash.
The submitter writes an exclusive phase attempt marker **before** jobsub and preserves it even on failure. An
interrupted or timed-out call must be reconciled against the actual queue and outputs before any retry; never
blindly remove its attempt marker.

The worker publishes its completion receipt only after both art outputs, log and provenance archive copy out.
An existing valid receipt plus both art files and log makes a retry skip. Any output without a matching receipt is
partial and causes refusal, never overwrite. This is conservative retry handling, not an atomic distributed commit;
workers do not claim remote filesystem locking. A completion receipt records log counts but explicitly does **not**
claim persisted ROOT event-count certification. Failed worker jobs retain their jobsub logs for diagnosis.

### Promotion gate format

`STATE/gate_pilot.json` (121) or `gate_remainder.json` (121–123) must use:

```json
{
  "schema": "crv-run2-extension-gate-v1",
  "campaign_sha256": "PREPARED_CAMPAIGN_SHA256",
  "shards": [{
    "subrun": 121, "seed": 20265220,
    "generated_events": 50000, "accepted_events": 1,
    "highest_cycle_counts_verified": true,
    "generator_values_verified": true,
    "copyout_verified": true, "runtime_verified": true,
    "evidence": [{"path": "/path/to/real/certification.json", "sha256": "REAL_SHA256"}]
  }]
}
```

The accepted-event value above is a schema placeholder, not a measured count. Populate only from completed
certification: highest-cycle dumped gen has 50k unique events, accepted gen/particles/steps agree with the worker's
successful acceptance summary, accepted generator identities/values agree with source gen, and actual release/FCL
and copy-out match. Evidence files must exist and match their hashes. No fake gate is generated by preparation.
The gate is an operator evidence index, not a signature or a substitute for checking the actual persisted output.
No automatic monitor submits the next phase.

## Storage and downstream analysis

Only small state and submit logs belong in exp. Art stays on pnfs; read known paths directly for bounded dumping.
Run Python analysis on UVA, reusing the staged 1M input. Derived 19M tables/evidence are estimated at about 12 GiB
(16 GiB conservative allowance), retained on pnfs or UVA rather than bulk staged into exp. Measure art bytes and
worker scratch usage in the pilot before full production. Both scratch areas are purged; decide archival retention
separately. Fixed-stock named V2 still needs its own matching Run 2 response-seed/receipt campaign.
