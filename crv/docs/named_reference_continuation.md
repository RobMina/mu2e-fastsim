# Bounded named-reference grid continuation

The user authorized the saved October 25 proposal: three 50k-event Run1A jobs and three Run2 jobs,
then sixteen 50k-event jobs per configuration only after successful pilot copy-out and persisted-count
certification. Subrun 101 is retained; pilots are 102–104 and remainder 105–120. This is G4 reference
production, not a fast-transport validation or a change to the Run2 V1 FAIL verdict.

`continue_named_reference_grid.sh STATE --approved-remainder` monitors the already-submitted pilots,
using their recorded cluster IDs. It is deliberately bounded to this campaign and reserved to gpvm07.
It uses the existing CVMFS SimJob MDC2025ay worker, staged FCL and previously certified dumper build;
no build, tarball, FCL staging, candidate transport, overwrite or automatic resubmission occurs.

## Gates and durable evidence

- A new data-area state directory and an exclusive campaign lock prevent concurrent controllers.
- Worker, dumper and certification code hashes are frozen; staged FCL must match frozen local files.
- Queue checks are five minutes apart. Held, removed or suspended pilot jobs stop the driver.
- All jobs must leave the queue, both art files and the worker log must appear in supported `ifdh`
  inventories, and fetched job logs must contain the worker's final completion marker.
- Dumps run serially on the reserved GPVM, reading art directly from known pnfs paths. No art is staged
  into the user's data area. At least 2 GiB of data quota headroom is required before dumping.
- `python3 -m crvfastsim.analyses.reference_certification` checks successful worker/dump logs against
  actual persisted identities from the highest ROOT tree cycles, including generated and accepted counts.
  It writes a success report only after all requested shards pass. No fast-candidate table is needed.
- The queue and all remainder output names are checked again before submission. Any existing remainder
  artifact or queued user job stops the automatic continuation for manual deduplication review.
- Each submission gets an exclusive `*.attempted` marker in the shared data-area ledger
  `thread_a_remainder_105_120_attempts/` before dispatch, with a copy in the state directory. Ambiguous
  outcomes stop and block fresh-state retries; they are never automatically retried. Both new cluster
  IDs and full submit logs are saved.
- The remainder is monitored and certified by the same gates. `status.txt` gives the latest phase;
  `driver_exit.txt` records final exit and UTC time. Each stage has a 12-hour monitoring deadline.

The controller may remain active after the interactive agent turn ends. Reserve gpvm07 until it exits.
Missing/failed evidence means STOPPED, not successful completion. Never rerun the controller to repair a
submission: inspect its attempt markers, queue, logs and existing outputs first. A new certification output
path is needed after a failed check. The original 50k/config baseline is not re-generated.

## Certification-only example

```bash
python3 -m crvfastsim.analyses.reference_certification DUMPDIR \
  --first 102 --last 104 --expected-events 50000 --output NEW_REPORT.json
```

This checks completeness relative to successful worker counts and consistency of the persisted accepted
identities with generated events. It does not independently reconstruct the Geant4 acceptance decision,
certify detector-response physics, or establish per-sector precision.
