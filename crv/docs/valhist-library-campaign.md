# Explicit multi-shard library replay plan

`valhist.library_campaign` schedules a small, explicit extension of the retained donor-bootstrap pilot. It is a
job runner, not a software build or grid submitter. The frozen physics checkout and already-built stock area
remain unchanged. Source/output identity and library validation are delegated to the existing sealed library
builder and pilot driver; the new planner refuses existing destinations rather than treating them as a resume.

## Commands

Run from the canonical CRV checkout, with frozen core on `PYTHONPATH`. All outputs are outside the code tree.
The script must run on one otherwise unused GPVM, with the ana environment already initialized and a fresh
named session. Do not change the original uncertainty checkout's `library_variants.py` or physics sources: its
sealed plan binds their exact paths and hashes. New analysis modules do not belong in the frozen physics checkout.

```bash
python3 -m valhist.library_campaign preflight --data "$CRVFS_DATA"
python3 -m valhist.library_campaign plan --data "$CRVFS_DATA" \
  --out "$CRVFS_DATA/valhist/thread_b_ensemble_plan_20261025.json" \
  --indices 1 2 3 4 5 6 --shards 11 12 --tag thread_b_ensemble_20261025
python3 -m valhist.library_campaign run "$CRVFS_DATA/valhist/thread_b_ensemble_plan_20261025.json" \
  --indices 1 2
```

`preflight` verifies every input hash of the accepted common-code shard-11 campaign, including the frozen stock
plugins, source, calibration and response assets. It does not rebuild anything. The checked inputs should remain
unchanged throughout the run; the underlying pilot also checks before/after provenance for each new cell.

The plan covers library indices from the original eight-replica donor plan and at least two unique validation
shards in 11–20. These shards are denominator-disjoint by design; actual full-key audits are performed in the
ensemble reporter. Existing original shard-11 draws are reused only after their completion/hash certification.
New cells use `3000000 + 10000*index + 100*shard + draw`, independently for draws 1 and 2. The original seeds
remain unchanged. Each assembled library is global across its draws and evaluation shards. Existing library
metadata and SHA256 must validate, otherwise the driver stops. Missing planned libraries are assembled by the
original Python data-library builder, not by any Offline build command.

Each `run --indices ...` call executes the selected indices sequentially. Distinct index lists can be assigned to
different free GPVMs, but never stack jobs on one host. The driver requires every non-reused output to be absent
at the start. A failure or timeout is not authorization to retry the complete list: inspect the real processes
and preserved completions, then explicitly plan remaining work with new output names if necessary. Never erase
a failed manifest or refresh its provenance.

The replay plan is distinct from the ensemble-report manifest. Use the tracked converter on an explicitly
selected, complete balanced subset of finished cells, then run the reporter:

```bash
python3 -m valhist.followup_summary ensemble-manifest --campaign PLAN.json \
  --out NEW_CELLS.json --indices 1 2 3 4 5 6
python3 -m valhist.library_ensemble --manifest NEW_CELLS.json --out NEW_REPORT.json
python3 -m valhist.followup_summary ensemble NEW_REPORT.json
```

The converter renames `expected_events` to `expected_generated_events`, omits `reused`, checks completion and
validates the balanced design. The reporter independently checks actual output hashes, event inventories and
physics identities; the converter alone is not certification.

## Statistical scope

This is a donor-only ensemble with a **fixed shared generator denominator**. The original eight shard-11 draws
are not eight independent libraries; they are four libraries with two conditional draws each. Independent
validation shards expand evaluation coverage but do not increase the number of independent library realizations.
Stock baseSeed, conditions and transport stay fixed. Two evaluation shards or six libraries are still small
samples; results remain conditional variance diagnostics, not calibrated systematic errors or blanket DQM acceptance.
