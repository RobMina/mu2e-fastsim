# Persisted-table transport validation

`python3 -m crvfastsim transport-validation` supplements the established `v1compare` without changing its
thresholds or physics. It audits explicit generator/reference/baseline/candidate files, keeps every named sector,
and adds event-cluster paired rate errors, truth-identity matching and entry-momentum diagnostics.

```bash
python3 -m crvfastsim transport-validation \
  --gen GEN.root --reference G4.root --baseline OLD_FAST.root --candidate NEW_FAST.root \
  --config run2_v10 --expected-events 20000 --gdml WORLD.gdml --output audit.json
python3 -m crvfastsim transport-validation --merge 'audit/shard_*.json' --output merged.json
python3 -m crvfastsim transport-validation --summary merged.json
```

Outputs must be new paths. `--artifact PATH` may be repeated for job logs, FCL or other provenance. ROOT reads use
the highest tree cycle. The audit records every cycle, row and unique event count, checks the expected generated
event count from persisted IDs, and verifies that the reference is a matching subset and both fast arms retain
the full generator table. Input, geometry and source hashes accompany the results.

**Reference completeness is external evidence.** Accepted-event G4 tables need not contain every generated event.
This audit counts persisted accepted events; compare that count with successful worker/dump logs or an accepted-event
manifest. It cannot infer that an absent accepted event was legitimately rejected. Source-generator rows need
`(run, subrun, event, idx)` uniqueness. G4 particle tables require `genidx`; only explicitly fast-format arms may
reconstruct it from generated-muon ordering and `mukey` numbering.

**Statistical unit is the generated event.** For paired crossing totals A/B, the ratio error is
`sqrt(n/(n-1) * sum_i (a_i - (A/B)*b_i)^2) / B`, with every generated event, including zeros. This retains
within-event crossing correlations. Disjoint-run/subrun shards merge through sums of a, b, a², b² and ab, not by
averaging ratios or errors. The merge requires matching physics source hashes, geometry, diagnostic hash and bin
edges. The estimate is conditional on these samples and does not include model/systematic uncertainty.

**Momentum bins are truth diagnostics, not efficiencies.** Each arm is binned by its own bar-entry momentum;
energy-loss corrections move crossings between bins. Matched crossings use `(run, subrun, event, genidx, bar)`,
never proximity or time. Multiple muons in one bar remain distinct. The historical V1 comparator aggregates
primary muons by event/bar instead, so its crossing counts can differ slightly and its reported union of particle
records is not the number of generated events.

Scripts `run_transport_validation.sh`, `audit_transport_validation.sh` and `audit_named_transport.sh` retain the
existing extracted and named-config campaign conventions. They use new output tags, run sequentially on one core,
and never build or submit jobs. The run script creates only corrected candidates; existing G4 and fast baselines
are read-only. Audit-only scripts reuse candidate outputs and never regenerate transport. Set `PYTHONPATH` to
fixed code revisions before running; preserve manifests/logs with the output data, not in the code repository.

**Comparator caveats:** a V1 group PASS is not per-sector precision or full response/reco acceptance. The existing
MPV fit can return unphysical values with very large errors that its tolerance accepts; histograms with no populated
comparison bins can also be marked passing. Treat such checks as unsupported evidence, not successful shape tests.
No histogram, fitting or acceptance policy is changed by this diagnostic.
