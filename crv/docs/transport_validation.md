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

## V1 strata and clustered binned ratios (`crvfastsim.analyses.v1strata`)

`scripts/run_named_candidate.sh CONFIG CAMPAIGN OUTDIR` transports a certified named-reference campaign (seed =
subrun - 100), runs `v1compare` per shard, merges it, then `scripts/run_v1strata.sh` (shard, merge). The strata split
each V1 group by layer, entry face, position along the bar, incidence, entry momentum, crossing rank and below-plane
x/X0, with event-clustered errors. With `--v1-stats` (the merged `v1compare_stats.json`) the strata also recompute
v1compare's 20-bin path-length and visible-energy ratios with bin variances clustered by generated event.
v1compare's ratio chi2 treats crossings as independent; in the Run 2 side walls one muon often crosses several bars
with nearly equal paths, so the clustered chi2 is the statistically consistent one. It is a diagnostic: v1compare's
verdict and thresholds are unchanged. `PATH_MAP=FROM=TO` runs a campaign from a local copy of its inputs.

## Below-plane world margin (`RunConfig.below_margin`)

For CRVs that wrap the DS (`below_world`), the material between the MSC plane and the bars is read inside the CRV's
bounding box grown by `below_margin`. Near-horizontal muons reach the Run 2 side walls through the hall walls and
dirt beside the hall, below the plane and more than 2 m outside the box; without them the fast arm had about 10 %
too many crossings at near-normal incidence on the side walls (short paths), which pulled the side-wall mean path
length down. `run2_v10` uses 20 m (100 m gives the same V1); extracted (no `below_world`) and Run 1A / 1B keep 2 m,
so their outputs are unchanged. `--below-margin MM` overrides it. `--below-msc` (bar-to-bar scattering below the
plane, `transport.scattered_crossings`) is opt-in: muons scatter out of the CRV there but never into it, and its V1
is worse than the straight lines.
