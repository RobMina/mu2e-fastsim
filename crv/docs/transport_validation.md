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

Outputs must be new paths. `--artifact PATH` may be repeated for job logs, FCL or other provenance.
`--baseline` is optional: omit it when only same-input candidate/reference evidence exists; no surrogate or reused
unpaired baseline is invented. Merges reject audits with different comparison sets. ROOT reads use
the highest tree cycle. The audit records every cycle, row and unique event count, checks the expected generated
event count from persisted IDs, and verifies that the reference is a matching subset and both fast arms retain
the full generator table. Input, geometry and source hashes accompany the results.

**Reference completeness is external evidence.** Accepted-event G4 tables need not contain every generated event.
This audit counts persisted accepted events; compare that count with successful worker/dump logs or an accepted-event
manifest. `--expected-accepted N` enforces agreement with that external successful-worker count using unique
reference-particle event IDs at the highest cycle, not particle rows. The count gate does not verify accepted-event
identities. Without it, completeness remains explicitly unchecked. It cannot infer that an absent accepted event
was legitimately rejected. Source-generator rows need
`(run, subrun, event, idx)` uniqueness. G4 particle tables require `genidx`; only explicitly fast-format arms may
reconstruct it from generated-muon ordering and `mukey` numbering.

**Statistical unit is the generated event.** For paired crossing totals A/B, the ratio error is
`sqrt(n/(n-1) * sum_i (a_i - (A/B)*b_i)^2) / B`, with every generated event, including zeros. This retains
within-event crossing correlations. Disjoint-run/subrun shards merge through sums of a, b, a², b² and ab, not by
averaging ratios or errors. The merge requires matching physics source hashes, geometry, diagnostic hash and bin
edges. The estimate is conditional on these samples and does not include model/systematic uncertainty.

**Paired means and energy per path.** Each generated event contributes `[sum_a, weight_a, sum_b, weight_b]`.
For length/energy crossing means, weights are crossing counts; for energy per path, weights are summed lengths.
`energy_per_path` is the ratio of summed energy to summed length, not the average of crossing-by-crossing ratios.
For the arm ratio `R = (Sa/Wa)/(Sb/Wb)`, the totals gradient is
`g = [Wb/(Wa*Sb), -R/Wa, -R/Sb, R/Wb]`. The variance is `N/(N-1) * g.T @ sum(x_i*x_i.T) @ g`;
`g.T @ sum(x_i) = 0`, so this equals the centered influence covariance. Raw totals and cross-products are stored
and summed across disjoint shards. `contributing_events` accompanies the full generated-event count; fewer than
two contributing events leaves SE unavailable, rather than a misleading zero. Materially negative variance fails;
only floating-point roundoff within a contraction-scaled tolerance is clamped. Nearly identical arms can suffer
raw-moment cancellation; this is not a high-precision numerical identity test.

`matched_only` conditions on common `(run,subrun,event,genidx,bar)` identities. `a_only`/`b_only` hold unmatched
counts, energy/path sums and means, with no unpaired uncertainty claim. These are conditional truth diagnostics,
not efficiencies. Per-shard entry-momentum quantiles are retained, never averaged into pooled quantiles. Tests
compare the uncertainties to a direct centered event-influence calculation and require pooled/merged equivalence.

Audit-time source fingerprints cover relevant transport/tables/eloss/materials/world/crossings and CRV
configuration/geometry/CLI-driver files. They do not by themselves prove producer code for existing ROOT tables;
retain the original production manifests/logs as artifacts. Merges require compatible recorded hashes.

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

## Fail-closed V1 evidence

A V1 group PASS is not per-sector precision or full response/reco acceptance. `v1compare` reports validity and
reason codes separately from physics discrepancies. Every configured group is required; empty arms, an invalid
event universe, or missing in-range matched-time evidence fail rather than disappearing from the report.
`matched` and `matched_in_time_range` expose time-histogram truncation; the median remains an in-range diagnostic.

The Moyal estimator, 15–70% fit core, >=100 histogram entries, numerical physics tolerances and 2-sigma allowance
are unchanged. Fit evidence additionally requires at least four populated core bins, positive amplitude/MPV/scale,
a peak inside the fitted core, and finite positive-definite covariance without optimizer warnings. Scale, MPV
standard error or scale standard error exceeding the fitted-core width makes the fit unidentified. These are evidence-validity guards,
not a retuning of the 1% MPV agreement threshold; failed fits are not clipped, refitted or replaced by peak finders.

Shape checks retain the 20-bin construction, >=400 reference entries per eligible bin and chi2/ndf <=1.5 threshold.
They require positive finite histogram totals, at least one eligible comparison bin, positive candidate entries
in each eligible bin, and finite chi2. Zero eligible bins is unsupported, not a perfect match. The eligible-bin
count remains explicit: one supported bin is not a claim of complete 20-bin shape coverage. Genuine chi2=0 passes.

Completed comparisons and merges write their JSON/figures before exiting **1 on FAIL** and **0 on PASS**. `--emit`
only accumulates a batch and does not certify it. Campaign drivers that collect several comparisons must record
exit codes and continue only when the expected stats JSON exists; an operational exception is not a physics FAIL.
`scripts/recheck_transport_v1.sh OLD NEW` rechecks the retained Thread A pilot tables and extracted batches into a
new directory, without transport or G4 generation. The dedicated tests are `tests/test_v1validity.py`.

## Completed-campaign replay

For a completed full-statistics campaign, use the read-only
[`named_reference_replay`](named_reference_replay.md) workflow: freshly certify the existing dumps,
verify historical input/source/result evidence, and recompute V1/paired statistics without transport.
The original generation driver remains frozen to its recorded producer revisions; importing the tools
from a newer branch does not recertify a newer transport implementation.

## Existing named-reference extension

`scripts/dump_named_transport_reference.sh OUT BUILT_ART_AREA [FIRST LAST]` reads existing Run1A/Run2 v1 art
pairs directly from pnfs, defaults to subrun 101, requires the exact already-configured `MUSE_WORK_DIR`, and records
worker/dump logs plus dumper source/library hashes. It neither builds nor generates events; configure the existing
art area before calling it. The output directory must be new. Input-file existence alone is never certification.

`scripts/validate_named_transport_references.sh REFS OLD_LEDGER NEW_OUT` handles the known subrun-101 50k references
and retained 20k pilots, using their externally checked accepted counts. It makes new seed-1 candidate tables only
for the 50k inputs, audits all sectors, re-audits (does not rerun) old candidates, and combines disjoint subruns to
70k per configuration. Keep the Run1A/Run2 directories distinct: both new references use run/subrun 1400/101.
This is a bounded workflow for those samples, not a claim that the planned 20-shard grid campaigns completed.
`scripts/audit_named_paired_means.sh REFS OLD_LEDGER CANDIDATE50K NEW_OUT` recomputes only the diagnostic audits
on those existing tables and merges them; it never reruns transport.
`scripts/propose_named_reference_extension.sh pilot|remainder` only prints exact commands for a 3-job/config
continuation and then 16-job/config remainder. It does not execute commands, stage FCL or refresh credentials.
Submission requires explicit user approval, fresh queue/output deduplication, unchanged staged FCL and a data-area
submit directory. The remainder additionally requires successful pilot copy-out and persisted-count certification.

## Saved path geometry diagnostics

```bash
python3 -m crvfastsim transport-paths --gen GEN.root --reference G4.root --candidate FAST.root \
  --config run2_v10 --expected-events 50000 --gdml WORLD.gdml --output paths.json
python3 -m crvfastsim transport-paths --merge paths20k.json paths50k.json --output paths70k.json
```

All configured sectors/groups are retained. The analysis matches truth identities, reconstructs each entry-ray
remaining chord in the axis-aligned bar, and classifies unique opposite thickness faces, edge/clipped faces, or
ambiguous endpoints (0.01 mm tolerance). It compares saved path length and entry-ray chords, G4 true length versus
entry-ray chord, and G4 true length versus the sum of each CrvStep record's endpoint chord. Ratios use the same
paired-event moments as the main audit. Decomposition factors multiply; percentage shifts must not be added as
if they were independent. `true/entry-ray` includes changes in exit face/direction, not just in-bar path excess.

Geometrical attribution uses `both_nonstopping_clean_single_record`: both arms have p1 >1 MeV/c, one record,
unique boundary faces, valid rays and end times, and no shortened path relative to endpoints. A single CrvStep
record can aggregate many G4 steps; it is only a conservative single-entry proxy. Fast stopped rows can retain
full-chord endpoints while their stored path is shortened; they are flagged and excluded from clean populations.
Nonfinite/reversed G4 end times are counted, never imputed into clean evidence; the last record is ordered by
its finite start time. `all_matched` includes these flagged cases and is not the attribution sample.

Clean incidence bins `[0,.2,.5,.8,1]` use the absolute G4 direction component along the thickness normal;
clean momentum bins use G4 bar-entry momentum. These are **truth-conditioned diagnostics, never correction models**.
Small populations remain explicit, including contributing-event counts and unavailable errors. No world-segment
history is invented: the tables do not carry one. Saved source/geometry fingerprints describe audit-time code,
not complete production provenance; reference-completeness certification stays with the main audit and logs.

`scripts/audit_named_paths.sh REFS OLD_LEDGER CANDIDATE50K NEW_OUT` applies this to the retained 20k and 50k
samples, then merges disjoint subruns. No model parameter, physics gate or transport result is modified.
