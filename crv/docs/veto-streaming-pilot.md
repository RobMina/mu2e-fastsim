# Bounded combined retained streaming pilot

`veto.streaming_pilot` integrates the frozen per-input indexed replay cache,
full-key optical kernel and bounded source-qualified downstream response. It is
a deliberately bounded new-frame integration test, **not** a full retained-sample
campaign, production path, coincidence validation or veto-fraction result.

## Prior evidence and immutable domains

The caller supplies explicit SHA256 hashes for the original retained plan summary,
the completed full-input cache audit summary and the completed four-frame saved
response summary. Admission repeats both saved verifiers, checks the complete
9,000-frame registry and all transitive conditions/source bindings, and reuses the
saved calibration and original cache-audit frame/batch fingerprints. Prior smoke
photons and downstream products are verified, never regenerated. Frozen helpers
and previous evidence are not modified.

The inherited saved-response verifier validates original raw and stock ROOT data
as well as saved products. It materializes some raw branches and performs the
original smoke-selection scans. Thus **saved verification means no physics
regeneration, not no raw reads**. The new admission first checks raw counts,
resident typed-payload estimates, exact branch schemas and header-only basket
limits with the frozen indexed-loader guards. This bounds those named inputs,
not pandas/uproot temporaries or total RSS. Cache/plan verification also retains
its saved-Parquet admission and hashing. The execution-loop timer starts after
admission and asset loading; external `/usr/bin/time -v` measures the whole
process. Cumulative RSS can include admission even when the loop timer does not.

Selection is deterministic: for each input take the lowest existing complete
planned batch that contains neither a saved smoke frame nor a completed
extreme-time control. Extreme controls are recovered by their original full
source/event identities from the timing census, not merely a short event-number
list. The additional historical exclusion list remains conservative. The pilot
accepts at most **three whole batches and nine unique frames**, one batch per
input, without changing the full registry or rebatching selected frames. Empty
and zero-output frames remain explicit. This selection is an integration domain,
not a representative sample or added independent exposure.

## Streaming and exact identities

The executor opens one cache at a time and selects each planned batch once. Every
raw column is checked against its immutable original cache slot; source-qualified
typed fingerprints and per-frame counts must exactly equal the completed cache
audit. It reuses that selection across both optical/downstream seed pairs
3027000 and 3027001. Loops are over files, bounded batches, seeds and four fixed
arms; row work and identity joins are vectorized.

`streaming_optics.make_batch` makes exactly two frozen keyed optical calls:
original raw endpoints and historical folded endpoints. It requires exact common
full photon keys, then builds the unchanged timing table. NaNs skip; infinities,
bad domains and resource overruns fail. No endpoint or photon is filtered away
based on a successful response. Original `source_row`, input SHA and complete
run/subrun/event identities remain authoritative; `event_row` is only a local
lookup. Limits are immutable, positive native integers which may be lowered but
not raised above4frames/20ksteps/2M simulated substeps/200ksubsteps per internal
chunk/300k photons per call. Photon/state/attempt/waveform/fit budgets remain those
of the existing bounded response.

One `RootUnion` spans all inputs, batches and seeds, preallocated for at most
1.8million int64 roots. It verifies each full key against the frozen bridge and
rejects any duplicate/collision before accepting a child. The final digest is
SHA256 of sorted little-endian int64 roots. This detects bridge collisions on the
actual union; it is not proof of independent random streams.

Each seed's candidate timing table feeds historical, arrival-only, rejection-only
and both-stock-timing arms. All status-live detector channels—including optically
quiet channels—receive the explicit1e-4/ns thermal model. Downstream seeds equal
the optical seeds and use the full-registry helper mapping. This is not a
factorial decomposition of optical versus downstream variance. Every output
stage is validated against explicit frame identity, conditions, phase, optical
ancestry, physical sample support and the reconstructed resource receipt.
Historical-versus-other-arm comparisons are structural diagnostics, not truth
matching and not required to be equal.

## Bounded persistence and failure

The top transaction is8MiB and each batch/seed child64MiB. Six children at most
therefore give **392MiB =411,041,792bytes** maximum logical pathname bytes through
the writers, including status, receipt copies and failure reserves. This is a
fixed bound, not a size extrapolation or disk-quota/RSS guarantee. Child files
include raw selected events/steps/particles, candidate photons, all12 response
stages for each arm and validation receipts. The top binds every child receipt,
source snapshots, full/selected registry, calibration and the exhaustive
frame×seed-pair×arm ledger. Candidate identities survive every arm's rejection.
Only one arm's full products plus the small structural reference are held at a
time. Results are canonicalized by batch/seed regardless of input census order.

Any cap, source mismatch, missing/duplicate frame or missing arm/seed leaves the
top incomplete. Completed children do not make an incomplete pilot acceptable.
No resume, overwrite, truncation, hidden frame split or denominator renormalization
is allowed. Output must be outside the input plan/cache/response directories.
The atomic-publication and final-name crash-durability limitations of
[veto-batch-artifacts.md](veto-batch-artifacts.md) apply unchanged.

```bash
python3 -m crvfastsim.veto.streaming_pilot \
  --plan /data/retained_plan --cache-audit /data/cache_audit \
  --response /data/saved_response \
  --expected-plan-sha256 PLAN_SHA --expected-cache-sha256 CACHE_SHA \
  --expected-response-sha256 RESPONSE_SHA --out /data/NEW_PILOT_PREFLIGHT.json --preflight
```

Preflight verifies the prior evidence and freezes the selection/budgets without
opening a new indexed cache or executing optics/response. It still has the
inherited raw/stock verification reads described above. Remove `--preflight` and
supply a new directory to execute only that bounded pilot after resource review.
No tool automatically launches a remote job or a full campaign.

`veto.streaming_pilot_report` checks saved receipts and all source bindings,
rebuilds photon timing fields/masks from saved raw/folded arrivals, verifies
original cache fingerprints/joins, response structure/receipts and complete
frame/arm/seed coverage, and reproduces the structural comparisons without
regenerating optical/SiPM/waveform/fit products. It requires the exact saved summary
and environment schemas, checks saved Parquet metadata before table loading, and
rechecks all top/child receipts after processing to detect late artifact changes.
Raw optical arrivals and response arithmetic remain hash-bound measurements, not
an independent physics rerun.

A combined full-retained executor still needs campaign-level output sizing and
admission, completion across the complete registry, coincidence/native pulse
membership and frozen-track coverage. CE joint selected exposure, physical source
reuse and finite-library covariance remain independent open gates. All outputs
retain `execution_admitted=false`, `timing_closure=false`, `F=null` and
`finite_library_uncertainty=null`. C4 stays stock.
