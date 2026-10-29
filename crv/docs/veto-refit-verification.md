# Saved-refit output verification

`veto.refit_membership_report` checks the existing saved-refit finder output without
calling the finder or reconstructing any response. It is structural and provenance
verification, not independent reconstruction of coincidence physics. The successful
generation modules and output directory remain immutable.

## Admission and persisted evidence

An external SHA256 of the refit summary is mandatory. Validate the exact completed
transaction, artifact inventory, byte cap and generation source map. Associate
`codeN.txt` with the original writer's code-path order, not sorted JSON-map order.
Require each snapshot's hash, including the inherited membership transaction's
snapshots, to match its declared source. The five original JSON report roles must
be uniquely schema-qualified and hash-bound. Their inter-report pins are checked
again through the frozen pure admission helpers; prior verifier entry points are
never executed.

Reuse the full 9,000-frame registry, complete per-input calibration, Run1A geometry
and stock ignore-channel source/configuration bindings. Rebuild the admitted saved
stock-to-refit pulse bridge through exact indices and genuine waveform links. This
uses the frozen pure admission helpers also used by generation: it is not an
independent implementation of those helpers. Compare every saved source pulse,
waveform, bridge and calibration column, including original storage widths and
floating bits, against those admitted source tables. No raw ROOT, ADC samples,
photon tables, response stages or selected-track payloads are opened.

Input bounds remain 32 MiB per file, 128 MiB declared Parquet uncompressed bytes,
64 columns, 4,096 row groups; row limits are 9,000 registry, 15,360 calibration,
20,480 conditions, 2,048 per pulse/waveform/bridge/native table and four frames per
block. JSON report discovery is bounded and excludes transaction marker names.
These metadata and output bounds are not process-RSS guarantees.

## Native consistency checks

- Preserve all full input-qualified identities and immutable registry frame IDs.
- Canonical pulse indices, structural keys, integrated float64 PE, original relative
  and reco times, and flags match the admitted source. Stored ordering is not repaired.
- Native member links resolve to the exact active pulse collection and existing
  cluster; indices and member ordinals are unique, ordered and dense where required.
- Member sector/layer agrees with the bound geometry, the sector is enabled, and
  cluster sector type and distinct-layer count agree with its member set. The frozen
  configuration requires at least three layers.
- Integer side-hit counts equal the linked positive-PE member counts on each side.
  An absent side has zero PE/time sentinels; a populated side has positive PE;
  `poscalc=true` requires both sides. No new pulse-quality selection is applied.
- Every selected frame is retained. Pulse/cluster/member counts, summary records,
  diagnostics, aggregate census and the nonempty flag agree with decoded products.

The adapter's original pulse column order differs from the older pilot reporter's
canonical column order. Validate the actual saved schema first, then project named
columns for the shared structural checker; never silently coerce numerical types.

Floating cluster PE, time, position and slope are hash-bound generation observations,
with finite/domain checks—not independently recalculated physics. In particular,
raw pulse PE sums need not equal status-compensated cluster PE, and raw times omit
sector propagation corrections. A consistent permutation of member ordinals cannot
be independently ruled out without rerunning the finder: this verifier confirms the
saved order's integrity and structure, not independent recovery of that order.

## Publication and scope

Recheck consumed files, code, inherited transactions and the complete output
transaction immediately before publishing a new report outside every input
location. The report includes source snapshots and is capped at 2 MiB. It uses
exclusive JSON creation; it does not claim the atomic completion or crash-durability
protocol of `BatchArtifacts`. Existing reports are never overwritten.

```bash
python3 -m crvfastsim.veto.refit_membership_report /data/EXISTING_REFIT \
  --expected-summary-sha256 REFIT_SUMMARY_SHA --out /data/NEW_VERIFIED.json
```

No finder, fitting, response, stock reconstruction, coverage calculation, grid
reconstruction or previous verifier runs. Stock-native membership is not compared.
No same-pulse stock equivalence, timing closure, representative-exposure estimate,
finite-library uncertainty or full-campaign admission follows. `F=null`; C4 stays
stock. The original generation runtime/RSS remain labeled historical observations;
verification timing and memory are reported separately.

## Completed retained check

The saved-output check passes for all 87 pulses, 80 waveform identities, three
coincidences and 17 native links across the three immutable full-registry frames.
No finder or upstream stage was repeated. Original generation summary/receipt
hashes are unchanged. The report is 736,475 bytes including executed-source
snapshots, below the 2 MiB cap. Whole-process verification took 2.35 s with
239,444 KiB peak RSS; its internal pre-publication observation was 0.829 s.

All 124 new targeted tests pass locally and remotely. Fixtures exercise real
Parquet/receipt operations and the pure bridge on synthetic authority/native
products, not independently verified detector physics. They include consistent
empty output and explicitly demonstrate the limitations: altered finite scalar
properties or a consistent alternative native order cannot be identified as a
finder defect from these saved structural checks alone.
