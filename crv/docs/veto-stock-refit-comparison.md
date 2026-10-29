# Saved stock–refit membership comparison

`veto.stock_refit_comparison` compares retained stock member sets with the already
saved finder output on stock-ADC refits. No finder, fit, response, stock job,
coverage calculation or earlier verifier is executed. This is a robustness test
under changed pulse values, not identical-input finder equivalence.

## Actual legacy schema and allowed comparison

The retained boundary-v4 stock-member tables contain full run/subrun/event, stock
pulse collection index, bar/SiPM/channel/sequence and the stored pulse time and PE.
They omit parent-cluster and member ordinals. The bound stock-cluster tables contain
exactly one cluster for each selected event. This permits an unambiguous **unordered
member-set** comparison after checking that one-cluster-per-event requirement on
both sides. A multi-cluster or missing-cluster event fails rather than guessing a
parent from time, channel proximity or collection position.

The stock cluster fields are only start, end, average time, float32 PE and integer
side-hit counts. The export lacks layer count, sector type, side PE/time, position,
slope and position-method flag. These fields and member ordinals are not invented.
The legacy inverse-reconstructed relative time is not used as a genuine original
pulse time or as a matching key.

## Binding and index mapping

Require external hashes of the saved-refit summary, original completion receipt
and saved-output verification report. Reuse their pure provenance checks, not the
verification entrypoint. The original receipt anchors all refit artifact bytes;
a resealed current receipt cannot substitute for it. Follow the refit summary's
bound boundary report to its exact stock-member and stock-cluster payload hashes.
No raw ROOT or original photon/response table is opened.

Preserve the full 9,000-frame registry and source-qualified identities. For each
block, map stock members by full input/run/subrun/event/frame plus `stock_idx` into
the complete saved stock-to-refit bridge. Require exact original stock time/PE
storage bits and matching channel/sequence fields. Check the bridge's refit index,
waveform/start-TDC/sequence/channel and saved refit PE/time/flags against the native
pulse table. Native links must resolve through that same bridge, not through a
nearest-time match. Retain every pulse and reject a changed ignored-channel domain.

Report common, stock-only and refit-only stock pulse indices per frame. Membership
differences are valid observations, not grounds for retrying seeds or changing cuts.
Legacy positive-PE side-hit counts are checked against the retained members.
Stored scalar values and explicit refit-minus-stock differences are reported,
without tolerance-based equality or independent reconstruction of cluster physics.
Stock float32 PE and refit float64 PE remain distinct storage representations.

## Bounds and output

Strict flat table schemas, finite numeric fields and metadata bounds precede decode.
Limits are four frames per block, 2,048 rows per member/pulse/bridge/cluster table,
9,000 registry rows, 32 MiB per input file/transaction, and 128 MiB declared Parquet
uncompressed data with at most 64 columns and 4,096 row groups. These are not RSS
guarantees. Code snapshots and all consumed evidence/receipts are checked again
before exclusive publication of a new JSON report, capped at 2 MiB. There is no
atomic-completion or crash-durability claim, and no overwrite mode.

```bash
python3 -m crvfastsim.veto.stock_refit_comparison compare /data/EXISTING_REFIT \
  --verified /data/EVIDENCE/EXISTING_REFIT_VERIFIED.json \
  --expected-summary-sha256 REFIT_SUMMARY_SHA \
  --expected-verified-sha256 REFIT_VERIFIED_SHA \
  --expected-complete-sha256 ORIGINAL_REFIT_COMPLETE_SHA \
  --out /data/SEPARATE_OUTPUT/NEW_COMPARISON.json
```

The `schema` subcommand inspects only the four bound legacy table schemas; it does
not perform the comparison or publish a physics certificate.

## Scope

These three deliberately selected events cannot determine an efficiency or F.
Coverage one in their previous saved-grid check came from event-gate rejection,
not complete window occupancy, so coverage agreement is not a substitute for this
member comparison. Native-order equality, missing scalar fields, production timing
closure, CE exposure, source-library independence and finite-library covariance
remain outside this check. `F=null`, `finite_library_uncertainty=null`; C4 stays stock.

## Completed retained result

All three unordered member sets agree exactly: 8, 4 and 5 common members, with
zero stock-only or refit-only members. Integer side-hit counts also agree. This
holds for the changed refit pulse values, not an identical-input experiment.

| Frame | Common members | Refit − stock average time (ns) | Refit − stock PE |
|---:|---:|---:|---:|
| 2504 | 8 | +2.7490271349961404e-5 | +2.4173980790465066e-6 |
| 3280 | 4 | −2.4940391767813708e-5 | +4.807376114968065e-7 |
| 5754 | 5 | +9.82080794074136e-5 | +8.610250461060787e-6 |

Start/end residuals are also retained explicitly; none of these scalar differences
is promoted to exact equality by a tolerance. Missing legacy ordinals still prevent
native-order comparison, and missing scalar fields remain untested.

All 174 new targeted tests pass locally (12.79 s) and remotely (51.46 s).
Fixtures use synthetic native products and assertions with real Parquet/receipt
operations; forbidden-entrypoint guards exclude physics, raw access, earlier
verifiers and coverage. The retained comparison completes on its first attempt:
4.93 s whole-process wall time, 219,352 KiB peak RSS, 797,400 report bytes including
source snapshots. No previous finder, verification or coverage result was rerun.
