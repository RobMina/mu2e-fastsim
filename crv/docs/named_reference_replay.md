# Reverify completed named references without rerunning transport

`python3 -m crvfastsim.analyses.named_reference_replay` verifies the completed
20-shard/configuration campaign against freshly generated reference certificates, then
re-merges saved V1 histogram accumulators and paired-event sufficient statistics.
It never generates G4 or fast transport, submits jobs, changes physics tolerances, or
rewrites an old artifact. Use a new output directory; an interrupted replay cannot resume.

First rerun `reference_certification` on each retained dump partition (101, 102–104,
105–120), writing three new certificates. Its checks read the highest ROOT tree cycle,
require exactly 50,000 generated event identities/shard, compare accepted identities and
all generator values to the original source, and require successful worker/dump counts.
The replay requires all 40 shards, even when a physics verdict fails.

```bash
python3 -m crvfastsim.analyses.reference_certification BASELINE_DUMPS \
  --first 101 --last 101 --expected-events 50000 --output NEW_CERT101.json
python3 -m crvfastsim.analyses.reference_certification PILOT_DUMPS \
  --first 102 --last 104 --expected-events 50000 --output NEW_CERT102_104.json
python3 -m crvfastsim.analyses.reference_certification REMAINDER_DUMPS \
  --first 105 --last 120 --expected-events 50000 --output NEW_CERT105_120.json
python3 -m crvfastsim.analyses.named_reference_replay --campaign COMPLETED_CAMPAIGN \
  --certificates NEW_CERT101.json NEW_CERT102_104.json NEW_CERT105_120.json \
  --outdir NEW_REPLAY
```

Fresh certificates must agree with every historical persisted artifact/count, not merely
the total. The replay checks the completed summary, disjoint shard coverage, retained
101 candidate, all candidate/audit/path/emit fingerprints, recorded production-log
fingerprints, geometry, before/after/final source snapshots, and per-config/overall
verdict consistency. It checks saved audit/path provenance against the **historical**
source snapshot, not the currently imported transport. Comparator, certifier and
statistical/path diagnostic implementations must match their historical hashes.

Recomputed audit dictionaries and V1 statistics must equal the saved dictionaries
exactly. This is tested on the same ana environment as production; a numerical-library
version difference that changes fit results is a failed exact replay, not permission to
relax agreement. The fresh reference checks establish completeness relative to the
successful worker evidence, not independent reconstruction of the G4 acceptance decision.
Hashes establish consistency, not cryptographic producer attestation.

Outputs are `verification.json` with input/source evidence, per-config `audit1M.json`
and `v1/` results, and `tables.md` with grouped verdicts and every sector's paired SE.
Verification PASS means evidence/replay success, **not** physics PASS: inspect `v1_pass`.
Partial output without `verification.json` is not successful verification. CLI exit 0
means verification completed; operational/consistency errors are nonzero.

V1 retains event/bar aggregation, while the paired diagnostic retains genidx; their
crossing counts can differ slightly. Generated-event totals come from certificates,
not V1's historical particle-record `events_union`. Uncertainties are paired statistical
SE clustered by generated event (including zero-crossing events), not model/systematic
uncertainty. Grouped V1 PASS does not certify each sector or response/reconstruction.

## Integration boundary

The original `named_reference_campaign` generation driver remains deliberately pinned
to the frozen validation core and compatible retained evidence. Porting it to a newer
branch does not authorize generating new candidates against old producer fingerprints.
Use the read-only replay above for existing evidence; a new transport campaign requires
its own revision-matched plan and reference provenance. Historical campaign-specific
submission/recovery scripts are retained as provenance and require fresh user approval;
none should be run merely to reproduce these statistics.
