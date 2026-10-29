# Bounded real-asset keyed optical smoke

`veto.keyed_optical_smoke` admits a verified raw-step census, fixed-stock response
checks and pinned geometry/lookup/source assets. It exercises the separate
`keyed_optics` diagnostic, not production response or the previous common-stream
probe. No SiPM, waveform, coincidence, selected-track F or stock regeneration runs.

## Selection and draw schedule

Before generating photons, select the first full-tuple frame in each input not
used by completed boundary controls, plus every non-exceptional frame with zero
stock-eligible steps. The prior exception frames are excluded by the census and
the historical boundary event numbers are explicitly excluded. This deterministic
smoke selection does not represent exposure. Include all steps/channels and retain
all selected frames even when zero photons survive. Reject more than ten frames
or 10,000 source steps per input, and more than 300,000 photons per generation call.
Two distinct explicit response seeds apply to every input; the input SHA domain
separates files. They remain two draws on the same selected frames, not twice the
exposure or an uncertainty estimate.

Validate all original input rows against the hash-bound census, the raw replay
manifest/tree census, full fixed-stock event coverage and complete channel
calibration/deviation coverage. Bind intended per-sector yield policy, source
version timing bridge and exact lookup/geometry hashes. No stock physics branches
beyond event/condition tables are needed. New output directories only.

Each seed generates raw and historically folded endpoint photons. Exact full keys
must match. A reversed three-part external partition with a different internal
substep batch must reproduce raw keys/times exactly; a fixed every-other-step
subset must reproduce exactly its parent photons. These are software-invariance
checks, not independent response replicas. Save four timing masks on the full
candidate table: historical, arrival-only, rejection-only, stock timing. Status
and inclusive upper eligibility are common; there is no lower photon cut. Preserve
raw arithmetic and the float32 bunch origin. Save per-frame and union-channel
ledgers separately for each response seed.

## Downstream root bridge

The full immutable optical key remains authoritative. A versioned vectorized
SplitMix64 digest of all input digest words, seed and integer key components yields
a nonnegative 63-bit `photon_id` usable as a diagnostic downstream root. It never
uses a rank or chunk index. Collision checks run on the full generated candidate
union across inputs and seeds and fail rather than resolve collisions by rank.
Any larger consumer must repeat that global check when pooling separately generated
chunks; a hash is not a collision-free encoding. The full tuple stays in the saved
ledger so the bridge can be recomputed. Existing downstream drivers are not
silently pointed at this new schema and do not yet accept multiple run/subrun
identities through their event-only keys.

## Evidence and verification

The summary includes input/asset/code hashes, dependency versions, selected full
event census, response seeds, invariance results, per-seed counts and measured
resources. A code snapshot accompanies the smoke. `--verify` reads saved products,
checks mandatory manifest coverage, repeats raw-input/fixed-stock admission,
recomputes roots, joins, four-arm masks and frame/channel ledgers, and never
generates optics. Repeating admission rereads source tables and calibration;
it does not rerun detector simulation. Conflicting hashes for the same path
across manifest sections are rejected rather than overridden. It does not independently
prove the saved raw photons against stock optics. Missing domains, duplicate keys,
changed assets, missing required hashes and overwritten outputs fail closed.

```bash
python3 -m crvfastsim.veto.keyed_optical_smoke \
  --audit /data/step_census --verified /data/step_census_verified.json \
  --paired /data/fixed_stock/paired.json --photon-dev /data/CRVPhoton.txt \
  --offline /release/Offline --lut-dir /data/lookup \
  --seed 3027000 --second-seed 3027001 --out /data/NEW_SMOKE
python3 -m crvfastsim.veto.keyed_optical_smoke \
  --verify /data/NEW_SMOKE --out /data/NEW_VERIFICATION.json
```

A successful smoke admits this bounded real-asset diagnostic only. A full retained
sample runner, complete downstream frame-key contract, response-seed statistics,
CE-selected joint exposure and upstream-library covariance remain separate work.
Do not scale runtime linearly from this sample or replace stock C4.
