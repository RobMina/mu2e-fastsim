# Bounded common-photon extreme-time probe

`veto.optical_timing_probe` follows a verified `step_timing_audit` by selecting every
frame containing an endpoint floor-fold/corrected-fmod residual. It includes **all
steps in those frames**, all represented sectors, both readout ends and all emitted
channels. It neither selects retained stock-member channels nor repeats an earlier
hand-picked boundary sample. A hard ten-frame cap prevents this diagnostic becoming
an unreviewed broad response campaign.

## Scope and reproducibility

This is a new optical draw on an unchanged raw replay. It is not historical stock
or fast RNG recovery, not a selected-track exposure sample and not a new F result.
Defaults, yields, production optics and stock outputs are unchanged. No SiPM,
thermal, waveform, pulse fit or coincidence stage runs. CE replay-v2 is rejected by
the existing input gate.

```bash
python3 -m crvfastsim.veto.optical_timing_probe \
  --audit /data/step_timing_audit \
  --verified /data/step_timing_verification.json \
  --paired /data/fixed_stock/paired_intended_per_sector_v1.json \
  --photon-dev /data/conditions/CRVPhoton.txt \
  --offline /release/v13_41_00/Offline \
  --lut-dir /cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0 \
  --seed 1027000 --out /data/new_optical_probe
```

The census manifest, complete source ledgers, fixed-stock checks, metadata,
conditions payload and deviation table are hash-bound. Particle PDGs join by full
`(run,subrun,event,SimParticle key)`, never nearest time or position. Selected replay
rows must exactly equal the census's original ROOT rows and timing payload. The
fixed-stock event census must include every input frame with matching float32 bunch
time, spill and window. Calibration/deviation channel coverage is complete and
unique. Source, lookup and geometry hashes are pinned. Existing outputs are refused.

The census used v13_39_00's photon generator, whereas fixed stock uses v13_41_00's
per-sector yield-sharing fix. Both source identities are retained. A checked bridge
requires the complete `CrvPhotonGenerator::produce` body (including timing) to be
**byte-identical** across these pinned sources. This is not general release or full
response equivalence. The nominal intended per-sector yields remain unchanged.

## Common draws and four arms

The unchanged production `photons.make_photons` kernel runs twice with identical
PCG64 initial states: raw absolute endpoints and historical folded relative
endpoints. The driver requires exact step/channel output identities and identical
final RNG state. Production optics does not branch its random draws on finite
endpoint magnitudes. NaN endpoints skip in both calls. Post-draw timing/step masks
consume no RNG and never discard candidate identities from the persisted ledger.

The stable photon key is **input hash + draw seed + source step row + channel +
ordinal**. `photon_id` also records original candidate order for downstream reuse.
These keys label a **fixed, ordered subset and draw**. This implementation is
explicitly **not a subset-, permutation- or chunk-invariant keyed optical generator**;
changing the selected rows or chunking requires a new draw contract. Common-stream
control within this bounded probe must not be described as the full-sample keyed
optical experiment.

| Arm | Photon generation | Final arrival wrap | Stock step predicate |
|---|---|---|---|
| `historical` | historical folded endpoints | no | no |
| `arrival_only` | raw endpoints | yes | no |
| `rejection_only` | historical folded endpoints | no | yes |
| `stock_timing` | raw endpoints | yes | yes |

Every arm also applies the same status-live mask and SiPM upper-time eligibility.
This is **not** avalanche/PE acceptance. The raw upper cut uses the stock absolute
comparison `wrapped <= origin + 1500`, preserving operation order. There is no
lower photon-time cut. The `[1400,1500]` diagnostic region is relative to the bunch
origin and is not the absolute coincidence flash gate.

A fifth saved comparison, wrapping the folded photon time, isolates numerical
non-equivalence to wrapping the **actual raw generator result**. At enormous raw
times, floating-point addition can quantize the optical delay before modulo; do
not reconstruct raw arrivals by adding an integer period to folded photons. No
absolute tolerance is used to suppress a difference. Saved residual distributions
and exact gate flips are separate checks.

## Outputs and reinspection

Each input produces source steps, candidate photons, all selected event rows and
the union-of-emitted-channels table. The summary binds every output, records the
source-version bridge, RNG states, seeds, arm counts, extreme-step contributions
and measured runtime. `F` and `finite_library_uncertainty` remain null and
`timing_closure` remains false.

```bash
python3 -m crvfastsim.veto.optical_timing_report /data/existing_optical_probe \
  --out /data/new_saved_report.json
```

The report verifies all recorded asset/input/code/output hashes and recomputes
photon identities, ordinals, wrap arithmetic, step masks, eligibility summaries,
per-frame and per-channel tables without generating photons. It retains up to 100
extreme-step photon rows for direct inspection; complete rows remain in parquet.
It reports counts of common accepted photons with time differences above declared
thresholds, so numerical inequality is not mistaken for a large physical effect.
The raw optical samples themselves are hash-bound observations, not independently
regenerated or proven against stock optics by this report.

## What remains

Matching eligibility does not prove unchanged pixel recovery, waveform samples,
fit parameters, membership or veto coverage. Reuse the saved photon IDs with
shared keyed downstream draws to test those effects. Optical seed variation,
full-sample keyed generation, all-channel/sector closure, CE joint exposure and
upstream-library uncertainty are separate gates. This selected-frame diagnostic
cannot authorize replacing stock C4 or tuning light yields.
