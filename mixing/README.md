# Stock-frame pileup sampler (opt-in)

This package samples joint tracker/CRV detector-step frames **before digitisation**.
It changes no default chain. Synthetic tests cover the probability laws, donor
ordering, empty events, particle-reference remapping and timing. Native response
validation and optional fast-digitiser integration are separate companion changes.

## Stock semantics

The implementation follows `Production/JobConfig/mixing/{prolog,epilog}.fcl` and
Offline `EventMixing/{MixBackgroundFrames,ProtonBunchIntensityLogNormal}`:

- Draw one primary-event bunch intensity, not one per stream. A non-beam primary
  uses the truncated lognormal; a beam primary uses its intensity-weighted law
  (`mu -> mu + sigma**2`). Truncate before nearest-integer POT rounding.
- Conditional on that intensity, independently draw the four Poisson counts.
  `stock_rates` multiplies the **campaign's SimBookkeeper efficiencies** and
  epilog factors. It never derives efficiencies from saved-event counts.
- Stream order: mu-stop pileup, electron flash, muon flash, neutral flash.
  The stop chain includes the stop prescale. Early streams require their own
  efficiencies and compensating prescales; the early reduction also applies to
  mu-stop pileup. Early and ordinary donor pools must not be interchanged.
- The stock production donor read mode is **sequential with wrapping**, preceded
  by one uniform skip in `[0, MaxEventsToSkip]`. `select_frames` supports that mode
  and explicit `randomReplace`; it does not silently replace sequential sampling
  with independent random donor choices. Carry the returned cursor across chunks.
- Ordinary mixing configures **no additional SimTimeOffset**. Source frames
  already contain pulse offsets. Do not reapply one, fold modulo the bunch period,
  or cut individual steps at the flash selection boundary. The flash filter
  selected whole events; it is not a payload-wide time cut. The mixer does not
  fold times; the detector response must, as stock digitizers do.
- Keep each donor occurrence's tracker and CRV arrays together, even if one or
  both are empty. Repeated donor uses have separate particle keys. If a caller
  supplies a calorimeter collection, `overlay` keeps it with the same donor too.

NumPy samples the same distributions, not the stock C++ random-number sequence.
Pair physics comparisons on the same primary-event IDs; exact donor and
response RNG replay requires the stock selection and seeds, not just equal seeds
passed to NumPy.

## Reading existing stock frames

`fcl/dump_frames.fcl` configures the existing `StrawGasStepTableDumper` and
`CrvStepTableDumper` on the same dts input. It requires an already-built runtime
containing both plugins, with `CrvStepTableDumper` supporting `staged_cat`.
There is no new C++ module or build step here. For an already-built older dumper
without association-policy support, `dump_frames_legacy.fcl` adds the stock
`EventLister`; pass its log via `read_event_log` and `read_frames(event_ids=...)`.
This keeps empty frames without inferring them from generator or step rows.

```bash
mu2e -c mixing/fcl/dump_frames.fcl -s DONORS.art -n -1 -T frames.root
```

`read_frames` reads the highest ROOT tree cycles from that output. The CRV census
is the event denominator, including empty frames; it is not inferred from the
union of nonempty step rows. The reader keeps full run/subrun/event identity and
joins particle keys only within their event. The selected compressed particle
collection is shared by the tracker and CRV input tags. The reader checks its
parent/generator/step references, not the original uncompressed ancestry.

## Python use

Run from the umbrella root (or add it to `PYTHONPATH`); dependencies are NumPy,
SciPy, Awkward and uproot, already included in the core analysis environment.

```python
import numpy as np
from mixing.frames import read_frames, overlay
from mixing.sampling import BunchLaw, STREAMS, draw_multiplicities, select_frames, stock_rates

rng = np.random.default_rng(1024)
signal = read_frames("signal.root")
pools = {name: read_frames(name + ".root") for name in STREAMS}
# Obtain efficiencies, prescale and bunch settings from the reference campaign.
rates = stock_rates(efficiencies, stop_prescale=stop_prescale)
intensity = BunchLaw.from_sdf(mean, sdf, cut_max).draw(
    rng, len(signal.event_ids), beam_primary=True)
counts = draw_multiplicities(rng, intensity, rates)
backgrounds, cursors = [], {}
for column, stream in enumerate(STREAMS):
    selection, cursors[stream] = select_frames(
        rng, counts[:, column], len(pools[stream].event_ids),
        cursor=None, max_events_to_skip=max_skip[stream])
    backgrounds.append((pools[stream], selection))
mixed = overlay(signal, backgrounds)
```

`overlay` remaps sparse particle keys, parents, generator indices and both
detectors' step pointers per donor occurrence. Original frame objects are not
modified. If explicitly configured, `time_offset_ns` is a per-primary-event
**additional** global-time shift for donors only, applied once to tracker time,
both CRV endpoint times, generator time and particle start/end global times.

## Saved-input checks

`python3 -m mixing.inspect frames.root --event-log frames.log --out summary.json`
reports counts, empty frames and time ranges; non-finite input times are counted,
not replaced. `mixing.check_overlay` tests assembly on those saved frames with an
explicitly synthetic Poisson mean of two frames. It checks row conservation and
particle references; it does not measure physical occupancy or production speed.

## Boundaries


- The dts reader extracts tracker and CRV only, not calorimeter steps. It is not a
  full-detector/trigger mixing implementation.
- Source compression, stopped-muon-daughter exclusions and detector-OR selection
  come from the stock donor files. The Python overlay does not redo those filters.
  Campaign-specific postmix `MixingFilter` selections must be applied by stock art;
  the Python merge alone only represents a universal/no-op postmix selection.
- The Python flat-table overlay does not write native art products. Native-art
  response uses the stock mixers instead. The existing tracker table producer
  does not import remapped donor GenParticles; feeding merged flat tables to it
  without resolving that mismatch is not a complete truth-preserving bridge.
- Before occupancy claims, use the same primary sample, geometry, conditions,
  bunch law and efficiency values as the reference; compare straw digis per plane
  and CRV pulses in **all sectors**, including empty events. Trigger-selected
  output cannot supply an unconditional occupancy denominator.
- `catalog.py` makes bounded read-only MetaCat queries. Cached dataset counts are
  not evidence of available files; empty membership is reported explicitly.

```bash
python3 -m pytest -q tests/test_mixing.py
python3 -m mixing.catalog DATASET_DID --limit 2 --out census.json
```
