# calo: calorimeter fast simulation (opt-in)

Detector-specific calorimeter code for the fast simulation. Nothing here changes a chain default or the `cesob`
calorimeter-acceptance proxy.

## Response (`calofastsim.response`, `calofastsim.propagation`, `calofastsim.hits`)

NumPy transcriptions of the stock Offline calorimeter response, from compressed `CaloShowerStep`s to readout
photoelectrons, and of the `CaloHitMaker` reco-digi grouping:

- `response`: Birks/G4 energy selection and longitudinal response (LRU) correction in float32, mean PE per SiPM,
  `std::lround` counts, the `fmod` bunch-time folding with the lower-time cut only, and the inverse-variance
  reduction of reco digis for a supplied group membership.
- `propagation`: the float32 sequential CDF of `CaloPhotonPropagation` built from the depth/time histogram, and the
  `lower_bound` lookup with caller-supplied uniforms.
- `hits`: the `CaloHitMaker` crystal/time grouping (at least two reco digis per crystal, split relative to the first
  member of each time group), vectorised over all events and crystals.

`response_jax` and `propagation_jax` are optional JAX twins; import them explicitly and enable x64 for the double
arithmetic. Conditions, geometry and tables are always caller-supplied: no calibration constant is embedded.

Sources: `Offline/CaloMC/src/CaloShowerROMaker_module.cc`, `CaloPhotonPropagation.cc`,
`Offline/CaloReco/src/CaloHitMaker_module.cc`.

Status: on the saved `CaloShowerStep`s of a 2,000-event CE sample, with Poisson statistics and travel time off in
both arms, the NumPy and JAX (eager and jit) primitives reproduce every stock readout count and photoelectron time
exactly (47,267 steps, 61,312 readouts, 4.43 M photoelectrons), and the stock propagation lookup matches on 6,390
prescribed draws.

## Readout and digitisation (`calofastsim.readout`, `calofastsim.pulse`, `calofastsim.digi`)

- `artsteps.steps_from_art`: reads `CaloShowerStep`s and event ids straight from a stock art file (uproot plus a
  NumPy decoder of art's memberwise `vector<CaloShowerStep>` blob) and applies the on-spill `EventWindowMarkerProducer`
  timing rule. `stockio.load_steps` accepts that art file or a table export (schema in its docstring).
- `readout.make_readouts`: `CaloShowerStep`s to per-SiPM photoelectrons (`CaloShowerROMaker`), all events at once,
  with photon statistics and travel time on or off.
- `pulse.PulseCache`: the `CaloPulseUtil` cache (TH1 interpolation to digiSampling/100, float32, unit maximum).
- `digi.digitize`: the `CaloDigiMaker` waveform, noise segments, pedestal, 12-bit ADC and peak/digi extraction,
  on the active readouts only, in NumPy or JAX (`backend="jax"`, x64 required), with the per-event DR-marker offset.
  Off spill, set `DigiConfig.event_length`. Noise comes from the stored noise histogram (`generate: false`, the
  production setting); the generated-noise cache is not implemented.

`fcl/stock_chain.fcl` runs the stock chain on saved steps in three arms and writes one EventNtuple per arm:
`det` (photon statistics, travel time and noise off), `noi` (the `det` readouts plus noise) and `prd` (the
production digitisation and reco configuration). Calo conditions and event timing are the Offline prolog defaults,
which production uses; the event-window marker is the production table with its per-job random phase fixed to one
draw. `scripts/stock_chain.sbatch` runs it with `container/offline/offline_uva.sh` and an existing runtime.
`scripts/compare_digis.py` compares fast and stock digis, reading the steps from the same art file.

Status (2,000 CE, stock Offline v13_39_00): with photon statistics, travel time and noise off, all 11,204 stock
digis are reproduced exactly (every ADC sample, t0 and peak position) by both backends. With noise on, the digi
count differs by -0.001 +- 0.014 per event and the length, peak, t0 and summed-ADC shapes agree (chi2/ndf 12/27,
23/51, 10/53, 9/39; a second stock seed gives 16/27, 13/51, 6/53, 12/39).

## Tests

```bash
PYTHONPATH=calo:core python3 -m pytest -q -p no:cacheprovider calo/tests
```

The JAX tests skip cleanly when JAX is not installed.
