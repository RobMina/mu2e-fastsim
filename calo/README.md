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

## Reconstruction (`calofastsim.templatefit`, `calofastsim.cluster`, `calofastsim.chain`)

- `templatefit.fit_digis`: the `CaloTemplateWFProcessor` template fit in JAX (baseline, primary and sequential
  secondary peak seeding, the source objective `sum (y-f)^2 / b` with parameter limits, component selection and
  refit), vmapped over digis. Damped Newton steps replace TMinuit, so values agree to tolerance. The second pass for
  `chi2/ndf > 1` digis is not implemented (about 0.1 % of digis; flagged `refit_skipped`).
- `hits.make_hits` (above, `backend="jax"` for the grouping rounds) builds `CaloHit`s; reco-digi energies use the nominal `CalCalib` (0.0625 MeV/ADC, 0.00375
  for the CAPHRI LYSO crystals).
- `cluster.make_clusters`: `CaloProtoClusterMaker` + `ClusterFinder` + `CaloClusterMaker` + `ClusterAssociator`
  (main and split clusters, split association, main merging, energy-weighted position in the disk front-face
  frame), vectorised over events.
- `chain.run_chain`: steps -> readouts -> digis -> reco digis -> hits -> clusters for all events.

The stock template fit divides its chi2 by the fitted baseline, so with noise off the objective is zero and stock
reco is degenerate; reco comparisons therefore use the noise-on arms.

Status (2,000 CE, production settings, against stock Offline v13_39_00 on the same steps):
- stage by stage on stock inputs: template fit on the stock digis matches the stock reco digis (energy within 1 %
  for all 9,598 paired, time within 0.1 ns for 99.96 %); hits from the stock reco digis are bit-exact (4,596);
  clusters from the stock hits are identical (837: energy, time, size, split flag; position within 3e-5 mm);
- full fast chain from the steps, paired by event against a second stock seed as reference: leading-cluster
  energy difference 0.03 +- 0.07 MeV with RMS 2.10 MeV (stock vs stock 2.15), time RMS 0.719 ns (0.709, dominated by
  the production 0.5 ns reco-PBT smearing), position RMS 2.3 mm (2.4); CE peak 87.6 vs 87.3 MeV (+0.31 +- 0.34), core width 4.25 vs 3.90 MeV (+0.35 +- 0.42), low-tail
  fraction 30.5 % vs 30.6 % (0.0 +- 0.8 %).

`scripts/compare_reco.py` runs the stage-by-stage comparison, `scripts/compare_chain.py` the full chain (figures and
`numbers.json`), `scripts/time_chain.py` the throughput.

## Tests

```bash
PYTHONPATH=calo:core python3 -m pytest -q -p no:cacheprovider calo/tests
```

The JAX tests skip cleanly when JAX is not installed.
