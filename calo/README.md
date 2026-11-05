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

## Tests

```bash
PYTHONPATH=calo:core python3 -m pytest -q -p no:cacheprovider calo/tests
```

The JAX tests skip cleanly when JAX is not installed.
