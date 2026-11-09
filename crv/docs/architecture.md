# CRV: architecture

## The chain

The fast chain replaces Geant4, the stock CRV response and the stock CRV reconstruction of the extracted (KPP) cosmic
configuration. Every stage reads and writes flat tables keyed by `(run, subrun, event)`, so each stage can be swapped
for its stock counterpart and compared on the same events.

```
CRY primaries ── gen tables (Geant4 reference, default) | native CRY (opt-in) | device CRY (opt-in)
     │
     ▼
transport (tier 1a)    primary muon to and through the CRV bars: stage C = GDML world in height bands, sliced
     │                 energy loss, correlated multiple scattering, range-out near the end of the range
     ▼
tier 1b                what Geant4 adds: the muon's descendants (deltas above the cut, decay electrons) and the
     │                 non-muon CRY primaries with their showers, resampled from a Geant4 step library
     ▼
CrvStep table   ◄──── art: CrvStepsFromTable injects it into stock Offline (comparison arm A1)
     │
     ▼
response               photons (port of MakeCrvPhotons) -> SiPM charges (port of MakeCrvSiPMCharges)
     ▼
digitisation           waveforms, zero suppression, digis (port of CrvWaveformsGenerator, MakeCrvDigis)
     ▼
reconstruction         pulse fit (port of MakeCrvRecoPulses) -> coincidence clusters (port of CrvCoincidenceFinder)
     ▼
outputs                per-event clusters and pulses, stats JSON; CrvRecoTableDumper layout for the comparisons
```

`runchain` runs the whole chain in one process; `e2e-bench` runs it with timing, fingerprints and prepared-context
reuse; each stage also has its own subcommand (`transport`, `tier1b-apply`, the `reco-*` subcommands).

## Backends

Every stage has a NumPy implementation, which is the reference and the default. JAX twins exist for the response,
waveforms, reco (`--jax`), the transport (`--transport-backend jax`) and tier 1b (`--tier1b-backend jax`); they run
on CPU, NVIDIA GPUs and Intel PVC. The JAX twins are bit-identical per key to themselves on one device, not paired
draws with NumPy, and are validated statistically against it ([device.md](device.md)).

## Who owns what

| Area | Package or directory | Notes |
|---|---|---|
| physics primitives | core `mu2efastsim` (energy loss, materials, world, crossings, transport pieces, tables, statistics) | shared with the tracker and beam stages |
| CRV geometry, transport, tier 1b | `crvfastsim` top level (`geometry`, `transport`, `tier1b`, `tier1b_assets`, `runconfig`) and their `*_jax` twins | the tier-1b sampling and its asset sets are frozen except by a validated change |
| response | `crvfastsim/response/` | lookup tables, photons, SiPM, the stock oracles through PyROOT |
| reconstruction | `crvfastsim/reco/` | waveforms, pulse fit, coincidences, background model |
| validation | `crvfastsim/validation/` | V0-V4 checks and the stock-DQM histogram comparison core (old paths `crvfastsim.analyses.<name>`, `valhist.<name>` are aliases for one release) |
| studies and arms | `crvfastsim/analyses/`, `valhist/` | runchain, e2e-bench, census and diagnostics; the stock-DQM arms (fast tables, deck rule, step closure) |
| objective and veto | `crvfastsim/objective/`, `crvfastsim/veto/` | the CRV response objective; the C4 veto rule |
| art modules | `art/CrvFastSim/` | table dumpers and table-to-art injectors for the comparison arms (built in a muse satellite area) |
| Geant4 references | `grid/`, `art/dump_*.sh`, `geometry/` | jobs that made the reference samples and the GDML export |
| endpoints | kit tasks (`kit_task`, `kit_uva`) and core's endpoint adapters | one Mu2eBO point = one job on UVA, EAF, Polaris or Aurora |

## Configuration

A run configuration (`crvfastsim.runconfig`: `extracted_v04` default; named `run1a_v01`, `run1b_v01`, `run2_v10`)
selects geometry, conditions, lookup tables, the transport variant and the tier-1b asset set. The response and reco
constants come from `crvfastsim/configs/response_v1.json` ([response-configuration.md](response-configuration.md)).
Site paths come from the environment (core `mu2efastsim.site`); nothing is built in.

## Validation design

Each stage is checked in isolation against its stock counterpart on the same input (V0 geometry, V1 crossings, V3
tier-1b steps, V4 photons/SiPM, V-R1 waveforms, V-R2 pulses, V-R3 coincidences), then in combination (V2, V-R4, the
A1/E2E histogram comparisons) and on the objective the chain feeds (V-R5). See [validation.md](validation.md).
