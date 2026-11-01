# mu2e-crv-fastsim

Geant4-free simulation and fast reconstruction of the Mu2e **Cosmic Ray Veto** for the extracted (KPP, no field)
cosmic configuration: fast CrvSteps (muon transport plus a Geant4 secondary library), the CRV response (photons,
SiPM charges), waveforms and digis, pulse and coincidence reconstruction, in NumPy with JAX twins for GPUs. It feeds
CRV MC/data and design studies (Mu2eBO) without art or Geant4, and reproduces the stock Offline chain stage by stage.

Stage 1 of the umbrella [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim) (submodule `crv/`), which pins
compatible commits; detector-independent physics comes from [`mu2e-fastsim-core`](https://github.com/RobMina/mu2e-fastsim-core)
(package `mu2efastsim`). Work from a clone of the umbrella.

This repository holds code only: no Mu2e-internal conditions, tables, data or results.

## Status

| Part | Default | State |
|---|---|---|
| transport (tier 1a) | stage C with `range_out`, extracted_v04 | V0, V1 pass; device twin opt-in, validated |
| tier 1b (secondaries) | asset set `strict_v4` | step yields within errors; two step-shape discrepancies open |
| response (photons, SiPM) | NumPy; JAX with `--jax` | V4 pass on CPU and GPU (A100, PVC, B200, T4) |
| waveforms, pulses, coincidences | NumPy; JAX with `--jax` | V-R1..V-R5 pass; fast response + reco on fast steps 0/133 histograms flagged vs stock |
| whole chain vs Geant4 + stock | | 5 M held-out events: 21/155 histograms flagged (steps), 13/133 end to end; counts within 0.25 % |
| named configurations (Run 1A, Run 2) | | Run 1A V1 pass, Run 2 V1 fail; no tier-1b library: cannot run end to end |
| device precision | float32 in the device kernels (float64 one variable away) | within the float64 seed spread on 100 shards; pulse PE offset bounded at <= 0.3 % (`docs/device-precision.md`) |

Numbers, samples, thresholds and commits: [`docs/validation.md`](docs/validation.md). What is missing or not
validated: [`docs/limitations.md`](docs/limitations.md).

Throughput of the default chain (cached CRY primaries to clusters, warm, events with CRV steps per second): one CPU
core 55, one A100 MIG slice 163, one Polaris node 990, one Aurora node 1,212, 8 A100 at UVA 1,695; stock Offline on
one grid job 0.92.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
cp site.env.example site.env        # fill M2FS_INPUTS (verified input release), M2FS_DATA (outputs); source it
source site.env
python3 -m pytest -q -p no:cacheprovider crv/tests          # JAX tests skip without JAX
G="$CRVFS_DATA/g4ref_v2_tables/gen.001400_00000022.root"; GEOM="$CRVFS_DATA/geometry"
python3 -m crvfastsim runchain "$G" --tier1b-dir "$CRVFS_DATA/tier1b_strict_v2" \
  --gdml "$GEOM/mu2e_extracted_v04.gdml" --calib "$GEOM/crvcalib_sim_best_v1_5_run1400.csv" \
  --deviation "$GEOM/crvphoton_sim_best_v1_5_run1400.csv" --max-events 1000 --seed 22 --outdir "$M2FS_DATA/crv-smoke"
```

Details, GPU options and the options that change the physics: [`docs/running.md`](docs/running.md). The inputs are
Mu2e-internal and not yet published as a collaborator release (umbrella `docs/running.md`).

## Map

| Path | Content |
|---|---|
| `crvfastsim/` | the package and its CLI (`python3 -m crvfastsim <subcommand> -h`); reference: [`crvfastsim/README.md`](crvfastsim/README.md) |
| `crvfastsim/response/`, `crvfastsim/reco/` | response and reconstruction stages, NumPy and JAX |
| `crvfastsim/validation/` | V0-V4 checks (`v0check` ... `v4sipm`, `transport_validation`) and the stock-DQM comparison core (compare, coverage, pilot, paired rates and shapes, event shapes, reco detail); `crvfastsim.analyses.<name>` and `valhist.<name>` are aliases for one release |
| `valhist/` | the stock-DQM arms and studies (fast tables, deck rule, step closure, device harnesses) |
| `art/` | muse satellite area: table dumpers and table-to-art injectors for the comparison arms |
| `grid/`, `geometry/`, `profile/` | jobs that made the Geant4 references, the GDML export and the baseline profile |
| `scripts/` | drivers of the validation campaigns (GPVM, UVA, grid) |
| `tests/` | pytest suite (synthetic inputs; geometry tests need the public Offline counters file) |

Docs: [`architecture.md`](docs/architecture.md) (stages, backends, ownership), [`running.md`](docs/running.md),
[`validation.md`](docs/validation.md), [`limitations.md`](docs/limitations.md), [`device.md`](docs/device.md) (JAX
transport and tier 1b), [`tier1b-default.md`](docs/tier1b-default.md), [`transport-range-out.md`](docs/transport-range-out.md),
[`fastreco.md`](docs/fastreco.md) (reco goals, V-R thresholds and results),
[`response-configuration.md`](docs/response-configuration.md), [`e2e-benchmark.md`](docs/e2e-benchmark.md); opt-in
features have their own page in `docs/`. [`docs/plan.md`](docs/plan.md) is the original 2026-10-15 plan (historical;
its goals and thresholds are still the agreed ones).

## Tests

`python3 -m pytest -q -p no:cacheprovider tests` from this directory with core and crv on `PYTHONPATH`. Large-sample
validations are `@pytest.mark.slow` (run with `--runslow`); `python3 -m mu2efastsim.testselect --also ../core` selects
the test files a change affects. CI runs the suite with and without JAX on every pull request (`CONTRIBUTING.md` in
the umbrella).

## Device precision

The device (JAX) chain (transport, tier 1b, photons, SiPM, waveforms, pulses, coincidences) runs in float32 by
default; `MU2EFS_DEVICE_FLOAT=float64` restores float64 (identical to the float64 output of earlier releases). On
5 M generated cosmics, float32 is within the float64 seed-to-seed spread on every validated quantity except the
mean pulse PE (see [device precision](docs/device-precision.md)). It runs 1.2-2.8x faster on cards with reduced FP64 rate
(MIG slices, A40, RTX PRO 6000), with no gain on A100, B200 and PVC. float64 output is identical on every GPU model
for a given seed. float32 output repeats on one model but its discrete results can differ slightly between models,
so `runchain` and `e2e-bench` record the precision and device model (`precision` in their stats, `--emit` shards
and merge). NumPy (CPU) paths are float64 and unchanged. The UVA kit (`kit_uva.py`) passes `FASTSIMKIT_DEVICE_FLOAT`
(default float32) to its jobs and shares a JAX compilation cache (`FASTSIMKIT_JAX_CACHE`).

## Studies kept off main

Finished or paused studies live on branches, each the full tree at the point it left main:

| Branch | Contents |
|---|---|
| `park/veto-closure` | the full veto-inefficiency closure (frame replay, CE normalisation, bounded response, refit and membership audits, deadtime frame production); main keeps the C4 rule evaluation (`crvfastsim.veto`, `veto.deadtime`, `veto.cosmic_ana`) |
| `park/tier1b-library` | tier-1b library studies: generator association, shower sampling and uncertainty, denominator audits, library ensembles, device-chain certificates and reports |
| `park/response-diagnosis` | joint response fits, response diagnosis and explicit-list validation |
| `park/named-run2` | named-run reference campaigns, production reuse/scope, Run 2 grid planning, scatter convergence and V1 strata |
| `park/pvc-reco` | PVC (Aurora) reco profiling and parity probes |
| `park/cpu-speed` | speed switches measured and not promoted: the pulse-fit Jacobian workspace (no gain over the normal-equation cache) and compact reco dispatch (faster, output not identical) |

## License

Apache-2.0 (`LICENSE`, `NOTICE`).
