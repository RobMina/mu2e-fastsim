# Primary-brems photons at the emitting foil exit

This opt-in calculation predicts the **uncollided primary-brems component** at
its first exit from its birth foil. It is not the complete photon flux out of the
whole stopping target and is not yet a prediction of tracker deposited energy.
It does not change the CE or beam defaults.

## Model

1. `flash_target_jax.target_source` carries DS-entrance electrons and positrons
   through the magnetic field and target/absorber layers using the core slab
   physics, with charge-specific tables and helices. `flash_navigator_jax` is a
   separate execution loop using target-geometry helpers shared with CE; it
   does not change CE physics or scheduling defaults. See
   [shared-transport.md](shared-transport.md). Slab steps with no active lanes are skipped.
2. `lane_step_photons` records brems energy and direction before parent recoil,
   along with the emission position and path. Births outside the finite foil
   annuli are counted separately, not reassigned to a foil.
3. `flash_exit.foil_exit` follows each photon on a straight ray to the first
   emitting-foil face, outer rim or hole. The transparent result has unit weight.
4. The uncollided weight is `exp(-mu_rho(E) * rho * path_mm / 10)`, using the
   untuned aluminium [NIST XCOM table](flash-attenuation.md). Every interaction
   removes a photon from this component; scattered/secondary photons are absent.

The charged-particle navigator still uses planar infinite transverse layers
once a foil is entered, local flat shells and frozen layer fields. A finite
photon exit does **not** repair this charged-particle geometry approximation.
No delta-ray cascades, annihilation photons, scattering redistribution, subsequent
foil interactions, downstream absorber transmission, or tracker response is
included. Omitted charged-secondary/annihilation counters are retained.

## Run and compare

```bash
PYTHONPATH=core:beam JAX_ENABLE_X64=1 python3 -m beamfastsim.flash_flux \
  --states source_states.npz --source-events 1000 \
  --geom overlay.txt --offline-root /path/to/geometry \
  --maps DSMap.header TSdMap.header --capacity 256 --max-steps 500000 \
  --h 1 --unroll --out flux_low
PYTHONPATH=core:beam python3 -m beamfastsim.flash_flux_compare \
  --low flux_low --high flux_high --source-events 1000 --out compare
```

Use the same ordered source states and seed for design arms. Outputs include
births, first foil-exit states, event totals, raw energy histograms, per-shard
sums, paired ratios/errors and figures. `--births saved_births.npz` performs only
photon postprocessing, without rerunning charged transport. A saved full-EM
birth file is supported: only mechanism 1 (brems) enters this calculation.

Energy bins below the attenuation table or across its narrow K-edge gap are
reported as unsupported, with their photon count and energy; no extrapolation
or transparent substitute enters the uncollided result. The transparent totals
include them. Buffer overflow and iteration-cap lanes are reported; a run with
either exits nonzero. Path-limited loopers are counted separately and still
contribute their recorded partial emission. Comparisons with loopers are partial
source diagnostics, not completed transport.

## Time and normalization

Source times are retained in host float64; device times are elapsed times from
the start, preventing large absolute times from erasing small flight increments.
Adding them back does not restore phase precision lost in the stored source.
The stock strict `fmod(t,T)>250 ns OR fmod(t,T)<10 ns` predicate is provided for
**detector-step selection**. Applying it at a photon exit is labelled diagnostic
only. No source is rejected for being late, and no bunch offset is added here.

`--source-events` includes the whole extracted input-event prefix, including
zero-hit and zero-photon events. It is neither generated POT nor a selected
tracker-event denominator. The compared arm event arrays must agree exactly.
Paired errors are calculated on event contributions; a separate cluster error
uses sums by run/subrun. A few subruns give a fragile cluster error. Shape pulls
include the fitted count-ratio covariance but are diagnostics, not independent
bin significance tests.

The board detector-deposit ratio is only a contextual reference: a DS source
sample and TS3 board primaries are not paired, and a deposit ratio need not equal
a photon-count or photon-energy ratio. No normalization is fitted to that value.

## Performance diagnostics

`python3 -m beamfastsim.flash_profile` accepts the source, geometry, maps and
transport settings above, plus `--reference flux_high/births.npz`. It separately
measures host table construction, JAX lowering/compilation, synchronized device
execution in fixed-shape chunks, and final device-to-host copying. Chunk reports
include live lanes, vacuum/slab lane steps and photon counts. Chunking retains
the species key, lane order/shape and shared slab RNG iteration; it does not
restart the random stream. Every available saved state and photon array is
compared at the end. These are diagnostic timings, not a matched stock speedup.

The source uses one fixed batch per charge. Finished lanes remain in those
arrays, and a small number of low-energy vacuum trajectories can keep the whole
batch active for hundreds of thousands of iterations. Scalar guards around
empty crossing solves and empty layer-entry frame construction can be tested
with `--skip-empty`; the normal source default remains unchanged. This option
does not alter stepping limits, cuts, source times, lane ordering or RNG calls.

A 4,000-input-event, 4,541-parent A100-80GB diagnostic gave the following single
measurement per variant, with the same float64 source, seed, 1 mm maximum step,
0.05 rad turn limit, 30 m path limit and 256 photon slots:

| Component | Baseline | Empty-work guards |
|---|---:|---:|
| Table construction | 21.64 s | 23.58 s |
| Lowering and compilation | 70.57 s | 70.21 s |
| Synchronized execution | 1279.39 s | 544.08 s |
| Cold instrumented source | 1375.13 s | 641.86 s |
| Cold input events/s | 2.91 | 6.23 |

All 27 saved charged-state, photon and completeness-flag arrays agreed exactly
with the reference output in both variants. Final host copying took less than
0.06 s. Electrons and positrons required 546,091 and 607,775 iterations;
80.4% of baseline execution time occurred after iteration 100,000 of each
species, by which point only 65/4,007 electrons and 16/534 positrons remained.
Useful vacuum-plus-slab lane-step occupancy was 2.83% and 3.72% over the whole
run. These are scheduling counters, not measured GPU hardware utilization.
The guarded version is still dominated by the long fixed-batch tails.

## Hybrid sparse-vacuum scheduling (opt-in)

Add `--cpu-vacuum 128 --cpu-adaptive` to `flash_flux`, or set those keyword
arguments on `target_source`. Material interactions remain on the original
device with their original species batch, gathered ordering, key and RNG
iteration. Only an empty-material vacuum population of at most 128 survivors
is compacted to CPU. A compiled CPU loop takes the same individual steps and
stops **before committing any surface-candidate iteration**. Hot state returns
to its original rows, the global RNG clock advances by the number of committed
vacuum steps, and the original device step handles the candidate/material work.
CPU buckets shrink through 128, 64, 16, 4 and 1 lanes. No photon buffers or slab
random-number generation move into the compact loop.

This hybrid is not an all-GPU transport. On one A100-80GB with four allocated
CPU cores, the same 4,000-event input gave:

| Point | Cold profiled source | Input events/s | Execution excluding setup/compile |
|---|---:|---:|---:|
| bpzup21 | 243.02 s | 16.46 | 134.78 s |
| bpzup12 | 254.51 s | 15.72 | 150.02 s |

The high-point cold rate is 5.40 times the instrumented original and 2.52 times
the empty-work guards; execution is 8.53 times the original. The public
`flash_flux` CLI subsequently measured 235.98/257.93 s cold for low/high,
including source setup/compilation and host materialization, or **16.95/15.51
input events/s** (5.79/5.35 times the original low/high rates). This timer
excludes process startup, initial map loading and photon postprocessing, as in
the original measurement. These are individual measurements, not a stock-G4
speedup or an extrapolation to larger samples.

Recomputed emitting-foil exits and attenuation retain 965/3,356 transparent
photons and identical transparent spectrum bins. The largest uncollided-total
difference is 1.59e-10 photons; the largest energy-total difference is
1.14e-13 MeV. The paired high/low central ratios remain 3.40923237117 for
uncollided counts and 3.25663370265 for energy.

For both points, all saved discrete state/crossing/step counters, fates, photon
validity masks and completeness flags match the original. Floating-point
arrays are **not bitwise identical** across CPU/GPU arithmetic and compiler
fusion. Observed maximum differences across these two samples are 3.61e-5 mm
for final charged position, 4.45e-6 for a final direction component, 5.62e-9 mm
for photon birth position, and 4.59e-14 MeV
for photon energy. This does not guarantee unchanged boundary decisions on
all future inputs. Keep the default device path for reproducibility checks.
The zero/5 path-limited loopers and all existing physics limitations remain.

Full material compaction remains a separate change: random samples depend on
gathered slot, array shape and the shared iteration. Copying the CE refill/shrink
driver would not preserve the existing per-parent realization. Do not present
early stopping or a larger vacuum step as a scheduling-only fix.
