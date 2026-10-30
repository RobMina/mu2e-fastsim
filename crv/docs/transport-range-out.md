# Stage-C transport: muon range-out (`range_out`)

From 2026-10-30 the stage-C transport of the **extracted** configuration (`extracted_v04`, `runconfig.range_out`)
follows the muon's kinetic energy near the end of its range; `--legacy-transport` restores the earlier transport and
the named configurations (`run1a_v01`, `run1b_v01`, `run2_v10`) keep it. `transport.transport(range_out=...)` is off
by default for direct callers. The opt-in JAX backend (`--transport-backend jax`) includes a device twin of these
rules; `--legacy-transport` retains its earlier kernel. New-default validation is distinct from the earlier
transport certificate and is described in [device-new-default.md](device-new-default.md).

## What changes

1. **Deltas above the cut leave the muon.** The bar loss is the restricted loss below the bar's delta cut (87.2 keV);
   the deltas above it are not transported, their deposits come from the tier-1b library. Without `range_out` the
   muon's kinetic energy also dropped only by the restricted loss, 1.65 MeV/cm at high momentum where Geant4's muon
   loses 2.42 MeV/cm in the same bars, so a muon reached its last bars with too much energy. With `range_out` the
   sampled delta energy (`eloss.sample_deltas`, per cut value) is taken from its kinetic energy as well; the bar's
   deposit is still the restricted loss, sampled at the (now lower) running momentum.
2. **Range-out inside a bar.** A bar's loss was sampled once at the entry momentum. When its mean loss exceeds
   `SUBSTEP_FRACTION` (10 %) of the kinetic energy, the bar is lost in `SUBSTEP_N` (10) sub-steps at the running
   momentum; a muon that stops inside deposits what it has left and its path ends there.
3. **Absorbers.** The aluminium between the layers used one mean dE/dx at the entry momentum over the whole path
   (median 11 mm): a 60 MeV/c muon (15.8 MeV, CSDA range 7.7 mm) kept 4.4 MeV and stopped at the top of the next bar
   with almost nothing. With `range_out` the mean loss follows the CSDA range of the full mean dE/dx,
   T(R(T) - L), with the sampled fluctuation scaled to it; where the mean loss exceeds `RANGE_NEAR` (30 %) of the
   kinetic energy the range is straggled instead (sigma_R / R = `RANGE_STRAGGLE`, 3.3 %, the Bohr value for protons
   scaled by sqrt(m_p / m_mu)), so few muons leave the sheet with a few MeV. The range table starts at 15 MeV/c with the
   residual range T0 / (2 dE/dx(T0)) and goes linearly to 0 at T = 0 (Bethe turns negative near 1 MeV/c; a table that
   starts there adds ~1e10 mm to every range).

Each part draws from its own stream spawned from the file's generator (bar sub-steps, deltas, absorber range, in that
order); without `range_out` no random draw changes and the output is the earlier transport's.

## Evidence (100 held-out g4ref_v2 shards 22-220, 5 M generated events; the primary muon's own crossings)

| | Geant4 | earlier transport | `range_out` |
|---|---|---|---|
| in-bar stops | 715 | 555 | 643 |
| entry kinetic energy of the stopping crossing, p10 / p50 / p90 [MeV] | 5.9 / 14.0 / 21.4 | 1.4 / 7.5 / 13.3 | 4.7 / 13.2 / 19.5 |
| path of the stopping crossing, p10 / p50 / p90 [mm] | 2.6 / 12.3 / 25.7 | 0.4 / 7.3 / 19.9 | 2.7 / 12.7 / 24.6 |
| no further crossing after leaving an inner EX layer at 60-70 MeV/c (layers 1 / 2 / 3) | 0.66 / 0.75 / 0.52 | 0.08 / 0.10 / 0.17 | 0.64 / 0.66 / 0.52 |
| chi2/ndf of log10 visible energy, crossings below 200 MeV/c | | 5.42 | 1.18 |
| chi2/ndf of the stopping crossing's entry momentum | | 16.66 | 2.01 |

Kinetic-energy loss per length of the exiting crossings agrees with Geant4 in every entry-momentum bin from 60 MeV/c
up (e.g. 60-80 MeV/c: 5.62 Geant4, 4.44 earlier, 5.61 MeV/cm; above 1 GeV/c: 2.42, 1.67, 2.38 MeV/cm). With the
default tier 1b on top, every step histogram of `valhist.placement_closure` stays at or below its earlier event-
clustered chi2/ndf; the per-crossing visible energy and the step yield are unchanged within errors.
