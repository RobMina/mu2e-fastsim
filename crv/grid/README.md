# G4 reference sample (plan section 5)

Stock `ExtractedCRY.fcl` (run 1400) from the CVMFS Musing `SimJob MDC2025ay`: no tarball. Each shard writes the
accepted events (`dts.*`: CrvSteps, SimParticles, GenParticles, ...) and every generated event's CRY primaries and
live time (`gen.*`: GenParticles, CosmicLivetime, GenEventCount), so the fast sim can rerun the same primaries.

| file | role |
|---|---|
| `fcl/ref_sim_extracted.fcl` | stock chain + the `GenOutput` stream |
| `grid_ref_sim.sh` | worker: shard k = subrun k+1, baseSeed 20261016+k; skips if both outputs exist |
| `submit_ref_sim.sh [--test\|--cluster -N n]` | stages the fcl, submits (lifetime 4 h, 3 GB memory, 4 GB disk) |

## g4ref_v1 (2026-10-16)

| item | value |
|---|---|
| output | `$CRVFS_PNFS/g4ref_v1/` (`dts`, `gen`, `logs/`) |
| test job | shard 0: exit 0; 50,000 generated, 2,413 accepted (4.83 %); wall 35:22, VmHWM 2.0 GB; dts 17.4 MB, gen 8.0 MB |
| full cluster | `-N 20`, shards 0-19 (shard 0 skipped, done in the test): **all 20 complete** 2026-10-16 ~09:10 |
| total | **1,000,000 generated, 50,008 accepted (5.00 %)**, per shard 2,413-2,603; 393 MB of dts+gen |
| live time | **15.587 s** total (CosmicLivetime from CRY `timeSimulated`), 0.774-0.790 s per 50 k shard: **15.59 us of cosmic exposure per generated event**, 312 us per accepted event |
| grid cost | wall 30-64 min per 50 k shard (node-dependent), VmHWM 2.01-2.09 GB |

The test job logs 108 `Mu2eG4TrackingAction::swapTrajectory: SimParticle is not found` warnings from one event with a
very large shower (SimParticleCollection size limit; event 8204). Present in the stock chain too; the event keeps its
steps but not every MCTrajectory.

## crygen_v1: CRY-only gen tables (2026-10-22)

The 20 `g4ref_v1` shards (1 M generated) are too few for production `chain_point` points (fastsimkit), and production
`dts.mu2e.CosmicCRYExtracted.MDC2025as` keeps only accepted events (no gen stream). The fast sim needs every generated
event's CRY primary, so `crygen_v1` reruns the generator alone: same `ExtractedCRY.fcl`, run 1400, but only
`generate`, `genCounter` and `cosmicTimeOffset`; `g4run` stays on a path behind a filter that rejects every event,
because `CosmicCRY` needs the `WorldG4` geometry that `g4run` adds at beginRun (without it: "Failed to retrieve detector
element of type WorldG4"). Output: the `gen` stream only (GenParticles, CosmicLivetime, GenEventCount), dumped later
with `art/CrvFastSim/fcl/dump_gen.fcl`.

| file | role |
|---|---|
| `fcl/gen_only_extracted.fcl` | generator-only configuration |
| `grid_crygen.sh` | worker: shard k = subrun 2001+k, baseSeed 20263000+k (subruns 1001-1020 belong to the tracker's gen_as_v1); skips if the output exists |
| `submit_crygen.sh [--test\|--cluster -N n]` | stages the fcl, submits (lifetime 3 h, 2.5 GB memory, 2 GB disk); default 100 x 1 M |

GPVM check (20,000 events, mu2egpvm03): 42 s CPU (2.1 ms per generated event), VmHWM 1.9 GB (Geant4 initialisation),
1.7 MB of art; dumped table has the g4ref_v1 columns, 20,105 primaries for 20,000 events, live time 0.308 s
(15.4 us per event, g4ref_v1 15.59 us).

Test job `30476047.0@jobsub04` (2026-10-22, before the subrun change): 1,000,000 generated, 15.565 s live, CPU 1,856 s,
wall 39:43, VmHWM 1.93 GB. It used subrun 1001, which the tracker's `gen_as_v1` already uses (different seed, so a
different sample, but the same run/subrun/event keys): its output was renamed `...crygen_test_subrun1001...` and is not
part of `crygen_v1`. Record every new sample (name, subruns, seeds) before submitting, so run/subrun/event keys never
collide.
