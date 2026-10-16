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
