# beam/: Geant4-free muon beam transport to the stopping target (prototype)

Package `beamfastsim` (imports `mu2efastsim` from `core/`). Replaces the DS stage of the stock beam chain
(`Production/JobConfig/pileup/MuBeamResampler.fcl`) for the target-stop question: muons from the DS entrance (as
written by the POT stage, `MuBeamCat`) are carried through the field map, the stopping-target foils and the IPA in
batched JAX lanes (CPU or GPU) to their stop, decay or exit.

No Mu2e-internal data or results are stored here; the field maps, geometry and reference samples are read at FNAL.

| Path | What |
|---|---|
| `beamfastsim/fieldmap.py` | G4BL `.header/.bin` maps; trilinear lookup as Offline `BFGridMap` |
| `beamfastsim/simpleconfig.py`, `geometry.py` | Offline geometry text files -> foils, TSdA, IPA, outer PA |
| `beamfastsim/transport.py` | RK4 stepping, layer energy loss (G4 Urban via `mu2efastsim.eloss_jax`) and Highland scattering, range-out, decay, lane compaction |
| `beamfastsim/g4ref.py`, `compare.py` | per-muon G4 reference table and fast-vs-G4 metrics/plots |
| `beamfastsim/bench.py` | throughput on synthetic muons (no inputs needed) |
| `fcl/g4ref_mubeam_ntpart.fcl`, `art/run_g4ref.sh` | stock DS stage of SimJob MDC2025ac with a per-muon tree |
| `art/run_fast_cpu.sh` | g4table + field check + fast run + comparison on a GPVM core |
| `art/eaf_beam.sh` | the same plus a benchmark on an EAF A100 slice (`core/eaf/eafhub.py run`) |

```bash
# GPVM: G4 reference (stock DS stage, ~12 ms/muon), then the fast arm on the same muons
bash beam/art/run_g4ref.sh $M2FS_DATA/beam_fastsim/g4ref_full 214310 1
bash beam/art/run_fast_cpu.sh $M2FS_DATA/beam_fastsim/g4ref_full $M2FS_DATA/beam_fastsim/fast_v1_20k 20000
# laptop/any: unit tests and benchmark
PYTHONPATH=core:beam python3 -m pytest -q beam/tests
PYTHONPATH=core:beam python3 -m beamfastsim.bench --n 2000
```

## Persisted reference audit

Audit existing outputs before reusing a partial reference; this does not run transport or require JAX/CVMFS:

```bash
PYTHONPATH=core:beam python3 -m beamfastsim reference-audit \
  --nts "$REF/nts.owner.beamfs_g4ref.version.sequencer.root" --g4 "$REF/g4ref.npz" \
  --artifact "$REF/job.fcl" --artifact "$REF/job_config.txt" \
  --artifact "$REF/job.log" --artifact "$REF/mubeamcat.txt" \
  --out "$REF/reference_audit_UNIQUE_TAG.json"
```

The audit selects the highest `mutree/ntpart` cycle, records every available cycle's entry count, counts distinct
persisted event/primary IDs, hashes the ROOT/table/configuration artifacts, and compares all raw primary columns to
the cached table in row order. Duplicate primary IDs or a table mismatch make the CLI exit nonzero.
Add `--geom PATH --offline-root ROOT` to recompute `q`, `at_rest` and `foil` from ROOT rows, compare them to the
cache, and record the effective geometry plus SHA-256/size/path of every file traversed by the beam geometry reader.
`--baseline-geom Offline/Mu2eG4/geom/geom_common.txt` additionally checks the parsed overlay contract below.
These are current-reader checks, not historical G4 binary/geometry certification. Empty-muon events and source art
contents are not certified. Review the saved job outcome separately; particle rows are not event counts.

`transport_all` enforces an exact positive-integer step cap, including the final partial chunk. Foil crossings own
the arrival endpoint and exclude departure in either direction. `make_stepper` rejects nonfinite/malformed foil
arrays, invalid radii/thicknesses, overlapping or unordered foils, and centre spacing smaller than `ds`. Treat the
geometry arrays as immutable after compilation. This is a stepping precondition, not a full geometry contract:
the inherited material, support and absorber approximations are not certified by this check.

`run` and therefore `point` reject effective parsed changes outside these target-overlay keys, relative to
`Offline/Mu2eG4/geom/geom_common.txt` in the selected Offline root:

- `stoppingTarget.{radii,halfThicknesses,zVars,deltaZ,z0InMu2e,holeRadius,holeRadii}`
- `protonabsorber.distFromTargetEnd`

This includes changes through nested includes, not just the top-level file. A scalar hole change shadowed by a
per-foil `holeRadii` vector is rejected. The baseline's unmodelled supports/absorbers remain approximations; parsed
material, tilt, field-axis and unrelated detector changes are rejected rather than silently scored. This guard
covers effective **parsed** keys only: the minimal SimpleConfig reader is not a general syntax validator, and a
user-selected Offline root is not an immutable approved baseline. Full G4 support for per-foil holes is release
dependent. Allowed target changes still need paired physics validation; they are not a certified BO search space.

For a foil stop, time and proper time use the crossing fraction of the pre-step flight, consistent with the saved
thin-plane stop position. No within-foil slowing time is integrated; surviving crossings, IPA stops, boundary/decay
ordering, grazing paths and the low-energy model remain approximations requiring physics convergence checks.
A shared seed is not layout-invariant: chunking and compaction affect random draws.

## Bounded numerical convergence pilot

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.convergence \
  --g4 "$REF/g4ref.npz" --out "$DATA/convergence_UNIQUE_TAG" --n 2000 --seeds 11 22
```

This runs four arms sequentially for each saved seed: nominal (5 mm, 3000 steps, 4 loss slices), half-step
(2.5 mm, 6000 steps: equal maximum path), doubled cap (5 mm, 6000 steps), and 8 slices. `--geom` selects an overlay;
`--arms` can restrict the list. Put `nominal` first for paired arm differences. `run --max-steps` also exposes the
numerical cap directly; it is not a BO design variable.

`--start ROW --n COUNT` selects a contiguous whole-event block. Requests outside the table or splitting an event
(including noncontiguous repeated IDs) fail before creating output. The selected input is saved as `input_block.npz`;
the manifest records its hash, row range, event count and endpoint IDs. `nominal` must be the first arm. Disjoint
blocks in the same table are not automatically independent upstream source samples, and equal event IDs across
different G4 resampling jobs do not establish matched primaries.

The output directory must be new. Each arm retains an NPZ; `summary.json` counts actual saved muons, fates and foil
profiles and reports event-cluster standard errors for paired stop-fraction differences and flight-time shifts
conditional on stopping in both arms. `manifest.json` hashes the input, current beam/core physics code, field files
and geometry include closure. The fixed input prefix is reused; different seeds are response replicas, not new
primaries. Same seeds across layouts are not invariant common-random-number streams. No acceptance threshold or
independent-reference uncertainty is implied. Avoid interpreting conditional timing shifts as G4 stop-time closure.

`--budget-seconds` (default 1200) prevents starting further arms after the budget, but does not interrupt a running
arm; an incomplete pilot lists unstarted arms. On a GPVM use one CPU affinity and the one-job-per-host rule. All
passes include their compilation overhead and `repeat=0`: the legacy `muons_per_s_warm` metadata label from `run`
is **not** a warm-throughput measurement here. Never extrapolate larger-job runtime from this pilot.

## Finer slicing on a fixed population

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.slicestudy \
  --baseline "$DATA/convergence_UNIQUE_TAG" --out "$DATA/slices_NEW_TAG" --seeds 11 22
```

Requires a whole-event pilot with `input_block.npz`; pass the same `--geom` for moved targets. Reuses the saved
4/8-slice arms and runs only 16/32 slices. Before any run, verifies the selected-input hash, beam/core physics
source hashes, geometry include hashes, field hashes, and saved-arm parameters/lengths against the pilot manifest
and summary. This is not a full binary/environment identity check; reused output hashes are recorded at extension
time, not retroactive proof of their original write. The new output directory must not exist.

For each seed, all timing differences use the same intersection of tracks stopping in all four arms. Subsets keep
those stopping in the same foil across all arms separate from those changing foil in any arm. Saved row indices
allow exact replay. Reports include event-cluster errors for adjacent-slice and 32-minus-4 flight-time, foil-index
and z differences. These outcome-conditioned populations are **diagnostics**, not production response/correction
models; the common intersection changes if more arms are added. Seed replicas are correlated through their shared
inputs. No G4 time closure, acceptance threshold or default-slice change follows from the report.

Status and validation: wiki `mu2e/wiki/sim/beam-muon-transport-fastsim.md`.
