# beam/: Geant4-free muon beam transport to the stopping target (prototype)

## Collaborator inputs

The umbrella [input inventory and setup](../setup/README.md) lists the MuBeamCat-derived
muon table, field map, geometry include tree and their publication gate.
No approved distributable beam input release is bundled. `beamfastsim.stage` can
fetch public map/geometry assets, but their hashes must be pinned alongside the
muon input. The streamed endpoint needs `FASTSIMKIT_FNAL_HOST` and reads `FASTSIMKIT_BEAM_DATA` (default
`$M2FS_INPUTS/beam_fastsim`) and `FASTSIMKIT_BEAM_CODE` (default `$FASTSIMKIT_SCRATCH/beam_code_kit`). The job
scripts take `M2FS_CODE`, `VENV` or `FASTSIMKIT_EAF_VENV` (a Python/JAX venv), `BEAMFS_DATA_ROOT` (the
`beam_fastsim` data area every closure output must be inside), `M2FS_DATA` and `PNFS_OUT` (grid staging) from your
site.env; each stops with a message naming a missing variable.

Package `beamfastsim` (imports `mu2efastsim` from `core/`; installable with `pip install ./beam`, metadata in
`beam/pyproject.toml`). Replaces the DS stage of the stock beam chain
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
| `beamfastsim/cechain.py` | conversion electrons from target stops or given starts to the tracker entry plane: helix steps in the field maps, foils / IPA / OPA crossed with the core `eslab` or `eslab_jax` slab physics; per-electron plane state in an npz |
| `beamfastsim/cesob.py` | sob stand-in of the stopping-target design studies: beam stops -> conversion electrons -> calorimeter-acceptance proxy (crystal map, disks' inner Al and carbon-fibre rings crossed with the slab physics) -> the studies' approximate sensitivity; the same estimate on a Geant4 EdepAna tree (`reduce`, `refsob`, bit-identical to the design studies), proxy checks against Geant4 (`calocheck`, `calodiag`), `hybrid` and `recalo`; fastsimkit task `beam_sob` |
| `beamfastsim/cereference.py`, `cecompare.py` | Geant4 CeEndpoint reference table from `ReadVirtualDetector` trees; fast vs Geant4 at the entry plane (filter-implying selection, energy MPV/FWHM/tail fractions) |
| `fcl/g4ref_mubeam_ntpart.fcl`, `art/run_g4ref.sh` | stock DS stage of SimJob MDC2025ac with a per-muon tree |
| `art/run_fast_cpu.sh` | g4table + field check + fast run + comparison on a GPVM core |
| `art/eaf_beam.sh` | the same plus a benchmark on an EAF A100 slice (`core/eaf/eafhub.py run`) |

```bash
# GPVM: G4 reference (stock DS stage, ~12 ms/muon), then the fast arm on the same muons
bash beam/art/run_g4ref.sh $M2FS_DATA/beam_fastsim/g4ref_full 214310 1
bash beam/art/run_fast_cpu.sh $M2FS_DATA/beam_fastsim/g4ref_full $M2FS_DATA/beam_fastsim/fast_v1_20k 20000
# conversion electrons: starts of a Geant4 reference table, or isotropic CE from a stop table
PYTHONPATH=core:beam python3 -m beamfastsim.cechain --offline-root $OFFLINE --maps DSMap.header TSdMap.header \
  --stops g4ref.npz --n 8000 --backend jax --out ce_fast.npz
PYTHONPATH=core:beam python3 -m beamfastsim.cecompare --ref ce_g4ref.npz --fast "ce_fast*.npz" --mode stops \
  --offline-root $OFFLINE --maps DSMap.header TSdMap.header --out compare.json
# sob stand-in for one target geometry: stops, CE, calorimeter proxy and the approximate sensitivity
PYTHONPATH=core:beam python3 -m beamfastsim.cesob point --g4 g4ref_run1baa_vd8.npz --z-up 3900 --geom overlay.txt \
  --electrons 100000 --procs 32 --out sob_point
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
- `hasTSdA`, only set to `false`: the TSdA absorber is then not built (the Mu2eBO foils studies run Geant4 without
  it); the default geometry keeps it

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

## Layer physics

Layers lose energy in slices with the local restricted dE/dx (fast Bethe law, Urban fluctuations) and scatter with
one Highland kick. The range-integrated alternative with Geant4 11.3.2 aluminium tables (`--physics g4`) and the
loss-model studies behind it (numerical convergence, finer slicing, frozen-entry loss histories, single-foil closure
against Geant4, slab and edge diagnostics) are kept on branch `park/beam-loss-studies`. Refining the layer loss model
did not improve agreement at the foil edges.
