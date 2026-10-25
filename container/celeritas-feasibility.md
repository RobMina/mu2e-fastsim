# Celeritas 0.6.3: load probe and flash integration gates

Status (2026-10-25): image packaging confirmed; GPU load/transport untested. No new allocation, build or physics
run was started. At 16:53 EDT, `squeue -u ram2aq` and `squeue -A ldmxuva` returned no jobs. This is not a lack of
project allocation: `allocations -a ldmxuva` confirms the active award through 2027-06-19, with 9,998,969 SU
remaining at this check. Image metadata was read from login host `udc-ba37-32c1`, with no Slurm job/step set;
shared `/scratch` was accessible there without an active compute job. The existing image is
`/scratch/ram2aq/mu2efastsim/mu2e-celeritas.sif`, 7,059,009,536 bytes, mtime `2026-10-25 00:00:03 -0400`.
The saved `celeritas_build.log` records `celeritas@0.6.3+covfie+cuda+geant4+hepmc3~root+shared+vecgeom`,
Geant4 11.3.2, VecGeom 1.2.11, CUDA 12.6.3/sm80 and HepMC3 3.3.1. It is not the JAX image.

## First gate: load only, inside an existing A100 allocation

`probe_celeritas.sh IMAGE NEW_OUTPUT_DIR` refuses to run without a Slurm job/step, a visible A100, a readable image,
and Apptainer. It refuses an existing output directory and preserves loader exit codes/stderr. It runs only
`celer-g4 --help`, `--version`, and `--dump-default` using `--cleanenv --nv`, never a transport input or a build.
CLI success does **not** establish GPU kernel execution or Mu2e compatibility. Image size/mtime are recorded, not
an immutable digest; hash the image before a physics campaign.

Once the script is available on UVA, from a shell in the chosen existing A100 allocation:

```bash
module load apptainer
srun --ntasks=1 bash /path/to/probe_celeritas.sh \
  /scratch/ram2aq/mu2efastsim/mu2e-celeritas.sif \
  /scratch/ram2aq/mu2efastsim/celeritas_probe_UNIQUE_TAG
```

This is not a submission command. Verify the allocation is free of other work first. The script has not yet been
deployed to UVA. Laptop guard checks (no container or GPU involved):

```bash
bash -n container/probe_celeritas.sh
PYTHONDONTWRITEBYTECODE=1 ~/mu2e/.venv/bin/python3 -m unittest discover -s container/tests -v
```

Seven mocked tests pass: missing allocation/step, wrong GPU, missing image, existing-output preservation, CLI-only
execution and loader failure propagation.

## Version-pinned interface findings

Inspected upstream tag **v0.6.3**, not moving main or v0.7 documentation. These are source findings, not results
from running the image. The build recipe contains no application patch for these interfaces.

| Interface | What exists | Missing gate for Mu2e flash |
|---|---|---|
| Field | `DetectorConstruction::construct_field` accepts only `uniform`/`rzmap`; `GlobalSetup` advertises those two | DSMap requires a Cartesian application adapter, not merely a JSON option. Do not replace it with RZ without measuring the approximation |
| Cartesian library | `CartMapFieldAlongStepFactory`, `CartMapMagneticField`, `CartMapFieldInput` | Wire the same map into G4 and Celeritas; test offsets, axis order, native units, interpolation and boundary policy |
| GDML | `GeantGdmlLoader` discovers `auxiliary auxtype="SensDet"` entries | Inspect the matched full DS/tracker GDML, material/copy-number identities and sensitive gas volumes; extracted KPP geometry is not the beam geometry |
| Primaries | `event_file` is HepMC3, or use mutually exclusive `primary_options` | Export matched MDC2025ab DS-start e± events with vertices, momentum, time, identity and multiplicity preserved; art is not read directly |
| Hit output | `sd_type="event_hit"` records total edep (MeV), pre-step time (s), logical-volume instance and copy number | Build has `~root`; `GlobalSetup` warns hits will not be written. Need an explicit output adapter or user-approved ROOT-enabled build |
| Aggregate score | `sd_type="simple_calo"` integrates edep over all events | Useful smoke diagnostic, not per-event uncertainty, ancestry, StrawGasStep eIon or completed-event denominator |
| Physics/cuts | Standalone physics list/options, not Mu2eG4 cut code | Match range cuts, Mu2e vacuum/air e± kill cuts, time window, processes and scorer semantics before comparison |

The library Cartesian field uses **native units** (cm/gauss for CGS builds), flattened `[X][Y][Z][3]`. Do not pass
Mu2e mm/T arrays unchanged. Verify the actual image unit system. Field-map availability is not an assertion that
the existing beam zero-outside policy matches Celeritas clamping or Offline's full field selection.

`event_hit` calls `GetTotalEnergyDeposit()`, not the Mu2e `StrawGasStepMaker` aggregation or an explicitly ionising
score. Establish total-vs-ionising and grouping/time semantics, including zero-deposit events, before calling the
output `flash_edep`. Keep generated-event and primary-particle denominators separately until the production
objective contract is resolved; multi-primary events make them different.

## Bounded follow-up sequence

1. Load probe in an existing allocation. Preserve output, version/features and errors before considering rebuilds.
2. Synthetic material/geometry EM smoke test, G4-only versus offload, with explicit scorer and completed counts.
   This tests the engine, not the Mu2e objective; no Mu2e data need leave FNAL.
3. Prepare a small Cartesian-field/scoring adapter against this exact library version. The user builds it; do not
   silently change the image or substitute current upstream interfaces. ROOT is one output option, not mandatory.
4. Verify matched GDML, DSMap and MDC2025ab primary export plus scoring/cut contracts. Internal data staging to UVA
   must use the approved per-job tmpfs/cleanup workflow; none has been transferred by this restart.
5. Prepare a uniquely tagged DS-start G4 pilot/reference and compare photon production and tracker deposits.
   Existing `beam/art/run_g4ref_flash.sh` selects **Run1Baa TS3-start**, so is not the DS reference launcher.
   Reuse the TS diagnostic only for its documented composition results. Do not launch the proposed 2M reference
   until source/scoring/geometry and resource choices are approved.

No `flash_point`, shower implementation, ranking validation, DS reference completion or speedup is claimed.

## Primary sources (v0.6.3)

- [CLI](https://github.com/celeritas-project/celeritas/blob/v0.6.3/doc/usage/execution/_usage/celer-g4.txt)
- [Input](https://github.com/celeritas-project/celeritas/blob/v0.6.3/app/celer-g4/RunInput.hh)
- [Field and SD construction](https://github.com/celeritas-project/celeritas/blob/v0.6.3/app/celer-g4/DetectorConstruction.cc)
- [ROOT output guard](https://github.com/celeritas-project/celeritas/blob/v0.6.3/app/celer-g4/GlobalSetup.cc)
- [Hit score](https://github.com/celeritas-project/celeritas/blob/v0.6.3/app/celer-g4/SensitiveDetector.cc)
- [GDML SD tags](https://github.com/celeritas-project/celeritas/blob/v0.6.3/src/geocel/GeantGdmlLoader.cc)
- [Cartesian input](https://github.com/celeritas-project/celeritas/blob/v0.6.3/src/celeritas/field/CartMapFieldInput.hh)
- [Along-step factories](https://github.com/celeritas-project/celeritas/blob/v0.6.3/src/accel/AlongStepFactory.hh)
