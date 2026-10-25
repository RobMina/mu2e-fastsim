# Celeritas 0.6.3: load probe and flash integration gates

Status (2026-10-25): image packaging, A100-node CLI loading and **synthetic GPU EM transport/scoring** confirmed.
Mu2e geometry, Cartesian field, cuts and flash objective remain unvalidated. Load-only job **21958048**
completed at 16:58 EDT (4 s elapsed, exit 0), on `udc-an36-19`, A100-SXM4-80GB, driver 595.71.05, Apptainer 1.4.5.
The probe ran `--help`, `--version` (0.6.3), and `--dump-default` successfully; the latter reports CGS input units.
Evidence: `/scratch/ram2aq/mu2efastsim/probe/celer_load_21958048/` and `celer-load-21958048.out`; synced to
`~/mu2e/remote-uva-data/mu2efastsim/probe/`. No build, transport or internal-data transfer occurred.
At 16:53 EDT, `squeue -u ram2aq` and `squeue -A ldmxuva` returned no jobs. This is not a lack of
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

An existing interactive allocation is **not** a prerequisite: submit a short batch job through the approved
`ldmxuva` queue workflow. Deployed through the UVA mirror as
`/home/ram2aq/mu2e/fastsim_uva/probe_celeritas_0a1589c.sh` (SHA256
`a63bae495671d9a3235b12d817d8c80ac84c816c3f3e342e6367cbdd557b8fd2`). The completed job used this command on the login host:

```bash
sbatch --parsable --job-name=m2fs_celer_load --account=ldmxuva --partition=gpu \
  --gres=gpu:a100:1 --cpus-per-task=2 --mem=8G --time=00:05:00 \
  --output=/scratch/ram2aq/mu2efastsim/probe/celer-load-%j.out \
  --wrap='module load apptainer && srun --ntasks=1 bash /home/ram2aq/mu2e/fastsim_uva/probe_celeritas_0a1589c.sh /scratch/ram2aq/mu2efastsim/mu2e-celeritas.sif /scratch/ram2aq/mu2efastsim/probe/celer_load_${SLURM_JOB_ID}'
```

Check existing jobs before submitting to avoid duplicates. Laptop guard checks (no container or GPU involved):

```bash
bash -n container/probe_celeritas.sh
PYTHONDONTWRITEBYTECODE=1 ~/mu2e/.venv/bin/python3 -m unittest discover -s container/tests -v
```

Seven mocked tests pass: missing allocation/step, wrong GPU, missing image, existing-output preservation, CLI-only
execution and loader failure propagation.

## Synthetic transport/scoring gate: passed, with a diagnostic-counter defect

Job **21958075**, October 25 17:03 EDT, A100-SXM4-80GB on `udc-an34-19`, completed in **5 s**, exit 0.
`smoke_celeritas.py` writes an original synthetic GDML: 1 m lead cube inside a 2 m vacuum world, a central
10 MeV source, uniform 1 T field, `celer_em`, and `simple_calo` on the absorber. Six serial-process arms run
64 events each: e-, e+, gamma, each with Geant4-only (`CELER_DISABLE=1`) and GPU offload. The same deterministic
primary specifications are used, not common stochastic transport histories. The default range cut is 0.1 cm;
this is not a Mu2e-like geometry, cut set or response model.

| Source | Finished event records per arm | G4 edep [MeV] | GPU edep [MeV] |
|---|---:|---:|---:|
| e- | 64 | 640.000000 | 640.000000 |
| e+ | 64 | 705.40786048 | 705.40786048 |
| gamma | 64 | 640.000000 | 640.000000 |

Counts come from persisted `result.time.events`, recorded by `EventAction::EndOfEventAction` **after** flushing
transport; not merely the input count. Positron energy includes annihilation with a material electron,
`64 * (10 + 2 * 0.51099891)` MeV. Full containment is deliberately an energy-bookkeeping smoke test, not a
shape/distribution comparison. Maximum absolute relative energy residual is 3.4e-15. GPU output reports CUDA,
A100, transport/hit-gather kernels, nonzero along-step action timing and 64 offloaded tracks per arm. The job
therefore demonstrates actual offload, rather than just executable loading. No speedup is claimed: these tiny
per-event-flush runs are dominated by GPU overhead (transport ~0.32-0.36 s versus G4 ~0.003-0.005 s).

**Counter defect:** GPU log summaries claim roughly 1.0715 billion hits. Upstream v0.6.3
`src/celeritas/ext/detail/HitProcessor.hh` declares `size_type num_hits_;` with no initializer; its constructor
in `.cc` does not initialize it, accumulation uses `+=`, and `exchange_hits()` resets it only after the first
flush. This is a source-supported explanation for the bogus diagnostic total, not a billion scored hits.
Exclude this log count; the separately accumulated calorimeter energy closes. Fix initialization in a future
user-built adapter/image. The existing binary has not been patched and no upstream issue/PR was submitted.

Evidence: `/scratch/ram2aq/mu2efastsim/probe/celer_smoke_21958075/` and `celer-smoke-21958075.out`, synced to
`~/mu2e/remote-uva-data/mu2efastsim/probe/`. The executed script hash is
`8e376c76768e489c9502eac156e44aecca626ebcc85b5da28f94d97e1627030e` (h1, before adding the offline checker).
`--check-only` inspected the saved results without rerunning either arm. `summary.json` holds the checked values.
The current runner automatically performs the same check after running; **17 local tests** cover the probe and
smoke card/environment/score/count/device guards. Only public synthetic data were used; no build or install.

```bash
# Submit after deploying both files through the UVA mirror and checking their hashes.
sbatch /home/ram2aq/mu2e/fastsim_uva/smoke_celeritas.sh /home/ram2aq/mu2e/fastsim_uva/smoke_celeritas.py
# Recheck completed output locally, with no new transport.
~/mu2e/.venv/bin/python3 container/smoke_celeritas.py \
  ~/mu2e/remote-uva-data/mu2efastsim/probe/celer_smoke_21958075 --check-only
```

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

1. Load probe completed as a normal batch job (21958048); no prior interactive allocation required.
2. Synthetic material/geometry EM smoke completed (21958075), G4-only versus offload, with explicit score/count
   checks. This tests the engine, not the Mu2e objective; no Mu2e data left FNAL.
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
