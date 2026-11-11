# EdepAna-format files from the fast chain (`beamfastsim.edepana`, opt-in)

The stopping-target design studies chain two analyses of the Mu2e analysis server: `edep` runs Offline's EdepAna
module (`Analyses/src/EdepAna_module.cc`, Offline v13_39_00 and later) on the CeEndpoint (`mustops_ce`) art files and
keeps its ROOT file; `approx_ce_sensitivity` reads that file's `EDepAna/tree` and returns S/sqrt(B). `edepana` writes
the same tree from the fast chain, so the second analysis, and the tree and summary readers of the first, run
unchanged on fast output. The physics definitions stay in the analysis server.

```bash
python3 -m beamfastsim.edepana ce --stops STOPS.npz --geom GEOM.txt --electrons 200000 --seed 1 --dtype float64 \
    --library SHOWER_LIBRARY.npz --gdml RUN1A.gdml --procs 15 --out DIR
python3 -m beamfastsim.edepana point --g4 BEAM_TABLE.npz --z-up 3900 --geom GEOM.txt --muons 0 ... --out DIR
```

fastsimkit task `beam_edep` (params `geom`, `muons`, `electrons`, `seed`, `dtype`) runs `point` as one GPU job and
returns the file with the results (kind `root`).

## Outputs

| File | What | Read by |
|---|---|---|
| `nts.mu2efastsim.edep.root` | `EDepAna/tree` (TTree), one row per stored event; `EDepAna/fastsim_provenance` (TObjString, JSON) | `read_edep_tree`, `generated_events`, `approx_ce_sensitivity` |
| `<file>.edep.json` | `{"max_events": null}`: one full run, the record `edep` writes beside its file | `approx_ce_sensitivity` (`stops_per_pot` route) |
| `edep_summary.txt` | the block EdepAna prints at endJob, same arithmetic and format | `parse_edep_summary` |
| `<file>.fastsim.json` | provenance (code commit, chain configuration, inputs, counters) and the summary numbers | |
| `metrics.json` (kit) | the summary, the events with `event_calo_edep_vis > 10` and their rate per generated event, stops | fastsimkit |

The file is a TTree written with uproot, not an RNTuple: EdepAna writes a TTree through TFileService and the readers
use uproot's TTree interface (`tree.arrays(..., library="np")` with jagged per-primary branches). The jagged branches
share the counter `nprimaries`, as in the stock file. Not written: EdepAna's `hist_N` histogram directories and its
endJob Landau-fit lines, which no analysis reads.

## Contract: `EDepAna/tree`

All per-primary branches are jagged with counter `nprimaries` (CeEndpoint: one primary, the conversion electron).
Frame: Mu2e (the analysis server's `primary_start_r` subtracts the detector origin x = -3904 mm itself).

| Branch | Type | Unit | EdepAna definition | Fast source | Class |
|---|---|---|---|---|---|
| `nprimaries` | int32 | | primaries of `FindMCPrimary` | 1 | exact |
| `primary_start_{x,y,z}` | float[] | mm | SimParticle start position | stop resampled from the stop table (the CeEndpoint `TargetStopResampler` draw) | exact given the stops |
| `primary_start_{px,py,pz}` | float[] | MeV/c | start momentum | isotropic, \|p\| = 104.9726 MeV/c (CeEndpoint on Al) | exact |
| `primary_start_e`, `primary_start_m` | float[] | MeV | start energy, mass | sqrt(p^2 + m_e^2), m_e | exact |
| `primary_start_pdg` | int[] | | PDG id | 11 | exact |
| `primary_trk_front_p` | float[] | MeV/c | \|p\| of the primary's first-in-time StepPointMC at VD `TT_FrontHollow` / `TT_FrontPA`; 0 if none | `cechain_jax` state at the tracker entry plane (z 8539.9 mm; Geant4 VD13 8539.48 mm); 0 if the electron never reaches it | approximate |
| `primary_trk_front_energy` | float[] | MeV | sqrt(p^2 + m_start^2) at that step | the same from the plane state | approximate |
| `primary_calo_edep`, `primary_calo_edep_vis` | float[] | MeV | CaloShowerStep G4 / Birks energy of the primary and its descendants | = the event values (in a CeEndpoint event every calorimeter step descends from the electron) | approximate |
| `event_calo_edep`, `event_calo_edep_vis` | float | MeV | sum of all CaloShowerSteps, G4 / Birks energy | whole-event Geant4 shower pattern of the calorimeter shower library drawn at the first disk-front crossing (both disks' steps); electrons without one: the `cesob` wall proxy, a pattern drawn at the first crystal contact when it accepts | approximate (gap below) |
| `event_trk_edep` | float | MeV | sum of StrawGasStep ionising energy | the electron's own fieldsteps rows plus the albedo returns | approximate, low (gap below) |
| `run`, `subrun`, `event` | int32 | | art event id | run 1 (option), one subrun per batch, event = birth index + 1 | bookkeeping |
| `weight` | float | | 1 | 1 | exact |
| `ngen` | int64 | | running sum of each subrun's `GenEventCount`, updated at beginSubRun | running sum of the births per batch | exact |

**Stored events.** A CeEndpoint art file keeps only the events passing `DetectorStepFilter` (Production
`JobConfig/primary/prolog.fcl`, `PrimaryFilter`): a SimParticle with at least 12 StrawGasSteps of ionising energy
> 1 eV and \|p\| > 50 MeV/c, or visible calorimeter energy of at least 45 MeV. EdepAna counts the generated events
of every subrun in `ngen`, so the stored fraction is part of every per-generated-event number. The writer applies
the same filter to the fast rows and the fast visible energy (approximate: it inherits the gaps below).

**Summary block** (`edep_summary.txt`): `Saw <sum of weights> events (<ngen> gen events) --> output rate = ...`, the
mean `event_calo_edep` per stored and per generated event, the weighted count with `event_calo_edep_vis > 50`, and
the mean `event_trk_edep` per stored and per generated event, printed as `std::cout` prints a double.

## Known gaps and the branches they change

- **No delta rays or conversions in the fast tracker steps** (unless `--deltas`, the opt-in delta-ray library):
  `event_trk_edep` is low (in Geant4 CE events the eIoni deltas carry about a third of the gas ionisation) and so are
  the `edep` tracker averages. The tracker-front branches are upstream of the tracker and do not depend on it; the
  storage filter counts the electron's own steps and is barely affected.
- **Calorimeter entries through a disk's inner edge.** The shower library holds Geant4 patterns of disk-front
  entries only; electrons that enter a disk from the bore go through the `cesob` wall proxy (the inner Al and
  carbon-fibre rings, then at least 17.8 mm in a crystal) and draw a front-entry pattern at their first crystal
  contact. This changes `event_calo_edep(_vis)`, `primary_calo_edep(_vis)`, the selection `event_calo_edep_vis > 10`
  (the CE acceptance of `approx_ce_sensitivity`, so its signal and both backgrounds) and the calorimeter branch of the
  storage filter. The front branches (the signal shape) do not depend on it.
- The shower library comes from Geant4 option-4 CeEndpoint events; the design-study boards ran FTFP_BERT (option 0),
  whose energy-loss tail before the tracker is about 1 % smaller (relative) at 1 MeV below the birth energy.
