# Approximations

Each workflow replaces some stock stages with a model: a condensed transport, a library, a response layer. This page
lists them per workflow, with the switch that selects a level where one exists. Each entry also gives the
stock-equivalent alternative. Some approximations have no switch yet. Those are listed as "no switch" and cannot be
selected.

## Selecting levels

Every chain command and the shared timer take the same three options:

```bash
--approx-preset default|fast|fidelity      # all approximations at once
--approx NAME=LEVEL                        # one approximation, after the preset (repeatable)
--approximations FILE                      # a config with an approximations section (JSON, or YAML with PyYAML)
```

A config section names one workflow's levels, or several workflows keyed by name. With `--approx-preset`, the
preset on the command line wins over the file. `--approx` is applied last.

```yaml
approximations:
  W5: {preset: fidelity, pulse_fit: short_window}
  W2: {calo_albedo: none}
```

The presets:

| preset | each approximation at |
|---|---|
| `default` | its default level: what the chain does without these options |
| `fast` | its fastest level |
| `fidelity` | the level closest to stock that the code has; an approximation with no such level stays at its default and is listed under `no_fidelity_path` |

### What the levels do and do not change

- **Defaults.** A default level sets no option, so a run without the options passes exactly the arguments it did
  before. The test suites check that the default levels set nothing.
- **Levels map to existing options.** Each other level sets options the chain already has, for example
  `runchain --below-msc` or `cetracker --complete-outer-wall`.
- **Conflicts are refused.** An option given explicitly with a different value is refused. The exception is a level
  declared to refine an option the run chose (`replaces`, for example one table name to another). Such a level needs
  that option given and never turns an output on by itself.
- **Missing levels are refused.** A level an approximation does not have is refused, and so is an approximation that
  has no switch. Nothing is invented where no path exists.
- **Bench entry points.** The shared timer (`python3 -m mu2efastsim.bench`) finds the declarations in
  `<package>.approximations` from the entry point's package and `--workflow`, and adds the levels' arguments to
  `--kwargs`.
  - An entry point that does not take a level's argument refuses that level. For example, the W1 bench entry runs
    only the condensed chain.
- **Precision is separate.** Device precision (float32 or float64) is not a level here. The chains keep their own
  precision options.

## Provenance

Every run records an `approximations` entry (schema 1):

- `workflow`, `preset` and `overrides`.
- `levels`: what ran.
- `selected`: what was asked for. It differs from `levels` where the run's other options decided otherwise, for
  example no albedo on the NumPy tracker, or the full-length pulse fit without `--jax`. The reason is under
  `effective`.
- `non_default`.
- `no_fidelity_path` and `no_stock_path`.
- `fixed`: the approximations with no switch.

Where each workflow writes it:

| workflow | written to |
|---|---|
| W1 | `beamfastsim.tsdevice`: the NPZ `meta` |
| W2-W4 | `beamfastsim.cetracker`: `handoff.json` |
| W5 | `crvfastsim runchain`: `runchain_stats.json`, the `--emit` file and the `--save-reco` NPZ. `runchain-merge` refuses shards with different levels. `crvfastsim.analyses.runchain_bench`: its report |
| W6 | `mixing.stream_benchmark`: `configuration.json` |
| W7 | the timer report only (the device chain has no command) |
| W8 | `trigger.device.run decide`: the decide JSON |

Every workflow also writes the entry to the timer report.

To print the record a selection would write:

```bash
python3 -m mu2efastsim.approximations crvfastsim --record W5 --approx-preset fidelity
```

## The table

The table is generated from the declarations; a test keeps it in sync. To regenerate it:

```bash
python3 -m mu2efastsim.approximations beamfastsim crvfastsim calofastsim mixing trigger
```

The alternative column says what gives the stock result. For an approximation with no switch, it gives the stock
behaviour and any stock-art route outside the chain's levels:

| alternative | meaning |
|---|---|
| exact port | a level of the chain runs the ported stock algorithm |
| stock art | stock products read from art or EventNtuple replace the approximated stage, through a different input of the chain or a stock job |
| none | the package has no stock-equivalent path yet |

<!-- table: python3 -m mu2efastsim.approximations -->
| workflow | approximation | levels (default first) | fast | fidelity | stock alternative | switch |
|---|---|---|---|---|---|---|
| W1 | `ts_transport`: TS stage: condensed device transport (RK4 in the field bank, ionisation with Urban fluctuations, cumulative Highland scattering, step caps), or the DS stage started from the reference's stock Geant4 DS-entrance states | condensed, stock_ds_entrance | condensed | stock_ds_entrance | stock art: stock DS-entrance states of the reference (art ReadVirtualDetector) feed the DS stage | tsdevice --arm (CLI only; the bench entry runs the condensed chain) |
| W1 | `ts_vacuum_step`: TS stage step length in vacuum: the fixed step everywhere (default), or a long step (up to 20 mm, at most 0.1 rad of turning and a 0.25 mm sagitta) where its chord crosses no material, stays inside the envelope and does not reach the DS plane | fixed, long | long | fixed | none: Geant4 takes field-limited steps in vacuum (deltaChord 0.25 mm); the fixed 2.5 mm step is the closer, slower model | tsdevice --vacuum-step; bench_chain(vacuum_step=); Settings.vacuum_step_mm |
| W1 | `pion_nuclear_interactions` (no switch) | always on | - | no level | not modelled: pions reaching the DS are counted, not transported | - |
| W1 | `highland_gaussian_scattering` (no switch) | always on | - | no level | Gaussian Highland, no single-scattering tails (stock: Geant4 Urban MSC) | - |
| W1 | `stopped_pion_and_capture_products` (no switch) | always on | - | no level | stopped-pion daughters, muon capture and decay electrons not produced | - |
| W1 | `ds_layer_physics` (no switch) | always on | - | no level | thin-plane foils with per-layer Highland and restricted loss (stock: Geant4 in the foil geometry) | - |
| W2-W4 | `calo_albedo`: tracker returns from the calorimeter (backscatter): sampled from a Geant4-built library, or none; the NumPy tracker has no albedo | library, none | none | library | none: Geant4 transport through the calorimeter; no stock path in the chain (the library is built from it) | cetracker --albedo (device tracker); DeviceChain(albedo=) |
| W2-W4 | `straw_wall`: straw wall material: counted only on chords that cross the gas (default), or the complete outer-minus-gas wall with wall-only chords | gas_chords, complete | gas_chords | complete | none: Geant4 straw walls; the complete wall is the closer model | cetracker --complete-outer-wall; DeviceChain(complete_outer_wall=) |
| W2-W4 | `tracker_rows`: the response layer's StrawGasStep rows per track (nrow): from the tracker and albedo transport (default), or drawn from a library of the default chain's rows per (|p|, tan dip, r_max) cell of the entry helix, without the tracker transport (track observables only: no steps, digis or delta rays) | transported, sampled | transported | transported | none: Geant4 tracker steps; the transported rows are the closer model (the library is built from them) | DeviceChain(nrow_library=); bench entry cetracker:bench_run (not the cetracker CLI, which writes steps) |
| W2-W4 | `primary_only_steps` (no switch) | always on | - | no level | no delta rays, conversions or other secondaries in the tracker besides the albedo (DeviceChain(deltas=) takes a Geant4 delta-ray library .npz, opt-in; no level until the library ships); stock StrawGasSteps from the production reco are the stock-art path | - |
| W2-W4 | `reco_response` (no switch) | always on | - | no level | no track fit: efficiency and residuals drawn from stock LoopHelix tables (default: measured on the full Geant4 events, with the empirical joint draw; CE-only tables by name, DeviceChain(response="ceonly")); stock art: the injector (fasttrk_field.root) through StrawGasStepsFromTable, stock digis and LoopHelix | - |
| W2-W4 | `target_slab_physics` (no switch) | always on | - | no level | flat slabs, constant field per layer, shell curvature as an axial field, no secondaries (stock: Geant4 in the target geometry) | - |
| W2-W4 | `rpc_internal_electron_only` (no switch) | always on | - | no level | W4: internal conversion only, the positron dropped (stock: external RPC with photon transport and both leptons) | - |
| W5 | `muon_eloss`: stage-C muon energy loss: the run configuration's model (range-out with sub-stepped bars where range_out is set) or one bar loss at the entry momentum | run_config, entry_momentum | run_config | run_config | stock art: Geant4 CrvSteps from art through runchain --steps-input (response and reco only) | runchain --legacy-transport |
| W5 | `below_msc`: run configurations with a below-plane world (Run 1A, 1B, 2): straight lines below the MSC plane, or multiple scattering bar to bar (NumPy transport) | straight, scattered | straight | scattered | stock art: Geant4 CrvSteps from art through runchain --steps-input (response and reco only) | runchain --below-msc |
| W5 | `pulse_fit`: with --jax: 12-sample, 3-round device pulse fit, flagged waveforms refit at full length; full_length runs the NumPy response, which fits every waveform at full length | short_window, full_length | short_window | full_length | exact port: full-length pulse fit port (reco.pulsefit.reconstruct_full; stock Minuit oracle reco.stockfit) | runchain --jax (absent: NumPy response, full-length fit) |
| W5 | `straight_line_transport` (no switch) | always on | - | no level | Geant4 transport; the stock-equivalent input is Geant4 CrvSteps from art through runchain --steps-input (response and reco only) | - |
| W5 | `secondaries_library` (no switch) | always on | - | no level | Geant4 secondaries; the stock-equivalent input is Geant4 CrvSteps from art through runchain --steps-input (response and reco only) | - |
| W6 | `finite_donor_pools` (no switch) | always on | - | no level | donor frames read in order and reused when the pool wraps; native art mixing (fcl/four_stream_response.fcl) | - |
| W6 | `independent_sampler_draws` (no switch) | always on | - | no level | one shared intensity and four conditional Poissons from the JAX RNG, no event replay; stock seeded replay: fcl/four_stream_fast_inputs.fcl | - |
| W6 | `tracker_crv_only` (no switch) | always on | - | no level | calorimeter and trigger not mixed; native art mixing (fcl/four_stream_response.fcl) | - |
| W7 | `digi_response`: crystal hits: photoelectrons, waveforms, digis and the template fit (default), or drawn per (event, crystal) from a table of the full chain's outcome (hit, energy ratio, time offset) per bin of visible energy, without digis (calofastsim.param_response) | waveform, parametric | parametric | waveform | none: stock CaloDigiMaker and CaloRecoDigiMaker; the waveform response is the closer model (the table is built from it) | device_chain.make_chain / make_steps_chain(digi_response=) |
| W7 | `shower_library` (no switch) | always on | - | no level | whole-shower library at the disk front, one of the K entries nearest in entry radius; stock CaloShowerSteps through device_chain.make_steps_chain / steps_batch (exact on stock steps) | - |
| W7 | `crystal_placement` (no switch) | always on | - | no level | library shower moved onto the crystal grid at the query position; stock CaloShowerSteps through device_chain.make_steps_chain / steps_batch (exact on stock steps) | - |
| W7 | `direct_front_entries_only` (no switch) | always on | - | no level | only direct disk-front entries get a response; stock CaloShowerSteps through device_chain.make_steps_chain / steps_batch (exact on stock steps) | - |
| W7 | `template_fit_single_pass` (no switch) | always on | - | no level | Newton template fit without the second pass for chi2/ndf > 1 (stock: TMinuit with the refit) | - |
| W8 | `compact_helix_fit` (no switch) | always on | - | no level | compact device helix fit in place of RobustHelixFinder/AgnosticHelixFinder; stock trigger replay on the same digis (trigger/replay.py --straw-digi-tag, trigger/hybrid pairing) | - |
| W8 | `kalseed_response` (no switch) | always on | - | no level | KalSeedFilter as cuts plus a per-path pass probability, per process (decide --response ce, dio, rpc, none or a file; default ce): the shipped binned response, rows in time-cluster hits and helix momentum (CE/DIO: the apr helix paths; RPC: every tpr/apr track and helix path), scalar factors from CE for the other paths; none: probability 1; stock trigger replay on the same digis (trigger/replay.py --straw-digi-tag, trigger/hybrid pairing) | - |
| W8 | `exact_weighted_median` (no switch) | always on | - | no level | time clusters use the exact weighted median where stock uses the P2 estimate; stock trigger replay on the same digis (trigger/replay.py --straw-digi-tag, trigger/hybrid pairing) | - |
| W8 | `tracker_path_subset` (no switch) | always on | - | no level | 20 single-helix tracker paths of 52 on the device; the replay runs all 52 | - |
| W8 | `no_delta_finder` (no switch) | always on | - | no level | DeltaFinder skipped (exact without pileup); stock trigger replay on the same digis (trigger/replay.py --straw-digi-tag, trigger/hybrid pairing) | - |
<!-- end table -->

## Notes per workflow

- **W1.** `ts_transport=stock_ds_entrance` runs the DS stage from the reference's stock DS-entrance states. This is
  `tsdevice --arm stock-ds`. `ts_vacuum_step=long` (opt-in, the `fast` preset) lets a lane take a step of up to
  20 mm where the chord crosses no material, stays inside the envelope and does not reach the DS plane, capped at
  0.1 rad of turning and a 0.25 mm sagitta (Geant4's deltaChord); elsewhere the 2.5 mm step is unchanged. Against
  the fixed step (12 seeds of each Run 1A cohort, about 25,000 inputs each): stops per input 1.005 +- 0.015 (float64)
  and 1.002 +- 0.015 (float32), DS-entering muons 0.999 +- 0.009 in both, foil, stop time, stop radius and
  DS-entrance momentum shapes consistent, with 2.7x fewer steps per lane. The other step sizes and caps are numeric
  settings (`tsdevice --step`, `Settings`), not approximation levels.
- **W2-W4.** One declaration serves the stops (W2), DIO (W3) and RPC (W4) electrons of `cetracker`.
  - The albedo levels act on the device tracker; the NumPy tracker has none.
  - `straw_wall=complete` is the existing opt-in complete outer wall.
  - `tracker_rows=sampled` (opt-in, by name only) draws each track's rows (`nrow`, read by the response layer only
    through its bin) from a library of the default chain's rows per (|p|, tan dip, r_max) cell of the entry helix
    (`trkfastsim.recoresp.nrow`), so the tracker and albedo are not transported. It serves the observables boundary
    only: no steps, digis or delta rays (DeviceChain refuses deltas with it, and a call with digis). Against the
    transported rows on the same births and keys (100,000 each of CE, DIO and RPC electrons, both dtypes), the fit,
    quality and CE-selected efficiencies agree within their paired errors (CE selection: CE -0.2 / +0.2 +- 0.17 %,
    DIO -0.1 / +0.1 +- 0.22 %, RPC -0.8 / -0.5 +- 0.5 %, float64 / float32), and so do the observable shapes.
  - The response layer is not a level. Its default tables are measured on the full Geant4 events and draw residuals
    and quality jointly from reference tracks (pools); the CE-only tables (`DeviceChain(response="ceonly")`) remain
    available by name.
  - `DeviceChain(deltas=)` appends Geant4 delta-ray straw steps from a library. It becomes a level once the library
    ships with the package; until then the primary-only steps are listed without a switch.
  - The stock route for the response layer and the primary-only steps is the stock reconstruction of the injector
    file. It is a separate job, not a level.
- **W5.**
  - `pulse_fit=full_length` runs the NumPy response, which fits every waveform at full length. It cannot be combined
    with `--jax`, and the runchain timer, which always runs on JAX, refuses it.
  - `below_msc` matters only for run configurations with a below-plane world.
  - The stock-art input for the transport and the secondaries is Geant4 CrvSteps through `--steps-input`.
- **W6, W8.** These have no switchable approximation yet. The stock alternatives are the native art mixers (W6) and
  the stock trigger replay on the same digis (W8).
- **W6 device approximations set by the capacity file.** The default Run 1A file
  (`mixing/capacities/run1a_mix1bb_batch256.json`) has two: donor-frame digi reuse for the tracker (`tracker_reuse`,
  [mixing/DEVICE_TRACKER.md](../mixing/DEVICE_TRACKER.md)) and CRV thermal noise only on counters with steps
  (`thermal_counters: "hit"`, [mixing/DEVICE_CRV.md](../mixing/DEVICE_CRV.md)). `run1a_mix1bb_batch256_exact.json` is
  the same file without either. The run records the capacity file it used.
- **W7.** `digi_response=parametric` (opt-in, the `fast` preset) draws each (event, crystal) hit from a table of the
  full chain's outcome per bin of visible energy (`calofastsim.param_response`) instead of photoelectrons, waveforms,
  digis and the template fit: steps -> hits + clusters, no digis. On the held-out CE files (14,218 events, the same
  steps, timing and keys, against the waveform response): leading-cluster peak ratio 0.9988 +- 0.0018, width
  +0.03 MeV, yield above 10 MeV 1.0018 +- 0.0006, energy chi2 28/41, x / y / radius chi2 3/52, 3/50, 2/11, leading
  time paired difference 0.01 +- 0.01 ns. Clusters above 10 MeV per event come out 1.2 % high (0.728 against 0.719,
  chi2 17/5): the table draws each crystal on its own, without the time structure of its deposits. The stock
  alternative of the other approximations is stock steps through `make_steps_chain`.
