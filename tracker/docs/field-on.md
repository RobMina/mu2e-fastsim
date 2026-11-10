# Tracker with the field on (opt-in)

`trkfastsim.sim.fieldsteps` transports charged primaries through the tracker in a magnetic field and writes their
StrawGasSteps; `python3 -m trkfastsim field-compare` compares them with Geant4 on the same tracks. Nothing in the
default (field-free, extracted) chain changes. The world transport above the tracker (CRY to the DS) is not part of
this: tracks start from a given state at the tracker envelope.

## Model

- **Stepping:** chords of the local helix (`mu2efastsim.ehelix.helix`, field at the helix midpoint of the step), at
  most `chord_mm` long and with a sagitta below `sagitta_mm` (defaults 20 mm and 0.02 mm; 10 mm for electrons in the
  comparisons). The field is a G4BL map (`ehelix.read_g4bl` + `grid_field`, e.g. the DS map), a constant vector, or
  none. The tracker volume is vacuum.
- **Straws:** gas crossings per chord (`ChordIndex`: panels sorted by z, only those overlapping the chord are tested;
  same rows as `crossings.straw_crossings` clipped to the chord); a crossing cut by a chord end is merged with its
  continuation, so a row is one passage through a straw.
- **Material:** straw walls (outer-cylinder chord minus the gas chord), sense wire at its closest approach, gas, and the
  support structure from the GDML (`fieldsteps.interior`: TrackerInterior's coaxial shells, limited to pieces reaching
  inside the envelope cylinder; no calorimeter).
- **Loss:** muons, the slice sampler of `mu2efastsim.transport.slab_loss` (ionisation, and the core default for
  bremsstrahlung, pair production and photonuclear losses in the solids; `radiative=` forces either); electrons,
  `fieldsteps.electron_loss` (`mu2efastsim.eioni` continuous loss below 0.1 MeV with Urban fluctuations, deltas and
  brem photons above it sampled one by one). Gas eIon per merged crossing: `eloss.sample_loss` (muons) or
  `electron_gas_eion` (restricted Moller loss, largest transfer T/2). The straw gas is `steps.STRAW_GAS`, with
  Geant4's gas branch of the density effect (the geometry's state); `gas=` takes another material.
- **Scattering:** one angular kink per chord at its end, Highland with the log term on the cumulative x/X0, plus the
  single-scattering tail for muons (none for electrons); the kink's lateral displacement is dropped.
- **Step content** as `MakeStrawGasSteps` with the field on: path in the gas, mean of entry and exit momentum, width
  = min(sagitta rms, bend rms) with the nominal field at the tracker centre, shape from |p| against the straw-radius
  momentum, ionisation type, time along the arc.
- Tracks stop when they leave the envelope (straws + 150 mm radially and axially), range out, or reach the path or
  chord cap. Tracks that leave and come back (loopers in the field, electrons returning from the calorimeter) are not
  followed.

## Comparison

`field-compare` reads StrawGasStepTableDumper tables of a field-on Geant4 sample, takes the primaries of a species
(and, in a resampled cosmic stage, their same-species continuation), restarts each from its first tracker VD crossing
(`--entry vd`, `--entry-vds`), its first tracker SurfaceStep (`--entry surface`) or the earlier of the two
(`--entry first`), and compares the primary steps after that time, paired per track: straws per track, path, eIon,
width, shape and ionisation types, momentum and position at the last Geant4 straw, time at the first straw, and the
same block for the first passage (Geant4 steps up to the fast exit time). It also checks the geometry (Geant4 step
end points against the straw surfaces) and writes primary-only tables for the art injector
(`--write-tables`, `art/TrkFastSim` `StrawGasStepsFromTable`).

## Results (Run 1A Geant4 samples, DS map; first passage)

| | Muons (cosmic, ~13,600 tracks) | Conversion electrons (~56,100 tracks) |
|---|---|---|
| straws per track, fast - Geant4 | +0.02 +- 0.07 to +0.28 +- 0.07 (two seeds; Geant4 39.5) | +0.05 +- 0.02 (Geant4 39.5) |
| path in the gas, chi2/ndf | 44/48 - 68/48, means within 0.1 % | means within 0.1 %, chi2/ndf about 4 |
| eIon | mean within 0.4 %, MPV within 1.5 %; per mm by step \|p\|: within 0.4 % below 10 GeV/c, +0.1 to +0.8 +- 0.7 % above (+1.9 to +2.9 % with the solid branch) | -0.02 +- 0.07 %, MPV within 0.1 % (+2.3 % and +2.8 % with the solid branch) |
| width | -1.9 to -2.3 %, chi2/ndf 1.3 | within 0.3 %, chi2/ndf 1.3-2.5 |
| time at the first straw | below 1 ps | below 0.1 ps |
| momentum at the last Geant4 straw | within 0.3 MeV/c below 500 MeV/c | +0.1 MeV/c (Geant4 drop 0.75) |

Control arm without field: -2.3 straws per track and zero width. Step end points of the Geant4 sample lie on the
straws of the GDML used (99 % within 5e-5 mm of the gas surface).

Throughput, NumPy, one CPU core, transport only: about 90 muons/s and 57 conversion electrons/s.

Diagnostics: `--offset-bound` (first-straw offsets of side entries against the Highland spread of the support),
`--matched-straws` (time, wire DOCA and length on the straws both arms cross), `--delta-census` (Geant4 e+- daughters
above 10 MeV in the straws), `big_drops` (where tracks lose more than 10 MeV between straws), `--self-pass-only` and
`--unbiased-only` (the selection of the reference sample, below).

- **Side entries.** Muons entering through the outer cylinder reach their first straw 1.1, 0.75, 0.43, 0.25 mm
  (median, 2-5, 5-10, 10-20, 20-50 GeV/c) from Geant4 when the fast arm has no support material: this is Geant4's own
  scattering in 1.6-3.2 X0 of support over about 330 mm (inside the Highland band of every bin, offset x p roughly
  constant); with the fast support the offsets grow by 1.15-1.7 (independent draws: sqrt 2). Explained.
- **Electron gas eIon.** The straw gas is a gas in the Geant4 geometry, and Geant4 uses the gas branch of the
  Sternheimer parametrisation for it; the solid branch puts the restricted loss of 100 MeV electrons 2.4 % high.
  Explained; the gas branch, now the default, closes it (table above), and removes the +1.9 to +2.9 % excess of the
  muon eIon per mm above 10 GeV/c (`eion_per_mm_by_p` in `field-compare`, two seeds).
- **Reference-sample selection.** The cosmic reference keeps an event when some particle's straw steps have |p|
  between 30 and 250 MeV/c (15 steps, 16 planes), or above 1 GeV/c with 20 steps (prescaled 1/100). Almost every muon
  above 1 GeV/c in it was kept because another particle passed, typically a hard delta electron, and low-momentum
  muons are kept when their own loss upstream of the straws brought them into the window. The Geant4 muons are
  therefore conditioned on their energy loss: above 1 GeV/c, 13.6 % lose more than 10 MeV between two straws
  (fast 0.13 %, the Bethe rate), and e- daughters above 10 MeV appear in the straws about 100x more often than that
  rate. The momentum-loss differences above 1 GeV/c (+5 to +75 MeV/c at the last straw) are this selection, not
  missing physics: the radiative losses move the mean above 5 GeV/c by about +0.3 MeV/c. An unbiased high-momentum
  reference needs a sample without that filter.

## Stock reconstruction on field-on steps

Primary-only steps (fast; Geant4 after the entry state; Geant4 first passage) injected into the reference events
(`art/TrkFastSim` `StrawGasStepsFromTable`), then stock Run 1A OffSpill digitisation, stock reconstruction (CentralHelix
`KKCHmu`, `bfcorr` false) and EventNtuple, the same release and seeds for every arm; compared per event with
`python3 -m trkfastsim field-reco` (20 reference files, about 17,700 events with Geant4 primary steps, about 470 tracks
per arm). The EventNtuple configuration needs the CentralHelix fit under `trk.fits` (`EventNtuple.PathOff`, `trk.fits :
[ Off ]`, `FitType : CentralHelix`, `trk.fillTrkDtDt : false`) with the EventNtuple release of the backing Offline.

| Fast against Geant4 first passage (paired per event) | |
|---|---|
| track yield per event | +0.23 +- 0.06 % absolute (ratio 1.08); all of it from muons below 300 MeV/c at the Geant4 first straw (selection, above); 0.3-1, 1-2, 2-5, > 5 GeV/c within errors |
| active hits per track | +0.24 +- 0.16 (36.4) |
| chi2/ndof, fit consistency | -0.05 +- 0.03; +0.02 +- 0.01 |
| reco momentum - own first-straw momentum, median (sigma from the IQR) | 1-2 GeV/c +0.3 % (1.1 %) vs -0.1 % (1.1 %); 2-5 GeV/c +0.05 % (1.8 %) vs -0.2 % (1.8 %); > 5 GeV/c -0.3 % (6.2 %) vs -0.7 % (7.3 %) |

On the same straws, hit edep was +2.2 % higher in the fast arm (mean; TOT +0.7 %), as the solid-branch density
effect of these steps predicts for mostly > 5 GeV/c tracks (the gas branch, the default since, lowers the restricted
loss by 1.5-2 % above 10 GeV/c); the steps differ by
35 um in wire DOCA (2-5 GeV/c, IQR) and below 1 ps in time. Geant4 all passages against first passage: yield -0.03 +- 0.03 %, hits
-0.13 +- 0.07.

## Opt-in finite wall material

`fieldsteps.transport(..., complete_outer_wall=True)` includes finite-cylinder wall-only crossings and exact wall
lever moments in loss, scattering and tails. It replaces the gas-gated wall contribution rather than adding it;
gas-step creation, wire handling, material fractions and current radiative/gas defaults are independent. The
option **defaults to False**; the device twin has the same opt-in (`fieldsteps_jax.Tracker(..., complete_outer_wall=True)`,
`DeviceChain(complete_outer_wall=True)`, `cetracker --complete-outer-wall`). Gas and wire MSC lever moments still use the crossing midpoint
`s_mid` approximation; exact wall moments do not make those contributions exact. Candidate expansion is not
memory-bounded by the subtraction batch size.

The finite-wall solver deliberately uses exact `== 0` parallel/stationary tests: a tiny positive transverse
quadratic coefficient is still a nonparallel crossing. This fixes near-parallel wall intersections; it must not
share the legacy gas search's near-parallel threshold. The legacy gas search and default route remain unchanged.

The separate cosmic validation note records the saved development comparison and its limitations; those results
are not duplicated here and do not justify changing the default.

Synthetic tests cover finite cap/parallel/grazing intersections, material consumers, segmentation, and default
versus explicit-False byte/RNG equality. Normal pytest also compares all six default-route synthetic cases against
[frozen main 9e81577 outputs](../tests/data/fieldwall_main9e81577.md), generated independently with core d041bb3.
Schema, integers and RNG states are exact; floats use `rtol=1e-10`, `atol=1e-11` for cross-platform rounding.
Shared-route perturbation tests demonstrate that the fixtures catch regressions which omitted/False parity alone
cannot. No baseline source or fixture regeneration is needed to run these tests. An optional main source file can
also be compared byte-for-byte in the same process and environment:

```bash
python3 tests/test_fieldwall_parity.py --baseline /path/to/main/fieldsteps.py
```

## Conversion electrons through stock LoopHelix reconstruction

The generated electron of each Geant4 CeEndpoint dts event goes through the umbrella `beamfastsim.cetracker`
(`cechain` to the tracker entry plane, then `sim/fieldsteps`); the reference arm is the Geant4 primary electron's own
steps from its first VD13 crossing (`python3 -m trkfastsim.validation.cearms`). Both arms are injected into the same
dts events and run through the same stock OffSpill digitisation, stock LoopHelix reconstruction (`KKDe`, recoMC
OffSpill) and EventNtuple (`scripts/ce_steps_slurm.sh`, `scripts/ce_reco_slurm.sh`; submit with `sbatch -A YOUR_ACCOUNT`); secondaries and calorimeter
steps are in neither arm. `python3 -m trkfastsim.validation.cemom` compares them per event over every dts event:
efficiency (fit status > 0, and the CE selection: e-, downstream at TT_Front, fit consistency > 1e-3, tan(dip),
d0 and r_max windows; no t0 window, since OffSpill digitisation does not fold the times into the microbunch), reco
momentum at TT_Front minus the arm's own momentum at the tracker entrance (Gaussian core in -1..+1 MeV/c, mean,
tails beyond +-1 and +-2 MeV/c), momentum shapes in 0.1 MeV/c bins with paired per-bin pulls, active hits and
chi2/ndof.

## On the device: `sim/fieldsteps_jax` and the reco response (`recoresp`)

`sim/fieldsteps_jax.transport` is the device twin of `fieldsteps.transport` for electrons (`loss=electron_loss`,
`tail=False`), float64 or float32. Inputs and outputs stay device arrays: start states in, fixed-shape per-track
record buffers out (`[tracks, R_CAP]` sgsteps columns plus `valid`; `to_frame` gives the host table). Every cap is
counted in the returned statistics. With loss and scattering off it reproduces the NumPy rows to rounding (float64);
with them on it is statistically equivalent (independent random streams). Tables are built once per process
(`Tracker.from_gdml`); compiling the chord loop takes minutes on a GPU, so use the JAX persistent compile cache.

With `complete_outer_wall=True` the device wall material is the complete outer-minus-gas wall of every straw the chord
passes (`fieldwall.WallIndex`, wall-only chords included), summed over the straw window of each candidate panel; it
equals `WallIndex` to 1e-9 mm per chord in float64 (a few um on near-tangent chords in float32). On Run 1A DIO electrons (NumPy fieldsteps, 11k events, same entries)
the gas-gated default loses 6 % less momentum in the tracker than Geant4 (median 0.605 vs 0.650 MeV/c over the track);
the complete wall gives 0.636. The loss tail above 2 MeV stays lighter than Geant4's (7.4 % vs 9.1 % of tracks).

`recoresp` is a reco response layer for conversion electrons: `recoresp.table` measures, from a reference arm run
through stock LoopHelix reconstruction, the fit and fit-quality efficiencies and the residual distributions (momentum
at the front, tan(dip), d0, r_max, fit consistency, chi2/ndf, active hits; inverse CDFs with a Gaussian copula),
binned in the event's step rows and its entry tan(dip). `recoresp.device.apply` draws them per fast event on the
device and applies the CE fiducial cuts to the smeared values. The shipped table (`recoresp/data`) is built from
development events only; check agreement on independent events. It extrapolates only as far as its reference:
conversion electrons in the Run 1A tracker.

Opt-in empirical joint draw (`table.build(..., pools=True)`, shipped as `recoresp/data/ce_loophelix_run1a_pools.npz`,
`DeviceChain(response="pools")`): each bin also stores its pool of reference quality tracks, and the response draws
one reference track's full vector (momentum, tan(dip), d0, r_max residuals and the fit-quality outputs) instead of the
Gaussian copula and inverse CDFs. The copula keeps rank correlations but not tail dependence: a large momentum residual
of the stock fit usually comes with an off-cut d0, r_max or tan(dip) and fails the fiducial cuts. On Run 1A DIO
electrons (stock reconstruction of 200 dts files, response applied to the same events' Geant4 truth), the copula puts
+2 to +6 % more CE-selected tracks above 100 MeV/c than stock; with pools the ratio is 0.99-1.00 on both halves, and
above 95 MeV/c and overall it is unchanged.

Default response, full events (`recoresp/data/ce_loophelix_run1a_fulldts_pools.npz`, `DeviceChain(response="default")`
or `"fulldts"`; the conversion electron's own steps only: `"ceonly"`, with pools `"pools"`):
the same bins, built from 21 development dts files (59k events, three times the CE-only tables' sample), measured
with every Geant4 StrawGasStep of the event (delta rays, other secondaries, the calorimeter albedo) through stock
reconstruction, with pools. The device transport supplies only the
conversion electron and its albedo, so this table carries the effect of the other secondaries on the fit.

### Stock reco inputs: per-configuration tables, run-time cuts, knobs

The response is measured, not modelled, so a change of a stock reco input (conditions, straw status, fit settings)
needs a table measured under that input. `recoresp.config` describes the stock chain of a build:
- A JSON manifest holds the Offline area, geometry, DbService purpose and version, and the digitisation seed. For each
  stage it holds an include plus FHiCL override lines. It also lists the Geant4 sample, the table options, and any
  input files by name and sha256.
- `recoresp/data/config_run1a_offspill.json` is today's chain.
- Its hash is the first 16 hex digits of the SHA-256 of the canonical JSON.

Building a table:
1. `python3 -m trkfastsim.recoresp.config render MANIFEST OUTDIR` writes the three FCL files.
2. `tracker/scripts/resp_build_slurm.sh OUTDIR LIST OUT_ROOT` runs them on the dts files of LIST. The job directory is
   put first on `MU2E_SEARCH_PATH`, so DbService text-file overrides are found by name.
3. `table.build(events, pools="all", config=manifest)` stores the manifest and its hash in the table.

Using a table:
- `DeviceChain(response=<hash>)` takes the table from `recoresp/data/registry.json`. Each registered table is a GitHub
  release asset tagged `resp-<hash>`. It is downloaded into `$TRKFASTSIM_RESP_CACHE` (default
  `~/.cache/trkfastsim/recoresp`) and checked against its sha256.
- `DeviceChain(response_config=manifest or hash)` refuses any table built under other inputs.

All-track pools (`pools="all"`) keep every fitted track with its direction flag, so the quality selection runs at run
time. `DeviceChain(response_cuts={...})` (`table.with_cuts`) changes the CE cuts:
- tan dip, d0 and r_max as (low, high), with any table;
- the fit-consistency cut, only with all-track pools. Its valid range is [0, 1); a tighter cut leaves fewer pool
  tracks per bin.

A continuous knob is covered by a grid of tables: `table.at_knob({value: table}, x)`.
- Between two neighbouring tables it interpolates the efficiencies and inverse CDFs linearly, and resamples the
  pools as a mixture of the two.
- Outside the grid it refuses.
- The registered straw-threshold grid (TrkPreampStraw thresholds scaled at both ends) is 0.8, 0.9, 1.1 and 1.2, with
  the shipped manifest's table at 1.0. Its valid range is [0.8, 1.2].

Validation, held-out full Geant4 events (35k generated; ±0.30 pp on CE):
- Each configuration was built from the 7 development files and compared per generated electron, f64 / f32, with
  stock under that configuration.
- In all of them the tables built under the change reproduce stock within the common offset of about −0.4 pp, and
  the old tables fail.

| Change | Stock CE | Tables built under it: CE, fast − stock | Old tables |
|---|---|---|---|
| none (shipped manifest) | 35.80 % | −0.39 / −0.21 pp | — |
| 16 panels without HV | 35.62 % (fit −1.06 pp) | −0.56 / −0.40 pp, fit +0.29 / +0.37 | fit +1.19 / +1.33 pp, active-hit shape 285/35 |
| KKDe drift extension, last four stages at maxdoca 2 mm | 38.79 % | −0.42 / −0.38 pp | −3.32 / −3.22 pp |
| thresholds x0.8 | 34.15 % | −0.44 / −0.36 pp | +1.26 / +1.45 pp, fit chi2/ndf shape 1,199/18 |
| thresholds x0.9 | 35.39 % | −0.45 / −0.38 pp | fit chi2/ndf shape 194/18 |
| thresholds x1.1, interpolated 1.0-1.2 | 35.87 % | −0.59 / −0.46 pp, fit chi2/ndf shape 20/18 | fit chi2/ndf shape 72/18 |

Notes:
- Stock CE efficiency is curved in the threshold scale. Linear interpolation over a step of 0.2 misses it by about
  0.4 pp at 0.9, which is why the grid uses a step of 0.1. With that step the curvature implies an interpolation
  error of about 0.1 pp.
- Run-time fit-consistency cut (baseline all-track table), shift from the build cut 1e-3:

  | Cut | Stock shift | Fast shift f64 / f32 |
  |---|---|---|
  | 0.01 | −2.99 pp | −3.09 / −2.99 pp |
  | 1e-4 | +1.52 pp | +1.51 / +1.46 pp |

- The (step rows, tan dip) bins were enough for the dead panels; no spatial bin axis was needed.
- A change of the simulation itself (geometry, material, field) needs a new Geant4 sample, not only a new table.

## Open

- An unbiased reference for muons above 1 GeV/c and for side entries (the selection above).
- No world transport to the tracker; no device track fit (the reco response layer stands in for it).
