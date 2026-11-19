# CRV module design points (`crvfastsim.design`, workflow W9)

Opt-in package that scores a parametric CRV module design on the sector layout of a stock counters file (Run 2:
`crv_counters_v10`), for design optimisation (e.g. a Mu2eBO study). Nothing else in `crvfastsim` imports it, and no
default of the cosmic chain changes.

```bash
python -m crvfastsim.design muons GEN.root ... --out muons.npz            # CE-like proxy muons from CRY gen tables
python -m crvfastsim.design geometry --design '{"tilt_deg": 15}'           # build and summarise a design
python -m crvfastsim.design evaluate --design design.json --muons muons.npz --dtype float64 \
    [--lookup-dir DIR] [--calibration deadtime.json] [--kappa K] --out metrics.json
python -m crvfastsim.design closure --muons muons.npz --dtype float64 --lookup-dir DIR \
    --photon-map DIR/photonMap.root --mode mean|sampled --out closure.json    # optics model vs the full chain
python -m crvfastsim.design stockcmp STOCK.root ... --dtype float64 [--design D] [--gen GEN.root ...] \
    [--geometry-check] --out stockcmp.json                                    # paired comparison with stock
```

## Design knobs (`module.Design`)

| Key | Meaning | Default |
|---|---|---|
| `n_layers` | scintillator layers per module | stock (4) |
| `thickness`, `width` | counter cross-section [mm] | stock (19.80, 51.34) |
| `max_length` | longest counter [mm]; longer sectors are split into equal segments (`segment_gap` between them), each with its own SiPMs | 0 = stock lengths |
| `absorber`, `absorber_material` | passive absorber in every gap between layers [mm]; `G4_Al`, `MildSteel`, `G4_Pb`, `G4_POLYETHYLENE` | stock (9.525 mm Al) |
| `tilt_deg` | rotation of each module about its counter length axis, i.e. the angle between the module normal and the stock normal | 0 |
| `fiber_diameter` | WLS fiber [mm] | stock fiber of each sector |
| `counters_per_module` | counters per layer and module | stock (16) |
| `coincidence_layers` | k of the k-of-N layer rule (at most the layer count of every sector) | stock: N minus the stock spare layers (3 of 4, 4 of 4 where stock asks for all four) |
| `spare_layers` | instead of `coincidence_layers`: k = N - spare_layers in every sector (follows per-sector layer counts) | unset |
| `pe_threshold` | PE per layer and read-out end for a hit, every sector | stock threshold of each sector (8 PE in Run 2) |

The default `Design()` reproduces the stock bars exactly (`module.to_crv_geometry` against
`geometry.from_offline_config`, tested). Modules are re-tiled to cover the stock sector extent. Tilted modules are
louvers: the pitch is the projected module width plus the module gap, so every line along the stock normal meets a
module; at small tilts the pitch grows until neighbours do not overlap. In sector-local coordinates every bar is an
axis-aligned box, so crossings use the core slab method.

Tilt needs no change to Offline and no per-layout table: the response of a counter is read on the muon state in
the counter's local frame (entry and exit point, path), so a tilted module is the same local problem as an untilted
one seen by a rigidly rotated muon (tested). This is the same rule as the component libraries of the fast chain:
bin on the entrance state in the component's frame, never on a global detector configuration.

## Figures of merit (`fom.evaluate`)

| Metric | Definition |
|---|---|
| `inefficiency` (+ `_err`) | mean over the muon sample of P_miss = prod over sectors (1 - P_coinc); P_coinc = P(at least k of N layers have a read-out end with at least the sector's PE threshold), Poisson about the expected PE (same-plus-adjacent counter sum), k = N - (4 - stock required layers); muons that cross no counter count as missed (`uncovered_fraction`); per sector, each muon is assigned to its highest crossed sector |
| `deadtime_rate_ratio`, `deadtime` | rate model (below); `deadtime` only with a calibration file |
| `coincidence` | per sector, the rule used: "k of N, threshold PE" |
| `ly_margin` | min over sectors of the design's minimum light yield along its longest counter (better end, normal incidence, aged by `aging_c1`) over the nominal design's (aged by `aging_ref`) |
| `cost`, `cost_rel`, `cost_terms` | scintillator volume [m^3], channels, absorber mass [t], fiber length x diameter^2 [m mm^2], each over the nominal design; `cost` is their mean |

The muon sample (`muons`) is the CRY muons whose line passes through a cylinder about the detector-solenoid axis
from the stopping target to the tracker end (`fom.PROXY`), a geometric stand-in for the CE-like selection. With
`--world GDML` (`sample.py`) the muons are first carried through the world material above the CRV (the fast stage-C
transport: straggled loss, Highland scattering to the MSC plane) and start on that plane with their momentum there;
muons that range out above it are dropped. Inside the CRV each muon loses the mean (Bethe) energy in the designed
counters and absorbers in path order (`veto.stack_range`): crossings after it stops are dropped, and a muon that
stops before the CE-like region is left out of the inefficiency (it cannot reach the tracker; `stopped_fraction`).
The P_coinc of a sector also carries the finder's slope window and minimum cluster PE (`veto.finder_factor`).

Against stock Geant4 + CRV reco on a thick-absorber Run 2 sample (25.4 mm Al per gap, 1M CRY events), over CE-like
events whose stock muon reached the detector region: stock 0.115 % (14 of 12,176) against the model's 0.177 %, a
difference of -0.06 +- 0.05 %, consistent in momentum bins. With generated lines instead of CRV-entry states the
model was 8 times too low (muons that lose most of their energy above the CRV).

### Optics model (`optics.py`)

Expected PE per counter end for a chord: scintillation photons (yield x `photon_yield_scale` x Bethe mean loss x
path, spread along the chord) times the arrival probability of a stock lookup table read at the scaled point
(thickness and width scaled by the ratio of scintillator cores, length by the distance to the read-out end, the last
bin extended with the table's own attenuation length beyond its length), times a capture factor
(d / d_stock)^alpha_d x (P_table / P_core)^alpha_g (alpha_d from the stock 1.8 / 1.4 mm fiber yields; P the core
perimeter), times PE per arrival of the SiPM model and a calibration scale `pe_scale` fitted to the full chain at
the nominal design. A sector whose stock table is not in the lookup directory uses the closest available table of
the same reflector kind (`table_substitutions`). Cherenkov light and visible-energy fluctuations are not modelled.

### Experimental charge/time calibration

`python -m crvfastsim.objective.charge_time_calib matches.npz --out calibration.json`
fits corrections relative to a nominal response. Supply aligned arrays `channel`,
`distance_mm`, `pe`, `expected_pe`, `time_ns`, and `expected_time_ns` for matched
track/channel observations. Optional `charge_weight` and `time_weight` are
inverse variances in log-PE and time, respectively.

The fit profiles channel log-gains and time offsets with a zero-mean gauge,
leaving one shared slope each for inverse attenuation length and inverse signal
speed. Within-channel distance variation is required. Select positive,
unsaturated observations; the fit does not estimate missed hits, model
zero-suppression, replace `objective.timecal`, or update production conditions.

`OPENBLAS_NUM_THREADS=1 python crv/scripts/bench_charge_time_calib.py` uses 128
synthetic channels and 10,240 training / 2,560 held-out observations. Median time
on one AMD EPYC 9V74 core (NumPy 2.5.5) was 1.03 ms profiled versus 42.4 ms for
the equivalent dense least-squares fit, with coefficients agreeing to
8e-15 in the model's fit units. The two injected slopes were recovered within
0.008% and 0.11%; held-out RMS was 0.030 in log-PE and 0.102 ns. This is
synthetic closure, not calibration against detector data.

### Deadtime model (`deadtime.py`)

No beam-induced CRV simulation exists in the fast chain. Two rate components per sector carry a calibrated nominal
deadtime to a design: accidental k-of-N coincidences of counter singles (singles scale with counter volume, light
yield^beta and the intensity scenario kappa) and correlated multi-layer showers (scale with sector area, kappa and the
absorber transmission for 1 MeV photons). A higher PE threshold lowers the singles rate like a lower light yield
(the same exponent beta); a different k changes the accidental combinatorics. Deadtime = 1 - exp(-veto window x total rate). The calibration file
(`{"source": ..., "deadtime": {sector: fraction}}` or `{"total": fraction}`) holds the stock nominal deadtime; it is
an input, not part of this repository.

## Comparison with stock (`stockcmp.py`)

Input: the TFileService files of a stock job that ran the CrvStepTableDumper (`dump/`) and the CrvRecoTableDumper
(`crvdump/`) on every event, in the geometry of the design being checked. Paired over events, per sector: stock PE
(reconstructed pulses) over the model's expected PE, per layer and read-out end; the fraction of (event, layer) rows
with an end at or above threshold against the model's mean probability; the fraction of events with a stock
coincidence cluster against the mean P_coinc. `--geometry-check` places every stock CrvStep in the design bar of the
same index (1.0 = the stock geometry is the design's). `--gen` (dumps of every generated event) adds the paired
CE-proxy inefficiency; events the stock job dropped for having no CRV steps count as stock misses.

Reference numbers (stock Offline, Run 2 conditions, 3 GeV muons on a top module, 33,334 per angle): stock/model PE
0.847 +- 0.001 at 0, 15 and 30 degrees from the counter normal, while the PE itself grows by 16 %. The tilt
dependence of the model therefore agrees with stock to 0.2 %, and `--pe-scale 0.847` calibrates the absolute scale.
Every layer was hit at 15 and 30 degrees in both arms; at normal incidence the model misses 0.2 % more layers
(counter edges).

## Approximations (W9)

All fixed (no switch yet); the table in [`../../docs/approximations.md`](../../docs/approximations.md) lists each
with what stock does instead. In short: primary-muon straight lines (no secondaries, scattering or energy loss in the
CRV), scaled optics instead of new lookup tables, expected PE with Poisson statistics instead of the full response,
the k-of-N rule without the finder's time, slope and distance cuts, a geometric CE-like proxy, and a calibrated rate
model for the deadtime.

## Precision

Every array path takes `dtype` (`--dtype float32|float64`, required on the command line); means and the Poisson
probabilities are accumulated in float64.
