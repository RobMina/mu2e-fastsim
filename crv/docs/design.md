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
| `profile` | counter cross-section: `rect`, or `triangle` (interleaved prisms, below; needs `table_mode` native) | `rect` |

The default `Design()` reproduces the stock bars exactly (`module.to_crv_geometry` against
`geometry.from_offline_config`, tested). Modules are re-tiled to cover the stock sector extent. Tilted modules are
louvers: the pitch is the projected module width plus the module gap, so every line along the stock normal meets a
module; at small tilts the pitch grows until neighbours do not overlap. In sector-local coordinates every bar is an
axis-aligned box, so crossings use the core slab method.

Tilt needs no change to Offline and no per-layout table: the response of a counter is read on the muon state in
the counter's local frame (entry and exit point, path), so a tilted module is the same local problem as an untilted
one seen by a rigidly rotated muon (tested). This is the same rule as the component libraries of the fast chain:
bin on the entrance state in the component's frame, never on a global detector configuration.

### Triangular counters (`profile="triangle"`)

Isosceles prisms, base `width` along the gap direction, height `thickness` along the layer direction, one fiber at
the centroid, alternately apex up and apex down every width / 2 with no gap, so a layer is a solid slab of
scintillator and a track crosses about two prisms in each layer. A module holds twice the stock counter count by
default (the stock module width and channel count: one fiber per prism against two per rectangle); the placed
layout tiles with the prism module pitch ((counters + 1) x width / 2 plus the module gap). Bars keep their bounding
boxes for the slab crossings, the placement and the FEB boxes; `veto.clip_prisms` clips each chord to the prism.

The layer response is the existing same-plus-adjacent sum: the two prisms a track crosses in a layer are neighbouring
counters, and the finder adds the PE of a counter and its neighbours, so the layer PE per end is their sum (no
per-counter threshold, which would lose the light split between the two prisms). Cost: a prism has half its box's
scintillator, one fiber and one channel per read-out end. Deadtime: the prism volume and half-base area enter the
singles and correlated terms; the model is not calibrated for prisms (nor in k). Offline has no triangular counter:
the placed counters text describes the module boxes only.

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

`OpticsConfig(table_mode="native")` with `Context(native_lookup_dir=...)` reads the design counters' own tables
(`crvfastsim.lookupgen`, made for their cross-section and fiber, docs/lookupgen.md) at the local point itself: no
scaling, no capture factor; the sector yield is taken back to the 1.4 mm fiber the tables are normalised to, and
`native_model_scale` (default 1 / 1.164, released over generated light on the stock counter) puts the generator's
optical model on the released-table scale that `photon_yield_scale` and `pe_scale` were set with. The nominal design
keeps the stock tables (scaled mode). Missing lengths are substituted as above (`native_table_substitutions`; the
kit refuses only stock substitutions). Native mode is the only one that reads triangular counters (table frame: x
from the centroid towards the apex, |y|, one fiber); their light-yield profile is the layer crossing midway between
neighbouring fibers (half the height in each prism, the two summed), where the rectangle's is the width centre,
between its two fibers.

### Deadtime model (`deadtime.py`)

No beam-induced CRV simulation exists in the fast chain. Two rate components per sector carry a calibrated nominal
deadtime to a design: accidental k-of-N coincidences of counter singles (singles scale with counter volume, light
yield^beta and the intensity scenario kappa) and correlated multi-layer showers (scale with sector area, kappa and the
absorber transmission for 1 MeV photons). A higher PE threshold lowers the singles rate like a lower light yield
(the same exponent beta); a different k changes the accidental combinatorics. Deadtime = 1 - exp(-veto window x total rate). The calibration file
(`{"source": ..., "deadtime": {sector: fraction}}` or `{"total": fraction}`) holds the stock nominal deadtime; it is
an input, not part of this repository.

## Placement in the detector envelope (`layout.py`, `envelope.py`, `placement.py`)

By default `fom.evaluate` re-tiles every sector on its stock position (`module.build`) and ignores everything around
the CRV. With `FomConfig(placement=PlacementConfig(...))` (kit: `place_policy`) the optimisation loop places each
design itself, then scores the placed geometry:

1. **Layout** (`layout.py`, the rules of the G1 option-B references, every constant from the counters file): the
   no-overshoot tiling per sector (counters per module within 6 of stock) with one-module end-strip *filler*
   sectors `<parent>F`, the scaled `layerOffset`, length segments `segment_gap` apart (2 x (CMBOffset +
   CMBHalfThickness) + 2 mm) with the joints staggered by `stagger_mm` per layer, FEB boxes scaled to the module
   pitch. Fillers take their parent's response, cuts, coincidence group and deadtime terms.
2. **Envelope** (`envelope.py`): ExtShieldUpstream boxes and ExtShieldDownstream extruded blocks built as Offline's
   makers build them (half-lengths, tolerances, orientation codes; blocks split into convex prisms, holes and notches
   solid), the floor (`yOfFloorSurface.below.mu2eOrigin`) and the counters file's support structures. The borated
   polyethylene boxes (ExtShieldUpstream types 15-19 on Run 2) are the movable sheets. Not modelled: the hall
   services (pipes, trays).
3. **Placement** (`placement.py`): each sector (with its segments and filler) is the union of its Geant4 mother
   parts and moves along its layer direction by `-shift`; it treats the envelope, the sectors that face it along its
   layer direction in the stock geometry and the sheets it does not push as hard, and takes the shift that needs the
   least sheet displacement, then the smallest |shift|. Sheets are pushed away from the CRV by the sectors on their
   stock CRV side (never toward it), grow away from the CRV when thicker, must clear the envelope and each other by
   `sheet_clearance` (1 mm) and the other sectors by `clearance` (3 mm), or the stock gap if smaller; a sector that
   reaches a sheet only through its re-layout cuts the sheet's end back by up to `max_trim`. Sectors whose FEB boxes
   hit anything drop them. The joint stagger is the largest candidate that places.
4. **Remedies**, when the base placement does not fit: a move of named sectors (e.g. `move:topmove:T1,T2,T3:0,0,1:50`,
   the top sectors downstream by up to 50 mm) or a thinner sheet (`thin:thinU:Outer CRV-U:10`, down to 10 mm). Each
   takes the smallest amount that places. Policy `each` scores the base placement, or every remedy on its own when
   the base does not place, so the effect of each remedy on inefficiency, deadtime and cost is in
   `placement_variants`; the result is the feasible variant with the lowest inefficiency. Policy `first` stops at
   the first that places.

`eval.json` records the chosen placement (`placement`: shifts, moves, FEB-less sectors, stagger, every sheet's move,
thickness and trims, remaining violations, the `shield_overlay` lines for a stock reference) and every variant;
`crv_counters_placed.txt` is the placed counters text a stock reference is built from. A variant that thins or trims a
sheet sets `deadtime_shield_uncalibrated`: the deadtime model is calibrated on the stock shielding and has no shield
term, so such a deadtime is not a prediction until a shield-thickness study calibrates it. `placement.check` tests a
given (e.g. hand-made) placement against the envelope without solving.

**Default rules since 2026-11-24** (the first placed scores and the G1 Geant4 references used the old v3 rules,
reproduced by `LayoutConfig(split=True, close_up=False)` with `PlacementConfig(tie_chains=False)`):
- stock counter lengths: `Design.max_length` no longer splits sectors (`LayoutConfig.split`), so a placed design has no
  length joints the nominal geometry does not have;
- close-up: sectors that follow each other in stock (T1..T5, R1-R2, L1-L2, E1-E2; `layout.sector_chains`) start where
  their predecessor's modules end, so the no-overshoot end strips do not open cracks through the stack at the
  boundaries (they collect at the chain's last end);
- tied chains: such sectors share one shift, so the top plane has no step.

`python -m crvfastsim.design.tiling_plot` draws the top sectors seen from above (layers a vertical track crosses, module
outlines, zooms on the sector boundaries) for the nominal geometry or a design, placed or on stock positions.
**Coincidence grouping** (kit: `coinc_groups`). Empty (the default) searches every sector alone. `sector_type` is
Offline's CrvCoincidenceFinder grouping (`fom.sector_type_groups`): sectors of one `crs.sectorType` are searched
together, each hit with its own sector's cuts, and placed fillers join their parent's group. Scoring the design loop
on placed geometry means `place_policy` and `coinc_groups=sector_type` together. (The G1 crvd04 Geant4 references
were built on the v3 layout rules, before the default rules above, so they do not validate the default layout.)

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
