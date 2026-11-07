# CE whole-shower library (opt-in)

This module samples deposited patterns from saved conversion-electron `CaloShowerStep`s. It does not run Geant4,
change a chain default, or replace `beamfastsim.cesob`'s acceptance proxy. Entry energy/direction/radius are
**simulated incident-state inputs**, not truth-conditioned corrections to measured reconstruction.

## Build

The reader accepts the numeric `CaloLibraryDumper` export of a single-CE StageParticle sample: `dump/events`,
`generated`, `particles`, `products`, `steps`, `showers`, `crystals`, `disks`, and `virtualdetectors`. It selects the
first forward **direct-electron** disk-front VD crossing (73 or 75), matched by persistent particle identity.
The StageParticle upstream parent is reseated into the raw g4run product under the same key. Other input formats
need an explicitly selected adapter; matching by proximity is not used.

### Production-input adapter (one development file checked)

The isolated [`art/`](art/README.md) satellite extends the existing numeric exporter with explicit
`inputMode: "primary"` and `dump_library_primary.fcl`. It reads the saved `PrimaryParticle` vector instead of
requiring discarded StageParticles. No input-type auto-detection, key-only reseating or fake generated rows is
used. The default Stage path remains the pilot path above.

`read(path, input_mode="primary")` requires exactly one `mu2eCeMinusEndpoint` electron per stored event and joins
its exact `(run, subrun, event, ProductID, key)` identity into the configured compressed transport collection.
Only `SimParticle::selfParent()` continuations extend that direct electron; ordinary electron/photon descendants
are not incident candidates. Parent forests must resolve and be acyclic. Every non-null generator link must
resolve, but the CE itself need not have one: modern CeEndpoint has a stopped-muon parent. Counts, presence flags,
primary vector identities and every step/shower pointer are checked. Whole-event CSS is still retained, including
both disks and empty responses; CE-descendant versus other-ancestry CSS counts are reported, not used to drop rows.
The parent traversal is vectorized pointer doubling, not a Python loop over events or particles.

For the first post-build **development-file** schema check, use the saved-only command, not the library builder:

```bash
PYTHONPATH=calo python3 -m calofastsim.showers.export TABLES.root \
  --input-mode primary --out production_input_check.json
```

Following the user build, a 10-event smoke and one complete development-file extraction passed. An independent
raw-art check matched all **2,813 stored event identities and 115,272 ordered CSS rows** exactly, including pointer
identities and every decoded CSS state field. It exposed a two-byte ProductID offset error in `artsteps`, now fixed
and regression-tested; response/reconstruction did not use that identity column. The production reader finds 2,396
direct-front entries, retains 56 empty-CSS events, and counts 389 shower-positive events outside direct-front scope.
Every CSS ancestor in this file belongs to the CE ancestry. These are input checks, not model closure results.

Original-release geometry inputs were used after checking the saved production configuration and relevant maker
changes; exported crystal frames and 9,221 saved front-VD crossings agree with the two-disk geometry. Details and
the `production_check --check-geometry` command are in [`art/README.md`](art/README.md). Synthetic tests alone would
not establish these runtime facts. This is not a full-material or every-field round-trip certificate.

Filtered production DTS supports only a **stored/selected-population** comparison, not unconditional generated-event
acceptance. Reserved confirmation shards remain unused. No production library or physics validation is claimed.

```bash
PYTHONPATH=calo:core python3 -m calofastsim.showers TABLES.root \
  --out ce_showers_dev.npz --split-row 1000 \
  --source-commit EXPORTER_COMMIT --code-commit LIBRARY_COMMIT
```

The first 1,000 source event rows are development; the remaining rows are held out. The event split is chosen
before examining shapes, and is **not independent production-shard validation**. The builder writes an NPZ
without pickle plus a short JSON description. Nothing is renormalized to total incident energy: leakage is already
part of the sampled deposits. The whole event's CSS, both disks and empty patterns remain correlated.

Stored per-shower features are total entry energy, forward cosine, radius and disk. Default bin edges are
`[0,50,80,95,110] MeV`, `[0,.4,.7,1.000001]`, and `[0,450,550,750,1000] mm`. Each CSS retains its G4/Birks
energies and time/position relative to the entry time and momentum frame. Donors are uniform within a bin.
An empty bin uses the nearest occupied bin in bin-index distance **within the same disk**, with a returned
`fallback` flag. A disk without donors is explicitly unsupported, not a genuine zero-response draw.

## Frozen library on one production development file

A four-arm comparison uses the checked 2,813-event production DTS file: stock reconstruction, fast reconstruction
of its original CSS, and the unchanged pilot library with geometric or crystal-aware placement. The common shower
comparison contains **2,396 direct-front events** (both disks); all source rows remain in reconstruction and saved
outputs. The original-CSS control also covers all 2,813 rows. This development file was not used to train the frozen
505-entry pilot library; the reserved confirmation files remain unused. Legacy library metadata records source-file
provenance, not full parent-event identities, so cross-file parent independence is not certified.

Stock uses the existing Offline v13_39_00 runtime, matched to the implemented CAL1 response/reconstruction. Original
v13_41_01 geometry/search inputs are explicit; calorimeter geometry and condition tables match. This is **not a
comparison against current Offline head**: later digi clock-tick/time-offset changes are not silently mixed in.
Stock and fast arms use on-spill phase zero, the same pulse/noise/propagation/calibration inputs and production
0.5 ns reconstructed-PBT smearing. Stock response is independently stochastic. Donor seed 20261105 is shared by
both placement variants; fast reconstruction seed is 20261108. No model setting, library bin or donor was tuned.

| Conditional result | Original-CSS fast control | Geometric library | Crystal-aware library |
|---|---:|---:|---:|
| Selected above 10 MeV (stock: 1,727) | 1,722 | 1,590 | 1,619 |
| Selection ratio to stock | 0.9971 ± 0.0015 | 0.9207 ± 0.0073 | 0.9375 ± 0.0077 |
| Mean energy difference, zeros retained [MeV] | −0.135 ± 0.087 | −6.37 ± 0.54 | −5.29 ± 0.48 |
| Energy shape χ²/ndf | 22.3/19 | 178.2/19 | 122.3/19 |
| Truth-E/p shape χ²/ndf | 25.5/20 | 236.3/21 | 162.3/20 |
| Paired time difference, jointly selected [ns] | −0.001 ± 0.019 | +0.415 ± 0.138 | +0.357 ± 0.139 |

The full-stored-population original-CSS control gives 1,963/1,968 selected, ratio **0.99746 ± 0.00127**, and mean
energy difference **−0.115 ± 0.073 MeV**. It isolates the much larger shower-library discrepancy from downstream
response/reconstruction. Nonfinite fitted reco digis are flagged: 2 control, 4 geometric, 3 crystal; default
unimplemented-refit counts are 50/35/34. They are not silently removed from event denominators.

For the **950 entries below 450 mm radius**, geometric/crystal selection ratios are **0.821 ± 0.019 / 0.855 ± 0.019**,
with energy deficits **−12.39 ± 1.09 / −10.50 ± 1.01 MeV**. On the same 1,541 events selected in all four arms, mean
transverse separations from stock are **105.7 ± 4.4 / 105.9 ± 4.4 mm**, versus **1.52 ± 0.34 mm** for original CSS;
these are stochastic paired distances, not detector resolution. Radial/tangential marginal χ²/ndf are 67.3/24 and
70.0/29 (geometric), 66.5/24 and 77.0/29 (crystal), versus 8.4/12 and 12.8/11 for the control.

There are 73 fallback queries, zero unsupported disks, and 76 sampled empty donors. Unmapped Birks energy is
7.48% geometric versus 5.66% crystal. Crystal placement helps energy and truth-E/p here but neither variant closes;
the truth-E/p benefit is not uniform across studies (the earlier cross-fit result worsened). **Neither is ready for reserved-file confirmation or
proxy replacement.** Errors use paired 100-stored-row blocks and exclude library-training/sampling uncertainty;
no new stock seed was generated to estimate a stock-to-stock spread. These are filtered-population diagnostics.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.production_compare \
  --tables TABLES.root --library FROZEN_LIBRARY.npz --stock-ntuple STOCK.root \
  --reco-inputs CONDITIONS_DIR --stock-matched-phase0 --out comparison
```

The driver checks geometry, full stock event identity, source-row retention and explicit timing compatibility;
`--prepare-only` checks inputs and sampling without reconstruction. `fcl/stock_primary.fcl` runs only the production
calorimeter arm on compressed DTS, with no transport or output art. One CPU call per fast arm is a physics check,
**not a throughput measurement**; no speedup is inferred across the different stock/fast hosts.

### Placement loss versus donor mismatch (saved-input diagnostic)

`placement_diagnostic` reuses the same production queries, frozen donor draw and saved reconstruction. It checks
full IDs, entries, geometry, donor indices, support/fallback and offsets, and verifies original CSS maps back to its
own crystals. No reconstruction or sampling change is made. The original comparison did not record donor-payload
or placement-code hashes: matching draw indices is not retrospective certification of those payloads; current input
hashes are recorded by this diagnostic.

Let O be the original whole-event Birks CSS energy, D the donor energy before placement, M the mapped energy and U
the discarded energy. The exact per-query identity is **M − O = (D − O) − U**, with both disks and zero-energy events
retained. This is not an identity for reconstructed leading-cluster energy or selection efficiency.

| Mean Birks CSS quantity, 950 queries with radius <450 mm [MeV/query] | Geometric | Crystal-aware |
|---|---:|---:|
| Donor − original, before placement | −5.04 ± 1.22 | −5.04 ± 1.22 |
| Additional energy discarded in placement | 11.65 ± 0.65 | 9.08 ± 0.51 |
| Mapped − original | −16.69 ± 1.25 | −14.12 ± 1.20 |

The <450 mm stratum is not the physical hole boundary (336 mm). Only **0.960 ± 0.203 MeV/query** of either arm's
loss has raw positions inside that hole. Geometric placement loses another **10.686 ± 0.698** inside the radial/axial
envelope; this does not imply narrow crystal gaps. Crystal placement's dominant remaining loss is the transformed
center failing its distance bound, **7.985 ± 0.509 MeV/query**, plus **0.132 ± 0.033** from center/proposed-envelope
checks. Raw outer-edge losses are 0.0015 MeV/query; axial losses are zero. Categories are exclusive, with raw
inner/outer/axial precedence before the crystal guards. In the inner stratum, **7.27 / 6.59 MeV/query** of geometric/
crystal losses lie more than 10 mm from the geometric placer's four candidate boxes, measured at raw positions—not
crystal-remapping displacement. Relaxing a microscopic containment tolerance is not the indicated remedy.

The radius association is particularly strong. With donor radius more than 25 mm below the query (293 queries),
D − O is **−17.25 ± 2.79 MeV** and geometric/crystal losses only **1.51 / 1.22 MeV**. With donor radius more than
25 mm above the query (273 queries), D − O is **+5.56 ± 1.98 MeV**, but losses reach **31.57 / 27.19 MeV**. Within
25 mm (384 queries), losses are **5.22 / 2.20 MeV**. These are descriptive subdivisions, not a tuned donor criterion
or causal proof: the groups have different incident populations, empty-donor rates and fallback counts. The inner
stratum's azimuth-rotated donor-direction mismatch has median 23.1°, but its simple angle strata are not monotonic
in selection; radius and direction effects are not disentangled by this check.

Of 132 crystal-arm inner-radius misses relative to stock, 40 drew zero-energy donors and 34 had positive donors
with no discarded Birks energy; 58 had positive placement loss. There are also 39 gains, for a net deficit of 93.
Thus discarded energy cannot explain every miss. Zero loss does not exclude changes in occupancy, timing, response
or clustering; restoring lost energy is not a validated recovery counterfactual. The CSS>10 MeV gate is kept
separate from reconstructed selection. Raw donor centroid marginals already disagree before discarding deposits:
radial/tangential descriptive chi2/ndf **73.7/23 / 62.6/29** on the same 704 inner queries whose original, donor and
both mapped CSS totals exceed 10 MeV. This conditional subset is not the full population or a resolution test.

The frozen-draw energy accounting points toward **more local donor-radius matching**, not energy rescaling or
snapping deposits into absent detector cells. This diagnostic itself changed no model; the separately specified
candidate below tests that recommendation. Defaults and confirmation remain reserved. Paired 100-source-row errors
exclude library/draw uncertainty; D − O includes finite shower sampling and incident-population mismatch, not uniquely
donor-model bias.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.placement_diagnostic \
  --tables TABLES.root --library FROZEN_LIBRARY.npz --comparison comparison --out placement_diagnostic
```

### One radius-locality development candidate

A single predeclared **K=8** candidate restricts the donor pool to the nearest incident radii within the **same
selected disk and energy/cosine/radius cell**. It retains the existing closest-occupied-cell fallback, flags and
empty showers. Equal distances break by library row; eligible rows return to library-row order before applying the
same uniform draw schedule. Pools of at most eight reproduce the baseline donor exactly. No hard radial-distance
cut, direction conditioning, energy rescaling or placement change is made. `sample(..., radius_neighbors=8)` is
opt-in; the default zero preserves the original arrays, output schema and RNG behavior. This is incident-state
shower sampling, not a measured-variable response correction. K=8 is not an optimized value.

On the same 2,396 queries, using one shared candidate draw and the unchanged reconstruction seed, only the two
candidate arms are reconstructed. All stock/control/baseline outputs are reused. Input identities and current
hashes match the preceding diagnostic; current source/condition/draw fingerprints are recorded, without claiming
retrospective certification of the original runtime. No new stock seed or confirmation file is used.

| Metric | Baseline geometric → K=8 | Baseline crystal-aware → K=8 |
|---|---:|---:|
| Selected / 1,727 stock | 1,590 → 1,643 | 1,619 → 1,655 |
| Selection ratio | 0.9207 ± 0.0073 → **0.9514 ± 0.0073** | 0.9375 ± 0.0077 → **0.9583 ± 0.0075** |
| Mean leading-energy bias [MeV] | −6.37 ± 0.54 → **−3.59 ± 0.41** | −5.29 ± 0.48 → **−2.80 ± 0.39** |
| Energy shape chi2/ndf | 178.2/19 → 116.7/19 | 122.3/19 → 108.9/19 |
| Truth-E/p shape chi2/ndf | 236.3/21 → 157.8/21 | 162.3/20 → 138.8/21 |
| Inner-radius selection ratio | 0.8212 ± 0.0192 → **0.8958 ± 0.0166** | 0.8554 ± 0.0190 → **0.9098 ± 0.0164** |
| Inner-radius leading-energy bias [MeV] | −12.39 ± 1.09 → −7.05 ± 0.84 | −10.50 ± 1.01 → −5.81 ± 0.80 |
| Discarded Birks energy | 7.48% → 3.85% | 5.66% → 1.96% |

The paired candidate-minus-baseline selected fractions improve by **2.21 ± 0.54 / 1.50 ± 0.53 percentage points**
(geometric/crystal). These are net gains of 53/36; they include 97/86 newly selected and 44/50 newly lost events,
not one-way recovery. Most gains occur below 450 mm. Candidate disk-0 ratios are 0.9678/0.9720; disk-1 ratios remain
lower, 0.9159/0.9287. Above 550 mm the crystal ratio remains 0.9968, not an additional yield improvement.

Median absolute donor-radius mismatch falls **29.2 → 3.5 mm**, and **32.5 → 4.4 mm** in the inner stratum. Of the
2,396 donors, 1,971 change; fallback remains 73, unsupported zero, sampled empty donors 76 → 67. Unique donors
500 → 492 and maximum reuse 16 → 18 are reported, not treated as independent library statistics. Inner rotated
direction mismatch changes little (median 23.1° → 21.7°). At inner radius, donor-minus-original Birks energy remains
**−4.43 ± 0.92 MeV/query**, while placement loss falls to **5.61 ± 0.39 / 3.00 ± 0.35 MeV/query**. Thus the main
CSS accounting improvement is reduced placement loss, not closure of the donor-energy distribution.

On the same **1,505 events selected by all six arms**, transverse paired separation changes from **104.0 ± 4.4 →
95.7 ± 4.6 mm** (geometric) and **104.3 ± 4.4 → 95.0 ± 4.7 mm** (crystal); these are not detector resolutions.
Radial/tangential marginal chi2/ndf remain **51.9/25, 64.6/29 / 61.4/24, 55.4/29**, respectively. Jointly selected
mean time shifts relative to stock are **+0.340 ± 0.114 / +0.383 ± 0.118 ns**. Shape disagreement persists.
Of 98 remaining inner crystal-arm stock misses, 31 have zero-energy donors and 36 have positive donors with no
discarded Birks energy; 31 have placement loss. Forty gains leave a net deficit of 58. This is not proof that
restoring losses or changing direction would recover the remaining misses.

Native-backend nonfinite reco-digi flags are 0/1; skipped-refit counts 50/54. All source rows remain, and no faulty
fit is hidden by dropping an event. One compile-inclusive call per arm is **not a throughput measurement**. Paired
100-source-row errors omit library/draw uncertainty; changing donor multiplicity changes downstream random histories.
The existing stock reference is still Offline v13_39_00, not current head, and this remains a filtered population.
**K=8 is a useful development improvement, not stock closure or a promotion to confirmation.** Keep defaults fixed;
assess the residual donor/timing/shape mismatch before another model change. No additional candidate was scanned.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.radius_trial \
  --tables TABLES.root --library FROZEN_LIBRARY.npz --baseline comparison \
  --prior-diagnostic placement_diagnostic/numbers.json --reco-inputs CONDITIONS_DIR --out radius8
```

`--prepare-only` performs identity/hash/draw/placement checks without reconstruction. The driver has fixed K/seeds,
rejects overwriting an existing run, and saves the draw, leading clusters, paired reports and CSS accounting.

### Saved K8 residual diagnosis

`residual_diagnostic` reads the frozen draw and leading clusters; it does **not** sample or reconstruct. It checks
full identities, saved baseline arrays, timing, draw/placement fingerprints and prior CSS accounting. All 2,396
queries and both disks remain. The exact both-disk Birks partition into spatial loss, mapped/readout-rejected and
mapped/readout-accepted energy closes with zero residual. The lower readout gate uses the stock folding convention;
there is no added upper cut or second PBT subtraction. Energy beyond the nominal end is reported only as CSS
window exposure, not photon/waveform loss.

The clearest localized residual is **disk 1, entry radius <450 mm, incident energy ≥95 MeV**:

| Saved diagnostic | Result |
|---|---:|
| Queries / distinct selected donors | 190 / 38 |
| Stock / original-CSS fast control selected | 146 / 145 |
| Geometric / crystal selected | 117 / 117 |
| Either candidate's selection ratio | **0.801 ± 0.032** |
| Crystal mean leading-energy bias | **−11.87 ± 2.19 MeV** |
| Crystal misses / gains | 30 / 1 |
| Original / sampled empty CSS patterns | 1 / 23 |
| Original / sampled CSS energy ≤10 MeV | 1 / 32 |

The weak-pattern frequency comparison uses the **same incident-defined population**, not a donor-energy cut:
its excess is **16.32 ± 2.91 percentage points**. All 32 sampled weak patterns lie in the existing cosine≥0.7
stratum (109 queries): original 0/109 versus sampled 32/109; selection is 61/89, with 28 misses and no gains.
This is not an incident-energy scaling effect: mean donor-minus-query incident energy there is −0.080 MeV,
while donor-minus-original CSS energy is **−21.24 ± 4.35 MeV/query**, before placement. Only 25 distinct donors
serve those 109 queries; maximum reuse is 18, and there is no fallback. Training/draw uncertainty is not measured.

Across all queries, 82 of 120 crystal-arm stock misses occur with empty or positive-≤10-MeV donors (43+39).
However, **removing those donors would bias the model**: shower emptiness and leakage are physical outcomes.
Globally the original/sample weak-pattern counts are 93/137, with 110 donor-only weak and 66 original-only weak
queries. Misses and gains must both remain; neither these counts nor donor-conditioned near-unity ratios establish
causal recovery or unconditional closure.

Timing is not the principal explanation indicated by this lower-gate diagnostic: only **1/2** global
geometric/crystal stock misses coincide with a mapped-to-time-accepted CSS >10-MeV downcrossing, and **zero** do
in the 190-query region. Its jointly selected crystal/stock time difference is **+0.003 ± 0.057 ns** (116 pairs).
Downstream photon propagation, waveform windows and clustering are not ruled out. Direction trends are mixed
with occupancy/cosine populations: inner high-energy disk-0 angle-stratum ratios are approximately flat, whereas
disk 1 worsens with angle. This does not yet justify direction matching.

Sparse cells also matter descriptively: selected-cell pools ≤8 give a crystal selection ratio **0.698 ± 0.043**,
versus **0.992 ± 0.007** for pools ≥33; the populations differ. Energy/spatial shapes still disagree, no model
changes or confirmation files were used, and no throughput result follows from saved-only postprocessing.
The next bounded diagnostic is an exact census of the existing eligible pools in the localized weak-pattern
region, to distinguish a frozen-draw fluctuation from the pool's response mixture before proposing another model.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.residual_diagnostic \
  --tables TABLES.root --library FROZEN_LIBRARY.npz --trial radius8 \
  --baseline comparison --out radius8_residuals
```

### Exact eligible-pool census and larger-library planning

`pool_census` enumerates the **existing** K8 pools without sampling or reconstruction. Same-disk cell fallback,
cell/radius ties, support flags and every saved donor's membership are checked. It retains both disks, all 2,396
queries, empty donors and the original full-event identities. Full-cell occupancy is distinct from K8 eligibility.

For the previously identified **disk-1, radius <450 mm, E≥95 MeV, cosine≥0.7** population:

| Quantity | Result |
|---|---:|
| Queries / donors in the selected cell and eligible union | 109 / 25 |
| Donor empty / weak (whole-event Birks CSS ≤10 MeV) patterns | 6 / 9 |
| Original empty / weak query patterns | 0 / 0 |
| Saved draw empty / weak counts | 23 / 32 |
| Exact K8 expected empty / weak counts | **22.0 / 34.0** |
| Conditional draw SD, empty / weak | **3.10 / 1.91** |
| Queries whose entire K8 pool is weak | **22** |

Thus the saved weak count is only 1.05 conditional draw SD below expectation: **the observed excess is not an
unlucky draw**. The frozen pool mixture itself predicts the discrepancy. This does not establish the population's
true weak probability or measure training uncertainty. Conditional SD is `sqrt(sum(p_i*(1-p_i)))`; independent
uniform choices remain conditionally independent when pools overlap, but reused donors do not become independent
training events. Expected distinct selected donors are 24.53, versus 25 actually selected. Full-cell uniform
sampling would predict 39.24 weak patterns, not an argument for increasing K or a reconstructed counterfactual.

Coverage is genuinely thin: **16 of the 40 queried cells have no donors**, and 29 have at most eight. The 73
fallback queries all substitute another occupied cell; none is outside the configured bins. In the problem cell,
the median farthest eligible radius distance is 17.7 mm, and its 90th percentile is 36.7 mm. Weak donors have
median radius 360.7 mm and radial direction component −0.131, versus 398.8 mm and +0.249 for the query population;
these are correlated incident-state coverage diagnostics, not a prescription to cut weak showers.

**Selection compatibility is an outstanding prerequisite, not merely a sample-size caveat.** The saved production
configuration and its released `DetectorStepFilter` implementation accept a qualifying tracker particle with at
least 12 steps **OR** a per-particle qualifying Birks sum >=45 MeV **OR** a total qualifying Birks sum >45 MeV.
Tracker step ionizing energy must exceed 1e-6 MeV with momentum strictly between 50 and 1e6 MeV/c; calorimeter
rows require positive Birks energy and momentum strictly between zero and 1e6 MeV/c. Grouping uses the exact
SimParticle, all PDGs are eligible, and no time cut or CRV route is active. This is not a 45-MeV calorimeter cut
on every stored event. The pilot configuration disables that detector filter; its conditional shower population
need not match selected production even within coarse incident bins. More donors alone cannot resolve that confound.

After establishing a compatible inclusive source and independent training/development split, an initial
**100 independent donor events per currently queried cell** is a useful coverage target: about 4,000 in those
40 cells, requiring 3,499 additions if the existing donors remain admissible. This gives at best a 5-percentage-point
worst-case marginal binomial standard error per cell. Targets of 200/400 require 7,496/15,496 additions and give
3.54/2.5 points. These are planning scenarios, **not** local-K8 precision, generated-event counts or closure criteria.
Fixed K8 still uses at most eight outcomes at each query; more library events primarily improve locality/coverage.
Do not infer a Geant4 campaign size or runtime from these conditional counts. No data were pulled or generated.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.pool_census \
  --tables TABLES.root --library FROZEN_LIBRARY.npz --trial radius8 --baseline comparison \
  --residual radius8_residuals --out radius8_pools
```

### File-role manifests and upstream-event identity gate

`source_split freeze` validates complete catalog pages and explicit file roles, then saves DID/FID/size/checksums
with input/source hashes. It does not choose files by event count, inspect detector data, sample donors or certify
independence. A bounded candidate contains ten training, ten development and ten reserved confirmation files;
all are file-disjoint. The bounded training/development/legacy-pilot upstream-event audit is now complete below;
confirmation remains unopened. Catalog counts alone do not establish donor coverage.

Production saves `TargetStopResampler::Primary` as an EventIDSequence containing the immediate upstream stop-event
IDs; the opt-in numeric exporter now retains it and has been checked on one complete development file.
Shared stop-file provenance alone neither proves nor excludes event overlap. The required conservative grouping key is the canonical original stop-file identity plus
source run/subrun/event: keep all muons and resamples from one stop event together. Copied subsets must retain
original provenance; a new path or checksum is not a new physical source. This is not original-proton independence.

`source_split overlap` validates explicit provenance ledgers against separately enumerated full output-event
censuses, checks source presence/cardinality/resolution and reports repeated/shared parents. Missing or ambiguous
provenance is an error, not an empty response. FIDs must already be resolved against the verified source manifest;
the structural checker cannot authenticate that mapping. It does not drop collisions or certify independence,
and its expected-census equality does not itself prove the census came from a complete input file. The completed
audit independently checked full EventAuxiliary censuses and catalog SHA256 for the allocated files. Confirmation
event contents remain unopened, and no library has been rebuilt from the proposed files.

```bash
PYTHONPATH=calo python3 -m calofastsim.showers.source_split freeze \
  --plan PLAN.json --catalog CATALOG_RECEIPT.json --out CANDIDATE_MANIFEST.json
PYTHONPATH=calo python3 -m calofastsim.showers.source_split overlap \
  --left TRAIN_LEDGER.npz --right DEVELOP_LEDGER.npz \
  --expected-left TRAIN_EVENTS.npz --expected-right DEVELOP_EVENTS.npz --out OVERLAP.json
```

The analyzer-only **opt-in source EventIDSequence extension is user-built and runtime-validated** on the already-used
2,813-event development file. `exportSourceEvents=false` leaves the previous schema and lookups unchanged.
Separate production/pilot FHiCL files opt in with explicit process tags. Source-presence/ProductID/count fields,
ordered integer source IDs and registry kind -6 feed `source_events`, requiring an independent full ordered census.
The actual branch is `art::EventIDs_TargetStopResampler__Primary.`. All 2,813 valid singleton source IDs match a
separate stock EventIDSequencePrinter run bound to full per-event ArtReports. `source_event_check` confirms all
218 old columns across 11 tables are unchanged, including jagged geometry neighbors (same-position NaNs equal).
An independent raw-art reconciliation also confirms all 115,272 CSS rows and the complete output event census.

The 2,813 distinct immediate source EventID tuples are only a within-file diagnostic: no original file identity or
stopped-muon key is inferred, and all file-resolution flags in the raw numeric validator remain false. This exporter
validation alone is **not** a training/development/pilot overlap result, library closure or inclusive acceptance.
The separately completed source-file binding and provenance-only overlap audit follows below; no library change
was made. See [the schema and runtime check](art/README.md#opt-in-immediate-source-event-identities);
confirmation remains unopened.

### Completed bounded original-parent audit

The predeclared production allocation is ten training files (sequencers 1,101,...,901) and ten development files
(0,100,...,900), plus all 2,000 previously exposed legacy-pilot events. Reads were provenance-only: stock source-ID
printer, complete independent EventAuxiliary census, source configuration and catalog SHA256/identity checks.
File0 reused its completed printer output. The legacy copied stop pool was tied back to its original source file
using copy configuration/log/hash evidence, pilot input receipts and actual source IDs equal to the independently
decoded copy census—not an assumed sequential-row mapping.

| Set | Files | Stored events | Distinct immediate parent events |
|---|---:|---:|---:|
| Proposed production training | 10 | 28,265 | 28,265 |
| Production development | 10 | 27,944 | 27,944 |
| Legacy pilot exposure | 1 | 2,000 | 2,000 |

Using **original stop-file identity + run/subrun/event**, training/development share **0** parent events and
training/pilot share **0**. Development/pilot share **165**, all in development sequencer **200**; **40** of those
parents occur among the frozen 505 donors. File0 shares **0** with the full pilot, so this particular dependence
does not affect the existing file0 K8 comparison. All 210 file pairs were checked; 209 have measured zero overlap.
There is no within-set parent reuse in this allocation. No rows were removed or reassigned.

Consequently the ten-file development allocation cannot be described as parent-disjoint from the old pilot/library.
The new production training allocation passes this immediate-parent disjointness gate against both development and
legacy exposure. This is **not original-proton independence**, confirmation certification, shower coverage, or an
inclusive-acceptance result. Keep the pilot and new selected-production training pool separate; any later pool-growth
study needs its own frozen selection/coverage criteria. No library/model changes or reconstruction were performed.

`source_audit` validates metadata, hashes and independently decodes the full census, then binds stock-printer output
to the verified single original parent. `source_pilot_audit` validates copy provenance and the frozen donor identity
subset. `source_overlap_report` checks all complete bundles and reports shared families without pruning. Bound
summaries authenticate generated ledger/census files by SHA256; jointly replacing both arrays without updating their
upstream receipt fails. Original proof authentication remains an explicit prerequisite, not an inferred property.

### Training-only selected-production coverage census

The fixed ten training files were subsequently read with the user-built analyzer only, retaining all **28,265**
stored events. All transport statuses were zero. Every numeric output/source identity matched the completed saved
parent audit; all **1,158,498** raw CSS rows matched independently in identity and state. Saved GeometryService
settings matched the verified production reference, using its original release include closure, and both disks'
crystal frames/front-VD crossings passed the coordinate checks. These are not new transport or material validations.

Unchanged energy/cosine/radius bins and both disks define **96 cells**. Whole-event response includes CSS in both
response disks; empty and weak showers remain included. No-front events are reported separately, never encoded as
empty front responses. Out-of-bin entries would be reported separately rather than clamped into coverage; none occurred.

| Coverage quantity | Frozen legacy donors | Selected-production training |
|---|---:|---:|
| First-forward direct-CE front entries | 505 | **24,120** |
| Occupied cells / 96 | 25 | **55** |
| Cells with more than 8 entries | 11 | 40 |
| Cells with at least 100 / 200 / 400 entries | 1 / 0 / 0 | 18 / 14 / 10 |
| Empty whole-event CSS responses | 17 | **259** |
| Weak whole-event Birks response ≤10 MeV, including zeros | 33 | **835** |

Training adds 30 occupied cells and loses none of the legacy occupied cells. Entry disks 0/1 contribute
16,753/7,367 front entries and occupy 28/27 cells. There remain 41 empty cells and 15 occupied cells with at most
eight entries; not every fixed-grid cell is necessarily physically reachable. Occupancy thresholds describe coverage,
not effective sample size, local K8 precision or development-query-weighted support.

Of the 4,145 stored non-front events, 3,877 have positive CSS. All stored events contain 527 no-CSS responses,
whereas the front subset contains 259. Front entries retain 1,034,261 CSS rows; 115,240 are in the disk other than
the entry disk, and were not discarded. No energy, time or weak-response veto was introduced.

The previously diagnosed **disk1, E≥95 MeV, radius<450 mm, cosine≥0.7** cell has **1,243** training entries,
including **1 empty and 3 weak** responses, versus 25 legacy donors with 6 empty and 9 weak. This is a substantial
change in available training support, **not evidence of reconstruction closure or the effect of sample size alone**:
selected production and the unfiltered legacy pilot differ in selection and simulation runtime. Incident-state bins
are simulation-library diagnostics, not a measured-variable correction model.

`coverage` computes the vectorized scalar census; `training_audit` checks complete stored/source identities and
saved receipts; `training_coverage` verifies externally frozen extraction/provenance hashes and writes only cell
arrays and a report. No candidate library was constructed, no donor drawn, no reconstruction run, and no reserved
confirmation event opened. K8, the frozen505 library and defaults remain unchanged. The known 165 development/pilot
parent overlaps (40 frozen donors) remain flagged without pruning. A separately authorized next experiment could
construct one production-only candidate, retaining every valid front entry including zeros, then diagnose support
on already exposed development input before proposing reconstruction. Inclusive generated acceptance remains open.

### Separate production candidate and deterministic K8 support

The subsequently authorized candidate packs **all 24,120 training front entries**, including **259 empty and
835 weak** responses, and all **1,034,261** attached CSS rows in both response disks. It contains no legacy-pilot
donors. The existing packer, bins and geometry are unchanged. Stable inventory-file/event order, complete stored
census, exact output/source identities and per-donor file/event provenance are retained; entry/scalar/offset fields
match exactly and relative position/time round trips agree within 1e-9. Packing is not a compiled software build.
The candidate is a separate artifact, not a replacement for the frozen505 library or any default.

The deterministic audit uses only the already-exposed development file: all 2,813 stored events, 2,396 valid front
queries. It enumerates the unchanged K8 pools without invoking the sampler or reconstruction. Every old-library
eligible member, pool flag/size, probability and radius/angle metric reproduces the saved census exactly. Historical
float-valued integer cell labels are checked for exact bounded integer values before comparison with int64 labels;
no identity or floating response tolerance is introduced. New training and the complete query file share zero
immediate parent events; this is not original-proton independence.

| Fixed-query support or CSS expectation | Legacy505 K8 | Production candidate K8 |
|---|---:|---:|
| Supported queries | 2,396 | 2,396 |
| Cell-fallback queries | 73 | **1** |
| Median worst eligible radius difference per query [mm] | 6.364 | **0.136** |
| Unique donors in the eligible-pool union | 505 | 12,863 |
| Expected empty responses (original CSS: 28) | 73.925 | **26.250** |
| Expected weak responses ≤10 MeV (original CSS: 93) | 143.575 | **82.625** |
| Mean expected whole-event CSS minus original [MeV] | −2.530 | **+0.055** |

These are **exact conditional uniform-pool expectations, not a new draw or reconstructed observables**. Conditional
empty/weak draw SDs are 4.345/6.820 for the new candidate, excluding training, model and population uncertainty.
The paired source-row-block error on the new mean CSS difference is 0.292 MeV; it also excludes library uncertainty.
Of 40 queried cells, 39 have direct training support; 2,303 queries lie in cells with at least 100 donors. Two
queries use four eligible donors each; the other 2,394 use eight. The one fallback is disk0, 50≤E<80 MeV,
cosine<0.4, 550≤radius<750 mm; it uses the unchanged nearest occupied-cell rule, not an added exception.

For the previously diagnosed **109-query disk1/E≥95/r<450/cosine≥0.7** region, the new eligible union contains
600 donors, **none empty and one weak**. No empty donor was removed from the candidate: the cell's one empty
response simply is not among these queries' nearest-eight pools. The weak donor appears in two pools, each with
probability 1/8. Thus expected empty/weak counts change from **22/34 to 0/0.25** (original CSS 0/0); weak draw SD
is 0.468. Median worst radius difference improves **17.711→0.328 mm**. Expected CSS-minus-original changes
−24.925→+0.757 MeV (new conditional source-block error 1.197 MeV), still before placement, timing and response.

Direction mismatch is not solved: the median per-query mean azimuth-rotated angle changes only **18.72→18.03°**
globally and **15.84→11.16°** in that region. No direction conditioning, K scan, energy rescale or response veto was
introduced. Donor exposure and expected unique draws are not effective sample sizes. The selected-production versus
unfiltered-pilot/runtime difference remains a confound; this is not a sample-size-only causal result, reconstruction
closure or inclusive-acceptance validation. No confirmation input was opened, and the development/pilot overlap
flag remains unchanged. A separately authorized next step can make one fixed draw for the two existing placement
variants and run their CPU reconstruction, reusing all completed stock/control/legacy results without tuning.

`production_candidate` creates the authenticated pack and donor ledger; `support_audit` is the pure deterministic
pool calculation; `candidate_support` binds the frozen candidate, full query provenance and historical pool census.
All outputs are new files. No sampling, reconstruction, simulation, default change or library promotion occurred
in this step.

### Fixed production-candidate reconstruction trial

A separately predeclared trial used **one** production-candidate K8 draw (seed 20261105), shared by the existing
geometric and crystal placements, and **exactly two CPU reconstruction calls** (seed 20261108). It reused every
completed stock/control/legacy/K8 reference. There was no library repacking, new physics file, seed/K scan, direction
conditioning, response veto, energy rescale, default change or confirmation access. All 2,813 stored rows are saved;
comparisons use the same 2,396 front queries, including zeros. Nonfront sampled zeros are bookkeeping, not an
inclusive acceptance prediction. Critical sources, calibrations, CPU/x64 versions and the native 15-iteration
fitter match the preceding K8 trial; this remains a historical-stock, not current-Offline-head comparison.

| Full-front result (stock selected: 1,727) | Legacy K8 geometric | Production geometric | Legacy K8 crystal | Production crystal |
|---|---:|---:|---:|---:|
| Selected >10 MeV | 1,643 | **1,714** | 1,655 | **1,729** |
| Selection ratio | 0.9514±0.0073 | **0.9925±0.0065** | 0.9583±0.0075 | **1.0012±0.0070** |
| Mean energy difference, zeros retained [MeV] | −3.586±0.414 | **−1.803±0.420** | −2.805±0.388 | **−0.753±0.404** |
| Energy shape χ²/ndf | 116.67/19 | **24.14/19** | 108.92/19 | **23.74/19** |
| Truth E/p shape χ²/ndf | 157.78/21 | 75.80/23 | 138.80/21 | 88.68/23 |

The original-CSS control is unchanged: 1,722 selected and −0.135±0.087 MeV mean energy difference. Relative to the
previous K8 draw, net gains are 71/74 events: geometric gains 122 and loses 51; crystal gains 119 and loses 45.
Against stock, the new arms still have 62 misses/49 gains and 54 misses/56 gains. Near-unity totals are not eventwise
agreement. Errors use paired original stored-row//100 blocks; they exclude training/model uncertainty and do not
measure a stock-seed spread. Histogram statistics are descriptive diagonal paired statistics, not nominal p-values.

In the existing **109-query disk1/E≥95/r<450/cosine≥0.7** region, stock/control select 89/88; both previous K8 arms
selected 61, while the new geometric/crystal arms select **86/88**, with three/one stock misses and no gains. Their
mean energy differences are −4.675±2.370 / +0.390±1.726 MeV. The draw contains no empty or weak donors there, versus
old 23/32; this follows changed eligible pools, not a veto. The broader 190-query region selects 144/146 versus 146
stock. Across all 950 inner-radius queries, selections rise 576/585→630/644 versus 643 stock; new mean energy biases
are −2.964±1.045 / −0.700±0.999 MeV. Both disks are retained; crystal disk0/disk1 ratios are 0.9992/1.0055.

The actual draw has 33 empty and 83 weak responses, versus fixed-pool expectations 26.25±4.34 and 82.625±6.82
(conditional draw SDs), and original 28/93. Whole-event donor CSS minus original is −0.291±0.371 MeV/query.
Geometric/crystal placement discards 2.385/0.586 MeV/query, or 2.851%/0.700% of donor energy. Mapped-minus-original
biases are −2.676±0.393 / −0.876±0.378 MeV. The exact per-query identity donor = spatial loss + mapped timing reject
+ mapped timing accept closes to zero. The float32 lower timing gate is unchanged; nominal late-window CSS exposure
is not a measurement of photon/waveform energy loss. The sole fallback remains in the results and misses stock
selection in both new arms; it was neither repaired nor excluded.

On the **same 1,469 events selected by all eight arms**, mean transverse separation from stock changes from
93.31/92.62 mm for previous K8 to 83.82±4.24 / 82.89±3.76 mm; the original-CSS control is 1.55±0.36 mm. These are
stochastic paired distances, not detector resolution. Jointly-selected mean time shifts are +0.003±0.027 /
−0.005±0.030 ns, but time shapes are 17.47/15 and 31.00/16. Truth E/p remains discrepant and crystal placement does
not uniformly improve all shapes. The production-versus-unfiltered-pilot selection/release confound remains.
**Strong development improvement, not closure certification, replacement promotion or confirmation readiness.**

`candidate_trial` authenticates saved inputs and refuses any existing output, persists the draw/placements before
reconstruction and each arm immediately, and never reconstructs a reference. `--validate-only` performs no draw;
`--report-only --execution-sha256 SHA` reads a completed execution, with a fresh `--report-out` for partial-report
recovery. `candidate_report` is saved-only. Synthetic tests use mocked calls; optional saved-artifact verification
must not invoke sampling or reconstruction. The two recorded call durations include compilation and are not a
throughput comparison. No additional reconstruction is needed to investigate the remaining saved residuals.

### Saved conditional-response diagnosis

A saved-only follow-up (`candidate_residual`) examines the same fixed trial without sampling, placement or
reconstruction. All 2,396 front queries and both disks remain; energy and truth E/p include zeros. Spatial/time
statistics distinguish each arm's own E>10 selection from a jointly selected pair. Conditional moments use original
stored-row//100 delete-block errors, including blocks with no selected entries. All 96 fixed cells are reported;
56 have no queries and another 27 have fewer than 20, so sparse-cell statistics are not promoted to conclusions.

**The strongest residual is a lost direction–centroid relationship, not a wrong overall width.** Let `u_r` be the
incident unit direction projected onto the disk-local radial axis. On each arm's selected population:

| Radial centroid quantity | Stock | Original-CSS control | Geometric | Crystal |
|---|---:|---:|---:|---:|
| Standard deviation [mm] | 70.705 | 71.263 | 75.102 | 71.315 |
| Correlation with `u_r` | 0.657 | 0.655 | **0.067** | **0.073** |

The geometric/crystal correlation differences from stock are −0.590±0.052 / −0.584±0.053. Joint-selected samples
show the same loss. It also persists within populated fixed cells: disk1, 95–110 MeV, cosine .4–.7, radius 450–550 mm
has 233 queries and **exactly the same 179 selected rows in all four arms**, yet stock/control correlations are
0.872/0.901 and geometric/crystal 0.016/0.024. This is not explained by selection migration or mixing the broad bins.
The centroid coordinates are correctly compared in the disk-front-face-local frame; entries are transformed from
the global frame. These are conditional-response distributions, not track resolution or eventwise shower accuracy.

The missing relationship is already present **before placement/reconstruction**: whole-event CSS radial-centroid
correlation with query `u_r` is 0.609 for original CSS but 0.097 for the saved donor draw; geometric/crystal placement
leaves it at 0.089/0.094. CSS centroids use their own positive-CSS denominators and are not substituted for leading
clusters. Current K8 matches radius inside disk/energy/polar-cosine cells, but does not match signed local radial and
tangential direction. The result motivates examining that missing conditioning, not rotating two-disk showers out
of the detector or correcting reconstructed output with truth.

For inward (`u_r<0`, 1,039 queries) versus outward (`u_r>=0`, 1,357) electrons, crystal leading-energy biases are
**+2.989±0.850 / −3.618±0.475 MeV**. The donor CSS mismatch already has the opposite-sign pattern,
+4.098±0.711 / −3.651±0.402 MeV. Crystal spatial-loss terms are only −0.395 / −0.732 MeV in these groups. This
localizes a conditional donor-response problem; the algebraic decomposition is not a causal recovery estimate.

The improved energy marginal does not imply E/p closure. Dividing each event by its own truth momentum and rebinning
reveals redistribution hidden by integrated energy: crystal E/p .80–.85 and .85–.90 bins have **−87±26.15 and
−89±28.31** entries relative to stock, while .95–1.00 has **+17±3.06**. The .85–.90 deficit includes −32 and −25
from the disk0/disk1 95–110 MeV, cosine .4–.7, radius 450–550 mm cells. All signed cell-bin contributions sum to the
global differences; regional χ² values are not summed. Flows are retained, with E/p>1.2 counts stock/control 12/12
and geometric/crystal 15/20. This diagnostic is not reconstructed track E/p or a nominal significance test.

Per query, the saved energy difference obeys
`F-S = (D-O) - spatial_loss - (mapped_time_reject-original_time_reject)
       + [(F-mapped_time_accept)-(C-original_time_accept)] + (C-S)`.
Here F/C/S are sampled/control/stock leading energies and D/O whole-event donor/original CSS. Dividing every term
by the same query momentum gives the E/p identity. Both identities close; the downstream term is a residual, not
a fitted response correction. No rescale or lower/upper timing-cut change is inferred.

**Priority update:** assess whether incident-direction-aware donor support can preserve the improved radius
locality, zeros and whole-event correlations. Only then define one bounded model test and broader development
validation. Do not spend a confirmation sample demonstrating this known conditional defect. Any candidate remains
opt-in; the frozen library, K8 defaults, previous trial and reserved files are unchanged.

### Deterministic signed-direction support feasibility

`direction_support` reads the frozen production library, saved K8 pools and completed trial entry states only.
There is no sampling, placement, reconstruction or additional physics file. It holds the existing chosen same-disk
cell fixed and compares two nested conditions: matching signs of local radial and tangential direction (zero is
nonnegative), then those signs plus an azimuth-aligned direction mismatch of at most 10°. Neither condition uses
shower energy or reconstructed outcomes. The 10° and 25 mm diagnostic boundaries are inherited, not scanned.

The strict existing K8 pools have a median of **2 direction-compatible donors**; **272/2,396 queries have none**.
Across each whole chosen cell, only five queries have no compatible donor and 60 have fewer than eight. A
nearest-up-to-eight compatible **support witness** reduces median pool-mean angle from **18.03° to 5.80°**, but
widens median maximum radius mismatch from **0.136 to 0.469 mm** and its 95th percentile from **2.98 to 18.51 mm**.
The worst mismatch is **107.40 mm**. Five zero-compatible queries explicitly retain original K8; unsupported disks
would remain unsupported. This witness is not a deployed policy, and its sparse-pool results are not eighth-neighbor
measurements. Compatible-only expectations stay undefined when there is no compatible donor.

| Population | Queries | At least 8 compatible donors within 25 mm |
|---|---:|---:|
| All | 2,396 | 2,284 (95.3%) |
| Disk 0 / disk 1 | 1,642 / 754 | 1,601 / 683 |
| Radius <450 / 450–550 / >=550 mm | 950 / 1,013 / 433 | 861 / 997 / 426 |
| Energy <50 MeV | 68 | 11 |
| Negative tangential direction | 28 | 0 |

All 592/233 queries in the two populous 95–110 MeV, cosine .4–.7, radius 450–550 mm cells have eight compatible
neighbors within 25 mm. Their witness maximum-radius medians are **0.273/0.412 mm**; the 109-query high-energy,
inner-radius disk1 group has median **0.881 mm**, maximum **4.539 mm**. Thus support is strong precisely where
saved conditional correlations failed, but it is not uniform in the low-energy/negative-tangent tail.

Whole-event both-disk outcomes remain intact. The uncapped witness has 12,809 eligible donors versus 12,863 for K8;
maximum expected donor exposure is 1.5 versus 0.75 (not effective sample size). Exact conditional expected empty/weak
counts are **26.875/79.418**, versus K8 **26.25/82.625** and original CSS **28/93**. These describe the witness,
**not** a radius-capped candidate or reconstructed closure, and were not used to select support.

**Single next candidate proposal, not executed:** retain the chosen cell and disk-axis shower orientation; use
nearest-eight sign-and-10°-compatible donors only when at least eight exist within 25 mm, otherwise retain original
K8 with an explicit direction/locality-fallback flag. That would use compatible support for 2,284 queries and keep
K8 for 112, including all 28 negative-tangent queries. The cap bounds the *new direction branch*, not the unchanged
fallback branch; it deliberately relaxes the original nearest-eight radius envelope. No guarantee of conditional
closure follows, especially in fallback regions. Any subsequent one-draw/two-placement reconstruction test needs
its own bounded execution plan; confirmation and all defaults remain untouched.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.direction_support \
  --candidate production_candidate --support production_candidate_support \
  --trial production_candidate_trial --out NEW_DIRECTION_SUPPORT
```

### Fixed capped-direction trial

The single candidate above was subsequently tested on the same exposed file: **one draw and exactly two CPU
reconstruction calls**, reusing stock/control and all earlier references. `direction_sampler` is a separate opt-in
module; `library.sample`, binning, placements, fitter, calibrations and defaults are unchanged. The fixed 10°/25 mm,
minimum-eight rule uses direction matching for 2,284 queries and original K8 for 112. All fallback donor identities
match the frozen production draw with the same RNG schedule; downstream response can still differ when the overall
input changes. No additional file, seed/parameter scan or confirmation was used.

**The main conditional defect improves substantially, but the model does not yet close.**

| Radial centroid–direction correlation, own E>10 | Stock | Production K8 | Capped-direction |
|---|---:|---:|---:|
| Geometric | 0.657 | 0.067 | **0.591** |
| Crystal | 0.657 | 0.073 | **0.596** |

Paired correlation changes versus production K8 are **+0.524±0.033 / +0.523±0.033** (geometric/crystal).
On each arm's identical three-way new/K8/stock selected subset, the new correlations are **0.625/0.627**, versus
K8 **0.060/0.058** and stock **0.655/0.655**; new-minus-stock differences are −0.030±0.043 / −0.028±0.044.
These are conditional-distribution diagnostics, not detector resolution. Radial own-selected widths are now
82.98/79.57 mm versus stock 70.70 mm, with differences +12.27±6.38 / +8.86±6.51 mm: correlation alone is insufficient.
Tangential correlations improve only to 0.272/0.264 versus stock 0.350; disk1 and fallback residuals remain.

The high-statistics fixed cells support the interpretation. In disk0 cell41 the radial correlations change
0.009/−0.038 to **0.904/0.894** (stock 0.815); in disk1 cell89, **the same 179 selected events in all arms** change
0.016/0.024 to **0.757/0.741** (stock 0.872). The latter still has a deficit. Whole-event donor CSS already recovers
radial correlation **0.611**, versus original CSS 0.609 and the prior donor draw 0.097; crystal placement gives 0.602.
This supports improving the missing incident conditioning rather than fitting a reconstructed-output correction.

| All 2,396 queries; zeros retained | Production K8 geometric / crystal | Direction geometric / crystal |
|---|---:|---:|
| Selected E>10 (stock 1,727) | 1,714 / 1,729 | **1,698 / 1,724** |
| Mean leading energy minus stock [MeV] | −1.803 / −0.753 | **−1.797±0.409 / −0.325±0.402** |
| Energy shape χ²/ndf | 24.14/19 / 23.74/19 | 23.35/19 / 22.19/19 |
| Truth-E/p shape χ²/ndf | 75.80/23 / 88.68/23 | **76.52/24 / 69.34/23** |

Against K8 the selected-count changes are −16/−5, or −0.00668±0.00445 / −0.00209±0.00361 of all queries.
Crystal has 49 misses and 46 gains relative to stock; near-unit totals are not eventwise agreement. Its inward/outward
energy biases change from **+2.989/−3.618** to **−0.287±0.755/−0.355±0.350 MeV**. The donor CSS mismatch likewise
changes from +4.098/−3.651 to +0.029/−0.095 MeV. However disk1 mean energy shifts −0.737±0.443 MeV versus K8, and
truth-E/p still has shape residuals (crystal .85–.90 bin −83±21.53 entries). No nominal p-value or closure claim follows.

The 112 fallback queries retain weak direction coupling; all 28 negative-tangent queries are in this category.
Fallback geometric/crystal selection remains 54/61, identical to K8 counts, without a reconstructed-identity claim.
Do not discard these queries or treat successful matching in the other 95.3% as full-population validation.

Crystal placement loss is **0.423±0.058 MeV/query (0.504% of donor energy)**, versus K8 0.586 (0.700%); geometric
loss is 2.456±0.157 (2.926%), versus 2.385 (2.851%). The exact donor/spatial/timing partition and leading-energy/Ep
accounting identities close. Actual empty/weak donors are **36/88**, versus capped-pool expectations **29.625/85.375**
and original CSS 28/93; no outcome veto. Conditional draw SDs are 4.20/5.96 counts and are separate from block errors
and training uncertainty. One crystal digi has nonfinite fit uncertainties; positive leading-cluster fields are finite,
and all events remain. Single-call timings include compilation and are not throughput measurements.

`direction_trial --validate-only` authenticates without drawing or reconstruction; ordinary execution refuses any
existing output. It persists each arm and permits externally pinned, saved-only reporting into a new destination.
Reports explicitly label their inherited `k8` key as **production K8**, not old505 K8. Direct conditional comparisons
use paired original-row blocks and identical masks when comparing joint new/K8/stock statistics.

**Development checkpoint:** retain this exact candidate, not a promoted default. The saved-only follow-up below
localizes the remaining discrepancies without changing settings or making another physics call.

### Saved-only localization of the direction candidate

`direction_localization_run --trial SAVED_DIRECTION_TRIAL --production SAVED_PRODUCTION_K8 --out NEW_REPORT`
reads only authenticated saved NPZ archives. It retains all 2,396 queries, zeros and E/p flows, all 96 fixed cells,
and all 112 fallback queries. Additive signed-bin count errors use the full 29 original stored-row blocks, including
blocks with zero contribution to a cell. Regional chi-squares/errors are not additive. Spatial statistics separate
own selections, identical new/K8/stock three-way selections, and a separate four-way intersection including control.
One control-unselected query is retained in the primary three-way population, not silently removed or given a centroid.

**The E/p peak discrepancy is mainly in direction-matched, middle-radius queries.** For crystal placement, the
.85–.90 bin deficit is **−83±21.53 entries**: −79 from the direction-used branch and only −4 from fallback.
The 450–550 mm radius band contributes −71; fixed cells41/89 contribute **−41±13.69 / −17±8.58**. Both disks
contribute (−56/−27). Fixing fallback coverage alone therefore cannot close this shape. The .90–.95 excess is
+48±12.98; stochastic migration between bins is substantial. On the fixed stock .85–.90 population, the mean E/p
bias is −0.0550±0.0059, improved from K8 −0.0935±0.0067. Its five saved accounting terms sum exactly, but conditioning
on a stock outcome makes this diagnostic—not a causal donor correction, energy-rescaling prescription or resolution test.

**The radial width is tail-sensitive.** Crystal has 62/1,724 own-selected queries beyond |radial|200 mm, versus
57/1,727 for stock; these contribute 70.48% versus 63.99% of each arm's full variance about its own mean.
On the identical 1,648-query three-way population, radial widths are 71.15/66.90/67.96 mm (new/K8/stock), with
new-minus-stock **+3.20±7.60 mm**. Disk1 has just one sampled query beyond 200 mm, already present in K8 fallback,
and none in stock. Its contribution dominates the disk1 width excess; it remains in every full-population result.
This does not establish tail closure or justify deleting the query.

**Disk1 tangential width is not merely a fitter-level issue.** On its identical 526-query leading-cluster population,
crystal widths are **39.63/35.98/29.74 mm**, and correlations 0.253/0.248/0.438 (new/K8/stock).
The new-minus-stock width is +9.89±6.14 mm. On the distinct 722-query common positive-CSS population, donor tangential
widths are already **52.00/48.02/38.68 mm** (new/K8/original), a new-minus-original excess of **13.33±4.57 mm**.
A nearly restored donor correlation does not ensure a correct width; CSS and leading populations are not interchangeable.

**Fallback remains a separate sparse-support limitation.** Of112 queries,15 have no capped-compatible donor and97
have one to seven; all28 negative-tangent queries fall back. Empty donors are9 versus expected4.875 with conditional
draw SD1.850; weak donors24 versus23.625 with SD3.466. These conditional draw uncertainties are distinct from paired
block errors and do not include library/model uncertainty. No reseeding, donor veto or sparse-population pruning follows.

The report has76 synthetic tests and an independent saved-output verifier covering all96cells, all117spatial regions,
all13CSS regions and all125fallback regions, including literal original-block errors. No new drawing, placement,
reconstruction, ROOT input, threshold scan or default change occurred. These are exposed-file development diagnostics,
not confirmation, inclusive acceptance, detector resolution or validated replacement performance.

**Next priority:** stop tuning this file and predeclare a small, fixed additional-development check of this exact
candidate, targeting E/p peak cells, disk1 tangential widths and full-population tails/fallback. Reuse the saved file0
references, retain both disks and all sparse groups, and leave confirmation closed. Additional files and physics calls
require a separately bounded execution plan; none were opened or run in this follow-up.

## Apply and response hand-off

`sample(library, entries, rng, orientation="disk")` rotates a whole donor pattern about the disk axis and translates
it to the query entry. It preserves longitudinal deposit positions relative to the entry, so the second disk does
not rotate out of the detector. Exact incident-direction rotation is also available as `orientation="incident"`;
it is unsuitable for transporting a two-disk pattern in a fixed detector. Neither placement rescales energy.

`place` associates the transformed deposits with crystal boxes using their saved geometry, and returns the crystal
ID, crystal-local position, global position and an explicit `mapped` mask. It checks up to four nearby boxes and
never snaps an outside deposit into a crystal. Unmapped deposits are a model loss, reported separately. Crystal
size comes from the exported geometry-service cell dimensions; wrapper/active-volume differences are not resolved.

`cluster_sum` is a temporary **one-cluster-per-entry energy-weighted sum**, not a fitted CaloCluster. It accepts
calorimeter corrected energies and time-acceptance masks. The saved comparison imports the calorimeter `response.py` explicitly,
uses the saved per-event timing and per-crystal LRU, and applies the same deterministic response to both arms:

```bash
PYTHONPATH=calo:core python3 -m calofastsim.showers.compare \
  --library ce_showers_dev.npz --input TABLES.root \
  --response-input RESPONSE.root --response-source calo/calofastsim/response.py \
  --out comparison --arm heldout --seed 20261105 \
  --digitization-start 450 --digitization-end 1795 --digitization-buffer 10
```

Those timing values belong to this saved control; supply the actual values for another input. The script produces
an energy overlay with paired pulls, `numbers.json`, and numeric cluster arrays. Errors use paired jackknife over
100-source-event blocks; the shape chi-square uses diagonal paired block variance and no fitted normalization.
The shared library's training uncertainty is not included. The E/p entry uses **true disk-front momentum**, and is
labelled a truth diagnostic, not reconstructed E/p. Package dependencies are NumPy, SciPy, uproot and matplotlib;
the hand-off additionally needs `mu2efastsim`.

## cechain / cesob adapter

```python
from calofastsim.showers.handoff import from_cechain
out = from_cechain(result, field, library, rng)
```

`result` is cechain's Result including absolute times. The adapter retains its generated-event denominator,
propagates tracker-entry states with the same midpoint-field helix stepping used by cesob, selects the first forward
disk-front entry, and returns query states plus sampled, placed CSS for the calorimeter response. Existing cesob NPZs omit times and
are not silently assigned fabricated times. No beam file is modified.

**This is a diagnostic adapter only:** it does not transport tracker/passive material or side-entry interactions.
Full generated-event comparisons and timing are below; their denominators include non-front-entry events.
The proxy remains unchanged.

## Pilot result and current limit

A 2,000-event stock pilot supplies 1,042 direct-electron front entries: 505 development (20,336 CSS, 17 empty
showers) and 537 held out. Of 1,195 shower-positive events, 189 have no direct-electron front entry; they remain in
the census but are outside this library's scope.

On the held-out front-entry population, sampled vs original CSS through the same deterministic response and simple
cluster sum gives:

| Quantity | Result |
|---|---|
| Acceptance above 10 MeV | ratio **0.9896 ± 0.0201**; difference **−0.745 ± 1.435 percentage points** |
| Mean energy difference | **−4.67 ± 1.46 MeV** |
| Energy shape, 5 MeV bins | **χ²/ndf = 52.1/21** |
| Empty-bin fallback | 16/537 entries; no unsupported disk |
| Visible energy outside mapped crystals | **10.0%** |

The energy shape is not adequate for proxy replacement. Independent production-shard confirmation and
side/photon entries remain open. Do not tune on these held-out results; any revised model needs a new confirmation
sample. The following full-chain measurement uses the same frozen model, not a post-confirmation retune.

## Full generated-event measurement

`prepare_chain` extracts the exact generated electron starts, full-event stock visible/response energies,
per-event timing and (optionally) stock leading clusters. `measure_chain` runs CE primary transport, the disk-front
adapter, frozen showers and response; it separately evaluates the cesob inner-wall proxy and its vacuum variant
on those same tracker-entry states. `reconstruct` connects the sampled CSS to the calorimeter digi/template-fit/
hit/cluster chain. No source Geant4, digitization or reconstruction reference is regenerated.

For the **1,000 held-out generated CE events**, double-precision device transport followed by the host disk-front
adapter and the frozen library gives:

| Selection | Fast | Stock | Paired fast/stock ratio |
|---|---:|---:|---:|
| Visible deposited energy >10 MeV | 543 | 566 | **0.9594 ± 0.0222** |
| Timed deterministic response sum >10 MeV | 422 | 441 | **0.9569 ± 0.0204** |
| cesob inner-wall proxy vs visible-deposit selection | 566 | 566 | **1.0000 ± 0.0139** |
| cesob vacuum proxy vs visible-deposit selection | 576 | 566 | **1.0177 ± 0.0157** |
| Reconstructed leading-cluster energy >10 MeV | 390 | 410 | **0.9512 ± 0.0224** |

The proxy targets visible-deposit acceptance and **must not be compared directly with the timed cluster selection**.
The new shower visible acceptance is 2.30 ± 0.78 percentage points below the wall proxy on identical fast states.
For reconstructed clusters, the all-event mean energy difference is −1.47 ± 0.82 MeV and the energy shape χ²/ndf
is 28.96/19. On the 356 events where both arms pass, the time difference is +0.899 ± 0.332 ns and the mean paired
transverse position separation is 251 mm. The large positional disagreement is not a good resolution or validation
result. There are 34 fast-only and 54 stock-only selected events. Independent fast/stock stochastic response and
one finite development library are used; the errors exclude library-training uncertainty.

**Measured complete-route throughput:** 1,000-event batches, one cold and two identical-seed warm calls, from
CE birth through reconstructed clusters. CPU and GPU use the same double-precision JAX transport and retain the
same acceptance counts. Device synchronization, host propagation, shower placement, response, digitization,
fitting and clustering are included; input loading and first-call compilation are separate.

| Allocation | First call | Median warm call | Generated CE/s | Stock grid-slot rates (23.9 CE/s) |
|---|---:|---:|---:|---:|
| One CPU core, JAX CPU | 134.44 s | **97.672 s** | **10.24** | **0.428** |
| RTX PRO 6000 1g.24GB MIG + four allocated host cores | 35.68 s | **14.162 s** | **70.61** | **2.954** |

Warm CPU transport/downstream times are 46.43/51.24 s; MIG transport/downstream 10.84/3.32 s. The wall proxy is a
separate comparison and not included in the new shower-chain timing. The fast route is primary-only, omits
tracker/passive material and fails the shower validity requirements; the stock timing covers the full production
CE simulation stage, not a matched reconstruction workload. These are rate comparisons, not validated replacement
claims or measurements of large-batch steady-state device saturation. Legacy NumPy transport did not finish its
first measurement before it was stopped; no rate is inferred from that partial run.

Reconstruction uses the calorimeter source at `656bd9b` with its saved timing snapshot; transport uses umbrella
`7bb64cd` and core `d041bb3`. The shower implementation is published in this PR series; the measurements used its frozen pre-publication snapshot.
No defaults changed. Package/CLI naming and test-file organization do not change the measured physics.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.prepare_chain \
  --tables TABLES.root --response RESPONSE.root --response-source calo/calofastsim/response.py \
  --sim-crystal Sim_crystal.txt --geometry geometry.resolved.txt --stock-ntuple STOCK.root \
  --out inputs --digitization-start 450 --digitization-end 1795 --digitization-buffer 10
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.measure_chain \
  --inputs inputs/chain_inputs.npz --library ce_showers_dev.npz --geometry inputs/geometry.txt \
  --maps DSMap.header TSdMap.header --response-source calo/calofastsim/response.py \
  --out chain --arm heldout --backend device --device gpu --repeat 3 \
  --digitization-start 450 --digitization-end 1795 --digitization-buffer 10 \
  --umbrella-commit UMBRELLA_SHA --core-commit CORE_SHA
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.reconstruct \
  --chain chain/chain.npz --chain-report chain/numbers.json --inputs inputs/chain_inputs.npz \
  --library ce_showers_dev.npz --maps DSMap.header TSdMap.header \
  --reco-code calo/calofastsim --reco-inputs RECO_INPUTS --reco-commit RECO_SHA \
  --live-transport device --transport-geometry inputs/geometry.txt --out complete --device gpu
```

The last command repeats the same transport/draws to time the full route and checks exact saved-state equality.
It does not draw a new held-out sample or choose a candidate. Omit `--live-transport` to reuse the saved states;
that mode reports a **sum of separately measured stages**, not a single-call end-to-end wall time.
`chain_report` renders saved summaries without re-executing held-out physics.

Tests: `PYTHONPATH=calo:core python3 -m pytest -q calo/tests/test_showers*.py`.

## Development replay with the merged reconstruction backend

`develop_reco` replays saved, five-fold cross-fit shower tables through the current calorimeter backend, using its
native template-fit cache and 15 iterations. It does not install the frozen-backend fit-cache adapter. The two
stock ntuples are explicitly selected, aligned on all full run/subrun/event identities, then restricted to the
first 1,000 development rows. The 505 direct-front queries are the reporting denominator. Other rows enter the
backend with no CSS: this is a **conditional stock-CSS control**, not a full-source acceptance measurement.

The default arms are original stock CSS, baseline sampled CSS and crystal-remapped sampled CSS. Each has one
cold and two warm calls with identical seeds; leading energy/time/x/y/z must match exactly across repeats,
including missing-value NaNs. Conditional row rates are not labelled generated-event throughput.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.develop_reco \
  --steps DEVELOPMENT_CROSSFIT_STEPS --inputs inputs/chain_inputs.npz \
  --reco-code calo/calofastsim --reco-inputs RECO_INPUTS \
  --stock-ntuple STOCK_SEED1.root --stock-ntuple2 STOCK_SEED2.root \
  --device gpu --repeat 3 --save-clusters --out development_replay
```

The saved cross-fit tables exclude each query's 200-source-row fold from its donor pool. This replay does not
resample showers or inspect the old held-out population. With the merged backend and updated stock timing:

| Development arm | Selected / 351 stock | Acceptance ratio | Mean energy difference [MeV] | Energy chi2/ndf |
|---|---:|---:|---:|---:|
| Original stock CSS through fast reconstruction | 350 | 0.9972 ± 0.0028 | +0.049 ± 0.103 | 13.59/18 |
| Baseline sampled showers | 324 | 0.9231 ± 0.0184 | −4.24 ± 1.27 | 35.72/19 |
| Crystal-remapped showers | 335 | 0.9544 ± 0.0166 | −3.30 ± 1.02 | 19.34/19 |
| Independent stock reconstruction seed | 351 | 1.0000 ± 0.0000 | +0.009 ± 0.072 | 17.86/17 |

The zero observed stock-seed acceptance error means no selection changes in this finite sample, not zero
population uncertainty. The paired errors above omit library-training uncertainty and cross-fold dependence.
CPU and MIG replay selections agree. The stock-CSS arm's leading arrays are exact across devices; one sampled
cluster per variant differs in energy by at most 1.91e-6 MeV and in transverse position by less than 8e-7 mm.
Same-device repeats are exact. The baseline sample flags one nonfinite reco digi; the crystal and stock-CSS arms
flag none. These flags are retained, not hidden by the selected-event totals.

### What the broader observables show

`validity_report` reads saved clusters only. It compares both-disk/radial strata, incident-relative radial and
tangential centroid distributions, time relative to the **folded** incident time, and truth E/p. Reconstructed
cluster time already subtracts reconstructed PBT; do not subtract MC PBT again from the folded incident state.
These incident-relative quantities are simulation diagnostics, not track resolution or reconstructed E/p.

- The deficit concentrates below 450 mm entry radius: baseline ratio **0.816 ± 0.043**, crystal **0.904 ± 0.041**;
  mean energy deficits **−10.25 ± 2.09** and **−8.18 ± 1.57 MeV**. The middle radial bin has ratio 1.000 ± 0.010
  in both variants. Both disks remain in the study.
- On the same 308 events selected by every arm, mean transverse fast/stock separation is **105.6 ± 9.8 mm** for
  baseline and **106.0 ± 9.9 mm** for crystal placement; stock-CSS and stock-seed controls are about 1.1 mm.
- Crystal placement improves the total-energy shape but **worsens the truth-E/p shape**, from **35.0/19** to
  **63.7/19**. This is not an across-observable improvement or a promotion decision.
- Paired time shifts are **+0.247 ± 0.345 ns** (baseline) and **+0.276 ± 0.341 ns** (crystal). The incident-folded
  time-shape chi2/ndf values are 7.78/11 and 5.51/10; out-of-range counts remain in the saved diagnostics.

Crystal placement is explicitly opt-in through `from_cechain(..., placement="crystal")` and
`reconstruct --placement crystal`. It moves donor crystal centers within half a cell diagonal and preserves
sub-crystal depth, energy and time; raw/proposed positions outside the detector and missing cells remain losses.
The default is `geometric`; direction matching is not included in this candidate.

Independent existing production DTS inputs require a reader for their compressed primary/GenParticle identity,
not the pilot's StageParticle-parent convention. Filtered stored events cannot establish unconditional generated
acceptance. No new independent confirmation, reconstructed-E/p validation, or stock replacement is claimed here.

### Complete-route development profile

`run_profile.sbatch` uses the first 1,000 generated rows only, prepares the same-seed transport once, then measures
one cold and two warm complete calls for each placement. Geometry/maps/calibrations remain outside the timer;
synchronization, transport, host disk-front propagation, sampling/placement and native reconstruction are inside.
The library contains development donors: these runs are timing workloads, **not unbiased physics tests**. Their
selection counts must not replace the cross-fit validation above. No old held-out draws are rerun.

| Allocation | Geometric cold / warm [s] | Geometric CE/s | Crystal warm [s] | Crystal CE/s |
|---|---:|---:|---:|---:|
| One CPU core | 124.83 / 75.450 | 13.25 | 67.229 | 14.87 |
| RTX PRO 6000 MIG 1g.24GB + four host cores | 31.18 / 9.272 | 107.85 | 9.400 | 106.39 |
| A100-SXM4-40GB + four host cores | 56.90 / 9.015 | 110.92 | 9.078 | 110.16 |

These are finite-batch measurements, not device-saturation rates. GPU comparisons process identical events and
leading E/time/x/y/z arrays agree exactly between the two GPU types. CPU geometric output is also exact; one
crystal-arm cluster differs by 3.82e-6 MeV and less than 9e-7 mm, with identical selected-event masks. Same-device
repeats match exactly. The raw outputs retain zero events and nonfinite-fit flags.

For geometric placement, median transport/hand-off/reconstruction times are **53.19/3.76/18.50 s** on CPU,
**7.88/0.91/0.48 s** on MIG, and **5.84/2.35/0.82 s** on A100. Faster A100 device work is offset by slower host
stages. The CPU arms ran sequentially: unchanged transport shifts from 53.19 to 40.98 s while crystal reconstruction
is actually slower (22.97 s). Do **not** attribute the apparent CPU total-rate gain to crystal placement.

The 23.9 CE/s stock grid-slot basis would give geometric rate ratios 0.555/4.512/4.641, but it covers different
physics and stage boundaries. They are **not validated replacement speedups**. These development profiles also
cannot be divided by the old held-out frozen-backend rates to infer an optimization gain.

`profile_report --run CPU=CPU_DIR --run MIG=MIG_DIR --run A100=A100_DIR --out profile.json` checks matching inputs,
settings, event rows and complete-call timing before summarizing saved results. `reconstruct --expect-clusters`
requires exact event rows and all leading observables against a caller-selected saved same-device reference.
`--stock-ntuple` explicitly selects an updated stock reference using full event identity; it never silently changes
an existing report.

The next performance experiment should target transport/output extraction or empty candidate work while holding
lane scheduling, random keys, precision and stepping fixed. Changing batch/compaction parameters changes the random
history and requires a separate physics comparison; neither the calorimeter fitter nor the beam transport is
silently altered here.
