# CRV step-layer campaign: opt-in placement and shower-source diagnostic

## Decision

Keep the placement refinements **opt-in**, with default sampling unchanged. Context plus endpoint containment
removes the large reconstructed-cluster y distortion; point placement additionally reduces the step-level layer
pattern. Mirror/translation provide no compelling further improvement. This is **not full tier-1b closure**:
other-bar descendants remain deficient, and the shower-source change is a diagnostic, not a certified replacement.

No source compilation is required. The working tree is based on CRV `1041b86`; measurements use core `a554571`
from the existing umbrella checkout. The response comparison predates the opposite-side changes on newer CRV main;
it must not be presented as a validation of that merged response/reco version.

## Sampling options

`crvfastsim tier1b-apply` accepts `--crossing-context`, `--contain-steps`, `--point-placement`,
`--mirror-offsets`, and `--translate-offsets`; all default off. The context option requires a library rebuilt
with the `libxing.ctx` column. Library momentum/path bins and the unchanged shower component stay fixed between
placement arms. Geometry is extracted v04, covering EX, T1, T2 and all taggers.

- Context conditions on layer rank and another module crossed within 300 mm upstream; cells with fewer than
  20 donors fall back to the original momentum/path bin. This is a generative simulation input, not a
  measured-efficiency selection. Origin/ancestry/context plots are truth diagnostics.
- Containment brings rotated same-bar endpoints inside their assigned bar. It preserves recorded deposit,
  path, momentum and time, so it is a placement approximation, not a new transport calculation.
- Point placement allows other-bar steps to land in a sector with another bar orientation.
- Mirror and translation retain nearest-bar fallback; neither models physical escape at module edges.

`valhist.steplayers` provides layer/origin breakdowns, step y/z scores and the same NumPy response on both step
arms. `valhist.stepclosure` adds source isolation, exposure/ancestry guards and a read-only shower identity audit.
Both modules run from the checkout with CRV and core on `PYTHONPATH`.

## Recovered placement results, 20 reference shards

The original layer excess belongs to muon descendants, not primary-muon transport. Pooling library crossings
across layers allows upward offsets from deeper donors to be placed on top-layer crossings; nearest-bar fallback
concentrates these deposits in the top layer and near module ends.

Step y diagonal deck scores (threshold 2; exact accumulated per-event second moments):

| Arm | EX | T1 | T2 | M | Sector y/z flags |
|---|---:|---:|---:|---:|---:|
| Original A1 | 53.08 | 9.5 | 13.3 | 2.8 | 6/8 |
| Context + contain | 3.95 | 5.2 | 2.8 | 2.5 | 6/8 |
| Context + contain + point (CCP) | 2.88 | 5.69 | 2.72 | 1.84 | 4/8 |
| All five options | 3.02 | 5.58 | 2.80 | 1.86 | 4/8 |

The five-option arm is not a closure improvement. CCP retains the production shower deficit and its total step
yield is low. No out-of-bar start positions above the 1-mm diagnostic tolerance remain after containment.

Saved NumPy-chain cluster y scores for original / context+contain / CCP are:

| Sector | Original | Context+contain | CCP |
|---|---:|---:|---:|
| EX | 7.56 | 1.16 | 1.47 |
| T1 | 1.63 | 0.99 | 1.24 |
| T2 | 1.65 | 0.58 | 0.66 |
| M | 4.26 | 0.69 | 0.35 |

These historical response reports clipped position flows into edge bins. They are retained and explicitly labeled
`legacy_clipped`; new response reports preserve ROOT flow bins and exclude them from visible-bin scores.

## Primary and tier-1b guards

CCP preserves **266,160 primary-muon steps exactly**, excluding only `nstep` (which counts added secondaries),
across all 20 shards. The existing V1 comparison passes every gate in EX/T1/T2/M: crossing changes
+0.163/+0.324/+0.147/-0.582%; mean path changes -0.01/+0.26/+0.16/-0.20%.
The V1 tool's 748,070 particle-union events are **not generated exposure**; this sample has one million generated
events. The held-out source experiment below separately checks its complete 500,000 generated identities.

On shards 11–20, CCP other-bar descendant energy per crossing is 0.085/0.104/0.069/0.034 MeV versus
0.101/0.154/0.105/0.079 MeV in G4 (EX/T1/T2/M). Therefore V3 is **not closed**. Its same-bar extra/muon energy
ratios agree at the few-percent level; nonmuon energy ratios are 0.918/0.875/0.864/0.962.

## Held-out shower-source experiment

Build showers on `g4ref_v1` shards 1–10 (500,000 generated events), test on shards 11–20 (500,000 events).
Only `libshower`, `libshowersteps`, and `libgen` are replaced together; all production descendant trees are kept.
No rate scaling is fitted. Full integer event tuples, highest cycles, source SHA256 and generated exposure are
recorded. Training and evaluation gen identities have zero overlap. An ancestry guard finds **all muon-family
step payloads identical** between CCP and the shower-source arm over the ten held-out shards.

| Held-out observable | Production showers | Reference-training showers |
|---|---:|---:|
| Total steps | 770,793 | 799,529 |
| Relative to 795,615 G4 steps | -3.12% | +0.49% |
| EX y / z score | 1.47 / 1.40 | 0.70 / 1.58 |
| T1 y / z score | 4.81 / 1.21 | 1.04 / 0.78 |
| T2 y / z score | 2.06 / 0.64 | 0.96 / 0.92 |
| M y / z score | 1.62 / 0.91 | 1.40 / 1.01 |
| Sector y/z flags | 2/8 | 0/8 |

Nonmuon-descendant step ratios to G4 change from -9.85% to +0.55% (EX), -13.84% to +3.24% (T1),
-11.13% to +3.33% (T2), and -4.81% to +32.99% (M, large statistical spread).
The source replacement removes the observed residual pattern on this held-out sample, but its 3,259 showers
are much fewer than the production library's 65,400. This is source sensitivity, not a finite-library uncertainty.

New NumPy response on both held-out step arms, ROOT flows kept:

| Sector | Cluster y / z score | Cluster count change (paired ratio SE) |
|---|---:|---:|
| EX | 0.65 / 0.56 | -0.05 +/- 0.33% |
| T1 | 0.90 / 0.12 | -0.83 +/- 0.88% |
| T2 | 1.27 / 0.63 | -0.47 +/- 0.70% |
| M | 1.19 / 0.76 | -2.88 +/- 1.21% |

All eight independent-arm deck scores are below 2; this is not a blanket DQM or efficiency pass. In particular,
M's paired z score is 5.52 (one occupied bin); its rate deficit remains visible in the paired statistic.

## Shower identity prevents promotion (historical pre-repair source study)

**Follow-up:** the production donor association is now repaired and strictly validated; see
[generator-association-repair.md](generator-association-repair.md). All 20 donors preserve legacy physics payloads,
with zero corrected generator mismatches or unsupported showers. The held-out replay improves the total step
count deficit from 3.12% to 2.38% but does not close the shower/position discrepancy (sector flags 2/8 -> 3/8).
The reference-training source study below remains historical and is not an identity-repaired replacement.


The read-only audit reuses `denominator_provenance.donor_showers` and `support`, verifies exact ordered shower
projection and the native reach formula, and does not delete, reassign or smooth any donor.

| Library | Showers | Primary/gen PDG mismatches | Negative native KE | Unsupported showers |
|---|---:|---:|---:|---:|
| Production context library | 65,400 | 279 | 24 | 61 |
| Reference-training source | 3,259 | 15 | 1 | 2 |

Every unsupported shower in both libraries has a PDG mismatch. No missing join occurs; 31/1 same-PDG showers
also have start/generated energy differences above 1 MeV. The persisted flat tables lack referenced GenParticle
ProductID, so the reference-training source is **not an identity-repaired library**. These defects do not, by
these counts alone, establish the cause or size of the ~10% nonmuon-descendant deficit. Recover collection
provenance, repair the association at the producer/dumper boundary, then rebuild and validate a separately
versioned library. Do not substitute true start energy or tune shower rates to force closure.

The descendant builder also has an inherited multi-muon limitation: crossings are grouped by event/bar and donor
attachment is not restricted to the ancestor muon. This change does not fix it; all new options remain experimental.

## Statistics and tests

The per-event sums are vectorized, with file/sector/depth loops only. `sum_e n_e^2` and `sum_e(c_e-r_e)^2` are
accumulated exactly, but are Poisson-bootstrap second moments, not centered fixed-N sample variances. The diagonal
scores omit cross-bin covariance and are not calibrated chi-square tests. Ratio SE uses
`sum_e(c_e-(C/R)r_e)^2 / R^2`, recovered from the stored paired moments; the earlier scaled-difference error was
incorrect as a ratio error. These estimates condition on a fixed library/conditions and exclude donor uncertainty.
The response RNG seeds match, but changing multiplicities does not imply event-aligned random streams.

Current campaign diagnostics fail closed outside run 1400 and chain-compatible event IDs. Exposure guards require
identical full generated identities for fast arms and reference-particle identities within that population.

The inherited full suite: 699 passed, 27 failed, 3 skipped. Of the failures, 25 are the old CRV/new core kit schema
mismatch, one is a pre-existing oneAPI mock without the `width` argument, and one is the new context requirement
missing from a synthetic geometry fixture. The fixture is updated; the final focused suite passes **42 tests**
(tier-1b, step closure and denominator provenance). With the unrelated
kit file and known oneAPI test excluded: 704 passed, 3 skipped, 1 deselected (before the last two diagnostic tests).
No unrelated test failure is hidden as a physics pass.

## Reproduction and retained evidence

Data root: `$M2FS_DATA/crv_step_layers_20261026/`.

- Saved placement products: `fast1b_ctx`, `fast1b_ccp`, `fast1b_ccpm`, `fast1b_all5`; each completed arm's
  `breakdown.json` / `posyz.json`, and `resp_{a1,ctx,ccp}_vs_a0g.s01-20.json`.
- New products under `takeover/`: `primary_guard.json`, `v1_ccp/`, `v3_ccp_s11_20/`,
  `lib_shower_g4train.{root,json}`, `ccp_shower_g4train/`, `posyz_heldout.json`,
  `ccp_heldout_breakdown.json`, `descendant_guard.json`, `exposure.json`,
  `response_shower_g4train.json`, `shower_{source,production}_audit.json`.
- `scripts/run_step_closure.sh source` performs the held-out source study; `guard` runs primary invariance,
  V1 and V3. The source command refuses an existing output library; use a new `STEP_LAYERS_OUT` for a new campaign.
- `python3 -m valhist.stepclosure --help` lists the standalone exposure, ancestry and shower-audit commands.
- `python3 -m valhist.steplayers clusters RESPONSE.json` and `report BREAKDOWN.json --views grp_cls_outside`
  summarize saved products without rerunning simulation.

Old baselines, libraries and outputs are preserved. No C++ build, grid submission, default promotion or merge
was performed. Integrate with current main and revalidate the response version before any production adoption.
