# Upstream beam prerequisites (experimental)

The default beam chain remains detector-solenoid entrance to stopping-target stops.
The tools below are opt-in reference preparation and vacuum development kernels,
**not full** PS/TS transport or a production-target yield model. They do not certify a hadron-production library,
transmission efficiency, Run 1A physics closure or a speedup.

**Paused source-only checkpoint:** [recording](TARGET_RECORDING.md) and
[original-product readback](TARGET_READBACK.md) proposals are uncompiled/uninstalled;
no verified inclusive target-exit library, qualified POT, causal launch frontier,
material-aware PS/TS closure or upstream throughput result is supplied.
The numerical modules below are prerequisites, not achieved physics validation.
See [UPSTREAM_CHECKPOINT.md](UPSTREAM_CHECKPOINT.md) for the evidence scope and resumption order.

## Census and selection contract

```bash
# In an environment providing the MetaCat client and METACAT_SERVER_URL:
PYTHONPATH=beam python3 -m beamfastsim.tscensus --out census.json
```

The inventory uses read-only MetaCat API methods, discovers production POT,
Beam, BeamCat and MuBeamCat datasets, and retains dataset counts plus a bounded
sample of file metadata. Dataset counts are not POT counts. Missing file and lazy
query/discovery errors are recorded as incomplete results. Actual retained VD
content, release geometry and normalization require file/configuration inspection.
The default namespace is `mu2e`; personal upstream studies are not covered unless
explicitly requested, e.g. `--namespace '*' --pattern 'sim.*.TS1*.art'`.
A completed bounded census is not a proof that no other suitable sample exists.
Use `--include-retired` explicitly to investigate retired metadata; the receipt records
that choice without certifying a usable replica. An empty ordinary query does not
rule out retired records. The [target-library acceptance contract](TARGET_LIBRARY.md)
defines the boundary, exposure, identity and event-correlation evidence still required.

The production POT stage writes charged particles reaching `DS2Vacuum` into
`Beam`; its filter preserves those particles and ancestors, with extra VD hits
only for preserved particles. Such upstream hits are **conditioned on downstream
survival**. They cannot provide an inclusive TS-entrance denominator or unbiased
target-exit library. BeamSplitter selects muon-containing events but keeps their
other retained products too. Establish the exact production configuration before
interpreting any catalogue as an independent muon sample.

Nominal comparison surfaces are `Coll1_In/Out` (VD1/2), `Coll31_In/Out` (VD3/4),
`Coll32_In/Out` (VD5/6), and `Coll5_In/Out` (VD7/8). The DS2 entrance is downstream
of VD8; they are not interchangeable. Particle comparisons must retain event,
collection and particle identity, including pion-to-muon ancestry, not match by
time or distance. First stored VD hits are not guaranteed to be the earliest
chronological or forward passage unless the underlying collection proves it.

The production-target optimization objective counts distinct negative muons and
pions at COL1 entrance per generated POT. This is a different boundary from a
DS-entering-muon objective and a different geometry from the stock production
target. Do not transfer downstream timing or normalization to this objective.

## Retained-row extraction

```bash
PYTHONPATH=beam python3 -m beamfastsim.tsreference retained.root \
  --tree mutree/ntpart --vd 1 --vd 8 --out retained.npz
```

This reads the highest persisted tree cycle and records every cycle's row count,
source hashes, file-scoped event/particle identity and parent identity. Both muon
and pion charges are retained by default. No-hit particles are not dropped; their
VD state is masked with NaNs. Unsafe floating-point identities and duplicate
particle keys fail closed. The JSON receipt explicitly labels survivor conditioning
and unverified normalization; particle rows are not generated-event counts. Full
parent-chain resolution, cross-collection identity, directional recrossings and an
inclusive denominator are still separate requirements.

## Field reference

```bash
PYTHONPATH=beam python3 -m beamfastsim.tsfield \
  --inner PSMap.header --inner TSuMap_fix.header --inner TSdMap.header \
  --outer PSAreaMap.header --outer WorldMap.header --point 3904 0 -4004
```

`tsfield` is NumPy-only. It reads modern G4BL binary maps, subtracts the header
origin shift (or the documented legacy default), and implements `extendY flip=By`.
Only unity standard field parameters are accepted. The binary marker, byte count,
dimensions and finite field entries are checked; source hashes and coverage boxes
are reported. Both marker-declared endian orders are decoded; the native Offline
reader is stricter about mismatches. Unknown formats/extensions fail closed.

`FieldMapBank` selects inner coverage first and then the first covering outer map;
it does not add overlapping fields. Inner maps with overlapping or touching closed
coverage boxes are rejected to avoid history-dependent face selection. Exact grid
endpoints are supported. An uncovered query raises `CoverageError`; explicitly
requesting `return_coverage=True` returns NaNs and a validity mask, never a silent
zero-field extrapolation. The CLI uses JSON null for each uncovered field component and
exits nonzero. These are source-level semantics and synthetic tests, not a stock
field-service or transport equivalence certificate.

## G4Beamline crossing tables

```bash
PYTHONPATH=beam python3 -m beamfastsim.tsg4bl input.root --out crossings.npz
# Optional upstream plane in the same stored coordinate system:
PYTHONPATH=beam python3 -m beamfastsim.tsg4bl input.root \
  --plane NTuple/Z3712 --out z3712.npz
```

This opt-in local-file reader preserves every stored muon/pion crossing at seven
collimator planes, both charges, raw nonnegative weights and file-scoped
`EventID/TrackID/ParentID`. Recrossings are not deduplicated or chronologically
reordered. Highest tree cycles, complete tree inventory, raw coordinate/component
ranges, source hashes and unique-identity counts are recorded. All requested-plane
species are validated before filtering; inconsistent PDG/parent identities and
unsafe float IDs fail closed. Duplicate or byte-identical input files are rejected.
Parents may be absent; parent identity alone does not establish parent species.

The production-target optimization deck differs from stock Offline production in
geometry, field maps and physics list. Its COL1 recorder has no explicit
later-survival requirement in the inspected source, unlike the stock Beam filter,
but dataset provenance, runtime overrides, acceptance and direction still require
verification. `coordinates=Centerline` is not a certified Mu2e output transform.
`Z3712` is upstream of COL1, not a production-target exit. `LostInTarget_Ntuple`
is a stopping-target-region loss recorder, not a production-target exit library.
Counts of represented events or unique tracks are not generated POT exposure.

## Shared-boundary diagnostic

```bash
PYTHONPATH=beam python3 -m beamfastsim.tsfaces \
  --inner PSMap.header --inner TSuMap_fix.header --inner TSdMap.header \
  --inner PStoDumpAreaMap.header --inner ProtonDumpAreaMap.header \
  --atol 0 --out shared_faces.json
```

This separate tool compares the two reference interpolants on every common
refinement node of a shared closed face, edge or corner, with source and evaluated
array hashes. Positive-volume overlaps are unsupported; oversize grids fail rather
than truncate. On continuous multilinear pieces the node differences bound the
componentwise difference in exact arithmetic; floating comparisons are approximate,
not a runtime certificate. A discontinuous reflected-By seam fails the gate even
when sampled nodes agree. The tolerance is explicit, never fitted to a transport
result. Mismatch/unsupported returns exit 2; malformed inputs return exit 1.

Offline keeps the previous inner map when it still covers a query, otherwise
searching inner then outer maps in their configured orders. Its field wrapper's
cache spans field queries, not an explicit particle identity. Exact upper-endpoint
corner indexing also needs characterization in the matched runtime. This diagnostic
does not resolve that runtime behavior or change the default `FieldMapBank`
rejection of touching maps. No stock-transport equivalence is inferred.

## Independent-stream field selection

`tsselect.MapSelector` is an opt-in selector separate from the strict bank. Its
`field(points, previous)` returns fields and fresh next-map indices; initialize
`previous` with int64 `-1` of the same batch shape and carry it with each query
stream through compaction/permutation. A previous covering inner map is retained;
otherwise configured inner/outer priority applies. Touching faces are allowed,
positive-volume inner overlaps rejected. Independent stream state is **not** the
native Geant4 global field-cache history. Endpoint interpolation remains the
mathematical NumPy reference, not characterized native upper-endpoint behavior.
Audit receipts distinguish on-disk observations from evaluated arrays/grid data.

## Native staged-reference identity diagnostic

`ts1_run1b_readvd.fcl` and `ts5_run1b_readvd.fcl` read existing Run 1B COL1 source
and nominal TS1-to-DS transport outputs using the stock analyzer only, with
immediate-parent navigation retained and no simulation producer. These are
Run 1B references, not a Run 1A substitute. `tsentries` counts highest-cycle art
`Events` entries; represented particle rows and generator counters are different
quantities.

```bash
PYTHONPATH=beam python3 -m beamfastsim.tsentries input.art --out entries.json
PYTHONPATH=beam python3 -m beamfastsim.tspair \
  --input entrance.root --output downstream.root --out identity.json
```

`tspair` joins native `(run,subrun,event,track)` keys, rejects duplicate native
event shards, and walks retained parent identities in vectorized ancestry layers.
Restarted tracks have **new IDs** and `mu2ePrimary` code with their original track
as parent; copied ancestors are not new endpoints. Pion-decay daughters keep
separate identities. Missing links, species conflicts, unknown codes and cycles
fail the identity gate. A decay-descendant label means decay occurs somewhere in
the ancestry, not necessarily the only production process. Collection ProductID,
simStage, unselected species, complete passages and generated exposure are not
certified by these flat rows. The result is an identity diagnostic, **not** fast
transport, transmission, efficiency or physics validity.

## Conditional charged-pion decay kernel

`tsdecay` provides NumPy-only proper-lifetime sampling and constant-momentum
proper-time/distance helpers, plus isotropic two-body pion decay and a
rotation-free Lorentz boost. Daughters conserve four-momentum and electric charge;
the caller must create distinct daughter identities and integrate the trajectory
to the actual decay point. Momentum is MeV/c, length mm, proper time ns.

This is **conditional on pi → mu nu**, not an exact branching-fraction sampler;
using it at every pion decay is a labelled 100% muonic-mode diagnostic. Rare modes,
muon polarization and subsequent muon decay are absent. The numerical domain is
limited to pion gamma <= 1000, well beyond beam momenta but not arbitrary extreme
boosts. No materials, hadronic interactions, apertures, navigation or full PS/TS
transport validity follow from this kernel or its conservation tests.

## Short vacuum RK4 primitives

`tsstep.step` uses core `ehelix.rk4` for one short muon/pion vacuum segment,
retaining momentum magnitude and updating lab/proper clocks for the actual path.
The path is capped by a 0.1-radian estimate using the start field only; this is
not an error bound in a varying field. `plane_step` solves an arbitrary-plane
endpoint bracket by reintegrating each fractional segment with the same RK4,
not a frozen-field helix. It requires an explicit `assume_monotone=True` caller
contract; tangencies, starts on the plane and zero paths are unsupported.

These kernels accept pure field functions only. Mutable selector history needs a
separate trial/commit protocol and cannot be hidden inside that callable. No
material-veto geometry, swept-path navigation, earliest physical passage or
stock-transport validity is supplied. Analytic helix and step-halving tests are
numerical checks, not full PS/TS physics validation.

## Conservative CSG clearance and lineage driver

`tsnav` provides positive-inside clearance fields for spheres, boxes, finite
cylinders/annuli, full tori, half spaces, rigid placements and CSG operations.
When the child contracts hold and placements are exactly rigid, these are
sign-correct conservative 1-Lipschitz fields in exact arithmetic, **not exact CSG distances**.
Near-orthogonal stored matrices and construction arithmetic need separate bounds;
see the stored transformed-solid section below. Intersections use minimum, unions maximum and
subtraction `min(first, -second)`. The finite cylinder has no artificial axis
boundary when its inner radius is zero. Placements take active local-to-global
rotations; this is not an implicit conversion of Geant4's passive placement
matrices. No nominal detector geometry or incomplete GDML import is supplied.

`tsdriver.propagate` composes short RK4 stepping, conservative clearance and the
conditional pion-decay kernel in a vectorized CPU lineage loop. Each numerical
segment is bounded by half its start clearance. For a complete modeled vacuum
region, a physical unit-speed trajectory of arclength less than that positive
clearance cannot leave its safety ball, regardless of curvature. Field-stage
queries and numerical endpoints are checked against the same ball. RK4 error,
floating-point clearance and geometry completeness remain separate uncertified
issues; the ball argument does not establish stock transport equivalence.

The driver explicitly requires `mode='no_decay'` or
`mode='conditional_muonic'`. Decay mode requires a caller-owned NumPy Generator;
proper lifetimes are sampled once per particle, not at each step. Pion parents,
daughter birth momentum and neutrino momentum are retained in a decay ledger.
The returned `(root_index, generation)` is a **local numerical identity**, not a
native TrackID: external event/particle keys remain attached by input-row index.
Muon decay terminates that lineage; its daughters are not transported. Pion
branching remains the labelled 100% muonic diagnostic, without polarization.
Compensated path/proper-clock sums and proper-clock endpoint scheduling avoid
lifetime-to-distance roundtrip misclassification; the scheduling allowance is
8 float64 epsilons relative to lifetime/current age, not a physics-fitted cut.

Statuses distinguish path limit, boundary pending, start not interior, muon decay
and iteration limit. **Boundary pending is unresolved, not absorption, loss or
transmission.** The driver stops within the caller's boundary tolerance and does
not nudge across a surface. Conservative CSG-internal seams can also stop it;
these do not prove physical material. A downstream material handler and complete
matched geometry are required before a full TS comparison or efficiency exists.
Inputs are copied; zero-progress coordinates, contradictory geometry queries and
field-coverage errors fail rather than silently accepting an endpoint.

## Matched-export inventory and restricted clearance compilation

The opt-in `ts_geometry_run1a.fcl` / `ts_geometry_run1b.fcl` configurations
inherit the stock GDML dumper: Minimal physics and one transport-only event.
They are geometry exports, not hadron-production samples or physics references.
Use each with its explicitly matched released runtime and field configuration.

```bash
PYTHONPATH=beam python3 -m beamfastsim.tsgdml geometry.gdml \
  --root TS1Vacuum --root TS3Vacuum --out structure.json
PYTHONPATH=beam python3 -m beamfastsim.tsgeom geometry.gdml \
  --root TS1Vacuum --root TS3Vacuum \
  --vacuum-material PSVacuum --vacuum-material DSVacuum --out clearance.json
```

`tsgdml` inventories every world-reachable physical instance (including repeated
logical placements) and requested subtrees, validates structural reference closure
and cycles, and retains raw solids/materials/defines/ancestor transforms. Unknown
solid shapes remain explicitly unsupported, not silently discarded. DTD/entities,
modular includes, malformed references and unsupported placement constructs fail.
File bytes, XML nodes/depth, physical depth, expanded instance count and retained
path items are bounded. The receipt does **not** evaluate transforms, prove
material physics, certify overlaps or provide a standalone geometry compiler.

`tssolid` adds slope-normalized conical frusta, angular sectors and simple
constant polygon extrusions. Nonconvex polygons use signed nearest-edge distance,
not an intersection of all edge half-spaces or triangulation with false seams.
Unrepresentable conical slopes are refused rather than approximated as cylinders.

`tsgeom` compiles a deliberately restricted subset directly from an audited GDML:
selected vacuum mothers and **all** their direct leaf daughters. Box, tube, cone,
full/sector torus, constant unscaled extrusion and complete Boolean operands are
supported. Unknown dimensions/attributes, symbolic values, nested daughters,
scales and unsupported units fail closed. Exact material references take priority;
pointer-suffix-stripped material aliases must be unambiguous. Accepted vacuum
daughters coalesce into their mother; material daughters remain exclusions. No
volume is dropped just because its name looks like a virtual detector.

Physical placement angles use the inverse frame rotation, while Boolean operand
angles use the direct rotation, following Geant4 v11.3.2 GDML read sources. First
operand transforms and all ancestor placements are retained. This source-level
conversion plus synthetic tests is **not** a native solid-query roundtrip
certificate. All queries are in the **export's world frame**, not an assumed Mu2e
frame. Optional `--point x y z` reports numerical clearance only.

Both inspected released exports compile all 31 volume instances across the five
TS vacuum mothers, including the shifted circle-intersected-with-strip COL3 holes,
antiproton disk/support/staircase and the COL5 section inside TS4. This is selected
subtree coverage, not a whole-world or PS/DS-entrance navigation certificate.
Other sibling overlaps/geometry remain outside the model. Shared zero-set seams
are unresolved; neither compilation nor point checks establish transmission.

## Explicit point owners and complementary-cut joins

`tslocate.Partition` retains selected vacuum mothers and direct leaf placements.
`Scene.locate(points)` returns strict vacuum/material interiors, exact exported
material/placement indices, local conservative clearance and root/material/zero
counts. Vacuum daughters coalesce into their mother; material daughters are
clipped by the mother. Zero sets, overlapping roots/leaves and points outside the
selected scope return explicit unresolved statuses with no material owner and
zero usable clearance. A unique zero set has a separate boundary candidate index,
not a certified crossing or a material transition. Internal CSG zeros can produce
that status too. CLI point receipts include these **numerical** classifications.
No closest-material guess, priority selection, tolerance or native query is used.

`tsnav.SplitUnion(A, B, H)` represents `(A intersect H) union (B intersect -H)`
using one explicit half-space and its algebraic opposite. Its clearance is
`max(min(a,h), min(b,-h), min(a,b))`: the final, set-redundant overlap term removes
the shared-plane zero where both **unclipped** remainders are strictly interior.
Min/max preserve the conservative 1-Lipschitz contract in exact arithmetic. This
is not universal seam removal: shared zeros in the remainders survive, and real
gaps are not filled. A synthetic straight vacuum trajectory crosses an exact
split without a nudge, while an inserted thin material slab remains unresolved
at its entry boundary. This does not authorize crossing an unresolved obstacle.

The exported TS placements are **not automatically welded** by this primitive.
Independent Euler/translation evaluation yields tiny numerical differences even
at nominal joints; comparing within an epsilon would change the represented
geometry. Matched native tolerances and proof of complementary cuts are still
required for those joints. Current GDML vacuum clearances and driver statuses
remain unchanged; no physical material crossing/transport is implemented here.

`tschord.partition` provides a bounded **straight-segment diagnostic**, not a
curved-path crossing algorithm. Adaptive dyadic subdivision returns a complete
ordered cover for each input row. A resolved interval fits strictly inside a
midpoint local-ownership clearance ball; uncertain surface/overlap/out-of-scope
intervals remain unresolved, with no material owner. `atol` bounds their chord
length only: it never enlarges geometry or certifies a passage. Depth caps are
separate unresolved results and interval-resource overflow raises without a
partial result. Loss of supplied endpoints in displacement arithmetic and
unrepresentable spatial subdivision fail. The mathematical ball argument does
not certify floating-point geometry or path interpolation errors.

```bash
PYTHONPATH=beam python3 -m beamfastsim.tschord geometry.gdml \
  --root TS3Vacuum --vacuum-material PSVacuum --vacuum-material DSVacuum \
  --assume-straight --chord 2 0 0 -2 0 0 --out chord.json
```

Material identity is preserved through exported instance/path/copy metadata;
unnamed physical placements remain unnamed. Duplicate aliases selecting the same
physical root fail instead of manufacturing overlap. Chord results retain input
row indices and interval coverage, not native particle identities or transport
probabilities. An actual curved trajectory needs a separately established path
enclosure/error bound; treating its endpoint chord as its path is unsupported.

## Conditional uniform-field curved enclosures

`tsarc.UniformHelix` reuses core's analytic uniform-field helix with immutable
initial states, per-row constant fields, charge-separated muon/pion PDGs and
unwrapped path lengths (supported phase up to 64 pi). It never substitutes a
frozen-field helix for variable-map transport. All evaluations restart from the
original state, so subdivision does not accumulate independent restart errors.

`tsarc.partition` makes a complete ordered arc-length cover using two separate
ownership-ball tests. For interval length L and curvature upper bound k, the
endpoint chord's maximum deviation is bounded by k L^2 / 8 in exact arithmetic.
Its midpoint ball radius is half the chord length plus that deviation and the
declared endpoint-position error. A separately evaluated arc midpoint has a ball
of radius L/2 plus its declared position error. **Each ball queries geometry at
its own centre**; taking a smaller radius with the other centre would be invalid.
The uniform-model bound k = K |B| / p is valid for both charges and any pitch.
Whole turns and coincident endpoints do not imply a zero path or zero curvature.

Position-evaluation error and geometry-clearance error are **required explicit
caller assumptions**, not values inferred from an RK4 step-doubling difference.
Obvious zero-displacement collapse on a positive arc with phase <= pi/2 fails;
legitimate closed full turns remain supported. This detects one representability
failure, not general floating-point accuracy.
A resolved interval has strict positive clearance margin after those errors;
width/depth/resource limits do not invent owners or transitions. Tangencies,
multiple passages and boundary-following paths may remain unresolved even when
the two endpoint owners agree. `atol` limits unresolved arc length, not a geometry
nudge. No material loss/scattering/decay is applied along these diagnostic arcs.

```bash
PYTHONPATH=beam python3 -m beamfastsim.tsarc geometry.gdml \
  --root TS3Vacuum --vacuum-material PSVacuum --vacuum-material DSVacuum \
  --field 0 0 1 --arc 2 0 0 -100 0 0 13 4 \
  --position-error-mm 0 --geometry-error-mm 0 --out arc.json
```

The example's zero errors request a **numerical idealization**, not a proven
zero-error calculation. Core trig/series evaluation, radius arithmetic and the
geometry clearances lack interval-arithmetic certification. CLI receipts record
these assumptions and explicitly deny native/floating-point certification,
variable-field transport and material physics. This is curved-path numerical
geometry development, not production TS transmission or boundary-root isolation.

`tsarc.vacuum_prefix` returns only the leading resolved vacuum portion, stopping
before the first unresolved/non-vacuum interval. It records the first contiguous
unresolved block and adjacent resolved owner indices for a future handoff; these
neighbors do **not** prove a unique crossing. A tangent contact can have the same
owner on both sides, and multiple passages may lie inside a bracket. Initial
material or uncertainty produces zero advancement. No clocks, pion decay,
material response or tolerance nudge is performed. Optional CLI `--vacuum-prefix`
includes this stop state without promoting later diagnostic material intervals
to traversed vacuum. Variable-map integration and native tolerances remain open.

## Remaining gates

- An inclusive, identity-preserving TS-entrance population and generated POT
  exposure, including rejected events and any production filters/resampling.
- The matched PS/TS maps, geometry, apertures, collimators, windows and supports.
- Pion decay feed-down, muon decay, range-out and relevant hadronic processes.
- Development/held-out shard separation, paired per-shard rates, momentum and time
  for each muon charge at TS exit and DS entrance.
- Same-boundary production log timing before quoting grid-slot replacement.

Existing DS field readers, transport, RNG sequences and defaults are untouched.

## Whole-map curvature-bound prerequisite

`tsbound.magnitude(map)` scans every map node in bounded chunks. Trilinear
interpolation is a convex combination of eight node vectors, so the largest
node magnitude bounds every interpolated magnitude in exact arithmetic. Y
reflection changes a component sign, not its norm. `covered_box(map, lo, hi)`
requires the whole closed requested box inside one map's coverage, including
reflected-map gaps. It deliberately uses the whole-map bound, not track samples
or a tight local estimate. This does not prove native selector/cache behavior,
seam continuity, floating-point interval accuracy or RK4 position error. A
curvature bound alone is not a propagated-state error certificate.

## Conditional varying-field model enclosures

`tsbound.spatial(map)` scans every parallel grid edge in bounded chunks. Each
trilinear Jacobian column is a convex combination of the corresponding edge
slopes; the root-sum-square of their maximum norms bounds the Jacobian operator
norm. Continuity extends this Lipschitz bound across grid cells. Reflected-map
lobes preserve it, but a zero-plane with nonzero nodal By is discontinuous and
cannot use it across that plane. Coverage gaps remain gaps regardless of slopes.

`tsvary.MapEnclosure` compares the **ideal varying-field Lorentz trajectory**
with a same-start uniform-field reference helix; it does not integrate an RK4
endpoint. It requires a single explicit `FieldMap` and the entire unit-speed
reachable box `x +/- length` within one continuous covered domain. No sampled
track-point maxima, selector history, map seams or field sums are inferred.
It freezes the measured map field at the start only as a reference curve,
not as a substitute trajectory claimed to follow the varying field.

Let `a = K/p`, `G` be the whole-map spatial Lipschitz bound and `eB` an explicit
bound on the initial frozen-field evaluation error. For direction difference
`w = u - v`, the Lorentz difference equation has a skew rotation term plus
`a_signed * v cross (B(r) - B0)`. The skew term contributes no norm growth.
Unit speed gives `|r(s)-x0| <= s`, hence

```
|u(s)-v(s)| <= a (G s^2 / 2 + eB s)
|r(s)-R(s)| <= min(2 s, a (G s^3 / 6 + eB s^2 / 2))
```

This establishes a conditional **model-derived position-error bound**, not an
RK step-doubling estimate. It compares paths from the original initial state;
restarting from an uncertain reference endpoint as if exact is unsupported.
`field_error_tesla` and the reference helix's all-position `position_error_mm`
are required caller assumptions. Zero explicitly requests a numerical
idealization, not certified zero floating error. Node differences, AABBs,
interpolation, helix evaluation, radii and geometry are not outward-rounded
interval arithmetic, so no floating-point or native certificate is issued.

`balls(source, s_lo, s_hi)` provides two enclosures for each ideal varying-field
subpath. The reference midpoint ball adds its error at the **absolute midpoint**
to halfwidth. The reference-endpoint chord ball adds the actual-map curvature
sagitta `K max|B|/p * width^2/8` and the maximum endpoint error at **absolute hi**.
Errors do not shrink merely because a late interval is subdivided. Each ball
has its own centre. `owners(..., geometry_error_mm=...)` separately queries
both centres and resolves only strict whole-ball containment. Discrete source,
owner and status arrays retain integer dtype and are immutable. Unresolved has
owner -1; it is not loss, a crossed boundary, or a transported state. A resolved
material interval is an identity diagnostic, not permission to transport vacuum
through material. No decay clocks, interactions, earliest crossings or material
entry/exit handoff are implemented here.

`beam/art/tsvary_audit.py ROOT --out NEW_RECEIPT.json` reads the existing seven
staged map pairs and original geometry exports without running simulation. Its
few short chosen segments are explicitly **not production tracks**; zero-error
receipts are conditional diagnostics, not transmission, native geometry
validation, RK4 accuracy certification or a performance benchmark.

## Carrying position and direction uncertainty

`tscarry.MapContext` snapshots one map into immutable byte-backed arrays and
binds its all-node/all-edge bounds and evaluated-grid fingerprint. A carried
state cannot silently select another map. Freezing a new reference field inside
the same context does not alter the actual ideal field; a map-selector handoff
would require a separate contract and is not implemented.

`tscarry.State` stores nominal position and a nonzero nominal direction vector,
a **separately supplied scalar momentum**, PDG, source-row tag, accumulated path,
and both position/direction error bounds. The scalar is treated as exact for
this ideal magnet-only model and is carried bit-for-bit; it is never re-derived
from a propagated vector's numerical norm. The nominal unit direction is defined
by mathematical normalization of the stored vector. Row tags are not substitutes
for native particle identity. Physical uncertainty in momentum is not supported.

For a segment with initial errors `ex, eu`, length `s`, spatial bound `G`, and
frozen-field evaluation error `eB`, the skew Lorentz difference argument gives

```
Eu(s) = min(2, eu + a ((G ex + eB) s + G s^2/2))
Ex(s) = min(ex + 2 s, ex + eu s + a ((G ex + eB) s^2/2 + G s^3/6))
a = K/p
```

The `G ex` term includes uncertainty in the field at the true initial position.
Both initial errors survive at zero length; the position cap is **ex + 2s**,
not 2s. The entire uncertain reachable box `x +/- (ex + length)` must be inside
one continuous covered map domain. Whole-ball geometric checks include these
errors at absolute local segment arclength; subdivision cannot erase them.

Every `Segment` requires explicit field-, reference-position- and reference-
direction-evaluation error declarations. The direction declaration bounds the
**normalized stored output direction** against the ideal frozen-field reference,
including normalization arithmetic; a position error cannot stand in for this.
`attempt(segment, geometry, geometry_error_mm=...)` accepts only rows whose WHOLE
interval has a strict conditional vacuum enclosure. It adds endpoint evaluation
errors to `Ex, Eu` before the next segment. Material/unresolved rows roll back
the entire nominal state, uncertainty, row tag and accumulated path. A vacuum
endpoint beyond a barrier cannot authorize an advance. The result retains the
material owner or unresolved status, not an invented loss or boundary crossing.

This produces a segmented **nominal/reference construction** with a bound on a
continuous ideal trajectory, not necessarily a continuous exact stitched helix.
Endpoint-evaluation discrepancies are carried rather than hidden. There is no
uncertainty reset, numerical-error estimator promotion, native field history,
material interaction, clock or decay treatment. Zero declarations remain explicit
numerical idealizations. All arithmetic lacks outward-rounded interval bounds;
this is not an RK4, floating-point, native geometry or physics-validation claim.

`beam/art/tscarry_audit.py ROOT --out NEW_RECEIPT.json` applies at most ten 5 mm
conditional reference attempts to chosen short interior paths using already
staged maps and the original geometry exports. It preserves earlier receipts
and labels the inputs as diagnostics, not production tracks or transmission.
`beam/art/tscarry_summary.py RECEIPT.json` summarizes the saved audit without
re-propagation; it reports retained error growth, status, path and fixed momentum.

## Bounded adaptive conditional vacuum prefixes

`tsvac.walk` composes the carried-state enclosures into an **accepted nominal
vacuum prefix**. Trial length starts at `min(max_step, remaining)`; a failed
whole-interval geometry, map-coverage or error-budget check halves the trial
from the SAME immutable prestate. Only a whole-interval vacuum enclosure inside
the configured position AND direction error budgets commits a candidate. All
other trials leave position, direction, both errors, scalar momentum, source
tags and cumulative path unchanged. No endpoint-only acceptance is used.

The explicit `tsbound.box_coverage` predicate handles per-row coverage/continuity
failures before segment construction. It does not catch arbitrary numerical
exceptions as coverage failures. Unsupported phase, malformed bounds, arithmetic
or contradictory geometry abort the batch. There is no automatic map switch.
Initial uncertain geometry/map/budget checks do not commit a zero-length handoff
or add evaluation errors. Initial strict vacuum/material owners and clearance
margins are retained; unresolved has owner -1 and a zero request leaves the
initial geometric status unqueried (-1). A zero request returns unchanged.

Controls are bounded and auditable: `max_trials` is a lifetime per-row trial
limit, `max_steps` an accepted-handoff limit, and `min_step` terminates a FAILED
trial, not a valid shorter positive final remainder. The worst-case ledger size
`N * max_trials` must fit `max_records` before work begins. Each trial records
source row, committed progress, attempted length, acceptance/rejection reason,
owner/margin and before/after error bounds. Discrete arrays remain immutable and
integer typed. Rejected ledger rows have unchanged after-errors.

Path consumption is fail-closed: error-free TwoSum checks require relative and
cumulative binary-float path sums, and each final remainder, to be EXACTLY
representable. Unsupported decimal-step combinations may raise rather than
silently discarding a positive remainder. No epsilon-based completion, zero
transport step or minimum-distance nudge is used. This bookkeeping restriction
does not make geometric or numerical trajectory arithmetic interval certified.

Terminal control status and the final failed-trial reason/length are distinct.
The returned state includes only accepted intervals. A map/geometry/budget stop
is **not** a physical crossing, a lost particle, or a root bracket; uncertainty
floors can stop refinement well before a surface. Resolved material remains a
stop, never a vacuum handoff. The declared evaluation/geometry errors still are
caller assumptions (explicit zero means a numerical idealization), and no native,
RK4, material, pion-decay or physics-validity certificate is issued.

`beam/art/tsvac_audit.py ROOT --out NEW_RECEIPT.json` exercises bounded interior
and near-limit diagnostic paths in already-staged maps and the original exports.
It preserves full trial ledgers and labels failed trials as non-root-brackets;
no new simulation, production tracks or transmission measurement is involved.

## Numerical error budget proposal (not achieved guarantees)

Separate model approximation, evaluation arithmetic, geometry representation and
native-model discrepancy. The field-model approximation above has an analytic exact-arithmetic
bound. The legacy carry still requires evaluation and geometry error declarations;
the later standalone outward-arithmetic modules do not retroactively certify that chain or the native model.
Required declarations remain assumptions; replacing zero with a small positive number does NOT prove it is a bound.

Initial engineering targets near a material face are 0.001 mm total position
uncertainty and 1e-6 normalized-direction difference. Reserve at most 0.0001 mm
of that position budget for numerical/geometry evaluation, leaving 0.0009 mm
for initial-state plus field-model carry. These are proposed thresholds, not
physics-validation criteria or claimed numerical accuracy. The 0.150 mm thin
window motivates this starting scale; grazing incidence and nearby surfaces can
require stricter targets. Path-contact interval width is a separate criterion:
normal-position uncertainty is amplified by the inverse transverse derivative.
No positive fixed threshold authorizes a crossing of an ambiguous rim or seam.

Qualification and integration priorities for defensible bounds follow. Restricted standalone
arithmetic prerequisites are implemented in the later sections; a complete qualified navigator
and practical long-path integration remain open.

- Use binary64 and explicit outward-rounded interval operations for grid
  indexing/interpolation, transforms, normalizations, norms, AABBs, clearances,
  radius sums and root arithmetic. Include input conversion and accumulated
  placement error; a final nextafter on an unbounded computed expression is not
  sufficient. First support bounded domains and reject unsupported arithmetic.
- Bound helix evaluation and normalized directions using range-reduced phase,
  explicit series/approximation remainder bounds, and interval arithmetic.
  Higher-precision cross-checks and step doubling are tests, not upper bounds.
- Bound map norms/slopes outward as well as query arithmetic. Field storage
  precision defines the selected ideal map; uncertainty relative to a different
  physical/native map remains a separate modeling issue. Nonzero map seams and
  native query-history differences are not numerical roundoff allowances.
- Charge every accepted handoff BOTH position and direction evaluation errors;
  keep initial-state errors and scalar-momentum assumptions explicit. Adaptive
  shortening reduces new model error but cannot erase accumulated uncertainty.
  An exhausted floor means unresolved or a future validated higher-order
  enclosure/reintegration from a retained anchor, never an error reset.
- Compare unresolved fractions, surface order, path and material lengths under
  tighter declared budgets on development samples before a frozen held-out
  physics comparison. Preserve unresolved weight; do not count it as loss.

No stochastic scattering, energy-loss or hadronic approximation is included in
these numerical budgets. Those require independent material-model validation.

## Isolated planar material first-contact intervals

`tssurface.faces(partition, material_leaf_index)` extracts six complete box
faces or two full-tube endcaps, including full-azimuth Sector wrappers and nested
placements. It retains the exact selected Volume, material, instance/path and
mother index. Arbitrary Boolean operand faces, partial-azimuth faces, curved
walls and unsupported solids are refused, not guessed. No geometry is welded.

Each face has an inward normal and a guard retaining every OTHER primitive
constraint, the selected mother, exclusions of competing material leaves and
other selected roots. A whole-interval trajectory ball must lie strictly inside
this guard. This excludes rims, holes, opposite faces, overlap and out-of-scope
joints. Complete exclusion of unmodeled world siblings is still an assumption.

`tssurface.entry(segment, face, plane_error_mm=..., projection_error=...,
geometry_error_mm=..., path_atol=..., max_iterations=...)` analyzes the segment
as a HYPOTHETICAL INCOMING VACUUM EXTENSION, not as valid motion after contact.
Strict uncertain endpoint signs establish a plane crossing. With unit inward
normal n, midpoint reference direction v, total direction error Eu and whole-map
curvature kappa, the bound

```
m = n.v - Eu(mid) - evaluation_direction_error
    - kappa * max(mid, length-mid) - projection_error > 0
```

makes the signed plane residual strictly increasing over the entire extension.
Together with the guard this proves one first material entry WITHIN this segment,
conditional on the bounds. No preceding history outside the segment is inferred.
Plane error must uniformly bound normal/origin representation discrepancy over
the whole tube; its derivative contribution is separately included in
projection_error. Stored-plane dot products and root arithmetic are outward
under the restricted tsinterval contracts below. Guard evaluation still uses
geometry_error_mm; trajectory balls, Ex/Eu, curvature and helix evaluation remain
uncertified or conditional, as in the carried-segment contracts.

Strict signed bisection narrows a valid interval. At an uncertain midpoint with
signed nominal residual f and total residual error e, intersect with
`mid +/- (abs(f)+e)/m`. This preserves the conditional unique-root interval;
a nonzero uncertainty floor can prevent the requested PATH-width isolation.
Iteration/numerical caps retain the prior interval, while failed initial guards,
signs or transversality make no root claim (and do not prove no entry).

The returned receipt distinguishes isolated, uncertainty-limited and capped
conditional intervals. It preserves row tags and does NOT modify the input state,
return a post-material endpoint, initialize material clocks, or implement exit,
loss, scattering, range-out or pion interactions. A material exit needs an actual
material-dynamics path enclosure; the incoming vacuum extension cannot provide
one. Reusing a root midpoint as an exact restart would discard uncertainty and
is not supported. Numerical/native certification and full TS transport remain open.

`beam/art/tssurface_audit.py ROOT --out NEW_RECEIPT.json` uses original exports
and a staged map for chosen thin-window face starts. Positive error declarations
are explicit unproved assumptions, not certified budgets or production tracks.

Suggested initial allocation of the 0.0001 mm numerical reserve is four
0.000025 mm contact-level contributions: field-evaluation-induced displacement,
reference-position/normalization-induced displacement, geometry/plane residual,
and enclosure/root arithmetic. Likewise reserve 1e-7 of the 1e-6 total direction
budget for numerical evaluation/normalization, leaving 9e-7 for initial/model
carry. These are cumulative CONTACT-level allowances, not per-step constants;
all propagated direction-error drift must be charged to position too. The current
carry stores total errors rather than this proposed provenance decomposition.

For a length h and a=K/p, a field-evaluation bound eB contributes at most
`a*eB*h^2/2` in position and `a*eB*h` in direction on that segment. Therefore an
allocated pair of remaining allowances dx,du implies
`eB <= min(2*dx/(a*h^2), du/(a*h))`, with prior carry and later propagation still
charged separately. There is no defensible universal eB independent of momentum,
step length and accumulated history. Near a face the allowable interval width
also scales as residual uncertainty divided by the positive derivative lower
bound. A zero/negative lower bound is unresolved, not a request to loosen budgets.

The proposed tight budgets are NOT a promise that the present frozen-field
reference can economically meet them over a full transport line. Whole-map
Lipschitz bounds are deliberately broad; many tiny steps still accumulate
initial-error coupling and direction-error drift. Long-path practicality likely
requires a validated higher-order trajectory enclosure with a tighter, genuinely
covered local field bound, rather than simply increasing trial counts. The first
planar-entry diagnostics start close to the face; they do not demonstrate that a
production trajectory can arrive there with comparably small carried errors.

## Restricted outward arithmetic and stored-plane enclosures

`tsinterval.Interval` accepts finite binary64 endpoints interpreted exactly,
not arbitrarily rounded integer/decimal source data. Basic add/subtract/multiply/
divide, square, sqrt and each reduction operation round outward with adjacent
representable floats. This contract assumes IEEE-754 correctly rounded binary64
basic operations/sqrt, gradual underflow and no fast-math reassociation. It does
not configure or prove the execution environment. Overflow, a divisor containing
zero, or an outward endpoint outside finite range fails closed. Signed zero and
underflow-to-zero are enclosed with neighboring subnormals, not discarded.
Scaled vector norms avoid raw square overflow; normalization encloses the exact
mathematical unit vector, never asserts rounded coordinates have exact unit norm.
No sine/cosine, decimal/GDML conversion or field evaluation is certified here.

`tsplane.signed` and `projection` enclose distance/projection for the mathematical
unit normal of a STORED vector and exact stored origin, operation by operation.
This covers their arithmetic, NOT uncertainty relative to an intended native or
exported plane. Representation/transform discrepancy still requires a separate
uniform plane residual and direction-projection bound. The root and derivative
arithmetic in `tssurface.entry` now uses these outward operations; tiny positive
root widths remain intervals rather than silently collapsed exact roots. Whole-
trajectory ball construction, guard geometry/placement, Ex/Eu, curvature and
helix evaluations are still uncertified or caller assumptions. The receipt flag
`plane_root_arithmetic_outward` applies only to that restricted layer; the overall
`floating_point_certified` and `native_query_certified` remain FALSE.

## Uncertain first-contact state sets, not point restarts

Every `tssurface.Entry` retains the EXACT incoming Segment and Face objects,
original per-row plane/projection/geometry error declarations and isolation
controls. `tscontact.contact(entry)` compacts only rows having a conditional
unique first-entry interval, preserving their original row indices/tags, status,
fixed scalar momentum, charge/species and full original entry provenance. Rows
without a uniqueness proof remain unresolved in the original receipt. Capped
or uncertainty-limited intervals may still produce a conditional set, not an
isolated point or permission to transport material.

For contact arclength s* in [lo,hi], evaluate reference x,u at LO, which is at or
before first contact. Do not use a possibly post-contact midpoint of the incoming
vacuum extension as a material state. Precontact unit speed and curvature give

```
Rx = Ex(lo) + evaluation_position_error + (hi-lo)
Ru = min(2, Eu(lo) + evaluation_direction_error + kappa*(hi-lo))
```

Width, sums, products and cumulative path endpoints are computed outward under
the restricted arithmetic contract. The position center is the evaluated x(lo)
and can be off the face. The direction center denotes MATHEMATICAL normalization
of the stored nonzero raw vector, not the rounded normalized coordinates; this
makes the unit-direction diameter cap 2 legitimate without hiding normalization
roundoff. Upstream direction evaluation error must use this same convention.
Bounds concern the one-sided INCOMING direction before interactions.

The contact set is a position ball intersected with the ORIGINAL CONDITIONAL
FINITE FACE and its guards, together with the direction ball and retained path
interval. If the intended face differs from the stored plane, the additional
`intended_plane_slab` bounds its stored signed residual by +/- the declared plane
representation error. It must NOT be replaced by zero residual on the stored
plane. No projection, exact surface restart, clock, material length or material
exit is returned. No State constructor or transport handoff is provided. Position,
direction and path correlations are not resolved by choosing arbitrary centers.

These are conditional sets: upstream root, map continuity/coverage, geometry,
physical precontact dynamics and evaluation bounds remain assumptions. This does
not certify full floating-point navigation, native equivalence or physics. A real
material integrator must start from an admissible correlated contact set and bound
its own dynamics; incoming-vacuum continuation cannot serve as an exit path.

`beam/art/tscontact_audit.py ROOT --out NEW_RECEIPT.json` reads original exports
and a staged map for chosen near-normal disk faces. It records off-plane centers,
radii, raw direction convention, cumulative intervals and exact face provenance;
its positive evaluation/geometry declarations remain unproved assumptions.

Bound scope is intentionally explicit: exact-rational arithmetic oracles exercise
interval endpoints, including cancellation, absorbed addends, underflow and mixed
broadcasts; decimal high-precision checks exercise stored-plane cancellation.
They are regression evidence under the assumed arithmetic environment, not a
certificate for other backends, GPU fast-math, native geometry or trajectory
transcendentals. Before device use, its arithmetic environment needs its own
qualification. Serializing a direction center as a rounded unit vector without
retaining its mathematical-normalization convention changes the set and is not
supported. Contact-source tags are row tags, not native ancestry identities.

## Outward whole-map and carried-polynomial prerequisites

`tsbudget.scan(map, chunk_nodes=...)` is a separate opt-in all-node/all-edge
scan. Per-operation interval norms bound every exact stored node; convexity
then bounds the ideal trilinear map. Each edge difference and spacing division
is outward; maxima of parallel-edge slope norms bound the Jacobian columns,
and their outward Euclidean norm bounds the operator norm. The fingerprint is
computed in bounded node chunks, including non-C-contiguous input grids. No
track sampling, fitted local bounds or seam waiver is used. Reflected zero-plane
By discontinuity is flagged separately; gaps and whole-domain coverage still
need explicit checks. The source map must not be mutated during scanning.

`tsbudget.errors(bounds, momentum, length, position_error_mm=...,
direction_error=..., field_error_tesla=...)` evaluates the carried Ex/Eu analytic
polynomials and unit-speed caps outward, as well as K*Bmax/p. Positive momentum,
exact stored binary64 K and nonnegative binary64 row inputs define the model.
Initial ex/eu and initial field-evaluation error eB remain assumptions, not
computed interpolation/normalization bounds. At zero length initial errors are
returned unchanged. Exact-zero factors may be recognized algebraically; positive
underflow terms are NOT discarded. Overflow fails closed even if a separate
crude cap might have been finite. All arithmetic inherits tsinterval assumptions.

The flags distinguish outward bound arithmetic from full floating-point field
or transport certification: interpolation, coordinate/index/domain arithmetic,
helix, geometry and native cache equivalence remain open. This module does NOT
replace the existing MapContext, Segment.errors, tsvac or tscontact defaults;
no mixed partial-certification claim is made. It supplies a prerequisite for a
future fully qualified path rather than silently changing existing receipts.

`beam/art/tsbudget_audit.py ROOT --out NEW_RECEIPT.json` scans all seven existing
maps and records selected error polynomials for exact scalar momentum, not
propagated tracks, material traversal, transmission or performance validation.

## Stored-grid field-query enclosures

`tsquery.evaluate(map, points, position_radius_mm=..., max_cells=...)` is an
opt-in standalone receipt for the mathematical trilinear map defined by exact
stored binary64 origin, origin shift, spacing and node values. It does not
reinterpret rounded `GridHeader.lo/hi` as exact physical boundaries. Six scalar
metadata domain endpoints are computed exactly as rational numbers and rounded
inward to binary64; the whole Cartesian query box must lie within them. Only
AFTER this independent coverage proof is a coordinate interval intersected with
[0,n-1]. Rounding is not permission to extend a domain or waive a seam. Decimal
header conversion and intended/native map discrepancies remain outside scope.

Coordinate subtraction/division, cell-local interpolation, reductions and vector
error norms use restricted outward arithmetic under tsinterval assumptions.
All intersected cells are retained, not just the nominal selected cell. A finite
whole-batch work cap counts nominal AND uncertain-region cells before cell/node
expansion; exceeding it fails closed, never drops cells. Input-sized arrays are
not bounded by this cap. Grid/source data must remain unchanged during the call.

The immutable receipt separates `evaluation_error_tesla`, bounding ideal nominal
field minus the actual stored `map.field(points)` result, from
`region_error_tesla`, which also contains position-box field variation. These
are alternatives for different purposes, not independent errors to add twice.
A scalar radius covers the Euclidean ball by its Cartesian box. Reflected maps
retain a gap around zero when present; boxes crossing a zero plane include both
By signs even if the map is discontinuous there. This is a value-range bound,
NOT a continuity/Lipschitz/native-map-selection claim. Exact zero and last-node
queries may include adjacent cells conservatively, without clipping outside
points into coverage. Unsupported/overflow arithmetic fails closed.

`beam/art/tsquery_audit.py ROOT --out NEW_RECEIPT.json` reads all seven existing
maps at chosen interior points and 1 nm query-box radii. No tracks are transported.
No existing field/carry/vacuum/contact defaults change; full floating-point,
helix, transformed geometry, native query/cache and material transport remain
unqualified. A tiny nominal interpolation bound cannot cure a broad trajectory
model enclosure or supply a material exit.

## Restricted outward uniform-helix evaluation

`tshelix.evaluate(x, direction, momentum, pdg, field, length)` is a separate
opt-in endpoint enclosure. Inputs are exact stored binary64 arrays; momentum
is an independently supplied positive scalar per row, and the initial direction
is the mathematical normalization of the stored raw vector. Supported charges
are mu+/mu-/pi+/pi-. The ideal Lorentz generator is
`W v = v cross (q*K*length*field/momentum)` with exact stored binary64 K.
An outward norm upper bound must establish `||W|| <= 1` rad. Larger phases fail
closed; no undocumented argument reduction or libm sin/cos error is assumed.

Operationwise interval recurrence encloses `W^j*u/j!` through order 18. The
exact direction remainder has norm at most `3*theta^19/19!` and the integrated
position remainder at most `length*3*theta^19/20!`, since `exp(theta)<3` for
`theta<=1`. Recurrence, remainders, normalization and reported norm/radius
arithmetic are outward under tsinterval assumptions. The remainder uses the
mathematical initial unit norm; interval dependency is retained separately in
the series enclosure. No fitted discrepancy or step-doubling proof is used.

The receipt retains interval position/direction, stored centers and computed
radii. The direction center means mathematical normalization of its stored raw
vector, not an exactly unit rounded vector. Tail values alone are NOT evaluation
errors: full radii also retain interval/normalization/arithmetic uncertainty.
A positive displacement absorbed at a large coordinate produces a finite set,
not a zero-motion certificate. Unsupported overflow/normalization fails closed.

`tshelix.difference(receipt, x, direction)` bounds errors versus explicitly
supplied evaluated values (including an existing core-helix result). It does not
assume that evaluation was accurate, nor bind it to an existing Segment. The
bound includes initial raw-vector normalization discrepancy and final direction
normalization under the same mathematical convention. No implicit field-query,
carried-state, geometry or varying-field-model error is included. Field error
belongs to a separate model account; do not silently change Segment declarations
or promote isolated endpoint bounds into a bound for every point on a segment.

`beam/art/tshelix_audit.py ROOT --out NEW_RECEIPT.json` uses seven existing maps
as frozen fields at chosen interior starts for 5 mm/100 MeV/c examples. It
compares existing core evaluations to computed ideal uniform-helix enclosures,
not measured physical trajectories or a varying-map transport. No defaults,
core, carry, material physics or device code change. Native agreement, decimal
conversion, transformed geometry, full continuous navigator and the practical
long-path model error remain open.

## Stored transformed-solid and finite-face guard arithmetic

`tsgeomiv.evaluate(solid, points: Interval, max_nodes=..., max_depth=...)` encloses
an explicitly supported solid expression on an entire Cartesian position box.
Placements use their exact STORED matrix/origin as an affine pullback; tolerance
acceptance by Placed does not prove exact orthogonality or a unit Lipschitz
constant. No subtraction of a ball radius from a supposedly 1-Lipschitz value
is used. Primitive operations, nested placements and min/max CSG are evaluated
outward directly. A strict interval sign is a stored-expression result, not a
native ownership/boundary or whole-world coverage certificate.

Supported forms include boxes, spheres, tubes, tori, cones, halfspaces, sectors,
constant polygon extrusions, the explicit CSG types and complete planar-face
remainders. Unknown subclasses and arbitrary clearance callables fail closed.
Work/depth caps bound expression traversal and polygon edges; input row count
is not bounded by these controls. No mutable source geometry is permitted during
the query. A zero interval or undecidable sign is not nudged or classified as loss.

Stored halfspace/sector normals define their scalar expressions directly. Their
construction-time normalization/trigonometric errors and intended plane angles
remain representation issues. Cone slopes and polygon edges/distances are
recomputed outward from defining stored radii/heights/vertices, not the earlier
rounded cached coefficients. Therefore this ideal defining-parameter expression
need not equal the exact-real reinterpretation of those older cached floats.
Polygon closest-segment projections are enclosed, and ray-crossing decisions
are certified per entire query box or marked ambiguous. Ambiguous winding keeps
both signs of the distance enclosure; it is never assigned a guessed interior.

`tsgeomiv.face_bounds(face, points: Interval, ...)` preserves the original Face
and guard identities. It composes the actual nested STORED affine pullbacks,
extracts the primitive-face covector and mathematically normalizes that covector
for signed-distance units. It bounds the difference from the existing extracted
stored plane uniformly over THIS query box, plus the normal-vector difference.
A stored matrix accepted near orthogonal can consequently have a nonzero plane
representation discrepancy; forward composition is not silently treated as an
exact inverse. These results do not bound decimal GDML conversion, intended
native transforms/tolerances, trigonometric construction or unmodeled siblings.

`beam/art/tsgeomiv_audit.py ROOT --out NEW_RECEIPT.json` reads both original exports,
evaluates each selected volume around its recorded placement origin, and bounds
actual antiproton-disk finite-face guards/plane discrepancies over chosen local
boxes. A volume origin is merely a diagnostic query, not necessarily an interior
point. No existing navigator, surface-entry, contact or carry defaults change.
Direct interval guard bounds are prerequisites for future whole-path integration,
not a completed material entry/exit algorithm or physics validation.

`face_bounds` requires an unchanged Face produced by `tssurface.faces`; extraction
provenance and agreement of a manually supplied/replaced guard are caller
preconditions, not established by its type check. Hand-assembled/replaced Face
objects are unsupported. Exact-rational regressions cover selected nested affine
box values, polygon box samples in both windings and axis-scaled/sheared/nested
plane discrepancies; finite sampling is regression evidence, not a proof for all boxes.

## Whole-interval stored-map path sets

`tstube.Context(map)` snapshots the stored metadata/nodes immutably and computes
outward all-node/all-edge bounds. `tstube.enclose(context, x, direction, momentum,
pdg, lo, hi, position_error_mm=..., direction_error=...)` supplies a separate
hypothetical incoming-vacuum path SET for every parameter in `[lo,hi]`, measured
from the ORIGINAL input. It never advances a State or resets uncertainty.

Initial position error is an assumed Euclidean ball around exact stored x.
Initial direction error is an assumed chord distance from mathematical
normalization of stored raw direction to the true unit direction, not an angle.
True scalar momentum and charge equal the supplied exact model values. Decay,
uncertain momentum, loss/scattering and material dynamics are not included.

The entire reachable Cartesian box x +/- (ex+hi) is checked against exact stored
grid domains by tsquery. Reflected-seam continuation additionally needs exact
zero alignment and exactly zero stored By on every seam node; a gap or nonzero
seam is rejected, not welded. The initial nominal field error is COMPUTED by
tsquery, separate from its broader uncertain-region field-range bound. The
whole-map Lipschitz bound, initial ex/eu and computed eB feed tsbudget's outward
carried Ex/Eu polynomials, including G*ex coupling, direction drift and caps.
No error contribution is inferred from small test discrepancies.

A qualified tshelix interval at a stored midpoint m covers ideal frozen-field
R(m),v(m). Unit speed covers R(s) within dt=max(m-lo,hi-m); frozen curvature
covers v(s) within k_frozen*dt. Inflate these interval sets by the whole-interval
model Ex(hi)/Eu(hi). Nondecreasing bounds justify using hi even when lo>0.
The endpoint has its own qualified interval and model inflation. All difference,
radius and inflation arithmetic is outward under the existing IEEE assumptions.
Unit directions may be intersected with coordinate bounds [-1,1]. No uncertified
rounded point is substituted for a reference set. This construction can be broad;
it does not yet solve practical long-path precision or implement adaptive stepping.

`tstube.guard(tube, face)` evaluates the ORIGINAL finite-face guard and primitive
plane directly over the WHOLE Cartesian position set using tsgeomiv. It makes no
1-Lipschitz clearance inference. A positive guard is not proof of entry, unique
contact, material crossing or world-sibling exclusion. The hypothetical vacuum
continuation beyond first contact is not actual material motion. Initial-error,
intended/native geometry, decimal/trigonometric representation and native field
selection/cache assumptions remain explicit; no full certification is promoted.

`beam/art/tstube_audit.py ROOT --out NEW_RECEIPT.json` reads original exported
antiproton-disk faces and a stored map for chosen short incoming-vacuum intervals.
Existing carry/surface/contact/vacuum defaults are unchanged. Endpoint sets are
not restart states; no material transport, acceptance or throughput is measured.
