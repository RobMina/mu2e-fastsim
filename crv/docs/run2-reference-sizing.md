# Run 2 reference sizing: production-reuse decision required

**Hold new production pending a reuse audit.** SAM contains
`sim.mu2e.CosmicDSStopsCRYAll.MDC2030af.art`: 17,841 files, 1,902,797,992 retained events (2026-10-29 query).
Its exact production configuration uses `geom_run2.txt` / `nominal_v10` and default CRY; top-level geometry,
CRV counters and CRY input bytes match the proposed reference. It is S1 DS-entry-filtered transport, stopped at DS
entrance, not the full field-off, all-generated-event reference. Inspect retained steps, normalization, selection
and field/material coverage before deciding which validation quantities can reuse it. Lack of all-event primaries
prevents exact paired all-generated-event replay, but does not by itself justify a new large G4 campaign; an
independent, correctly normalized production comparison may answer the physics question.

The 20M sizing below is conditional, not evidence that the extra production is needed. The one test job
failed namespace preflight before Geant4 and produced no physics output; no retry or additional phase is planned
until reuse is evaluated and a residual need is demonstrated.

Reuse the existing 1,000,000 generated-event reference. A read-only replay on current software gives exact V1
(event/bar) event-clustered statistical standard errors of 2.183% for C crossing rate, 1.196% for C length and
0.516% for U length. Under independent sampling from the same population, statistical design scaling gives:

| Total generated | C crossing SE (%) | C length SE (%) | U length SE (%) |
|---:|---:|---:|---:|
| 1M retained | 2.183 | 1.196 | 0.516 |
| 10M | 0.690 | 0.378 | 0.163 |
| 20M | 0.488 | 0.267 | 0.115 |

These are statistical planning estimates, not runtime projections, guarantees of passing the 1%/0.5% gates,
or evidence of individual sparse-sector coverage. The current scatter-in algorithm still needs convergence and
large-angle validity control; additional reference events must not hide that uncertainty.

A 20M target means **19M additional generated events**, or **380 retained-size 50k chunks**. Existing source and
seed/subrun ranges must be audited for disjointness before commissioning any extension. This note does not start
a new G4 campaign. Existing 50k Run 2 G4 evidence measured 1:08:52 wall and 2,033,344 kB maximum RSS on its worker;
retain that measurement without extrapolating a campaign runtime or SU. The extension runs on the FNAL grid,
using the already proven 1 CPU, 3000 MB RAM, 4000 MB disk and 4h lifetime request. A new grid test must verify
runtime, memory, output bytes and successful pnfs copy-out before the two pilots and remainder. No build or tarball
is needed: the original SimJob MDC2025ay Musing remains on CVMFS. See [grid setup](run2-grid-extension.md).

## Storage

The retained full 1M Run 2 generator/CrvStep set, logs, manifest and GDML occupy **647 MiB** in the shared input
area. Scaling that conservative retained-set size by 19 gives about **12.0 GiB** for the additional 19M-derived
tables/evidence, before art or response products. Reserve **at least 16 GiB** for that set and its validation summaries;
fixed-stock response replicas, original art and per-node CVMFS caches need separate measured allowances.

**The production destination is pnfs scratch, not the quota-limited exp data area.** Grid workers write dts/gen art,
logs and completion receipts directly to pnfs; the exp area holds only small submit/control records. Its 12.84 GiB
free quota is therefore **not a blocker for grid reference production**. The 16 GiB estimate above concerns optional
derived tables if retained together, not an exp reservation. Dump inputs directly from pnfs and move bounded
postprocessing to UVA, or persist derived products on pnfs; never bulk-copy the whole extension into exp. Scratch
on either site is purged and is not archival storage. Reuse staged inputs and never delete another campaign's outputs.

## Fixed-stock V2 floor

The retained Run 2 stock reference is Offline v13_39_00/p107, not the fixed release. The validated p109 3×3000
fixed-stock demonstration is **Run 1A**, not a Run 2 population. A Run 2 floor needs at least eight independent
fixed-stock response seeds and corresponding fast seeds with explicit input-event identity, timing/condition
bindings, spectra and prospective two-arm optical/runtime evidence. That is a separate response campaign; it does
not require new G4 if the existing Run 2 dts input is reused. Until those products exist, the named V2 result stays
unsupported rather than rebranding the old K=4 conditional fast-only seed spread as a full seed floor.
