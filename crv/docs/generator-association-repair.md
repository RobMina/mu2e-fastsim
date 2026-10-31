# Generator association repaired; held-out shower/position closure remains open

> Paths in this record identify the original campaign, not a published input
> service or guaranteed readable copies. Use the umbrella collaborator setup
> with a publisher-approved checksum manifest; do not symlink private areas.

**Follow-up:** [corrected-shower-uncertainty.md](corrected-shower-uncertainty.md) separates the repaired library's
exact sampler expectation, finite donor statistics and corrected reference-source variation. The held-out total
step deficit is 0.85% at the production expectation versus 2.38% in the retained draw; T1 position and nonmuon
energy residuals remain. A 20-file full-generator census audit independently certifies the unchanged exposure.

## Mechanism and executed probe

`CompressDetStepMCs::compressGenParticles()` copies referenced GenParticles into its output collection and
rewrites SimParticle generator pointers to that collection's ProductID/key. The legacy dumper selects
SimParticles from `compressDetStepMCs` but generator rows from `cosmicTimeOffset`, saving only the key.
The resulting event/key join is not a valid association when compression changes the index.

A FHiCL-only probe changed the selected generator collection to `compressDetStepMCs`, using the existing,
unmodified satellite dumper on one complete production donor, sequence `001400_00000990`. It exited zero over
25,176 accepted events. All **814,872 step rows, 795,346 particle rows, and the subrun row are bit-identical**
to the retained flat table (legacy columns, row order included).

| Shower diagnostic | Legacy collection | Compressed collection |
|---|---:|---:|
| Retained showers | 3,315 | 3,315 |
| Primary/gen signed-PDG mismatches | 14 | 0 |
| Negative native KE | 2 | 0 |
| Same-PDG start/gen energy difference >1 MeV | 1 | 0 |
| Largest absolute start/gen energy difference | 26,970.258 MeV | 0.001739 MeV |

Fifteen shower associations change; fourteen change KE bin. The remaining energy difference is consistent with
float SimParticle energy versus double GenParticle energy. This establishes the repair for this file's
observed mismatches, **not closure of the nonmuon shower-yield discrepancy**. The legacy probe does not persist
ProductIDs, so final products must be re-dumped using the strict schema below.

Evidence: `$M2FS_DATA/crv_genassoc_20261027/probe/`, files
`crvsteps.001400_00000990.root`, `dump.001400_00000990.log`, `comparison.json`, `comparison.log`.
Probe binary SHA256 `1f924c9cfb2629f8106a6d4743516239344debe12468a3df8330470eae33679a`;
source SHA256 `4f6216079bfe1d99405bb09bc20625a7b0da50f23d9b4ca8ca53027af5daf644`.

## Permanent contract

The changed dumper validates **before filling any event payload**:

- Every step SimParticle pointer and nonnull parent pointer must resolve in the selected SimParticle product.
- Every nonnull generator pointer must have the selected GenParticle ProductID and an in-range key, and resolve.
- Referenced and selected generator PDGs must agree. SimParticle/gen signed PDG must agree for **true primaries**
  only; a compressed descendant can legitimately inherit an ancestor's generator pointer.
- Referenced generator energy/time must be finite; key/count narrowing to the legacy int fields is guarded.

The default `retained_primary` policy has no non-strict switch. An explicit staged-Cat policy has a separate
schema and preserves pointer validation without assuming stage-primary/generator PDG equality; see
[`veto-cat-primary-policy.md`](veto-cat-primary-policy.md). Schema-3 staged-Cat output is rejected by the
retained-primary cosmic donor gate. Failed jobs can leave partial files; only successful completion plus the Python
census guard authorizes use.

Schema v2 adds:

| Tree | Added fields |
|---|---|
| `events` (new, one row per input event, empty events included) | full event IDs; `schema_version=2`; `associations_checked`; `full_generator_stream`; selected `gen_product_id`, `sim_product_id`; `n_gen`, `n_particles`, `n_steps`; configured tags |
| `gen` | actual selected `gen_product_id` |
| `particles` | `sim_product_id`, actual pointer `parent_product_id`, actual pointer `gen_product_id` |
| `steps` | actual pointer `sp_product_id` |

Null pointers use ProductID zero and key -1. Counts are signed 64-bit; ProductIDs and event IDs unsigned 32-bit.
Existing physics columns keep their original types and values. Numeric ProductIDs are compared **within an event**;
no assumption that IDs match between independent files/streams is made.

`dump_crvsteps.fcl` and the explicit donor config `dump_crvsteps_genassoc.fcl` select compressed generators.
`dump_gen.fcl` explicitly retains `cosmicTimeOffset`, disables steps and declares `fullGeneratorStream:true`.
The accepted/compressed donor gen population is **never a generated-exposure denominator**.

The Python `crvfastsim.genassoc` gate requires the schema, checked flag, exact integer identities and census
counts, rejects broken references/cycles and requires every step to reach a true retained primary. It runs before
any library binning and also protects direct `_build_one` use. Old donor tables must be re-dumped; there is no
kinematic fallback or legacy bypass. Old libraries remain readable for retained-baseline comparison.
Historical complete-generator files are allowed as denominators with explicit **unverified legacy stream role**
metadata; schema-v2 donor tables cannot be used as denominators. Schema-v2 reach normalization uses the event
census, including zero-generator events, rather than only identities found in generator rows. The planned
historical external bank is kept hash-identical. Eligibility is the retained-primary cosmic donor model, not
arbitrary staged or genealogy-truncated SimParticle collections.

Local focused validation: **171 tests passed**, covering association schema/identity/ancestry, campaign plan and
build guards, tier-1b, step-layer diagnostics and denominator/library variants. Shell syntax and diff checks pass.

## User build gate

Only the isolated worktree satellite is changed. Main checkouts and their existing binaries are untouched.
Its `backing` symlink points to `/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay`.
Existing `SConscript` already lists canvas and cetlib_except; no library dependency change is needed.

Run on **mu2ebuild02**, then report that the build completed:

```bash
cd /path/to/crv/art
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
muse setup
muse build -j 32
```

The user confirmed a successful full build on 2026-10-27. The isolated plugin is newer than the source;
`.musebuild` records 06:31:32 to 06:31:44 CDT. Source SHA256 is
`dda6d5c3c194de6747dd43e05ef255b79d3ffa90a95ffd3ec1f5315fb97397a2`; plugin SHA256 is
`ff28a1f3963b82548c6ee66f7d6128e89c40e4a1396e3c24ec0953f4442cabe6`.

## Executed strict runtime gates

The strict first-donor run completed on gpvm05: **25,176 census events, 25,193 generators, 795,346 particles,
814,872 steps**, with successful pointer/census/ancestry validation (highest ROOT cycles), zero ancestry cycles,
and exact legacy step/particle/subrun payloads. The association diagnostics reproduce the probe table above.
The measured art wall time was **14.687 s**; Python validation/comparison is additional. A lost SSH return did
not mean the run failed: its immutable completion record and outputs were checked before proceeding.

`valhist.genassoc_smoke` then ran both configuration checks:

- Wrong donor generator tag `cosmicTimeOffset`: expected art exit 1 at the ProductID guard, before any event,
  generator, particle or step row was filled.
- Full-generator input shard 2: successful 100-event / 101-generator smoke, schema-v2 full-stream role and
  `cosmicTimeOffset` tag, with every legacy generator value identical to the retained full-generator table.
  This is a configuration smoke, not a replacement exposure bank.
- Dynamic-loader initialization traces prove all three jobs loaded the exact isolated plugin above, not the
  main satellite. Evidence is in `strict_v2/logs/loader_refreshed.*` and `strict_v2/smoke/smoke.json`.

The initial credential failures were stale gpvm05 SSH-master credentials, **not an expired local ticket**.
Refreshing only that master restored home access. The existing `fpana` tmux session was left untouched;
subsequent jobs use the fresh connection. Failed setup logs remain as provenance and contain no donor output.

All 20 donors completed sequentially on gpvm05 with loader traces retained. `valhist.genassoc_report donors`
verified all completion records, matching source/binary/plan bindings, exact legacy payloads, the expected plugin
trace count and zero corrected association diagnostics. `library-invariants` compares `libbins`, `libxing` and
`libdesc` exactly against the retained production library before replay. The sequential Python-only follow-up
launcher is `scripts/run_genassoc_followup.sh ROOT` (with `M2FS_CORE` set); it does not compile anything.

Current local selection: **215 tests passed**, including the new loader-evidence and aggregation tests; shell
syntax and diff checks pass. This is a selected suite, not a new claim about the full inherited test suite.

## Campaign sequence

Plan: `$M2FS_DATA/crv_genassoc_20261027/strict_v2/plan.json`.
It names all 20 original production art paths recovered from successful dump logs, the old table hashes,
**500,000 generated events per file / 10,000,000 total exposure**, and the unchanged full-gen denominator bank.
There was no dCache directory scan. Final outputs have new paths; legacy tables/libraries/baselines are preserved.

After checking the completed build, host occupancy and quota:

1. Run one strict donor (`--first 1 --last 1`), require successful art exit and schema validation, and repeat the
   legacy-payload/association comparison. Also run a negative config selecting `cosmicTimeOffset`: it must fail
   at the collection-ID guard on an affected event. Confirm the gen-only config still reads the full stream.
2. Re-dump remaining donors sequentially, or disjoint file intervals on otherwise idle GPVMs. At most one job
   per GPVM. `art/run_genassoc_redump.sh PLAN FIRST LAST` sets up the existing built area and invokes
   `valhist.genassoc_campaign redump`. It never builds. It records command, binary/source/config hashes,
   output hash, observed runtime and successful validation. Incomplete output is refused, not overwritten.
3. In the analysis environment, run `python3 -m valhist.genassoc_campaign rebuild PLAN`. This validates every
   corrected donor, rebuilds the library with the unchanged external bank and exposure, and rejects missing
   support or probabilities above one before publication. It does not merge bins, discard bad rows or tune rates.
4. Run `python3 -m valhist.stepclosure shower-audit ROOT/library/library.json --out ROOT/library/association_audit.json`.
   Require zero generator mismatches/missing joins and understand every support limitation before replay.
5. `scripts/validate_genassoc.sh ROOT 11-20` reuses the saved stage-C transport and CCP baseline, applies the
   corrected library with the same seeds/options, checks all muon-family payloads stay identical, checks exposure,
   and computes step positions/origins, V1/V3 and NumPy response. No G4 or baseline transport is regenerated.
6. If the residual closes or changes, quantify it on the same held-out events before considering all 20 shards,
   stock-response DQM and integration with newer main. The retained NumPy response version is the old worktree
   version; it is not a claim about the newer opposite-side implementation.

Example analysis imports: `PYTHONPATH=$LAYERS:$CORE`, with the existing core checkout and `pyenv ana`.
The re-dump wrapper takes `M2FS_CORE`; never invoke it from an unrelated active muse work area.

## Completed campaign and held-out result (2026-10-27)

**Decision: the extraction defect is repaired for this campaign; the shower/position discrepancy is not closed.**
The sequential follow-up exited zero; no campaign process remains. No rate tuning, row deletion, energy
substitution, baseline regeneration or main-checkout change was used.

All 20 production donors contain **499,702 accepted events, 500,062 retained generators, 15,477,323 particles and
15,928,094 steps**. Accepted-event counts are not the generated denominator: the planned **10,000,000 generated
exposure** and all ten external full-generator input hashes are unchanged. All legacy step/particle/subrun values
and row order are exact. Actual-plugin loader traces are present for every donor. Measured art wall times range
from 12.85 to 14.69 seconds per file; that excludes validation/comparison.

| All-donor shower diagnostic | Legacy | Corrected |
|---|---:|---:|
| Retained showers | 65,400 | 65,400 |
| Primary/gen signed-PDG mismatches | 279 | 0 |
| Negative native KE | 24 | 0 |
| Same-PDG start/gen energy difference >1 MeV | 31 | 0 |
| Missing generator joins | 0 | 0 |
| Unsupported showers | 61 | 0 |
| Maximum absolute start/gen energy difference [MeV] | 352,507.0759 | 0.0017863 |

There are 310 changed shower associations/times and 280 changed KE-bin assignments. The corrected library passes
support/probability checks, and its ordered shower projection is exact. The `libbins` (36 rows), `libxing`
(2,655,560 rows), and `libdesc` (8,733,057 rows) are bit-identical in values to the retained library. Corrected
library SHA256: `ebb015a76d56edcf96c5bc533daa96729e3851ee3e59185abf0be5fd469257c7`.

### Held-out shards 11-20: 500,000 generated events

The same saved stage-C transport, per-shard seeds and context+containment+point options were replayed. The full
exposure/identity check passes. All muon-family step payloads (including descendants) remain identical, excluding
only the event-wide `nstep` count, which changes when showers change. **All V1 gates pass**. Its printed 373,998
union-particle-event count is not the generated exposure.

| Sector | Nonmuon energy fast/G4, old -> corrected | Total step deficit, old -> corrected | Step-y diagonal score, old -> corrected |
|---|---:|---:|---:|
| EX | 0.9182 -> 0.9406 | -1.78% -> -0.88% | 1.47 -> 0.85 |
| T1 | 0.8745 -> 0.9515 | -7.74% -> -5.08% | 4.81 -> 2.76 |
| T2 | 0.8641 -> 0.8787 | -3.75% -> -4.67% | 2.06 -> 3.06 |
| M | 0.9623 -> 0.8193 | -5.08% -> -10.15% | 1.62 -> 3.09 |

Total steps are **770,793 old -> 776,654 corrected**, against **795,615 G4**: -3.12% -> -2.38%. Sector y/z flags
at score >2 change **2/8 -> 3/8**; corrected z scores are EX 1.31, T1 1.00, T2 0.69, M 1.02. Muon-descendant
other-bar energy per crossing is exactly unchanged and remains low: EX 0.0854/0.1009, T1 0.1045/0.1537,
T2 0.0693/0.1054, M 0.0342/0.0790 MeV (fast/G4).

### NumPy response on the corrected steps

| Sector | Cluster y/z diagonal score | Cluster count difference (paired ratio SE) |
|---|---:|---:|
| EX | 1.02 / 0.50 | -0.18 +/- 0.33% |
| T1 | 0.57 / 0.24 | -1.19 +/- 0.88% |
| T2 | 1.14 / 0.60 | -0.49 +/- 0.69% |
| M | 0.97 / 0.85 | -3.05 +/- 1.20% |

All eight cluster diagonal scores are below 2, and all position flow counts are zero. This is **not** a blanket
response pass: M's paired z score is 6.26 with one occupied bin, and the M cluster deficit remains. These are the
old worktree's NumPy response, not validation of newer main/opposite-side code.

Changing shower bins can reorder donor selection and change multiplicities; matching seeds do not give aligned
nonmuon random draws. The observed sector changes are one fixed-library replay, not a calibrated causal effect
or finite-library uncertainty. The discrepancy must remain open. A subsequent study should quantify corrected
library/sampling uncertainty and source representativeness, and separately address the unchanged other-bar
muon-descendant model; do not tune rates to absorb either residual.

### Evidence and verification

Under `$M2FS_DATA/crv_genassoc_20261027/strict_v2/`:

- `donor_summary.json`, `comparisons/*.json`, `logs/dump.*.done.json`: exact donor comparison, strict validation,
  immutable hashes and actual loader evidence.
- `smoke/smoke.json`: negative ProductID and full-generator configuration gates.
- `library/library.{root,json}`, `library/muon_library_guard.json`, `library/association_audit.json`:
  completed corrected library, frozen geometry/input provenance, invariant muon tables, zero support defects.
- `validation/s11-20/{exposure,muon_family_guard,position_comparison,response}.json`, `v1/`, `v3/`,
  `ccp/breakdown.json`, `breakdown.txt`, `clusters.txt`: held-out checks and response.
- `closure_summary.json`: old/corrected physics comparison; `logs/followup.exit` is **0**.

Local selected suite: **215 passed**; shell syntax and `git diff --check` pass. No default promotion, C++ rebuild
or grid job was part of this change.

## Acceptance and limits

Association repair and physics closure are separate gates. Zero pointer/PDG errors and complete exposure close
the extraction defect. The yield/position/response comparisons decide whether that repair explains the shower
deficit. If the deficit survives, retain it as an open library-model discrepancy rather than applying a rate
correction. The known event/bar-only multi-muon descendant grouping is not changed in this patch.
Finite-library uncertainty is not covered by event-paired error bars or the diagonal deck score.
