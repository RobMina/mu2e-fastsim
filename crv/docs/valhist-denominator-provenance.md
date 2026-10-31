# Donor/generated energy and missing-support provenance

`valhist/denominator_provenance.py` is a **read-only truth-provenance diagnostic** for a sealed
`library_variants` donor build. It writes one new JSON report, never a library or revised plan. It does not
run transport, response, reconstruction, generation or bootstrap, and it does not authorize denominator
variants. Missing support is not repaired by merging, smoothing, clipping, alternate masses, momentum-derived
KE, omitted rows or a replacement physics checkout.

## Concrete source findings

Reviewed canonical `crvfastsim/tier1b.py`, `crvfastsim/analyses/census.py`, core `tables.py`,
`art/CrvFastSim/src/CrvStepTableDumper_module.cc`, and the current workspace Offline
`EventGenerator/src/CosmicCRY.cc`, `EventGenerator/defaultConfigs/extractedCRYconfig.txt`.
The current Offline sources/configuration are **review evidence, not certified historical production inputs**.

1. The dumper writes `gen.e = GenParticle::momentum().e()` and generator momentum components as **double**.
   It writes `particles.se = SimParticle::startMomentum().e()` and its start momentum as **float**.
   It writes `gen.idx` as the index of the configured `cosmicTimeOffset` collection; `particles.genidx` is
   `p.genParticle().key()`. The referenced **ProductID is not persisted**. Thus `(event, genidx)` equality
   cannot by itself prove that the pointer belongs to the dumped collection. PDG, momentum, position, time
   and energy checks are discriminants, not a replacement for collection provenance.
2. `_build_one` traces step particles through parents to the primary. Its shower PDG comes from the
   **SimParticle primary**, while its energy/time come from that donor file's `gen` join on full
   `(run, subrun, event, genidx)`. The formula is `donor_gen.e - MASS[abs(primary.pdg)]`.
   It does not use SimParticle start energy, CRV-step momentum or deposited energy. The joined generator
   PDG is called `gpdg` but is not checked against the primary PDG. Missing joins are not explicitly
   rejected in the builder; NaN KE is assigned the last bin by the current search/clip helper.
3. `_reach` and `_showers_for_gen` instead subtract `MASS[abs(gen.pdg)]` from their respective external/evaluation
   `gen.e`. The mass table is shared; neutron/proton values are 939.565/938.272 MeV. Unknown species get
   zero mass. A PDG identity mismatch can therefore make donor and denominator conventions disagree even
   though both use the same mass table.
4. `_ke_bin` searches **right** in 19 geometrically spaced edges from 1 to 1e6 MeV, then clips to bins 0–17.
   Bin 0 includes every finite KE **below 2.154434690 MeV**, including negative KE and underflow below 1.
   Interior edge equality goes into the bin to its right. Bin 17 includes high overflow. The audit exposes
   the raw KE and unclipped index; it does not reinterpret native counts.
5. In the reviewed `CosmicCRY.cc`, `secondary->ke()` is MeV kinetic energy, the minimum-energy cut is applied
   to that quantity, and persisted total energy is `ke + ParticleDataList::mass()`. Momentum is calculated
   from that total energy and the same mass. The current extracted config sets **minShowerEn = 50 MeV**.
   Fifty MeV is in bin 5, not bin 0. Therefore genuine sub-2.154 MeV primary nucleons are inconsistent with
   that cut **if the donor production actually used it**. This is a concrete reason to check donor producer
   configuration/collection identity, rather than simply requesting more of the same generator bank.
   The rounded native nucleon mass constants cannot plausibly account for a ~48 MeV discrepancy under
   an otherwise consistent mass convention; inferred masses and momentum-based KE test that assumption.

The previous production audit reported nine missing cells / 61 showers: electrons bins 12–15, photons 13–14,
neutrons 0 and 14, protons 0. The high cells cover KE 10–100 GeV for electron bins 12–14, 100–215.443 GeV
for electron bin 15, 21.544–100 GeV for photons 13–14, and 46.416–100 GeV for neutron bin 14.
Before execution, finite generator-bank tails were only a hypothesis. The remote audit below instead finds
PDG identity mismatches in every one of those unsupported shower occurrences; that hypothesis must not be
promoted to a reason for extending the generator bank before identity repair.

## Executed frozen-library result

`thread_b_denominator_provenance_v3_20261025.json` completed on GPVM03 using the original frozen uncertainty/core
imports, all sealed hashes unchanged, with exact ordered shower projection and nominal libgen closure. It reads
14 unique selected donor files (20 occurrences), 45,891 unique / 65,742 occurrence-weighted showers, and ten
external generator files / 500,000 persisted generated events. Per-donor reads took approximately 1.1–1.3 s;
per-generator reads approximately 0.09 s, excluding the surrounding hash checks.

All nine support holes are absent in raw external counts, none rounded to zero. **All 61 shower occurrences
(43 unique) in those cells have mismatched primary/gen PDGs.** All 25 bin-0 nucleon occurrences (19 unique)
have negative native KE, while diagnostic SimParticle start KE spans 55.393–2414.419 MeV. Unsupported high-bin
showers have diagnostic start KE 50.262–1861.149 MeV. Across the whole selected population, there are 209 unique /
296 occurrence-weighted PDG mismatches; no missing join. The compact `provenance_summary` preserves both unique
and occurrence-weighted counts. No row is removed or reinterpreted.

The immediate gate is to recover complete referenced GenParticle collection identity and fix the association,
then rebuild a separately versioned library and reassess genuine support. PDG equality alone is insufficient:
the retained report also contains same-PDG rows with substantially different start/generated kinematics.
Do not substitute start energy, smooth/merge bins, or request more of the same bank as a cure. The missing
ProductID prevents identifying the exact historical collection/remapping mechanism from these flat tables alone.
The existing donor-only variances remain frozen-implementation diagnostics, not a calibrated physics systematic.

## What the report establishes

- Accepts only a checksum-sealed donor build with its sealed sibling `plan.json`; uses the existing strict
  `denominator_variants._base` validation. Nominal inputs, donor/generator banks, geometry and actual
  imported physics/core/helper identities must remain those bound by the original audit.
- Reads highest ROOT cycles and records all available cycles and branch types for each consumed tree.
- For each **unique selected donor file**, traces ancestry with the bound vectorized `primary_of`, reproduces
  the builder's shower ordering and primary/gen join, and retains **every reconstructed shower row**, not
  just a few examples. Each row has full event/primary/gen identities, original persisted SimParticle fields,
  all original GenParticle fields with `gen_` prefixes, native mass/KE/bin, raw bin index, mass-shell residual,
  inferred invariant mass, momentum-derived KE/bin, join/PDG flags, and start-minus-gen energy/momentum/time/
  position differences. Existing serialization precision and a different transport stage can affect these
  differences; they are not all asserted to be errors.
- Retains summaries for **all nonmuon rows of each selected donor's accepted-event gen tree**, separately from
  the shower-producing subset. That accepted population must never substitute for complete generated exposure.
- The original ordered donor selection and file multiplicities reconstruct library-global shids: add the
  cumulative shower count of earlier occurrences to the per-file shid. Repeated bootstrap donor files count
  again. Requires **exact ordered `shid/apdg/kebin/ke` closure against persisted `libshower`**, without rerunning
  the physics builder. This closes the provenance projection, not all donor payload trees.
- For every original external generator file, records full input identity, highest-cycle/type metadata,
  complete persisted event count, species/bin counts, energy/mass/KE ranges, underflow/overflow and anomalous
  counts. Raw rows retain each bin's KE extrema, all negative-KE/unknown-mass/bin-disagreement cases and all
  bin-0 nucleons. Those rows retain full original generator columns and full-width integer event IDs.
  External raw-row examples are not a full generator-table export; the input SHA256 binds the complete source.
- Counts external nonmuon primary rows and scales with the original donor generated exposure using native
  `rint`. Separates **zero raw external rows** from **positive raw counts rounded to zero**. Checks the exact
  nominal `libgen` formula and `generated_nonmu` metadata. A persisted histogram/binning discrepancy fails
  publication instead of masquerading as sparse support.
- Rechecks unchanged base/plan/input/source/driver/evidence identities before exclusive new-file publication.
  Existing outputs, dangling symlinks and code-tree outputs are refused. Inputs are opened read-only.
  No manifest is refreshed or retrospectively certified. The report itself is not a signed certificate;
  only a completed successful invocation should be consumed.

Species/bin operations are vectorized inside each file. Python loops are over files, selection occurrences,
small bin groups and the bound ancestry-depth traversal, never individual events or particles. Per-file elapsed
seconds are printed. No production runtime estimate is inferred from synthetic tests. Memory includes the
selected donor shower projections and external species/bin columns; full generator/step tables are processed
one source file at a time, and retained report size grows with the full donor shower census.

## Remote invocation: keep the bound physics, run only the new driver

The canonical development physics has changed independently of the accepted Thread B audit. **Do not launch
`python -m valhist.denominator_provenance` from canonical crv against the historical product**, waive its hash
checks, or copy this driver into the frozen checkout. Execute the canonical new file directly while cwd and
PYTHONPATH select the original uncertainty checkout and frozen core. Absolute imports intentionally support this.

On a free GPVM, after checking host state and that the code is in place:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && pyenv ana
DRIVER=/path/to/mu2e-fastsim/crv/valhist/denominator_provenance.py
FROZEN=/path/to/frozen/uncertainty/crv
CORE=/path/to/frozen/core
# CRVFS_DATA must be the same data root as the retained Thread B audit.
LIBRARY_JSON=$(jq -r '.base.json.path' \
  "$CRVFS_DATA/valhist/thread_b_denominator_audit_20261025.json")
OUT="$CRVFS_DATA/valhist/thread_b_denominator_provenance_20261026.json"
cd "$FROZEN"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FROZEN:$CORE" \
  python3 "$DRIVER" "$LIBRARY_JSON" --out "$OUT"
```

The output parent must already exist and `OUT` must be new. If the retained report path differs, read its
`base.json.path`, rather than guessing a bootstrap-library directory. This does not need held-out replay files
or new generator production. It does need **every original donor/generator/nominal file** named by the sealed
plan for hash checks, the selected donors' `dump/{particles,gen,steps}`, every external `dump/gen`, base ROOT/JSON,
plan JSON plus checksum sidecars, and the unchanged geometry/source paths. Production-data content cannot be
recovered from notes or summaries. Missing files or source drift are blocking conditions, not reasons to
redirect dependencies to current physics.

Optional `--evidence FILE ...` hashes producer/configuration review snapshots before/after. Useful candidates:

- the exact donor and generator dumper source/FHiCL used to write the ROOT files;
- actual expanded historical donor and external generator FHiCL, CRY config text, ParticleDataList/global
  constants, release/backing identity, and original art product provenance/collection tags;
- the reviewed `CosmicCRY.cc` and `extractedCRYconfig.txt`, clearly labeled current snapshots if historical
  versions cannot be established.

These files are not parsed into a claim of production configuration equality. Do not label the current `off/`
files or current dumper source as historical certification. This module never reads art through gallery/ACLiC,
rebuilds a dumper or re-dumps art; collection provenance not persisted in the flat tables remains an explicit
follow-up input need.

## Interpreting low-KE nucleons without a repair

1. Require exact shower and libgen closure. Failure means do not attribute the report to the retained library.
2. Inspect raw donor `gen_e`, `gen_p*`, `gen_pdg`, primary `pdg/genidx`, mass-shell diagnostics and the full
   SimParticle start quantities. Negative KE folded into bin 0, unknown mass, mismatched PDG, failed join,
   nonphysical invariant mass or a different diagnostic momentum bin are concrete convention/identity clues.
3. If the native total-energy and momentum KE both genuinely lie below 2.154 MeV with the correct nucleon
   mass and matching primary identity, check the historical donor minimum energy and production stage versus
   the external bank. More samples under a strict 50-MeV source cut cannot supply that physical region.
4. If donors and external generator use matching production conventions and their kinematics close, zero raw
   rows establish **missing support in this finite bank**, not zero probability in nature. Tail coverage needs
   additional independent, configuration-compatible inputs or an explicitly validated support model.
5. A positive raw count scaled to zero is an exposure/rounding issue, not an empty source bin. A library count
   disagreement is a persisted formula/provenance problem. Keep both separate from genuinely absent raw rows.

Momentum-derived quantities and inferred masses are **truth diagnostics**, not new response-conditioning
variables or replacement generator energies. No outcome here approves changing the frozen physics or strict
support gate.

## Synthetic tests

```bash
cd /path/to/mu2e-fastsim/crv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../core \
  python3 -m pytest -p no:cacheprovider -q \
  tests/test_valhist_denominator_provenance.py tests/test_valhist_denominator_audit.py \
  tests/test_valhist_denominator_variants.py tests/test_valhist_library_variants.py
```

Tests use real tiny ROOT TTrees: exact edge/flow behavior, negative total-minus-mass KE, unknown masses,
particle-sign handling, float32 start versus double gen, highest cycles, descendant-to-primary identity,
missing/duplicate/mismatched joins, full-width JSON identity, missing versus rounded-to-zero support,
sealed-base end-to-end closure/read-only behavior, output refusal, physics/input drift and projection failure.
A separate synthetic check compares the projection with **real `tier1b._build_one`** using an in-memory tiny
geometry, without transport, download or production execution. The CLI was also checked by executing this new
canonical file with the locally mirrored frozen uncertainty/core paths. That only checks import/CLI viability,
not remote accessibility or production hashes.
