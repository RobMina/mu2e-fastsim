# Shared generator-denominator variants (Thread B)

`python3 -m valhist.denominator_variants` creates **denominator-only** whole-generator-file bootstrap
libraries from an existing, audited `library_variants build` product. It does not rebuild donors, run
transport/reconstruction, submit jobs, or modify physics, nominal libraries, or old sealed manifests.
All products go into **new directories outside the code tree**. This is a conditional diagnostic,
not a confidence interval or a DQM decision.

## Source review and scope

The reviewed `crvfastsim/tier1b.py` dependency chain is:

- `build`: `_bin_edges(step_paths, ...)` and `_build_one([donor], ...)` make all five donor trees.
  The external `gen_paths` argument is used only by `_reach`, after those trees have been assembled.
- `_reach`: counts nonmuon primary **rows**, grouped by absolute PDG and kinetic-energy bin; counts
  distinct complete `(run, subrun, event)` tuples **before filtering muons**; scales and rounds the
  primary counts to the explicitly asserted donor generated-event exposure. Its other output is
  the unscaled `generated_nonmu` metadata count.
- `_build_one` does read the **donor file's own** `gen` tree to define shower primary kinetic energy,
  bin and generation-relative times. That is a different input from the external denominator bank;
  those donor payloads must remain fixed in this denominator-only operation.
- `_showers_for_gen` uses `nshower / ngen` for nonmuon shower acceptance, with absent/zero denominators
  treated as zero. The following conditional shower draw uses the fixed donor shower population.
  Muon crossing bins, descendant sampling, geometry and quantile edges do not use `libgen`.
- `read_library` delegates to the core table reader, which reads the highest cycle of each `dump/`
  tree. The new writer copies the entire base ROOT file and appends a new highest-cycle `dump/libgen`
  TTree. All five donor trees and their cycles/payloads remain copied unchanged. The previous `libgen`
  cycle is retained for history, **not** an additional denominator to sum. Standard loading sees only
  the new cycle. This also avoids a physics rebuild or re-estimating donor quantile edges.

Accordingly replacing `libgen` and its generated-nonmuon count is sufficient for the external-bank
perturbation. This is checked against the base library's actual `libgen` before any plan is published.
All selected-code/source hashes remain strict. The nominal historical build identity remains only
as certified (or unavailable) in the old audit; this tool does not retrospectively certify it.

Only sealed `library_variants` build products with their unchanged sibling `plan.json` are accepted.
An unaudited nominal ROOT/JSON pair, or a previously produced denominator variant, is not accepted as
base. Half or bootstrap donor builds are acceptable **fixed bases**; their selection is recorded.
There is no joint donor/denominator generation API. Applying this tool to independently assembled
donor replicas is possible, but would require a separately designed joint experiment and must not
be labeled a joint bootstrap by this module. In particular it supplies neither crossed-replica
pairing nor joint uncertainty decomposition.

## Formula and resampling unit

For source generator file `f`, let `E_f` be its complete unique generated-event count (including
muon-only events) and `C_fb` the number of nonmuon primary rows in species/energy bin `b`.
Let `G` be the existing donor library's asserted generated-event exposure and `S_b` its shower count.
A replica draws the original number of generator **files** with replacement. With file multiplicity
`m_f`, it stores

```text
E*       = sum_f m_f E_f
C*_b     = sum_f m_f C_fb
ngen*_b  = np.rint(C*_b * (float(G) / E*)).astype(int64)
nshower_b = S_b
p*_b     = S_b / ngen*_b       (for positive supported denominators)
```

This is the ratio of weighted totals, not an average of file rates. Heterogeneous file event counts
are allowed and weighted as shown; exchangeability of such production blocks is still an assumption.
The kinetic-energy bin calculation uses the existing `tier1b.MASS` (unknown species mass zero) and
`tier1b._ke_bin`, including clipping at the outer edges and NumPy ties-to-even rounding. Bins with
no generated primary rows are absent, matching the existing builder. Generator-only bins have zero
showers. Unsupported shower bins, zero rounded denominators for positive shower populations, and
probabilities above one **fail the requested build**. There is no clipping, pseudocount, redraw,
replacement with nominal values or silent omission of a failed replica. An all-nonmuon-empty draw
also fails. Record such failures in any statistical campaign; analyzing only successful replicas
would condition the distribution and can bias it.

All primaries and events of a file share the same multiplicity. No per-primary, per-event or
per-evaluation-shard resampling is performed. Event keys are deduplicated **within original files
only** when measuring exposure. Original cross-file duplicates are rejected by exact tuple audits.
Bootstrap copies intentionally repeat the entire event population and must repeat its exposure.

**Do not call `tier1b._reach` directly with repeated filenames.** Its global event-ID deduplication
would retain repeated primary rows while collapsing repeated-file event exposure. The new module
instead aggregates vectorized per-file sufficient statistics. Tests compare its formula both to
direct counts and to the original helper on synthetic repeated files whose full occurrence keys
have been relabeled, demonstrating this distinction without changing physics code.

## Public Python API and CLI

```python
plan(library_json, outdir, held_out, generator_events, held_out_events,
     replicas=1, seed=0) -> dict
build(plan_json, variant, outdir) -> dict
main(argv=None)
```

`library_json` is a sealed audited donor build with its same-stem `.root` and sibling donor `plan.json`.
All original sources, nominal inputs, geometry and code named by that audit must still be available
and unchanged. No old manifest is refreshed or resealed by these commands.

`generator_events` is a required list of positive integer complete-event counts in the **original
nominal `gen` list order**, not sorted filename order. `held_out` is a nonempty explicit file list;
`held_out_events` supplies one positive integer complete-event count in that order. Both inventories
must match their asserted counts exactly. This is an operator assertion of completeness: omitted
zero-primary events cannot be recovered from a primary table, and if such events exist a complete
event-census implementation is needed instead of this tool. No count is inferred from particle rows,
filename ranges, or surviving donor events. Donor exposure comes from the validated old audit.

```bash
# From crv/ in the uncertainty checkout, with data outside code:
export PYTHONPATH=.:../core

python3 -m valhist.denominator_variants plan "$AUDITED_DONOR_BUILD/library.json" \
  --outdir "$DATA/denominator_plan" \
  --generator-events 50000 50000 50000 \
  --held-out "$HELD11" "$HELD12" \
  --held-out-events 50000 50000 \
  --replicas 20 --seed 173

python3 -m valhist.denominator_variants build "$DATA/denominator_plan/plan.json" \
  --variant denominator_0001 --outdir "$DATA/denominator_0001"
```

The illustrative file counts above must be replaced by actual **verified complete exposure** for
every listed generator and evaluation file; they are not production results or inferred defaults.
Replica names are `denominator_0001`, etc. `replicas` is positive; `seed` is nonnegative. Each uses
`PCG64(SeedSequence([seed, one_based_replica, 1]))`; the extra domain separates it from existing
`library_variants` donor streams. Adding replicas preserves earlier selections. Ordered indices,
multiplicities and selected event exposure are frozen, together with NumPy/pandas/uproot versions.
For fixed inputs and environment the plan bytes and numerical denominator are deterministic. ROOT
UUID/time metadata need not be byte-identical across independent builds; each actual artifact is hashed.

**Use the same published `library.root` for every evaluation shard in a replica.** There is no shard
argument or runtime redraw in this API. Downstream replay must pin and verify that file's SHA256 for
every arm/shard using the realization; this module does not enforce downstream configuration or run
those evaluations. Averaging independent per-shard denominator draws would incorrectly average away
this shared uncertainty. Response seeds and validation-shard exposure belong to a separate design.

## Audits, provenance and immutability

Planning rechecks the old sealed donor plan and assembled library provenance, library ROOT hash,
selection, donor exposure/file count, geometry, source ordering and current source identity. It
re-inventories all original donor `dump/particles`, original denominator `dump/gen`, and supplied
held-out `dump/gen` keys using the existing full-width vectorized highest-cycle inventory routines.
Repeated primaries within one event are valid; duplicates across original files in each bank fail.
Donor/generator, held-out/generator and held-out/donor overlaps fail with explicit counts. Held-out
donor checks conservatively cover the **entire original donor bank**, not just selected donors.
No raw event-key arrays are put into JSON; audited per-file cycles/counts/key hashes and zero-overlap
results are sealed. Memory scales with complete persisted inventories; primary bin counts are
computed vectorially within each file, with Python loops only over file or replica lists.

The plan records base JSON/ROOT/donor-plan identities, full source SHA256s, sufficient bin counts,
explicit exposures, the base donor selection, deterministic generator selection, version/source
identity and limitations. `source_sha256` hashes the canonical source records; `exposure_sha256`
hashes the fixed donor exposure and complete original/evaluation event-count lists. Each replica's
`selection_sha256` additionally covers ordered indices, multiplicities and **selected** exposure.
The base donor selection also has its own canonical hash. Hash encoding uses the existing
sorted/indented JSON serializer, including its trailing newline.

`build` reconstructs the full audit and all selections, comparing the entire manifest, before copying
the base ROOT. It rechecks all dependencies and the plan after writing. The existing `_read_sealed`
and publication helpers are reused. No heavy physics builder, counters lookup, download, or remote
operation is involved. Failed preflight publishes nothing; ordinary write failures clean staging.
Existing files/directories and dangling output symlinks are refused; parents must already exist.

Plan output: `plan.json`, `plan.json.sha256`. Build output: `library.root`, sealed `library.json`,
and a sealed copy of the **new denominator plan** as `plan.json`. The base donor plan is not rewritten;
its original path/hash remain referenced. Metadata retains donor statistics and `inputs`, changes
`gen` to the ordered bootstrap occurrences and `generated_nonmu` to their repeated raw primary count,
and uses new denominator provenance rather than misrepresenting the old provenance as this output.
The repeated `gen` list documents selection; it is **not** a safe rebuild recipe for the old `_reach`
helper. `generated_scaled_to` remains the donor exposure, not the selected generator event count.

Published files are read-only. Checksums are accidental-change detection, not signatures or a defense
against deliberately forged manifests and sidecars. Use trusted inputs. A process kill can leave an
incomplete publication; require the complete output set and verify hashes. There is no resume or
transactional-filesystem guarantee. Strict inherited source hashes may block old products after
physics/helper revisions; preserve them and produce a fresh audit, never waive checks.

## Statistical cautions and validation

- Whole-file bootstrap assumes the files are independent exchangeable generator blocks. Shared
  production conditions are not made independent by disjoint event identities; more resamples do
  not create more independent source files.
- Held-out certification covers supplied complete persisted gen IDs and persisted donor particle IDs.
  It cannot establish independence from unpersisted donor events or certify evaluations whose actual
  processed event census was never compared to the supplied held-out bank. Use shards 11–20 only
  after checking their actual IDs; this tool does not infer independence from shard numbers.
- The same denominator is shared by all shards. Full-campaign uncertainty must retain that covariance;
  conditional paired-event/shard errors from other tools do not include it automatically.
- This isolates denominator variation at fixed donor population, quantile edges and donor exposure.
  It does not include finite-donor uncertainty, uncertain production exposure, response randomness,
  finite validation exposure or joint covariance. Cross-species and energy-bin fluctuations are
  coupled through the file draws and the random selected exposure, not independent-bin errors.
- Sparse support failures and small numbers of files limit ordinary bootstrap inference. Do not
  silently discard failed replicas, relabel their spread as calibrated significance, or combine
  variance terms by quadrature without a justified independence/covariance model.

Synthetic local validation (no production data or physics run):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../core \
  /Users/rmina/mu2e/.venv/bin/python3 -m pytest -p no:cacheprovider -q \
  tests/test_valhist_denominator_variants.py tests/test_valhist_library_variants.py
```

Fixtures use real small ROOT TTrees and a mocked donor assembly, but the real tier-1b reach helper,
loader and writer. Tests exercise unequal event exposure, repeated files, multi-primary/muon-only
exposure, exact formulas, prefix-stable streams, latest-cycle loading, unchanged donor trees,
full-width held-out overlap, sealed/resealed manifest drift, source/code drift, output refusal,
postflight failures, sparse support, probability limits and ties-to-even rounding. They do not certify
production accessibility, its actual exposure, exchangeability, or downstream replay compatibility.
