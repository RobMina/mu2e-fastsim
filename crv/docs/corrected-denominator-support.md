# Corrected-library denominator support and uncertainty integration

## Decision boundary

`valhist.corrected_denominator` audits the separately versioned, identity-corrected production library using
its retained strict projection cache and the unchanged full-generator bank. It writes a **new diagnostic JSON**,
never a library, transport table, response output or downstream covariance. A nominal support pass is not a
bootstrap-support pass. Any requested draw with a shower-populated zero denominator or probability above one
blocks nonlinear propagation; no clipping repair, redraw, dropped replica or nominal substitution is permitted.

The existing `denominator_audit.propagate` preserves native sampler acceptance for every requested draw: zero
support rejects and a ratio above one always accepts. Its covariance remains a parameter-level diagnostic of that
behavior, not a valid ratio-ensemble physics covariance when a support gate fails. The report labels that scope
explicitly and never authorizes downstream propagation, even when the support-only audit passes.

## Reuse and provenance

The strict association repair, re-dumps, versioned library and additive-observable studies already exist; see
[generator-association-repair.md](generator-association-repair.md) and
[corrected-shower-uncertainty.md](corrected-shower-uncertainty.md). No repeat capture or software build is required.

The audit:

1. Binds the corrected library/plan/geometry, strict donor projection cache, donor completions and cache payloads.
   Donor association validation is reused by hashes, not asserted to be a fresh re-dump or independent pointer audit.
2. Requires schema-v2 donor evidence, unchanged ordered input banks and generated exposure. It never uses accepted
   donor-event counts as generated exposure.
3. Reads every original denominator file and its strict full-generator counterpart, compares every legacy gen and
   subrun value exactly, validates the full-stream role and census (including zero-generator events), and checks
   complete event identities and source disjointness. Original generator files remain the actual denominator inputs.
4. Recomputes file/cell sufficient statistics and native rounded normalization, requiring exact agreement with both
   the cache and persisted `libgen`; checks all shower counts against `libshower`.
5. Resamples whole generator files with one multiplicity vector shared across every species/KE cell. Count and
   exposure multiplicities remain coupled. It retains all failed draws and cross-cell covariance, and reports
   per-cell file support and exact raw-count loss probability `(1-k/F)^F` under the empirical file bootstrap.
6. Optionally audits the original nominal library with the same bank/multiplicity draws, keeping its distinct donor
   selection explicit. It refuses changed denominator bins/counts rather than inventing an A/B pairing.
7. Rechecks bound inputs and current analysis-source hashes before exclusive publication. Existing paths, dangling
   symlinks and code-tree outputs are refused. Historical source hashes are retained as provenance, not rewritten.

The binomial Wilson interval for the invalid-draw fraction is **Monte Carlo uncertainty conditional on the finite
file bank**, not a physical confidence interval or calibration of the bootstrap procedure. The audit's nominal
support scope is shower-populated cells; evaluated tail cells without templates need the separate held-out audit.

## Interface

With CRV/core on `PYTHONPATH`, from the CRV checkout:

```bash
python3 -m valhist.corrected_denominator CORRECTED_CACHE FULLGEN_CERTIFICATE \
  --legacy-library ORIGINAL_NOMINAL.root --replicas 1000 --seed 20261025 \
  --out NEW_SUPPORT_REPORT.json
```

`CORRECTED_CACHE` is the completed production directory created by `shower_uncertainty prepare`.
`FULLGEN_CERTIFICATE` is the completed `genassoc_fullgen` certificate; all referenced local ROOT/evidence files
must remain accessible. No dCache metadata discovery or art reading occurs. Output parent must exist.
Exit zero certifies report completion, **not physics or support acceptance**. Read `corrected.status`,
`nominal_support_pass`, `ensemble_support_pass`, and the per-cell failures.

## Historical uncertainty-tool port

The previously unpublished campaign, paired-shard shapes, half-library contrast, nested donor variance,
report/summary and multi-shard ensemble tools are integrated with current main. Existing `event_shapes`,
`library_variants`, `denominator_variants`, and `denominator_audit` already matched the preserved versions and
were not overwritten. The original frozen worktrees, sealed plans and output files remain untouched.

The historical replay wrappers and `library_campaign` intentionally name frozen physics/build areas and original
campaign paths. They are **not a corrected-library production campaign** and were not executed by this port.
Current `tier1b.build` requires strict donor identities; no legacy bypass was added to make historical donor
builds run against current code. Historical reports can still be read with their recorded source identities;
new campaigns need new compatible plans and strict donors.

No default runtime or physics parameter changed. The only existing-file test adjustment permits an absolute
`8 * float64 epsilon` in one synthetic enumerated covariance comparison: exact mathematical zero can be
`-5.55e-17` after Linux BLAS accumulation. Statistical implementation and production gates are unchanged.

## Validation and remaining work

Tests cover shared-file covariance, native ratio-of-totals/rounding, sparse support, saturation refusal,
integer/exposure guards, retained ROOT/cache closure, drift, exclusive outputs, full-width event identities and
zero-generator census rows. The port retains tests for whole-shard pairing, flow bins, immutable source/output
bindings, balanced library/draw/shard designs, exposure pooling, negative variance components and support gates.

To run the scoped suite:

```bash
python3 -m pytest -p no:cacheprovider -q tests/test_corrected_denominator.py \
  tests/test_valhist_*.py tests/test_tier1b.py tests/test_genassoc*.py tests/test_shower*.py tests/test_stepclosure.py
```

Missing empirical support is not fixed by additional bootstrap draws. A follow-up needs compatible independent
full-generator inputs or an explicitly validated support model, followed by a new nominal and whole-ensemble
support check. Do not reuse covariance from the misassociated library as repaired physics uncertainty. Nonlinear
cluster/DQM propagation, finite-library calibration and the persistent T1/other-bar descendant model are separate
open gates; no new simulation or generator production is authorized by this audit.
