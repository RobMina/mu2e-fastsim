# Audited finite-library diagnostics

`python3 -m valhist.library_variants` plans and rebuilds **whole-source-file** variants of a tier-1b library.
It does not change the nominal library, run transport, run stock processing, or make a validation claim.
Donor event keys were discarded in the nominal library ROOT trees: independent halves cannot be certified by
splitting `libxing` or `libshower`. This tool returns to the nominal JSON's source files.

## Exact CLI

Run from the umbrella checkout with `PYTHONPATH=crv:core`, or from `crv/` with `PYTHONPATH=.:../core`.
Use an existing data directory outside the code tree. Every `--outdir` must be **new**; its parent must exist.
The paths recorded in nominal metadata must be accessible on the host executing the command. Local unit tests
use synthetic ROOT files; they do not certify inaccessible production files.

```bash
export PYTHONPATH=crv:core

python3 -m valhist.library_variants plan "$NOMINAL_JSON" \
  --outdir "$DATA/library_variants_plan" \
  --generated-per-file 500000 --bootstrap 20 --seed 173

python3 -m valhist.library_variants build "$DATA/library_variants_plan/plan.json" \
  --variant half_even --counters "$LOCAL_COUNTERS" \
  --outdir "$DATA/library_half_even"

python3 -m valhist.library_variants build "$DATA/library_variants_plan/plan.json" \
  --variant half_odd --counters "$LOCAL_COUNTERS" \
  --outdir "$DATA/library_half_odd"

# Optional explicit sequential builds, not an implicit all-variants campaign:
for variant in bootstrap_0001 bootstrap_0002; do
  python3 -m valhist.library_variants build "$DATA/library_variants_plan/plan.json" \
    --variant "$variant" --counters "$LOCAL_COUNTERS" \
    --outdir "$DATA/library_$variant" || break
done
```

`--bootstrap B` defaults to zero (halves only); `--seed N` defaults to zero. Both must be nonnegative integers.
Replica names are `bootstrap_0001` through `bootstrap_B` with at least four digits. Half names refer to
**zero-based indices in the original metadata list**, not filename digits. There is no filename sorting,
glob expansion, file subsampling, truncation of event inventories, or hidden exposure inference in planning.
Relative metadata paths resolve relative to the nominal JSON directory; absolute paths are preferred.

The nominal JSON must contain nonempty `inputs` and `gen` path lists and a positive integer
`generated_scaled_to`. Its same-stem `.root` library must exist and is hashed. An even donor-file count is
required. `--generated-per-file` is mandatory and its product with the donor-file count must equal
`generated_scaled_to`, exactly. This asserts uniform **generated-event exposure**, not the number of
particle rows, surviving events, donor showers, or gen-table events. It is the operator's responsibility to
verify this per-file exposure against production provenance. Heterogeneous file exposures are not supported.

`--counters` is a required existing local Offline counters file matching the nominal geometry. It is hashed,
then passed directly to `geometry.from_offline_config`; no asset lookup, download or cache fetch occurs.
If nominal metadata did not record its geometry, this tool cannot retrospectively verify that match.

## Plan audit and deterministic variants

Before publishing `plan.json` and `plan.json.sha256`, `plan`:

1. Validates metadata and explicit exposure, rejects repeated nominal paths, and fingerprints nominal JSON/ROOT,
   every donor ROOT, and every fixed gen ROOT with streaming SHA256.
2. Reads **every** `(run, subrun, event)` row from the highest cycle of `dump/particles` in every donor file.
   It vectorizes unique full integer tuples within a file; multiple particle rows for one event are normal.
   It rejects missing/empty identity trees, invalid integer IDs, and any event identity shared across donor files.
   It never packs IDs into restricted bit fields or substitutes file/subrun ranges for an event audit.
3. Performs the same inventory on every fixed gen file's `dump/gen`. Multiple generated particles per event
   are allowed, but cross-file gen duplicates fail. Any donor/gen event-key overlap also fails rather than
   being silently interpreted as independence. Persisted event identities are assumed globally meaningful.
4. Records per-file tree cycles, selected cycle, total rows, unique-event counts and SHA256 of lexicographically
   sorted unique keys encoded as little-endian signed 64-bit triples. Counts and hashes, **not raw event-key
   arrays**, are written. Missing generated events cannot be recovered from `particles`; the inventory is a
   full persisted-donor audit, not a certification of all generated events.
5. Selects alternating disjoint halves and optional whole-file bootstrap replicas. Each bootstrap draws the
   original number of donor files with replacement. Ordered donor indices and per-file multiplicities are
   retained, including repeated filenames; exposure is the number of draws times the asserted per-file exposure,
   **not** the number of unique selected filenames. Each replica uses NumPy `PCG64` with
   `SeedSequence([seed, one_based_replica_index])`; adding replicas preserves earlier draws. NumPy version is
   recorded. The selection itself is frozen in the manifest.
6. Records hashes of current CRV/core Python source and this driver. Historical nominal code identity is copied
   from metadata (`code_identity`, `revision`, `commit`, `git`) as **reported, not verified**, or explicitly marked
   **unavailable**. Current source hashes are never presented as proof of which code built the nominal library.
   Do not claim an exact historical-code rebuild when that evidence is absent.

File audit timings are printed to stderr, not embedded in the deterministic manifest. Measure a local per-file
cost before projecting a campaign runtime; no remote production timing has been measured by the synthetic tests.
ROOT reads and within-file key operations are vectorized; only small file/variant lists use Python loops.
All distinct event-key arrays are retained for exact cross-file comparisons, so total key memory scales with
persisted event count even though full particle tables are never loaded.

## Build outputs and failure behavior

`build` accepts one explicit variant name. It verifies the plan sidecar, source hashes, current source-code
identity, nominal exposure, source ordering and deterministic selections **before** invoking the heavy builder.
It calls `tier1b.build` directly with the selected ordered donor list (duplicates preserved), the same fixed gen
list, and `generated` equal to that variant's exposure. The existing builder offsets IDs for each input occurrence
and re-estimates momentum/path-length quantile edges for the selected donors. These are **not fixed-edge** variants.
There is no workaround that merges repeated donor rows before calling the builder.

Each new build directory contains:

- `library.root`: standard tier-1b library trees written by `tier1b.write_library`;
- `library.json` and `library.json.sha256`: normal `inputs`, `gen` and builder statistics plus a provenance block
  with selection/multiplicities, generated exposure, geometry/code hashes, plan hash, library SHA256 and limitations;
- `plan.json` and `plan.json.sha256`: a frozen copy of the source audit.

Inputs, code, geometry and the original plan are checked again after building and before publication. Builder
file-count/exposure inconsistencies fail. Empty quantile populations or other tier-1b build failures are not
silently filled or repaired. Products are staged in a temporary sibling directory and cleaned on ordinary Python
failures; a new output directory is exclusively reserved for publication. Existing directories, files and dangling
symlinks are refused, including empty output directories. Published files are read-only; retry with a different new
output path, never edit a published manifest. Failed preflight leaves no new output or staging directory. A process
kill or machine failure during publication can leave incomplete products: require all listed files and verify hashes
before use. This is not a transactional filesystem or a resume protocol.

SHA256 sidecars and read-only modes provide reproducibility and accidental-change detection, **not signatures or
protection against an owner deliberately modifying both manifest and checksum**. Audit only trusted manifests. The
nominal source files must not be modified while either command runs. No source-file copying or physics outputs are
written into the code checkout. Source dependency hashes are strict: if code changes, make a new plan. This module
avoids Python-3.10-only annotation syntax and Python-3.11-only `hashlib.file_digest`; the installed project/dependency
requirements still determine which interpreters can execute the full heavy builder.

## Interpretation: diagnostic, not an error bar

- **Half-library differences are not one-sigma uncertainties.** Each half changes donor population and library size;
  split sensitivity alone has no calibrated error-bar interpretation.
- Whole-file bootstrap treats donor files as exchangeable independent source blocks. It is a conditional diagnostic,
  not a demonstration that production files meet that sampling assumption. Common production conditions remain common.
- The **fixed gen denominator is shared across all variants and with validation shards 1–10**. These builds do not
  resample denominator uncertainty, remove that dependence, or make those shards independent validation data.
- Re-estimated quantile edges and changed finite-bin populations are part of the diagnostic. Empty-bin failures are
  informative limitations, not grounds for silently switching back to the nominal library.
- Passing donor overlap checks does not certify independence from a validation sample that was not audited here.
  Audit complete validation event identities separately before claiming held-out validation.
- Comparing downstream variants also includes sampling/reconstruction noise unless evaluation streams and seeds are
  controlled. This tool neither runs those downstream comparisons nor turns their spread into a significance or a
  blanket DQM pass. It does not include finite-library uncertainty in existing event-paired error estimates.

## Local verification

```bash
# From crv/, using an existing environment with numpy/uproot/pytest and core available:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../core \
  python3 -m pytest -p no:cacheprovider -q tests/test_valhist_library_variants.py
```

Tests create only synthetic ROOT files in pytest temporary directories, mock the heavy physics builder, and cover
split/replica exposure, repeated donors, deterministic seeds, full-tuple overlap rejection, gen inventories, latest
cycles, immutable outputs, bad exposure, input/code/plan drift, failed-build cleanup, and no-output invalid preflight.
No production event numbers or physics comparison results are generated or recorded here.
