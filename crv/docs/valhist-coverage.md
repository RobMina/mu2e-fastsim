# ROOT histogram coverage and rate/shape audit

`valhist.coverage` is a standalone, read-only complement to `valhist.compare`.
It does not modify that comparator or merge files. Its purpose is to expose
coverage gaps and describe differences, **not certify physics agreement**.

## Run locally

From the `mu2e-fastsim` umbrella:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:crv \
  ~/mu2e/.venv/bin/python3 -B -m valhist.coverage \
  /data/reference/valhist.001.root /data/candidate/valhist.001.root \
  --outdir /tmp/coverage-001-new \
  --alias CrvDigis_fastCrvDigis=CrvDigis_Sim
```

Supply **original, unmerged historical files**, ideally matching the same
subrun/events. The old `compare merge` output dropped flow bins and source
Sumw2: this audit cannot restore either, or reliably identify that an input
has suffered this loss. Reading an old merged file cannot establish flow
closure. File provenance remains the caller's responsibility.

The output directory must not exist; existing paths are refused. The audit
opens inputs read-only and never writes ROOT output. It handles one file per
arm, not globs, event pairing, or shard aggregation.

`--select` defaults to the same CRV regex as `valhist.compare`:

```text
^(crvDigiDQM|crvRecoDQM|dqmCrv)/|^Validation/Crv
```

Use `--select ''` for all non-directory objects. Selection is applied **after**
canonicalization. `--alias OLD=NEW` is repeatable and applies to the immediate
child of `Validation/`, either an exact child or a prefix followed by `_`.
The longest matching alias applies once, without chained substitutions; other
directories and unrelated textual prefixes are untouched. Aliases apply to
both arms. Several source paths mapping to one canonical path are an explicit
`alias_collision`, never summed or arbitrarily selected.

## Inventory and coverage

The inventory chooses the highest numeric ROOT cycle per source path before
class filtering. TH1, TH2, TH3 and all TProfile families are inventoried;
selected non-histogram objects are retained as unsupported rather than hidden.
Original path, explicit cycle key, cycle number and ROOT class remain in each
report, including missing-object and collision reports.

The union of selected canonical keys is audited. Missing keys, exact ROOT-class
mismatches (including different storage types), dimension/edge/axis-label
mismatches, alias collisions, unsupported objects, read errors and invalid
source numbers are explicit coverage gaps. Edge matching is exact, not an
`allclose` tolerance that could mask a changed booking. Type mismatches are
not compared, even if the binning looks compatible. Empty selection is
explicit in the summary, not a successful agreement result.

**Profiles are unsupported.** They are reported as `unsupported_profile` and
never summed, normalized as counts, or silently omitted. A future profile
comparison needs profile entry/effective-entry conventions, error modes and
paired-event covariance; simply reusing TH1 contents is not defensible.

## Classification and quantities

Count semantics cannot be proved from a TH class alone. Names/titles indicating
rates, MPV, efficiency, occupancy, fractions, ratios, means, RMS, sigma,
pedestal or gain are classified `non_count`. In addition, **all objects under
the three DQM directories are conservatively non-count**, including ambiguous
names that might actually contain fill counts. This intentionally sacrifices
some rate/shape coverage to avoid interpreting derived maps as Poisson data.
Other supported TH histograms are `count_like`, not certified event counts.
Review classification/reasons for each object before interpreting results.
A derived object with an arbitrary name outside these conventions cannot be
automatically identified; there is no universal ROOT count/map metadata.

For supported compatible TH1/2/3 objects, all comparisons use source
`values(flow=True)` and the full source `fSumw2` array in ROOT bin order.
There is no merge, rebinning, overwrite or inferred variance fallback.

* `visible` means every axis is in range.
* `flow` is the complement: at least one underflow or overflow coordinate.
  Corner/edge flow cells are counted once, not once per axis.
* `inclusive` is the whole array.

Each region records bins, value sums and source-variance sums. Count-like sums
are fill-weight totals, **not event counts or exposure-corrected physical
rates**. Non-count map sums are only arithmetic diagnostics, not physical
totals; their normalization, normalized-shape and weighted statistics are
explicitly `not_applicable_non_count`. They still receive exact-bin and
absolute-difference diagnostics.

`closure` checks exact array equality for both values and source variances,
including flow. It separately records differing bins, L1/max absolute
differences and Sumw2 presence. `exact_values_and_variances` is `null` if either
source lacks Sumw2, not an assumed pass based on Poisson variances. Explicit
zero-valued Sumw2 is retained as zero, and malformed, negative or nonfinite
variance arrays are coverage gaps. Closure covers these arrays only, not ROOT
metadata, event equivalence or complete reconstruction closure.

For count-like objects the report separates:

1. **Normalization:** reference/candidate regional totals, absolute difference,
   and `(candidate-reference)/reference`. Zero reference gives a null relative
   difference with an explicit status.
2. **Normalized shape:** total variation `sum(abs(p-q))/2` and maximum absolute
   bin-fraction difference, after each regional array is normalized separately.
   These are descriptive distances (including multidimensional cells), not
   significance tests. Zero totals or negative weights make this probability
   shape undefined. Inclusive shape includes flow; visible shape does not.
3. **Source-weight diagnostics:** where both Sumw2 arrays exist, the sum of
   `difference**2 / (variance_ref + variance_cand)` over positive-denominator
   bins is reported, without ndf or p-value. Zero-variance changed bins are
   counted explicitly and make the full diagnostic null with
   `undefined_zero_variance_difference`; any positive-variance contribution
   remains separately available. An all-zero-variance disagreement therefore
   cannot masquerade as a finite zero discrepancy.

No Poisson assumption is used, including for missing Sumw2 or non-count maps.
Source Sumw2 can be present for unit-weight fills too; its presence is not
proof of weighting or of correct event-level uncertainty. For actual weighted
fills it preserves the booked weight-square information that the old merge
lost. Negative weights retain algebraic totals and exact/variance diagnostics,
but are not converted into probability distributions for shape tests.

## Output contract

* `histograms.json`: canonical-key mapping to source inventory, status/issues,
  classification, source metadata/edges/labels and available diagnostics.
* `summary.json`: input paths, selection/aliases, selected/excluded counts,
  status counts, `coverage_gaps`, missing-key lists, value/variance difference
  lists, `variance_coverage_gaps`, empty-selection indicator and limitations.

There is no blanket pass flag. `compared` means diagnostics were computed,
not agreement. Missing Sumw2 is a separate variance coverage gap and does not
prevent descriptive value comparisons. Numeric overflow is reported and all
JSON is strict: nonfinite numbers serialize as `null`, never NaN/Infinity;
invalid source arrays are marked and not compared. Inspect statuses as well
as nulls. CLI exit 0 means the reports were written, even with coverage gaps;
exit 2 denotes invalid invocation/input/output errors handled by the CLI.

## Uncertainty work remaining

Paired arms can share events, and a single event can fill several bins or fill
a bin repeatedly. Source Sumw2 does not encode those correlations. Consequently
this audit provides no p-values, calibrated thresholds, effective independent
sample count, multiple-testing claims, confidence intervals or bootstrap.
Normalized shape likewise has no supplied covariance.

Paired-shard mode is deliberately not implemented. A defensible extension
must first match stable run/subrun/shard or event identities (not positional
file order), check coverage and exposure, and establish independent resampling
units. It could then retain paired per-unit contributions and estimate the
covariance of rate and normalized shape separately. Resampling aggregated
bins would be a pseudo-bootstrap and is not a substitute for those data.

## Local regression suite

No ROOT installation, remote work or extra packages are required beyond the
existing Python environment. Synthetic ROOT fixtures use uproot's writable
TH1/TH2/TH3/TProfile API, including explicit flow and Sumw2 arrays.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:crv \
  ~/mu2e/.venv/bin/python3 -B -m pytest -p no:cacheprovider \
  crv/tests/test_valhist_coverage.py -q
```

Tests cover flow-only changes, multidimensional corners and variance ordering,
weighted and variance-only closure, missing/malformed Sumw2, zero variance,
zero/signed/nonfinite values, numeric overflow, rate-only and shape-only
changes, missing keys, strict edge/type mismatches, highest-cycle replacement,
unsupported profiles/objects, non-count maps, alias collisions/boundaries,
selection, strict JSON, source-file immutability and refusal of existing output.

## Independent catalogue coverage

`--catalogue FROZEN.txt` additionally checks the explicit Digi+Reco client inventory written by DQMHistSet,
including missing-in-both entries, extra histograms, exact ROOT classes and axis bookings. The Status client is
not implicitly required for MC. Fixed axes compare `nbins/xmin/xmax`, avoiding floating-point edge-generation
roundoff; variable axes compare edges. No `allclose` tolerance hides booking differences. The frozen catalogue
uses default ostream precision: if differing endpoints match six-significant-digit serialization, that is
reported as a precision candidate **without clearing the mismatch**. For the pilot this affects Z_extracted
(20787.95–24903.05 in both ROOT files, 20788–24903 in text); 84/85 exact bookings, all 85 names/classes present.

## Source-reviewed overrides and event-level rates

`--count-key EXACT_CANONICAL_PATH` explicitly declares a histogram's count semantics, overriding the conservative
DQM heuristic. Review the source before using it; no regex or automatic promotion is supported. The pilot driver
`audit_pilot.sh` names only cluster PE, NPc and position histograms filled once per cluster by the frozen
Offline `CRVDQM/src/CRVRecoDQM.cc` and DQM `DqmCrv_module.cc`. It does not promote rate/MPV maps.
The summary records every override and any unused names. Profiles remain unsupported even with an override.
`TNamed` objects (including DQM binning-version records) are compared by name/title; mismatches are coverage gaps.
`report.tsv` is a compact view of totals, flow and normalized-shape distances; detailed provenance stays in JSON.

`python3 -m valhist.paired_rates REF_DUMP CAND_DUMP --out NEW.json` supplies a separate, limited uncertainty
calculation where both arms have full event trees. It validates identical sorted run/subrun/event populations,
rejects duplicates, checks cluster-table totals against the per-event census, and retains zero-activity events.
For per-event values a and b and ratio R=sum(b)/sum(a), its delta-method standard error is
`sqrt(N * sample_var(b - R*a)) / sum(a)`. The difference error is `sqrt(N * sample_var(b-a))`.
These are conditional event-paired errors assuming independent generated events, not uncertainty from the fixed
secondary library, calibration/model variation or paired histogram shapes. Missing event coverage is an error,
not an invitation to pad absent reference events as zero. No acceptance threshold or p-value is attached.

Pilot execution: `run_pilot.sh NEW_OUT FASTC LIBRARY GEN_ART SEED` reuses retained transport, writes fresh tables
and A1/E2E art histograms in the already-built frozen `CRVVAL_AREA`, and dumps event/products without DigiMC so
completion can be checked. `M2FS_CORE` names an isolated core checkout; `CRVFS_DATA` is the data root. No build,
reference rewrite or resume-by-file-existence occurs. `pilot.py` captures source/input/conditions/binary hashes;
completion requires successful art exits and exact full generated-event IDs in the highest persisted tree cycle.
The initial pilot manifest predates added LUT/template/core-source hashing; retain that qualification in its record.
