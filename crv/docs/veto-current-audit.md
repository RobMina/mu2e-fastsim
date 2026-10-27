# Current-finder replay and exposure audits

These tools are validation-only. They do not change response, coincidence, mixing,
selection or BO defaults. Use a new output tag, keep fixed-stock products, and keep
the deadtime objective on stock until its independent promotion gates close.

## Fast-only replay

In an analysis-only Python environment with this checkout and core on `PYTHONPATH`,
set `CRVFS_LOOKUP_DIR` and the matching Offline geometry search path. Then:

```bash
bash scripts/replay_veto_current.sh "$BASE" "$NEW_FAST" "$CONDITIONS" "$PHOTON" \
  "$SELECTION" "$REFERENCE" 8100
bash scripts/replay_veto_current.sh "$BASE" "$NEW_REPLICA" "$CONDITIONS" "$PHOTON" \
  "$SELECTION" "$REFERENCE" 18200
```

The bounded driver expects `block0`, `block1`, `block2_fullinventory`, each with
3,000 saved `mixed.root` events and retained stock under
`fixed_stock_v13_41_00_20261025/<block>/stock.root`. It never runs stock or exports a
new mixture. Seeds increment by block; chunk size is 100. Timings, source hashes,
fast pulses, frames, bound checks and paired metrics are retained. Existing output
directories fail rather than overwrite. This is not a general campaign submitter.

## Membership, timing and replica covariance

```bash
bash scripts/audit_veto_current.sh "$BASE" "$NEW_FAST" "$NEW_REPLICA" \
  "$NEW_AUDIT" "$CAT_PROBES" "$COUNTERS"
```

`veto.replay_recover` reconstructs memberships from the ordered retained pulse table
with the current reference finder and requires its summaries to reproduce the saved
frames. Its pure API uses flat arrays and offsets for the unavoidable per-event
reference-finder dispatch. No response or fit runs. The exported pulse adapter is
diagnostic only: placeholder fields are explicitly labelled, never measurements.
This CLI defaults to the **current** finder; it must not be used to claim recovery of
historical attachment-OFF summaries with attachment ON.

`veto.replay_audit` requires exact ordered pulse identity before comparing native
pulse-index memberships with the retained same-pulse stock-finder control. Start,
end, average time and PE residuals are reported without hiding floating-point
mismatches. Across independent response draws only observed **channel sets** can
be compared; these are neither pulse identity nor truth-particle matches.
`frames.parquet` retains all empty frames, per-frame coverage and gate outcomes.

Two fast draws on one raw input and one stock draw are averaged **within frame**
before calculating the paired uncertainty: still 9,000 unique observations, not
18,000. The block SE assumes independent blocks and is descriptive if source pools
overlap; it is not finite-library uncertainty. Disjoint filenames do not establish
independence. No response uncertainty is inferred from just two draw means.

## Exposure and Cat denominators

```bash
python3 -m crvfastsim.veto.exposure_audit --base "$BASE" --probes "$CAT_PROBES" \
  --efficiencies "$SAVED_QE_CSV" --out "$NEW_EXPOSURE_JSON"
```

- Persisted explicit event intensity/bunch-time arrays are checked against their
  manifest, with conditional Poisson mean diagnostics per species.
- Source **content hashes** expose duplicates within and across filename subsets.
  No duplicate is silently discarded; physical exposure weights need a separate
  production/provenance decision before changing the library.
- Highest-cycle generator IDs use full `(run,subrun,event)` tuples and are compared
  with historical packed IDs. Generator rows are not events. Missing empty-generator
  events cannot be detected by the generator tree alone; retained processing logs
  are corroboration, not a replacement persisted event census.
- `--efficiencies` takes a previously queried `sim.efficiencies2` CSV. Purpose,
  version, extension, group and IOV linkage must be retained separately. Decimal
  products and stock's `std::stof` parsing effect are both reported.
- Analytic PBI moments distinguish an unconditional truncated lognormal from the
  beam-conditioned `x * lognormal(x)`. Offline classifies CE endpoint/leading-log as
  beam-based; those primaries require the latter, not the mixer's current unconditional
  draw. The selected-track/PBI correlation is an additional factorisation question.
  Do not multiply an existing F by the ratio of PBI means and call it corrected.

`veto.replay_timing` is a diagnostic-only port of the reviewed stock timing predicates:
endpoint rejection, corrected-fmod arrival wrapping, inclusive digi-start limits and
full waveform tails. Its tests cover all bunch phases and exact edges. These helpers
are not wired into production response and do not constitute full-sample timing
closure.

Tests:

```bash
python3 -m pytest -q tests/test_veto*.py tests/test_coinc_attachment_cpu.py
```

The audit/recovery/timing tools never build, submit, change retained files, tune
physical yields or widen timing gates. Numeric physics results and site paths belong
in external validation records, not this public repository.
