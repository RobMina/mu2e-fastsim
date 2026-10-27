# Thread B campaign and uncertainty workflow

This extends the corrected-transport pilot without rerunning its reference or transport arms.
These tools preserve historical campaign provenance; their reports are conditional diagnostics, not physics acceptance.
No script below compiles software or submits grid jobs. `library_variants build` assembles a ROOT data library
through the Python tier-1b builder; it is not an Offline/software build.

## Source ownership and immutable runs

Use the frozen pilot checkout for every `run_pilot.sh` invocation. `pilot.prepare` hashes all Python files under
its checkout, including analysis helpers not imported by simulation. Therefore **do not develop analysis code in
the pilot checkout while any replay runs**. Keep the uncertainty checkout separate. A drift-rejected completion
must remain rejected; do not refresh its manifest or remove its hash checks. Reuse any independently certified
assembled library and rerun the downstream arm into a new output directory through the frozen pilot checkout.

The user stages and commits changes. Do not commit/push or update shared branches as part of this workflow.
The analysis mirror deploys new files; verify their remote content before running. Outputs and logs belong in
the configured data area, never the code tree. One compute job per GPVM, including report generation.

## Remaining-shard extension

`run_shards.sh TAG SHARD...` runs the existing pilot/audit drivers sequentially, refuses existing directories,
and excludes already completed shard 11. Supply these environment variables:

- `THREAD_B_CODE`: frozen CRV pilot checkout.
- `M2FS_CORE`: frozen core checkout.
- `CRVVAL_AREA`: already-built frozen stock work area.
- `CRVFS_DATA`: CRV data root.
- `THREAD_B_TRANSPORT`: retained corrected `fastC` table directory.
- `THREAD_B_LIBRARY`: fixed nominal library ROOT file.
- `THREAD_B_GENROOT`: known generator-art input directory.

Each output is `CRVFS_DATA/valhist/TAG_sNNNNNNNN`. The source pilot records input/conditions/LUT/template/source
and binary hashes; completion checks art exits, actual highest-cycle event counts and exact transport event IDs.
Seeds retain the one-file-per-shard convention: tier1b draws use `[shard_seed,1000]`, fast response uses the shard
seed, while stock A1 retains the frozen FHiCL base seed. Common event IDs do not imply identical random streams.

`python3 -m valhist.campaign status --data DATA --tag TAG` reports completion status without treating an existing
file as success. After all shards finish:

```bash
python3 -m valhist.campaign prepare --data DATA --tag TAG --out NEW_REPORT_DIR
python3 -m valhist.campaign analyze NEW_REPORT_DIR --replicas 10000
python3 -m valhist.report_uncertainty full NEW_REPORT_DIR --replicas 999
```

`prepare` uses a matched-revision `TAG_s00000011` when that directory exists, otherwise the retained shard-11
pilot. It never falls back to the old pilot when a new one exists but is incomplete. It checks output SHA256 and
actual persisted IDs, requires one million unique generated events per arm and consistent nominal seeds/library/
shared physics inputs, and emits explicit paired manifests plus event-paired rates. If later response/reco source
changes make the legacy pilot incompatible, preserve it and replay shard 11 as a new candidate, never waive the
physics identity check. Missing historical pre-run hashes are recorded, not retroactively certified. A0g/A0/A0p references remain original unmerged files;
expected generated exposure is not promoted to verified event pairing where those files lack a census.

## Separate statistical questions

- `paired_shapes`: bootstrap whole paired shard histogram arrays, preserving within-shard correlations and flow;
  exact source-reviewed cluster count allowlist, separate normalized shape and total ratio. Ten-thousand draws
  improve Monte Carlo precision but do not turn twenty independent shards into more data. Centered-TV tails are
  approximate conditional diagnostics, not a blanket DQM acceptance test. See `valhist-paired-shapes.md`.
- `event_shapes`: paired per-event influence covariance and Gaussian multiplier fluctuations for complete A1/E2E
  dumps, including zero-cluster events. Explicit coarse edges in `report_uncertainty.FEATURES` define a separate
  diagnostic; these are not the stock booked histograms. See `valhist-event-shapes.md`.
- `report_uncertainty full` repeats shard diagnostics on shards 11–20, which do not enter the shared generator
  reach-denominator estimate. The full twenty-shard sample is still useful, but is not wholly held out from that
  normalization. The stock reseed floor is analyzed with the same shard bootstrap, not subtracted from TV.
- `uncertainty_summary REPORT...` prints compact rate/shape summaries without inline analysis scripts.

## Finite secondary-library effects

`library_variants plan` audits hashes and persisted source event identities, records explicit generated exposure,
and seals alternating disjoint donor-file halves and optional whole-file bootstrap selections. The nominal
library lacks donor event provenance, so random crossing/shower-ID halves would not establish independent donor
sets. All descendants and showers from a chosen source file remain coupled. See `valhist-library-variants.md`.

`run_library_halves.sh` is the initial sequential half-library assembly/replay wrapper. Use separate frozen
physics and uncertainty checkouts in production; `resume_library_halves.sh` demonstrates reusing an assembled
half and replaying through the untouched pilot checkout. `library_contrast` certifies the two successful fresh
replays, verifies disjoint ten-file halves and correct exposure, then reports paired rates and event shapes:

```bash
python3 -m valhist.library_contrast --data DATA --out NEW_CONTRAST_DIR --replicas 999
```

A single half-library contrast is **sensitivity**, not a full-library standard error. Shared denominator
uncertainty, finite validation exposure, response randomness and bin-edge re-estimation must remain explicit.
Do not relabel ordinary seed repeats as independent libraries.

`run_library_bootstrap.sh INDEX...` assembles only the explicitly named planned full-size donor-file bootstrap
libraries, then evaluates two draws per library on the same retained shard. It uses independent tier1b/fast
response seeds `200000 + 100*INDEX + DRAW`, with DRAW in {1,2}, rather than common seeds across replicas. A1 stock
baseSeed stays fixed, so this estimates conditional donor/sampling variability, not all stock-response variance.
Unique output directories are required; replay products stay separate from the nominal campaign.

```bash
python3 -m valhist.library_uncertainty --data DATA --indices 1 2 3 4 --out NEW_VARIANCE.json
```

For a balanced library-by-draw table, the method-of-moments diagnostic is
`var(library means) - mean(within-library draw variance)/draws`. Negative estimates are retained with an explicit
flag and a separately clipped nonnegative SD display, never hidden as proof of zero uncertainty. Per-bin
normalized-shape variance uses the same decomposition; no independent-bin or global significance test is implied.
A few libraries and one evaluation shard do **not** calibrate confidence coverage or justify extrapolating the
estimated variance to the entire campaign. More libraries, validation-shard coverage and denominator resampling
remain separate gates.

## Validation

Run the `test_valhist_*` suites and `test_tier1b.py` in the project venv before deployment. Synthetic fixtures
exercise full event identity, original flow contents, undefined ratios/shapes, paired correlations, input drift,
donor exposure and multiplicity. Existing-output refusal is part of the contract, not a resume mechanism.
