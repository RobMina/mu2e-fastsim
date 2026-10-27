# Saved response diagnosis

Run `python3 -m crvfastsim.objective.diagnose --directory REPLICAS --out NEW_DIRECTORY` to inspect the existing
four `a_nominal/a_best/b_nominal/b_best` outputs without regenerating response, changing calibration, or rescoring
the BO objective. `--min-entries 100` is the default common-cell threshold, applied separately to data and both
arms in each block. `--no-plots` writes only the JSON. Output directories must not exist.

The paired validator checks finished counts, fixed inputs/conditions/runtime/seed identity within each pair,
independent block seeds and the required 32-term score. This diagnostic additionally checks cross-block runtime
and knob identity, hashes the recorded data-reference file before and after reading, verifies saved point files
are unchanged, and checks the accumulator shapes and nonnegative finite counts. It does not reopen ROOT inputs,
recheck event-key independence, refit time corrections or modify any saved point. The report records both the
original provenance and the diagnostic code identity (including a dirty-worktree flag and module hashes).

## Missing reference timing diagnostics

Some frozen reference files have zero-filled `dtp`/`ftp` even though their ordinary objective histograms are
complete. By default this is reported as no common timing support, never fabricated closure. Optional
`--timing-dir DIR` reads the **data_files list from the fixed time-calibration asset** and merges those saved
per-file accumulators. It requires exact equality with the frozen scoring reference for events, files, exposure,
all line counts, every existing `h` array and all out-of-time counts. Only absent timing diagnostics are then
supplemented in memory; a nonempty differing timing array is refused. Every source file and the calibration asset
is hashed. This is evidence of identical accumulated observables, not a new full-event-identity audit. Nothing is
written back to the reference and no time offset is fitted or applied.

## What is measured

For all EX, T1, T2 and aggregate TAG groups:

- Positive selected-cluster layer PE, corrected first-pulse time RMS and selected-cluster minus EX time shapes.
- Pulse PE divided by measured line path factor, conditional on measured along-bar position and readout side.
- Raw pulse PE conditional on measured line path factor.
- Timing by (EX module, other module), including separate T10/T11 and all eight taggers when populated.

These are **binned, in-range** quantities. Means and RMS use bin centers; quantiles interpolate uniformly inside
the occupied bin. Sparse/absent cells stay explicit, with null summaries for empty histograms. The figures draw
conditional medians only for common populated cells. They are not confidence intervals or unbinned fit results.

## Population versus conditional-distribution checks

`marginal` uses all saved cells. `support_marginal` restricts each sample to exactly the same eligible cells, while
retaining that sample's cell fractions. `standardized` combines each sample's conditional distribution using
**fixed data cell fractions** on that same support. This is a diagnostic comparison, never a new response model
or a change to the objective. Coverage is recorded for each sample. Compare support-matched quantities, not an
unrestricted marginal against a truncated standardized one.

For means, the symmetric two-component decomposition on common support is exact up to floating-point rounding:

```
mean_new - mean_old = sum_cells mean(w_old,w_new) * (mean_new_cell - mean_old_cell)
                    + sum_cells mean(mean_old_cell,mean_new_cell) * (w_new - w_old)
```

The first term is a within-cell change, the second a cell-population change. Conditioning is on measured bin
membership, not truth. Cell membership itself can change with response/selection; neither term identifies a
physical causal mechanism. `tv_to_data` is total variation of normalized in-range shapes, with no bias correction
or inferred uncertainty. It is descriptive and is **not** the BO objective.

## What the saved states cannot establish

- No timing-versus-position, timing-versus-PE or joint position/path correlation is retained.
- PE and time-spread histograms do not resolve T10 and T11 individually; only `dtp` does.
- No paired event identities or selected-hit lists are saved, so one cannot hold the selected population fixed
  event by event. A common seed does not guarantee matching random draws after a knob-dependent branch.
- Histogram underflow/overflow counts were not saved. Selection already requires cluster timing within 20 ns of
  EX, and summary ranges differ (`dt_cluster` +/-25 ns, module-pair `dtp` +/-40 ns). The diagnostic does not measure
  the rejected timing tail or full distribution moments.
- The full best setting changes five knobs, not only T1 attenuation; attribution to one knob requires a separate
  controlled intervention. The P4 time correction and empirical background stay frozen here.

`diagnosis.json` contains every sector/cell and input hash. For one-dimensional shapes it also decomposes the
saved deterministic bias-corrected score into per-bin contributions, checking their sum against the saved term;
there is no new resampling or change of normalization. `t1_score_bins_*.png` displays which bins improve or worsen.
Other PNGs show scalar shapes, conditional PE, module timing, and timing with fixed data module-pair fractions for
each block. No significance/noise estimate, optimization or physics-acceptance decision is made by this command.

## Single-knob interventions

`scripts/run_response_diagnosis_controls.sh DATA NEW_OUT a|b` runs two complete single-knob controls serially for
one block: `atten_T1=0.4367132104009329` and `speed_EX=0.9840676938698677`. The shell must provide the **same frozen
CRV/core PYTHONPATH and Python environment as the original replicas**; do not run it with this diagnostic worktree
on the response path. One GPVM per block; no new nominal/full-best run. The driver records its own checksum,
refuses an existing output directory and writes `complete.txt` only after both validations succeed. The existing
runner checks full exposure/inputs/assets/clean code and records original measurement identity.

After both complete markers exist, compare them without simulation:

```bash
python3 -m crvfastsim.objective.control_diagnosis \
  --baseline-report diagnosis_saved_v2/with_score_bins/diagnosis.json \
  --controls-a controls_a_cpu_v1 --controls-b controls_b_cpu_v1 --out NEW_CONTROL_REPORT
```

It reads the original four point files through their saved SHA256 receipts, requires complete single-knob runs,
exact original nominal provenance except knobs, identical result flags and the exact intended knob value. The
supplemented timing source hashes are also rechecked. `control_diagnosis.json` reports all sectors, per-term deltas
and conditional/module diagnostics, with `sample_role` explicitly distinguishing each single-knob candidate from
the full-best point. The reused summary key `best` means the explicitly labelled candidate within that summary,
not necessarily the full-best point. No original input/conditions file is regenerated or rescored.

These interventions test the T1-attenuation and EX-reference-clock hypotheses separately. They do not constitute
a complete factorial decomposition of the five-knob setting, and differences of their totals must not be added
as if nonlinear response, selection and shared random draws were independent.
