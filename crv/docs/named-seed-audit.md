# Conditional named fast-seed audit

`crvfastsim.analyses.named_seed_audit` combines existing
[named-response audits](named-response-diagnosis.md), without reading ROOT rows or
running simulation. Each audit must use the same frozen saved stock reference and
event population. Candidates are fresh fast-response seed replays of **fixed G4
and fixed stock samples**, using declared named yields and the opt-in
[geometry readout mask](readout-mask.md).

```bash
python3 -m crvfastsim.analyses.named_seed_audit \
  --audits seed1/audit.json seed2/audit.json seed3/audit.json \
  --outdir NEW_SEED_AUDIT
```

The exclusive new directory receives `seed-audit.json` and `table.md`. Existing
outputs, including dangling output symlinks, are refused. Input validation and
aggregation finish before directory creation; input files are never modified.
There are no per-event loops: work is over files, seeds and fixed sector/SiPM
metric groups. The input products need not be reopened as ROOT files.

## Admission and grouping

Backend grouping is automatic from the candidate's `<candidate>.metadata.json`
`chain_config.jax` and `chain_config.fused_response` flags: `numpy`,
`jax_nonfused`, or `jax_fused`. Fused without JAX is rejected. These are **declared
backend labels**, not evidence that every kernel ran on that backend. Top-level
backend/noise declarations must agree with the full chain config.

Every backend needs at least **three distinct seeds**. A reused integer seed is
allowed across different backends, but does not imply common random numbers.
Any duplicate candidate resolved path or byte hash is rejected, even across
backends. A repeated seed within one backend is rejected independently of files.
`small_ensemble` flags K < 30, including K=8. This is a descriptive warning, not a
power calculation or admission threshold; reaching 30 does not certify adequacy.

Validation requires:

- The current saved-response audit schema/status, named Run1A or Run2, positive
  event count, and valid event-identity/conditions-config SHA256 values. These
  identities must agree across all input audits.
- The same **resolved reference path, hash and size**, with current bytes checked
  against each audit. A different path holding identical reference bytes is not
  accepted as the same frozen reference; symlink aliases resolving to it are.
- Current candidate bytes/hash/size match their audit records. Audit JSON,
  producer metadata, reference, candidate and conditions-manifest files are
  recorded and rechecked before output creation. Relative audit input paths are
  interpreted relative to the containing audit JSON directory.
- The same conditions-manifest resolved path and bytes. Producer conditions
  metadata, including its config digest, must agree with the audit and across
  seeds/backends. This compares saved evidence, not a fresh conditions export.
- Full chain configurations compare equal after removing **only the top-level
  `jax` and `fused_response` fields**. Nested fields with those names are not
  removed. All currently defined ChainConfig fields must be present; extra
  fields are retained and compared rather than silently discarded.
- Named `chain` step mode, consistent stage/window/noise/finder/timing metadata,
  and equal conditions/readout evidence across every backend. The
  `geometry_readout` policy, counts and mask hash must have the supported shape.
  No `diagnostic_override` key is permitted, even if null. Named sector response
  declarations and photon yield scale must retain the registry's declared
  values; shared-yield emulation cannot enter this sweep.
- Exactly all named sectors and `all`, `0`, `1`, `2`, `3` SiPM groups, with all
  nine paired-event metrics. Each metric's event count must equal the audit's
  full event count. Means and supported ratios/paired SEs must be finite numbers,
  not booleans or numeric strings. Ratios are recomputed from candidate and
  reference means and checked against stored relative values at 1e-12 relative
  and absolute tolerance. Reference means and descriptive reference medians
  must be identical across every replay. Missing/malformed medians also fail.

The complete producer metadata is retained for each input. Optional observer or
trace records are preserved without requiring an unsettled trace schema or
interpreting them as execution certification. This tool cannot independently
establish fixed G4 input identity, trace completeness, runtime/platform identity,
or actual absence of a JAX fallback when the producer does not bind those facts.
The fixed-G4 replay design remains the sweep operator's responsibility. Saved
hash consistency is not a producer signature or provenance admission.

## Reported quantities

For candidate seed means `C_s`, common reference mean `R`, and K seeds, each
supported metric uses

```text
relative_s = C_s / R - 1
mean       = mean(relative_s)
sample_sd  = std(relative_s, ddof=1)
se_mean    = sample_sd / sqrt(K)
```

The JSON reports mean, sample SD, SEmean, min, max and K for both relative values
and **absolute candidate means** in every sector/SiPM/metric group. Sample SD is
the **single-replay conditional fast-seed spread**. SEmean describes the mean
across those fast-seed replays under the usual independent-replay assumption;
it is not the uncertainty of a single replay. The compact table lists all-sector
summed pulse-PE relative **mean ± seed sample SD**, in percent, and K per backend.

Zero-reference metrics remain explicit, with null relative mean/SD/SE/min/max,
not missing rows or silently dropped channels. As in the source audit, other
nonpositive-reference metrics or fewer than two events also have unsupported
relative values. Their finite absolute candidate means and seed spreads are
still reported. Each metric records `reference_zero` and `candidate_all_zero`.
Run2 additionally lists the documented R4 forbidden SiPM indices 1 and 3 and
whether every metric is exactly zero across all candidate seeds. Nonzero values
are retained as diagnostic failures of the expected zero, not suppressed or
reinterpreted as a meaningful relative excess over zero. Other absent channels
remain visible in the full metric inventory; a mask hash alone cannot identify
their positions. Seq0 PE medians are retained per seed, not pooled into an event
median or assigned a statistical error.

Input paired-event SEs are checked but **never added in quadrature**, averaged,
or folded into the seed spread. No unpaired backend-difference error is reported;
compare the separate backend columns without assuming common random streams.
The reference sample is reused, not resimulated. Therefore these diagnostics are
**not a full independent fast+stock seed floor**, an uncertainty over fresh G4,
a finite-library uncertainty, systematic error estimate, spectral comparison,
full named V2 result, response closure or physics certification. Small ensembles
can miss rare response behavior; neither a small SD nor a zero observed channel
establishes an execution or physics certificate.

## Focused tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 CRVFS_NO_FETCH=1 \
PYTHONPATH=/tmp/r1-core-233771b:$PWD \
/Users/rmina/mu2e/.venv/bin/python3 -m pytest -p no:cacheprovider -q \
  tests/test_named_seed_audit.py
```

The synthetic tests use byte-blob candidate/reference files with real hashes and
complete mock producer metadata; they do not execute response kernels or certify
ROOT products. Coverage includes all groups/metrics, known spread formulas,
three backend declarations, K=8 warnings, duplicate seeds/path/symlink/hash,
config/conditions/event/reference drift, mixed diagnostic modes, missing or
malformed metrics, input mutation, zero-reference nonzero candidates, preserved
observer metadata, exclusive outputs, and CLI output.
