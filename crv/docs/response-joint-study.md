# Thread C joint yield and attenuation diagnostic

The frozen October 25 controls show a T1 longitudinal-shape improvement without a yield-scale correction, and
separately identify EX optical speed as the main common-reference cluster-time shift. This continuation keeps
EX speed nominal, P4 background and module time calibration fixed, and all reconstruction/numerical settings
unchanged. It is a bounded diagnostic factorial, not a new BO campaign or calibrated optimum.

## Coordinates and gates

`scripts/run_response_joint_block.sh VALIDATION_ROOT NEW_OUTPUT a|b` uses the existing held-out manifest and seed
for that block. It runs these four coordinates, serially on one free GPVM:

| Arm | T1 yield photons/MeV | T1 attenuation ratio |
|---|---:|---:|
| nominal | 28610 | 1 (no override) |
| atten | 28610 | 0.4367132104009329 |
| yield | 20027 | 1 (no override) |
| joint | 20027 | 0.4367132104009329 |

The 0.70 yield factor is an intentionally coarse amplitude intervention motivated by the saved measured PE
scale mismatch (roughly 120.5/165.5 for layer PE and 49/72 for low-path pulse PE). Those ratios are **not** a fitted
photon-yield calibration. Thresholds, SiPM nonlinearity, pulse fitting, selection and position/path correlations
make post-hoc histogram scaling inadequate. Full response is evaluated. The schema-v1 sector yield config is
used rather than editing nominal source defaults; no other sector yield or global scale changes.

The first two passes collect previously absent event identities and joint observables. They are instrumented
replays, not new independent baselines: their complete response accumulators must match the existing nominal
and attenuation-only outputs exactly, respectively. Each replay gate stops the driver on failure. The frozen
outputs remain the references and are never overwritten or rescored. New response-config and passive-capture
code remains explicitly unpublished, with before/after source hashes and dirty Git status recorded. Do not edit
this worktree during a pass; all artifacts live outside it. A final complete marker exists only after all arms.

A and B retain the original disjoint input identities and seeds. They are two consistency blocks, not a
paired-difference uncertainty calibration. A shared seed is not a guarantee of equal stochastic photon or
background draws after a changed knob. The four-arm interaction contrast is descriptive; its components must
not be claimed independent or additive causes.

## Passive capture

See [response-joint.md](response-joint.md) and [validation-diagnostic.md](validation-diagnostic.md). R3 retains
actual input run/subrun/event triples, including generated events with no CRV steps. Full-input one-pass capture
is required. For empty-waveform/no-step events only the no-cluster exposure is captured; their background-only
pulses still follow the original out-of-time histogram path and are not fabricated as reconstructed candidates.
No RNG calls, pulse fitting or objective changes are introduced by the capture hook. The capture-off/on gate
compares every response accumulator field, not just total score.

All sectors/bodies, measured line coordinates and path, PE, propagation-corrected first-pulse times, centered
residuals, spread group membership, selected clusters and raw rejected cluster timing are retained. Fit and
range flags distinguish unavailable evidence from empty in-range histograms. Full keys support common-selected
and common-valid comparisons; pulse ordinal is not a stable cross-arm physical identity. Truth coordinates are
not used in a response or correction model.

For the measured P4 data, `objective.joint_data` reads an existing EventNtuple directly, with no time correction
or new reconstruction. It verifies highest tree cycle and full identities, preserves bounded raw NPZ shards,
and can require exact scoring-field equality against a saved per-file accumulator. Its raw shards are indexed
by `summary.json`; it does not produce one monolithic `joint.npz`. A bounded subset is a diagnostic reference,
not a replacement for the 5M-event P4 scoring reference or an independent calibration holdout.

## Completed acquisition and independent identity audit

All eight A/B passes completed with 49,730/49,829 exposure events per arm. Both nominal and attenuation-only
capture replays exactly match their frozen complete response accumulators. No nominal/attenuation reference
was overwritten. The first P4 file capture contains 500,000 events, highest tree cycle 89, and exactly matches
all scoring fields of its saved per-file accumulator, with no data time correction. It is part of the original
reference population, not an independent calibration holdout.

`python3 -m crvfastsim.objective.joint_identity --directory BLOCK --out NEW.json` independently reopens each
manifest-listed generated-event census and verifies every captured source/run/subrun/event/slot, including
no-step exposure. It requires exact sorted identities for each of four slots, not just shared labels across
arms or matching event counts. Both A and B audits passed for every arm. Hashes are rechecked before/after;
this read-only audit does not reproduce pulse or stochastic draw identities. Existing per-pass source hashes
remain immutable even when the later reporting tools are improved.

## Interpretation boundary

Compare measured distributions and common-support residual second moments before considering a time refit.
Changes in selected-event membership, measured line/path, within-cluster pulse population and finite support
must be reported separately. Centered residuals have a shared estimated mean and correlated hits: pulse counts
are not independent sample sizes. A module-wide constant offset cancels from a fixed group's RMS; only changes
in selection or the group definition can make such an offset affect the observed spread population. No GP noise,
background, timing calibration, acceptance threshold, default config or optimizer board is changed here.

## Retained timing continuation

See [response-joint-timing.md](response-joint-timing.md) for the read-only multiplicity/hardware/pair diagnostic.
`joint_timing` reuses the eight completed captures and unshifted 500k-event data; no response rerun is needed.
Exact-N common-support T1 moments remain 6.41–7.13 versus data 2.48–2.63 ns² (data coverage 59–63%).
Same-bar/same-end distinct-SiPM differences, with both pulses 100–200 PE, are nominal 7.41–8.03 versus
2.25–2.42 ns² in data. The line correction cancels there, but pair populations are not fully standardized.
The excess is not solely a measured-line correction artifact; arrival statistics versus fitting remain
unidentified because the capture lacks those upstream observables. Hardware offsets were not fitted.

Saved reports are `diagnosis_saved_v2/timing_{a,b}_20261025_v2.json`; readable summary and inspected PNGs are in
`timing_summary_v3/`. Generate those locally with `python3 -m crvfastsim.objective.joint_timing_report
--a A.json --b B.json --outdir NEW`. This reports all bodies and checks exact A/B data-table equality and
raw/corrected decomposition/cancellation. The saved reports retain detailed results and unresolved evidence boundaries.
