# Reference diagnostics and campaign tooling

## Scope and prerequisites

These diagnostics build on [inclusive recording and source validation](flash-inclusive.md)
and [causal response and residual accounting](flash-response.md). They analyze
saved evidence or prepare explicit campaign inputs; they do not turn development
anchors into design holdout, recover lost timing precision, or establish historical
board closure. Keep zero-response events, unresolved exposure and residual energy
visible, and retain the original complete input-event denominator.

## Unknown causal attribution

`flash_response_unknown` explains unresolved attribution from the original saved
response and compact ancestry trace. It reports strict crossing/child-birth bracket
comparisons without changing ownership. Very late absolute times can give float32
particle births wide rounding intervals even when step times are double precision.
A resolved spatial exit and complete ancestry do not resolve that causal order.
Keep these deposits unknown; subtracting an offset after float32 storage cannot
recover the lost precision. Independent raw tracker identities and the original
response depth limit are not available in these compact archives. Supply
`--response-max-depth` only from the original frozen configuration. Without it,
ancestor comparison rules remain conditional, and a truncated diagnostic trace
is not proof that the original response exhausted its depth limit.

## Matched S1 spectra and concentration

The inclusive reference's `flash_s1_reference` comparison matches saved fast
uncollided first-foil weights to G4 primary brems on the full input-event universe.

`flash_s1_spectrum` decomposes this comparison into fixed absolute energy bins,
cumulative photon-energy tails and ranked event contributions. Event-level
covariance is retained before estimating tail uncertainties. It separately
inventories unsupported attenuation and unknown G4 exits. Thresholding a weighted
fast event expectation is not equivalent to thresholding a realized G4 event;
these event-total tail tables describe estimator concentration, not physical
multiplicity, survival probability or detector acceptance. Do not interpret the
associated indicator differences as a physics disagreement. The linear count and
energy-sum comparisons remain the appropriate primary diagnostics.

## Reference figures

`flash_reference_figures` renders the saved absolute spectra and inclusive
ancestry components from JSON reports. It checks matching denominators and energy
components before plotting. Figure labels preserve the primary-only and raw-TS3
scope; no historical asset is re-labelled as corrected-model closure.

## Campaign planning

`flash_reference_campaign` writes an explicit design/geometry/source/skip/count/seed
manifest without launching jobs. Shards cover disjoint source ranges, use distinct
seeds, and cannot wrap past the declared pool size. The runner accepts any explicit
frozen geometry path; historical design IDs are not an inference API. Keep unused
source ranges and new-design test points separate from development.

Use the saved-response and ancestry mergers described in the response reference
for disjoint shards; shard-local ownership pointers must not be concatenated.
Campaign manifests do not replace recording checks or certify successful transport.
Choose resources from measured output sizes and completed-job runtimes, not linear
extrapolation from a tiny pilot. Keep held-out events and genuinely unseen designs
separate from development, and freeze the model before evaluating design holdout.

## Source profiling

`flash_profile` separates input/geometry/field-map setup, table construction, JIT
lowering and compilation, fixed-shape execution chunks, and final host-copy time.
It records live lanes and vacuum/slab work without introducing physics cuts. Saved
step counters provide an iteration lower bound, not exact lane termination times.
Optional saved-reference comparisons report exact array equality and discrepancies;
optional flux summaries recompute foil exits and attenuation after timing. Execution
throughput excludes setup and compilation and is not end-to-end campaign throughput.
Profiling or agreement with a saved run does not establish physics closure. Earlier
throughput results apply only to their frozen code and geometry, not automatically
to the corrected geometry implementation.

## Unfinished source lanes

`flash_progress` inventories capped and path-limit/looper source lanes from a saved
flux archive without rerunning transport. It reports saved identities, state and
step counters, and remaining primary kinetic energy. That remaining energy is not
missing cascade or annihilation energy, nor a completed-flux or detector-response
estimate. This inventory diagnoses truncation; it does not repair it or justify
removing affected events from the normalization.
