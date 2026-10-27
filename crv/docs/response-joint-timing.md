# Retained-array pulse timing attribution

`python3 -m crvfastsim.objective.joint_timing --directory BLOCK --data DATA --out NEW.json`
reads the four frozen MC captures and the complete P4 capture without running response, refitting lines,
changing time calibration/background, rescoring, or pairing pulses across arms. Existing replay, configuration,
exposure and full-identity admission gates are reused from `joint_compare`. Source and input artifacts are
hashed before and after reporting. Output is exclusively created. MC arrays are released before loading data.

## Populations and standardization

The pulse population is complete selected event/slot spread groups with both measured projections, >=3 members,
and finite residual/time/distance for every member. A contaminated group is excluded whole, not silently
recentered on finite members; flows report this separately. Body denotes the actual pulse body. Event-weighted
multiplicity and RMS tables instead use the original event/slot representative-body convention.

Sparse residual cells retain body × measured along × measured path × measured PE, with the previous bin edges.
Compare three versions: no multiplicity coordinate, bands N=3–4, 5–8, 9–12, 13–16, 17–24, >=25, and exact N.
The >=25 band has an int64-max sentinel; actual complete-group counts cannot reach it. Common cells require
>=30 finite hits **and >=10 distinct event/slot groups in every MC arm and data**. Fixed data hit fractions
weight moments on that common support. Both support-matched own-fraction moments and coverage are reported.
These are hit-weighted descriptive moments, not independent-hit uncertainties or intrinsic sensor resolution.
Each multiplicity mode has its own support and target weights; cross-mode differences do not isolate a
multiplicity effect. Exact-N results characterize only the supported population, not the excluded tails.
Per-PE tables use the same within-PE common cells, not unrestricted PE marginals. Empty/sparse sectors remain
explicit; no cell threshold is relaxed to make a plot.

For each retained pulse define a = time - full-group mean(time), b = distance/140 - full-group mean(distance/140),
and r = a-b (the saved propagation-corrected residual). Reports retain E[r], E[r²], E[a²], E[b²], E[ab]. The
identity E[r²] = E[a²]+E[b²]-2E[ab] separates algebraic contributions, **not causal sources**: measured lines,
selection, PE and pulse timing covary. Hardware moments by layer, SiPM index and mapped FEB are unstandardized.

## Within-group pair correlations

Pairs are unordered pulses within the same event/slot **and actual body**; they never cross arms or events.
Relations include all, same/different layer, same/different SiPM index, same/different mapped FEB, same channel,
same-bar/same-end distinct SiPM, and same-bar/opposite-end. Relations overlap and must not be summed. SiPM index
is 0–3 within each bar, not a global sensor identity; channel = 4*bar+SiPM. Repeated pulses in one channel remain
in the same-channel report but are excluded from the distinct-SiPM control.

Each relation retains pair and distinct-event/slot counts, symmetric pooled endpoint moments/covariance/
correlation, E[(r_i-r_j)²], and its raw/propagation/cross decomposition. Pair weights grow as N(N-1)/2; an
additional group-weighted mean pair difference moment is supplied. The common estimated group mean induces
anticorrelation: a complete, single-body, fixed-N group's pooled pair correlation is -1/(N-1) identically.
An observed negative correlation is therefore not evidence for an electronics mechanism.

Same-bar/same-end pulses share the same line propagation correction. Their difference removes that correction
and the group mean exactly, while retaining optical/sensor/fitting/selection effects. Additional pair tables
require **both** pulse PEs in the same PE bin. They have no multiplicity/position/path standardization; differing
pair populations still limit MC/data or arm comparisons. Neither pulse count nor pair count is an independent
sample size. Sufficient-statistic reductions avoid event loops and quadratic pulse-pair materialization.

## FEB and timing evidence limits

The only map admitted here is the calibration CSV already hashed in the frozen MC manifest; only its channel,
ROC and FEB labels are read. `(ROC,FEB)` jointly identifies a mapped FEB. Data labels are **projected through
that MC map**, not certified as physical P4 cabling. In MC v05 the phase is common per event, regardless of those
physical labels. The retained T1 hits occupy two mapped FEBs per body (ROC 2, FEBs 1/2 for T10 and 3/4 for T11).
Uninstrumented CSV channels may have paired ROC/FEB=-1 sentinels; a selected hit with unavailable mapping is
refused. Do not infer a missing contrast is zero correlation or that EX/T2's layer-pair proxy describes taggers.

The capture has reconstructed first-pulse time and PE, not photon-arrival samples, fit beta/chi2, waveforms or
clock-phase samples. PE-dependent trends cannot distinguish arrival statistics from fit bias or electronics.
The 500k data events are a subset of the original scoring reference, not an independent calibration holdout;
the two MC blocks demonstrate consistency, not calibrated significance. No model, offset, optimizer setting,
GP noise or detector default is changed.

## Tests

`tests/test_joint_timing.py` checks tiny explicit pair oracles against vectorized reductions, cancellation and
centering identities, exact-N cells, group support gates, invalid hardware/map rejection, nonfinite whole-group
exclusion, empty inputs, row permutation, all-sector coverage and overwrite refusal. Synthetic tests do not
establish physics acceptance.
