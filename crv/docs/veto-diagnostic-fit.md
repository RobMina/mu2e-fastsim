# Diagnostic fit arithmetic and waveform allocation admission

`veto.diagnostic_fit` is a separate diagnostic path. The production pulse fitter,
existing waveform helper, saved downstream control and frozen qualified-response
adapter remain unchanged. No completed output is relabelled with this contract.

## Fit-window width is part of the arithmetic contract

The production vectorized fitter pads the fit windows of active waveforms to the
largest current width. Even masked zeros can change the floating-point reduction
order. Keyed random inputs do not make those fits partition-exact.

The diagnostic uses the same peak search, calibration, failure flags, Gumbel fit
and subtraction equations, but groups fit windows by their **own valid width**.
Each group passes an unpadded contiguous matrix to a diagnostic copy of the LM
kernel. The equations are unchanged, but a linear-algebra exception or nonfinite
arithmetic aborts the batch: the production kernel's batch-wide fallback must not
silently alter healthy companion rows. There is no scalar retry or pulse dropping.
Loops are over width buckets, pulse rounds and the sample axis, not events or
individual pulses.
Full recovery continues past the default eight-pulse cap. A waveform's full tail
is retained. Consecutive digis are joined by the original rule; nonconsecutive
digis remain separate even when their padded zero-suppressed tails overlap.

Synthetic tests hold the ADCs fixed and exercise subset, permutation, matrix
padding, long pulse trains and the previously nonexact qualified-response case.
Exact comparisons use every fit field and flag, not a numerical tolerance. These
are within-environment tests, not a cross-platform bit-equivalence claim, stock
Minuit equivalence or certification of untested real inputs. The new arithmetic
can differ slightly from the previous batch-dependent fit; that difference must
remain explicit when comparing outputs. Production defaults do not change.

## Allocation guards

The fitting path currently admits the default reconstruction configuration only,
integer ADCs in `[0,4095]`, pedestal in that range, and pulse-area/height calibration
in `[1e-12,1e12]`. Finite offsets and origins are required within `±1e12 ns` and
start TDC within `±2**40`. These are conservative diagnostic safety domains, not
new detector cuts. A domain rejection leaves the batch incomplete.

`diagnostic_fit.reconstruct` bounds total sample rows, concatenated waveform
length and the dense **waveform-count × maximum-length** matrix before calling
concatenation. `reconstruct_matrix` repeats admission before making fit scratch
arrays. Rejection is a failed batch, never sample clipping or exposure removal.

`veto.bounded_waveform` checks charge count, template size, charge-count × reach
and a conservative cluster-count × maximum-length upper bound **before** calling
the frozen diagnostic synthesis helper. The reach axis is independently capped
even for empty charge input. Configuration, windows, template values and numeric
dtypes are validated, and a conservative cluster amplitude bound includes the
keyed Box–Muller maximum (38 sigma), preventing unsafe int16 narrowing. Charge /
reference-charge ratios above one million are rejected, never clipped. The
supported ADC configuration has integer bounds/pedestal in `[0,4095]`, finite
nonnegative noise/threshold and finite positive conversion. The latter cell bound is calculated from
cluster endpoints and the maximum possible charge reach, without allocating the
two large matrices. It deliberately counts below-threshold clusters too. Quiet
channel phases and full zero-suppression tails are unchanged.

Cell caps bound the named arrays, not peak RSS, all simultaneous temporaries,
ROOT decompression, pandas storage or physical disk quota. Photon/thermal/cascade
limits, artifact transactions and a complete provenance-bound executor remain
separate admission requirements. A passing synthetic fit test does not admit a
full retained-sample campaign or promote the veto objective.
