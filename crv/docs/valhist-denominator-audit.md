# Native reach-denominator support audit

`valhist.denominator_audit` is a read-only complement to strict `denominator_variants`. It diagnoses a failed
support preflight and propagates whole-generator-file variability into the **existing reach acceptance
parameters**, not into downstream DQM/rate errors. It never writes a new library or weakens the strict builder.

```bash
# Use the unchanged uncertainty checkout/core required by the original sealed donor plan.
python3 -m valhist.denominator_audit AUDITED_LIBRARY/library.json \
  --out NEW_REPORT.json --replicas 1000 --seed 20261025
```

The nominal per-bin generator counts must exactly reproduce `libgen` on its existing support. All shower bins,
including bins absent from `libgen`, enter the report. Each bootstrap draw samples the original number of generator
files with replacement. The **same multiplicity vector applies globally** across all bins. Both primary counts
and complete generated-event exposure are weighted by occurrence before scaling to donor exposure and rounding.
This avoids deduplicating repeated file exposure while retaining repeated primaries.

The report records the raw ratio `nshower/ngen`, missing support, ratios above one, and every replica's support and
saturation diagnostics. Its covariance/quantiles refer to the native sampling behavior in
`tier1b._showers_for_gen`: a uniform random value in [0,1) is compared to the ratio. Consequently a ratio above
one already always accepts; an absent or zero denominator already rejects. Writing that existing behavior as
`min(ratio,1)` in the **diagnostic** is not an approved physics correction, nor permission to build a library whose
strict support checks fail. No failed bootstrap draws are omitted or replaced.

The full generator bank is assumed to comprise independent exchangeable source blocks. Only the denominator is
resampled; donors, conditional shower distributions and evaluation inputs are fixed. These are parameter-level
conditional covariance estimates. They are not a full finite-library systematic, downstream prediction covariance,
coverage calibration, or evidence for zero uncertainty in saturated bins. Missing support needs an explicit
physics/model decision before downstream denominator variants can be certified. Input/source hashes and the
base sealed audit are rechecked before writing a new report; no old plan is refreshed.
