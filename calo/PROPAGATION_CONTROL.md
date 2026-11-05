# Propagation-table and public-stock control

This additive control does not change stock producers, the shower library, chain defaults or the acceptance proxy.
The histogram audit runs in Python. The separate user-built public-stock analyzer has passed a scoped runtime
comparison with the supplied histogram and prescribed uniforms; any future C++ edit requires another user build.

## Two evidence levels

1. `scripts/check_propagation.py` reads an actual `TH2F` through uproot, excluding flow bins. It reconstructs
   ROOT TAxis regular-bin arithmetic, float32 first depth-bin width and time-bin centers, then the source-order
   CDF. It compares supplied-uniform NumPy and optional eager/JIT JAX lookups against an independent host
   `searchsorted` reference. This is **source/table conformance, not observed stock cache execution**.
2. `CaloPropagationDumper` constructs the **unmodified public stock** `CaloPhotonPropagation`, calls `buildTable`
   once and calls `propTimeSimu` with prescribed uniforms delivered by CLHEP `NonRandomEngine`. Its actual
   returned times and counted engine calls are independent observations. `--stock-control` checks these outputs
   and requires the full expected probe registry. This level has **passed a scoped user-built runtime control**
   against NumPy and eager/JIT CPU JAX with caller x64 enabled. This is a finite-probe result for the pinned inputs,
   not every representable depth/uniform, another table or another device.

The private stock `cdf_` is never accessed. The analyzer exports actual ROOT contents/centers and separately labelled
**reconstructed probe thresholds**, not a dump of the stock cache. Passing public calls establishes scoped
build-plus-lookup behavior, not the contents of every internal byte or production RNG equivalence.

## Numeric contract

- Stock uses only the first X-bin width, without subtracting the depth-axis origin. Depth is supplied as
  `crystal_length - local_z`; there is no interpolation, `FindBin` replacement or analytic stand-in.
- Regular TH2F contents are float32. Stock adds them sequentially to a float32 accumulator initially `1e-6f`,
  then divides each cumulative float by the final float sum. Flow bins are ignored, even when nonzero.
- ROOT variable-bin centers use `low_edge + 0.5 * width`. Uniform-bin centers use `xmin + (bin - .5) * width`.
  Python axis reconstruction remains a prediction until checked against the analyzer's observed ROOT calls.
- Stock casts the float32 depth/width quotient to signed int **before** clamping. The Python host rejects
  nonfinite or out-of-int32 truncated quotients, where stock has no portable result. JAX callers must prevalidate
  this domain and all input/table validity on the host.
- Each supplied float32 uniform selects the first CDF bin `>= uniform`, including plateaus and exact 0/1.
  A double CLHEP draw below one can narrow to float32 one; endpoint tests are intentional.
- All-zero slices produce an all-ones CDF and select the first bin. They are not dropped.

Probe design covers each reconstructed CDF threshold and its float32 neighbors, every depth boundary and its
neighbors, endpoint uniforms, and finite negative/high depth clamps. The source-order CDF and independent
`searchsorted` reference are not stock observations. Counts and bit comparisons are reported rather than fitted.

## Build and execution gates

Only the user builds the satellite. New `CaloPropagationDumper` dependencies on `CaloMC` and `ConfigTools` are
explicit in both CMake and SConscript. Earlier response/export analyzers and stock sources are unchanged.
Do not load the new plugin until successful user build completion, completed build metadata, source/dependency
freshness and clean `readelf`/`ldd -r` linkage are verified.

The one-event FHiCL uses `EmptyEvent`, not a saved art input or Geant4. Geometry is still required because stock
`buildTable` reads the refractive index. `propTimeLine` is outside this control; recording the refractive index
alone does not validate the straight-line method. No new transport or random production sample is needed here.

`scripts/propagation_control.sh` requires explicit `CALO_CODE`, `CALO_AREA`, `CALO_OUT`, `CALO_SIM_OFFLINE`,
`CALO_BACKING`, `CALO_ENVSET`, `CALO_PLUGIN_SHA256`, `CALO_TABLE_SHA256`, `CALO_STOCK_SOURCE_SHA256`.
It requires the already-set-up work area/envset/backing, a fresh output directory and fresh verified plugin.
It verifies the resolved table and source hashes, requires the linked CaloMC library to be the pinned backing
library, records its hash/linkage, effective FHiCL, runtime, geometry and process receipt. It never builds.
External preflight must verify the expanded FHiCL and resolved table path; the analyzer records its own path too.

Python invocation, with paths/checksums supplied by the operator:

```bash
PYTHONPATH=calo python3 calo/scripts/check_propagation.py TABLE.root \
  --input-sha256 TABLE_SHA --stock-source CaloPhotonPropagation.cc \
  --stock-source-sha256 SOURCE_SHA --out TABLE_REPORT.json
# Add --backend jax, optionally --jax-jit; enable caller x64 when desired.
# After successful stock execution, add all three pins:
# --stock-control STOCK_OUTPUT.root --stock-control-sha256 OUTPUT_SHA
# --stock-resolved-file EXPECTED_RESOLVED_TABLE_PATH
```

Reports refuse overwrite and retain numeric failures. Stock comparisons require an expected output hash and
observed resolved file/histogram metadata; the driver receipt/source-build manifest supplies the external binary
and condition provenance. Unit ROOT-output fixtures are synthetic and are not evidence that the analyzer compiled
or ran. CPU JAX exactness does not imply another device/compiler's exact division.

## Separately scoped stochastic response follow-up

1. **Travel only:** `TRAVEL_CONTROL.md` defines the now-executed stock arm with PE statistics disabled and travel
   enabled, fixed-phase unsmeared timing and unchanged conditions. Reusing the saved deterministic baseline,
   exact event/CSS/conditions/summaries, pointer/SiPM NPE and all conditional float32-arrival support checks passed.
   The separate post-support ideal-uniform diagnostic is not actual engine-law or stochastic-response certification.
2. **Poisson only:** `POISSON_CONTROL.md` defines the now-executed PE-statistics-on/travel-off arm. Exact saved
   inputs/conditions/summaries and no-travel float32 times passed. Zero-fill every accepted CSS/SiPM input, including
   absent positive-mean readouts, before a separately frozen ideal independent-Poisson CDF diagnostic. That model
   was not rejected; actual stock RandPoissonQ uses a Gaussian approximation at means at least 100, so this is not
   stock-sampler/engine validation or eventwise seed identity. No sample zero means occurred (synthetic tests only).
3. **Combined:** `COMBINED_CONTROL.md` defines the now-executed both-switches-on arm. Exact input/condition/summary
   invariants, a complete zero-filled count ledger and all conditional float32 arrivals passed. Prospective count,
   conditional-travel and joint count/mark ideal-model diagnostics were not rejected (suite union bound .03 under
   the ideal null). No separate-arm realization equality, actual-engine law or downstream certification is claimed.

The analyzer exports observed PE times/count realizations, but not production uniforms, engine state or the
sampler's internal draws. Predicted means are not observed stock intermediates.
Matching seeds across CLHEP and JAX is **not** an exact-realization control. Either instrument an explicitly reviewed
realization ledger, or use separately labelled statistical tests with fixed bins, a predeclared multiplicity correction
and calibrated null coverage before looking at a candidate. No parameter tuning to the response is allowed.

For travel tests, do not infer a sampled bin by subtracting folded time: stock first float-narrows folded time, then
float-adds the sampled bin center. At large absolute times distinct bins can collapse to the same stored float.
Construct conditional probabilities for the **observed float32 arrival values** by grouping all table bins that map
to the same `float32(base_float + time_center)`. Use CDF increments (including the floor and plateaus), not normalized
histogram counts. Exact CLHEP-to-float draw discretization near thresholds is not validated by a continuous-CDF test.
Keep late finite inputs and use an explicitly scoped distributional tolerance/null calibration; no clipping/dropping.
If `G(x) = P(float32(engine_double) <= x)`, the actual bin masses are `G(F[0])` and `G(F[j])-G(F[j-1])`,
not necessarily the continuous-uniform CDF differences. Even continuous engine doubles acquire rounding-cell
boundaries when narrowed to float32; pin and model the engine law or label that distribution test approximate.

The separate and combined arms establish only their stated hard gates and ideal-model diagnostic results.
Production RNG, actual sampler law, pulse/noise caches, digitization, TemplateFit, current hit/cluster/match
reconstruction and unbiased-library certification remain open.
