# Diagnostic keyed optical draws

`crvfastsim.veto.keyed_optics` is an opt-in preparation kernel for retained-sample
timing and response-seed studies. It is not wired into production response,
`replay_fast`, CE replay-v2 consumers, or the completed bounded timing controls.
No new stock history, veto fraction, finite-library error or promotion follows
from its synthetic validation.

## Identity and random contract

A call supplies an SHA256 of the original input, a uint64 response seed, and steps
with original `run, subrun, event, source_row` integer identities. The source row
is the original zero-based ROOT row, **never** an index regenerated after filtering
or concatenation. Within an input, source rows are unique. Validated identity columns
normalize to int64 (seed remains uint64), including empty partitions, so merging
signed/unsigned chunks cannot silently promote keys to float64. Empty inputs return
a typed empty photon table; an external runner must retain the all-event census,
including events with no steps or photons.

The full photon key is `(input_sha256, draw_seed, run, subrun, event, source_row,
chan, substep, photon_type, ordinal)`. Types 0/1/2 mean scintillation,
scintillator Cherenkov and fibre Cherenkov. Ordinal is the arrival number within
one substep/type, not a rank among all photons in a selected frame. Output order
is not identity. Compare sorted full keys when concatenating chunks.

The versioned optical domain uses the existing diagnostic FNV-1a/SplitMix64
uniform/Box-Muller primitive, with all four 64-bit words of the input digest and
all event/step integers. Photon-count, binomial-arrival, delay and emission draws
have separate domains; count keys omit substep/ordinal. No timing values, array
indices, total row counts or chunk boundaries enter the keys. This is a
non-cryptographic pseudorandom coupling, not collision-free mathematical random
independence or stock RNG equivalence. Its underlying helper and dependency
versions must be included in a run receipt.

Counts retain the production law: Gaussian(mean, sqrt(mean)) above 10 and
Poisson below/equal 10, divided by the geometric substep count and rounded to
nearest-even, clipped below at zero. Binomial thinning uses the existing lookup
probability/deviation/attenuation law with upper saturation at one. Discrete
Poisson/binomial draws use SciPy inverse CDFs at keyed 53-bit uniforms, clipped to
`[2^-53, 1-2^-53]` to avoid quantile singularities. This preserves the requested
laws to finite-grid/quantile precision, not NumPy RNG bit patterns. Delay and
emission bins use the production first-cumulative-weight-at-least-threshold rule;
zero-sum rows produce index zero as in that kernel. Positive fibre decay uses
an inverse-Gamma draw only for positive emission count. Platform/dependency
changes need revalidation; exact invariance is a within-environment guarantee.

## Geometry and arithmetic

Deterministic row preparation reuses production `iter_rows` with a zero-returning
adapter, discards its count placeholders, then computes independently keyed
counts with the same means. Thus local-coordinate transforms, reflectors,
per-sector yield selection, simulated-versus-output channel conventions and
propagation timing remain the production implementation. This adapter does not
consume a sequential RNG. The diagnostic substep kernel reuses the production
geometry/bin helpers and evaluates raw times in the same arithmetic order,
including delay quantization at enormous absolute times. It does not synthesize
raw arrivals by adding periods to folded photons.

NaN endpoints skip; infinities fail closed. Unknown PDGs, malformed identities,
negative/nonfinite active optical inputs, invalid probabilities and resource
limits fail rather than silently truncate. Substep expansion is bounded and may
split a step across internal chunks; substep identity remains its original rank.
A separate per-call photon cap is a resource guard, not a physics selection.
The total-work preflight defaults to 20 million simulated-channel substeps per
call, in addition to the 200,000-substep batch and five-million-photon limits;
even zero-yield steps cannot bypass that work bound. Exceeding any cap raises.
Lookup weights must use the production uint8 representation with checked sums,
bin dimensions and ordered finite edges. Propagation ratios must be finite and
positive. Content-qualified copies of lookup names protect the production
propagation cache from a same-name/different-content asset reload, without
mutating the original tables or changing the production module.

```python
from crvfastsim.veto.keyed_optics import make_photons

q = make_photons(steps, pdg, geometry, response_config, tables, deviation,
                 input_sha256=original_input_sha256, seed=3027000,
                 max_substeps=200_000, max_total_substeps=20_000_000,
                 max_photons=5_000_000)
```

The receipt must bind the underlying original input and asset hashes; this kernel
only validates the digest syntax, not a file at that path. Chunk-local validation
cannot detect duplicated source rows across separate calls; the future runner
must enforce global uniqueness and full event-census coverage. Numeric limits
are diagnostic admission bounds, not universally valid physics limits.

## Validation and next execution gate

Tests must establish exact sorted-key/time equality for permutation, subset,
external partition and internal substep chunk changes; common raw/folded keys;
separate seeds/input digests/full event tuples; typed empty/NaN-only cases;
fail-closed inputs and limits; all photon types and reflector orientations;
production deterministic geometry/timing parity; and count/thinning/delay laws.
Production source and prior saved evidence remain unchanged.

A retained-sample runner is still required. It must bind the input/asset/source
hashes, all-event census and original source rows; freeze a seed schedule; save
all four timing arms without deleting candidate identities; provide stable
full-key downstream roots; preserve every frame and union channel; and record
measured resources before a wider campaign. A replica is another response draw
on the same frames, **not additional exposure or upstream-library uncertainty**.
CE-selected joint exposure, physical source reuse and finite-library covariance
remain independent gates. C4 stays stock.
