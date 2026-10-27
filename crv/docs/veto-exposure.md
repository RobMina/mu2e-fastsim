# Versioned PBI exposure for veto replay

The new CE-conditioned target is **opt-in**. The default remains
`legacy_unconditional_v1` so historical seeds, draws, frame choices and results
remain reproducible. This change does not rescale a saved veto-loss factor, change
response/reconstruction defaults, or certify a production library.

## Distribution contract

Let `M = extended_mean`, `s = sqrt(-ln(sdf))`,
`mu = ln(M) - s²/2`, `C = cut_max`, and `f` be the lognormal density
with normal parameters `(mu, s)`. The retained defaults are `M = 1.58e7`,
`sdf = 0.8`, `C = 9.48e7`.

| Mode | Raw density on `0 < x < C` | PBI used for Poisson means |
| --- | --- | --- |
| `legacy_unconditional_v1` (default) | `f(x) / integral_0^C f(t) dt` | Floating draw, no rounding |
| `ce_conditioned_v1` | `x f(x) / integral_0^C t f(t) dt` | Integer `llrint` of the raw draw |

Multiplying a lognormal density by `x` shifts its normal log-mean by `s²`;
its normal width stays `s`. CE sampling therefore draws a lognormal with
normal mean `mu + s²` and rejects values at or above the **same strict cutoff**.
The cutoff is applied before rounding. `sdf=1` is the degenerate point-mass case
and is supported only if that value is below the cutoff.

Offline **v13_39_00 and v13_41_00** `ProcessCode`, together with the retained
production configuration, classify `mu2eCeMinusLeadingLog` as beam-based.
Consequently the CE target uses the size-biased distribution, not the historical
unconditional draw. This corrects the previous false non-beam-based comment;
it does not relabel historical outputs as CE-conditioned.

Rounding implements stock `llrint` with the normal `FE_TONEAREST` rounding mode:
nearest integer, with exact half-integer ties going to even. The implementation
uses vectorized `numpy.rint` followed by signed `int64` conversion. It is not
truncation, floor, or half-away-from-zero rounding. No equivalence to a changed
floating-point rounding environment or to Offline's RNG sequence is claimed.

For each species, `n_frames ~ Poisson(N * efficiency_product * extra_factor)`,
where `N` is the floating legacy PBI or rounded CE PBI. Existing mixer constants,
species iteration order, bunch-time choices and step-selection operations are
unchanged. Counts include drawn stepless frames, not only frames contributing
steps. There is no additional scalar adjustment to method F.

The lower-level mixer still accepts `intensity_scale`; its explicit ordering is
truncate raw PBI, multiply by this positive scale, then round for CE. The replay
exporter fixes this scale to one. A nonunit scale is a diagnostic override, not an
automatic production correction.

## APIs and production guard

- `exposure.ExposureContract(mode=...)` validates parameters, samples raw PBI,
  and supplies a JSON-safe `crv-pbi-exposure-v1` mathematical record.
- `mixframes.draw_intensity(..., mode=...)` returns the raw floating draw.
- `mixframes.mix(..., exposure_mode=..., return_events=True)` also returns
  event conditions; CE mode retains raw/rounded PBI and species counts without
  consuming any additional random numbers. The normal four-item return remains
  available.
- The lower-level mixer accepts synthetic legacy-shaped pools for unit tests.
  **Calling it directly is not a production-validation gate.**
- `replay.export(..., exposure_mode="ce_conditioned_v1")`, and the replay CLI
  equivalent, require **every** source table to declare metadata schema
  `crv-cat-v2`. They then load each species through `catsteps_v2.PoolV2`, which
  validates its all-event census, tuple-keyed identities, payload bindings and
  provenance. Legacy generator-row denominators cannot pass this gate; there is
  no permissive legacy override.
- The `mixframes` response CLI remains legacy-only. It does not expose an ungated
  CE production route.

An explicit CE replay invocation, only after separately reviewing the new library,
is:

```bash
python3 -m crvfastsim.veto.replay --tables "$VALIDATED_V2_TABLES" \
  --events "$EVENTS" --seed "$SEED" --exposure-mode ce_conditioned_v1 \
  --out "$NEW_REPLAY_ROOT"
```

Omitting `--exposure-mode` preserves the legacy behavior. Both modes refuse to
replace existing replay files. Unsupported mode names, nonfinite/nonpositive
parameters, `sdf` outside `(0, 1]`, invalid sample counts and unsafe integer PBI
ranges fail. Extremely inefficient rejection configurations are rejected below
acceptance `1e-6`, and the sampler has a 10,000-batch failure limit rather than an
unbounded hang. No production generation is part of the local tests.

## Replay schemas and provenance

The legacy replay remains **`crv-mixed-replay-v1`**, with exactly the historical
`dump/events` branches: `run`, `subrun`, `event`, `pbtime`, `intensity`.
Its scope text explicitly labels the legacy mode. No exposure/count branches or
new required contract objects are added to legacy output.

Opt-in CE replay is **`crv-mixed-replay-v2`**. It preserves those base branches and
adds:

| Event branch | Type | Meaning |
| --- | --- | --- |
| `intensity_raw` | float64 | Truncated raw PBI, before rounding (scale is one in export) |
| `intensity_rounded` | int64 | Rounded PBI actually used for Poisson means |
| `n_frames_<species>` | int64 | Sampled count for each of the four configured Cat species |

The existing `intensity` float64 branch contains the rounded PBI represented as
a double, preserving the response reader's base branch types. All events,
including fully stepless mixtures, retain these columns. The exporter validates
the raw/rounded/Poisson relationship and nonnegative integer counts.

The CE JSON records the exposure schema/mode, mathematical density, normal
parameters, cutoff, rounding, scale order, target classification, species
constants, branch types, `crv-cat-v2` validator, source metadata and hashes,
output hash, seed and package versions. Code hashes include `exposure.py`,
`mixframes.py`, `replay.py` and `catsteps_v2.py` (plus the legacy column-definition
module). This is a versioned provenance record, not a certification label.

The legacy `Pool` rejects versioned tables even when the exposure option is omitted.
The current `replay_fast` and `replay_check` paths reject replay-v2 pending explicit
consumer integration, rather than silently dropping its exposure contract.
Downstream consumers must explicitly accept the v2 schema and preserve/bind its
exposure contract before pooling or comparing results. Historical retained
replay remains v1 and must never be silently upgraded or pooled with CE exposure.
A matching base ROOT branch layout alone does not establish exposure equivalence.

## Remaining review gates

`PoolV2` establishes structural census/provenance consistency. It does **not**
certify source completeness, source independence across production subsets,
efficiency/IOV provenance, the production sampling plan, response closure, or
the factorisation between selected tracks and PBI. Those remain separate review
gates. In particular, a ratio of PBI means cannot be applied to a saved F as a
physics correction; the nonlinear occupancy/veto calculation needs a new,
properly conditioned sample after those gates are reviewed.

Local synthetic tests cover exact historical RNG/output regression, the shifted
and truncated law, nearest-even rounding before Poisson, retained per-species
counts and empty events, malformed inputs, legacy schema preservation, rejection
of legacy libraries for CE export, and a census/provenance-bound synthetic v2
export. Run with this checkout and the detector-independent core on `PYTHONPATH`:

```bash
python3 -m pytest -q tests/test_veto_exposure.py tests/test_veto_replay.py \
  tests/test_veto_catsteps_v2.py
```
