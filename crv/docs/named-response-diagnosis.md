# Named response: sector-resolved diagnostics

The named response comparison aggregates sectors of a common type. That can hide
configuration defects within a group. `analyses.named_response_audit` adds a
saved-product diagnostic for every individual sector and SiPM index, retaining
all events, including those with no response. It does not replace V2 or admit
producer provenance.

```bash
python3 -m crvfastsim.analyses.named_response_audit \
  --reference stock.root --candidate candidate.root \
  --conditions-manifest conditions/manifest.json --run-config run2_v10 \
  --outdir NEW_AUDIT
```

Inputs must have identical nonempty, unique full-width run/subrun/event triples;
foreign digi/pulse rows fail. The module uses vectorized NumPy/Awkward operations,
not event loops. It writes `audit.json` and `table.md` into a new directory. All
input hashes are recorded and rechecked. Every sector and all four SiPM indices
are retained, including zero-reference channels with explicitly unsupported
relative differences. No absent sensor is silently removed from the audit.

Metrics include digi/pulse counts, pulse PE sums, sequence-zero PE sums, saved
ADC sample count and pedestal-subtracted ADC sums. ADC sums are sums over saved
zero-suppressed samples, not deposited energy. Per-channel area-gain-normalized
ADC is diagnostic and is not a calibrated PE estimator with a sample-period
factor. Sequence-zero medians are descriptive, without an error estimate.

For event totals C and R and ratio q=mean(C)/mean(R), the relative difference is
q−1 and its paired delta-method standard error is
`std(C - q*R, ddof=1) / (sqrt(N)*mean(R))`. Zero-response events participate.
This accounts for shared-event covariance, not independent-seed fluctuations,
finite simulation-library uncertainty or systematic errors. Sparse sectors are
not a basis for physics promotion.

## Stock shared-lookup yield behavior

Offline `v13_39_00` `CRVResponse/src/CrvPhotonGenerator_module.cc` shares its
mutable `MakeCrvPhotons` object when lookup filenames match. It calls
`SetScintillationYield` only when constructing the first object, not before each
sector's step. Consequently the first sector's yield is reused:

| Configuration | Affected sector | First sector for `6000_0` | Declared yield | Effective stock yield |
|---|---|---|---:|---:|
| Run1A | T2 | T1 | 30482 | 39794 |
| Run2 | T5 | T3 | 30482 | 39794 |

The proposed narrow stock correction and actual-module regression oracle are
recorded in [stock-shared-yield-correction.md](stock-shared-yield-correction.md).
That is a source-reviewed plan only, not an applied or built Offline patch.

The fast photon code uses the declared yield for each sector. The stock debug
message prints the declared value even for a shared object, so that message does
not diagnose the discrepancy. The source behavior is not evidence that the
larger yield is the physically correct calibration.

`response.shared_yield.emulate(response, resolved_fcl)` reproduces only this
first-yield behavior. It validates the resolved declared sector order, lookup
names, yields, reflectors and global scales against the named response before
creating a new configuration. It never mutates a registry object.

```bash
python3 -m valhist.fasttables chain g4steps.root diagnostic.root \
  --run-config run1a_v01 --conditions-manifest conditions/manifest.json \
  --event-reference stock.root --stock-shared-yield-fcl resolved.fcl
```

This opt-in requires named step mode and a stock event reference, refuses existing
outputs, and records `diagnostic_override` with exact changes and resolved-FCL
hash. It cannot enter normal receipt admission. It is stock-behavior emulation,
not yield tuning, a stock fix, or a promoted named configuration.

On 1000 matched stock events, the summed pulse-PE relative differences were:

| Sector | Declared fast response | Shared-yield diagnostic |
|---|---:|---:|
| Run1A T2 | −22.777 ± 0.344% | −0.673 ± 0.381% |
| Run2 T5 | −23.240 ± 0.440% | +0.377 ± 0.445% |

Run1A's implemented aggregate metrics pass in that diagnostic. Run2's top
aggregate metrics pass for yield-only, but this is not full named V2 and no
independent seed floor was evaluated. Changing one yield alters subsequent RNG
consumption, so unchanged sectors are statistically comparable, not bit-identical.

## Independent physical readout omission

Run2 R4 has reflector 0 but `crs.sipmsAtSide1R4=false`. Stock checks
`hasCMB(SiPM % 2)` separately from status. Status and ordinal mapping are valid
on R4's nonexistent side; neither substitutes for sensor presence. The default
fast response lacks this geometry gate.

The saved stock/candidate R4 digi counts by SiPM were `28→30, 0→27, 27→29, 0→27`.
An opt-in [physical readout mask](readout-mask.md) prevents charges at indices
1/3 without rewriting conditions. Named fasttables exposes it as
`--geometry-readout` in step mode only, not reco or normal receipt mode. Metadata
records its policy, present/absent counts and mask hash. Run2 has 19264 physically
present channels and 2112 absent: 1984 reflector-side channels plus 128 R4
channels that photon generation otherwise populates. All physically responsive
channels are mapped.

The R4 pulse-PE excess is +84.003 ± 16.180% in the original saved response and
+86.518 ± 16.391% with yield emulation only. Adding the physical mask gives
−2.252 ± 1.748%; forbidden-side response disappears. The combined diagnostic's
T5 difference is −0.247 ± 0.465%. Production defaults remain unchanged.

The combined Run2 diagnostic still has a top spectrum metric failure and sparse
E/U/D/C spectrum support, despite the resolved large PE defects. Small-sample
spectral gates fluctuate as RNG consumption changes. Do not tune to those bins,
relax thresholds, or claim all-sector response closure.
