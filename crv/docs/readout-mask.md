# Opt-in physical readout mask

Offline `v13_39_00`, `CRVResponse/src/CrvSiPMChargeGenerator_module.cc`,
checks `bar.hasCMB(SiPM % 2)` before channel status. Physical sensor presence is
not calibration status, channel-map/ordinal coverage, or an optical reflector.
In particular, Run2 R4 has reflector 0 but no side-1 sensors: channels with
SiPM indices 1 and 3 must not produce charges even when mapped and status-good.
A signed reflector changes photon propagation/routing, not this presence rule.

## API

```python
from dataclasses import replace
from crvfastsim import readout

# geom and counters_cfg must come from the same geometry configuration.
mask = readout.sensor_mask(geom, counters_cfg)
conditions = replace(conditions, readout_mask=mask)
```

`sensor_mask` returns a dense one-dimensional NumPy boolean array of length
`4 * geom.nbars`. For bar `b` and SiPM `s`, entry `4*b+s` is taken from
`crs.sipmsAtSide{s % 2}{geom.sector_names[geom.sector[b]]}`. Every named sector
must supply both side flags, even if it has no bars. Values must be actual
booleans (Python or NumPy), not strings, numbers, or truthy substitutes.
Missing flags, invalid geometry shapes/types, duplicate names, and unknown
sector indices raise `ValueError`. If counters include `crs.sectorNames`, it
must match the geometry's names and order. No response configuration, yield,
reflector, calibration, hardware map, or ordinal data are consulted.

`Conditions.readout_mask` is the last dataclass field and defaults to `None`;
existing positional construction and CSV loading remain unchanged.
`readout.effective_status(conditions)` returns the **original status object**
when the mask is `None`. With a mask, it requires a one-dimensional boolean
NumPy array exactly matching the status length, then returns a copy with bit 0
ORed into absent channels. Neither the stored status nor the mask is mutated;
all existing status bits are retained. This temporary status is an execution
adapter, not an edited calibration or a claim that absence is a database fault.

`chain.charges` applies the effective status at its existing pre-SiPM boundary
for NumPy and nonfused JAX. Fused JAX receives it through the existing
`response_batched` status argument; the device implementation is unchanged.
Photon generation is unchanged, so the mask does not suppress photon draws.
Absent channels do not enter SiPM simulation. With no mask, no additional RNG
draws are made and the original status object follows the existing path.

This primitive is opt-in. It does not change registry/configuration defaults,
CSV files, or named-condition bundles, and does not establish physics closure.
Named fasttables exposes `--geometry-readout` for named step modes only. It
refuses reco-only mode, existing output paths and normal receipt admission;
metadata records the policy, mask hash and channel counts. The inventory now
reports physical presence separately from reflector routing and hardware mapping.

## Named runchain opt-in

Ordinary `runchain` also accepts the mask explicitly:

```bash
python3 -m crvfastsim runchain gen.root --run-config run2_v10 \
  --conditions-manifest conditions/manifest.json --library library.root \
  --gdml matching-world.gdml --geometry-readout --outdir NEW
```

This mode requires Run1A or Run2 plus a strict conditions manifest and refuses a
counters override. The existing manifest/CSV mutual exclusion remains. It does
not certify the supplied library or world, establish named E2E closure, or change
timing. `--jax` and `--jax --fused-response` use the existing charge-stage mask
routes. Prepared contexts and benchmark callbacks reject the option **before
resolution or callbacks**: their current sealing/asset contracts do not admit it.

`readout.apply(conditions, geom, counters_cfg)` constructs replacement conditions
and returns the same `geometry_readout` evidence used by fasttables: the policy,
present/absent counts, and SHA256 of the compact JSON boolean mask. It validates
status/mask size without modifying either stored status or input conditions.
Runchain stats and emitted shards carry this evidence separately from the
unchanged conditions-bundle record. Disabled runs do not acquire this key.

Before summing counts, merge rejects marked/unmarked mixtures, unknown or
malformed policy, or differing mask hashes/counts. Homogeneous historical
unmarked shards remain unmarked, not retroactively certified. These checks
compare declarations; they do not authenticate execution or bind all response,
world, library, input and runtime assets.

## Focused local tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 CRVFS_NO_FETCH=1 \
PYTHONPATH=/path/to/core:$PWD \
python3 -m pytest -p no:cacheprovider -q \
  tests/test_readout_mask.py
```

`test_readout_mask.py` tests validation and mocked routing. The separate
`test_readout_execution.py` runs the actual NumPy, nonfused JAX and fused JAX
charge paths on CPU with tiny in-memory synthetic lookup tables and photon map;
no response implementation or loader is patched. It retains every registered
Run1A/Run2 sector name/yield/reflector but uses synthetic geometry and side flags.
In particular, its Run1A T2 missing-side case is artificial, not a detector census.

The execution regression covers surviving and forbidden channels, all-absent
output, status bits, unmodified condition arrays, and exact output/RNG equality
for no mask versus all-true. Call profiling refuses silent NumPy SiPM fallback
when testing JAX. On 2026-10-27, **18 tests / 42 real charge executions passed**
with NumPy 2.4.6 and JAX 0.11.2 pinned to `cpu:0`; no skips. The focused run took
3.06 s and peaked at about 554 MiB RSS. This is neither a performance benchmark
nor cross-backend random-stream identity: each backend is checked against its own
seeded controls.

`test_runchain_readout.py` adds 31 synthetic integration/admission/merge checks,
including both named configurations, serial/prefetch forwarding and legacy
metadata preservation. Its upstream/downstream pipeline is mocked, distinct from
the real charge-stage execution test. Run test files in separate processes or use
`python3 -m mu2efastsim.testselect --run --isolate -- -q`.

None of these tests fetches assets, runs a GPU, validates production optical
payloads, establishes an independent-seed response floor, or certifies a run.

## Saved Run1A product smoke

A separate local `fasttables chain --jax --geometry-readout` run used the saved
1000-event Run1A stock population, its G4 steps, strict named conditions and
existing real optical/map/waveform files, with `JAX_PLATFORMS=cpu`. Declared sector
yields were retained: **no shared-yield emulation**. It finished in 13.40 s,
maximum RSS 2.77 GiB. This is a single measured smoke, not a timing extrapolation.
Run1A has all 5120 sensors present, so this checks the all-true physical-mask path;
it does not exercise missing real Run2 sensors.

The vectorized saved-product audit verified 1000 identical full-width event
identities, retaining zero-response events. Against the reused NumPy A4 output,
summed pulse PE differed by **+0.021 ± 0.172% (T1)** and **−0.112 ± 0.463% (T2)**.
Against stock, T1 was **+0.131 ± 0.160%**, while T2 retained **−22.863 ± 0.357%**,
as expected from the unresolved stock shared-yield behavior. Errors are paired
event SE, without an independent-seed floor; these two sector means do not
establish spectral or full response closure.

Unlike the synthetic execution tests, this CLI smoke was not call-profiled to
exclude nonfused SiPM fallback; it records JAX-requested CPU execution, not proof
that every real-data kernel stayed on JAX. The diagnostic mask remains outside
normal receipt admission. No fresh stock, Geant4, GPU or Run2 real-asset run was
made for this smoke.

## Run2 real-asset traced seed checks

A later CPU-only Run2 check used the saved 1000-event population and identical
G4/conditions/optical assets, with declared yields and the physical mask enabled.
Four fresh seeds in each of NumPy, nonfused JAX and fused JAX produced **12**
complete outputs. All passed the [strict call observer](response-execution-trace.md):
requested response paths were observed, JAX array arguments were on `cpu:0`,
and no NumPy SiPM fallback was observed in any JAX replay. Every input/source/
optical pre/post hash check passed and matched across the sweep.

Run2's **19264 present / 2112 absent** channels use mask SHA256
`c8e77c3019f6a830825fe7e5b686c0900fe6ff595d2eef184fc2bb4bbc7f05b7`.
R4 SiPM indices 1/3 have **zero digi and pulse output in all 12 replays**.
All 23 sectors and every SiPM index are retained in the saved-product audits.

Summed pulse PE versus the fixed stock sample, **mean ± single-replay fast-seed
sample SD**, not paired-event SE or an error on the mean:

| Sector | NumPy, K=4 | Nonfused JAX, K=4 | Fused JAX, K=4 |
|---|---:|---:|---:|
| R4 | −0.527 ± 0.469% | −0.683 ± 1.614% | −0.562 ± 1.167% |
| T5 | −22.982 ± 0.268% | −23.257 ± 0.566% | −23.269 ± 0.124% |
| C2 | −3.457 ± 3.449% | −4.742 ± 3.219% | −2.919 ± 3.171% |

The mask removes the large unphysical R4 excess, while declared-yield T5 retains
the stock shared-yield discrepancy in every backend. C2 illustrates the large
seed sensitivity of sparse sectors. Do not rank backend precision from four-seed
SDs or tune to these means. The conditional seed audit (branch `park/named-run2`)
reports the full inventory and flags the small ensembles. This is no stock-seed
replication, spectral closure, cross-backend equivalence gate, GPU validation,
full V2 certification or default promotion.

Observed traced production wall times were NumPy **11.17–12.09 s**, nonfused JAX
**31.60–33.07 s**, fused JAX **32.01–34.13 s**; peak per-process RSS across the
12 replays was **5.30 GiB**. Each replay ran in a fresh serial process. Profiling
and cold starts are included, so these are execution measurements, not throughput
benchmarks or runtime projections.
