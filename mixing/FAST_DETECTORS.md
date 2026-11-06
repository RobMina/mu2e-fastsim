# Mixed steps to fast tracker and CRV response (development)

This opt-in path replaces stock CRV response/reconstruction with the existing
staged JAX kernels and can run the fast tracker in the same process. It does not
simulate transport, change pileup laws, or concatenate independently digitized
donors. All sources' CRV steps share one event/channel recovery and waveform history.

## Cached inputs

`fcl/four_stream_detector_steps.fcl` retains the native four-stream mixer, omits
unused calorimeter steps, and exports all five tracker and CRV input collections.
No tracker/CRV response or reco is scheduled. The EventLister log is the independent
all-event census for the existing legacy CRV dumper. `MIXING_EXPORT_CRV_CONDITIONS=1`
in `run_response.sh` exports campaign CRVPhoton deviations. This is a one-time input
cache for response development, not a new Geant4 campaign.

The current paired replay explicitly reuses two saved timing tables: `--truth` is
the tracker-input reference whose `crvdump/events.pbtime` exposes unsmeared timing;
`--stock` is the original stock response with reconstructed bunch time. IDs/order
and lengths must agree with the extraction. The adapter does not invent missing
timing or infer it from pulses. Both timing arrays, including zero-step events,
are supplied to `crvfastsim.onspill.EventTiming`.

Tracker step tables must be exactly equal to the saved full extraction. CRV PDGs
are joined to SimParticles by full event IDs and particle key within each source.
Calibration/status/channel mapping comes from the saved stock dump; photon
deviations come from the explicit campaign export. No nominal conditions fallback.
The parsed expanded FHiCL is checked against the supported Run1A response settings.

## GPU-backed driver

```bash
python3 -m mixing.fast_detectors --input MIXED_STEPS.root --truth TIMING_TRUTH.root \
  --stock STOCK_RESPONSE.root --photon-csv db_CRVPhoton.csv --out NEW_OUTPUT \
  --repeats 3 --chunk 32
```

The CRV package requires the opt-in onspill adapter and JAX thermal support. Supply
its existing staged lookup tables, geometry/counters and waveform template through
normal package asset variables. The project image is reused; nothing is installed
or built. For a combined response timing add `--tracker-conditions conditions.npz`
and `--tracker-reference SAVED_FAST_TRACKER.root`. Tracker seed defaults to the
recorded development seed; its detector columns must reproduce the saved output.
`--max-events` selects an explicit development prefix, not a runtime extrapolation.

Passes reset RNGs and must repeat their arrays exactly. The first pass includes
cache loading/compilation; later passes measure warm response. Input validation,
conditions/table loading and file writing are timed separately. Chunk size is part
of the random-domain definition and must be held fixed in comparisons. The driver
uses the host for packing, thermal attempts, transfers and validation; GPU-backed
does not mean fully device-resident. Stock and JAX use independent response draws:
CRV closure is statistical, unlike the exact mixer/table or tracker replay checks.

Output `fast.root` contains full event IDs/timing, CRV diagnostic digi/pulse/cluster
tables under `crvfast/`, and optional native-compatible tracker detector tables
under `trkdigi/`. CRV output is intentionally a reduced diagnostic schema, **not**
a complete `CrvRecoTableDumper` or art-product bridge. No reconstructed tracker
tracks, calorimeter products, trigger decisions or complete MC truth are produced.
`report.json` records stage timing and paired totals, all represented CRV sectors,
PE and time shapes with under/overflow bins. Empty events remain in denominators.
No held-out occupancy asset or response parameter is changed by this development path.

## Direct flat-frame input (opt-in)

`--frames-manifest direct_manifest.json` replaces `--input`. It loads the complete
joint signal/four donor pools, calls the existing bunch law, four conditional
Poissons and sequential/wrapping selections, then feeds the overlaid steps directly
to the same two detector consumers. There is no art mixer or intermediate mixed
step export in this path. The existing native-cache path remains available.

Extract each source once with `run_frame_cache.sh SIGNAL STOP ELE MU NEUTRAL`,
using the same existing runtime/plugins as above. It reads all events (`-n -1`),
retains the legacy EventLister census and exports actual `SimEfficiencies2` for
Sim_best/v1_3 run 1430:1253. `prepare_direct CACHE --expanded expanded.fcl
--beam-primary` checks successful extraction, reads the campaign laws/skip bounds,
requires universal/no-offset mixing, and writes the manifest. The beam-primary
flag is an explicit declaration about this saved CE signal, not an inference from
an input-tag name. Efficiencies use the stock float32 parse of exported values,
not rounded log values, file counts or recomputed numerator/denominator ratios.

```bash
python3 -m mixing.fast_detectors --frames-manifest CACHE/direct_manifest.json \
  --expanded expanded.fcl --truth TIMING_TRUTH.root --stock STOCK_RESPONSE.root \
  --photon-csv db_CRVPhoton.csv --tracker-conditions conditions.npz \
  --out NEW_OUTPUT --repeats 3 --chunk 32
```

True/reconstructed bunch times still come from the explicit saved primary timing
censuses, aligned by full IDs. Input steps are never folded by the mixer. The
manifest and sampling receipt preserve full IDs, sparse particle/generator links,
joint donor occurrences and empty events. Output remains reduced diagnostics,
not complete native MC truth. No saved tracker equality is required or allowed:
NumPy draws the stock probability law, **not stock C++ RNG sequences**. Comparisons
are paired by primary ID but have independently sampled pileup as well as response.

`sampling.json`/`sampling.npz` are saved before GPU response, including intensities,
per-event counts, selected donor indices/source IDs and stream cursors. Frame read,
sampling, overlay/validation and adapter times are separate from warm response.
`setup_s` includes `input_s`; do not add them twice. Source extraction is a one-time
external cost. Full host overlay currently precedes response chunking: `--chunk`
bounds detector batches, not host assembly memory. Changing `--max-events` changes
the sampler RNG realization; a short run is not an exact prefix replay. Held-out
inputs/assets are not used here.

The full direct development run completed 2,741 events, with 3,470,922 tracker and
1,338,722 CRV steps. It produced 1,694,559 tracker digis, 74,857 CRV digis,
78,155 pulses and 41 clusters; all three repeated response passes were identical.
Paired ratios to stock: tracker `1.00077 +/- 0.01490`, CRV digis
`1.00037 +/- 0.01738`, pulses `0.99806 +/- 0.01763`; T1/T2 pulse ratios
`0.99822 +/- 0.01762` / `0.98850 +/- 0.08442`. There are only 38 stock clusters.

Warm combined response takes `124.318/124.308 s` (`22.05 events/s`): CRV
`28.141/28.220 s`, tracker `94.315/94.251 s`. First response is `184.020 s`.
Input handling takes `255.963 s`: read/hash/validation `90.827 s`, sampling
`0.089 s`, overlay/validation `100.688 s`, adapter/validation `64.296 s`.
Setup totals `258.705 s` including input; output writing is `2.408 s` separately.
This demonstrates the direct architecture, **not an end-to-end speedup**. Host
assembly and repeated reference validation now dominate preparation. Complete
source extraction took 680 s job wall time once; new samples reuse those pools.
The cached-native measurements below use different pileup draws and remain a
separate response-only benchmark, not a controlled throughput comparison.

### Host preparation optimization

Event-census and particle-reference checks now use integer tuple lookup indexes
instead of repeatedly sorting structured arrays. Each fresh reference validation
builds one particle index for parents and all detectors, and returns the checked
flat row indices. CRV PDGs use those indices directly rather than another full
particle DataFrame/merge. Checks are not cached on mutable `FrameBatch` objects.
Sparse keys, full uint64 IDs, null links, empty frames and stable row order remain
covered by independent structured-array oracle and mutation tests.

For repeated assemblies in one process, create `FrameSourceCache` and pass it to
`load_manifest(..., source_cache=cache)`. It owns byte-backed read-only snapshots,
reconstructs independent wrappers and fingerprints per caller, and hashes source
ROOT/log bytes on each request. Misses re-read/revalidate and re-hash after reading;
a changed source fails rather than publishing mismatched provenance. A new cache
is required in a new process; this is not a persistent trusted-file cache.

The driver option `--preparation-repeats 3` measures one cached load and two reuse
assemblies. It checks full flat truth, offsets, detector columns and PDGs by exact
ordered-buffer digests between repeats. `--sampling-reference OLD_OUTPUT` additionally
requires all saved choices, campaign values and source/census hashes to match before
response. Response `--repeats` remains separate. `preparation.passes` excludes exact
regression digest/comparison time, reported as `preparation.exact_check_s`;
`setup_s`/`input_s` include all requested preparation repeats and checks, so they
must not be described as single-assembly latency. The usual default prepares once.

On the full saved 2,741-event development sample, optimized preparation takes
`24.936 s` on its first load, then `11.854/11.851 s` with pool reuse, compared
with the previous `255.963 s` input stage. First-load components are
`14.116 s` read/hash/validation, `0.085 s` sampling, `7.323 s` overlay/validation
and `3.406 s` adapter/validation. On reuse, read/hash/validation is `1.081 s`.
All choices match the saved sampling receipt; full flat-truth/input digests match
between all preparation repeats. Every column in all five persisted detector
tables exactly matches the saved output, including 2,741 event rows and
25,418,385 tracker ADC samples. No baseline response was rerun.

Warm response is `123.343/123.324 s`, effectively unchanged; CRV is
`27.954/28.024 s` and tracker `93.604/93.496 s`. Output writing takes `2.308 s`.
The validation job includes three preparations, `5.470 s` of exact-input checks,
and three response passes; its `56.927 s` setup and 514 s job wall time are not
single-pass latency. The roughly 10x first-load preparation reduction is relative
to the saved run on another node with the same allocation class, not a controlled
same-node end-to-end benchmark. Pool extraction and detector compilation remain
outside warm preparation+response timings. Full host overlay still precedes
response chunking; this optimization does not change that memory boundary.

## Complete native-cache development measurement

On 2,741 events (3,474,473 tracker steps and 1,350,426 CRV steps), the corrected
on-spill path gives CRV digi ratio `1.002646 +/- 0.002168` and pulse ratio
`0.999336 +/- 0.002362` to the saved stock response. Pulse ratios are
`0.999416 +/- 0.002388` in T1 and `0.994632 +/- 0.019411` in T2. Coincidence
counts are 37 versus 38: ratio `0.9737 +/- 0.1299`, too sparse for strong closure.
Diagonal paired chi-square/ndf: digi TDC `19.05/13`, pulse PE `15.59/10`, pulse
time `23.89/14`. Cluster time is `17.70/10`, with only 38 reference clusters.
All repeated CRV arrays match; all tracker detector columns match the saved fast
reference (1,693,464 digis), with no tracker implementation changes.

On an A100, CRV response/reco takes `28.571/28.579 s` warm and tracker digitization
`90.533/90.781 s`; the combined response timer is `121.012/121.270 s`
(`22.65/22.60 events/s`). The first cache-loaded/partly compiling pass is
`186.293 s`, not a cold-cache benchmark. Validation/input/table setup is `33.126 s`
and output writing `2.516 s` separately. CRV device precision is float32; tracker
is float64. Existing pulse full-length retries remain visible (444 waveforms).

The one-time CPU mixing/export costs `219.789 s` wall (`139.242 s` CPU), writes
1.522 GB, and includes a long first-event mixer wait. It is not faster than the
previous smaller export in measured wall time. It can be reused for response-only
studies, but fresh primaries or pileup draws still pay that cost. No integrated
from-art speedup, resident GPU mixer, native CRV truth bridge or fast tracker fit
is established by the response benchmark.

An initial diagnostic exposed a JAX ZS-window bug: post-scan cropping let a rejected
pre-window start consume samples and hide a valid later start. Applying the
inclusive gate inside the scan fixes the 5.2% digi deficit without retuning any
response parameter. The opt-in mode preserves the stock 12-sample advance and
digi tails; all offspill defaults remain unchanged. Invalid step endpoint NaNs
are skipped as stock does (156,798 steps); timing rejection removes another
234,518. These exclusions are reported, not silently repaired or counted as zero.

## Single-pass end-to-end throughput

`fast_detectors --skip-comparisons --repeats 1 --preparation-repeats 1` is the
explicit direct-frame throughput mode. It still hashes and validates all inputs,
loads timing/conditions/assets, samples and overlays all frames, runs both
requested detector consumers, and writes the sampling receipt and closed ROOT
output. Only paired diagnostic comparisons are deferred; multi-repeat preparation,
response, or an inline sampling reference is rejected in this mode. The default
driver behavior is unchanged.

Time a **fresh process through exit**, not the sum of stage timers. The standard-
library-only `python3 -m mixing.process_timer --receipt process.time.json --
python3 -m mixing.fast_detectors ...` includes child imports, GPU initialization,
setup, response, output and shutdown. The measurement launcher also times
container launch through exit with GNU time. It runs one fresh
private JAX/CUDA cache followed by two fresh processes sharing the now-populated
caches, on the same allocated GPU. These are not in-process warm passes; operating-
system/storage caches are uncontrolled. Cache file inventories are saved.

After all timers stop, `python3 -m mixing.throughput ROOTDIR --reference
SAVED_DIRECT_DIR --stock SAVED_STOCK_DIR --out report.json` checks finished
highest-cycle event entries (including empty detector events), full-ID agreement,
all five detector tables and all saved sampling choices against the existing
reference. It reuses the original stock output and launch-to-exit `art.time`,
reporting art's internal `TimeReport` separately. Failed, partial or multi-pass
measurements cannot become throughput points.

Boundary: existing complete source-frame pools and explicit timing/conditions to
tracker digis/ADCs plus CRV digis/pulses/clusters. Source-pool extraction, prior
transport, staging, queue wait and post-run comparisons are separate. This is not
calorimeter simulation, tracker track fitting, trigger processing, or a complete
native art truth/product output. A historical stock UVA measurement is not a grid
or GPVM rate; different hardware, saved timing inputs, mixing implementation and
diagnostic schemas preclude a certified grid-slot equivalence. The earlier cosmic
CRV deck's grid denominator cannot be reused for this pileup workflow.

### Measured single-pass result (6 November 2026)

All three 2,741-event runs completed on one A100-80GB with two host CPUs and
64 GB RAM. Each uses a fresh process, one preparation and one detector pass.
Highest-cycle persisted entries, all sampling choices/source identities and every
column in all five detector tables match the saved validated direct reference.

| Cache state | Process wall (s) | End-to-end events/s | Container + process (s) |
|---|---:|---:|---:|
| Fresh private JAX/CUDA caches | 408.044529 | 6.717404 | 409.36 |
| Populated caches, fresh process 1 | 218.420718 | 12.549176 | 219.20 |
| Populated caches, fresh process 2 | 219.471473 | 12.489095 | 220.29 |
| Historical stock, one UVA CPU | 443.27 | 6.183590 | not measured comparably |

The historical stock process receipt is reused, not rerun; art's internal
436.465571 s is a separate boundary. The roughly 2x populated-cache/stock rate
ratio is descriptive, not a certified speedup or grid-slot equivalence.
The earlier response-only ~22 events/s must not replace these end-to-end rates.

Populated-cache setup takes 27.39/27.38 s (including 24.72/24.70 s input), combined
response 178.25/179.15 s, and ROOT writing 2.32/2.31 s. Fresh-process import/GPU
initialization, remaining compilation, output conversion and shutdown are also
inside the external process timer. The fresh-cache response takes 364.32 s.
No cache thresholds changed: the fresh directory has zero files before the first
run, 38 afterwards, and 38 before/after both populated-cache runs. Storage/OS
caches were not flushed. These measurements do not predict an in-process steady-
state rate or a different sample size.

`mixing.plot_throughput REPORT --out FIGURE --numbers NUMBERS` makes the separate
slide handoff from this validated report, checks the recorded endpoint labels,
and strips private paths/job IDs from the public numbers. The figure records the opt-in chain and timing boundary. Original occupancy
assets stay unchanged.

The user's 100x stock-throughput requirement applies to large production-scale jobs,
not this 2,741-event fresh-process latency test. Neither these process timings nor
the earlier repeated-response measurements establish the large-job limit. A future
measurement needs many realizations streamed through one persistent process with
reused source pools/assets, representative occupancy, bounded memory and actual
output writing. Larger event counts alone do not enlarge the present 32-event
response chunks; batching and sustained throughput must be measured, not extrapolated.

Tracker status checked 6 November: TRK2 is developing field-on reconstruction;
shared digitizer optimization is explicitly deferred. The earlier device-chain
wrapper enables `digi_clusters_jax.MINION_UNROLL=16` and
`SEARCH_METHOD="compare_all"`, whereas this direct caller uses the module defaults
`1` and `"scan"`. These existing settings are a candidate integration test, not a
measured PU1 speedup. Dense on-spill equality/physics and memory checks are needed
before adopting them; no detector kernel or runtime setting was changed here.
