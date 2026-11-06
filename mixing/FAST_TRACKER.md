# Experimental fast tracker digitisation

Companion to [native overlay validation](NATIVE.md); opt-in and development-only.


The tracker on-spill/field-on extension adds support to the existing NumPy/JAX
response with explicit MC bunch time, actual event windows, stock ghost charges,
field-dependent drift and a sample-specific detector origin. It is required by
`python3 -m mixing.fast_tracker`; the stock mixing chain does not change defaults.

`four_stream_steps.fcl` extracts all five tracker input collections with existing
dumpers, exposing unsmeared MC timing through the CRV event table. Its reconstructed
CRV timing differs from the normal chain, so use it only for development extraction.
`four_stream_fast_inputs.fcl` instead omits `makeSD`, pins the development job's
recorded module seeds, and preserves normal CRV reconstruction. A second EWM
producer with the same seed exports unsmeared timing without changing the physics
producer's 0.5 ns reconstructed-PBT smearing. Both are campaign-specific configs,
not general-purpose random-seed defaults. `MIXING_EXPORT_CONDITIONS=1` exports the
same campaign's tracker conditions in the response driver.

The comparison driver consumes those tables, the explicit origin and matching
GDML; it checks the saved extraction prefix and unchanged CRV products, preserves
all primary IDs and compares plane occupancy/TDC/TOT/PMP/ADC to the saved stock
tracker. Both backends run in bounded event chunks. It writes native-compatible
`trkdigi/events` detector tables, not full `StrawDigiMC` truth.

The complete 2,741-event development sample contains 3,474,473 mixed steps.
JAX gives 1,693,464 digis versus 1,693,258 stock: ratio `1.000122 +/- 0.000205`,
paired plane chi-square/ndf `29.40/36`. TDC/TOT/PMP/ADC shape chi-square/ndf are
`47.89/44`, `35.71/44`, `18.31/15`, `12.25/15`, `24.44/21`, `9.62/9`.
CRV event, digi, pulse and cluster products are unchanged; the exposed timing
column is explicitly MC truth. All three GPU output arrays repeat identically.
These are development results; held-out mixing results and assets are not rerun.

On one A100, warm tracker calls take `92.26 / 92.11 s` (`29.71 / 29.76 events/s`);
the cache-loaded first call takes `124.58 s`. Setup/input reading is `4.46 s` and
ROOT output `2.23 s` separately. Native mixing+CRV+step export takes `207.27 s`.
Do not quote tracker-only throughput as pipeline speed, or compare this mixed
CPU/GPU pipeline to a stock grid slot without a matched boundary. NumPy is not
faster for this workload (100-event development pilot `30.97 s`).

`import_fast_tracker.fcl` uses the existing `StrawDigisFromTable` plugin and
re-dumps its detector products; `MIXING_DIGI_TABLE` supplies the input table path.
The completed import round trip retains all 2,741 events, 1,693,464 digis and
25,401,960 ADC samples exactly in every field and ordering; import plus re-dump
costs 15.31 s. The imported digis can replace stock tracker detector products,
but consumers requiring `StrawDigiMC` still need a truth bridge. No new C++ build
is needed. The sequential sum of measured stage times, including import, is
353.84 s for the cache-loaded pass and 321.52/321.38 s warm (7.75 and 8.53 events/s),
versus the saved stock 436.47 s. This is a descriptive stage sum using different
CPU/GPU resources, not a controlled speedup or measured integrated wall timer.
Environment startup and queue latency are excluded; split-job turnaround is worse.

### Development profiling

Add `--profile --repeats 5 --fast-reference SAVED_FAST.root` to the same
`mixing.fast_tracker` command, with a new output directory. Keep the development
input, conditions, seed and chunk size unchanged. Pass 0 loads/warms the kernels;
pass 1 writes `host_warm.pstats` and cumulative/self-time text reports; pass 2
reuses `trkfastsim.validation.digi_stage_profile` and writes `stages_warm.json`.
Passes 3 and 4 have neither instrument enabled and are the warm throughput
measurements. Every pass must reproduce all output arrays; the optional saved
fast reference additionally checks every detector column against the prior run.

Synchronized stage timings include kernel-call dispatch and waiting, not just
GPU arithmetic, and child cluster timings are nested inside `make_clusters_total`.
Do not sum those nested rows or label the instrumented rate as production speed.
Host profiles without synchronization can attribute pending device work to
array conversions; they do not measure pure transfer bandwidth. Input reading,
ROOT writing and native preparation/import remain separate timing boundaries.

On the complete development input, the dedicated profiling run reproduces the
saved detector columns and every repeat exactly. Its uninstrumented warm passes
are `91.657/90.456 s` (`29.91/30.30 events/s`); the cache-loaded first pass is
`113.503 s`. In the separate `90.329 s` synchronized diagnostic, cluster production
costs `45.643 s` (including `38.505 s` in sampling), electronics `10.999 s`, and
charge/drift `0.671 s`. Remaining time includes host packing, transfers, association
and output assembly. These measurements prioritize cluster sampling and host/device
boundaries for further investigation; no kernel optimisation is included here.
