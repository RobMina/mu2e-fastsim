# BO track-selection prototype and measured-validation gate

R7 is paused by user as of 2026-11-05 06:10 EST; no R7 jobs are running. The
[public pause/resume handoff](PAUSE_RESUME.md) separates saved measured results
from unfinished response-control and independent-validation work.

The first JAX component is a **measured-reconstruction cut-scan prototype**, not an end-to-end trigger emulator.
It targets `tpr_TrkDe_80m70p_D0200` (bit 100) and `tpr_TrkDe_80m70p` (bit 110), sharing reconstructed
`TrigKSFDe` tracks. Production thresholds come from the embedded Digitize configuration, not current menus.
The kernel does not emulate tracker transport, hit reconstruction, pattern recognition or the shared fit.
No production threshold changes are made. In the completed full-history quiet pair, these two filters consume
0.1523% of stock / 0.1597% of hybrid attributed trigger work; the five shared fits consume 71.17% / 70.38%.
The hybrid's 4.307% lower trigger wall attribution accompanies 0.652% longer total art real time, not a
whole-job speedup. Those measurements do not time this JAX kernel.

## Source and configuration binding

The production runtime reports Offline `v13_41_01`. The corresponding public release resolves to commit
`b132ab6567ac021851353e05b7a1feb45bfb7c14`; its
[`KalSeedFilter_module.cc`](https://github.com/Mu2e/Offline/blob/b132ab6567ac021851353e05b7a1feb45bfb7c14/TrkFilters/src/KalSeedFilter_module.cc)
has SHA256 `51b6dbfe6f5dcd115afdfb8b9ada7aa187b8610f717cdbbc1527e7b573fabd39`. Tag and commit URL contents
were compared exactly. This is a release-source binding, **not** binary/source attestation or measured parity.
Keep the original configuration hash with each compiled selection contract. Unsupported configuration options
must fail rather than acquire guessed defaults. Do not substitute current-head source logic.

The released filter requires SeedOK status, a usable t0 segment and a supported fit; optionally checks fitted
particle hypothesis and momentum direction, then applies the configured kinematic/quality/hit cuts. A track
passing either cut entry is counted once. It counts all passing tracks, then applies `minNTrks`. Direction
rejection uses `pz * direction < 0`, not `pz <= 0`. Kinematic bounds are strict; hit minima are inclusive.

## Measured feature boundary

Each event needs explicit presence/reach metadata and zero-track coverage, plus stable collection/track indices.
A padded track table has a validity mask; padding is not a missing collection. Required measured values are:

- SeedOK status, usable t0 segment and supported-fit predicates;
- fitted particle hypothesis (not generator/truth PDG), momentum, pz, tan dip and signed d0;
- momentum error from the t0 segment, chi-squared/NDOF, fit consistency and active straw-hit count.

These are the **actual derived C++ quantities**, not algebraically convenient approximations. For example,
`1/tan(momentum.Theta())` must not silently become `pz/pt`, and LoopHelix signed minimum-axis distance must
not become a different helix's d0. Preserve source precision and arithmetic order when extracting features.
The limited contract excludes nonzero stereo/plane minima, calorimeter-association requirements and momentum
surface overrides; supporting those requires additional measured fields and source-bound extraction. In the
released data product, `chisquared()` returns float and `nDOF()` int: the C++ division evaluates as float before
comparison with the double cut. Export that evaluated expression, not a double-precision recomputation. Also,
`t0Segment()` throws on an empty segment vector: a reached SeedOK track without segments is an error, not a
normal zero-track event or a silently false t0 predicate.
No source-birth momentum, truth-triggerable flag or transport truth is an allowed surrogate.

## Completed measured evidence and remaining gates

The completed tracker-fast hybrid retains `TriggerResults` only in its decision output. Its input artifacts
contain original stock reconstruction, not the new fast replay's `TrigKSFDe`. The stock regeneration output
retains stock KalSeeds, but these cannot stand in for fast reconstructed tracks. A separate full-history
feature run has now completed art and produced a real measured fast table: 28,044 events, 18,038 tracks
and 10,006 zero-track events. Its actual std::vector schema, decision-art IDs/order and released-source
status predicates pass the strict saved-table checks. The full feature postflight completed successfully:
ordered source/preparation and executed-configuration checks, pinned worker receipt, saved-baseline packed
decisions and measured arrays all pass. A separate saved-output check independently reproduces the array
comparison: eager and JIT have zero track-identity/count, reached-filter-return or composed
path-bit mismatches for both selections across the full 28,044 events. D0200 reaches 17,931 events / 17,929
tracks (two empty reached collections); nominal reaches 17,988 / 17,981 (seven empty). All 52 replay and all
56 original packed decisions exactly match the saved hybrid baseline. This is a nonvacuous measured-array
result with observed source/configuration/runtime gates, not loaded-binary attestation, independent feature
extraction, C++ boundary certification, nonoverlapping confirmation, upstream reconstruction emulation or
speedup. The deliberately stronger metadata/parity flags remain false. Boundary/shape tests remain synthetic.

A separately declared **feature-export** run must use the exact admitted fast input population,
production runtime and full traversal. Keep all 52 scheduled paths and their own evolving counters. The R7-owned
[measured-feature analyzer satellite](art/README.md) has now been user-built against the exact released
musing/p109; source/build hashes, library freshness and build-host `ldd -r` verified. Actual art execution
completed with the transferred library and all measured-array checks pass. This is operational evidence,
not loaded-plugin tracing or independent verification of every exported derived value. Stock EventNtuple's
float, intersection-based quantities cannot certify this t0-segment contract.
`feature_replay.py` adds that analyzer only to the existing unconditional DecisionOutput endpath and redirects
the exact production TFileService `/dev/null` placeholder; source-only preflight never authorizes runtime.
The vectorized `jax_feature_validate.py` checks supplied arrays, per-path reach, accepted-track keys, eager/JIT
filter returns and composed bits, but has no artifact provenance admission and explicitly withholds measured
parity. `feature_runtime.py` now captures/rechecks the user-built library and source hashes, ELF/linkage
and explicit backing, freezes one plugin with an adapted private launcher, and checks worker relocations
before art. `feature_table.py` validates the exact ROOT schema/dtypes, highest cycles, IDs/order, empty
coverage, aligned tracks and accepted keys. `feature_validate.py` composes these checks with exact all-52
saved-baseline packed-word equality and eager/JIT array checks. A separately declared full-history
`feature_replay.sbatch` experiment has completed art exit 0; a bounded saved-output measured-array
comparison passes as above, and the original full postflight also completed successfully. The worker passed
its pinned library/relocation checks and the analyzer ran on all 28,044 events. `feature_saved.py` checks copied saved
outputs against the exact completed baseline and never reopens parents or substitutes for full postflight. The local tests, build verification
and pre-execution worker receipt alone do not satisfy measured parity gates. Do not change an immutable
campaign or call feature export a retry of successful art. Reading physics uses measured ntuples plus uproot,
not gallery or ACLiC.

`feature_status.py` supplies a separate optional source-bound consistency check over an admitted table.
With `feature_validate.py --status-source-directory`, exact released TrkFitFlag definitions/name mapping,
BitMap, KalSeed accessors, direction definitions and filter source bytes are rehashed before and after the
comparison. Every track's raw status is checked against exported SeedOK and supported-fit predicates,
including unreached tracks; accepted keys must follow single-collection traversal order. Multiple fit bits
retain the released LoopHelix/CentralHelix/Line precedence rather than being rejected arbitrarily.
This additional check does not change the structural loader's declarations, authenticate the actual loaded
plugins, independently verify t0 kinematics or ProductID provenance, certify C++ numerical boundaries, or
promote the measured-parity flags. A saved table can be rechecked without rerunning successful art.

## Per-path validation, before claiming an emulator

1. Bind feature artifacts to exact runtime, source bytes/order, digi tags, configuration, source implementation
   and exported event/track identities. Preserve explicit empty events; missing/unrun/error cases remain distinct.
2. Establish the exact feature-export run's all-52 decision equality against the saved hybrid on identical
   inputs/order. An output-only transformation is not assumed inert without that check.
3. Compare eager and JIT JAX selections to each reached C++ filter call, including accepted track identities
   and counts. Verify strict thresholds at boundaries with the released implementation, not only a Python oracle.
4. Compose path results with their actual preceding prescale/time-cluster/helix-filter reach. Report each path's
   false positives, false negatives, states and mapped endpoints against **TriggerReplay**, not original Digitize.
   Existing stock-to-fast migrations are not JAX errors. Never copy preceding stock-arm reach or counters.
5. Freeze the candidate and confirm on nonoverlapping independent source history. Common-seed parent resets
   in the development campaign do not supply independent replication or a shard SEM.

## Saved measured CPU batch benchmark

`selection_benchmark.py` first requires the saved measured-array/status/baseline checks, then evaluates both
unchanged selection contracts together on all 28,044 events, padding width one and 18,038 actual tracks.
Actual fast reach composes path bits; it is an input, not an emulated result. All output leaves are synchronized
for every timed call. Thirty warm batch repeats after three warmups on the local arm64 CPU, JAX 0.11.2/x64,
measure **0.120646 ms median per complete two-selection batch** (min 0.107250, max 0.139208, observed batch
p95 0.133913 ms). The amortized median is 0.004302 microseconds per input event, **not a single-event latency**.
This CPU process is not core-pinned and JAX threading is not constrained; it is not a single-core rate.

Costs excluded from warm timing are recorded separately: synchronized device placement 0.849458 ms,
lowering 28.365083 ms, compilation 65.507208 ms, first execution 0.320708 ms. In-memory JAX caches are cleared
after validation and persistent compilation caching is disabled during lowering/compile; backend initialization
remains paid, so this is not cold process startup. Saved validation costs 18.500 s and the additional
load/padding/target check costs 9.302 s; neither belongs to the warm kernel rate. Exact input/source receipts
and output digest/counts are rechecked; composed accepted events remain 17,788 / 17,813.
These are fixed-reconstruction BO selection costs only: no reconstruction, standalone full-path latency,
production-C++ hardware ratio, whole-trigger/full-chain acceleration, GPU rate or SU claim.

## Cost boundary

In the completed selected-stock quiet replay, these two final filters together account for approximately
0.122% of attributed trigger work; all KalSeedFilters approximately 1.46%. This is not fast-arm timing and not
an end-to-end speedup opportunity by itself. The five dominant shared LoopHelix fits account for roughly
three quarters of that replay's trigger work (71.2% in the recovered full-parent production attribution).
BO utility here is repeated selection/cut evaluation over fixed **measured reconstruction**, not predicting
how geometry changes the reconstructed tracks. Cost-dominant pattern-recognition/fit emulation remains open.
Any timing must synchronize JAX device results, separate compilation, and use the same event population and
hardware scope. No speedup, GPU or SU claim is established by this prototype.
