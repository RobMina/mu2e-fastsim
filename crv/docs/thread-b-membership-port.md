# Thread B: opposite-side attachment port and full-finder replay

Continuation of CRV commit `1041b863a4576d4c40f375a3279f4c2eedb55f4e`, in the isolated branch
`thread-b-membership-20261026`. The shared development checkout and all frozen DQM/uncertainty/core inputs remain
untouched. This is a candidate implementation on a dedicated branch, not a released default or blanket DQM acceptance.

## Mechanism and implementation

The [attachment-only diagnostic](valhist-membership-boundary.md) established that frozen fast reconstruction
omitted stock `addOppositeSidePulses`. The reviewed stock source is
`crvval/backing/Offline/CRVReco/src/CrvCoincidenceFinder_module.cc`, backing
`51a5125d_19cf240c_2241f345`, SHA256
`402cfb29ddeb7ba171b7abb01a3b0ddb1dc5d564b0ff1df16ffc8c733a8923b3`.
Its `prolog_v12.fcl` enables the stage (SHA256
`07df2a94e0774545dad84eff29a655b30249b21adddf5844a7740c3d29081526`).

The production candidate now runs attachment **after final coincidence clustering and before cluster properties
and minimum-cluster-PE rejection**, on CPU and JAX. `FinderConfig.add_opposite_side_pulses` defaults to `True`;
`False` explicitly selects the old membership behavior. The flag is recorded in the response JSON/schema,
forwarded by named run configurations, and carried as runtime JAX parameter data. It is a reconstruction
compatibility control, not a response calibration parameter. Old frozen source checkouts retain their old defaults.

The stage:

- Uses the original PE-filtered hit population, with channel compensation and corrected double pulse times.
- Requires both readout sides in the sector type.
- Uses a snapshot of all native clusters, including ones subsequently rejected by minimum PE.
- Excludes pulses already in any native cluster and counter sides already occupied in the target cluster.
  Secondary members still occupy their sides; candidates and opposite-end anchors must have `seq == 0`.
- Matches the **same physical counter**, opposite side, within the strict window
  `max_dt + 2 * half_length / fiber_signal_speed`.
- First selects the nearest cluster for each candidate, then the nearest candidate for each
  `(cluster, bar, SiPM)`. Exact ties retain stock cluster order and original candidate order. Losing candidates
  are not offered to another cluster; attachments never become new anchors.

CPU retains stock's ordered-map append order. JAX uses the native membership snapshot and quadratic-size array
masks, retaining existing roots. Its tie rank is derived from stock's side / initial-component seed / pulse-index
ordering, **not** its internal minimum packed-root index. Packing includes physical bar, SiPM and first-pulse
identity; promotion, splitting, asynchronous collection and whole-event CPU fallback preserve the configuration.
JAX property reductions still use their existing arithmetic/order; this port does not claim bitwise stock
cluster-property equivalence.

## Full retained-input execution

`valhist/membership_replay.py` runs the **entire candidate coincidence finder**, not merely attachment on retained
seeds. It consumes both retained A1 pulse tables and the independent retained hybrid pulse tables, with attachment
on and off for each. No waveform refit, art job, transport, response, library rebuild or reference replacement occurs.

The driver validates the original completed attribution before and after, binds candidate/helper/core source
inventories, and exclusively publishes a fresh JSON outside code and attribution directories. Actual imports must
come from the candidate CRV checkout and explicit frozen core. Geometry and conditions retain their original bound
paths. The historical separately-unbound counters-file limitation is not retroactively repaired.

Cluster identity is an exact event/sector/sorted-native-member-set identity, not collection order, nearest time,
position or truth matching. Duplicate exact identities withhold numerical pairing. Event/sector/pulse occupancy
addition/removal counts are reported separately from regrouping/unmatched-cluster counts. Full-width event keys
map to dense census rows; empty events remain in the denominator.

Executed on **mu2egpvm01**, one CPU (`taskset -c 0`), existing `crv_fastsim/venv_eaf`, JAX 0.11.2 CPU, NumPy 2.5.2,
pandas 3.0.5. Result:

```text
$CRVFS_DATA/valhist/thread_b_membership_replay_20261026.json
SHA256 f4dee57972a8d3d1a9f5becdf6ee00c7f4e964a8e7f6822489486c2bcb6b311e
```

All **450 input/source bindings** passed before/after. The complete invocation took **177.05 s**, including checks,
reads, compilation and comparisons. The first finder call took 38.36 s; subsequent measured calls took
14.01–17.45 s. These are observed timings on these files, not an extrapolated campaign/GPU performance claim.
No finder call needed overflow fallback.

| Shard | Complete events | Stock / candidate clusters | Added links vs disabled | Extra or residual links vs stock | Exact stock member-set matches |
|---|---:|---:|---:|---:|---:|
| 11 | 50,000 | 2,868 / 2,868 | 609 | 0 | 2,868 |
| 12 | 50,000 | 3,007 / 3,007 | 645 | 0 | 3,007 |

These results hold for **both A1 and hybrid persisted pulse inputs**. Disabled reconstruction reproduces every
retained hybrid member set on both pulse arms. Enabled reconstruction matches all **5,875 stock member sets**.
There are no changed cluster counts. The added 1,254 links comprise **51 EX and 1,203 T2**, with no T1/tagger
additions; 15 EX and 171 T2 events are affected. The evaluation exposure is **100,000 unique events**, not four
times that exposure because there are four controls. All 142,280 pulse identities and waveform links are checked.

The position-method flag (`poscalc`) matches stock for every enabled cluster. This directly checks the property
recomputation that the earlier attachment-only audit withheld.

## Remaining numerical property differences

Matching memberships does **not** imply identical numerical cluster properties. On the A1-pulse control, the
largest absolute enabled-candidate minus retained-stock differences across both shards are:

| Property | Maximum absolute difference | Interpretation |
|---|---:|---|
| PE | 0.000248433 PE | Every cluster lies inside the diagnostic's explicit float32 serialization envelope. |
| Start / end | 0 ns | Exact on A1 persisted pulse times. |
| Slope | 1.18562e-7 | Exact equality is not met; no physics tolerance applied. |
| Average time | 0.01221935 ns | Residual needs stock property-arithmetic investigation. |
| X | 1.4693112 mm | Not dismissed as a generic float-position round-trip effect. |
| Y | 0.000244117 mm | Numerical/dump difference, not an acceptance criterion. |
| Z | 0.000976397 mm | Numerical/dump difference, not an acceptance criterion. |
| `poscalc` | 0 mismatches | Exact on both retained pulse controls. |

With hybrid pulses the largest PE difference is 0.000331879 PE, still inside that envelope; tiny pulse-time
changes remain in start/end. The X/time residuals are essentially unchanged. Source review shows stock
accumulates side PE in `float` while the fast properties use double reductions; with large absolute pulse times,
a small relative denominator difference can affect reconstructed time and longitudinal position. This is a
**specific follow-up hypothesis**, not a completed causal isolation. The port deliberately does not silently
change property arithmetic or cast pulse times to float32.

## Tests and reproduction

Focused local suite: **277 passed** (project Python 3.14, JAX CPU), covering CPU scalar/vectorized combinations,
JAX dense/promotion/fallback/splitting, multiple batches/events, occupied/anchor secondary pulses, no cascading,
strict boundaries, collection-order and stock-cluster-order ties, minimum-PE rescue, runtime configuration,
existing chain integration, immutable diagnostic publication and full-census/set matching.

An initial remote ana run passed 117 tests with the JAX module skipped because ana does not contain JAX. That
skip is not JAX validation. Reusing the existing `venv_eaf` on CPU then passed **133 focused/config tests**.
The final expanded remote suite passed **277 tests in 354.94 s**, matching the local suite with no skipped JAX
tests. Its log is `$CRVFS_DATA/valhist/thread_b_membership_regression_20261026.log`. No dependency was installed.
GPU backend/performance validation is still pending.

```bash
cd /exp/mu2e/app/users/ram2aq/analysis/crv-thread-b-membership-20261026
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../core-thread-b-dqm-20261025 \
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 /exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf/bin/python3 \
  -m valhist.membership_replay \
  --attribution /exp/mu2e/data/users/ram2aq/crv_fastsim/valhist/thread_b_reco_attribution_20261025 \
  --out /exp/mu2e/data/users/ram2aq/crv_fastsim/valhist/NEW_membership_replay.json \
  --backend cpu --core /exp/mu2e/app/users/ram2aq/analysis/core-thread-b-dqm-20261025
```

Use a fresh output; the command refuses existing files. Run only after checking host availability, source sync,
and normal environment setup. The retained result binds the exact candidate source bytes at execution, not just
its base commit. Do not waive any old hash checks after editing a frozen checkout.

## Merge validation on main (2026-10-26/27)

The port applied cleanly to `main` `1041b86` (no conflicts). Full suite with the port: 811 passed / 3 skipped on the
laptop, 814 passed on a GPVM; the one failure in both, `test_oneapi_batch.py::
test_chain_digis_bucketed_default_equals_unbucketed` (a monkeypatched helper without the `width` keyword), fails
identically on `main` without the port.

**V-R3 with attachment on** (`crvfastsim reco-vr3 --both`: numpy and JAX finders on the stock pulses of each dump,
one GPVM core, JAX CPU):

| Sample | Events | Stock clusters / member links | numpy = stock | JAX = stock | JAX vs numpy |
|---|---:|---:|---|---|---|
| Thread B A1 dumps, shards 1-20 | 44,390 | 58,981 / 948,836 | identical | identical | identical |
| G4 reference stock dumps (A0), shards 1-20 | 44,509 | 59,165 / 956,091 | identical | identical | identical |
| new stock dump for the timing (shards 11-12, other seed) | 4,440 | 5,894 / 95,378 | identical | identical | identical |
| control, attachment off, A1 shards 11-12 | 4,404 | 5,875 / 93,972 | 186 clusters, 1,254 links missing | same | identical |

"Identical" is the exact (event, sector type, member set) list in every sector type. No JAX event needed the numpy
fallback. Properties, numpy vs stock over the 118,146 enabled clusters of the first two rows: PE 7e-4, start/end
exact, slope 2.4e-7, average time 0.018 ns, X up to 2.4 mm, Y/Z 1e-3 mm. `poscalc` differs for one cluster (EX): its
two-sided timing position falls 0.2 mm inside the counter end with the fast double-precision side times, and just
outside with stock's float side times, so stock falls back to the PE-weighted centre (X differs by 3,000 mm). The
same cluster has the same membership with attachment off; it is the property-arithmetic item of the next gates,
not an attachment effect. JAX vs numpy: PE exact, average time 4e-11 ns, slope 6e-10, X 5e-9 mm.

**GPU (one A100-SXM4-80GB, JAX 0.11.2):** V-R3 on A1 shards 11-12 passes with the JAX finder on the GPU (5,875
clusters identical to stock and to numpy; properties within 3.4e-10 slope, 3e-11 ns, 3e-9 mm of numpy), the same as
the JAX CPU control in the same job.

**Downstream DQM** (fast reco tables re-made with attachment on and off, injected into the stock Validation and
CRV DQM modules, compared with the deck rules: first-bin cut, k rule for time histograms, exact per-bin
event-clustered variance for position histograms):

| Comparison | flags before | flags with attachment | clusters below the reference's 1 % PEc quantile (420 PE), ref / before / after |
|---|---:|---:|---|
| A3 (fast reco on stock digis) vs A0, shards 11-20 | 0 / 147 | 0 / 133 | 356 / 723 / **356** |
| A4 (fast digis + fast reco on G4 steps) vs A0 | 2 / 147 | **0 / 133** | 356 / 726 / 339 |
| E2E vs A1, shards 1-20 | 8 / 133 | **0 / 133** | 668 / 1,457 / 639 |
| E2E vs A0g | 25 / 133 | 17 / 133 | 643 / 1,457 / 639 |

With attachment on, A3's cluster PE, NPc and Y histograms equal A0's bin for bin. Attachment off reproduces the old
tables exactly. The low-PE excess is gone. The 17 E2E-vs-A0g flags left are digi and pulse occupancy (ADC, BarId,
SiPM) and cluster Y, upstream of the coincidence finder.

**Throughput against post-#2048 stock** (same 4,982 art events; one GPVM core of an AMD EPYC VM; stock: Offline CI
build containing `813ec61e0`, TimeTracker means; fast: `reco-bench --stage reco` on the stock digis/pulses, best of
3 (CPU) or 5 (GPU), compilation excluded; events/s per art event):

| Stage | stock, 1 core | fast numpy, 1 core | fast JAX, 1 core | fast JAX, one A100 |
|---|---:|---:|---:|---:|
| pulses | 2,238 | 521 (0.23x) | 240 (0.11x) | 22,461 (10.0x) |
| coincidences | 24,598 | 271 (0.011x) | 191 (0.008x) | 17,587 (0.71x) |
| pulses + coincidences | 2,051 | 178 (0.087x, serial sum) | 25 (pipelined, 8 chunks); 106 (1 chunk) | 2,623 (1.28x, pipelined, 8 chunks) |

On one CPU core the fast reco is 11-80x slower than the post-#2048 stock modules; the JAX kernels' GPU-sized
padded batches dominate on CPU. On the A100 the pipelined chain on this 5 k-event sample is bound by per-chunk
dispatch (coincidence dispatch 0.86 s, kernels 0.25 s of 1.9 s); the standalone stages add up to about 9,900
events/s (4.8x stock). Earlier speed-ups in `fastreco.md` are against pre-#2048 stock and larger samples.

## Next gates

1. Isolate the remaining X/average-time differences with a separately labeled stock property-arithmetic control,
   including sequential float side-PE accumulation and cluster traversal order. Preserve this membership-only
   result as the baseline; do not replace it or tune tolerances to absorb the discrepancy.
2. ~~Validate the candidate on an actual GPU, then produce new full same-digi and downstream A1/E2E DQM outputs
   under distinct tags.~~ Done (merge validation above). No twenty-shard response replay or refreshed campaign
   acceptance is claimed here.
3. Independently repair complete donor/GenParticle collection identity before rebuilding a separately versioned
   library. All nine denominator-support holes were populated by misassociated donor energies; this coincidence
   repair does not unblock generator-denominator uncertainty propagation.

No source build, old-manifest rewrite or denominator repair was performed. The candidate is published only on
its dedicated branch; publication does not update shared development checkouts or declare the remaining gates closed.
