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

## Next gates

1. Isolate the remaining X/average-time differences with a separately labeled stock property-arithmetic control,
   including sequential float side-PE accumulation and cluster traversal order. Preserve this membership-only
   result as the baseline; do not replace it or tune tolerances to absorb the discrepancy.
2. Validate the candidate on an actual GPU, then produce new full same-digi and downstream A1/E2E DQM outputs
   under distinct tags. No twenty-shard response replay or refreshed campaign acceptance is claimed here.
3. Independently repair complete donor/GenParticle collection identity before rebuilding a separately versioned
   library. All nine denominator-support holes were populated by misassociated donor energies; this coincidence
   repair does not unblock generator-denominator uncertainty propagation.

No source build, old-manifest rewrite or denominator repair was performed. The candidate is published only on
its dedicated branch; publication does not update shared development checkouts or declare the remaining gates closed.
