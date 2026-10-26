# Thread B retained-membership isolation

Two read-only, vectorized diagnostics continue `reco_detail`. Neither changes the frozen physics source, produces
new physics ROOT files, reruns art, or relaxes a coincidence cut. Both validate the completed same-digi attribution
with `reco_detail._validate` before and after reading, including the original frozen sources, all bound inputs,
complete event census, and persisted waveform/pulse identity. A fresh output filename outside the attribution
is required; existing files are never overwritten. New diagnostic and reviewed stock-source hashes are bound too.

## Combination boundary control

```bash
python3 -m valhist.membership_boundary \
  --attribution /data/valhist/thread_b_reco_attribution_20261025 \
  --out /data/valhist/NEW_boundary.json \
  --stock-source /frozen/backing/Offline/GeometryService/src/CosmicRayShieldMaker.cc
```

The driver imports only the separately hash-bound frozen `geometry.py` and `coinc.py` using explicit file paths.
It enumerates same-side, distinct-layer combinations among **retained stock cluster members**, using flat joins
and array cuts. The fixed loops are over shards, sectors, layer choices and the four controls, not events/pulses.
A 2M intermediate-combination guard refuses excessive expansion. All extracted sectors remain included.

Controls use fast versus source-derived stock arithmetic grouping for bar centers, crossed with A1 versus hybrid
persisted double pulse times. No float32 time cast or numeric tolerance is used. Per-edge support records whether
any passing combination contains the link. Maximum margins are diagnostic only: their separate maxima can come
from different combinations, and `joint_margin` mixes ns and dimensionless quantities, so only its sign is useful.
Nonfinite margin counts denote edges with **no available required-layer combination**, not malformed pulse data.

This is not a full finder replay: it does not redo the initial PE filter, greedy clustering or opposite-side
attachment. A stock member can legitimately have no own-side coincidence support if stock attaches it later.
The historical counters-hash limitation is retained, not retroactively repaired.

The executed controls (`thread_b_membership_boundary_v2_20261025.json`) find identical bar centers for both
arithmetic orders, no changed decisions when substituting A1/hybrid times, and no native combination support for
all 1,254 removed edges. Every retained hybrid edge has such support. This negative boundary result led to
review of the actual frozen stock backing, not the older local Offline checkout.

## Exact opposite-side attachment control

```bash
python3 -m valhist.opposite_side_audit \
  --attribution /data/valhist/thread_b_reco_attribution_20261025 \
  --out /data/valhist/NEW_opposite_side.json \
  --stock-source /frozen/backing/Offline/CRVReco/src/CrvCoincidenceFinder_module.cc \
  --stock-fcl /frozen/backing/Offline/CRVReco/fcl/prolog_v12.fcl
```

Frozen stock `51a5125d_19cf240c_2241f345` contains `addOppositeSidePulses`, required by its config and set true in
`prolog_v12.fcl`; the frozen fast finder predates that stage. The stock code applies it after both-side coincidence
clustering, before cluster properties. It selects unclustered first pulses on missing counter sides, matched to a
first pulse at the opposite end of the **same counter**, with a strict time window. It chooses the nearest cluster,
then nearest pulse per `(cluster, counter, SiPM)`, retaining collection-order/cluster-order ties.

The diagnostic implements that stage with vectorized joins and stable selections, starting from all retained
hybrid cluster seeds and the full retained pulse table. The PE filter includes channel-status compensation,
same/adjacent-counter sums and the inclusive adjacent time threshold. Pulses already in any cluster are excluded;
a side already represented in a cluster is never augmented. The frozen calibration and geometry are explicit.

It runs two controls: A1 persisted pulses and hybrid persisted pulses. On shards 11–12 **both recover exactly
609 + 645 missing links**, no extra links, zero residual edges after union with the hybrid membership. Breakdown:
EX 25 + 26, T2 584 + 619; T1/taggers zero. Cluster counts remain the original 2,868 + 3,007 and generated exposure
remains 50,000 + 50,000. This isolates the membership mechanism on this retained sample. It is not full
production equivalence, a new acceptance test, or a response-model correction. Observed `poscalc` transitions
are reported, not recomputed with stock's float side-PE accumulation. A production port still needs independent
full-finder CPU/JAX/fallback regression and downstream revalidation with new, separately labeled outputs.

**Continuation:** the [production candidate port and full-finder replay](thread-b-membership-port.md) now pass
membership closure and `poscalc` on both retained shards. That new result is separate from this attachment-only
audit; residual numerical properties, GPU validation and downstream DQM remain open.

## Tests

```bash
PYTHONPATH=.:../core PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/test_valhist_membership_boundary.py tests/test_valhist_opposite_side_audit.py \
  tests/test_valhist_reco_detail.py
```

Synthetic tests exercise exact slope/time boundaries, arithmetic grouping, same-side exclusion, already-clustered
pulse exclusion, nearest/tie behavior, full event keys, adjacent filtering and immutable outputs. No remote command,
build, art run or real sample is executed by the tests.
