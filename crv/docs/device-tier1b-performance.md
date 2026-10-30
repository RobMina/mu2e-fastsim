# Experimental tier-1b device result

The backend is opt-in (`--tier1b-backend jax` for tier1b-apply, runchain and
e2e-bench); NumPy remains the default. Only extracted strict_v3 is supported.
No changes to the NumPy sampler, asset sets or default physics are included.
The historical results below are pinned to strict_v3 and the earlier straight-line
transport; new-default strict_v4/range_out work is recorded separately in
[device-new-default.md](device-new-default.md).

## Correctness boundary

Fixed-uniform replay on 100 held-out even shards (5 million generated events)
reproduces all published steps and particle ancestry to roundoff: integer fields
exact, maximum floating absolute difference 3.64e-12. Complete donor histories
are expanded without a step cap. CPU synthetic tests also cover missing support,
context fallback, unvisited-level precedence, empty crossings, padding and ray
cells. Library-only float32 chord calculations are precomputed with reference
NumPy intermediate dtypes; runtime geometry remains float64. Simply recomputing
them with fused device arithmetic changed a containment branch by 16.59 mm and
was rejected. Bounds-comparison ordering also matters for diagnostic counts.

Independent device draws on the same fixed bank give 7,956,947 reference steps
and 7,965,391 candidate steps: +0.1061% +/-0.3940% same-event clustered SE.
**The required all-histogram deck certificate fails:** bar, t, t2, lt, lE and
posx remain flagged. For example bar chi2/ndf is 13.415 with Poisson variance,
0.502 with exact clustered variance. Per-sector y/z shapes are unflagged. These
are recorded failures, not a seed/threshold-selection exercise, and the backend
is not promoted or certified. The clustered compatibility is not substituted
for the stated deck gate. Rare/nonmuon donor histories produce substantial
within-event correlation. No default physics was changed to improve this test.

## Measured one-A100 boundary

Ten 50k-event shards, four prepared passes per backend (cold then three warm),
opt-in device transport in **both** arms, independent tier-1b streams. Full
boundary is cached primary tables through detailed pulse/coincidence output.
All reported stage times are wall seconds; async nested timers overlap and must
not be added. Host metadata, CSR expansion and publication are included.

| Warm median over 500k generated events | NumPy tier1b | Device tier1b |
|---|---:|---:|
| Full audited wall | 89.212 | 64.940 |
| Transport | 2.780 | 2.665 |
| Tier1b | 26.723 | 3.853 |
| Photons | 5.432 | 5.233 |
| SiPM | 10.282 | 10.417 |
| Waveforms | 3.250 | 3.174 |
| Reco | 18.159 | 17.756 |
| Detail publication | 13.345 | 12.988 |
| Context audit | 4.851 | 4.855 |

Observed chain improvement 1.374x, tier1b 6.94x. Warm wall ranges
88.674–89.332 and64.741–65.034s respectively (three measurements each), not
precision hardware uncertainty. Context preparation56.60/56.51s is separate;
first-pass compilation is not warm time. Array footprint is about2.94GB;
preparation in the standalone run took7.76s after38.21s library/sidecar load.
Multiple shapes still compile when primary/shower populations change.

Per-arm replay fingerprints, immutable prepared-context audits and final
source/input hashes pass. Clusters29,768 vs29,717: -0.1713% +/-0.2996% paired;
pulses725,714 vs723,120: -0.3574% +/-0.3711%. Cluster/pulse PE/time and cluster-y
are unflagged in the retained-output diagnostic. Cluster-z is all-flow under
that diagnostic's range and unsupported, not a passed shape. This is not full
response/reco deck certification.

## Device metadata and CSR follow-up

An additional full fixed-bank check (100 shards, 5M events) verifies exact muon
ids, ordered visits, context/bin and UV support cells on JAX, then passes the
same tight fixed-uniform frame/ancestry replay (maximum3.64e-12). Complete
main/unvisited histories now expand through immutable CSR and fixed-size device
placement blocks; event/time/PDG/path/energy/relative-identity fields stay on
device until the compatibility API collects them.88 focused CPU tests pass.

On the first ten50k-event shards, one explicit independent-stream warmup per
arm/shard precedes three measured repeats. Summed per-shard medians are
24.9546s NumPy vs2.3895s JAX (observed10.44x standalone sampler); per-shard
median ranges2.4312–2.5433 vs0.2101–0.2621s. This includes host ingress and
publication. Compile-inclusive independent warmups remain separate, about
0.78–1.82s for JAX. It is not a new whole-chain/node rate or a same-job comparison
with the earlier implementation. Summed component medians: main0.6661s,
showers0.6500s, unvisited0.0940s, near-miss0.3184s, final publication0.6232s;
individual medians need not sum exactly. Load37.918s, preparation7.745s,
uploaded arrays3.0475GB. Independent histogram certification was not repeated;
the six failed gates above remain in force.

## Remaining work

This port is **not a fully device-resident chain**. Device CRY generation and the device
transport currently publish host tables; tier1b showers, near-miss candidate
grouping and ancestry/schema publication still use pandas. A follow-up moves
crossing metadata and complete main/unvisited CSR expansion to JAX, exposes
retained device placement buffers, and checks support-defining metadata exactly.
The timing table above describes the earlier validated hybrid boundary, not this
follow-up. A supplementary
three10k-event device-generated chain reached all four sector types, but its
second/third batches still compiled new shapes (11.47/6.83s total) and are not a
steady-state throughput certificate. No 8GPU node result, host-only-I/O claim,
zero-copy handoff, endpoint rollout or named-configuration validation is made.
The reference API must remain available for fixed-donor replay as device-buffer
integration progresses. The failed literal deck gate needs review, not tuning.
