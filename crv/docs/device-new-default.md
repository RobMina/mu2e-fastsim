# Opt-in extracted default on the device

The opt-in JAX path implements the extracted model's strict_v4 hard-muon donor
selection and stage-C range_out transport. NumPy remains the backend default.
The reference physics and asset sets are not modified. Historical strict_v3
certificates and timings do not certify this new configuration.

```bash
python3 -m crvfastsim runchain gen.root --gdml world.gdml --calib calib.csv \
  --jax --transport-backend jax --tier1b-backend jax --outdir device_chain
```

## Numerical contract

The sampler uses the same hard catalogue, threshold, pattern keys, sparse-key
rate fallback, whole-history replacement and 50-round complement redraw as NumPy.
Fixed-uniform replay includes the two per-muon uniforms and each active redraw;
integer tuple lower bounds avoid lossy pattern packing. See
[device-tier1b.md](device-tier1b.md).

The transport's immutable CSDA knots come from the reference range table. Delta
counts use a full Poisson draw; the spin-weight rejection sampler has no count
or rejection cap. 32 independent proposals per lane are evaluated at once and
blocks repeat until every electron is accepted; there is no discarded tail.
Each bar uses distinct main/substep/delta keys; absorber range straggling has its
own stream. Near-stop bars use 10 steps at running momentum. Only restricted loss
is quenched into visible energy; the kinetic ledger and exit momentum also
include above-cut deltas. The reference exit-position and generated-beta clock
conventions are retained, including stopped crossings.

Legacy mode retains its existing numeric kernel and RNG. Stage A with range_out
is refused rather than interpreted as stage C. Unsupported world/MSC variants
remain fail-closed. Prepared contexts must agree on range_out as well as geometry.

## Validation status

Experimental, not promoted. CPU tests cover reference CSDA interpolation, delta rates and distributions, fixed-loss
and delta bar-loop agreement with NumPy, substep stop ledgers, padding, repeats, hard-pattern/rate/donor replay, empty
catalogues, sparse-rate boundaries, the exact 50-round redraw residual and the inherited caller contracts. Selected
tests: 484 passed; one known baseline stopped-lane assertion fails (residual 3.568e-6 MeV/c vs 1e-6 tolerance).

Range_out transport, 100 fixed shards (5 M generated events): 1,329,978 NumPy vs 1,330,469 JAX steps,
+0.0369 % ± 0.0465 % (same-event clustered SE). All 21 step shapes are unflagged, deterministic crossings and repeats
are exact, and the unchanged V1 checks pass in EX/T1/T2/M against both the saved NumPy tables and stock. The saved
baseline was reused, so no standalone throughput is inferred from this comparison.

Strict_v4 fixed-uniform replay passes all 100 shards with exact identity and ancestry (maximum field difference
3.64e-12). Independent sampler streams on the same range_out tables: 7,958,555 NumPy vs 7,977,547 JAX steps,
+0.2386 % ± 0.3936 % paired. **The literal all-shape gate fails:** bar, t, t2, lt and posx stay flagged (deck
chi2/ndf 8.846, 41.307, 38.578, 35.479, 14.573; exact-clustered 0.367, 0.448, 0.535, 0.871, 0.526, which does not
replace the deck gate). All sector y/z shapes are unflagged. No seeds, thresholds or assets were tuned. EX
adjacent-layer descendant-energy correlation 0.26394 ± 0.01572 (NumPy) vs 0.27520 ± 0.01598 (JAX); both arms have 38
events with at least 1,000 steps out of 5 M.

On ten 50 k-event shards (an independent-stream warm-up, then 3 repeats) the summed per-shard sampler medians are
25.2508 s NumPy vs 2.5270 s JAX (9.99x), including host ingress and publication; this is not a whole-chain or node
rate. Device CRY generation reaches all four sector types through this path, but new shapes still compile and its
three 10 k-event smoke batches are not a throughput measurement.

## Audited new-default chain timing

One A100-80GB, the same ten 50 k-event shards, a cold pass plus 3 warm passes per arm. Both arms run range_out +
strict_v4 physics and the GPU response and reco; only the transport and tier-1b backends differ. The wall time
includes input I/O, context audits, detailed pulse/cluster output and fingerprints. Repeated fingerprints, prepared
context audits and provenance checks pass.

| Warm medians for 500 k generated events [s] | NumPy transport + tier 1b | JAX transport + tier 1b |
|---|---:|---:|
| Full audited wall | 85.223 | 64.723 |
| Transport | 8.119 | 4.548 |
| Tier 1b | 24.799 | 2.968 |
| Photons | 4.901 | 5.113 |
| SiPM | 9.987 | 9.779 |
| Waveforms | 2.919 | 3.116 |
| Reco | 13.038 | 17.385 |
| Detailed output | 12.891 | 13.178 |
| Context audit | 4.915 | 4.935 |

Warm chain 1.317x faster (5,867 to 7,725 generated events/s), transport 1.785x, tier 1b 8.355x. Warm full-wall ranges
84.789-85.512 s vs 64.637-65.063 s over 3 repeats (not hardware confidence intervals). Nested and asynchronous stage
timers overlap and must not be summed. Preparation (53.191 / 53.200 s) and cold passes (211.496 / 377.656 s) are
separate; the device arm is slower cold. With independent streams downstream, clusters 29,842 to 29,723
(-0.3988 % ± 0.3168 % paired) and pulses 727,020 to 724,721 (-0.3162 % ± 0.3894 % paired); cluster/pulse PE and
time and cluster y are unflagged, cluster z has no in-range support. These are diagnostics, not a response/reco
certification.

This is still a hybrid chain. Crossing metadata, donor picks, CSR expansion, placement and transport numerics run on
the device; showers, near-miss candidate grouping, final ancestry and several chain interfaces use host tables. No
host-only I/O, zero-copy or 8-GPU node claim.
