# Device (JAX) backends of the CRV chain

The response (photons, SiPM charges), waveforms and reco have had JAX twins since the first port (`runchain --jax`).
This page covers the two later device backends for the steps: **transport** (`--transport-backend jax`) and
**tier 1b** (`--tier1b-backend jax`). `runchain` uses both by default when it runs on JAX (`--jax`, backends `auto`)
on the extracted configuration; `--transport-backend numpy --tier1b-backend numpy` keeps the host path, which stays
the reference and the default without `--jax` and for `transport`, `tier1b-apply` and `e2e-bench`. JAX is imported
only when a device backend is used. The transport implements the extracted default model
(`range_out` stage C); the device tier 1b implements both strict sets, the default `strict_v5` (root-coherent draws)
and the previous default `strict_v4` (hard-muon draws). Neither changes the reference physics or the asset sets.

```bash
python3 -m crvfastsim transport gen.root --stage C --gdml world.gdml --transport-backend jax --outdir device_steps
python3 -m crvfastsim runchain gen.root --gdml world.gdml --calib calib.csv --jax --outdir device_chain
```

`e2e-bench` and `tier1b-apply` accept the same options.

## Transport contract

- Stages A and C. Single-scattering tails, sliced MSC and every below-plane world/MSC variant raise instead of
  falling back; stage A with `range_out` is refused. Not validated for the named Run 1A/Run 2 worlds.
- `crossings_jax` clips slabs and keeps the entry-ordered crossings of each track on the device (equal distances keep
  ascending bar index). At most 64 crossings per track; an overflow raises before any table is written. Batches are
  padded to 4096 muons. A host prefilter admits the union of the NumPy near-plane mask and the original-ray detector-box
  hits; it is not an acceptance cut.
- `transport_jax.DeviceTransport` prepares geometry and world constants once, uses core's clipping, slice, tail-loss
  and stopping rules and processes materials in the NumPy world's order (a missing material raises; NumPy has a
  concrete fallback). Above-plane scattering uses the band covariance and geometric-mean momentum.
- `range_out`: the CSDA knots come from the reference range table; delta counts are a full Poisson draw with an
  uncapped spin-weight rejection sampler (32 proposals per lane per block until all are accepted); near-stop bars take
  10 sub-steps at running momentum; only the restricted loss is quenched into visible energy, while the kinetic ledger
  also loses the above-cut deltas. Each bar has its own main/substep/delta keys, absorber straggling its own stream.
- The CRV-local loss sampler evaluates the Poisson/Gamma rounds in parallel (same K=32 cap, ten Gamma rounds and
  fallback as core); parallel summation changes float64 rounding, so agreement with core is numerical, not bitwise.
- Seeds: one draw of the caller's NumPy generator makes the JAX key. Same inputs, key, device, batch size and
  capacities repeat exactly; NumPy and JAX draws are not paired, and a different batch size gives a different stream
  of the same law. The exit-position and generated-beta clock conventions of NumPy are kept.

## Tier-1b contract

- Sets `strict_v5` and `strict_v4` (context, contain, point, topology, unvisited, near-miss, chord, showers, plus the
  root-coherent draws of `strict_v5` or the hard-muon draws of `strict_v4`). Mirror, translate, follow, link and
  stop-product sampling are not ported.
- Preparation builds immutable sorted support and CSR donor histories from the checked library and sidecars; every
  bound comes from the whole library, never from a validation shard. Missing positive-probability support raises;
  there is no history cap and no truncation. Geometry arithmetic is float64.
- Main donors are uniform over the (group, momentum, path bin, context) support, falling back to the bin when it has
  fewer than 20 context donors; whole histories are kept. Unvisited rows are sampled separately on the finest of five
  position levels with at least 20 donors. Near-miss selection excludes whole events with a fast crossing; showers draw
  reach by |PDG| and kinetic energy, then a complete shower.
- Hard draws (v4): each fast muon's ordered group/layer pattern is encoded in two base-17 integers with the momentum
  class of its first crossing; exact lookup gives the catalogue and rate (sparse keys use the class/count rate, more
  than 24 crossings have zero rate); other crossings redraw while their donor is hard, for at most 50 rounds, as the
  NumPy sampler does. Residual hard picks are reported.
- Root-coherent draws (`strict_v5`): each fast crossing draws in its (bin, context, next-crossing symbol) cell, else
  its (bin, context) cell, else its bin, as `tier1b._support_next`; the step trees anchored on the picks are expanded by
  CSR and each row goes to the fast crossing at its library distance along the muon, else the first one with its
  symbol, else stays on the anchor (same layer) or is free (offset from its anchor's library entry); crossings their
  predecessor's pick does not feed keep the rows their pick's trees bring in from behind. The walks are fixed loops
  bounded by the library's largest tree distance and `MAX_WALK`.
- Separate JAX keys give independent draws; a fixed-uniform replay mode drives selection and placement from given
  uniforms so they can be compared exactly with NumPy.

## Validation status

| Check | Sample | Result | Verdict |
|---|---|---|---|
| transport, JAX vs NumPy | 100 held-out shards, 5 M generated | steps 1,330,469 vs 1,329,978, +0.037 +- 0.047 % (same-event clustered); 21 step shapes unflagged; crossings and repeats exact; V1 passes in EX/T1/T2/M against NumPy and stock | pass |
| tier 1b `strict_v4`, fixed-uniform replay, float64 | same 100 shards | every step and ancestry exact (max field difference 3.6e-12) | pass |
| tier 1b `strict_v4`, fixed-uniform replay, float32 | same 100 shards | not exact by construction: entry positions within 1e-3 mm; 10 shards change a discrete outcome | diagnostic |
| tier 1b `strict_v4`, independent streams, float64 | same 100 shards | steps 7,977,547 vs 7,958,555, +0.24 +- 0.39 % paired; 0 of 21 shapes flagged: bar 8.85, t 0.49, t2 0.46, lt 0.42, posx 0.53 (floor 13.87, 0.43, 0.42, 0.22, 0.53) | pass (deck rule with floor) |
| tier 1b `strict_v4`, independent streams, float32 | same 100 shards | steps +0.19 +- 0.40 % paired; 0 of 21 shapes flagged: bar 18.96, t 0.43, t2 0.38, lt 0.30, posx 0.47 (same floor) | pass (deck rule with floor) |
| tier 1b `strict_v5`, fixed-uniform replay, float64 | same 100 shards | every step and ancestry exact (max field difference 3.6e-12) | pass |
| tier 1b `strict_v5`, fixed-uniform replay, float32 | same 100 shards | not exact by construction: 80 shards change a discrete outcome; 978 of 7.97 M steps change their (event, bar) cell | diagnostic |
| tier 1b `strict_v5`, independent streams, float64 | same 100 shards | descendant steps +0.24 +- 0.25 % paired; 0 of 21 shapes flagged: bar 21.3, t 0.32, t2 0.40, lt 0.29, posx 0.41, NStep 0.96 (floor 19.9, 0.43, 0.38, 0.33, 0.59, 0.94); EX adjacent-layer descendant correlation 0.43 +- 0.03 (NumPy 0.42 +- 0.04) | pass (deck rule with floor) |
| tier 1b `strict_v5`, independent streams, float32 | same 100 shards | descendant steps -0.11 +- 0.23 % paired; 0 of 21 flagged: bar 24.9, t 0.42, t2 0.44, lt 0.28, posx 0.64, NStep 1.04 (same floor); correlation 0.41 +- 0.03 | pass (deck rule with floor) |
| downstream, both device backends, `strict_v5`, float32: device steps through the fast response (E2E) and the stock response (A1) | 100 held-out shards, 5 M generated | paired per-shard statistic (family-wise 1 % limits): against the NumPy chain 0 of 133 (E2E) and 0 of 155 (A1) histograms flagged on either half; cluster PE 0.95 / 1.16, NStep 1.24, step x 1.23, NDigis 0.79 / 1.03; above 2000 PE 0.994 +- 0.012 / 0.999 +- 0.012; digis -0.12 +- 0.13 %. Against Geant4 (A1): cluster PE 1.23 (NumPy 1.09), above 2000 PE 0.991 +- 0.015 (0.991 +- 0.014), NStep 2.33 (2.20), NDigis2 1.27 (3.04), NPc 1.84 (2.02); E2E cluster PE 1.05 (NumPy 1.50, a second NumPy tier-1b seed 0.99) | pass |
| downstream, both device backends (`strict_v4`) | 500 k generated | clusters -0.40 +- 0.32 %, pulses -0.32 +- 0.39 % paired; PE, time and cluster y unflagged | diagnostic |

The deck rule (`valhist/deck_rule.py`) is the one of the validation deck: errors scaled by the NStep multiplicity for
t, t2 and lt, exact clustered errors for posx, Poisson errors for the other shapes, and a flag above
max(2, 3 x floor). The floor is the NumPy sampler against itself with a new seed. Earlier versions of this table used
Poisson errors for every shape, which flagged t, t2, lt and posx at 14.6-41.3. bar passes through the floor: it is
above 2, but NumPy against reseeded NumPy (13.87, threshold 41.61) is further apart than the device at float64, and at
float32 the paired chi2/ndf is 0.95 against 0.69 for the floor, so no shape residual is seen.

The NumPy backends stay the reference. With `strict_v5` the device chain is indistinguishable from the NumPy chain
under the deck's paired rule on 5 M events, and against Geant4 its histograms move within the spread of two NumPy
tier-1b seeds.

## Production rates, `strict_v5` (runchain, 11 shards of 50 k generated, warm compilation cache, float32)

Steady = shards 2-11; whole = the 11-shard process including setup (about 35 s on these nodes) and the first shard. Two
jobs per card; generated events per second.

| GPU | NumPy steps (steady / whole) | device tier 1b only | device transport + tier 1b (default with `--jax`) |
|---|---:|---:|---:|
| RTX PRO 6000 (full card) | 16,000-17,200 / 7,200-7,650 | 18,600-19,300 / 7,600-7,700 | 19,700-20,900 / 6,800-7,100 |
| RTX PRO 6000 MIG 1g.24gb | 13,600-14,800 / 6,900-7,500 | 15,400-15,700 / 7,100-7,300 | 16,400-17,200 / 6,600-7,000 |
| A100-SXM4-80GB | 8,600-8,800 / 3,700-3,900 | 8,700-8,900 / 3,600 | 10,100-10,800 / 3,500 |

The device steps gain 17-25 % in steady state (device tier 1b alone 6-15 %) but pay a fixed cost per process: the device transport kernel takes
about 7 s to load from a warm persistent cache (57-60 s to compile without one) and the device tier 1b 2-3 s to
prepare. At 11 shards per process the whole job is 0-10 % slower than with NumPy steps; it gains from about 20-25
shards per process. Without a persistent compilation cache the first shard takes 100-190 s with device steps against
30-55 s with NumPy steps. Host memory: MaxRSS 12-17 GB with the device steps against 9-11 GB.

## Timing (one A100-80GB, warm, `strict_v4`)

Ten 50 k-event shards, cold pass plus three warm passes per arm, both arms with GPU response and reco; only the
transport and tier-1b backends differ. Medians for 500 k generated events, wall seconds including I/O, context audits
and detailed output:

| | NumPy transport + tier 1b | JAX transport + tier 1b |
|---|---:|---:|
| full audited wall | 85.2 | 64.7 |
| transport | 8.1 | 4.5 |
| tier 1b | 24.8 | 3.0 |
| response, waveforms, reco | 30.8 | 35.4 |

Warm chain 1.32x (5,867 -> 7,725 generated events/s). Preparation (53 s) and the cold passes (211 s NumPy, 378 s JAX,
compilation) are separate; short one-shot jobs can be slower on the device. Stage timers overlap and do not add up.

## Boundary

This is a hybrid chain: transport numerics, crossing metadata, donor picks, CSR expansion and placement run on the
device; showers, near-miss grouping, final ancestry and the interfaces between stages use host tables (pandas). It is
not a zero-copy or fully device-resident chain. `python3 -m valhist.device_transport --help` describes the transport
comparison driver (held-out receipts, crossings, clustered step histograms, paired yields, V1, timing).
