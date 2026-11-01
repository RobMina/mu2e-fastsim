# Device (JAX) backends of the CRV chain

The response (photons, SiPM charges), waveforms and reco have had JAX twins since the first port (`runchain --jax`).
This page covers the two later opt-in device backends for the steps: **transport** (`--transport-backend jax`) and
**tier 1b** (`--tier1b-backend jax`). NumPy stays the default for both; JAX is optional and imported only when one
of them is requested. Both implement the extracted default model (`range_out` stage-C transport, tier-1b set
`strict_v4`); neither changes the reference physics or the asset sets.

```bash
python3 -m crvfastsim transport gen.root --stage C --gdml world.gdml --transport-backend jax --outdir device_steps
python3 -m crvfastsim runchain gen.root --gdml world.gdml --calib calib.csv \
  --jax --transport-backend jax --tier1b-backend jax --outdir device_chain
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

- Sets `strict_v3` and `strict_v4` (context, contain, point, topology, unvisited, near-miss, chord, showers; v4 adds
  hard-muon draws). Mirror, translate, follow and stop-product sampling are not ported.
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
- Separate JAX keys give independent draws; a fixed-uniform replay mode drives selection and placement from given
  uniforms so they can be compared exactly with NumPy.

## Validation status

| Check | Sample | Result | Verdict |
|---|---|---|---|
| transport, JAX vs NumPy | 100 held-out shards, 5 M generated | steps 1,330,469 vs 1,329,978, +0.037 +- 0.047 % (same-event clustered); 21 step shapes unflagged; crossings and repeats exact; V1 passes in EX/T1/T2/M against NumPy and stock | pass |
| tier 1b, fixed-uniform replay | same 100 shards | every step and ancestry exact (max field difference 3.6e-12) | pass |
| tier 1b, independent streams | same 100 shards | steps 7,977,547 vs 7,958,555, +0.24 +- 0.39 % paired; sector y/z shapes unflagged; bar, t, t2, lt, posx flagged (deck chi2/ndf 8.8, 41.3, 38.6, 35.5, 14.6; exact-clustered 0.37-0.87) | **fail** (deck rule) |
| downstream, both device backends | 500 k generated | clusters -0.40 +- 0.32 %, pulses -0.32 +- 0.39 % paired; PE, time and cluster y unflagged | diagnostic |

Because the tier-1b deck rule fails, physics results use the NumPy tier 1b. The device tier 1b is a speed option until
that is resolved.

## Timing (one A100-80GB, warm)

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
