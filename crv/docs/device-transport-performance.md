# Device transport validation and timing

## Scope

The opt-in JAX backend reproduces extracted-default straight-line stage A/C
transport. NumPy remains the default. Tier-1b, response and reco algorithms are
unchanged; named-run worlds and optional tail/sliced/below-plane transport are
not certified. See [the contract](device-transport.md).

## Execution changes

The original device implementation spent most captured execution in many tiny
nested sampler kernels. For two warmed 50k-event passes, the CUDA API trace
recorded 770,430 individual kernel launches and 66,502 graph launches. Batching
independent sampler rounds, unrolling the bounded Poisson recurrence and skipping
unused Gamma proposals reduced those to 10,986 and 6,914. Captured API/kernel
times overlap and profiling perturbs latency; these counts are structural
evidence, not an additive percentage breakdown of application wall time.

The implementation retains float64, original material and loss helpers, K=32
Poisson and ten Gamma rounds, per-round random subkeys, material order, slice
and stopping rules. Parallel ion sums change floating-point reduction order.
Material columns use fixed reductions rather than nondeterministic atomic adds.
Bounded top-k selection preserves crossing distance/bar ordering and overflow
rejection. Bar IDs are published as int64, matching the host schema.

A timing-only batch sweep selected 4096: approximately 0.262 s per 50k-event
shard versus 0.290 s at 2048 and 0.47 s at 8192. No physics score was used to
choose it. Changing batch shape changes the independent JAX stream, so physics
was revalidated at the selected shape.

## Validation

- Forty synthetic CPU tests cover ordered ties, overflow, all sampler branches,
  mixed/zero/empty lanes, replay, schema, many-piece reductions and range-out.
- The original 14 fluctuation-law points pass at 4M samples per arm on A100,
  including both original core and optimized sampler versus NumPy, tight
  same-key optimized/core agreement, and the exact dE/dx grid.
- The fixed 100-even-shard held-out bank contains 5M generated events. JAX gives
  1,331,412 steps versus NumPy 1,330,959: +0.0340% ± 0.0466% same-event clustered
  standard error. NumPy and JAX random draws are independent.
- Deterministic track/bar crossings and distances agree exactly. Same-backend
  replay is exact. All 21 aggregate step-shape flags are false, including all
  extracted sector groups and flow bookkeeping.
- Unmodified V1 passes in EX/T1/T2/M both for JAX versus NumPy and for each arm
  versus stock. No cuts, physics formulas, seeds or thresholds were tuned.
- The selected regression suite has 422 passes and one unchanged NumPy
  stopped-momentum residual failure, reproduced separately on pristine main.

A previous 1024 batch had an extra port-V1 energy score 1.501429 against a 1.5
limit. That failed result is retained, not rounded into a pass. Four-slice outer
loop unrolling was rejected because it increased warm time and compilation.

## Throughput boundary

Ten 50k-generated-event shards, one A100-80GB, prepared geometry, five warm JAX
passes and four warmed NumPy passes per shard:

| Quantity | NumPy | JAX |
|---|---:|---:|
| Per-shard median transport wall | 0.677–0.904 s | 0.230–0.302 s |
| Per-shard speedup | — | 2.68–3.14x |
| Transport within a 500k-event chain | 6.750 s | 2.752 s |
| Complete audited chain warm median | 88.221 s | 88.236 s |

Transport includes preparation, transfer and pandas table assembly. The chain
includes input/output, audits, host tier-1b, response and reco. Stage timers overlap
and must not be summed. The independently sampled downstream realization offsets
the transport gain; no full-chain acceleration or whole-node rate is established.
Compilation remains about two minutes and is excluded only from warm figures.

With the existing `--prefetch` enabled for BOTH backends on the same ten shards,
three warm passes give medians 62.930 s NumPy versus 60.194 s JAX (observed 1.045x).
The ranges overlap: 59.817–63.878 s and 57.683–61.299 s respectively. This modest
net result is not a precise or universal chain-rate claim. JAX detailed-output
fingerprints are exactly identical to the serial schedule; cold/warm replay and
final provenance pass. No response, tier-1b or reco algorithm was changed.

The retained chain's cluster rate differs by -0.1476% ± 0.3178% paired, pulse rate
by -0.0180% ± 0.3861%. Cluster/pulse PE and time and cluster y have no flag in the
extra event-clustered diagnostic. Cluster z is all-flow in its diagnostic booking
and remains unsupported, not passed. These diagnostics do not replace full DQM.

## Reproduction tools

Run on allocated compute nodes using the existing project image, not a login
node. Scripts take code/input/output/image paths from environment variables;
inputs are read in place. Producer snapshots must stay immutable until jobs end.

- `valhist.device_transport`: synchronized repeats, exact crossing checks,
  clustered physics, stock V1, repeat fingerprints; rejects existing outputs.
- `valhist.device_sampler_validation`: all 14 full-size loss-law checks.
- `valhist.device_transport_profile`: warm-only CUDA capture and separate
  component benchmarks; `--full-only` avoids extra component compilations.
- `valhist.device_transport_certificate`: strict audit of retained tables,
  generated populations, identities, stopping and repeated timing receipts.
- `valhist.device_transport_bank_report`: combines nonoverlapping fixed-bank
  accumulators; does not infer timing for reused NumPy tables.
- `valhist.device_chain_compare`: paired event-clustered retained-output
  diagnostics, explicitly separate from DQM certification.

Cold setup, compilation and full invocation time must be reported separately
from steady-state speed. A transport-only certificate is not a whole-chain one.
