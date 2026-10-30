# Opt-in native CRY primaries

The ordinary chain can generate extracted-configuration primaries in-process,
without a gen-table file. This requires core's `cosmic_cry` module and the
CRY-enabled image/shim described in core's `docs/native-cry.md`. The shared
library is host-CPU code even when the CRV response uses a GPU.

```bash
# Use the CRY-enabled image and bind a current core/crv code snapshot.
python3 -m crvfastsim runchain native --gen native --gen-events 50000 \
  --gen-seed 1234 --seed 5678 --gen-subrun 1 \
  --gdml /inputs/mu2e_extracted_v04.gdml \
  --calib /inputs/crvcalib.csv --deviation /inputs/crvphoton.csv \
  --tier1b strict_v3 --tier1b-dir /inputs/tier1b_strict_v2 \
  --outdir /outputs/native
```

The positional `native` is a placeholder, not an input file. `--gen-events`
counts **accepted events**, including all their primary particles; one invocation
currently permits 1 through 2**20−1 events to fit the chain's packed keys. `--gen-seed`
defaults to `--seed` but uses a separate RNG from transport/response. Subruns must
be in [0,2**28−1] so the event/SiPM packed label fits int64. The native
receipt, measured live time, generation timer and clusters per live second are
in `runchain_stats.json`. All detector conditions, photon tables, geometry and
tier-1b library/sidecars are still required. Native generation removes only the
pre-made **gen** input, not the response/secondary inputs.

Only extracted_v04 is configured initially. Prepared contexts, benchmark
callbacks, prefetch, `--max-events`, and paired `--compare` are rejected for
native mode. Use `--gen-events` to size a native job. `--emit shard.json` retains
the native receipt, event range, generation timer and live exposure.
`runchain-merge` sums independent native exposures and cluster counts; it refuses
mixed table/native shards, incompatible generator configurations, repeated
fresh-source seeds and overlapping packed event ranges. Choose distinct seeds
and subruns for separate processes. This detects common duplication mistakes,
not RNG-substream independence or provenance tampering.

Core's `chain_point` evaluator admits opt-in `gen_set=4`, with positive
`gen_events` per sub-job and `run_config=0`, `files_per_job=1`,
`shard_offset=max_events=0`. Prefetch (`chain_flags & 1`) is refused; fused
response remains opt-in. It uses seed+1000*k and subrun k+1 within each point;
callers must choose nonoverlapping seeds across points. Native generation needs
a CRY-enabled image and matching pinned core/crv commits. Endpoint metrics keep
their existing per-generated-event meaning; verified native exposure, generation
time, receipts and clusters/live-second are additive result metadata.
Admission/rendering tests are not a live endpoint deployment certificate.

The original table route is unchanged: omit `--gen` or pass `--gen table` and
supply gen files normally. No tier-1b or in-bar physics changes accompany this
hook. The stock-style wrapper preserves full correlated raw showers and the
stock over-energy carry behaviour; see core docs for coordinate/time units,
geometry signature, exposure and differences from raw-shower identity.
Independent native/table samples require independent-sample statistics, not
paired-event errors or bit-identical comparisons. The `native_cry_check` analysis
provides a smoke end-to-end comparison with event-clustered count errors
conditional on the measured exposure; it is not a production certificate.

## Device backend

`--gen device` uses core's JAX CRY sampler and Mu2e selection on the GPU, rather
than the C++ sampler. It uses the same positional `native` placeholder,
`--gen-events`, `--gen-seed`, `--gen-subrun` and emitted exposure contract. It
requires a GPU explicitly and fails rather than silently selecting CPU. Add
`--jax` for the existing GPU photon/reco path; generation is independently
on-device even when downstream transport returns to NumPy. Core's
`docs/device-cry.md` defines the precise host/device boundary and pool/RNG limits.
`chain_point gen_set=5` selects it; `gen_set=4` stays the CPU reference. Device and
CPU-native receipts are distinct and cannot be merged as one generator arm.

Cold JAX compilation can dominate small jobs. Report synchronized warm sampling,
selected-buffer transfer/packing, cold initialization and whole-chain wall time
separately; neither a cold small-N run nor a GPU allocation establishes speed-up.
