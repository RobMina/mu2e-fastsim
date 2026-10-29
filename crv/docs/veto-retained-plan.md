# Retained-sample frame identities and resource admission

`veto.frame_registry` provides a frozen **input-SHA256/run/subrun/event** registry
for diagnostic downstream helpers. The full tuple, not an event number or a packed
legacy identifier, is the frame identity. The registry includes zero-step and
zero-output frames and their saved bunch phase/intensity. It is not an upstream
shower-independence or CE exposure certificate.

Registry rows are canonical-sorted and assigned dense `frame_id` labels below
2^20 only to satisfy the unchanged waveform helper's label domain. Bind and reuse
the **complete registry** for every subset, batch, arm and response replica. Never
rebuild it from a selected batch. A digest of the registry and the explicit uint64
response seed derives the helper RNG seed; changing the registry creates a new
random contract rather than silently reusing event-only streams. Full identities
are restored on every stage output. Kernel-local cluster/waveform ranks remain
local labels, not global truth identities. `qualified_response` requires the saved
registry digest at entry, verifies optical root digests/seed domains, builds all
status-live detector channel conditions (including optically quiet frames), and
restores full tuples on every returned stage. This is not a production consumer.

**Pulse-fit partition invariance is not certified.** Synthetic tests show exact
candidate/charge/ADC/phase results when unrelated frames are removed, but very
small fitted PE differences under changed batch composition. The adapter reports
`pulse_fit_partition_exact_certified=false`; no tolerance is used to convert that
failure into equality. Freeze the batch plan for comparisons until deterministic
fit-padding/bucketing is implemented and independently validated. Frame/seed keys
alone do not guarantee identical floating-point reduction in the fit.

`veto.retained_plan` scans the verified retained raw replay one input at a time,
in bounded ROOT baskets, without generating photons or rerunning stock. It binds
all raw/census/source/conditions hashes and the intended response policy. For each
frame it records actual row counts/ranges/contiguity, finite endpoint counts,
geometric simulated-channel substep counts, and the upper support of live channels
from finite raw steps. The all-live-detector thermal domain is explicit and is not
replaced by the optical channel union. Runtime and output sizes are not extrapolated
from a small-N smoke.

The planner splits canonical consecutive frame groups by vectorized bisection until
**whole-frame** raw-step/substep/live-channel caps hold. It never splits a frame's
SiPM recovery history across batches. A single frame exceeding a cap is a blocker,
not permission to drop it. Input spans and actual source-row contiguity are saved;
a later executor must choose bounded ranged reads or a one-file-at-a-time cache
rather than assume event rows are contiguous. Batches cover every frame exactly
once; changing batch limits changes the plan, not the frozen registry or RNG keys.

These are measured workload descriptors and proposed admission caps, **not a proof
of peak memory or detector output size**. Gaussian optical counts and stochastic
cascades are unbounded laws. The executor still needs per-call photon/attempt caps,
pre-allocation waveform-cell guards, output-byte budget checks and an atomic
completion receipt. A limit raises and leaves the run incomplete; it never truncates
physics output or renormalizes the denominator.

```bash
python3 -m crvfastsim.veto.retained_plan \
  --smoke /data/verified_keyed_smoke --smoke-verified /data/smoke_verified.json \
  --out /data/NEW_RETAINED_PLAN
```

The command requires the previously successful keyed real-asset smoke and its
saved verification as admission evidence, but does not regenerate that smoke.
It freezes an explicit seed schedule in its summary; additional replicas are
response variation on the same registry, not independent exposure. No production
response, CE-v2 consumer, stock baseline or C4 default is changed. A full campaign
is separate from this read-only plan.
