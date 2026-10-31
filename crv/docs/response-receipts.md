# Prospective response-output receipts

`crvfastsim.response_receipt` binds a **new fasttable output** to input bytes,
producer metadata, implementation-source bytes, event identities, and strict named
conditions. It does **not** certify the stock input's producer configuration,
transport acceptance, end-to-end equivalence, or full named V2. In particular,
matching requested geometry to conditions is not proof that the stock ROOT input
was produced with that geometry.

There is no receipt-upgrade command for historical files. A legacy
`OUT.root.metadata.json` is descriptive metadata, not a receipt, and is rejected
by receipt validation.

## Producer API

```python
import dataclasses
from crvfastsim import response_receipt

# Before configuration loading and before creating the output:
inputs_before = response_receipt.snapshot_inputs({
    "input": input_path,
    "conditions_manifest": conditions_manifest,
    # "event_reference": stock_root,  # for chain/digis, when used
})

# Configure and execute the actual producer, creating a new output ROOT file.
# Use its returned metadata and actual configuration, not the requested analysis
# layout. The CLI integration belongs to valhist.fasttables.
metadata["chain_config"] = dataclasses.asdict(actual_chain_config)
metadata.update(stage=stage, seed=seed, jax=jax, noise=noise,
                window_applied=stage != "reco")
receipt_path = response_receipt.write_receipt(output, metadata, inputs_before)
```

- `snapshot_inputs(mapping)` accepts named paths and returns a dict-like object
  whose records contain absolute resolved `path`, SHA256 `sha256`, and byte
  `size`. `input` is the required producer input name; `event_reference` is the
  optional stock population input. Additional named assets are also checked.
  Keep the **live returned object**. Saving/reloading it as JSON, copying it into
  an ordinary dict, modifying its records, or reusing it for another receipt is
  rejected. It captures implementation-source hashes and its completion time.
- `write_receipt(output, metadata, inputs_before, *, path=None)` returns a `Path`.
  The default is `OUT.root.receipt.json`; an explicit `path` selects a different
  new receipt path. The output must exist and have modification/change times
  after the snapshot. Input/output aliasing is rejected. All recorded input and
  implementation bytes are rechecked around output inspection. Existing receipt
  paths, including dangling symlinks, are rejected. Publication uses exclusive
  creation, not replacement; an I/O failure can leave an incomplete receipt,
  which must not be used. This is not atomic ROOT-plus-sidecars publication.
- Metadata must use `crv-fasttables-named-v1`, contain the embedded
  `crvfastsim.named_conditions.v1` manifest and complete RunConfig identity, and
  record `stage`, `seed`, `jax`, `noise`, finder settings, configured/effective
  windows, timing policy, and `window_applied`. The metadata is preserved as a
  JSON-normalized object, not reconstructed from the requested validation layout.
  If `chain_config` is supplied, every ChainConfig field must be present and its
  window/finder/backend/noise settings must agree with the metadata. The complete
  configuration is retained unchanged. Without it, integrity validation remains
  possible but conditional-response support is explicitly withheld.

The prospective guard prevents accidental post-hoc hashing of an unchanged old
output. It is **not an execution attestation**: deliberate timestamp/evidence
rewriting or a dishonest caller can construct internally consistent evidence.
The producer must actually bracket its invocation with these calls.

## Event and table contract

`event_identity(root_path)` reads `crvdump/events` and returns
`{"encoding": ..., "count": ..., "sha256": ...}`. Run, subrun, and event must be
integer-typed columns in `[0, 2**32 - 1]`; floats, negative/overflow values, empty
universes, and duplicate triples fail. The digest is SHA256 of the lexicographically
sorted `(run, subrun, event)` matrix, each element encoded as little-endian
unsigned 64-bit. It does not pack or truncate keys. Input row ordering therefore
does not affect the digest.

The separate fasttable producer contract is narrower: exactly one run and
`subrun,event < 2**20`, with at most 32,768 channels. Receipt publication and
validation reject violations, including source step/particle keys, rather than
certifying an output after the producer's bit packing has lost information.

Stage checks require:

| Stage | Required output trees |
| --- | --- |
| `digis` | `events`, `digis`; reco trees must be absent |
| `reco`, `chain` | `events`, `digis`, `pulses`, `clusters` |

Every row key must belong to the events tree. Required columns, integer keys and
indices, finite numeric values, bar/SiPM ranges, ADC integer samples, collection
ordering/index coverage, jagged collection references, and per-event tree counts
are checked. Empty data tables are allowed; the event universe must be nonempty.
These are structural checks, not recomputation of response or reconstruction.
All event operations are vectorized; the module contains no physics simulation.

## Consumer API and evidence

```python
evidence = response_receipt.validate_receipt(
    output, receipt_path,
    run_config=rc,             # registry name or RunConfig
    geom=geom,
    conditions_manifest=external_manifest,
)
```

Validation calls **fresh** `named_conditions.load_conditions(external_manifest,
rc, geom)`. It compares the entire loaded manifest exactly with the embedded
manifest, including source-file/event provenance; there is no semantic reduction
or silently accepted substitute bundle. It also checks the complete requested
RunConfig, geometry identity, configured/effective windows, and default named
finder. If full `chain_config` exists, it must exactly match the named default
chain configuration with only the recorded fixed-window/backend/noise choices.
A finder control or other chain override is not admitted as normal response
evidence.

The output's path/hash/size, sorted event digest, stage tables, recorded input
bytes, and implementation-source inventory/hashes must still match. Recorded
paths are absolute; there is **no relocation option**. This includes checking the
current implementation, so a changed producer source requires new production,
not silent validation using new code. Runtime dependencies and external compiled
libraries are not attested.

Population/time evidence is stage-specific:

- **`reco`:** output identities must exactly equal the `input` stock events,
  including events with no digi rows. Input digi keys must belong to that universe.
- **`chain`/`digis` with `event_reference`:** output identities must exactly equal
  that stock universe, including zero-step events. Source step/particle key
  capacities are also checked.
- **`chain`/`digis` without `event_reference`:** population status is
  `unverified_population_no_event_reference`. Neither generated exposure nor
  stock population is inferred from the step input.
- Conditional timing support requires fixed absolute **0–100000 ns offspill**
  metadata and persisted stock event columns satisfying **`spill=0`,
  `evlen=100000`, `pbtime=0` for every event**. Present nonfinite or contradictory
  offspill columns fail. Absent columns produce an explicit unverified status,
  never inferred values.
- Fixed 200–1500 ns **onspill is unsupported**, even if the stock values happen
  to resemble that crop. A claim of `onspill_equivalence: true` is rejected;
  `false` is not itself evidence of equivalence.

A successful integrity check returns compact evidence with
`status: bound_fast_output_and_assets`, stage, run configuration, event digest,
population/window status, `conditional_response_supported`, and limitations.
That conditional flag means only the recorded default fast configuration and
fixed-offspill event/population checks are supported. It is not a physics pass.
Lookup tables, photon maps, and waveform payloads are not automatically discovered
or snapshotted; pass any additional assets explicitly in `snapshot_inputs` to
bind their bytes. Their absence from the automatic binding is reported for step
arms. Those records alone still do not attest which runtime files were opened.

Every successful result retains:

- `stock_source_configuration_not_certified`;
- `no_transport_or_e2e_admission`;
- `not_full_named_v2_certification`;
- `integrity_not_authenticity`;
- `named_v2_pass: false`, `transport_admitted: false`, `e2e_admitted: false`.

The conditions bundle describes its own stock conditions job, not necessarily
this receipt's input campaign. A matching input hash is not stock geometry proof.
Full V2 guardrails remain outside this API; neither receipts nor metric agreement
upgrade historical smoke files to named V2 certification.

## Local tests

```bash
PYTHONDONTWRITEBYTECODE=1 CRVFS_NO_FETCH=1 \
PYTHONPATH=/path/to/core:$PWD \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
~/mu2e/.venv/bin/python3 -m pytest -p no:cacheprovider -q \
  tests/test_response_receipt.py
```

Tests use tiny synthetic ROOT files and a mocked strict conditions loader. They
exercise receipt exclusivity, prospective guards, drift, exact metadata/config
matching, full-width identities and packing rejection, stage schemas, population
checks, missing/nonfinite window evidence, and unsupported onspill. They do not
run a stock job, validate a real conditions campaign, or establish physics closure.
