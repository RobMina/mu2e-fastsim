# Bounded per-batch artifact transactions

`crvfastsim.veto.batch_artifacts.BatchArtifacts` is a standalone persistence
prerequisite, not a batch executor or campaign admission mechanism. It changes no
response, selection, registry, denominator, retained plan, or production defaults.
Use **one new transaction directory and one context-manager invocation per batch**.

## Contract and example

```python
from crvfastsim.veto.batch_artifacts import BatchArtifacts

context = {
    "registry_sha256": saved_full_registry_digest,
    "batch_identity": {
        "plan_sha256": saved_plan_digest,
        "batch": batch_number,
        "frame_first": first_frame_id,
        "frame_stop": stop_frame_id,
    },
    "arms": ["nominal", "comparison"],
    "seeds": [0, 17],
}

with BatchArtifacts(
    "/data/new-batch-0007",
    max_bytes=64 * 1024 * 1024,
    context=context,
    expected_artifacts=["frames.parquet", "diagnostics.json"],
) as artifacts:
    artifacts.write_parquet("frames.parquet", whole_batch_frame, compression="zstd")
    artifacts.write_json("diagnostics.json", batch_diagnostics)
```

`whole_batch_frame` can be a pandas DataFrame or an Arrow Table. Pandas conversion
and Parquet encoding are table operations, not Python per-event loops. Parquet is
encoded directly into the bounded sink, including metadata and its final footer;
there is no whole-Parquet in-memory buffer followed by an after-the-fact size check.
`write_json` streams encoder chunks. For another streaming encoder, use
`with artifacts.open_artifact("output.bin") as stream:` and write bytes to `stream`.
The sink is nonseekable, has no exposed file descriptor, and must not be closed by
the encoder. Encoders must propagate failures; a sink error is latched even when
caught. Do not use one transaction concurrently from multiple threads.

The caller must declare 1–128 distinct expected artifact names up front. Missing,
extra, duplicate, or unsuccessful artifacts prevent commit. Names have one
extension and match `[a-z0-9][a-z0-9_-]{0,95}\.[a-z0-9]{1,16}`. Protocol names,
portable device names, paths, hidden names, uppercase characters, extra dots,
Unicode, and trailing whitespace are rejected. Output files use exclusive creation;
existing directories, files, and directory symlinks are never reopened or replaced.
The parent directory must already exist. The new directory is private (mode 0700)
and files are created mode 0600, subject to a more restrictive process umask.

Provenance is snapshotted at construction as canonical, finite JSON: sorted string
keys, ASCII escaping, compact separators, and a trailing newline. Its SHA256 is
bound into all status records. The top-level fields are exactly `registry_sha256`,
`batch_identity`, `arms`, and `seeds`. The registry digest must be lowercase SHA256;
batch identity must be a nonempty JSON object; arms are distinct nonempty safe
names; seeds are distinct uint64 JSON integers, not floats or booleans. Provenance
is limited to 16 KiB encoded. No implicit default arms, seeds, or batch identity are
provided. Bind the **complete frozen registry** and the frozen plan, not a registry
rebuilt from the selected batch. Digest syntax checks do not establish that the
caller used that registry or that it executed all declared arms/seeds.

## Byte budget, including failures and completion

`max_bytes` is a hard cap on the sum of logical file sizes persisted **through this
transaction**, including initial status, failure status, output artifacts, and both
receipt names. Each proposed `write` is checked **before its first byte**; a write
that does not fit is rejected in its entirety with `ByteLimitExceeded`. Successfully
written prefixes remain diagnostic partial files and are never valid physics output.
OS short writes are accounted as they occur. Encoder errors, disk errors, and cap
errors poison the transaction even if caught inside the surrounding `with` block.

Space for the fixed-size `failed.json` payload is reserved before any artifact
writes. That reservation remains unavailable even on success. Initial status plus
the failure reservation must fit before the directory is created. Completion is
also checked against the budget before staging its receipt; a budget sufficient
for Parquet alone is **not** necessarily sufficient to complete. The receipt size
depends on the declared artifact names and provenance; no automatic budget or
campaign-sizing estimate is supplied.

`persisted_path_bytes` counts bytes actually written plus the second receipt name's
logical size on success. The staged and final receipts are hardlinks to one inode,
but both names are conservatively charged. Directory metadata, allocation blocks,
filesystem journals, compression, copy-on-write costs, and unrelated writers are
not included. This is **not a disk-quota, free-space, peak-RSS, encoder-allocation,
or runtime certificate**. Arrow/pandas tables, JSON chunks, and encoder buffers
can consume memory before a bounded write occurs. External writes bypass this
budget and are not supported; unexpected entries or precommit alterations fail
verification but cannot be retroactively bounded.

## Durable status and atomic completion

1. Exclusively create a new directory, write and fsync `incomplete.json`, then fsync
   the directory and its parent before returning control to the caller.
2. Create each artifact exclusively. On a successful encoder return, fsync its
   bytes and retain its byte count, streaming SHA256, and file identity.
3. On normal context exit, require the complete declared output set. Check directory
   identity and entries; reopen status and artifacts without following symlinks;
   reject changed identities, sizes, hardlinks, or nonregular files; fsync and
   independently rehash the saved bytes. Detect same-size content changes too.
4. Write and fsync `.complete.pending.json`, independently verify it, and fsync the
   directory. The receipt binds provenance, byte cap, initial-status SHA256, and
   artifact byte counts/SHA256 values. It always has `execution_admitted=false`.
5. Publish `complete.json` with an atomic, exclusive hardlink. Publication never
   overwrites an existing name. No fallible data operation follows publication.
   The staging name is intentionally retained: there is no unlink or destructive
   cleanup, even on failure.

Only **`complete.json`** is the completion name; the pending receipt is never
admission evidence, even if its JSON contains `state="complete"`. The original
`incomplete.json` is immutable historical evidence and remains after success. A
consumer must validate the receipt's schema, expected context, output hashes and
sizes, and reject a batch with `failed.json`; mere directory or Parquet existence
is insufficient. This utility does not implement an executor or saved-output
consumer verifier.

On any prepublication exception, persist and fsync `failed.json` and the directory
using the reserved budget. The original exception is propagated. If stream close
also fails, its failure is attached as an exception note where supported, rather
than replacing the active error; this applies to control/status writes as well as
artifacts. A close failure without an active error is propagated and poisons the
transaction. Failure-marker errors (including close-error notes) are attached to
the original exception where supported. A swallowed write, encoder, or close error
instead causes `TransactionFailed` at context exit. All files are
retained, no existing results are modified, and a failed directory cannot be resumed.
Choose a **different new directory** for a deliberate retry. The failure status
records `physics_truncated=false` and `denominator_renormalized=false`: failure is
not permission to use a shortened sample or change its exposure denominator.
Those fields assert the failure policy, not the validity of partial artifacts.

This protocol assumes a local POSIX filesystem supporting directory fsync,
exclusive file creation, `O_NOFOLLOW`, and atomic hardlinks in the same directory.
The caller must own a trusted parent/directory and exclude concurrent mutation.
Precommit checks are not a lock or an adversarial-writer security boundary; files
are not made OS-immutable against their owner after completion. The API itself
never overwrites previously created output.

The final link is deliberately the last fallible publication operation. Its parent
is **not fsynced after publication**: a power failure can lose the final name, in
which case the durable initial status means the batch is incomplete. The outputs
and staged receipt are fsynced before publication; this design does not certify
crash-durable acknowledgement of the final name. Abrupt termination before commit
leaves an incomplete directory, not necessarily a failure record. No software can
guarantee a new durable status when the filesystem itself refuses status writes or
fsync: such errors are preserved where possible, and absent or invalid completion
still means incomplete. Initialization failure may leave a directory without a
valid initial record and must also be treated as incomplete.

## Scope and validation

This utility does not certify whole-frame coverage, arm/seed execution, registry
contents, upstream independence, exposure, waveform partition invariance, or
scientific correctness. Expected artifact names are a caller-supplied output
contract, not a physics completeness proof. It does not allocate photon budgets,
cap cascade attempts, guard waveform cells, run retained plans, or automatically
admit any full campaign. These remain separate prerequisites described in
[veto-retained-plan.md](veto-retained-plan.md).

The focused synthetic tests cover JSON and pandas/Arrow Parquet round-trips,
frozen provenance, strict paths, byte-cap boundaries including the final Parquet
footer and receipt overhead, reserved failure status, caught encoder errors,
partial OS writes, altered outputs before commit, existing directories, no
overwrite, atomic receipt publication, publication/fsync failures, simultaneous
encoder/write and close failures (including control/status cleanup), and transaction
reuse. Run only this module when checking the utility:

```bash
python3 -m pytest -q tests/test_veto_batch_artifacts.py
```
