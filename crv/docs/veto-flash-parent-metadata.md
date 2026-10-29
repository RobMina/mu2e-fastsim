# Flash parent-metadata discrepancy: exact worker evidence

The exact cataloged worker log for Flash subrun 13147 resolves the discrepancy
reported in `veto-historical-production.md`. No physics file was opened or rerun.

## Evidence

`log.mu2e.NeutralsFlash.MDC2025ac.13147-1759506564.log` has catalog file ID
102817457, size 226,284 bytes and SHA-256
`8795f8d78c74ede772f129c87fab1544aae7ce6a899209b1a9ea07e5932eac00`.
The bytes read at its exact catalog location match that hash.

The log records:

- prodtools **v1.3.7**, global/local job index **13147**;
- generated FCL using NeutralsCat **32644**, not catalog parent **32205**;
- run 1430, subrun 13147, seed 13148, 1,340,000 events, maximum skip 1,335,651;
- SimJob/MDC2025ac, al9-prof-e29-p084, art 3.15.00 and Offline v10_47_00;
- wrapping at event 381176 after 381175 secondary events read;
- art exit zero, 22 Flash and one EarlyFlash output events.

The Flash catalog's 22-event count agrees. Both the output catalog record and the
worker-log catalog record name 32205 as parent. The worker log does not provide a
fresh hash certificate of the retained physics file, so configuration/runtime
observations remain distinct from payload inspection.

## Cause in the named worker software

The inspected v1.3.7 `runjobdef.py` computes parent filenames using
`Mu2eJobIO.job_inputs` from `jobiodetail.py`, then writes them to
`parents_list.txt`. Separately, FCL generation uses `Mu2eJobFCL` from `jobfcl.py`.

Those implementations disagree for `sequential_aux: true`:

- `jobfcl.py` honors sequential selection and chooses index `13147 % 40 = 27`,
  giving **32644**. This matches the generated FCL printed in the worker log.
- `jobiodetail.py` samples an auxiliary filename without checking
  `sequential_aux`. Running only this read-only helper on the pinned small
  job-definition archive gives **32205**, exactly matching the catalog record.

Thus the observed discrepancy has a concrete **parent-list generation defect**,
not evidence of an off-by-one physics assignment. No historical source or catalog
record was modified. A second parent match does not validate the defective helper:
its sampled result can happen to agree with the sequential input.

Source hashes for the inspected worker version:

| File | SHA-256 |
|---|---|
| `runjobdef.py` | `54bc9bcedde63b0bb3330465e3a67fd09f2cdddba841a8a42cbf5df5ab16e817` |
| `jobiodetail.py` | `60d46a46b71f26c63085adddc4bb828c4b9d0807e012f389fc5be1842dacc9e5` |
| `jobfcl.py` | `2cbe4b443b5ca322c0516f1d8053008d6dbd51f0846a019cf3db56902fd79e4f` |
| `job_common.py` | `368e005c47890ac9faa847eb3e70407a7a5da73bb10101c1172f697ff94a2eb9` |

The helper exited zero, with warnings about its package importing the module
before `runpy` execution and unavailable `samweb_client`. It performed no SAM
query, physics read or production execution. The exact output is retained rather
than treating the warnings as a clean test-suite result.

## Consequences

- The previously queried 32205 ancestor metadata describes the **catalog path**,
  not the physical ancestor identified in this worker's generated FCL.
- Do not silently use deep SAM parentage as physical ancestry for this campaign.
  Bind actual worker versions and generated input assignments before deriving
  source-reuse weights or covariance.
- This observation does not invalidate the already measured immediate Cat-to-Flash
  overlap or payload aliases. Those remain separate evidence and are not rerun.
- No catalog repair, upstream PR, new physics sample, exposure renormalization or
  finite-library uncertainty estimate is included here. C4 remains stock.
