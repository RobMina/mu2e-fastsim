# Recover a completed named-reference grid submission

`scripts/certify_completed_named_grid.sh OLD_STATE NEW_OUT` certifies the already-submitted
subruns 105–120 for Run1A and Run2 without any submission, resubmission, G4 generation or build.
Use only after inspecting the actual queue and the old attempt ledger. Preserve the original
controller's exit/status logs: a stopped monitor does not mean the grid jobs stopped.

The recovery verifies frozen worker/dumper hashes from `OLD_STATE`, requires an empty user queue,
checks all subrun-101–120 art pairs and worker logs through `ifdh ls`, fetches each remainder
process's logs, and requires its exact worker completion marker. It then dumps only 105–120,
serially, with the existing built art area; known art paths are read directly from pnfs.
At least 2 GiB of quota headroom is required. It runs `reference_certification` on the new dumps
and writes `status.txt`, `driver_exit.txt`, the certificate and the complete log evidence.
No outputs from the retained 101 baseline or 102–104 pilot are rewritten or regenerated.

For the October 26 recovery, the old controller had certified all six pilots and submitted both
16-job remainders, then exited when the first Run2 queue snapshot showed held status. Successful
remainder logs were available the next morning. This script deliberately does not release or
cancel jobs; monitoring policy and grid lifecycle are separate.

## Full-statistics comparison

`scripts/run_named_reference_campaign.sh all NEW_OUT` waits for the specific October 26 completion
certificate, pins the approved core/CRV worktree HEADs and retained audit/path JSON hashes, checks
quota, and invokes the documented `named_reference_campaign` module. It uses an explicit ana
Python executable: an initialized parent shell's setup guard can prevent a child `bash` from
recreating the `pyenv` function. `all` runs both configurations serially on one reserved GPVM;
a single configuration is also supported. No build or grid submission occurs in this launcher.

The driver distinguishes execution completion from physics V1 PASS/FAIL; review its final
`summary.json`, not just the process exit code. See `named_reference_campaign.md` for the full
coverage, provenance, reuse and output contracts. A failed launch before creating any output can
be retried after its cause is fixed and actual process/file state is checked. Preserve its log.
Partial candidate outputs require explicit recovery, not an automatic restart or overwrite.
