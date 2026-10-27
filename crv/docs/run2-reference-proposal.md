# Larger Run2 reference: proposal only

No submission or staging is performed by this proposal. Reuse the completed 1M generated-event reference and
current straight 20 m material-margin baseline. The experimental scatter-in path has failed 50k probes at two
step sizes and has not passed convergence; a larger reference must not be used to hide that model uncertainty.

## Statistical sizing

The completed 1M sample's event-paired statistical SEs are about 2.19 percentage points on C crossing rate,
1.20 points on C length mean, and 0.52 points on U length mean. Under independent same-distribution sampling,
SE scales as the inverse square root of the generated count (a statistical design approximation, **not runtime**):

| Total generated | C rate SE (%) | C length SE (%) | U length SE (%) |
|---:|---:|---:|---:|
| 1M retained | 2.19 | 1.20 | 0.52 |
| 10M | 0.69 | 0.38 | 0.16 |
| 20M | 0.49 | 0.27 | 0.12 |

**Recommendation: target 20M total (19M additional)** for grouped C rate SE about half of the fixed 1% tolerance.
This does not guarantee a pass, certify sparse individual sectors such as C2, or control systematic uncertainty.
A smaller 10M total tier is useful diagnostically but leaves the C rate SE at about 0.69%.

## Bounded jobs and resources

Reuse the existing **50k generated/job** subdivision: 380 additional jobs, subruns 121–500, base seed 20265100
(worker seeds 20265220–20265599). The completed Run2 subrun-101 job measured **1:08:52 wall**, 4084.82 CPU seconds,
peak resident 2,033,344 kB. Do not infer a campaign completion time from it. The established per-job request is
one CPU, 3000 MB memory, 4000 MB disk, 4h expected lifetime, CVMFS SimJob MDC2025ay, no tarball.
This exceeds the preferred 30-minute target; retain the measured successful chunk rather than infer a new runtime.
A fresh 20k benchmark could commission shorter chunks, but would change the job count and seed/subrun proposal.

Stage approval: one job at 121, then two jobs 122–123, then 377 jobs 124–500 only after persisted generator/accepted
count certification, successful copy-out and queue/output dedup. Do not reuse an attempt directory after an ambiguous
submission. Keep source hashes and staged FCL identical to the retained reference. Current worker seeds must be
checked against all prior campaigns, not just filenames. Reuse existing reference outputs; never overwrite them.

The established submit-script invocation for the first test, **only after explicit authorization**, is:

```bash
bash "$CRVFS_CODE/grid/submit_ref_sim.sh" --test --config run2 --nevts 50000 \
  --first-subrun 121 --baseseed 20265100 --tag v1 \
  --outdir "$CRVFS_PNFS/g4ref_run2_ext_r1_v1"
```

This submitter stages FCL and submits immediately; it is **not a dry run**. Read its files, verify hashes and obtain
approval first. Logical write scope is `/mu2e/scratch/users/ram2aq/crv_fastsim/g4ref_run2_ext_r1_v1`.
The first-two-pilot/remainder invocations need separately audited disjoint ranges, not blind reruns.

## Storage and admission

Keep art outputs on pnfs scratch, read directly for dumping, and process the table stream in bounded batches.
The retained Run2 50k dump directory occupies about 32 MiB locally, so 19M more tables would be of order 12 GiB
if the output distribution and compression remain comparable. This is a storage planning estimate, not measured
new output. Art sizes and worker disk margin require fresh pilot measurement. Reserve at least 16 GiB for derived
tables plus a separately sized allowance for stock response/DQM if those are requested. The data quota was about
133.6 GiB used of 150 GiB during this study: do not launch a full copy-in under that headroom, and never delete
other outputs without user approval. Sequential batch validation or a different approved storage location is required.

Still required before launch: user go-ahead; current queue and supported pnfs inventory; staged input/source hashes;
source/extraction generation counts; token write scope; pilot output/disk/memory checks; explicit final retention plan.
