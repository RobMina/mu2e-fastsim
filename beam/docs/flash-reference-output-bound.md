# Saved-output bound for the inclusive reference

This design is not an enforced campaign gate yet. The budget writer and payload
sandbox are separate opt-in components, and neither certifies an existing runner.
No transport may use them until the exact compute-node combination passes the
failure probes and an independently reviewed end-to-end integration.

## Pause checkpoint — 2026-11-05

All enforcement/certification flags remain false; no owned transport or pilot-slot
use has occurred. Pidfds are ruled out, not a pending deployment option. The
[separate owned-child prototype](flash-reference-owned-children.md) has 114 portable
mocked tests and no host launch. Its outer watchdog and invocation-bound receipt
are unimplemented. The pidfd experiment and observer below are historical
rejected-route evidence, not launch instructions. Quota/scheduler containment is
a separate conditional alternative, not a prerequisite merely to test owned-child
identity. See [the pause/resumption handoff](r5-pause-20261105.md).

## Quantity and trust boundary

The approved limit is 20 GiB of saved output file contents over four non-reusable
slots, at most 5 GiB each. It includes raw art, audit/export ROOT, merged logs,
configuration, manifests and failed partial publications. This is not a physical
block quota: filesystem metadata, scheduler-owned spool/accounting, existing
read-only inputs and node-local transient CVMFS cache are different quantities.
Do not silently describe a saved-content limit as a cap on all allocated storage.
A stronger simultaneous temporary-plus-published bound needs a separate bounded
filesystem and an overlap budget; cross-filesystem move is still copy-then-remove.

The supervisor, fixed configuration and runtime binaries are trusted but may fail,
crash or write excessive output. Protection against a malicious same-UID host
process, kernel vulnerabilities, arbitrary network services or readable host
secrets is not claimed. An attestation flag is never a substitute for observed
confinement and descendant termination.

## Four permanent reservations

`flash_reference_budget.SerializedBudget` exclusively creates one of exactly four
fixed slot directories. Neither a failed reservation nor a failed start is reused.
The same object serializes every saved write, charges actual partial-write bytes,
reserves space for a final status, and accepts only fixed output filenames. The
5 GiB maximum cannot be increased through its constructor. Merged log input is
continuously drained with at most 1 MiB retained. Publishing rejects symlinks,
hardlinks, nonregular files, traversing paths and existing destinations. Incomplete
bounded prefixes remain charged and visible; no automatic retry or deletion exists.

Only a trusted supervisor may hold writable saved-directory descriptors. The
payload receives pipes, not writable saved-output descriptors. Slurm stdout and
stderr must both be `/dev/null`, with setup and application messages drained into
the counted stream. No uncounted wrapper log, timing file, copied snapshot or
second export directory may be created later. Each additional retained artifact
must use the same remaining reservation, including post-run analysis products.

This primitive does not launch, kill or sandbox anything. Its caller must prove
that all payload descendants have terminated before opening outputs to publish.
A post-run census is diagnostic, not enforcement.

`flash_reference_guarded_run.execute_and_publish` is a separate trusted supervisor
primitive: it drains merged output with a parent deadline, reaps or escalates the
wrapper on failure, and publishes only fixed output names after explicit external
invocation-bound verification. The external receipt/callback are responsibilities,
not automatically checked security evidence. Its report always keeps
`fully_enforced=False`; an unrestricted command can still write outside its budget.
No transport CLI or automatic retry is supplied. After the external callback
returns exactly True, the supervisor now requires `flash_reference_inventory`
before opening any publication source: exactly the three fixed files and optional
empty tmp/home/shm directories. Additional numbered RootOutput files, unique
ROOT temporaries, hidden objects, subdirectories, SQLite residuals, symlinks and
hardlinks refuse the entire publication. This is intentionally conservative;
zero-byte fixed files can pass structure but never establish valid art/ROOT.
The inventory is immutable measured metadata with `verified=False`, not another
termination/confinement assertion or replacement for publication-time checks.
End-to-end review must still bind the supervisor to a verified boundary,
invocation-bound descendant lifecycle, complete art validation and disabled outer
logs. The separate nested-directory synthetic fixture diagnostic is not a
production-compatible flat publication layout.

## Current single-container candidate

`flash_reference_apptainer` replaces nested namespace experiments with ONE
Apptainer invocation. Its plan remains `verified=False`. It disables configured
binds, hostfs, CWD, sys and devpts mounts, uses the pinned read-only worker image,
and remaps only declared read-only code, satellite area, exact input and CVMFS.
The only writable bind sources are fresh private node-local work and its aliases.
Runtime/session/cache directories are separate and are never bound into work.
The launcher environment is constructed from an allowlist; `--cleanenv` alone
would not stop launcher-side `APPTAINER_*`/`SINGULARITY_*` injection. The trusted
caller must pass both `plan.argv` and `plan.environment`, with cwd `/`, stdin
`DEVNULL`, merged pipe output and closed inherited descriptors.

The fixed internal handoff applies the existing no-new-privileges/capability and
seccomp policy directly after container setup, closes all descriptors above 2,
and unblocks inherited SIGINT/SIGTERM before setup/application exec. These source
properties do not establish actual mount topology, helper-process isolation,
termination or publication safety. The child probe records the PID namespace,
other visible processes and their proc access, mounts/propagation, descriptors,
write refusals, minimal devices, ordinary threads and selected denied syscalls.
A successful diagnostic is still not an aggregate-output certificate. The existing
guarded supervisor accepts explicit copied `launch_environment` and canonical
`launch_cwd`; the caller must pass the plan environment and cwd `/`, rather than
its inherited-default mode. A separate externally established descendant-lifecycle
gate remains necessary. Apptainer refuses a container HOME environment override;
the trusted inner handoff validates the private home directory and sets the fixed
HOME before importing security code or executing setup/application commands.

Apptainer 1.4 documentation describes `bind-paths`, `hostfs` and system mount
disabling; the exact v1.4.5 `Launcher.setNoMountFlags` implementation additionally
confirms `cwd`, `sys` and `devpts`. The installed site configuration is hash-pinned;
the builder returns the complete bytes with only `mount slave = yes` changed to
`no`, plus the fixed private `--config` path. The trusted caller exclusively creates
that file, chmods it 0400 and passes the exact returned environment. The actual
non-setuid installation must permit this global option; `--userns` alone does not
make a setuid-built installation accept custom config. There is no config fallback,
build-config workaround or privilege elevation. Actual final mounts must reject
propagation links; the configuration flag itself is not evidence. Resolve image
aliases (for example `/var/run` to `/run`) and check actual read-only mounts and
all their descendant mounts, rather than accepting a missing literal mountpoint.

## Rejected nested candidate (retained for evidence)

Two pinned compute attempts failed before payload at nested bubblewrap root or
`/usr` bind setup. Do not reuse login-only success as compute evidence, retry
blind nested variants, or fall back to the unrestricted runner.

`flash_reference_sandbox` builds a fixed bubblewrap plan after the trusted outer
runtime has mounted CVMFS and configured the declared Offline release. It builds
an empty temporary root with a fixed set of read-only runtime prefixes, not a
recursive host-root bind. A fresh node-local work tree is writable through
its work/temp/home/shared-memory aliases. It masks host runtime/proc/device paths,
exposes no FUSE device or writable CVMFS cache, drops capabilities, requires
no-new-privileges and installs a fail-closed libseccomp policy before exec.
Namespace/mount/ptrace operations are denied, while ordinary thread creation must
remain usable. Individual file-size and core limits are additional controls, not
an aggregate bound. Network access remains available for the declared conditions
service, and is not a general hostile-code sandbox claim.

The old Offline container runner creates broad writable binds and must not run
inside this boundary or be mistaken for it. CVMFS cache setup remains outside;
application output must remain inside private work until bounded publication.
Plans/results deliberately remain `verified=False` until actual probes support
an explicitly narrower verified configuration. Never fall back to an unrestricted
runner when a feature, capability check or seccomp load fails.

## Required probes

On an allocated compute node with the pinned image, plugin and runtime hashes:

- verify actual mount flags, private PID/proc view, capability state, seccomp and
  ordinary threads; reject writes to persistent inputs/code/home/output/cache;
- test absolute/relative escapes, replacement/rename, inherited descriptors,
  additional numbered RootOutput files, device/FUSE and runtime socket access;
- test oversized files, many files, log floods, syscall refusal, crashes, orphan
  descendants, termination and broken-output/interrupt cleanup;
- use the real RootOutput lifecycle: it opens a uniquely named temporary file and
  renames on close, so a two-preopened-file assumption is insufficient;
- test publisher overflow/partial writes/unsafe objects and duplicate starts,
  confirming retained content never exceeds the four fixed allowances;
- run the analytic synthetic fixture and saved-art reread inside the final
  combination before any transport, retaining failures and exact receipts.

`flash_reference_lifecycle_payload` is an opt-in finite no-Geant4 test payload,
not a controller or production gate. Its normal/nonzero/crash/linger/closed-stdio
modes exercise a session-detached double-forked orphan in private work. Its small
ready/heartbeat files and emergency 30-second expiry are diagnostic safety nets,
not termination proof; an independently bounded external controller/watchdog and
identity-safe cleanup are mandatory before running parent-death fault injection.
Namespace-local PIDs must never be used as host signal targets. A ready artifact
is a partial observation, not admission. Test primary/init/master/outer-supervisor
death independently and require cleanup before emergency expiry. Unconfirmed
termination retains transient work and refuses publication.

The exact Apptainer PID-1 shim and kernel namespace-init termination can cover
normal primary exit and established master death, but do not alone enforce the
outer supervisor death/startup-race path. Process-group reap or pipe EOF must not
substitute for invocation-bound prepublication termination evidence. Immediate
payload death on supervisor death is not necessary for the logical saved-content
invariant if the sole writer fails stopped, slots stay consumed, all payload
writes remain genuinely node-local transient, and scheduler cleanup has a finite,
verified bound. Death during publication can leave already charged prefixes;
there is no rollback, recovery publisher or uncounted later salvage. This is a
separate bounded-cleanup claim, not an immediate parent-death guarantee. Startup
uncertainty still forbids publication and unsafe work deletion. The exact
scheduler admission/teardown and all payload/helper lifetimes require evidence.
Record the installed CVMFSEXEC mode/control-FD closure and explicitly discard or
account for repository syslogs; do not copy node-local logs/cache/session trees
into saved storage outside the counted writer.

## Historical pidfd ancestry-domain experiment (rejected, not certified)

`flash_reference_subreaper` is a smaller candidate for invocation-bound descendant
observation than scanning host procfs. Use only a dedicated single-threaded host
process with no unrelated children, competing reapers, signal-fork handlers or
external brokered launches. Set/check PR_SET_CHILD_SUBREAPER before the sole fork,
keep the trusted launch stub gated until its pidfd is acquired, and irreversibly
spend the gate. Close all descriptors above stderr in the child before exec.
The controller must own the CVMFSEXEC startup and its helpers, not merely the
inner Apptainer process. No same-UID hostile process protection is claimed.

Cleanup enumerates only direct/adopted children as an aid, binds waitable child
membership without an intervening reap, and signals through pidfds only. Session
detachment, changed groups and intermediate subreapers do not by themselves break
the rooted ancestry invariant. Nested namespace/helper behavior still requires
actual runtime observation. Drain `waitpid(-1, WNOHANG | __WALL)`: zero means
children remain, not success. Only ECHILD after the original root status and a
permanently spent launch gate can support ancestry-domain drainage under these
assumptions. PID reuse must not overwrite the original root status or suppress
signals to a newly adopted process. This does not certify external services,
confinement, valid artifacts, completeness, publication or safe work deletion.

The historical first prepared compute experiment was normal exit only: no parent/controller
kill injection, no Geant4, no pilot slots and no saved publication. The existing
600-second bounded collector observes a dedicated 540-second subreaper setup
child, within a 12-minute one-core/8-GiB standard job. The finite payload creates
an actual setsid/double-fork orphan with a 30-second emergency expiry; completion
must be observed before 20 seconds from its initial heartbeat, not credited to
expiry. Source flags cannot establish that result. Unknown ancestry retains all
private work and refuses post-run record opening. Future controller-death tests
still need a surviving cleanup authority; the existing collector's process-group
fallback is not one and must not be presented as descendant proof. Slurm teardown
or a timeout does not create a production termination certificate.

## Separate pidfd-free alternative (user direction, 2026-11-04)

Pidfds are ruled out for this environment. The replacement design is a separate
[owned-child controller](flash-reference-owned-children.md), not a fallback that
removes checks from the existing pidfd implementation. A dedicated subreaper with
exclusive real-child wait ownership can retain identity until terminal reap:
a live child that exits during the check-to-signal interval keeps its PID as an
unreaped zombie. Fresh waitid WNOWAIT membership immediately before individual
signalling, no competing reapers/ptrace/auto-reap, and explicit terminal-reap
phases replace the handle requirement. No signal is allowed on an old PID,
unknown proc candidate, process group or namespace-local PID.

The four-slot content cap, sole bounded saved writer, actual confinement,
invocation-bound all-descendant termination before publication, independent
watchdog/startup/helper coverage and physics completeness requirements are
unchanged. The new candidate requires its own review and actual host/runtime
probes; a supporting design argument is not admission. It needs no quota claim
merely to test this process-ownership primitive. If the owned-child route cannot establish complete containment, stop for a
separately verified alternative rather than reuse the rejected pidfd route; quota does not waive lifecycle or artifact-validity gates.

## Observed lifecycle blocker (2026-11-04)

Normal-only diagnostic 22482090 closed FAILED 1:0 after two seconds on
udc-aw32-5c1. The reviewed Linux x86_64 pidfd_open syscall returned ENOSYS (38)
after the sole gated fork, before gate release. The missing Python wrapper was
therefore not the only obstacle. This observation does not distinguish a kernel
without the syscall from a policy that returns the same errno.

Read-only Slurm metadata subsequently reports the failed node and all 300
standard-partition nodes on Linux 4.18.0-553.124.1.el8_10.x86_64. Upstream
pidfd_open first appeared in Linux 5.3; vendor backports mean the release string
alone is not definitive capability evidence. No newer supporting standard node
has been established. A separate self-only host diagnostic 22484355 was submitted
on the failed node: one CPU, 1 GiB, two minutes, no requeue, suppressed Slurm logs,
a ten-second alarm and one bounded receipt. It probes uname/self seccomp metadata,
self pidfd_open and pidfd signal zero only; it creates no child or container.
It closed COMPLETED 0:0 in one second on udc-aw32-5c1, with
capability_available=false: self pidfd_open returned ENOSYS (38), Seccomp=0,
NoNewPrivs=0 and no effective capabilities. No child/container/transport was
created. This is a completed negative capability diagnostic, not a successful
lifecycle test. The observed host kernel lacks a usable pidfd_open on this path;
no process seccomp filter was active. Combined with the common older standard
kernel release, missing kernel support is the leading cause, not the earlier
Python-wrapper omission or the unlaunched payload seccomp policy. Vendor/site
confirmation is still needed to name a supporting kernel or permitted host.

The controller now checks self pidfd_open and pidfd signal zero before pipe/fork,
signal-disposition changes or subreaper mutation. Catchable INT/TERM are deferred
through FD acquisition/use/one-shot close. Missing capability therefore refuses
before attempting a root launch; actual gated-root capture remains required on
a supporting host. This is an earlier capability check, not a repaired kernel,
permission bypass or termination certificate. It does not retroactively establish
cleanup of the original failed trial.

The original failed trial report records launch_attempted=true, launch_released=false, root_status=null,
ancestry_drained=false and zero captures/reaps. No CVMFSEXEC, container or test
payload was released. Do not infer descendant termination from gate EOF, wrapper
reaping or a closed Slurm job; retained work remains unknown-lifecycle evidence.
Do not signal the old numeric root PID. All verified/fully_enforced flags remain
false, and this candidate cannot authorize publication or transport. Earlier
boundary and synthetic-fixture successes are not upgraded by this attempt.

There is no unchanged retry or numeric-PID/process-group fallback. A supported
alternative must independently establish confinement, invocation-bound stopped
writers and the required startup/watchdog/helper lifetime controls. A verified
filesystem quota can provide a stronger saved-storage ceiling, but neither free
space nor an unrelated account quota proves the four-slot bound or termination.
No alternative filesystem quota or scheduler cleanup authority is established.

A separate quota/scheduler alternative is conditional, not launch-ready: retain the serialized
writer and permanent reservations, combine a site-supported dedicated quota with
an independently owned scheduler lifecycle domain, and revalidate the pinned
single-container combination. Evidence must identify quota scope, enforcement,
bypass/reset permissions, sparse/compressed-file semantics, temporary/rename
behavior and all helper/output locations. A block quota is not automatically a
logical-content limit. Existing account capacity figures (200 GB home, 10 TB
scratch) do not establish four dedicated 5-GiB enforcement domains. A private
size-limited tmpfs could constrain temporary storage, but shared node /tmp,
RLIMIT_FSIZE, cgroup memory limits and free-space checks cannot replace this
saved-content contract.

The lifecycle domain must admit every runtime/helper before workload release,
prevent membership escape, expose trustworthy invocation-bound empty-domain
evidence, and retain independent cleanup authority across supervisor death or
blocked syscalls. Cgroup configuration and scheduler timeouts alone establish
none of those properties. Primary/init/master/supervisor failures and interrupted
startup need distinct bounded tests before transport. Quota enforcement does not
waive stopped-writer proof, and unknown lifecycle still forbids publication or
work deletion. No site request, quota modification or new diagnostic submission
is implied by this assessment.

## Observation and failure-control continuation (2026-11-04)

`flash_reference_lifecycle_observer` is a historical pidfd-based, unlaunched host
observation primitive retained for evidence, not the missing controller or an admission check. Its caller must
independently bind a host PID to the invocation and retain a trusted live-member
PID-namespace FD. It opens a no-follow proc directory before numeric pidfd_open,
then requires bounded descriptor-relative stat/status reads, stable starttime and
NSpid, repeated matching pinned namespace identities and a non-ready pidfd.
Holding the old proc directory prevents a same-tick PID replacement from silently
becoming the observed process. Captured namespace/proc/pidfds remain open until
explicit close; cleanup attempts every remaining close even when interrupted,
without retrying a possibly recycled descriptor. Poll readiness records only that
captured process's exit; it never proves all descendants stopped or permits
publication/deletion. Caller trust assertions are not evidence. Cooperative
monotonic deadline checks cannot cancel a blocked syscall. Portable mocked tests
are not Linux race, watchdog, cgroup, helper or actual lifecycle validation.

Integrated local publication fault tests compose the real supervisor, inventory
and writer using an explicitly fake security callback in temporary directories.
They cover all four permanent small-budget slots, exact overflow prefixes, actual
short writes followed by EIO, manifest/status failures, strict extra-object
refusals, merged stdout/stderr floods, and post-write file/log/directory fsync
failures. Retained contents are independently summed against charged bytes.
A status can look complete after its fsync fails: the caller's run report still
fails and the slot remains consumed; do not admit on a status file alone. These
tests do not establish crash durability, stable writers, real RootOutput closure,
valid art or any runtime security/termination claim.

Read-only site inspection reports proctrack/cgroup and task/cgroup, KillWait 30s,
UnkillableStepTimeout 300s and no UnkillableStepProgram. These configuration fields
are not observed admission or a guaranteed finite cleanup bound; unkillable
processes remain a distinct failure. Installed CVMFSEXEC 4.54 script SHA256
`c5ecbd4a580a05522013db6a7269ab8bdec92317da58767d5a0b3eba456f5390`
and mountrepo SHA256
`99a8002ea9740a6dabfb67a35688a4d1955442b7e8dc881278dc64c0d470cc57`
support `-N` to close the command FD/unset WAITFIFO before command launch.
Source selects `-pf` only on its USERFUSE path without masked proc mounts; actual
mode, helper identities and teardown still require compute observation. The new
unlaunched diagnostic revision pins both helper scripts, requests `-N`, and
explicitly discards repository USYSLOG to `/dev/null` and disables DEBUGLOG in
both default.local and final per-repository local overrides. It does not copy
these helper logs into saved output. This source revision does not relabel or
upgrade any completed diagnostic, which used the earlier setup; effective final
config, descriptor closure and helper lifetime still need runtime checks.

A successful ordinary namespace `true` probe, a unit suite, or a loadable plugin
alone does not close these gates. If this environment cannot support the complete
boundary, stop for a separately verified alternative rather than weaken the cap.
The owned-child route remains unlaunched and paused; quota alone cannot close its
lifecycle or publication gates.
