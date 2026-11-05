# Pidfd-free owned-child lifecycle alternative

This is a separate controller design and diagnostic prototype, not a fallback in
the rejected pidfd controller. It does not authorize transport, publication or
work deletion. The user ruled out pidfds; this route needs no newer host kernel,
software build, privileged mount, site quota or partition change merely to test
its ownership primitive. Actual runtime/watchdog/confinement gates remain open.

## Pause checkpoint — 2026-11-05

The separate `flash_reference_owned_children.run_owned_tree` prototype is prepared
and covered by 114 portable mocked tests, including a post-comment-cleanup rerun.
It has not been launched on a host. The outer watchdog and invocation-bound
receipt are unimplemented; an asserted `external_watchdog_present` argument is
not evidence that either exists. Reports retain work and keep `verified=False`
and `certificate=False`. No runtime diagnostic or transport is authorized by
this checkpoint. See [the resumption handoff](r5-pause-20261105.md).

## Retain the output invariant

Keep the existing four permanent reservations, at most 5 GiB of saved contents
each, bounded serialized sole writer, counted logs/config/status/failed prefixes,
and no retry or later recovery publisher. Keep the prescribed single-Apptainer
confinement and genuinely node-local payload/helper work. Neither raw art nor
exports are opened for publication until invocation-bound descendant termination
is established. Inventory, artifact validity and transport completeness remain
separate checks. Replacing process handles changes none of those requirements.

## Identity through exclusive child ownership

A dedicated single-threaded subreaper owns its real direct/adopted process
children. It explicitly sets SIGCHLD to SIG_DFL, without SA_NOCLDWAIT, before its
sole gated fork. It has no other children, competing reapers, ptrace tracees,
callbacks that fork/wait/change dispositions, or external broker launches.
Ordinary threads in the payload remain allowed; they are not arbitrary signal
candidates. The controller itself must not be traced or trace other processes.

For each positive host PID candidate, bounded host-proc metadata first checks
PPid equals this controller, Tgid equals the candidate and TracerPid is zero.
The controller itself requires the trusted host proc/namespace view. A child may
legitimately have multiple NSpid entries in Apptainer's nested PID namespace;
the first must be its host PID, and no trailing local PID is a signal target.
Proc enumeration/metadata is only a candidate/rejection aid. Then, in a serialized
critical section with catchable INT/TERM deferred, the controller calls:

    waitid(P_PID, pid, WEXITED | WNOHANG | WNOWAIT | __WALL)

- A successful no-event result means an eligible child with no reportable exit
  yet. Immediately signal that individual positive PID; do not call wait/reap,
  arbitrary callbacks or disposition-changing handlers between check and signal.
- A terminal event means an exited child retained by WNOWAIT. Do not signal it;
  leave it for the separate explicit terminal-reap phase.
- ECHILD means no signalling authority for that candidate. Skip it, never treat
  that per-PID result as exhaustion of the invocation.
- A mismatched PID, unexpected event, other error or lost ownership fails closed.

If the child exits between the successful membership check and the signal, it
remains an unreaped zombie. Its PID cannot become an unrelated process while the
real parent remains the exclusive reaper. This is the replacement identity
mechanism: a retained child relationship, not an old PID, process-group signal,
proc scan, start-time comparison or an atomic-handle claim. Nothing permits
signalling the numeric PID left over from an earlier diagnostic.

The lease expires at terminal reap. Never cache signalling authority across a
reap. Record the original root's first terminal status separately; clear per-PID
TERM/KILL suppression on every terminal reap so a later adopted generation is
not mistaken for the old one. Unexpected stopped/traced waitpid results are not
terminal completion. Credential changes may still prevent signalling: correct
identity does not prove permission or eventual termination.

## Entire startup and invocation exhaustion

Set/check subreaper status before creating the root. The root is a minimal gated
stub; no CVMFSEXEC, mount helper, Apptainer or application is started until the
parent records its child ownership. The stub closes inherited descriptors above
stderr before exec. The parent irreversibly spends its launch gate before it can
accept emptiness. Releasing a gate can let the child run before the parent closes
its pipe; the proof requires no residual launch authority at final admission,
not an impossible parent-close-before-exec ordering.

After primary/root exit or failure, alternate terminal-reap and fresh owned-child
signal phases, peeling the adopted frontier with bounded TERM/KILL escalation.
Detached sessions, double forks and intermediate subreapers do not alone escape
the ancestry domain. A missing/empty proc list never proves drainage. Only
waitpid(-1, WNOHANG | __WALL) returning ECHILD after the original root's terminal
status and permanently spent gate can establish exhaustion under the assumptions.
A zero return means children remain. Operations, total reaps/signals and enumeration
counts are bounded; cleanup has a cooperative deadline, not a hard syscall-time
bound. Blocked syscalls can exceed that deadline. Exhaustion leaves lifecycle
unknown and work retained.

## Independent watchdog and publisher placement

The production layout must separate failure domains:

    bounded sole publisher -> outer dedicated subreaper/watchdog
                           -> inner dedicated controller
                           -> gated complete CVMFSEXEC/Apptainer ancestry

The outer watchdog establishes its own subreaper state before the inner controller
can launch. The inner must independently set its state; the flag is not inherited
across fork. The outer owns the inner as a real child. If the inner exits or is
terminated, its surviving descendants become the outer's responsibility. The
outer uses the same fresh owned-child protocol and must drain its own domain
before producing a successful lifecycle receipt. Only explicit failure/status
communication descriptors, not saved-output authority, may reach controllers;
the payload stub closes all descriptors above stderr before runtime exec.

An inner report, pipe EOF or a watchdog Boolean is not sufficient. The receipt
must bind the exact invocation/configuration and actual outer-domain exhaustion;
the sole publisher then performs strict inventory, artifact audit and bounded
copying. Publisher death fails stopped and leaves charged prefixes/reservations;
there is no salvage publisher. Controllers hold no usable saved-output authority.

A finite stack does not prove cleanup after arbitrary death of its outermost
member. A stopped or unkillable inner process can also retain children that the
outer cannot yet treat as its own. Slurm cleanup and startup admission remain
external proof obligations, not facts supplied by KillWait configuration. No
finite deadline guarantees termination of uninterruptible kernel I/O. Unknown
cleanup always forbids publication/deletion; no pending SIGKILL counts as exit.

## Staged validation, no transport implied

1. Prepare and independently review the new primitive with portable mocked race,
   stale PID, exit-before-signal, terminal status, clone-child, interrupt and bound
   tests. Leave the existing pidfd implementation unchanged.
2. Separately review a bounded host-only normal/root-exit orphan diagnostic. Test
   the actual Linux 4.18 wait/ownership behavior without containers or publication.
3. Implement/review the outer watchdog and invocation-bound receipt. Test inner
   controller death, interrupted startup, detached orphans and closed stdio with
   an independently surviving cleanup authority and finite emergency payload.
4. Revalidate the exact CVMFSEXEC helper lifetime, single-Apptainer boundary,
   production-compatible flat synthetic fixture and all publication failures.
5. Only after the original output/runtime/completeness gates pass, use the already
   approved first 10 events/design, followed by a separate manual 990-event gate.

No step grants automatic retry, campaign expansion, build, grid or GPU permission.

## Supporting interface evidence

- Linux wait documentation: https://man7.org/linux/man-pages/man2/waitid.2.html
  documents WNOWAIT retention, SIGCHLD/SIG_IGN and SA_NOCLDWAIT distinctions,
  zombies and __WALL support in waitid since Linux 4.7.
- Subreaper semantics: https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html
  documents adoption by the nearest living ancestor subreaper and non-inheritance
  of the setting across fork.
- Upstream Linux v4.18 kernel/exit.c and kernel/signal.c: wait_task_zombie's WNOWAIT
  branch does not perform the terminal reap transition; do_notify_parent's
  autoreap condition depends on SIGCHLD ignore/SA_NOCLDWAIT; reparenting precedes
  parent exit notification. These support the invariant, not vendor-runtime proof:
  https://github.com/torvalds/linux/blob/v4.18/kernel/exit.c
  and https://github.com/torvalds/linux/blob/v4.18/kernel/signal.c
