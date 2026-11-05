"""Trusted supervisor primitive, NOT a sandbox, runtime gate, or certificate.

The fixed argv must already be a reviewed namespace wrapper. An unrestricted
command can write persistent files despite close_fds and these output bounds.
The caller owns stable private node-local work and saved root, prevents external
writers, and independently verifies confinement AND descendant termination.
A receipt is an opaque reference to external evidence, never self-certification.
The callback must check that evidence against this exact invocation and confirm
all descendants have stopped after wrapper reaping; a process-group kill alone
cannot establish that (descendants can escape it). This module never claims
fully_enforced, never invokes transport automatically, and never retries a slot.
The wrapper inherits blocked SIGINT/SIGTERM and must safely unblock them before
its workload; Popen restore_signals does not restore the thread signal mask.
Cleanup uses SIGKILL as its fallback; no threaded-unsafe preexec_fn is installed.

All scheduler/outer logs must separately be disabled or routed through the counted
pipe. Saved-content bounds do not bound temporary storage, metadata, caches, or
writes made by an unsandboxed command. No CLI is supplied; explicit launcher
environment and cwd are trusted caller inputs, not evidence of confinement.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from .flash_reference_inventory import (
    OutputLayout as InventoryOutputLayout, validate_inventory,
)
from .flash_reference_budget import (
    CHUNK_BYTES, DEFAULT_STATUS_RESERVE, MAX_LOG_BYTES, SLOT_BUDGET_BYTES,
    SerializedBudget,
)


class OutputLayout(Enum):
    REFERENCE = "reference"
    NOG4_FIXTURE = "nog4_fixture"


# No caller-provided paths, destination names, globbing, or numbered-file search.
_OUTPUTS = {
    OutputLayout.REFERENCE: (("raw_1.art", "output.art"),
                             ("export.root", "output.root"), ("job.fcl", "config.fcl")),
    OutputLayout.NOG4_FIXTURE: (("flash_reference_fixture_1.art", "output.art"),
                               ("export.root", "output.root"), ("job.fcl", "config.fcl")),
}
_STOP_TIMEOUT = 0.5


@dataclass(frozen=True)
class RunReport:
    slot: str
    state: str
    returncode: int | None
    wrapper_reaped: bool
    publication_verified_by_caller: bool
    independently_verified_confinement_receipt: str | None
    published: tuple[str, ...]
    saved_bytes: int
    discarded_log_bytes: int
    failure: str | None = None
    launch_state: str = "not_started"
    fully_enforced: bool = field(default=False, init=False)


class GuardedRunError(RuntimeError):
    """Failed run; retained slot and report must not be retried or removed."""

    def __init__(self, report):
        super().__init__(report.failure)
        self.report = report


def _terminate_and_reap(child):
    """Bounded best effort; even a successful reap is not descendant proof."""
    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            # Still try the tracked child if group signalling was refused.
            try:
                child.send_signal(sig)
            except OSError:
                pass

    try:
        send(signal.SIGTERM)
        try:
            child.wait(timeout=_STOP_TIMEOUT)
        except BaseException:
            pass
    finally:
        # Kill remaining group members even if the direct child already exited.
        send(signal.SIGKILL)
    child.wait(timeout=_STOP_TIMEOUT)


def _drain(child, deadline):
    """Yield bounded chunks without waiting for a newline, buffer fill, or EOF."""
    with selectors.DefaultSelector() as selector:
        os.set_blocking(child.stdout.fileno(), False)
        selector.register(child.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("parent timeout while draining wrapper output")
            if not selector.select(min(remaining, 0.1)):
                continue
            try:
                chunk = os.read(child.stdout.fileno(), CHUNK_BYTES)
            except (BlockingIOError, InterruptedError):
                continue
            if not chunk:
                return
            yield chunk


def execute_and_publish(root, slot, command, *, work, parent_timeout,
                        independently_verified_confinement_receipt=None,
                        verify_callback=None, output_layout=OutputLayout.REFERENCE,
                        byte_limit=SLOT_BUDGET_BYTES, log_limit=MAX_LOG_BYTES,
                        status_reserve=DEFAULT_STATUS_RESERVE,
                        launch_environment=None, launch_cwd=None):
    """Execute trusted fixed argv, reap its wrapper, then publish fixed sources.

    ``verify_callback(receipt, argv, work, returncode) is True`` is required for
    publication, in addition to a nonempty external receipt reference and a zero
    wrapper exit. The callback is trusted caller code, NOT a confinement probe
    performed here. It must independently confirm confinement and termination of
    every descendant. After that confirmation, strict inventory of the matching
    fixed layout must succeed before any output publication; it is a metadata
    measurement, not independent verification or a runtime certificate.
    Without both inputs execution is diagnostic only: no files
    are published and status is failed/unverified. Reports always keep the runtime
    gate false. Callback execution and filesystem operations must themselves be
    trusted/bounded: parent_timeout bounds subprocess supervision, not arbitrary
    Python callbacks or stalled filesystems. Failures retain partial files and
    failed lifecycle markers; exceptions expose a report via GuardedRunError.
    launch_environment=None inherits the parent environment; an explicit mapping
    is copied and passed as the entire launcher environment, including when empty.
    launch_cwd=None uses work; an explicit canonical absolute directory changes
    only the launcher cwd, not publication sources or the callback's work argument.
    These handoff options do not establish confinement or certify execution.
    A launch_state of unknown requires callers to retain work, never delete it:
    no child handle exists to prove termination or permit publication.
    """
    if not isinstance(command, (tuple, list)) or not command or any(
            not isinstance(arg, str) or "\0" in arg for arg in command) or not command[0]:
        raise ValueError("command must be fixed nonempty argv, not shell text")
    argv = tuple(command)
    if isinstance(parent_timeout, bool) or not isinstance(parent_timeout, (int, float)) or \
            not math.isfinite(parent_timeout) or parent_timeout <= 0:
        raise ValueError("parent_timeout must be finite and positive")
    if not isinstance(output_layout, OutputLayout):
        raise ValueError("output_layout must be a fixed OutputLayout enum")
    receipt = independently_verified_confinement_receipt
    if receipt is not None and (not isinstance(receipt, str) or not receipt.strip() or
                                len(receipt) > 4096 or "\0" in receipt):
        raise ValueError("receipt must be a nonempty bounded external evidence reference")
    if verify_callback is not None and not callable(verify_callback):
        raise ValueError("verify_callback must be callable")
    work = Path(work)
    if not work.is_absolute() or not work.is_dir():
        raise ValueError("work must be an existing absolute private directory")

    environment = None
    if launch_environment is not None:
        if not isinstance(launch_environment, Mapping):
            raise ValueError("launch_environment must be a mapping of strings")
        environment = dict(launch_environment)
        if any(not isinstance(key, str) or not key or "=" in key or "\0" in key or
               not isinstance(value, str) or "\0" in value
               for key, value in environment.items()):
            raise ValueError("launch_environment requires nonempty keys without '=' or NUL "
                             "and string values without NUL")
    cwd = work
    if launch_cwd is not None:
        try:
            cwd = os.fspath(launch_cwd)
        except TypeError as exc:
            raise ValueError("launch_cwd must be a canonical absolute directory") from exc
        if not isinstance(cwd, str) or not cwd.startswith("/") or "\0" in cwd or (
                cwd != "/" and any(part in ("", ".", "..") for part in cwd.split("/")[1:])):
            raise ValueError("launch_cwd must be a canonical absolute directory")
        if not Path(cwd).is_dir():
            raise ValueError("launch_cwd must be an existing directory")

    # Exclusive construction leaves failed reservations permanently consumed.
    writer = SerializedBudget(root, slot, byte_limit=byte_limit, log_limit=log_limit,
                              status_reserve=status_reserve)
    child = None
    launch_state = "not_started"
    reaped = verified = False
    discarded = 0
    published = []
    state = "failed"
    failure = None
    caught = None
    prior_mask = None
    try:
        writer.start()
        deadline = time.monotonic() + parent_timeout
        try:
            if os.name == "posix" and hasattr(signal, "pthread_sigmask"):
                prior_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            launch_state = "unknown"
            child = subprocess.Popen(argv, cwd=cwd, env=environment,
                                     stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     close_fds=True, start_new_session=True, bufsize=0)
            launch_state = "acquired"
        finally:
            if prior_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
                prior_mask = None
        # Count only observed bytes, never claim unread pipe data as discarded.
        observed = 0
        saved_before = writer.saved_bytes
        def counted_chunks():
            nonlocal observed
            for chunk in _drain(child, deadline):
                observed += len(chunk)
                yield chunk
        try:
            writer.stream_log(counted_chunks())
        finally:
            discarded = max(0, observed - (writer.saved_bytes - saved_before))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("parent timeout before wrapper reap")
        child.wait(timeout=remaining)
        reaped = True
        if child.returncode != 0:
            raise RuntimeError("wrapper exited nonzero: " + str(child.returncode))
        if receipt is not None and verify_callback is not None:
            if verify_callback(receipt, argv, work, child.returncode) is not True:
                raise RuntimeError("caller verification refused publication")
            verified = True
            validate_inventory(work, InventoryOutputLayout[output_layout.name])
            for source, destination in _OUTPUTS[output_layout]:
                writer.publish(work / source, destination, confinement_verified=True,
                               payload_terminated=True)
                published.append(destination)
            writer.write_manifest()
            state = "complete"
        else:
            state = "unverified"
    except BaseException as exc:
        caught = exc
        failure = type(exc).__name__ + ": " + str(exc)
        if launch_state == "unknown":
            failure += "; launch state unknown: retain work; termination unconfirmed"
        if child is not None:
            try:
                _terminate_and_reap(child)
                reaped = True
            except BaseException as cleanup:
                reaped = False
                failure += "; cleanup unconfirmed: " + type(cleanup).__name__
    finally:
        if child is not None and child.stdout is not None:
            try:
                child.stdout.close()
            except BaseException as exc:
                caught = caught or exc
                failure = failure or "pipe close failed"
                state = "failed"
        try:
            writer.finish("complete" if state == "complete" and caught is None else "failed")
        except BaseException as exc:
            caught = caught or exc
            failure = (failure + "; " if failure else "") + "final status failed: " + str(exc)
            state = "failed"
        finally:
            try:
                writer.close()
            except BaseException as exc:
                caught = caught or exc
                failure = (failure + "; " if failure else "") + "writer close failed: " + str(exc)
                state = "failed"
            finally:
                # Pending handlers can raise before the first restoration takes effect.
                if prior_mask is not None:
                    try:
                        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
                    except BaseException as exc:
                        caught = caught or exc
                        failure = (failure + "; " if failure else "") + "signal mask restore failed: " + str(exc)
                        state = "failed"
    report = RunReport(slot, "failed" if caught else state,
                       child.returncode if child is not None else None, reaped, verified,
                       receipt, tuple(published), writer.saved_bytes, discarded, failure,
                       launch_state)
    if caught is not None:
        if not isinstance(caught, Exception):
            caught.report = report
            raise caught
        raise GuardedRunError(report) from caught
    return report


def run_guarded(root, slot, command, *, work, parent_timeout,
                confinement_receipt=None, verify_callback=None,
                output_layout=OutputLayout.REFERENCE,
                launch_environment=None, launch_cwd=None):
    """Convenience spelling; same trusted primitive, never a runtime gate."""
    return execute_and_publish(
        root, slot, command, work=work, parent_timeout=parent_timeout,
        independently_verified_confinement_receipt=confinement_receipt,
        verify_callback=verify_callback, output_layout=output_layout,
        launch_environment=launch_environment, launch_cwd=launch_cwd)
