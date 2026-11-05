"""Bounded no-Geant4 diagnostic ONLY; never consumes or certifies pilot slots.

The trusted outer diagnostic keeps <=1 MiB of merged log and one small status.
Transient work is retained for job-lifecycle inspection, never copied to saved
output. This deliberately does not implement approved transport or publication.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time


BASE = Path("/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103")
DATA = Path("/scratch/ram2aq/mu2efastsim/flash_r5_20261103/out")
DIST = Path("/scratch/ram2aq/mu2efastsim/environments/cvmfsexec-el8")
FILES = ("flash_reference_apptainer.py", "flash_reference_sandbox.py",
         "flash_reference_apptainer_probe.py")
REPOSITORIES = ("config-osg.opensciencegrid.org", "mu2e.opensciencegrid.org",
                "fermilab.opensciencegrid.org", "singularity.opensciencegrid.org")
CVMFSEXEC_SHA256 = "c5ecbd4a580a05522013db6a7269ab8bdec92317da58767d5a0b3eba456f5390"
MOUNTREPO_SHA256 = "99a8002ea9740a6dabfb67a35688a4d1955442b7e8dc881278dc64c0d470cc57"


def configure_cvmfsexec(ce, cache):
    """Prepare an unlaunched copied runtime; discard helper logs explicitly."""
    import hashlib
    for name, expected in (("cvmfsexec", CVMFSEXEC_SHA256), ("mountrepo", MOUNTREPO_SHA256)):
        if hashlib.sha256((ce / name).read_bytes()).hexdigest() != expected:
            raise ValueError("unreviewed CVMFSEXEC helper: " + name)
    config = ce / "dist/etc/cvmfs"
    domain = config / "domain.d"
    domain.mkdir(parents=True, exist_ok=True)
    (domain / "opensciencegrid.org.local").write_text(
        'CVMFS_SERVER_URL="http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/@fqrn@;'
        'http://cvmfs-s1bnl.opensciencegrid.org:8000/cvmfs/@fqrn@"\n')
    (config / "default.local").write_text(
        'CVMFS_HTTP_PROXY="DIRECT"\nCVMFS_CACHE_BASE=' + str(cache) +
        '\nCVMFS_QUOTA_LIMIT=6000\nCVMFS_TIMEOUT=30\nCVMFS_TIMEOUT_DIRECT=30\n'
        'CVMFS_USYSLOG=/dev/null\nCVMFS_DEBUGLOG=\n')
    final = config / "config.d"
    final.mkdir(parents=True, exist_ok=True)
    for repository in REPOSITORIES:
        # mountrepo emits repository.local last, after repository/domain config.
        (final / (repository + ".local")).write_text(
            'CVMFS_USYSLOG=/dev/null\nCVMFS_DEBUGLOG=\n')
    print(json.dumps({"cvmfsexec_nonesting_requested": True,
                      "repository_logs_disposition": "discard:/dev/null",
                      "cvmfsexec_sha256": CVMFSEXEC_SHA256,
                      "mountrepo_sha256": MOUNTREPO_SHA256,
                      "runtime_mode_observed": False, "verified": False}), flush=True)


def mounted(local, apptainer, fixture=False):
    spec = importlib.util.spec_from_file_location("r5_apptainer", str(local / "checks" / FILES[0]))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    command = ["/usr/bin/python3", "-I", "-B", "/opt/r5/checks/flash_reference_apptainer_probe.py",
               "--outer-pid-namespace", os.readlink("/proc/self/ns/pid")]
    for target in ("/scratch/ram2aq/r5-must-not-write-" + local.name,
                   "/home/ram2aq/r5-must-not-write-" + local.name,
                   "/opt/r5/checks/must-not-write", "/opt/r5/area/must-not-write",
                   "/cvmfs/mu2e.opensciencegrid.org/must-not-write",
                   str(local / "cache" / "must-not-write")):
        command.extend(("--persistent-target", target))
    if fixture:
        command = ["/bin/bash", "/opt/r5/checks/run_confined_fixture.sh", os.readlink("/proc/self/ns/pid")]
    plan = module.build_command(apptainer=apptainer, runtime=str(local / "runtime"),
                                work=str(local / "work"), code=str(local / "checks"),
                                area=str(local / "area"), command=command,
                                file_size_limit=16 * 1024**2)
    configuration = local / "runtime/config/apptainer.conf"
    with configuration.open("xb") as stream:
        stream.write(plan.configuration)
    configuration.chmod(0o400)
    import hashlib
    print(json.dumps({"candidate_argv": plan.argv, "launcher_environment": plan.environment,
                      "configuration_sha256": hashlib.sha256(plan.configuration).hexdigest(),
                      "verified": False}), flush=True)
    print("image_resolved=" + os.path.realpath(module.PINNED_IMAGE), flush=True)
    child = subprocess.Popen(plan.argv, env=plan.environment, cwd="/", stdin=subprocess.DEVNULL,
                             close_fds=True)
    return child.wait()


def setup(local, apptainer, fixture=False, lifecycle=False):
    """All fallible copies/mount startup are children of the bounded collector."""
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    print("transient_work=" + str(local), flush=True)
    for name in ("runtime", "work", "checks", "area", "cache"):
        (local / name).mkdir(mode=0o700)
    for name in ("home", "tmp", "cache", "config", "session", "empty"):
        (local / "runtime" / name).mkdir(mode=0o700)
    for name in ("tmp", "home", "shm"):
        (local / "work" / name).mkdir(mode=0o700)
    source = BASE / "apptainer_boundary_v1"
    for name in (*FILES, *(("run_confined_fixture.sh",) if fixture else ()),
                 *(("flash_reference_lifecycle_payload.py",) if lifecycle else ())):
        shutil.copyfile(source / name, local / "checks" / name)
    shutil.copyfile(__file__, local / "diagnostic.py")
    if lifecycle:
        shutil.copyfile(BASE / "apptainer_lifecycle_probe.py", local / "lifecycle.py")
        shutil.copyfile(source / "flash_reference_subreaper.py", local / "subreaper.py")
    deployed = BASE / "exporter_runtime_v1"
    shutil.copytree(deployed / "FlashReference", local / "area" / "FlashReference")
    (local / "area" / ".muse").write_text("ENVSET p109\n")
    (local / "area" / "backing").symlink_to("/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_41_01")
    library = local / "area" / "build/al9-prof-e29-p109/FlashReference/lib"
    library.mkdir(parents=True)
    for kind in ("Exporter", "Fixture"):
        shutil.copyfile(deployed / ("FlashReference" + kind + ".plugin"),
                        library / ("libflashreference_FlashReference" + kind + "_module.so"))
    shutil.copytree(DIST, local / "cvmfsexec", symlinks=True)
    ce = local / "cvmfsexec"
    configure_cvmfsexec(ce, local / "cache")
    script = "lifecycle.py" if lifecycle else "diagnostic.py"
    command = [str(ce / "cvmfsexec"), "-N", *REPOSITORIES, "--",
               sys.executable, "-I", "-B", str(local / script), "--mounted", str(local),
               "--apptainer", apptainer]
    if fixture:
        command.append("--fixture")
    if lifecycle:
        command.extend(("--host-pid-namespace", os.readlink("/proc/self/ns/pid")))
        return _supervise_lifecycle(local, command)
    os.execve(command[0], command, os.environ.copy())


def _supervise_lifecycle(local, command):
    """One normal trial; collector deadline is outside this dedicated subreaper."""
    from dataclasses import asdict
    import stat
    spec = importlib.util.spec_from_file_location("r5_subreaper", local / "subreaper.py")
    subreaper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = subreaper
    spec.loader.exec_module(subreaper)
    report = subreaper.run_tree(command, os.environ.copy(), "/", timeout=540,
                               external_watchdog_present=True)
    print(json.dumps({"subreaper_observation": asdict(report), "verified": False}), flush=True)
    if not report.success or not report.ancestry_drained:
        return 1
    if (local / "cvmfsexec/log").exists():
        raise ValueError("unexpected uncounted repository-log directory")

    def read_bounded(name):
        fd = os.open(local / "work" / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4096:
                raise ValueError("invalid bounded lifecycle observation file")
            data = os.read(fd, 4097)
            if not data or len(data) > 4096:
                raise ValueError("missing or oversized lifecycle record")
            return data
        finally:
            os.close(fd)

    ready = json.loads(read_bounded("lifecycle-ready.json"))
    heartbeats = [json.loads(line) for line in read_bounded("lifecycle-heartbeat.jsonl").splitlines()]
    if (ready.get("nonce") != "normal_" + local.name or ready.get("mode") != "normal"
            or ready.get("intermediate_reaped") is not True
            or ready.get("emergency_expiry_seconds") != 30
            or ready.get("verified") is not False or not 1 <= len(heartbeats) <= 30
            or heartbeats[0].get("sequence") != 0
            or type(heartbeats[0].get("monotonic_ns")) is not int):
        raise ValueError("incomplete or mismatched normal lifecycle fixture")
    elapsed = (time.monotonic_ns() - heartbeats[0]["monotonic_ns"]) / 1e9
    if not 0 <= elapsed < 20:
        raise ValueError("cleanup not observed safely before emergency expiry")
    print(json.dumps({"normal_lifecycle_ready": ready, "heartbeat_count": len(heartbeats),
                      "drained_before_emergency_seconds": elapsed,
                      "ancestry_domain_only": True, "verified": False,
                      "fully_enforced": False, "transport_events": 0}), flush=True)
    return 0


class DiagnosticTerminated(Exception):
    pass


def _termination(signum, frame):
    raise DiagnosticTerminated("received signal " + str(signum))


def _reap(child):
    """Direct-child best effort only, never a descendant-lifecycle certificate."""
    errors = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            errors.append(type(exc).__name__ + ": " + str(exc)[:256])
        try:
            child.wait(timeout=2)
        except BaseException as exc:
            errors.append(type(exc).__name__ + ": " + str(exc)[:256])
    return child.returncode is not None, errors


def _report_text(value, limit=160):
    """Printable ASCII keeps even hostile exception text within receipt bounds."""
    return "".join(c if 32 <= ord(c) <= 126 else "?" for c in str(value)[:limit])


def diagnostic(apptainer, fixture=False, lifecycle=False):
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal():
        raise RuntimeError("compute allocation required")
    out = DATA / ("apptainer_boundary_probe_" + job)
    out.mkdir(mode=0o700)
    saved, observed = 0, 0
    child = local = prior_mask = None
    state, error, launch_state = "failed", None, "not_started"
    reaped, teardown = False, []
    handlers, dispositions, normalized = {}, {}, []
    try:
        deadline = time.monotonic() + 600
        for sig in (signal.SIGINT, signal.SIGTERM):
            handler = signal.getsignal(sig)
            dispositions[sig.name] = ("None" if handler is None else
                                      "SIG_DFL" if handler == signal.SIG_DFL else
                                      "SIG_IGN" if handler == signal.SIG_IGN else
                                      "callable" if callable(handler) else "other")
            # Normalize unknown native handlers to DFL in this one-shot CLI.
            handlers[sig] = signal.SIG_DFL if handler is None else handler
            if handler is None:
                normalized.append(sig.name)
        for sig in handlers:
            signal.signal(sig, _termination)
        with (out / "merged.log").open("xb", buffering=0) as log:
            local = Path(tempfile.mkdtemp(prefix="r5-apptainer-boundary-", dir="/tmp"))
            os.chmod(local, 0o700)
            command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                       "--setup", str(local), "--apptainer", apptainer]
            if fixture:
                command.append("--fixture")
            if lifecycle:
                command.append("--lifecycle")
            prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(handlers))
            launch_state = "unknown"
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
                                     cwd="/", env={"PATH": os.path.dirname(apptainer) + ":/usr/bin:/bin",
                                                   "HOME": str(local), "LANG": "C", "LC_ALL": "C"})
            launch_state = "acquired"
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            prior_mask = None
            os.set_blocking(child.stdout.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("diagnostic deadline")
                    if not selector.select(0.1):
                        continue
                    try:
                        chunk = os.read(child.stdout.fileno(), 65536)
                    except (BlockingIOError, InterruptedError):
                        continue
                    if not chunk:
                        break
                    observed += len(chunk)
                    retained = memoryview(chunk)[:max(0, 1024**2 - saved)]
                    while retained:
                        count = log.write(retained)
                        if count is None or count <= 0:
                            raise OSError("diagnostic log write made no progress")
                        saved += count
                        retained = retained[count:]
            code = child.wait(timeout=max(0.1, deadline - time.monotonic()))
            state = "child_checks_passed" if code == 0 else "failed"
    except BaseException as exc:
        error = _report_text(type(exc).__name__ + ": " + str(exc), 256)
        state = "failed"
    finally:
        # Complete fallible cleanup before writing the final receipt.
        for sig in handlers:
            try:
                signal.signal(sig, signal.SIG_IGN)
            except BaseException as exc:
                teardown.append("handler ignore " + sig.name + ": " + str(exc))
        if child is not None:
            try:
                reaped, reap_errors = _reap(child)
                teardown.extend(reap_errors)
            except BaseException as exc:
                teardown.append("reap: " + str(exc))
            try:
                if child.stdout is not None:
                    child.stdout.close()
            except BaseException as exc:
                teardown.append("pipe close: " + str(exc))
        if prior_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
                prior_mask = None
            except BaseException as exc:
                teardown.append("mask restore: " + str(exc))
        for sig, handler in handlers.items():
            try:
                signal.signal(sig, handler)
            except BaseException as exc:
                teardown.append("handler restore " + sig.name + ": " + str(exc))
        if not reaped or teardown:
            state = "failed"
    report = {"state": state, "returncode": child.returncode if child else None,
              "error": error, "log_bytes": saved, "discarded_log_bytes": observed - saved,
              "transient_work_retained": str(local) if local else None,
              "wrapper_reaped": reaped, "launch_state": launch_state,
              "teardown_errors": [_report_text(item) for item in teardown[:6]],
              "initial_signal_dispositions": dispositions,
              "none_handlers_normalized_to_SIG_DFL": normalized,
              "verified": False, "fully_enforced": False, "transport_events": 0,
              "requested_payload": ("normal_lifecycle_only" if lifecycle else
                                    "fixture_and_reread" if fixture else "prescribed_child_checks"),
              "scope": "No-G4 diagnostic only; no lifecycle/publication or physics certificate."}
    content = (json.dumps(report, sort_keys=True) + "\n").encode("ascii")
    if len(content) > 4096:
        raise RuntimeError("diagnostic receipt exceeds fixed bound")
    with (out / "status.json").open("xb") as stream:
        stream.write(content)
    return 0 if state == "child_checks_passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mounted", type=Path)
    group.add_argument("--setup", type=Path)
    parser.add_argument("--apptainer", required=True)
    payload = parser.add_mutually_exclusive_group()
    payload.add_argument("--fixture", action="store_true", help="Synthetic art and saved reread only; never Geant4")
    payload.add_argument("--lifecycle", action="store_true", help="One normal finite no-Geant4 lifecycle fixture")
    args = parser.parse_args()
    binary = os.path.realpath(args.apptainer)
    if args.mounted:
        if args.lifecycle:
            parser.error("lifecycle uses its separate fixed mounted script")
        return mounted(args.mounted, binary, args.fixture)
    if args.setup:
        return setup(args.setup, binary, args.fixture, args.lifecycle)
    return diagnostic(binary, args.fixture, args.lifecycle)


if __name__ == "__main__":
    raise SystemExit(main())
