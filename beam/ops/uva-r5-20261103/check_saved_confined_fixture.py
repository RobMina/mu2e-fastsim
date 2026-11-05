"""Bounded read-only analysis of the closed synthetic outputs from job 22478642.

No art or transport is rerun. This is not the physics saved-output guard. The
analysis container receives the old private work read-only and no saved-output
bind. Only this trusted outer writer retains at most 1 MiB log and 4096 status.
"""

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


BASE = Path("/home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103")
OLD_WORK = Path("/tmp/r5-apptainer-boundary-8zywj47s/work")
APPTAINER = "/apps/software/standard/core/apptainer/1.4.5/bin/apptainer"
IMAGE = "/scratch/ram2aq/mu2efastsim/mu2efastsim.sif"


def main():
    job = os.environ["SLURM_JOB_ID"]
    if not job.isdecimal():
        raise ValueError("compute job id required")
    out = Path("/scratch/ram2aq/mu2efastsim/flash_r5_20261103/out") / ("confined_fixture_checks_" + job)
    out.mkdir(mode=0o700)
    child, code, error = None, None, None
    saved = observed = 0
    deadline = time.monotonic() + 240
    try:
        if not (OLD_WORK / "fixture/export.root").is_file() or not (OLD_WORK / "reread/export.root").is_file():
            raise ValueError("closed node-local outputs unavailable; never rerun automatically")
        command = [APPTAINER, "exec", "--userns", "--containall", "--cleanenv", "--no-home",
                   "--no-mount", "bind-paths,hostfs,cwd", "--disable-cache", "--cwd", "/mnt/work",
                   "--bind", str(OLD_WORK) + ":/mnt/work:ro",
                   "--bind", str(BASE / "apptainer_boundary_v1") + ":/opt/r5/checks:ro",
                   IMAGE, "python3", "-I", "-B", "/opt/r5/checks/check_confined_fixture.py"]
        environment = {"PATH": os.path.dirname(APPTAINER) + ":/usr/bin:/bin", "HOME": "/tmp",
                       "LANG": "C", "LC_ALL": "C"}
        with (out / "merged.log").open("xb", buffering=0) as log:
            child = subprocess.Popen(command, cwd="/", env=environment, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     close_fds=True, start_new_session=True)
            os.set_blocking(child.stdout.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("saved-output checker deadline")
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
                        if not count:
                            raise OSError("log write made no progress")
                        saved += count
                        retained = retained[count:]
            code = child.wait(timeout=max(0.1, deadline - time.monotonic()))
    except BaseException as exc:
        error = (type(exc).__name__ + ": " + str(exc))[:800]
    finally:
        if child is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    error = "cleanup: " + str(exc)[:800]
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    error = "checker wrapper not reaped"
            child.stdout.close()
    report = {"checker_exit": code, "error": error, "log_bytes": saved,
              "discarded_log_bytes": observed - saved, "source_job": "22478642",
              "art_rerun": False, "transport_events": 0, "physics_completeness": False,
              "passed": code == 0 and error is None}
    content = (json.dumps(report, sort_keys=True) + "\n").encode("ascii")
    if len(content) > 4096:
        raise ValueError("status exceeds fixed limit")
    with (out / "status.json").open("xb") as stream:
        stream.write(content)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
