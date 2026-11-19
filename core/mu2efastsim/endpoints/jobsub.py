"""The FNAL grid (FermiGrid) through jobsub_lite for the fast chain: hependpoints.jobsub with the Mu2e settings, plus
content-addressed staging of job inputs to dCache scratch.

Every command runs on an FNAL submit host (an interactive node where ``jobsub_submit`` and ``ifdh`` work with a valid
grid token) reached by ``ssh -o BatchMode=yes <host>``; the endpoint never obtains or refreshes credentials. Settings
come from arguments or the environment, never from the package:

  host        $FASTSIMKIT_JOBSUB_HOST (else $FASTSIMKIT_FNAL_HOST)   ssh host of the submit node
  workdir     $FASTSIMKIT_JOBSUB_WORKDIR    submit-host directory for job files, the submit CWD and code snapshots
                                            (a data area, never a synced code tree)
  output_dir  $FASTSIMKIT_JOBSUB_OUTPUT_DIR dCache scratch directory the jobs copy their results to
  stage_dir   $FASTSIMKIT_JOBSUB_STAGE_DIR  dCache scratch directory the kit stages image, code and inputs to
  group / role  mu2e / Analysis ($FASTSIMKIT_JOBSUB_GROUP, $FASTSIMKIT_JOBSUB_ROLE)
  disk        $FASTSIMKIT_JOBSUB_DISK (default 4GB: image, its sandbox, code, inputs and outputs of one CPU job)
  host_setup  $FASTSIMKIT_JOBSUB_SETUP: shell text run before every submit-host command (default: the Mu2e
              environment and ``muse setup ops``, which puts ifdh on the PATH)
  worker_setup  shell text run first on the worker (default: the Mu2e environment and ifdhc)

``--need-storage-modify`` is the logical form of output_dir (the dCache mount prefix ``pnfs`` dropped): the job token
may write only there. Every job also carries ``--lines=+FERMIHTC_AutoRelease=True`` (a held lifetime overrun is
re-queued) and the default usage model DEDICATED,OPPORTUNISTIC. Job layout, recovery, status, fetch and cancel are
hependpoints.jobsub's (its docs/jobsub.md).

Staging (``stage_file``): a file on the submit host is copied with ``ifdh cp`` to
``<stage_dir>/<kind>/<sha256[:16]>/<name>``, then copied back to a temporary file on the submit host and rehashed;
only a verified copy is recorded (locally, ``<state>/staged/``) and later reused. dCache is never listed or stat-ed.
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from hependpoints import EndpointError
from hependpoints.jobsub import JobsubEndpoint
from hependpoints.submission import durable_json, read_json, submission_lock

from mu2efastsim import site

from . import mu2e_config as cfg

MU2E_SETUP = "source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh"
HOST_SETUP = f"{{ {MU2E_SETUP}; muse setup ops; }} >/dev/null 2>&1 </dev/null || true\n"
WORKER_SETUP = (f"{{ {MU2E_SETUP}; setup ifdhc; }} >/dev/null 2>&1 </dev/null || true\n"
                "command -v ifdh >/dev/null || { echo 'no ifdh on the worker'; exit 89; }\n")
AUTO_RELEASE = "--lines=+FERMIHTC_AutoRelease=True"
DISK = "4GB"
_SHA = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def token_path(path: str) -> str:
    """The logical dCache path jobsub_lite expects in --need-storage-modify: the mount path without its mount prefix."""
    mount = "/pnfs/"
    return path[len(mount) - 1:] if path.startswith(mount) else path


def quiet_ssh_runner(host: str, connect_timeout: int = 15):
    """ssh -o BatchMode=yes <host>, with stdin from /dev/null when the command has no input: a kit served over stdio
    (MCP) must never let a helper ssh read the client's requests. Uses the user's ssh configuration (an existing
    ControlMaster for the host is reused)."""
    def run(cmd: Sequence[str], stdin: Optional[bytes], timeout: float) -> subprocess.CompletedProcess:
        remote = " ".join(shlex.quote(c) for c in cmd)
        argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}", host, remote]
        if stdin is None:
            return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
        return subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    return run


class FermiGridEndpoint(JobsubEndpoint):
    GROUP_VARS = ("FASTSIMKIT_JOBSUB_GROUP",)
    WORKDIR_VARS = ("FASTSIMKIT_JOBSUB_WORKDIR",)
    OUTPUT_VARS = ("FASTSIMKIT_JOBSUB_OUTPUT_DIR",)
    CONFIG_HINT = site.HINT

    def __init__(self, host: str | None = None, workdir: str | None = None, output_dir: str | None = None,
                 stage_dir: str | None = None, group: str | None = None, role: str | None = None,
                 disk: str | None = None, host_setup: str | None = None, worker_setup: str | None = None,
                 extra_args: Sequence[str] = (AUTO_RELEASE,), state_dir: str | Path = "~/.mu2efastsim/fermigrid",
                 runner=None, name: str = "fermigrid", timeout_s: float = 300.0, **kw):
        kw.pop("max_su_per_job", None)            # the kit passes the UVA budget cap; FNAL slots carry no SU charge
        host = host or site.setting("FASTSIMKIT_JOBSUB_HOST") or site.require("FASTSIMKIT_FNAL_HOST")
        group = group or site.setting("FASTSIMKIT_JOBSUB_GROUP", "mu2e")
        role = role or site.setting("FASTSIMKIT_JOBSUB_ROLE", "Analysis")
        output_dir = (output_dir or site.require(*self.OUTPUT_VARS)).rstrip("/")
        self.stage_dir = (stage_dir or site.require("FASTSIMKIT_JOBSUB_STAGE_DIR")).rstrip("/")
        if not self.stage_dir.startswith("/"):
            raise EndpointError(f"{name}: stage_dir must be an absolute path")
        if host_setup is None:
            host_setup = site.setting("FASTSIMKIT_JOBSUB_SETUP", HOST_SETUP)
        self.host_setup = host_setup if not host_setup or host_setup.endswith("\n") else host_setup + "\n"
        super().__init__(group=group, workdir=workdir or site.require(*self.WORKDIR_VARS), output_dir=output_dir,
                         host=host, role=role, storage_modify=token_path(output_dir),
                         disk=disk or site.setting("FASTSIMKIT_JOBSUB_DISK", DISK),
                         worker_setup=WORKER_SETUP if worker_setup is None else worker_setup,
                         extra_args=extra_args, state_dir=state_dir, runner=runner or quiet_ssh_runner(host),
                         name=name, environment=cfg.ENVIRONMENT, timeout_s=timeout_s, **kw)

    def _raw(self, script: str, stdin: Optional[bytes] = None, timeout: Optional[float] = None) -> bytes:
        return super()._raw(self.host_setup + script, stdin, timeout)

    # -- staging ---------------------------------------------------------------------------------------------------
    def staged_path(self, kind: str, sha256: str, name: str) -> str:
        if not _NAME.fullmatch(kind) or not _NAME.fullmatch(name) or not _SHA.fullmatch(sha256):
            raise EndpointError(f"{self.name}: bad staging key {kind!r}/{sha256!r}/{name!r}")
        return f"{self.stage_dir}/{kind}/{sha256[:16]}/{name}"

    def _staged_record(self, dest: str) -> Path:
        d = self.state.root / "staged"
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        return d / (hashlib.sha256(dest.encode()).hexdigest()[:32] + ".json")

    def remote_sha256(self, path: str) -> Optional[str]:
        """sha256 of a dCache file, copied to a temporary file on the submit host; None when the copy fails."""
        q = shlex.quote(path)
        out = self._sh(f"t=$(mktemp -d) || exit 1; if {self.copy_command} {q} \"$t/f\" >/dev/null 2>&1; then "
                       f"sha256sum \"$t/f\" | cut -c1-64; else echo ABSENT; fi; rm -rf \"$t\"", timeout=1800).strip()
        return out if _SHA.fullmatch(out) else None

    def stage_file(self, src: str, kind: str, name: str, sha256: str, size: int) -> str:
        """Copy the submit-host file `src` (pinned sha256 and size) to <stage_dir>/<kind>/<sha16>/<name> once and
        return that dCache path. A destination recorded as verified is reused without a remote call; otherwise the
        source is rehashed (a changed source fails), copied, and the dCache copy is rehashed before it is recorded."""
        dest = self.staged_path(kind, sha256, name)
        rec = self._staged_record(dest)
        with submission_lock(rec.parent, rec.stem):
            old = read_json(rec)
            if old and old.get("sha256") == sha256 and old.get("size") == size and old.get("dest") == dest:
                return dest
            q, qd = shlex.quote(src), shlex.quote(dest)
            got = self._sh(f"set -e; sha256sum {q} | cut -c1-64; stat -c %s {q}", timeout=1800).split()
            if got != [sha256, str(size)]:
                raise EndpointError(f"{self.name}: {src} on the submit host is not the pinned file "
                                    f"({got} != {[sha256, str(size)]})")
            self._sh(f"{self.remove_command} {qd} >/dev/null 2>&1; {self.mkdir_command} "
                     f"{shlex.quote(os.path.dirname(dest))} >/dev/null && {self.copy_command} {q} {qd} >/dev/null",
                     timeout=1800)
            if self.remote_sha256(dest) != sha256:
                raise EndpointError(f"{self.name}: staged copy {dest} does not match {src}")
            durable_json(rec, {"src": src, "dest": dest, "sha256": sha256, "size": size})
            return dest
