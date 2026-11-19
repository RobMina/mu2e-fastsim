"""The FNAL grid (FermiGrid) through jobsub_lite for the fast chain: hependpoints.jobsub with the Mu2e settings.

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

``--need-storage-modify`` is hependpoints' default, the logical form of output_dir (``token_path``): the job token
may write only there. Every job also carries ``--lines=+FERMIHTC_AutoRelease=True`` (a held lifetime overrun is
re-queued) and the default usage model DEDICATED,OPPORTUNISTIC. Job layout, recovery, status, fetch, cancel and
content-addressed staging (``stage_file``) are hependpoints.jobsub's (its docs/jobsub.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from hependpoints.apptainer import sandbox_fallback  # noqa: F401  (re-exported for the task packages)
from hependpoints.jobsub import JobsubEndpoint, token_path  # noqa: F401  (token_path: re-exported for callers)

from mu2efastsim import site

from . import mu2e_config as cfg

MU2E_SETUP = "source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh"
HOST_SETUP = f"{{ {MU2E_SETUP}; muse setup ops; }} >/dev/null 2>&1 </dev/null || true\n"
WORKER_SETUP = (f"{{ {MU2E_SETUP}; setup ifdhc; }} >/dev/null 2>&1 </dev/null || true\n"
                "command -v ifdh >/dev/null || { echo 'no ifdh on the worker'; exit 89; }\n")
AUTO_RELEASE = "--lines=+FERMIHTC_AutoRelease=True"
DISK = "4GB"


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
        stage_dir = stage_dir or site.require("FASTSIMKIT_JOBSUB_STAGE_DIR")
        if host_setup is None:
            host_setup = site.setting("FASTSIMKIT_JOBSUB_SETUP", HOST_SETUP)
        self.host_setup = host_setup if not host_setup or host_setup.endswith("\n") else host_setup + "\n"
        super().__init__(group=group, workdir=workdir or site.require(*self.WORKDIR_VARS),
                         output_dir=output_dir or site.require(*self.OUTPUT_VARS), host=host, role=role,
                         stage_dir=stage_dir, disk=disk or site.setting("FASTSIMKIT_JOBSUB_DISK", DISK),
                         worker_setup=WORKER_SETUP if worker_setup is None else worker_setup,
                         extra_args=extra_args, state_dir=state_dir, runner=runner, name=name,
                         environment=cfg.ENVIRONMENT, timeout_s=timeout_s, **kw)

    def _raw(self, script: str, stdin: Optional[bytes] = None, timeout: Optional[float] = None) -> bytes:
        return super()._raw(self.host_setup + script, stdin, timeout)
