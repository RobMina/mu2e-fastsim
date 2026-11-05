"""One normal no-Geant4 lifecycle diagnostic; no transport/publication interface.

The existing bounded collector is the independent outer deadline. This revision
never injects controller/supervisor death and never consumes pilot reservations.
A subreaper observes the whole copied CVMFSEXEC/runtime ancestry domain. Kernel
D-state, external launch services, controller death and scheduler-wide teardown
remain outside this diagnostic; no complete guard certificate is emitted.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def mounted(local, apptainer, host_pid_namespace):
    runtime_namespace = os.readlink("/proc/self/ns/pid")
    if runtime_namespace == host_pid_namespace:
        raise ValueError("CVMFSEXEC private PID mode was not observed")
    if any(key in os.environ for key in ("CVMFSEXEC_CMDFD", "CVMFSEXEC_WAITFIFO")):
        raise ValueError("CVMFSEXEC -N control environment was not cleared")
    print(json.dumps({"host_pid_namespace": host_pid_namespace,
                      "cvmfsexec_pid_namespace": runtime_namespace,
                      "cvmfsexec_private_pid_observed": True,
                      "cvmfsexec_control_environment_absent": True,
                      "verified": False}), flush=True)
    plan_module = module("r5_apptainer", local / "checks/flash_reference_apptainer.py")
    command = ["/usr/bin/python3", "-I", "-B",
               "/opt/r5/checks/flash_reference_lifecycle_payload.py",
               "--enable-lifecycle-payload", "--mode", "normal",
               "--nonce", "normal_" + local.name,
               "--outer-pid-namespace", os.readlink("/proc/self/ns/pid")]
    plan = plan_module.build_command(
        apptainer=apptainer, runtime=str(local / "runtime"), work=str(local / "work"),
        code=str(local / "checks"), area=str(local / "area"), command=command,
        file_size_limit=16384)
    config = local / "runtime/config/apptainer.conf"
    with config.open("xb") as stream:
        stream.write(plan.configuration)
    config.chmod(0o400)
    print(json.dumps({"lifecycle_mode": "normal", "candidate_argv": plan.argv,
                      "verified": False, "transport_events": 0}), flush=True)
    os.chdir("/")
    os.execve(plan.argv[0], plan.argv, plan.environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted", type=Path, required=True)
    parser.add_argument("--apptainer", required=True)
    parser.add_argument("--host-pid-namespace", required=True)
    args = parser.parse_args()
    return mounted(args.mounted, args.apptainer, args.host_pid_namespace)


if __name__ == "__main__":
    raise SystemExit(main())
