"""Self-only host capability diagnostic; no fork, container, payload or admission.

An unsupported syscall is an observed diagnostic result, not successful lifecycle
validation. Kernel release and Seccomp fields are evidence, not a proof of why a
syscall is unavailable. Only signal zero through a pidfd referring to this process
is permitted. Output is one exclusive bounded receipt; no stdout or helper logs.
"""

import ctypes
import json
import os
from pathlib import Path
import platform
import signal
import sys

OUT = Path('/scratch/ram2aq/mu2efastsim/flash_r5_20261103/out')
MAX_RECEIPT = 16384


def self_pidfd():
    """Use the native API or the reviewed Linux x86_64 syscall with flags zero."""
    if sys.platform != 'linux':
        raise RuntimeError('Linux required')
    native = getattr(os, 'pidfd_open', None)
    if native is not None:
        return native(os.getpid(), 0)
    if platform.machine() != 'x86_64' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError('raw pidfd_open requires reviewed Linux x86_64 ABI')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(os.getpid()), ctypes.c_uint(0))
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return fd


def observe():
    """Observe this process only; exceptions remain diagnostic refusal evidence."""
    uname = platform.uname()
    report = {
        'scope': 'Self-only pidfd capability; no fork/container/transport/lifecycle certificate',
        'verified': False, 'fully_enforced': False, 'transport_events': 0,
        'kernel_release': uname.release, 'kernel_version': uname.version,
        'machine': uname.machine, 'node': uname.node,
        'python': sys.version, 'python_pidfd_open': hasattr(os, 'pidfd_open'),
        'python_pidfd_send_signal': hasattr(signal, 'pidfd_send_signal'),
        'pidfd_open_self': False, 'pidfd_signal_zero_self': False,
        'self_fd_closed': False, 'capability_available': False,
        'diagnostic_completed': False, 'error': None,
    }
    fd = None
    try:
        with open('/proc/self/status', 'rb', buffering=0) as source:
            status = source.read(16385)
        if len(status) > 16384:
            raise RuntimeError('bounded self status exceeded')
        keys = (b'NoNewPrivs', b'Seccomp', b'Seccomp_filters', b'CapEff')
        report['self_status'] = {
            key.decode(): value.strip().decode('ascii')
            for line in status.splitlines() if b':' in line
            for key, value in [line.split(b':', 1)] if key in keys
        }
        report['self_pid_namespace'] = os.readlink('/proc/self/ns/pid')
        fd = self_pidfd()
        report['pidfd_open_self'] = True
        signal.pidfd_send_signal(fd, 0, None, 0)
        report['pidfd_signal_zero_self'] = True
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'errno': getattr(exc, 'errno', None),
                           'message': str(exc)[:1024]}
    finally:
        if fd is not None:
            try:
                os.close(fd)
                report['self_fd_closed'] = True
            except Exception as exc:
                report['error'] = {'type': type(exc).__name__, 'errno': getattr(exc, 'errno', None),
                                   'message': str(exc)[:1024]}
    report['capability_available'] = (report['pidfd_open_self']
        and report['pidfd_signal_zero_self'] and report['self_fd_closed']
        and report['error'] is None)
    report['diagnostic_completed'] = True
    return report


def main():
    signal.alarm(10)
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isascii() or not job.isdigit() or not 1 <= len(job) <= 12:
        raise RuntimeError('explicit Slurm job required')
    directory = OUT / ('pidfd_capability_' + job)
    directory.mkdir(mode=0o700)
    receipt = observe()
    receipt['job_id'] = job
    payload = (json.dumps(receipt, sort_keys=True) + '\n').encode('utf-8')
    if len(payload) > MAX_RECEIPT:
        raise RuntimeError('bounded receipt exceeded')
    with (directory / 'receipt.json').open('xb', buffering=0) as target:
        if target.write(payload) != len(payload):
            raise OSError('short diagnostic receipt write')
    signal.alarm(0)
    return 0 if receipt['diagnostic_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
