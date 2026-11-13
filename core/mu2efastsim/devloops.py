"""Device-side loops on a GPU (XLA command buffers): MU2EFS_DEVICE_LOOPS=1 (default) or 0.

XLA records most GPU kernels of a compiled program into command buffers (CUDA graphs), but by default it runs a
`lax.while_loop` or `lax.cond` from the host: after each iteration the predicate is copied to the host, which then
launches the next iteration. Kernels with many short device loops (the tracker electronics, the SiPM and pulse-fit
loops) spend part of their time waiting on those round trips. With the setting on (the default),
`--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE` is appended to XLA_FLAGS: XLA keeps its default command-buffer
types and also records loops and conditionals, which then run on the device (CUDA conditional nodes). It is meant
to change how the kernels are launched, not what they compute (the W6 mixing outputs were checked identical with and
without it over full passes). The calorimeter device chain is an exception: its clustering (nested device loops) gave
wrong clusters with it on A100 GPUs, so calofastsim.device_chain compiles its chains without command buffers (a
per-jit XLA option). It has no effect on CPU backends. MU2EFS_DEVICE_LOOPS=0 leaves XLA's default.

XLA reads XLA_FLAGS when JAX creates its first backend, so `apply()` runs when the `mu2efastsim` package is imported,
as for the determinism setting (determinism.py). A value the user already put in XLA_FLAGS for the same flag is left
alone. `state()` (recorded by `precision.provenance()`) says whether the flag is set and whether a JAX backend existed
already when it was appended (then it took no effect in this process). Standard library only.
"""
import os

from .determinism import _backend_started

FLAG = "--xla_gpu_enable_command_buffer"
VALUE = "+CONDITIONAL,+WHILE"
_STATE = {}


def requested():
    """The setting: MU2EFS_DEVICE_LOOPS, on unless "0" (or "false"/"off"/"no")."""
    value = os.environ.get("MU2EFS_DEVICE_LOOPS", "1").strip().lower()
    if value not in ("1", "0", "true", "false", "on", "off", "yes", "no", ""):
        raise ValueError(f"MU2EFS_DEVICE_LOOPS must be 1 or 0, not {value!r}")
    return value not in ("0", "false", "off", "no")


def apply():
    """Append the device-loop flag to XLA_FLAGS when the setting is on (once per process)."""
    if _STATE:
        return dict(_STATE)
    flags = os.environ.get("XLA_FLAGS", "")
    user = FLAG in flags
    on = requested()
    if on and not user:
        os.environ["XLA_FLAGS"] = (flags + " " if flags.strip() else "") + FLAG + "=" + VALUE
    _STATE.update(requested=on, flag_source="user" if user else ("setting" if on else "none"),
                  late=bool(on and not user and _backend_started()))
    return dict(_STATE)


def state():
    """Device-loop record for provenance: the setting, where the flag came from, the flag value XLA sees now, and
    `late` (appended after a JAX backend already existed, so not in effect for this process)."""
    s = dict(_STATE) if _STATE else {"requested": requested(), "flag_source": "not applied", "late": False}
    value = None
    for tok in os.environ.get("XLA_FLAGS", "").split():
        if tok.startswith(FLAG):
            value = tok.partition("=")[2]
    s["xla_gpu_enable_command_buffer"] = value
    return s
