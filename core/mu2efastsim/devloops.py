"""Device-side loops and concurrent graph regions on a GPU (XLA command buffers).

MU2EFS_DEVICE_LOOPS=1 (default) or 0. XLA records most GPU kernels of a compiled program into command buffers (CUDA
graphs), but by default it runs a `lax.while_loop` or `lax.cond` from the host: after each iteration the predicate is
copied to the host, which then launches the next iteration. Kernels with many short device loops (the tracker
electronics, the SiPM and pulse-fit loops) spend part of their time waiting on those round trips. With the setting on
(the default), `--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE` is appended to XLA_FLAGS: XLA keeps its default
command-buffer types and also records loops and conditionals, which then run on the device (CUDA conditional nodes).
It is meant to change how the kernels are launched, not what they compute (the W6 mixing outputs were checked
identical with and without it over full passes). The calorimeter device chain is an exception: its clustering (nested
device loops) gave wrong clusters with it on A100 GPUs, so calofastsim.device_chain compiles its chains without command
buffers (a per-jit XLA option). It has no effect on CPU backends. MU2EFS_DEVICE_LOOPS=0 leaves XLA's default.

MU2EFS_GRAPH_CONCURRENCY=1 (default) or 0. By default XLA records the kernels of a command buffer as a chain, one after
another. With the setting on, `--xla_gpu_graph_enable_concurrent_region=true` is appended: kernels that do not depend
on each other become parallel branches of the CUDA graph, so many small kernels (a batch step launches tens of
thousands, most of them a few microseconds long) can share the GPU. The dependencies are XLA's own, so the results do
not change (W6 mixing outputs identical with and without it over full passes, every capacity file and dtype). The two
settings are independent; concurrency acts on every command buffer, with or without device loops.

XLA reads XLA_FLAGS when JAX creates its first backend, so `apply()` runs when the `mu2efastsim` package is imported,
as for the determinism setting (determinism.py). A value the user already put in XLA_FLAGS for either flag is left
alone. `state()` (recorded by `precision.provenance()`) says whether each flag is set and whether a JAX backend existed
already when they were appended (then they took no effect in this process). Standard library only.
"""
import os

from .determinism import _backend_started

FLAG = "--xla_gpu_enable_command_buffer"
VALUE = "+CONDITIONAL,+WHILE"
CONCURRENCY_FLAG = "--xla_gpu_graph_enable_concurrent_region"
CONCURRENCY_VALUE = "true"
_STATE = {}


def _setting(name):
    value = os.environ.get(name, "1").strip().lower()
    if value not in ("1", "0", "true", "false", "on", "off", "yes", "no", ""):
        raise ValueError(f"{name} must be 1 or 0, not {value!r}")
    return value not in ("0", "false", "off", "no")


def requested():
    """The device-loop setting: MU2EFS_DEVICE_LOOPS, on unless "0" (or "false"/"off"/"no")."""
    return _setting("MU2EFS_DEVICE_LOOPS")


def concurrency_requested():
    """The concurrent-region setting: MU2EFS_GRAPH_CONCURRENCY, on unless "0" (or "false"/"off"/"no")."""
    return _setting("MU2EFS_GRAPH_CONCURRENCY")


def apply():
    """Append the device-loop and concurrent-region flags to XLA_FLAGS when their settings are on (once per
    process)."""
    if _STATE:
        return dict(_STATE)
    on, conc = requested(), concurrency_requested()   # both checked before anything is changed
    flags = os.environ.get("XLA_FLAGS", "")
    user, conc_user = FLAG in flags, CONCURRENCY_FLAG in flags
    added = []
    if on and not user:
        added.append(FLAG + "=" + VALUE)
    if conc and not conc_user:
        added.append(CONCURRENCY_FLAG + "=" + CONCURRENCY_VALUE)
    if added:
        os.environ["XLA_FLAGS"] = " ".join(([flags] if flags.strip() else []) + added)
    late = _backend_started()
    _STATE.update(requested=on, flag_source="user" if user else ("setting" if on else "none"),
                  late=bool(on and not user and late),
                  concurrency_requested=conc,
                  concurrency_source="user" if conc_user else ("setting" if conc else "none"),
                  concurrency_late=bool(conc and not conc_user and late))
    return dict(_STATE)


def _value(flag):
    value = None
    for tok in os.environ.get("XLA_FLAGS", "").split():
        if tok.startswith(flag):
            value = tok.partition("=")[2]
    return value


def state():
    """Record for provenance: each setting, where its flag came from, the flag value XLA sees now, and `late` /
    `concurrency_late` (appended after a JAX backend already existed, so not in effect for this process)."""
    if _STATE:
        s = dict(_STATE)
    else:
        s = {"requested": requested(), "flag_source": "not applied", "late": False,
             "concurrency_requested": concurrency_requested(), "concurrency_source": "not applied",
             "concurrency_late": False}
    s["xla_gpu_enable_command_buffer"] = _value(FLAG)
    s["xla_gpu_graph_enable_concurrent_region"] = _value(CONCURRENCY_FLAG)
    return s
