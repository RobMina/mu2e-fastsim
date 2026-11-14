"""Device-side loops on a GPU (XLA command buffers), chosen per workflow: MU2EFS_DEVICE_LOOPS=auto (default), 1 or 0.

XLA records most GPU kernels of a compiled program into command buffers (CUDA graphs), but by default it runs a
`lax.while_loop` or `lax.cond` from the host: after each iteration the predicate is copied to the host, which then
launches the next iteration. Kernels with many short device loops (the tracker electronics, the SiPM and pulse-fit
loops) spend part of their time waiting on those round trips. With device loops on,
`--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE` is appended to XLA_FLAGS: XLA keeps its default command-buffer
types and also records loops and conditionals, which then run on the device (CUDA conditional nodes). It is meant
to change how the kernels are launched, not what they compute (the W6 mixing outputs were checked identical with and
without it over full passes). The calorimeter device chain is an exception: its clustering (nested device loops) gave
wrong clusters with it on A100 GPUs, so calofastsim.device_chain compiles its chains without command buffers (a
per-jit XLA option). It has no effect on CPU backends.

Which is faster depends on the workflow, so the default (auto) follows the workflow:
WORKFLOWS lists the measured ones (the W6 mixing chain is faster with device loops; the W2 CE chain with digis, whose
digitiser makes many small launches, is 11 % (float64) / 16 % (float32) slower with them), and a workflow not listed
keeps them on. MU2EFS_DEVICE_LOOPS=1 or 0, or the entry point's --device-loops on|off (select's `loops`), forces the
setting for any workflow.

XLA reads XLA_FLAGS when JAX creates its first backend. `apply()` runs when the `mu2efastsim` package is imported, as
for the determinism setting (determinism.py): it appends the flag unless the setting is 0, so an entry point that
names no workflow behaves as before. A workflow entry point then calls `select(workflow, loops)` before it creates a
JAX backend (mu2efastsim.bench does, from --workflow; beamfastsim.cetracker does for W2), which adds or removes the
appended flag for that workflow. A value the user already put in XLA_FLAGS for the same flag is left alone. A
select after a JAX backend exists cannot change the setting: it is recorded as `late` (and warned about when it would
have changed it). `state()` (recorded by `precision.provenance()`) gives the setting, the workflow, whether the flag
is set and where it came from.

Concurrent graph regions: MU2EFS_GRAPH_CONCURRENCY=1 (default) or 0, independent of the device-loop setting and of
the workflow. By default XLA records the kernels of a command buffer as a chain, one after another. With the setting
on, `apply()` also appends `--xla_gpu_graph_enable_concurrent_region=true`: kernels that do not depend on each other
become parallel branches of the CUDA graph, so many small kernels (a W6 batch step launches tens of thousands, most of
them a few microseconds long) share the GPU. XLA keeps its own dependencies, so the results do not change (W6 mixing
outputs identical with and without it over full passes, every capacity file and dtype). A value the user already put
in XLA_FLAGS for this flag is left alone; `concurrency_state()` is its provenance record.

Concurrent regions together with device loops hang some chains: the W2 CE chain (transport while loop in
cechain_jax.run) waits forever on a device value on A100 GPUs with both on, in both dtypes, while either one alone
completes. NO_CONCURRENCY_WITH_LOOPS lists those workflows: when `select` leaves device loops on for one of them (forced
with --device-loops on or MU2EFS_DEVICE_LOOPS=1, or a user value of the loop flag), it also removes the appended
concurrent-region flag (warned when MU2EFS_GRAPH_CONCURRENCY was set explicitly; a user value in XLA_FLAGS is left
alone). Their defaults (W2: loops off) keep concurrency. Standard library only.
"""
import os
import warnings

from .determinism import _backend_started

FLAG = "--xla_gpu_enable_command_buffer"
VALUE = "+CONDITIONAL,+WHILE"
# device loops per workflow in auto mode (measured on A100 GPUs, same card, in sequence); unlisted workflows: on
WORKFLOWS = {"W2": False, "W6": True, "mixing": True}
# workflows whose chains hang with device loops and concurrent graph regions both on (A100): select drops concurrency
NO_CONCURRENCY_WITH_LOOPS = {"W2"}
CONCURRENCY_FLAG = "--xla_gpu_graph_enable_concurrent_region"
CONCURRENCY_VALUE = "true"
_STATE = {}
_CONCURRENCY = {}


def setting():
    """MU2EFS_DEVICE_LOOPS: "auto" (default or empty), "on" (1/true/on/yes) or "off" (0/false/off/no)."""
    value = os.environ.get("MU2EFS_DEVICE_LOOPS", "auto").strip().lower()
    if value in ("", "auto"):
        return "auto"
    if value in ("1", "true", "on", "yes"):
        return "on"
    if value in ("0", "false", "off", "no"):
        return "off"
    raise ValueError(f"MU2EFS_DEVICE_LOOPS must be auto, 1 or 0, not {value!r}")


def requested():
    """Whether device loops are requested at import (before any workflow is named): on unless the setting is off."""
    return setting() != "off"


def wanted(workflow=None, loops=None):
    """Device loops for a workflow: loops ("on"/"off", True/False) first, then an explicit MU2EFS_DEVICE_LOOPS, then
    WORKFLOWS (unlisted or None: on)."""
    if loops not in (None, "auto"):
        if loops in (True, "on", "1"):
            return True
        if loops in (False, "off", "0"):
            return False
        raise ValueError(f"device loops must be on, off or auto, not {loops!r}")
    s = setting()
    if s != "auto":
        return s == "on"
    return WORKFLOWS.get(workflow, True)


def _flags():
    return os.environ.get("XLA_FLAGS", "")


def _set(on):
    """Add or remove the appended flag token (never a value the user set)."""
    tok = FLAG + "=" + VALUE
    parts = [p for p in _flags().split() if p != tok]
    if on:
        parts.append(tok)
    os.environ["XLA_FLAGS"] = " ".join(parts)


def concurrency_requested():
    """MU2EFS_GRAPH_CONCURRENCY: on unless "0" (or "false"/"off"/"no")."""
    value = os.environ.get("MU2EFS_GRAPH_CONCURRENCY", "1").strip().lower()
    if value not in ("1", "0", "true", "false", "on", "off", "yes", "no", ""):
        raise ValueError(f"MU2EFS_GRAPH_CONCURRENCY must be 1 or 0, not {value!r}")
    return value not in ("0", "false", "off", "no")


def _apply_concurrency():
    if _CONCURRENCY:
        return
    on, user = concurrency_requested(), CONCURRENCY_FLAG in _flags()
    if on and not user:
        os.environ["XLA_FLAGS"] = " ".join(_flags().split() + [CONCURRENCY_FLAG + "=" + CONCURRENCY_VALUE])
    _CONCURRENCY.update(requested=on, flag_source="user" if user else ("setting" if on else "none"),
                        late=bool(on and not user and _backend_started()))


def concurrency_state():
    """Concurrent-region record for provenance: the setting, where the flag came from, the flag value XLA sees now,
    and `late` (appended after a JAX backend already existed, so not in effect for this process)."""
    s = dict(_CONCURRENCY) if _CONCURRENCY else {"requested": concurrency_requested(), "flag_source": "not applied",
                                                 "late": False}
    value = None
    for tok in _flags().split():
        if tok.startswith(CONCURRENCY_FLAG):
            value = tok.partition("=")[2]
    s["xla_gpu_graph_enable_concurrent_region"] = value
    return s


def _loops_in_flags():
    """Whether XLA_FLAGS records loops or conditionals in command buffers (our token or a user value)."""
    for tok in _flags().split():
        if tok.startswith(FLAG + "="):
            value = tok.partition("=")[2].upper()
            return "WHILE" in value or "CONDITIONAL" in value
    return False


def _guard_concurrency(workflow):
    """Before the first backend: drop the appended concurrent-region flag when device loops stay on for a workflow in
    NO_CONCURRENCY_WITH_LOOPS (it hangs). A user value of the concurrency flag is left alone."""
    tok = CONCURRENCY_FLAG + "=" + CONCURRENCY_VALUE
    if workflow not in NO_CONCURRENCY_WITH_LOOPS or not _loops_in_flags() or tok not in _flags().split():
        return
    if _CONCURRENCY.get("flag_source") == "user":
        return
    if os.environ.get("MU2EFS_GRAPH_CONCURRENCY", "").strip():
        warnings.warn(f"concurrent graph regions are off for {workflow} with device loops on (the chain hangs with "
                      "both); MU2EFS_GRAPH_CONCURRENCY is ignored", RuntimeWarning, stacklevel=3)
    os.environ["XLA_FLAGS"] = " ".join(p for p in _flags().split() if p != tok)
    _CONCURRENCY.update(flag_source="none", dropped_for=workflow)


def apply():
    """Append the device-loop flag to XLA_FLAGS unless the setting is off, and the concurrent-region flag unless
    MU2EFS_GRAPH_CONCURRENCY is off (once per process)."""
    _apply_concurrency()
    if _STATE:
        return dict(_STATE)
    user = FLAG in _flags()
    on = requested()
    if on and not user:
        _set(True)
    _STATE.update(setting=setting(), workflow=None, requested=on,
                  flag_source="user" if user else ("setting" if on else "none"),
                  late=bool(on and not user and _backend_started()))
    return dict(_STATE)


def select(workflow=None, loops=None):
    """The workflow's device-loop setting (wanted), before the first JAX backend: adds or removes the flag apply()
    appended. Returns the state. The first select of a process decides; a later one changes it only with an explicit
    loops (on/off). After a backend exists nothing changes (late; a warning if it would have). A workflow in
    NO_CONCURRENCY_WITH_LOOPS that keeps device loops also loses the appended concurrent-region flag."""
    apply()
    if loops in (None, "auto") and "decided_by" in _STATE:      # an earlier select (e.g. the entry point's) stands
        return dict(_STATE)
    on = wanted(workflow, loops)
    src = "flag" if loops not in (None, "auto") else ("setting" if setting() != "auto" else "workflow")
    if _STATE["flag_source"] == "user":
        _STATE.update(workflow=workflow)
        if not _backend_started():
            _guard_concurrency(workflow)
        return dict(_STATE)
    if _backend_started():
        current = _loops_in_flags()
        if on != current:
            warnings.warn(f"device loops for {workflow or 'this workflow'} would be {'on' if on else 'off'}, but a "
                          "JAX backend already exists: they stay as they are in this process (select before any "
                          "JAX computation)", RuntimeWarning, stacklevel=2)
        if (workflow in NO_CONCURRENCY_WITH_LOOPS and current
                and (CONCURRENCY_FLAG + "=" + CONCURRENCY_VALUE) in _flags().split()):
            warnings.warn(f"{workflow} runs with device loops and concurrent graph regions both on, which hangs: "
                          "select before any JAX computation, or set MU2EFS_GRAPH_CONCURRENCY=0",
                          RuntimeWarning, stacklevel=2)
        _STATE.update(workflow=workflow, late=on != current)
        return dict(_STATE)
    _set(on)
    _guard_concurrency(workflow)
    _STATE.update(workflow=workflow, requested=on, flag_source="setting" if on else "none", decided_by=src,
                  late=False)
    return dict(_STATE)


def state():
    """Device-loop record for provenance: the setting (auto/on/off), the workflow named by select, whether loops are
    requested, where the flag came from (and, after select, what decided it: workflow, setting or flag), the flag
    value XLA sees now, and `late` (the requested setting could not take
    effect because a JAX backend already existed)."""
    s = dict(_STATE) if _STATE else {"setting": setting(), "workflow": None, "requested": requested(),
                                     "flag_source": "not applied", "late": False}
    value = None
    for tok in _flags().split():
        if tok.startswith(FLAG):
            value = tok.partition("=")[2]
    s["xla_gpu_enable_command_buffer"] = value
    return s
