"""GPU command-buffer settings chosen per workflow: device-side loops (MU2EFS_DEVICE_LOOPS) and concurrent graph
regions (MU2EFS_GRAPH_CONCURRENCY), each auto (default), 1 or 0.

Device loops. XLA records most GPU kernels of a compiled program into command buffers (CUDA graphs), but by default it
runs a `lax.while_loop` or `lax.cond` from the host: after each iteration the predicate is copied to the host, which
then launches the next iteration. With device loops on, `--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE` is
appended to XLA_FLAGS: XLA keeps its default command-buffer types and also records loops and conditionals, which then
run on the device (CUDA conditional nodes). The calorimeter device chain compiles its chains without command buffers
(a per-jit XLA option; its clustering gave wrong clusters with device loops on A100 GPUs).

Concurrent graph regions. By default XLA records the kernels of a command buffer as a chain, one after another. With
them on, `--xla_gpu_graph_enable_concurrent_region=true` is appended: kernels that do not depend on each other become
parallel branches of the CUDA graph, so many small kernels share the GPU.

Both change how kernels are launched, not what they compute (outputs compared per jitted call against command buffers
off and the CPU for W1, W2, W5, W6, W7, W8, both dtypes; W6 output files identical with and without concurrency). Both
have no effect on CPU backends. Which is faster depends on the workflow, so in auto each follows WORKFLOWS, built from
same-job A100-80 rates (warm events/s over 2 warm passes, cold compile cache):

    workflow   loops on / off, f64 | f32               loops   concurrency
    W1         624 / 608 | 869 / 844 (+3 %)              on      off
    W2         76 / 101 | 35 / 42 (5,000 / pass);        off     off
               200k / pass: 1,865 / 2,087 | 2,036 / 2,424
    W3, W4     W2's CE chain                             off     off
    W5 plain   2,810 / 2,989 | 2,762 / 2,823 (-2..-6 %)   off     off
    W5 refit   3,119 / 3,132 | 3,040 / 3,045 (~0)        off     off  (same "W5" entry)
    W6         279 / 260 | 295 / 272 (+7..+8 %)          on      on (x1.03-1.07, output files identical)
    W7         3,551 / 3,557 | 5,459 / 5,460 (0)         off     off
    W8         7,013 / 7,848 | 7,509 / 8,600 (-11..-13 %) off     off

Concurrency is on only where it is measured faster and its outputs were checked (W6). A workflow not in the table, and
a process that names none, gets both off. The CE chain (W2, W3, W4: the transport while loop in cechain_jax.run) hangs
on A100 GPUs with device loops and concurrency both on, in both dtypes, while either alone completes: for those
workflows the combination cannot be selected (CE_CHAIN): when device loops are on, by any override, concurrency is
turned off with a warning.

Overrides, per setting: select's `loops` / `concurrency` argument (an entry point's --device-loops /
--graph-concurrency on|off), then MU2EFS_DEVICE_LOOPS / MU2EFS_GRAPH_CONCURRENCY = 1 or 0, then the table. A value
the user put in XLA_FLAGS for either flag is left alone (except the CE-chain combination above).

XLA reads XLA_FLAGS when JAX creates its first backend. `apply()` runs when the `mu2efastsim` package is imported (as
for determinism.py): it appends a flag only when its setting is forced on (1). A workflow entry point then calls
`select(workflow)` before it creates a JAX backend (mu2efastsim.bench does, from --workflow; beamfastsim.cetracker
for W2; beamfastsim.tsdevice for W1; mixing.device_backend for W6). The first select of a process decides; a later
one changes a setting only with an explicit argument. A select after a JAX backend exists cannot change anything: it
is recorded as `late`, with a warning when it would have. `state()` and `concurrency_state()` (recorded by
`precision.provenance()` and the bench report) give the setting, the workflow, what decided it and the flag value XLA
sees. Standard library only.
"""
import os
import warnings

from .determinism import _backend_started

FLAG = "--xla_gpu_enable_command_buffer"
VALUE = "+CONDITIONAL,+WHILE"
CONCURRENCY_FLAG = "--xla_gpu_graph_enable_concurrent_region"
CONCURRENCY_VALUE = "true"
# auto mode, per workflow: (device loops, concurrent graph regions); measured numbers in the module docstring.
# Unlisted workflows and None: both off.
WORKFLOWS = {
    "W1": (True, False),
    "W2": (False, False),
    "W3": (False, False),
    "W4": (False, False),
    "W5": (False, False),
    "W6": (True, True),
    "mixing": (True, True),
    "W7": (False, False),
    "W8": (False, False),
}
# workflows that run the CE chain, which hangs with device loops and concurrency both on
CE_CHAIN = {"W2", "W3", "W4"}
_STATE = {}
_CONCURRENCY = {}


def _parse(name, value):
    value = value.strip().lower()
    if value in ("", "auto"):
        return "auto"
    if value in ("1", "true", "on", "yes"):
        return "on"
    if value in ("0", "false", "off", "no"):
        return "off"
    raise ValueError(f"{name} must be auto, 1 or 0, not {value!r}")


def setting():
    """MU2EFS_DEVICE_LOOPS: "auto" (default or empty), "on" (1/true/on/yes) or "off" (0/false/off/no)."""
    return _parse("MU2EFS_DEVICE_LOOPS", os.environ.get("MU2EFS_DEVICE_LOOPS", "auto"))


def concurrency_setting():
    """MU2EFS_GRAPH_CONCURRENCY: "auto" (default or empty), "on" or "off" (same spellings)."""
    return _parse("MU2EFS_GRAPH_CONCURRENCY", os.environ.get("MU2EFS_GRAPH_CONCURRENCY", "auto"))


def requested():
    """Device loops at import (before any workflow is named): on only when MU2EFS_DEVICE_LOOPS forces them."""
    return setting() == "on"


def concurrency_requested():
    """Concurrency at import: on only when MU2EFS_GRAPH_CONCURRENCY forces it."""
    return concurrency_setting() == "on"


def _forced(name, value):
    if value in (None, "auto"):
        return None
    if value in (True, "on", "1"):
        return True
    if value in (False, "off", "0"):
        return False
    raise ValueError(f"{name} must be on, off or auto, not {value!r}")


def wanted(workflow=None, loops=None):
    """Device loops for a workflow: loops ("on"/"off", True/False) first, then an explicit MU2EFS_DEVICE_LOOPS, then
    WORKFLOWS (unlisted or None: off)."""
    f = _forced("device loops", loops)
    if f is not None:
        return f
    s = setting()
    return s == "on" if s != "auto" else WORKFLOWS.get(workflow, (False, False))[0]


def concurrency_wanted(workflow=None, concurrency=None):
    """Concurrency for a workflow, before the CE-chain rule: the argument, then MU2EFS_GRAPH_CONCURRENCY, then
    WORKFLOWS (unlisted or None: off)."""
    f = _forced("graph concurrency", concurrency)
    if f is not None:
        return f
    s = concurrency_setting()
    return s == "on" if s != "auto" else WORKFLOWS.get(workflow, (False, False))[1]


def _flags():
    return os.environ.get("XLA_FLAGS", "")


def _value(flag):
    value = None
    for tok in _flags().split():
        if tok.startswith(flag + "="):
            value = tok.partition("=")[2]
    return value


def _put(flag, value, on):
    """Add or remove our own token of a flag (never a value the user set)."""
    tok = flag + "=" + value
    parts = [p for p in _flags().split() if p != tok]
    if on:
        parts.append(tok)
    os.environ["XLA_FLAGS"] = " ".join(parts)


def _loops_on():
    """Whether XLA_FLAGS records loops or conditionals in command buffers (our token or a user value)."""
    v = (_value(FLAG) or "").upper()
    return "WHILE" in v or "CONDITIONAL" in v


def _concurrency_on():
    return (_value(CONCURRENCY_FLAG) or "").lower() in ("true", "1")


def _set(on):
    _put(FLAG, VALUE, on)


def apply():
    """At package import (once per process): append each flag only when its setting forces it on."""
    if _STATE:
        return dict(_STATE)
    for flag, value, on, rec in ((FLAG, VALUE, requested(), _STATE),
                                 (CONCURRENCY_FLAG, CONCURRENCY_VALUE, concurrency_requested(), _CONCURRENCY)):
        user = flag in _flags()
        if on and not user:
            _put(flag, value, True)
        rec.update(workflow=None, requested=on, flag_source="user" if user else ("setting" if on else "none"),
                   late=bool(on and not user and _backend_started()))
    _STATE.update(setting=setting())
    _CONCURRENCY.update(setting=concurrency_setting())
    return dict(_STATE)


def _source(arg, env):
    return "flag" if arg not in (None, "auto") else ("setting" if env != "auto" else "workflow")


def select(workflow=None, loops=None, concurrency=None):
    """The workflow's device-loop and concurrency settings, before the first JAX backend: adds or removes the flags.
    Returns the device-loop state. The first select of a process decides; a later one changes a setting only with an
    explicit argument for it. For a CE-chain workflow with device loops on, concurrency is turned off (it hangs).
    After a backend exists nothing changes (late; warnings when it would have, or when the CE chain runs with both)."""
    apply()
    redo_loops = _forced("device loops", loops) is not None or "decided_by" not in _STATE
    redo_conc = _forced("graph concurrency", concurrency) is not None or "decided_by" not in _CONCURRENCY
    if not (redo_loops or redo_conc):           # an earlier select (e.g. the entry point's) stands
        return dict(_STATE)
    on = wanted(workflow, loops) if redo_loops else _loops_on()
    conc = concurrency_wanted(workflow, concurrency) if redo_conc else _concurrency_on()
    if _backend_started():
        now_loops, now_conc = _loops_on(), _concurrency_on()
        if (on, conc) != (now_loops, now_conc):
            warnings.warn(f"device loops / graph concurrency for {workflow or 'this workflow'} would be "
                          f"{'on' if on else 'off'} / {'on' if conc else 'off'}, but a JAX backend already exists: "
                          "they stay as they are in this process (select before any JAX computation)",
                          RuntimeWarning, stacklevel=2)
        if workflow in CE_CHAIN and now_loops and now_conc:
            warnings.warn(f"{workflow} runs with device loops and graph concurrency both on, which hangs the CE "
                          "chain: select before any JAX computation, or set MU2EFS_GRAPH_CONCURRENCY=0",
                          RuntimeWarning, stacklevel=2)
        if redo_loops:
            _STATE.update(workflow=workflow, late=on != now_loops)
        if redo_conc:
            _CONCURRENCY.update(workflow=workflow, late=conc != now_conc)
        return dict(_STATE)
    if redo_loops:
        if _STATE["flag_source"] != "user":
            _set(on)
            _STATE.update(requested=on, flag_source="setting" if on else "none")
        _STATE.update(workflow=workflow, decided_by=_source(loops, setting()), late=False)
    if redo_conc:
        if _CONCURRENCY["flag_source"] != "user":
            _put(CONCURRENCY_FLAG, CONCURRENCY_VALUE, conc)
            _CONCURRENCY.update(requested=conc, flag_source="setting" if conc else "none")
        _CONCURRENCY.update(workflow=workflow, decided_by=_source(concurrency, concurrency_setting()), late=False)
    _CONCURRENCY.pop("dropped_for", None)
    if workflow in CE_CHAIN and _loops_on() and _concurrency_on():
        warnings.warn(f"graph concurrency is off for {workflow} with device loops on (the CE chain hangs with both)",
                      RuntimeWarning, stacklevel=2)
        os.environ["XLA_FLAGS"] = " ".join(p for p in _flags().split() if not p.startswith(CONCURRENCY_FLAG + "="))
        _CONCURRENCY.update(requested=False, flag_source="none", dropped_for=workflow)
    return dict(_STATE)


def state():
    """Device-loop record for provenance: the setting (auto/on/off), the workflow named by select, whether loops are
    requested, where the flag came from (and, after select, what decided it: workflow, setting or flag), the flag
    value XLA sees now, and `late` (the requested setting could not take effect because a JAX backend existed)."""
    s = dict(_STATE) if _STATE else {"setting": setting(), "workflow": None, "requested": requested(),
                                     "flag_source": "not applied", "late": False}
    s["xla_gpu_enable_command_buffer"] = _value(FLAG)
    return s


def concurrency_state():
    """Concurrency record for provenance, as state(), plus `dropped_for` when the CE-chain rule turned it off."""
    s = dict(_CONCURRENCY) if _CONCURRENCY else {"setting": concurrency_setting(), "workflow": None,
                                                 "requested": concurrency_requested(), "flag_source": "not applied",
                                                 "late": False}
    s["xla_gpu_graph_enable_concurrent_region"] = _value(CONCURRENCY_FLAG)
    return s
