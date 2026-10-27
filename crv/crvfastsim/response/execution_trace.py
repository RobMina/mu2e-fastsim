"""Opt-in, synchronous Python-call evidence for the existing response backends."""
from collections import Counter
import _thread
import sys
import threading


_PREFIX = "crvfastsim.response."
_THREAD_STARTERS = tuple(getattr(_thread, name) for name in
                         ("start_new_thread", "start_new", "start_joinable_thread")
                         if hasattr(_thread, name))
_TARGETS = {
    "photons": {"make_photons", "_substep_photons"},
    "photons_jax": {"make_photons", "substep_photons", "device_photons"},
    "sipm": {"simulate", "_simulate_chunk"},
    "sipm_jax": {"simulate", "run_device_sipm"},
    "device": {"response_batched", "response"},
}
_REQUIRED = {
    "numpy": ("photons.make_photons", "photons._substep_photons",
              "sipm.simulate", "sipm._simulate_chunk"),
    "jax": ("photons_jax.make_photons", "photons_jax.substep_photons",
            "sipm_jax.simulate", "sipm_jax.run_device_sipm"),
    "fused": ("device.response_batched", "device.response",
              "photons_jax.device_photons", "sipm_jax.run_device_sipm"),
}
_LIMITATIONS = (
    "Python call/dispatch evidence, not execution authentication or physics validation.",
    "JIT bodies, native workers and every device operation are not instrumented.",
    "Argument device placement is not proof of every operation's architecture.",
    "Only synchronous execution in the main Python thread is supported.",
    "No-SiPM workloads cannot establish absence of fallback.",
)


def _symbol(frame):
    module_name = frame.f_globals.get("__name__", "")
    if not module_name.startswith(_PREFIX):
        return None
    module = module_name[len(_PREFIX):]
    name = frame.f_code.co_name
    if name not in _TARGETS.get(module, ()):
        return None
    loaded = sys.modules.get(module_name)
    function = getattr(loaded, name, None)
    if getattr(function, "__code__", None) is not frame.f_code:
        return None
    return f"{module}.{name}"


def _device_record(device):
    return {"id": int(device.id), "platform": str(device.platform),
            "device_kind": str(device.device_kind), "description": str(device)}


def _has_workers():
    return len(threading.enumerate()) != 1 or _thread._count() != 0


class ExecutionTrace:
    """Single-use context; inspect summary or require_backend after leaving it."""

    def __init__(self, backend):
        if not isinstance(backend, str) or backend not in _REQUIRED:
            raise ValueError("backend must be numpy, jax, or fused")
        self.backend = backend
        self._calls = Counter()
        self._photon_workload_calls = 0
        self._sipm_workload_calls = 0
        self._unsupported = []
        self._entered = False
        self._closed = False
        self._completed = False
        self._callback = self._profile
        self._jax = None
        self._available_devices = []
        self._observed_devices = {}
        self._owner = None

    def __enter__(self):
        if sys.implementation.name != "cpython" or not hasattr(_thread, "_count"):
            raise ValueError("execution tracing requires CPython thread/profile support")
        if self._entered:
            raise ValueError("execution trace contexts are single-use")
        if threading.current_thread() is not threading.main_thread():
            raise ValueError("execution tracing requires the main Python thread")
        if sys.getprofile() is not None or threading.getprofile() is not None:
            raise ValueError("execution tracing rejects nested or existing profilers")
        if _has_workers():
            names = [t.name for t in threading.enumerate() if t is not threading.main_thread()]
            raise ValueError(f"execution tracing does not support active Python worker threads: {names}")
        if self.backend != "numpy":
            import jax
            self._jax = jax
            self._available_devices = [_device_record(d) for d in jax.devices()]
        if sys.getprofile() is not None or threading.getprofile() is not None:
            raise ValueError("execution tracing rejects nested or existing profilers")
        if _has_workers():
            raise ValueError("execution tracing does not support active Python worker threads")
        self._owner = threading.get_ident()
        self._entered = True
        sys.setprofile(self._callback)
        return self

    def _profile(self, frame, event, arg):
        if ((event == "call" and frame.f_code is threading.Thread.start.__code__)
                or (event == "c_call" and any(arg is start for start in _THREAD_STARTERS))):
            self._unsupported.append("Python thread creation attempted during observation")
            raise ValueError(self._unsupported[-1])
        if event != "call":
            return
        symbol = _symbol(frame)
        if symbol is None:
            return
        self._calls[symbol] += 1
        values = frame.f_locals
        if symbol in {"photons._substep_photons", "photons_jax.substep_photons",
                      "photons_jax.device_photons"}:
            if len(values.get("rows", {}).get("nsub", ())):
                self._photon_workload_calls += 1
        if symbol == "sipm._simulate_chunk" and len(values.get("chans", ())):
            self._sipm_workload_calls += 1
        if symbol == "sipm_jax.run_device_sipm":
            parent = frame.f_back
            caller = _symbol(parent) if parent is not None else None
            count_name = {"device.response": "nv", "sipm_jax.simulate": "n"}.get(caller)
            if count_name and parent.f_locals.get(count_name, 0) > 0:
                self._sipm_workload_calls += 1
        if self._jax is not None:
            arrays = [values.get("key"), getattr(values.get("dt"), "prob", None)]
            arrays.extend(values.get("args", ())[:5])
            for array in arrays:
                if isinstance(array, self._jax.Array):
                    for device in array.devices():
                        record = _device_record(device)
                        self._observed_devices[(record["platform"], record["id"])] = record

    def __exit__(self, exc_type, exc, tb):
        if threading.get_ident() != self._owner:
            raise ValueError("execution trace must exit on its owning thread")
        current = sys.getprofile()
        changed = current is not self._callback
        if current is self._callback:
            sys.setprofile(None)
        if changed and not self._unsupported:
            self._unsupported.append("profiler changed during observation")
        if threading.getprofile() is not None or _has_workers():
            self._unsupported.append("threaded execution or thread profiler detected")
        self._closed = True
        self._completed = exc_type is None
        if self._unsupported and exc_type is None:
            raise ValueError("; ".join(self._unsupported))
        return False

    def summary(self):
        """Return JSON-safe evidence; empty or partial paths are never confirmed."""
        calls = dict(sorted(self._calls.items()))
        missing = [name for name in _REQUIRED[self.backend] if not calls.get(name)]
        fallback = self.backend != "numpy" and bool(calls.get("sipm.simulate"))
        unexpected = []
        for name in calls:
            if self.backend == "numpy" and name.split(".")[0] in {"photons_jax", "sipm_jax", "device"}:
                unexpected.append(name)
            elif self.backend != "numpy" and name in {"photons.make_photons", "photons._substep_photons"}:
                unexpected.append(name)
            elif self.backend == "jax" and name.startswith("device."):
                unexpected.append(name)
            elif self.backend == "fused" and name in {"photons_jax.make_photons", "photons_jax.substep_photons", "sipm_jax.simulate"}:
                unexpected.append(name)
        reasons = list(self._unsupported)
        if not self._closed:
            reasons.append("observation context has not finished")
        elif not self._completed:
            reasons.append("observation context exited with an exception")
        if missing:
            reasons.append("missing required response calls: " + ", ".join(missing))
        if not self._photon_workload_calls:
            reasons.append("no nonempty photon-row workload observed")
        if not self._sipm_workload_calls:
            reasons.append("no nonempty SiPM workload observed; fallback coverage unsupported")
        if fallback:
            reasons.append("NumPy sipm.simulate called during requested JAX/fused execution")
        if unexpected:
            reasons.append("unexpected backend path: " + ", ".join(unexpected))
        confirmed = not reasons
        status = ("unsupported" if self._unsupported else "fallback" if fallback else
                  "confirmed" if confirmed else "insufficient_evidence")
        evidence = {
            "schema_version": 1, "observer": "python_sys_setprofile",
            "requested_backend": self.backend, "status": status,
            "backend_confirmed": confirmed, "calls": calls,
            "required_calls": list(_REQUIRED[self.backend]), "missing_calls": missing,
            "unexpected_calls": unexpected, "fallback_detected": fallback,
            "fallback_observation": ("not_applicable" if self.backend == "numpy" else
                                     "observed" if fallback else
                                     "not_observed_in_observed_calls" if confirmed else "insufficient_evidence"),
            "photon_workload_calls": self._photon_workload_calls,
            "sipm_workload_calls": self._sipm_workload_calls,
            "completed": self._closed and self._completed,
            "unsupported": list(self._unsupported), "reasons": reasons,
            "limitations": list(_LIMITATIONS),
        }
        if self.backend != "numpy":
            evidence["jax"] = {
                "version": self._jax.__version__ if self._jax is not None else None,
                "available_devices": [dict(d) for d in self._available_devices],
                "observed_argument_devices": [dict(d) for _, d in sorted(self._observed_devices.items())],
                "placement_evidence": "argument placement only; not per-operation execution",
            }
        return evidence

    def require_backend(self):
        """Return confirmed evidence, otherwise raise ValueError with the reasons."""
        evidence = self.summary()
        if not evidence["backend_confirmed"]:
            raise ValueError("response execution trace: " + "; ".join(evidence["reasons"]))
        return evidence


def observe(backend):
    """Observe numpy, nonfused jax, or fused response calls without changing kernels."""
    return ExecutionTrace(backend)
