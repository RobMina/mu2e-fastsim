"""Fail-closed audits of retained setup data, not of per-event output.

Host fingerprints describe content, not allocation identity or pandas caches.
JAX arrays are the deliberate exception: their immutable, synchronized buffers
must remain the *same retained arrays*, with the same shape/dtype/placement.
There is no device-to-host conversion. Keep the context alive across audits;
these fingerprints are not portable device-content hashes.

Freezing is a guard, not a security boundary: Python containers and pandas
columns remain replaceable, and pre-existing writable aliases may still exist.
Compare fingerprints around use to detect content changes. Equal-content host
replacement is intentionally equivalent. Neither operation deep-copies data.
"""

import dataclasses
import hashlib
import json
import struct
import sys
import types

import numpy as np
import pandas as pd

__all__ = ["freeze_arrays", "content_fingerprint"]


def _typename(value):
    cls = type(value)
    return cls.__module__ + "." + cls.__qualname__


def _numpy_dtype(dtype):
    if not isinstance(dtype, np.dtype) or dtype.fields or dtype.subdtype or dtype.metadata:
        raise TypeError(f"unsupported context dtype: {dtype!r}")
    if dtype.kind not in "biufcSUO":
        raise TypeError(f"unsupported context dtype: {dtype!r}")
    return dtype.str


def _pandas_dtype(dtype):
    # Admit primitive pandas strings, not arbitrary extension arrays or nullable numerics.
    if isinstance(dtype, pd.StringDtype):
        return repr(dtype)
    return _numpy_dtype(dtype)


def _object_json(array):
    # Vectorized validation restricts object buffers to string labels rather than pointer bytes.
    inferred = pd.api.types.infer_dtype(array.ravel(), skipna=True)
    if inferred not in ("string", "empty"):
        raise TypeError(f"unsupported context object array: {inferred}")
    try:
        return json.dumps(array.tolist(), ensure_ascii=True, separators=(",", ":"),
                          allow_nan=True).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise TypeError("unsupported context object array; expected string labels") from exc


def _readonly(array):
    # Freeze backing buffers so future Series views inherit the read-only guard.
    current = array
    while isinstance(current, np.ndarray):
        current.setflags(write=False)
        current = current.base


class _Audit:
    def __init__(self, freeze):
        self.freeze = freeze
        self.active = set()
        # Retain temporary pandas views too: id reuse must not hit a stale memo.
        self.memo = {}

    def visit(self, value):
        ident = id(value)
        if ident in self.active:
            raise ValueError("cycle in prepared context")
        if ident in self.memo:
            return self.memo[ident][1]
        self.active.add(ident)
        try:
            result = self.node(value)
            self.memo[ident] = (value, result)
            return result
        finally:
            self.active.remove(ident)

    def node(self, value):
        h = None if self.freeze else hashlib.sha256()

        def emit(data):
            if h is not None:
                h.update(struct.pack(">Q", len(data)))
                h.update(data)

        def child(item):
            emit(self.visit(item))

        emit(_typename(value).encode("utf-8"))
        if value is None:
            pass
        elif type(value) is bool:
            emit(b"1" if value else b"0")
        elif type(value) is int:
            emit(str(value).encode("ascii"))
        elif type(value) is float:
            emit(struct.pack(">d", value))
        elif type(value) is complex:
            emit(struct.pack(">dd", value.real, value.imag))
        elif type(value) is str:
            emit(value.encode("utf-8", errors="surrogatepass"))
        elif type(value) is bytes:
            emit(value)
        elif self.is_jax_array(value):
            value.block_until_ready()
            # No np.asarray/device_get/tobytes: identity binds immutable buffers.
            child(id(value))
            child(tuple(value.shape))
            child(str(value.dtype))
            child(tuple(sorted((d.platform, int(d.process_index), int(d.id), d.device_kind)
                               for d in value.devices())))
        elif isinstance(value, np.generic):
            _numpy_dtype(value.dtype)
            child(np.asarray(value))
        elif type(value) is np.ndarray:
            child(_numpy_dtype(value.dtype))
            child(tuple(value.shape))
            if value.dtype.kind == "O":
                encoded = _object_json(value)
                emit(encoded)
            elif not self.freeze and value.size:
                # Canonicalize C order only when storage is non-contiguous.
                contiguous = value if value.flags.c_contiguous else np.ascontiguousarray(value)
                emit(memoryview(contiguous).cast("B"))
            if self.freeze:
                _readonly(value)
        elif isinstance(value, pd.DataFrame):
            child(value.index)
            child(value.columns)
            child(value.attrs)
            child(bool(value.flags.allows_duplicate_labels))
            for i in range(len(value.columns)):
                column = value.iloc[:, i]  # also handles duplicate column labels
                child(_pandas_dtype(column.dtype))
                child(self.pandas_array(column))
        elif isinstance(value, pd.Series):
            child(value.name)
            child(value.index)
            child(_pandas_dtype(value.dtype))
            child(self.pandas_array(value))
            child(value.attrs)
            child(bool(value.flags.allows_duplicate_labels))
        elif isinstance(value, pd.MultiIndex):
            child(tuple(value.names))
            child(tuple(value.levels))
            child(tuple(value.codes))
            child(value.sortorder)
        elif isinstance(value, pd.Index):
            child(value.name)
            child(_pandas_dtype(value.dtype))
            child(self.pandas_array(value))
        elif isinstance(value, dict):
            self.check_slots(value)
            child(len(value))
            for key, item in value.items():
                child(key)
                child(item)
            if type(value) is not dict:
                child(vars(value) if hasattr(value, "__dict__") else {})
        elif isinstance(value, (tuple, list)):
            self.check_slots(value)
            child(len(value))
            for item in value:
                child(item)
            if hasattr(value, "__dict__"):
                child(vars(value))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            # Do not use asdict(): it copies arrays and misses dynamic attrs.
            fields = dataclasses.fields(value)
            self.check_slots(value, {f.name for f in fields})
            for field in fields:
                child(field.name)
                child(getattr(value, field.name))
            if hasattr(value, "__dict__"):
                child(vars(value))  # includes geom.lib_group and future attrs
        elif (hasattr(value, "__dict__") and not callable(value)
              and not isinstance(value, (types.ModuleType, type))
              and (type(value) is types.SimpleNamespace
                   or type(value).__new__ is object.__new__)):
            self.check_slots(value)
            child(vars(value))
        else:
            raise TypeError(f"unsupported context type: {_typename(value)}")
        return b"" if h is None else h.digest()

    def pandas_array(self, value):
        array = value.to_numpy(copy=False)
        if isinstance(value.dtype, pd.StringDtype) and value.dtype.na_value is pd.NA:
            # Normalize missing string labels while freezing the original buffer too.
            if self.freeze:
                _readonly(array)
            return value.to_numpy(copy=False, na_value=None)
        return array

    @staticmethod
    def is_jax_array(value):
        # Do not import JAX or initialize a backend for host-only contexts.
        jax = sys.modules.get("jax")
        return jax is not None and isinstance(value, jax.Array)

    @staticmethod
    def check_slots(value, fields=()):
        # Hidden slot state on an otherwise simple object must not be ignored.
        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            # Native dict subclasses may retain C-level state outside __dict__/__slots__.
            hidden = (name for name, item in cls.__dict__.items()
                      if isinstance(item, types.MemberDescriptorType))
            if any(s not in ("__dict__", "__weakref__") and s not in fields
                   for s in (*slots, *hidden)):
                raise TypeError(f"unsupported context slots: {_typename(value)}")


def freeze_arrays(value):
    """Validate recursively and mark reachable NumPy buffers read-only in place.

    Return ``value`` for convenience. Cycles raise ValueError; unsupported
    values/dtypes raise TypeError. A failure may leave earlier buffers frozen.
    Containers are not replaced, and JAX arrays are only synchronized.
    """
    _Audit(freeze=True).visit(value)
    return value


def content_fingerprint(value):
    """Return an order-sensitive SHA256 hex digest of supported context data.

    Shared references are legal and equivalent to equal independent host
    values; cycles fail closed. NumPy writeability and pandas' internal caches
    are deliberately excluded, so freezing or index lookup cannot cause drift.
    """
    return _Audit(freeze=False).visit(value).hex()
