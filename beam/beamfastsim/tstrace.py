"""Columnar observer for explicitly selected transport rows."""
import numpy as np

STEP_SCHEMA = dict(row=(np.int64, ()), step=(np.int32, ()), x0=(float, (3,)),
                   proposed_x=(float, (3,)), p0=(float, (3,)), t0=(float, ()),
                   path0=(float, ()), h=(float, ()), pdg0=(np.int64, ()), valid=(bool, ()),
                   x1=(float, (3,)), p1=(float, (3,)), t1=(float, ()), path1=(float, ()),
                   pdg1=(np.int64, ()), fate=(np.int8, ()))
MATERIAL_SCHEMA = dict(row=(np.int64, ()), step=(np.int32, ()), order=(np.int32, ()),
                       name=("U64", ()), requested_mm=(float, ()), fraction=(float, ()),
                       stopped=(bool, ()), p_before=(float, (3,)), p_after=(float, (3,)))


def _stack(chunks, schema):
    return {k: np.concatenate([c[k] for c in chunks]) if chunks else np.empty((0, *shape), dtype)
            for k, (dtype, shape) in schema.items()}


class Trace:
    """Collect copied selected-lane states, without RNG access or physics-state references."""

    def __init__(self, rows):
        self.rows = np.asarray(rows).copy()
        self._steps, self._materials = [], []
        self._current = None
        self._order = 0

    def begin(self, **state):
        if self._current is not None:
            raise ValueError("trace begin without end")
        self._current = {k: v.copy() for k, v in state.items()}
        self._order = 0

    def material(self, name, **state):
        if self._current is None:
            raise ValueError("material outside accepted step")
        index = np.searchsorted(self._current["row"], state["row"])
        if np.any(index >= len(self._current["row"])) or not np.array_equal(self._current["row"][index], state["row"]):
            raise ValueError("material rows absent from current step")
        if len(name) > 64:
            raise ValueError("material name exceeds trace schema")
        record = {k: v.copy() for k, v in state.items()}
        record.update(step=self._current["step"][index].copy(),
                      name=np.full(len(index), name, dtype="U64"),
                      order=np.full(len(index), self._order, dtype=np.int32))
        self._materials.append(record)
        self._order += 1

    def end(self, **state):
        if self._current is None:
            raise ValueError("trace end without begin")
        self._current.update({k: v.copy() for k, v in state.items()})
        if len(self._current["row"]):
            self._steps.append(self._current)
        self._current = None

    def arrays(self):
        if self._current is not None:
            raise ValueError("unfinished trace step")
        return {**{"step_" + k: v for k, v in _stack(self._steps, STEP_SCHEMA).items()},
                **{"material_" + k: v for k, v in _stack(self._materials, MATERIAL_SCHEMA).items()}}
