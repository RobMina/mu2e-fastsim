"""Observe two selected development muons while exactly replaying existing full-cohort runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from .tscrossings import TARGETS, reference_identity
from .tsreference import lookup
from .tsstudy import VARIANTS, cohort

SEEDS = (20261111, 20261112, 20261113)
LIMITS = [
    "Selected stock survivors are diagnostics, not a new rate sample.",
    "Full positive-source batch and original order preserve the saved RNG stream.",
    "Material rows record actual sampler call order, not geometric within-chord interval order.",
    "Requested column times sampled fraction is the transport's material-path accounting.",
    "Accepted step endpoints are fast states, not exact curved trajectories or stock histories.",
    "Cold-at-start lanes have terminal state but no accepted step rows.",
]
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
    """Columnar selected-lane observer with no RNG access and no physics-state references."""

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


def exact_replay(result, saved):
    """Compare every output and exposure array in original order, excluding only runtime."""
    if set(result) != set(saved):
        raise ValueError("replay output keys differ")
    unequal = [k for k in result if k != "seconds" and
               (result[k].dtype != saved[k].dtype or not np.array_equal(result[k], saved[k], equal_nan=True))]
    if unequal:
        raise ValueError("saved replay mismatch: " + ", ".join(unequal))
    return sorted(set(result) - {"seconds"})


def replay(paths, out):
    """Replay only the three preselected baseline seeds, refusing old output paths."""
    paths, out = [Path(p) for p in paths], Path(out)
    runs = [json.loads(p.read_text()) for p in paths]
    if not runs or len({r["seed"] for r in runs}) != len(runs):
        raise ValueError("nonempty unique saved runs required")
    for r in runs:
        if r["seed"] not in SEEDS or r["variant"] != "baseline" or r["settings"] != list(VARIANTS["baseline"]):
            raise ValueError("only the three selected existing baseline runs are authorized")
    if len({json.dumps(r["inputs"], sort_keys=True) for r in runs}) != 1:
        raise ValueError("saved inputs differ")
    if out.exists():
        raise ValueError("fresh trace output directory required")
    inputs = runs[0]["inputs"]
    r, _ = cohort(inputs["reference"])
    targets = np.asarray(TARGETS, dtype=np.int64)
    reference_identity(inputs["reference"], targets)
    rows = lookup(r["id"], targets)
    if np.any(rows < 0):
        raise ValueError("target absent from full positive cohort")
    order = np.lexsort(r["parent_shard_ids"].T[::-1])
    for key in ("parent_shard_ids", "parent_shard_event_counts"):
        r[key] = r[key][order]
    from . import tsfield, tsgeometry, tstransport
    field = tsfield.from_config(inputs["field_config"], inputs["offline_root"], inputs["maps"])
    geometry = tsgeometry.Geometry(inputs["gdml"])
    out.mkdir(parents=True, exist_ok=False)
    for path, run in zip(paths, runs):
        trace = Trace(rows)
        step, material, loss = VARIANTS["baseline"]
        f = tstransport.transport({k: r[k].copy() for k in ("x", "p", "t", "pdg")}, field, geometry,
                                 seed=run["seed"], step_mm=step, material_step_mm=material,
                                 loss_fraction=loss, trace=trace)
        f.update({k: r[k] for k in ("id", "shard", "event_id", "parent_shard_ids", "parent_shard_event_counts")})
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            checked = exact_replay(f, saved)
        arrays = trace.arrays()
        arrays.update(target_row=rows, target_id=targets)
        arrays.update({"terminal_" + k: f[k][rows] for k in
                       ("x", "p", "t", "pdg", "fate", "path", "steps", "material_path", "pion_decay")})
        stem = out / f"baseline-{run['seed']}"
        with stem.with_suffix(".npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        info = dict(seed=run["seed"], settings=run["settings"], inputs=inputs, saved_run=str(path),
                    exact_replay=True, exact_keys=checked, ninput=len(r["id"]), events=len(r["event_id"]),
                    seconds=float(f["seconds"]), target_id=targets.tolist(), target_row=rows.tolist(),
                    target_fates=[tstransport.FATES[i] for i in f["fate"][rows]],
                    step_rows=len(arrays["step_row"]), material_rows=len(arrays["material_row"]), limits=LIMITS)
        with stem.with_suffix(".json").open("x") as handle:
            json.dump(info, handle, indent=2, allow_nan=False)
        print(json.dumps({k: info[k] for k in ("seed", "exact_replay", "seconds", "target_fates", "step_rows", "material_rows")}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path, help="existing baseline run JSON files")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    replay(args.runs, args.out)


if __name__ == "__main__":
    main()
