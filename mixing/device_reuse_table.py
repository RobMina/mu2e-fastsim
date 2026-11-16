"""A saved donor-frame reuse table (mixing.device_reuse), made once and loaded by every job instead of recomputed.

The reuse precompute digitises every donor frame of the reused streams once per setup copy: about 75 s on an A100
but hours on one CPU core for the Run 1A pools. The table it returns depends only on
- the tracker seed (the configured `tracker_seed`; fresh-input passes change the per-batch seeds, not the setup's);
- the reused donor pools (their event IDs and tracker steps, hashed by content, so the paths do not matter);
- the tracker conditions;
- the reuse setup capacities (`precompute`, `frames_per_batch`, `precompute_rows` or the tracker row capacity,
  `streams`, `shifts`, `phases`, `periods`) and the tracker device float;
- the set of event time shifts (with `shifts: "events"`, one copy per distinct mc_pbtime) and the event length.

A saved table holds all of the first five in its key and refuses to load on any difference. It holds the copies'
shifts and the event length it was made for, and refuses a job with a time shift that is not one of its copies (to
1e-6 ns) or a longer event. Build it once for the union of the production's shifts (`build`, or the first job with
`reuse_table_shifts`), then point every job at it with `reuse_table`.

Only `shifts: "events"` tables are saved: with the phase grid a copy's shift depends on the job's mean shift.

The table's donor frames are stored by (pool, index in the pool), so a job with another signal pool loads it.
A shared table freezes each donor frame's setup draws over the whole production, as one job of that size would:
a frame drawn many times repeats the same digis. Rebuild the table after any change to the tracker digitiser.

    python3 -m mixing.device_reuse_table build CONFIG.json OUT.npz [--shifts S1,S2,...]

CONFIG.json is a `mixing.device_bench` configuration (its `out` is not used); the shifts are added to those of the
configuration's signal events.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

VERSION = 1                     # bump when the table's content or layout changes
SHIFT_TOLERANCE = 1e-6          # ns: a job's shift must equal one of the table's copy shifts to this
FIELDS = ("frame", "copy", "sid", "tdc", "tot", "adc", "pmp", "t_analog", "jitter", "dtime")


def _digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def pools_digest(data, pools):
    """sha256 of the reused donor pools: their event IDs, tracker step counts and tracker step columns."""
    trk = data["collections"]["tracker"]
    offsets, columns = trk["offsets"], trk["columns"]
    parts = []
    for k in pools:
        lo, hi = int(data["starts"][k]), int(data["starts"][k] + data["sizes"][k])
        if "ids" in data:
            parts.append(np.asarray(data["ids"][lo:hi], np.uint64))
        parts.append(np.diff(offsets[lo:hi + 1]).astype(np.int64))
        r0, r1 = int(offsets[lo]), int(offsets[hi])
        parts += [np.asarray(columns[name][r0:r1]) for name in sorted(columns)]
    return _digest(parts)


def conditions_digest(c):
    """sha256 of the tracker conditions (scalars and arrays)."""
    scalars = json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in c.s.items()}, sort_keys=True,
                         default=str)
    return hashlib.sha256(scalars.encode() + _digest([np.asarray(c.a[k]) for k in sorted(c.a)]).encode()).hexdigest()


def table_key(c, capacities, data, seed, rows_capacity):
    """What the reuse table depends on, apart from the event shifts and length (the saved table's key)."""
    from trkfastsim import devfloat
    from .device_reuse import ReuseCapacities
    caps = ReuseCapacities(**capacities)
    if caps.shifts != "events":
        raise ValueError("only tracker_reuse tables with shifts \"events\" can be saved")
    setup = {"precompute": caps.precompute, "frames_per_batch": caps.frames_per_batch,
             "rows": caps.precompute_rows or int(rows_capacity), "streams": caps.streams, "shifts": caps.shifts,
             "phases": caps.phases, "periods": caps.periods}
    return {"version": VERSION, "tracker_seed": int(seed), "tracker_float": devfloat.NAME,
            "pools": pools_digest(data, caps.reused_pools()), "conditions": conditions_digest(c),
            "setup": hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()}


def _small(a):
    """The smallest integer dtype that holds an integer array exactly (floats unchanged)."""
    a = np.asarray(a)
    if a.dtype.kind not in "iu" or not a.size:
        return a
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if a.min() >= info.min and a.max() <= info.max:
            return a.astype(dtype)
    return a


def save(path, table, key, starts, length):
    """Write the reuse table (device_reuse.precompute output) with its key, its donor frames as (pool, index in the
    pool), and the event length it was made for. Written to a temporary file, then renamed."""
    starts = np.asarray(starts, np.int64)
    frame = np.asarray(table["frame"], np.int64)
    pool = np.searchsorted(starts, frame, side="right") - 1
    if np.any(pool < 1):
        raise ValueError("reuse table frames must be donor frames")
    arrays = {name: _small(table[name]) for name in FIELDS if name != "frame"}
    arrays.update(pool=pool.astype(np.int8), local=_small(frame - starts[pool]),
                  copy_shifts=np.asarray(table["copy_shifts"], np.float64), shift=np.float64(table["shift"]),
                  phases=np.int64(table["phases"]), periods=np.int64(table["periods"]), length=np.float64(length),
                  key=np.array(json.dumps(key, sort_keys=True)))
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load(path, key, starts, event_shifts, length):
    """Read a saved reuse table for a job; refuse it if its key differs from the job's (table_key), if a job event
    shift is not one of its copy shifts, or if the job's events are longer than the table's."""
    with np.load(path, allow_pickle=False) as f:
        saved = json.loads(str(f["key"]))
        bad = sorted(name for name in set(saved) | set(key) if saved.get(name) != key.get(name))
        if bad:
            raise ValueError(f"reuse table {path} was made for another {', '.join(bad)} (table {[saved.get(n) for n in bad]}, "
                             f"job {[key.get(n) for n in bad]})")
        copies = f["copy_shifts"]
        job = np.unique(np.asarray(event_shifts, np.float64))
        near = np.min(np.abs(job[:, None] - copies[None, :]), axis=1)
        if np.any(near > SHIFT_TOLERANCE):
            raise ValueError(f"reuse table {path} has no copy at event shifts {job[near > SHIFT_TOLERANCE].tolist()} "
                             f"(copies at {copies.tolist()})")
        if float(length) > float(f["length"]) + 1e-9:
            raise ValueError(f"reuse table {path} was made for events up to {float(f['length'])} ns, the job's reach "
                             f"{float(length)} ns")
        starts = np.asarray(starts, np.int64)
        table = {name: f[name] for name in FIELDS if name != "frame"}
        table["frame"] = starts[f["pool"].astype(np.int64)] + f["local"].astype(np.int64)
        for name in ("copy", "sid"):
            table[name] = table[name].astype(np.int64)
        table.update(copy_shifts=copies, shift=np.float64(f["shift"]), phases=np.int64(f["phases"]),
                     periods=np.int64(f["periods"]))
    return table


def shifts_and_margin(shifts):
    """The setup's mean shift and window margin for a set of event shifts (as device_physics: the largest distance
    from the mean plus 5 ns)."""
    shifts = np.asarray(shifts, np.float64)
    mean = float(np.mean(shifts))
    return mean, float(np.max(np.abs(shifts - mean))) + 5.0


def obtain(path, c, capacities, data, length, seed, *, rows_capacity, event_shifts, extra_shifts=(), log=None):
    """The reuse table for a job: loaded from `path` if it exists, else computed (for the job's shifts plus
    `extra_shifts`) and, with a path, saved there. Without a path: computed, as before."""
    from . import device_reuse
    shifts = np.asarray(event_shifts, np.float64)
    if path is not None and Path(path).exists():
        key = table_key(c, capacities, data, seed, rows_capacity)
        table = load(path, key, data["starts"], shifts, length)
        if log is not None:
            log(f"donor reuse table loaded from {path}: {len(table['sid'])} digis, {len(table['copy_shifts'])} copies")
        return table
    union = np.r_[shifts, np.asarray(extra_shifts, np.float64)]   # per event (the mean as before), then the extras
    mean, margin = shifts_and_margin(union)
    table = device_reuse.precompute(c, capacities, data, length, seed, rows_capacity=rows_capacity, shift=mean,
                                    margin=margin, log=log, event_shifts=union)
    if path is not None:
        save(path, table, table_key(c, capacities, data, seed, rows_capacity), data["starts"], length)
        if log is not None:
            log(f"donor reuse table saved to {path}")
    return table


def build(config, out, shifts=()):
    """Compute and save the reuse table of a device_bench configuration, for its signal events' shifts plus
    `shifts`."""
    from trkfastsim.digi import digicond
    from .device_backend import Source
    from .device_bench import configuration
    from .device_source import Layout, pack_source
    from . import crv_inputs
    args = configuration(config)
    capacity = json.loads(Path(args.device_capacities).read_text())
    if "tracker_reuse" not in capacity.get("response", {}):
        raise ValueError(f"{args.device_capacities} has no tracker_reuse block")
    source = Source(args.frames_manifest, seed=args.sampling_seed, max_selected_rows=args.max_selected_rows)
    layout = Layout(args.batch_events, capacity["max_donors"], capacity["rows"], args.max_selected_rows)
    data, _ = pack_source(source, layout)
    mc, _, length = crv_inputs.read_timing(source.signal.event_ids, args.truth, args.stock)
    if Path(out).exists():
        raise FileExistsError(out)
    return obtain(out, digicond.Conditions.load(args.tracker_conditions), capacity["response"]["tracker_reuse"],
                  data, float(np.max(length)), args.tracker_seed, rows_capacity=layout.rows["tracker"],
                  event_shifts=np.asarray(mc), extra_shifts=shifts, log=print)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python3 -m mixing.device_reuse_table", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="compute and save the reuse table of a device_bench configuration")
    b.add_argument("config")
    b.add_argument("out")
    b.add_argument("--shifts", default="", help="comma-separated event time shifts (ns) to add")
    a = p.parse_args(argv)
    shifts = [float(s) for s in a.shifts.split(",") if s.strip()]
    build(a.config, a.out, shifts)


if __name__ == "__main__":
    main()
