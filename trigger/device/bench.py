"""Entry points for the shared warm-rate timer (`python3 -m mu2efastsim.bench`), boundary "StrawDigis -> decisions".

    TRIGGER_DEVICE_DUMPS=a.root,b.root TRIGGER_DEVICE_MENU=replay.fcl TRIGGER_DEVICE_MVA=TimeCluster.weights.xml \
    [TRIGGER_DEVICE_RESPONSE=ce|dio|rpc|none|response.json] [TRIGGER_DEVICE_BATCH=1024] [TRIGGER_DEVICE_PRECISION=float32] \
    [TRIGGER_DEVICE_ORDER=hits|input] [TRIGGER_DEVICE_HMIN=64] [TRIGGER_DEVICE_COMPACT=0|1] [TRIGGER_DEVICE_PAIRS=0|1] \
    python3 -m mu2efastsim.bench trigger.device.bench:step --prepare trigger.device.bench:load --events N \
        --dtype float64 (or float32, matching TRIGGER_DEVICE_PRECISION) --workflow trigger --boundary "StrawDigis -> tracker trigger path decisions" --out bench.json

TRIGGER_DEVICE_RESPONSE is a named response of trigger/device/response.py (default ce, the binned CE/DIO response;
rpc for RPC events; none for no response layer) or a response JSON file.
Batch, hit-axis floor and compact path default to run.CPU_BATCHING on a CPU backend (16, 32, on) and to 1024, 64, off
otherwise; the variables override them. `load(N)` reads the first N events' digis into host memory and packs them into dense event blocks (not timed).
`load(N, seed=S)` (the timer's `--fresh-inputs`, called before every pass) takes the disjoint slice of events
[S x N, (S + 1) x N) of the dumps read in order, so each seed gets its own events; it raises if the dumps hold fewer
than (S + 1) x N events. The chain and the read dumps are kept for later calls in the process (built once, as in a
production job);
`step(data)` runs a whole pass: host -> device transfer, hit reconstruction, time clusters, helices, every evaluated
path's decision on the device, and the host-side D0-phi prescale counter; it returns the decisions.
"""
from __future__ import annotations

import os
from pathlib import Path

import jax
import numpy as np

from . import chain as chainmod, menu, response, run


_CHAINS = {}
_DUMPS = {}


def _chain(dumps, tree):
    compact = os.environ.get("TRIGGER_DEVICE_COMPACT", "1" if _cpu().get("compact") else "0") == "1"
    pairs = os.environ.get("TRIGGER_DEVICE_PAIRS", "0") == "1"
    key = (dumps[0], tree, os.environ["TRIGGER_DEVICE_MENU"], os.environ["TRIGGER_DEVICE_MVA"],
           _response(), os.environ.get("TRIGGER_DEVICE_PRECISION", "float64"), compact, pairs)
    if key not in _CHAINS:
        cfg = menu.parse_fcl(Path(os.environ["TRIGGER_DEVICE_MENU"]).read_text())
        cond = run.conditions(dumps[0], tree)
        resp = response.load(_response())
        _CHAINS[key] = chainmod.Chain(cfg, cond, os.environ["TRIGGER_DEVICE_MVA"], response=resp,
                                      precision=os.environ.get("TRIGGER_DEVICE_PRECISION", "float64"), compact=compact,
                                      pairs=pairs)
    return _CHAINS[key]


def _response():
    """TRIGGER_DEVICE_RESPONSE: a named response or a file (default response.DEFAULT)."""
    return os.environ.get("TRIGGER_DEVICE_RESPONSE") or response.DEFAULT


def _cpu():
    """run.CPU_BATCHING on a CPU backend (the TRIGGER_DEVICE_* variables override it), else {}."""
    return run.CPU_BATCHING if run.on_cpu() else {}


def _order():
    """TRIGGER_DEVICE_ORDER: "hits" (default: events batched by digi count, results put back in event order; the same
    decisions bit for bit, less padding of the hit axis: A100 batch 1024 +51 % float32 / +58 % float64, one CPU core
    batch 64 2.0x / 2.2x) or "input" (the dumps' event order in each batch, the order before 2026-11-10)."""
    order = os.environ.get("TRIGGER_DEVICE_ORDER", "hits")
    if order not in ("input", "hits"):
        raise ValueError(f"TRIGGER_DEVICE_ORDER must be input or hits, not {order!r}")
    return order


def _dump(path, tree):
    if (path, tree) not in _DUMPS:
        keys, ev, _ = run.load_dump(path, tree)
        _DUMPS[path, tree] = keys, ev
    return _DUMPS[path, tree]


def load(n, seed=None):
    if n < 1:
        raise ValueError("n must be positive")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    dumps = os.environ["TRIGGER_DEVICE_DUMPS"].split(",")
    tree = os.environ.get("TRIGGER_DEVICE_TREE", "trigph")
    ch = _chain(dumps, tree)
    size = int(os.environ.get("TRIGGER_DEVICE_BATCH", _cpu().get("batch", 1024)))
    hmin = int(os.environ.get("TRIGGER_DEVICE_HMIN", _cpu().get("hmin", 64)))
    skip = 0 if seed is None else seed * n           # events of the dumps (in order) before this slice
    blocks, left = [], n
    for d in dumps:
        if left <= 0:
            break
        keys, ev = _dump(d, tree)
        lo = min(skip, len(keys))
        skip -= lo
        keys = keys[lo:lo + left]
        if not keys:
            continue
        ev = {k: v[lo:lo + len(keys)] for k, v in ev.items()}
        left -= len(keys)
        if _order() == "hits":
            blocks.append([(nb, b, rows) for rows, nb, b in run.sorted_batches(keys, ev, size, hmin=hmin)])
        else:
            blocks.append([(nb, b, None) for _, nb, b in run.batches(keys, ev, size, hmin)])
    if left > 0:
        if seed is None:
            raise ValueError(f"only {n - left} events in the dumps")
        raise ValueError(f"the dumps hold too few events for seed {seed}: its slice is events [{seed * n}, "
                         f"{(seed + 1) * n}) and only {n - left} of its {n} events exist (fresh inputs never reuse "
                         f"events; add dumps or lower --seed or --events)")
    return ch, blocks


_WARM = set()


def _warmup(ch, blocks):
    """TRIGGER_DEVICE_COMPACT: in the first pass (inside the timer: setup), compile the compact path's capacities for
    every hit-axis size from the smallest block's up to 512 hits (larger blocks compile when they come)."""
    if not getattr(ch, "compact", False) or not blocks or not blocks[0]:
        return
    _, b, _ = blocks[0][0]
    E, h0 = b["valid"].shape
    hmax = max(512, max(bb["valid"].shape[1] for fb in blocks for _, bb, _ in fb))
    h = 1 << int(np.ceil(np.log2(max(h0, 1))))
    while h <= hmax:
        if (id(ch), E, h) not in _WARM:
            ch.warmup({k: np.zeros((E, h) + v.shape[2:], v.dtype) if v.ndim >= 2 and v.shape[1] == h0 else v
                       for k, v in b.items()})
            _WARM.add((id(ch), E, h))
        h *= 2


def step(data):
    ch, blocks = data
    _warmup(ch, blocks)
    out = []
    for file_blocks in blocks:                       # one file = one prescale-counter sequence
        decs, auxs, ns = [], [], 0
        for nb, b, _ in file_blocks:
            d, a, _ = ch.decide(b)
            decs.append((nb, d))
            auxs.append((nb, a))
            ns += nb
        # one host transfer of every block's decisions and stage arrays (not one blocking copy per array)
        dh, ah = jax.device_get(([d for _, d in decs], [a for _, a in auxs]))
        dec = [{k: np.asarray(v)[:nb] for k, v in d.items()} for (nb, _), d in zip(decs, dh)]
        aux = [{k: np.asarray(v)[:nb] for k, v in a.items()} for (nb, _), a in zip(auxs, ah)]
        if file_blocks[0][2] is not None:            # hit-sorted blocks: back to event order (prescale counter)
            rows = [r for _, _, r in file_blocks]
            dec, aux = run.unsort(rows, dec), run.unsort(rows, aux)
        else:
            dec = {k: np.concatenate([d[k] for d in dec]) for k in dec[0]}
            aux = {k: np.concatenate([a[k] for a in aux]) for k in aux[0]}
        out.append(ch.finish(dec, aux, ns))
    return out
