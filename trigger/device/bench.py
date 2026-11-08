"""Entry points for the shared warm-rate timer (`python3 -m mu2efastsim.bench`), boundary "StrawDigis -> decisions".

    TRIGGER_DEVICE_DUMPS=a.root,b.root TRIGGER_DEVICE_MENU=replay.fcl TRIGGER_DEVICE_MVA=TimeCluster.weights.xml \
    [TRIGGER_DEVICE_RESPONSE=response.json] [TRIGGER_DEVICE_BATCH=1024] [TRIGGER_DEVICE_PRECISION=float32] \
    python3 -m mu2efastsim.bench trigger.device.bench:step --prepare trigger.device.bench:load --events N \
        --dtype float64 (or float32, matching TRIGGER_DEVICE_PRECISION) --workflow trigger --boundary "StrawDigis -> tracker trigger path decisions" --out bench.json

`load(N)` reads the first N events' digis into host memory and packs them into dense event blocks (not timed).
`load(N, seed=S)` (the timer's `--fresh-inputs`, called before every pass) takes the disjoint slice of events
[S x N, (S + 1) x N) of the dumps read in order, so each seed gets its own events; it raises if the dumps hold fewer
than (S + 1) x N events. The chain and the read dumps are kept for later calls in the process (built once, as in a
production job);
`step(data)` runs a whole pass: host -> device transfer, hit reconstruction, time clusters, helices, every evaluated
path's decision on the device, and the host-side D0-phi prescale counter; it returns the decisions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import numpy as np

from . import chain as chainmod, menu, run


_CHAINS = {}
_DUMPS = {}


def _chain(dumps, tree):
    key = (dumps[0], tree, os.environ["TRIGGER_DEVICE_MENU"], os.environ["TRIGGER_DEVICE_MVA"],
           os.environ.get("TRIGGER_DEVICE_RESPONSE"), os.environ.get("TRIGGER_DEVICE_PRECISION", "float64"))
    if key not in _CHAINS:
        cfg = menu.parse_fcl(Path(os.environ["TRIGGER_DEVICE_MENU"]).read_text())
        cond = run.conditions(dumps[0], tree)
        resp_path = os.environ.get("TRIGGER_DEVICE_RESPONSE")
        resp = json.loads(Path(resp_path).read_text()) if resp_path else {}
        _CHAINS[key] = chainmod.Chain(cfg, cond, os.environ["TRIGGER_DEVICE_MVA"], response=resp,
                                      precision=os.environ.get("TRIGGER_DEVICE_PRECISION", "float64"))
    return _CHAINS[key]


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
    size = int(os.environ.get("TRIGGER_DEVICE_BATCH", "1024"))
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
        blocks.append([(nb, b) for _, nb, b in run.batches(keys, ev, size)])
    if left > 0:
        if seed is None:
            raise ValueError(f"only {n - left} events in the dumps")
        raise ValueError(f"the dumps hold too few events for seed {seed}: its slice is events [{seed * n}, "
                         f"{(seed + 1) * n}) and only {n - left} of its {n} events exist (fresh inputs never reuse "
                         f"events; add dumps or lower --seed or --events)")
    return ch, blocks


def step(data):
    ch, blocks = data
    out = []
    for file_blocks in blocks:                       # one file = one prescale-counter sequence
        decs, auxs, ns = [], [], 0
        for nb, b in file_blocks:
            d, a, _ = ch.decide(b)
            decs.append((nb, d))
            auxs.append((nb, a))
            ns += nb
        jax.block_until_ready([d for _, d in decs])
        dec = {k: np.concatenate([np.asarray(d[k])[:nb] for nb, d in decs]) for k in decs[0][1]}
        aux = {k: np.concatenate([np.asarray(a[k])[:nb] for nb, a in auxs]) for k in auxs[0][1]}
        out.append(ch.finish(dec, aux, ns))
    return out
