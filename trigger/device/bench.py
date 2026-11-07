"""Entry points for the shared warm-rate timer (`python3 -m mu2efastsim.bench`), boundary "StrawDigis -> decisions".

    TRIGGER_DEVICE_DUMPS=a.root,b.root TRIGGER_DEVICE_MENU=replay.fcl TRIGGER_DEVICE_MVA=TimeCluster.weights.xml \
    [TRIGGER_DEVICE_RESPONSE=response.json] [TRIGGER_DEVICE_BATCH=1024] [TRIGGER_DEVICE_PRECISION=float32] \
    python3 -m mu2efastsim.bench trigger.device.bench:step --prepare trigger.device.bench:load --events N \
        --dtype float64 (or float32, matching TRIGGER_DEVICE_PRECISION) --workflow trigger --boundary "StrawDigis -> tracker trigger path decisions" --out bench.json

`load(N)` reads the first N events' digis into host memory and packs them into dense event blocks (not timed);
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


def load(n):
    dumps = os.environ["TRIGGER_DEVICE_DUMPS"].split(",")
    cfg = menu.parse_fcl(Path(os.environ["TRIGGER_DEVICE_MENU"]).read_text())
    cond = run.conditions(dumps[0], os.environ.get("TRIGGER_DEVICE_TREE", "trigph"))
    resp_path = os.environ.get("TRIGGER_DEVICE_RESPONSE")
    resp = json.loads(Path(resp_path).read_text()) if resp_path else {}
    ch = chainmod.Chain(cfg, cond, os.environ["TRIGGER_DEVICE_MVA"], response=resp,
                        precision=os.environ.get("TRIGGER_DEVICE_PRECISION", "float64"))
    size = int(os.environ.get("TRIGGER_DEVICE_BATCH", "1024"))
    blocks, left = [], n
    for d in dumps:
        if left <= 0:
            break
        keys, ev, _ = run.load_dump(d, os.environ.get("TRIGGER_DEVICE_TREE", "trigph"))
        keys = keys[:left]
        ev = {k: v[:left] for k, v in ev.items()}
        left -= len(keys)
        blocks.append([(nb, b) for _, nb, b in run.batches(keys, ev, size)])
    if left > 0:
        raise ValueError(f"only {n - left} events in the dumps")
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
