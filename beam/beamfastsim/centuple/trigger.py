"""Stage 2 (GPU, own process): device digis -> tracker trigger path decisions (trigger.device), per event.

The device trigger evaluates the tpr/apr paths of a prepared menu (``trigger/replay.py``) from the StrawDigis, with
the KalSeedFilter response layer (``trigger/device/run.py response``). It sets JAX's process-wide x64 flag from its
precision, so it runs in a process of its own after the record stage. The D0-phi prescale counter of HelixFilter
runs on the host per subrun (one counter sequence per input file, as stock).

Output: ``centuple_trigger.npz`` with run, subrun, event and one bool array per evaluated path (``trig_<path>``),
and a JSON ``meta`` (menu, response, precision, skipped paths).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

TRIGGER = "centuple_trigger.npz"


# The hit times subtract the proton bunch time, as stock subtracts its reco estimate (evtinfo.pbtime). Digis from the
# record stage carry the per-event estimate (pbtime_reco, the stock MC proton bunch time plus its reco spread); pbt
# overrides it with one value for every event.


def decide(digis_path, menu, conditions, tc_mva, response, precision="float64", block=1024, spill=1,
           conditions_tree="w8ph", pbt=None):
    """Decisions {path: bool[n]} for every event of a record's digis file, and the meta record."""
    from trigger.device import chain as C, digis as D, menu as M, run as R
    f = np.load(digis_path)
    run, sub, evn = (np.asarray(f[k], np.int64) for k in ("run", "subrun", "event"))
    n = evn.size
    dg = {k: f[k] for k in ("ev", "sid", "tdc", "tot", "pmp")}
    cfg = M.parse_fcl(Path(menu).read_text())
    resp = json.loads(Path(response).read_text()) if response else None
    tc = C.Chain(cfg, R.conditions(conditions, conditions_tree), tc_mva, response=resp, precision=precision)
    tic = time.perf_counter()
    decs, auxs = [], []
    if pbt is not None:
        pbts = np.full(n, float(pbt))
    elif "pbtime_reco" in f.files:
        pbts = np.asarray(f["pbtime_reco"], float)
    else:
        pbts = np.zeros(n)
    for lo, m, b in D.blocks(dg, n, block, events=evn, spill=spill, pbt=pbts):
        d, a, _ = tc.decide(b)
        decs.append((m, d))
        auxs.append((m, a))
    dec = {k: np.concatenate([np.asarray(d[k])[:m] for m, d in decs]) for k in decs[0][1]}
    aux = {k: np.concatenate([np.asarray(a[k])[:m] for m, a in auxs]) for k in auxs[0][1]}
    out = {k: np.zeros(n, bool) for k in dec}
    for s in np.unique(sub):                  # one prescale-counter sequence per subrun, in event order
        msk = sub == s
        fin = tc.finish({k: v[msk] for k, v in dec.items()}, {k: v[msk] for k, v in aux.items()}, int(msk.sum()))
        for k in out:
            out[k][msk] = fin[k]
    meta = dict(stage="trigger", precision=precision, menu=str(menu), response=str(response), spill=int(spill), pbt=("per event" if pbt is None else float(pbt)),
                paths=sorted(out), skipped=sorted(tc.skipped), wall_s=round(time.perf_counter() - tic, 2),
                events=int(n))
    return dict(run=run, subrun=sub, event=evn), out, meta


def run(digis_path, out, **kw):
    keys, dec, meta = decide(digis_path, **kw)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / TRIGGER, meta=json.dumps(meta), **keys, **{"trig_" + k: v for k, v in dec.items()})
    return out / TRIGGER, meta
