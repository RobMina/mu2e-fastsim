"""Fast response on a saved raw replay sample, preserving its explicit all-event denominator."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .. import chain
from . import fastframes as FF
from .replay import sha256


def stock_shared_table_yields(response):
    """Diagnostic for stock photon-maker aliasing: first sector yield per shared lookup table."""
    first = {}
    sectors = {}
    for name, sector in response.sectors.items():
        value = first.setdefault(sector.table, sector.yield_per_mev)
        sectors[name] = dataclasses.replace(sector, yield_per_mev=value)
    return dataclasses.replace(response, sectors=sectors)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--photon-dev", required=True)
    ap.add_argument("--seed", type=int, default=7311)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pulses", help="optional per-channel pulse parquet from this same response draw")
    ap.add_argument("--stock-shared-table-yields", action="store_true", help="diagnostic only: reuse the first sector's yield for a shared LUT, as stock photon-maker aliasing does")
    a = ap.parse_args(argv)
    path, out = Path(a.input), Path(a.out)
    if out.exists() or out.with_suffix(".json").exists() or (a.pulses and Path(a.pulses).exists()):
        raise ValueError("choose new output paths")
    manifest = json.loads(path.with_suffix(".json").read_text())
    if manifest["output_sha256"] != sha256(path):
        raise ValueError("replay input hash mismatch")
    with uproot.open(path) as f:
        st = f["dump/steps"].arrays(library="pd")
        pa = f["dump/particles"].arrays(library="pd")
        ev = f["dump/events"].arrays(library="pd")
    if len(ev) != manifest["events"]:
        raise ValueError("replay event count mismatch")
    keys = chain.event_key(ev)
    w = -ev.pbtime.to_numpy().astype(np.float32).astype(float)
    st, lost = FF.fold_steps(st, keys, w)
    if lost:
        raise ValueError("steps missing event conditions")
    cond = FF.conditions_from_dump(a.conditions, a.photon_dev)
    timers = {}
    pulse_sink = [] if a.pulses else None
    setup = FF._setup("run1a_v01")
    if a.stock_shared_table_yields:
        setup["cfg"] = dataclasses.replace(setup["cfg"], response=stock_shared_table_yields(setup["cfg"].response))
    fr = FF.frames_from_steps(st, pa, cond, "run1a_v01", keys, keys, w, seed=a.seed,
                              chunk=a.chunk, timers=timers, run_no=manifest["run"], pulse_sink=pulse_sink, setup=setup)
    if len(fr) != len(ev):
        raise ValueError("fast response changed frame denominator")
    out.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(fr, out)
    if a.pulses:
        cols = {k: np.concatenate([p[k] for p in pulse_sink]) if pulse_sink else np.zeros(0)
                for k in ("ev", "chan", "time", "pe", "seq")}
        order = np.argsort(keys)
        cols["time"] += w[order][np.searchsorted(keys[order], cols["ev"])]
        ak.to_parquet(ak.Array(cols), a.pulses)
    result = {"input_sha256": manifest["output_sha256"], "seed": a.seed, "frames": len(fr),
              "coincs": int(ak.sum(ak.num(fr.coincs.time))), "timers": timers,
              "pulse_file": a.pulses, "pulse_sha256": sha256(a.pulses) if a.pulses else None,
              "code_sha256": {n: sha256(Path(__file__).with_name(n)) for n in ("replay_fast.py", "fastframes.py")},
              "chunk": a.chunk, "stock_shared_table_yields_diagnostic": a.stock_shared_table_yields,
              "effective_yields": {k: v.yield_per_mev for k, v in setup["cfg"].response.sectors.items()},
              "conditions_sha256": sha256(a.conditions), "photon_dev_sha256": sha256(a.photon_dev),
              "timing": "Current fast step folding; pbtime rounded to float32 as in stock event products; no reco smearing."}
    out.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
