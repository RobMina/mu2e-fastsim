"""Fast response on a saved raw replay sample, preserving its explicit all-event denominator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .. import chain
from . import fastframes as FF
from .replay import sha256


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--photon-dev", required=True)
    ap.add_argument("--seed", type=int, default=7311)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    path, out = Path(a.input), Path(a.out)
    if out.exists() or out.with_suffix(".json").exists():
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
    fr = FF.frames_from_steps(st, pa, cond, "run1a_v01", keys, keys, w, seed=a.seed,
                              chunk=a.chunk, timers=timers, run_no=manifest["run"])
    if len(fr) != len(ev):
        raise ValueError("fast response changed frame denominator")
    out.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(fr, out)
    result = {"input_sha256": manifest["output_sha256"], "seed": a.seed, "frames": len(fr),
              "coincs": int(ak.sum(ak.num(fr.coincs.time))), "timers": timers,
              "conditions_sha256": sha256(a.conditions), "photon_dev_sha256": sha256(a.photon_dev),
              "timing": "Current fast step folding; pbtime rounded to float32 as in stock event products; no reco smearing."}
    out.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
