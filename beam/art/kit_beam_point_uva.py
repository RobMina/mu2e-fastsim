#!/usr/bin/env python3
"""Live test of beam_point on UVA (one A100), from the laptop on the UVA network/VPN.

The kit streams the muon table from the FNAL host (ssh) into the UVA job's tmpfs; the job fetches the public inputs
from the CVMFS Stratum-1. check -> submit -> status (poll) -> results for each geometry given.
  python3 beam/art/kit_beam_point_uva.py PREFIX geom[,geom...] [muons] [repeat]
Env: FASTSIMKIT_STATE_DIR (default ~/mu2e/scratch_uva/fastsimkit_beam), FASTSIMKIT_FNAL_HOST (mu2egpvm03).
"""
import json
import os
import sys
import time

from mu2efastsim.endpoints import kit
from beamfastsim import kit_task

if "beam_point" not in kit.DELEGATES:
    kit_task.register(kit.TASKS, kit.Task, kit.Resources, kit.FNAL_TASKS)
    kit_task.register_delegate(kit.DELEGATES)


def main():
    prefix, geoms = sys.argv[1], sys.argv[2].split(",")
    muons = int(sys.argv[3]) if len(sys.argv) > 3 else 105350
    repeat = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.environ.setdefault("FASTSIMKIT_ENDPOINT", "uva")
    k = kit.FastsimKit(state_dir=os.path.expanduser(os.environ.get("FASTSIMKIT_STATE_DIR", "~/mu2e/scratch_uva/fastsimkit_beam")))
    handles = []
    for i, g in enumerate(geoms):
        name = f"{prefix}.{i:02d}"
        params = {"task": "beam_point", "geom": g, "muons": float(muons), "seed": 1.0, "repeat": float(repeat)}
        c = k.check(name, params, [], [])
        print("check", name, c, flush=True)
        if not c["ok"]:
            sys.exit(2)
        print("submit", k.submit(name, params, [], []), "job", k.ep.state.get(name).job_id, flush=True)
        handles.append((name, g))
    t0, out = time.time(), {}
    while handles:
        for name, g in list(handles):
            s = k.status(name)
            print(f"{time.time() - t0:6.0f}s {name} {s['state']} {s['message']}", flush=True)
            if s["state"] in ("completed", "failed", "cancelled"):
                if s["state"] == "completed":
                    out[g] = k.results(name)["metrics"]
                    print(json.dumps(out[g]), flush=True)
                handles.remove((name, g))
        if handles:
            time.sleep(30)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
