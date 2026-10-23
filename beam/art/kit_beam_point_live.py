#!/usr/bin/env python3
"""Live test of the fastsimkit task beam_point on the EAF endpoint, driven from a GPVM.

Registers the task at runtime (beamfastsim.kit_task.register) so kit.py is not edited, then runs the contract:
check -> submit -> status (poll) -> results, for each geometry given. EAF_TOKEN_FILE must be set.
  python3 beam/art/kit_beam_point_live.py PREFIX geom[,geom...] [muons] [repeat]
"""
import json
import os
import sys
import time

from mu2efastsim.endpoints import Resources, kit
from beamfastsim import kit_task

kit_task.register(kit.TASKS, kit.Task, Resources, kit.FNAL_TASKS)


def main():
    prefix, geoms = sys.argv[1], sys.argv[2].split(",")
    muons = int(sys.argv[3]) if len(sys.argv) > 3 else 105350
    repeat = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    os.environ.setdefault("FASTSIMKIT_ENDPOINT", "eaf")
    k = kit.FastsimKit(state_dir=os.environ.get("FASTSIMKIT_STATE_DIR", "/exp/mu2e/data/users/ram2aq/beam_fastsim/kit_state"))
    handles = []
    for i, g in enumerate(geoms):
        name = f"{prefix}.{i:02d}"
        params = {"task": "beam_point", "geom": g, "muons": float(muons), "seed": 1.0, "repeat": float(repeat)}
        c = k.check(name, params, [], [])
        print("check", name, c, flush=True)
        if not c["ok"]:
            sys.exit(2)
        print("submit", k.submit(name, params, [], []), flush=True)
        handles.append((name, g))
    t0 = time.time()
    out = {}
    while handles:
        for name, g in list(handles):
            s = k.status(name)
            if s["state"] in ("completed", "failed", "cancelled"):
                print(f"{time.time() - t0:7.0f}s {name} {s}", flush=True)
                if s["state"] == "completed":
                    out[g] = k.results(name)["metrics"]
                    print(json.dumps(out[g]), flush=True)
                handles.remove((name, g))
        if handles:
            time.sleep(30)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
