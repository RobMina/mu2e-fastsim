#!/usr/bin/env python3
"""Live UVA check of the per-node input cache: submit two crv_response points back to back (same inputs, different
knobs), poll both, report each point's input wait, whether it reused the node's inputs, and its node.

    PYTHONPATH=<core>:<crv> FASTSIMKIT_CRV_BRANCH=<branch> python3 live_kit_crv_response_uva_pair.py
"""
import json
import os
import time

from mu2efastsim.endpoints import get
from mu2efastsim.endpoints.kit import FastsimKit


def main():
    state = os.path.expanduser(os.environ.get("FASTSIMKIT_STATE_DIR", "~/.mu2efastsim/kit_uva_crv"))
    ep = get("uva", state_dir=os.path.join(state, "jobs"), max_su_per_job=60.0)
    kit = FastsimKit(endpoint=ep, state_dir=state)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pts = {f"crvpair_{stamp}_a": {"speed_EX": 1.04}, f"crvpair_{stamp}_b": {"speed_EX": 1.12}}
    t0 = time.time()
    for n, kn in pts.items():
        print("submit", kit.submit(n, {"task": "crv_response", "seed": 1, **kn}, [], []), flush=True)
    left = set(pts)
    while left:
        for n in sorted(left):
            st = kit.status(n)
            if st["state"] != "working":
                left.discard(n)
                print(f"{time.time() - t0:6.0f} s  {n} {st['state']}", flush=True)
        time.sleep(30)
    for n in pts:
        r = kit.results(n)
        m = r["metrics"]
        inbox = ep.read_file(n, "inbox").split()
        print(n, json.dumps({k: m.get(k) for k in ("objective", "objective_sigma", "input_wait_s", "inputs_reused")}),
              "node", inbox[0] if inbox else "?", "slurm", ep.times(n).get("elapsed_s"))


if __name__ == "__main__":
    main()
