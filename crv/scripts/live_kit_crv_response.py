#!/usr/bin/env python3
"""Live EAF test of the fastsimkit task crv_response from a GPVM: check, submit, poll, results, stop the server.
The job runs the code checkout CODE (a dir holding core/ and crv/; the EAF default is the shared umbrella).

    EAF_TOKEN_FILE=... PYTHONPATH=CODE/core:CODE/crv python3 live_kit_crv_response.py CODE [key=value ...]
"""
import json
import os
import sys
import time

from mu2efastsim.endpoints.eaf import EAFEndpoint
from mu2efastsim.endpoints.kit import FastsimKit

D = os.environ.get("CRVFS_DATA", "/exp/mu2e/data/users/ram2aq/crv_fastsim")


def main():
    code = sys.argv[1]
    params = {"task": "crv_response", "seed": "1", **dict(a.split("=", 1) for a in sys.argv[2:])}
    slices = os.environ.get("EAF_SLICES", "1g.10gb").split(",")
    ep = EAFEndpoint(workdir=f"{D}/eafkit", state_dir=f"{D}/eafkit_state/jobs", slices=slices, code=code)
    kit = FastsimKit(endpoint=ep, state_dir=f"{D}/eafkit_state")
    name = f"live_crv_response_{time.strftime('%Y%m%d_%H%M%S')}"
    print("check", kit.check(name, params, [], []), flush=True)
    t0 = time.time()
    try:
        print("submit", kit.submit(name, params, [], []), flush=True)
        while True:
            st = kit.status(name)
            print(f"{time.time() - t0:6.0f} s  {st['state']}  {st['message']}", flush=True)
            if st["state"] != "working":
                break
            time.sleep(30)
        if st["state"] == "completed":
            r = kit.results(name)
            m = r["metrics"]
            print(json.dumps({k: m[k] for k in ("objective", "objective_sigma", "events_with_steps", "wall_s")}))
            print(f"{len(m)} metrics; files: {[f['name'] for f in r['files']]}")
    finally:
        print("stop_idle:", ep.stop_idle(), flush=True)


if __name__ == "__main__":
    main()
