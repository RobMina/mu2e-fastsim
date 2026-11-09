#!/usr/bin/env python3
"""Live UVA test of the fastsimkit task crv_response from a host on the UVA network that also reaches the FNAL host:
check, submit, stream the inputs (GPVM -> this host -> job tmpfs), poll, results.

    PYTHONPATH=<core>:<crv> FASTSIMKIT_CRV_BRANCH=crv-response-kit python3 live_kit_crv_response_uva.py [key=value ...]
Env: FASTSIMKIT_STATE_DIR (default ~/.mu2efastsim/kit_uva_crv), FASTSIMKIT_SIF, FASTSIMKIT_FNAL_HOST, FASTSIMKIT_CRV_*.
"""
import json
import os
import sys
import time

from mu2efastsim.endpoints import get
from mu2efastsim.endpoints.kit import FastsimKit


def main():
    params = {"task": "crv_response", "seed": 1, **dict(a.split("=", 1) for a in sys.argv[1:])}
    state = os.path.expanduser(os.environ.get("FASTSIMKIT_STATE_DIR", "~/.mu2efastsim/kit_uva_crv"))
    ep = get("uva", state_dir=os.path.join(state, "jobs"), max_su_per_job=60.0)
    kit = FastsimKit(endpoint=ep, state_dir=state)
    name = f"crvresp_{time.strftime('%Y%m%d_%H%M%S')}"
    print("check", kit.check(name, params, [], []), flush=True)
    t0 = time.time()
    print("submit", kit.submit(name, params, [], []), f"({time.time() - t0:.0f} s incl. code update)", flush=True)
    while True:
        st = kit.status(name)
        print(f"{time.time() - t0:6.0f} s  {st['state']}  {st['message'][:150]}", flush=True)
        if st["state"] != "working":
            break
        time.sleep(30)
    if st["state"] == "completed":
        r = kit.results(name)
        m = r["metrics"]
        print(json.dumps({k: m.get(k) for k in ("objective", "objective_sigma", "events_with_steps", "wall_s",
                                                "input_wait_s")}))
        print(f"{len(m)} metrics; metadata {r['metadata']}")
        times = ep.times(name) if hasattr(ep, "times") else {}
        print("slurm", times)


if __name__ == "__main__":
    main()
