"""Summarise admitted named conditions and stock event populations without inferring physics closure."""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from .. import runconfig, readout
from ..named_conditions import load_conditions


def inventory(manifest, reference=None):
    m = json.loads(Path(manifest).read_text())
    rc = runconfig.get(m["run_config"])
    geom = rc.geometry()
    cond, metadata = load_conditions(manifest, rc, geom)
    unmapped = (cond.roc < 0) | (cond.feb < 0)
    disabled = (cond.status & 5) != 0
    ignored = (cond.status & 2) != 0
    sector = np.repeat(geom.sector, 4)
    reflector = np.array([rc.sectors[name].reflector for name in geom.sector_names])[sector]
    parity = np.arange(len(cond.status)) % 2
    simulated = (reflector == 0) | ((reflector > 0) & (parity == 0)) | ((reflector < 0) & (parity == 1))
    present = readout.sensor_mask(geom, rc.counters_cfg())
    out = {"run_config": rc.name, "scope": metadata["scope"], "channels": len(cond.status),
           "physical_readout_present": int(present.sum()), "physical_readout_absent": int((~present).sum()),
           "photon_simulated_but_absent": int((simulated & ~present).sum()),
           "unmapped_physical_response": int((unmapped & ~disabled & simulated & present).sum()),
           "mapped": int((~unmapped).sum()), "unmapped": int(unmapped.sum()),
           "unmapped_status_enabled": int((unmapped & ~disabled).sum()),
           "unmapped_simulated": int((unmapped & ~disabled & simulated).sum()),
           "unmapped_reflector_side": int((unmapped & ~simulated).sum()),
           "response_disabled": int(disabled.sum()), "reco_ignored": int(ignored.sum()),
           "nonzero_offsets": int(np.count_nonzero(cond.timeoffset)),
           "nonzero_deviations": int(np.count_nonzero(cond.deviation)), "sectors": {},
           "source_event_identity": metadata["stock_event_identity"],
           "assets": metadata["assets"], "physics_closure": "not_evaluated"}
    for i, name in enumerate(geom.sector_names):
        mask = sector == i
        out["sectors"][name] = {"channels": int(mask.sum()), "unmapped": int((mask & unmapped).sum()),
                                "physical_readout_present": int((mask & present).sum()),
                                "physical_readout_absent": int((mask & ~present).sum()),
                                "photon_simulated_but_absent": int((mask & simulated & ~present).sum()),
                                "response_disabled": int((mask & disabled).sum()),
                                "unmapped_simulated": int((mask & unmapped & ~disabled & simulated).sum()),
                                "unmapped_reflector_side": int((mask & unmapped & ~simulated).sum())}
    if reference:
        with uproot.open(reference) as f:
            out["reference_trees"] = {name: int(f[f"crvdump/{name}"].num_entries)
                                      for name in ("events", "calib", "charges", "digis", "pulses", "clusters")}
            ev = f["crvdump/events"].arrays(["run", "subrun", "event", "spill", "evlen", "pbtime"], library="np")
            ids = np.column_stack([ev[k] for k in ("run", "subrun", "event")])
            if len(ids) != len(np.unique(ids, axis=0)):
                raise ValueError("duplicate reference event identity")
            out["reference_event_window"] = {key: np.unique(ev[key]).tolist() for key in ("spill", "evlen", "pbtime")}
            cl = f["crvdump/clusters"].arrays(["sectortype"], library="np")["sectortype"]
            types, counts = np.unique(cl, return_counts=True)
            out["reference_cluster_types"] = dict(zip(map(str, types), map(int, counts)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = inventory(args.manifest, args.reference)
    with open(args.out, "x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
