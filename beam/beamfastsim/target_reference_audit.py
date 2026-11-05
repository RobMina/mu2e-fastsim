"""Census persisted ReadVirtualDetector rows and primary/pair identities before transport validation."""
import argparse
import json

import numpy as np


def audit(paths, code):
    import uproot
    reports = []
    for path in paths:
        with uproot.open(path) as root:
            tree = root["cetree/ntpart"]
            names = ["run", "subrun", "evt", "trk", "code", "pdg", "isvd", "x", "y", "z", "px", "py", "pz", "time"]
            a = tree.arrays(names, library="np")
            cycles = {k: int(root[k].num_entries) for k in root.keys() if k.split(";")[0] == "cetree/ntpart"}
        event = np.column_stack([a[k] for k in ("run", "subrun", "evt")])
        particles = np.column_stack([event, a["trk"]])
        primary = a["code"] == code
        sel = primary & (a["pdg"] == 11)
        eevent, mult = np.unique(event[sel], axis=0, return_counts=True)
        front = np.stack(a["isvd"])[:, 12].astype(bool)
        p = np.sqrt(a["px"] ** 2 + a["py"] ** 2 + a["pz"] ** 2)
        x = a["x"][sel].astype(float)
        y = a["y"][sel].astype(float)
        z = a["z"][sel].astype(float)
        t = a["time"][sel].astype(float)
        reports.append(dict(path=str(path), tree_cycles=cycles, persisted_particle_rows=len(p),
            persisted_events=int(np.unique(event, axis=0).shape[0]), duplicate_particle_ids=int(len(p) - np.unique(particles, axis=0).shape[0]),
            all_creation_codes={str(int(c)): int(n) for c, n in zip(*np.unique(a["code"], return_counts=True))},
            process_code=int(code), primary_electrons=int(sel.sum()), primary_positrons=int((primary & (a["pdg"] == -11)).sum()),
            primary_electron_events=len(eevent), max_electrons_per_event=int(mult.max()) if mult.size else 0,
            primary_electrons_at_vd13=int((sel & front).sum()),
            start_ranges={k: [float(v.min()), float(v.max())] if v.size else None for k, v in
                          (("x", x), ("y", y), ("z", z), ("t", t), ("p", p[sel]))}))
    return reports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--code", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = audit(a.roots, a.code)
    with open(a.out, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
