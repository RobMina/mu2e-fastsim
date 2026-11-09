"""Where a fast-vs-G4 crossings excess sits for a run configuration: per sector type, crossings of the primary muon
binned in |p| at the bar entry (fast: px0..; G4: CrvStep momentum) and in the material crossed below the MSC plane
before the first bar (fast arm, from the below-plane world). Diagnostic for V1 side sectors (Run 2).

    python3 scripts/side_excess.py FAST.root G4.root --run-config run2_v10 [--gdml ...]
"""
import argparse
import json

import numpy as np

from crvfastsim import runconfig, tables
from crvfastsim import transport as tr

P_EDGES = np.array([0, 300, 600, 1000, 2000, 5000, 1e9])


def xings(path, geom):
    st = tables.read_many([path], "steps")
    pa = tables.read_many([path], "particles", tables.KEY + ["key", "pdg", "isprim"])
    prim = pa[(pa.isprim == 1) & (pa.pdg.abs() == 13)][tables.KEY + ["key"]].rename(columns={"key": "sp"})
    s = st.merge(prim, on=tables.KEY + ["sp"]).sort_values(tables.KEY + ["bar", "t0"])
    s = s.drop_duplicates(tables.KEY + ["bar"])
    s["p"] = np.sqrt(s.px0 ** 2 + s.py0 ** 2 + s.pz0 ** 2)
    s["sector"] = geom.sector[s.bar.to_numpy()]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fast")
    ap.add_argument("g4")
    ap.add_argument("--run-config", default="run2_v10")
    a = ap.parse_args()
    rc = runconfig.get(a.run_config)
    g, c = rc.geometry(), rc.counters_cfg()
    names = runconfig.sector_type_names(g, c)
    st_of = np.array([int(c[f"crs.sectorType{n}"]) for n in g.sector_names])
    f, r = xings(a.fast, g), xings(a.g4, g)
    out = {}
    for t, nm in sorted(names.items()):
        hf = np.histogram(f.p[st_of[f.sector] == t], P_EDGES)[0]
        hr = np.histogram(r.p[st_of[r.sector] == t], P_EDGES)[0]
        out[nm] = {"p_edges": P_EDGES[:-1].tolist(), "fast": hf.tolist(), "g4": hr.tolist(),
                   "ratio": np.round(hf / np.maximum(hr, 1), 3).tolist()}
        print(f"{nm:6s} fast/g4 per |p| bin {P_EDGES[:-1].astype(int).tolist()}: {out[nm]['ratio']}  (g4 {hr.tolist()})")
    print(json.dumps({"run_config": rc.name}))


if __name__ == "__main__":
    main()
