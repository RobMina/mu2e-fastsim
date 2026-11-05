"""Geant4 conversion-electron reference table from ReadVirtualDetector `ntpart` trees (CeEndpoint dts inputs).

One row per conversion electron (creation code `code`, default mu2eCeMinusEndpoint): start position, momentum and
global time, end code, and the first crossing of the tracker entry virtual detector (default TT_FrontHollow). Every
input tree's highest cycle is read; the number of rows per file is recorded so it can be checked against the event
count of the input file. Needs uproot.
"""
import argparse
import json
from pathlib import Path

import numpy as np

CE_CODE = 167        # ProcessCode mu2eCeMinusEndpoint
VD_FRONT = 13        # VirtualDetectorId TT_FrontHollow
START = ("x", "y", "z", "px", "py", "pz", "time")
VD = ("isvd", "xvd", "yvd", "zvd", "pxvd", "pyvd", "pzvd", "tvd")


def read(paths, tree="cetree/ntpart", code=CE_CODE, vd=VD_FRONT, extra=()):
    """extra: further (VD id, suffix) pairs, e.g. (10, "stout") for ST_Out."""
    import uproot
    vds = [(vd, "front")] + list(extra)
    cols = {k: [] for k in START + ("codestop", "evt", "subrun", "ifile")}
    cols.update({f"{k}_{sfx}": [] for _, sfx in vds for k in VD})
    rows = []
    for i, p in enumerate(paths):
        f = uproot.open(p)
        cyc = max((k for k in f.keys() if k.split(";")[0] == tree), key=lambda k: int(k.split(";")[1]))
        a = f[cyc].arrays(list(START) + ["codestop", "evt", "subrun", "code", "pdg"] + list(VD), library="np")
        sel = (a["code"] == code) & (a["pdg"] == 11)
        rows.append(int(sel.sum()))
        for k in START + ("codestop", "evt", "subrun"):
            cols[k].append(a[k][sel])
        cols["ifile"].append(np.full(sel.sum(), i))
        for v, sfx in vds:
            for k in VD:
                cols[f"{k}_{sfx}"].append(np.stack(a[k][sel])[:, v - 1] if sel.any() else np.empty(0))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    return out, rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.cereference", description=__doc__)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gencount", type=int, required=True, help="generated events per input file (SAM dh.gencount)")
    ap.add_argument("--code", type=int, default=CE_CODE,
                    help="creation code of the primaries (default mu2eCeMinusEndpoint; 166 mu2eMuonDecayAtRest for DIO)")
    ap.add_argument("--extra-vd", nargs=2, action="append", default=[], metavar=("ID", "SUFFIX"),
                    help="also keep the first hit in this virtual detector (e.g. 10 stout for ST_Out)")
    a = ap.parse_args(argv)
    out, rows = read(a.roots, code=a.code, extra=[(int(i), sfx) for i, sfx in a.extra_vd])
    np.savez_compressed(a.out, gencount=np.full(len(a.roots), a.gencount), rows=np.array(rows), **out)
    print(json.dumps(dict(files=len(a.roots), rows=int(sum(rows)), front=int(out["isvd_front"].sum()),
                          inputs=[Path(p).name for p in a.roots]), indent=1))


if __name__ == "__main__":
    main()
