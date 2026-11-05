"""Extract event-keyed physical pion stops from existing ReadVirtualDetector trees."""
import argparse
import json

import numpy as np

from . import geometry as G


def read(paths, geom, tree="cetree/ntpart", tmin=350., stopping_code=78):
    import uproot
    if not paths or not np.isfinite(tmin):
        raise ValueError("need input trees and finite pion time cut")
    names = ("run", "subrun", "evt", "trk", "pdg", "xstop", "ystop", "zstop", "tstop", "pstop", "codestop")
    cols = {k: [] for k in names}
    event_counts = []
    for path in paths:
        with uproot.open(path) as root:
            a = root[tree].arrays(list(names), library="np")
        sel = (a["pdg"] == -211) & (a["pstop"] < 1e-6) & (a["codestop"] == stopping_code) & (a["tstop"] >= tmin)
        for k in names:
            cols[k].append(a[k][sel])
        event_counts.append(int(np.unique(np.column_stack([a[k] for k in ("run", "subrun", "evt")]), axis=0).shape[0]))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["at_rest"] = out["pstop"] < 1e-6
    out["foil"] = G.foil_index(geom, out["xstop"], out["ystop"], out["zstop"])
    keep = out["foil"] >= 0
    out = {k: v[keep] for k, v in out.items()}
    identity = np.column_stack([out[k] for k in ("run", "subrun", "evt", "trk")])
    if identity.size and np.unique(identity, axis=0).shape[0] != len(identity):
        raise ValueError("duplicate event/track identities in pion stop inputs")
    return out, event_counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_run1_a.txt")
    ap.add_argument("--tmin", type=float, default=350.)
    ap.add_argument("--stopping-code", type=int, default=78)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out, events = read(a.roots, G.from_offline(a.geom, a.offline_root), tmin=a.tmin, stopping_code=a.stopping_code)
    np.savez_compressed(a.out, **out)
    print(json.dumps(dict(pion_stops=len(out["pdg"]), input_events=events,
                           eligible_events=np.unique(np.column_stack([out[k] for k in ("run", "subrun", "evt")]),
                                                     axis=0).shape[0], tmin=a.tmin, stopping_code=a.stopping_code)))


if __name__ == "__main__":
    main()
