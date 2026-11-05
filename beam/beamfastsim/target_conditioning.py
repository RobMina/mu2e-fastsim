"""Same-model independent-draw null: quantify outcome conditioning, not G4 closure."""
import argparse
import json

import numpy as np

from . import cechain as C
from . import cecompare as CC
from .target_compare import clustered_mean


def selection_null(base, candidate, groups):
    """Compare unconditional outcomes and candidate acceptance after selecting base successes."""
    a, b = np.asarray(base, bool), np.asarray(candidate, bool)
    if a.shape != b.shape or a.ndim != 1 or np.shape(groups) != a.shape or not a.size:
        raise ValueError("same nonempty row population and source groups required")
    na, nb, both = int(a.sum()), int(b.sum()), int((a & b).sum())
    return dict(starts=int(a.size), base_selected=na, candidate_selected=nb, both_selected=both,
                unconditional=CC._paired_mean(a, b),
                unconditional_clustered_diff=clustered_mean(b.astype(float) - a, groups),
                candidate_acceptance_given_base=float(both / na) if na else None,
                base_acceptance_given_candidate=float(both / nb) if nb else None,
                interpretation="selection on one independent stochastic realization loses migrations on replay even for the same model; no G4 bias correction follows")


def compare(base_path, candidate_path, geom, field):
    a, b = CC.fast_plane([base_path]), CC.fast_plane([candidate_path])
    with np.load(base_path) as x, np.load(candidate_path) as y:
        for k in ("idx", "x0", "u0", "p0", "t0"):
            if not np.array_equal(x[k], y[k]):
                raise ValueError("conditioning null requires exact identical source starts and row IDs")
    full = []
    for s in (a, b):
        obs = CC.observables(geom, field, s)
        values = {}
        for k in ("p", "rmax"):
            values[k] = np.full(s["ok"].size, np.nan)
            values[k][s["ok"]] = obs[k]
        full.append(values)
    out = dict(mode="same-model-independent-draw-conditioning-null", physics_validated=False,
               caveat="VD-selected fast base is not the actual downstream G4 PrimaryFilter; diagnostic only", cuts={})
    groups = np.arange(a["ok"].size) % 20
    for rc in CC.RMAX_CUTS:
        sa = a["ok"] & (full[0]["p"] > CC.P_MIN) & (full[0]["rmax"] > rc)
        sb = b["ok"] & (full[1]["p"] > CC.P_MIN) & (full[1]["rmax"] > rc)
        out["cuts"][str(rc)] = selection_null(sa, sb, groups)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_run1_a.txt")
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = compare(a.base, a.candidate, C.from_offline(a.geom, a.offline_root), C.field_maps(a.maps))
    with open(a.out, "w") as stream:
        json.dump(r, stream, indent=2, allow_nan=False)
    print(json.dumps(r, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
