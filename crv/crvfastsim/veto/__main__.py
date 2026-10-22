"""python -m crvfastsim.veto SNAPSHOT.root [...] (--card JSON | --cosmic-ana CUTS.yaml CUTSET) [--pmin 85 --pmax 200]
    [--track-time BRANCH] --outdir DIR

C4 on selected-track snapshots: N. Harrison's singleElectron outputs (outTree: trkfit_mom_R, trkfit_time,
crvcoincs_reco.*), one row per selected track. Reports the vetoed fraction (inefficiency = 1 - it, for cosmics; the
deadtime, for signal + beam) with Jeffreys intervals, overall and per sector of the vetoing coincidence. This is the
V-E1 entry point: the same card on the same events must give the analysis's own count.
"""
import argparse
import json
import os

import awkward as ak
import numpy as np

from . import VetoCard, fraction, vetoed
from .apply import gate

PFX = "crvcoincs_reco."


def read_snapshot(paths, track_time="trkfit_time"):
    import uproot
    cols = ["trkfit_mom_R", track_time] + [PFX + c for c in ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType")]
    arrs = [uproot.open(p)["outTree"].arrays(cols, library="ak") for p in paths]
    a = ak.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    co = ak.zip({c: a[PFX + c] for c in ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType")})
    return a["trkfit_mom_R"], a[track_time], co


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--card", help="VetoCard JSON")
    ap.add_argument("--cosmic-ana", nargs=2, metavar=("CUTS_YAML", "CUTSET"))
    ap.add_argument("--pmin", type=float, default=None)
    ap.add_argument("--pmax", type=float, default=None)
    ap.add_argument("--track-time", default="trkfit_time")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    if bool(a.card) == bool(a.cosmic_ana):
        ap.error("give exactly one of --card and --cosmic-ana")
    card = VetoCard.from_dict(json.load(open(a.card))) if a.card else VetoCard.from_mu2e_cosmic_ana(*a.cosmic_ana)
    p, t, co = read_snapshot(a.inputs, a.track_time)
    sel = np.ones(len(p), bool)
    if a.pmin is not None:
        sel &= ak.to_numpy(p) > a.pmin
    if a.pmax is not None:
        sel &= ak.to_numpy(p) < a.pmax
    g = gate(co, card)
    v, sec = vetoed(t[:, None], co, card, return_sector=True)
    v, sec = ak.to_numpy(ak.firsts(v)), ak.to_numpy(ak.firsts(sec))
    out = {"card": card.__dict__ | {"sha": card.sha()}, "inputs": a.inputs, "tracks": int(len(p)),
           "selected": int(sel.sum()), "passing_gate": int((sel & g).sum()), "vetoed": fraction(v[sel & g])}
    out["vetoed_by_sector"] = {int(s): int(((sec == s) & sel & g).sum()) for s in np.unique(sec[sel & g & v])}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.outdir, "veto_stats.json"), "w"), indent=1, default=float)
    f = out["vetoed"]
    print(f"{card.name} ({card.sha()}): {f['k']} / {f['n']} vetoed = {f['frac']:.5f} [{f['lo']:.5f}, {f['hi']:.5f}] "
          f"(68 % Jeffreys); gate kept {out['passing_gate']} of {out['selected']} selected; by sector {out['vetoed_by_sector']}")


if __name__ == "__main__":
    main()
