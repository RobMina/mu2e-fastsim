"""P0 adapter for S. Grant's mu2e_cosmic_ana (github sam-grant/mu2e_cosmic_ana): run his own cut definitions
(Analyse.define_cuts, a cut set from config/common/cuts.yaml) on EventNtuple files, combine every active cut except the
CRV veto ("unvetoed") and the momentum windows narrower than the requested one (--window, default 85-200), and write one row per selected track: its time at the tracker middle (electron hypothesis, the
time his veto uses), its momentum there, and its event's CRV coincidences. crvfastsim.veto then applies a VetoCard.
His selection code is imported, never copied (rule E11: the analysis owns the selection).

    python -m crvfastsim.veto.cosmic_ana --mca /path/to/mu2e_cosmic_ana --cutset dev FILE.root [...] --out sel.parquet
Needs the Mu2e ana environment (pyutils, uproot, awkward); input files on disk (prestaged) or xrootd URLs.
"""
import argparse
import os
import sys

import awkward as ak
import numpy as np

BRANCHES = {
    "evt": ["run", "subrun", "event"],
    "crv": ["crvcoincs.time", "crvcoincs.nHits", "crvcoincs.pos.fCoordinates.fZ", "crvcoincs.timeStart",
            "crvcoincs.timeEnd", "crvcoincs.PEs", "crvcoincs.nLayers", "crvcoincs.sectorType"],
    "trk": ["trk.nactive", "trk.pdg", "trkqual.valid", "trkqual.result"],
    "trkfit": ["trksegs", "trksegpars_lh"],
    "trkmc": ["trkmcsim"],
}


def _import_mca(mca):
    for sub in ("src/core", "src/utils"):
        p = os.path.join(mca, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from analyse import Analyse
    from cut_manager import CutManager
    return Analyse, CutManager


NARROWER = {"wide": ("within_ext_win", "within_sig_win"), "ext": ("within_sig_win",), "sig": ()}
GATE = ("within_coinc_start_time", "within_coinc_end_time")   # flash gate: kept out of the selection for deadtime


def select_file(path, analyse, CutManager, location="tape", drop=NARROWER["wide"]):
    """One file -> awkward record per selected track (all active cuts but 'unvetoed'). A bare file name is resolved by
    pyutils through mdh at `location` (as mu2e_cosmic_ana does); a path or URL is opened as given."""
    from pyutils.pyprocess import Processor
    remote = "/" not in path
    data = Processor(use_remote=remote, location=location, verbosity=0).process_data(file_name=path, branches=BRANCHES)
    cm = CutManager(verbosity=0)
    data = analyse.define_cuts(data, cm)
    cm.toggle_cut({"unvetoed": False, **{c: False for c in drop}})
    mask = cm.combine_cuts(active_only=True)                      # [E, T]
    at_mid = analyse.selector.select_surface(data["trkfit"], surface_name="TT_Mid")
    segs = data["trkfit"]["trksegs"]
    t_mid = ak.firsts(segs["time"][at_mid], axis=-1)              # [E, T]
    m = segs["mom"]["fCoordinates"]
    p_mid = ak.firsts(np.sqrt(m["fX"] ** 2 + m["fY"] ** 2 + m["fZ"] ** 2)[at_mid], axis=-1)
    keep = ak.any(mask, axis=-1)
    ev = data["evt"][keep]
    co = data["crv"][keep]
    return ak.zip({
        "run": ev["run"], "subrun": ev["subrun"], "event": ev["event"],
        "t_mid": t_mid[keep][mask[keep]], "p_mid": p_mid[keep][mask[keep]],
        "coincs": ak.zip({k.split(".", 1)[1]: co[k] for k in BRANCHES["crv"]}),
    }, depth_limit=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--mca", required=True, help="mu2e_cosmic_ana checkout")
    ap.add_argument("--cutset", default="dev")
    ap.add_argument("--on-spill", action="store_true", default=True)
    ap.add_argument("--location", default="tape", help="mdh location for bare file names (tape, disk, scratch)")
    ap.add_argument("--window", choices=sorted(NARROWER), default="wide",
                    help="momentum window: the cut set's windows narrower than this are switched off (55234's table is 'wide', 85-200)")
    ap.add_argument("--no-gate", action="store_true",
                    help="also switch off the coincidence start/end gate cuts (deadtime: the gate is applied with the veto)")
    ap.add_argument("--files", help="text file of inputs, one per line (added to the positional ones)")
    ap.add_argument("--out", required=True, help="parquet with one row per event holding its selected tracks")
    a = ap.parse_args(argv)
    Analyse, CutManager = _import_mca(a.mca)
    analyse = Analyse(on_spill=a.on_spill, cut_config_path=os.path.join(a.mca, "config/common/cuts.yaml"),
                      cutset_name=a.cutset, verbosity=0)
    inputs = list(a.inputs) + ([ln.strip() for ln in open(a.files) if ln.strip()] if a.files else [])
    parts = []
    for i, f in enumerate(inputs):
        r = select_file(f, analyse, CutManager, a.location, NARROWER[a.window] + (GATE if a.no_gate else ()))
        parts.append(r)
        print(f"{i + 1}/{len(inputs)} {os.path.basename(f)}: {len(r)} events, {int(ak.sum(ak.num(r.t_mid)))} tracks",
              flush=True)
    out = ak.concatenate(parts) if parts else ak.Array([])
    ak.to_parquet(out, a.out)
    print(f"wrote {a.out}: {len(out)} events, {int(ak.sum(ak.num(out.t_mid)))} selected tracks")


if __name__ == "__main__":
    main()
