"""CLI of the fast CE EventNtuple chain (opt-in). Stages, in order:

  python3 -m beamfastsim.centuple record   --out DIR (--states FILE ... | --stops NPZ --electrons N) [chain inputs]
  python3 -m beamfastsim.centuple trigger  --digis DIR/centuple_digis.npz --menu replay.fcl --conditions trgdump.root
                                           --tc-mva TimeCluster.weights.xml --response response.json --out DIR
  python3 -m beamfastsim.centuple donor    --files nts.*.root --covered DIR/centuple_trigger.npz --out donor.npz
  python3 -m beamfastsim.centuple assemble --record DIR/centuple_record.npz --trigger DIR/centuple_trigger.npz
                                           --donor donor.npz --out DIR
  python3 -m beamfastsim.centuple write    --plan DIR/centuple_plan.npz --donor-files nts.*.root --out nts.root

record and trigger run in the fast-simulation image on a GPU (trigger in its own process); donor and assemble need
NumPy, awkward and uproot; write needs PyROOT with the EventNtuple dictionaries of the stock release.
"""
import argparse
import json
import sys


def _births(a):
    import numpy as np
    from . import record
    if a.states:
        from mu2efastsim import tables
        g = tables.read_many(a.states, "gen")
        g = g.sort_values(tables.KEY).reset_index(drop=True)
        for k in tables.KEY:
            g[k] = g[k].astype(np.int64)
        if a.electrons:
            g = g.iloc[:a.electrons].reset_index(drop=True)
        return g
    st = dict(np.load(a.stops))
    return record.births_frame(st, a.electrons, a.seed, run_number=a.run)


def cmd_record(a):
    from . import record
    b = _births(a)
    paths, cnt = record.run(b, a.out, a.seed, batch=a.batch, offline_root=a.offline_root, maps=a.maps, gdml=a.gdml,
                            library=a.library, cal1=a.cal1, digi_cond=a.digi_cond, geom=a.geom, dtype=a.dtype,
                            response=a.response, edge_library=a.edge_library)
    print(json.dumps(dict(paths={k: str(v) for k, v in paths.items()}, **cnt)))


def cmd_trigger(a):
    from . import trigger
    path, meta = trigger.run(a.digis, a.out, menu=a.menu, conditions=a.conditions, tc_mva=a.tc_mva,
                             response=a.response, precision=a.precision, block=a.block, spill=a.spill)
    print(json.dumps(dict(path=str(path), **{k: meta[k] for k in ("events", "wall_s", "precision")},
                          paths=len(meta["paths"]))))


def cmd_donor(a):
    import numpy as np
    from . import donor
    if a.covered.endswith(".npz"):
        covered = sorted(k[5:] for k in np.load(a.covered).files if k.startswith("trig_"))
    else:
        covered = [p for p in a.covered.split(",") if p]
    print(json.dumps(donor.build(a.files, covered, a.out)))


def cmd_assemble(a):
    from . import assemble
    path, meta = assemble.build(a.record, a.donor, a.out, trigger=a.trigger, seed=a.seed)
    print(json.dumps(dict(path=str(path), classes=meta["classes"], covered=len(meta["covered_paths"]),
                          missing=meta["covered_missing"])))


def cmd_write(a):
    from . import writer
    st = writer.write(a.plan, a.donor_files, a.out, n_generated=a.n_generated, check_sha256=not a.no_check)
    print(json.dumps(st))


# fixed names of the point's inputs under --data (the kit task's data directory)
DATA = dict(library="shower_library.npz", edge="edge_library.npz", gdml="tracker.gdml", cal1="calo_response",
            digi_cond="digicond_onspill.npz", menu="trigger/replay.fcl", conditions="trigger/trgdump.root",
            tc_mva="trigger/TimeCluster.weights.xml", trig_response="trigger/response.json", donor="donor/donor.npz")
FILE = "nts.mu2efastsim.centuple.root"


def summary(path, triggers=("apr_TrkDe_80m70p", "cpr_TrkDe_80m70p")):
    """ce_chain-style counts of a written file: events, events with a track passing the analysis server's default
    trigger_efficiency_ntuple selection (status 1, e-, downstream at TT_Mid, p_front > 80 MeV/c, >= 15 active hits,
    chi2/ndof < 5), and those of them accepted by any of `triggers`."""
    import awkward as ak
    import numpy as np
    import uproot
    t = uproot.open(path)["EventNtuple/ntuple"]
    a = t.arrays(["trk.status", "trk.pdg", "trk.nactive", "trk.chisq", "trk.ndof", "trksegs"]
                 + ["trig_" + p for p in triggers], library="ak")
    seg = a["trksegs"]
    mom = seg["mom"]["fCoordinates"]
    p = np.sqrt(mom["fX"] ** 2 + mom["fY"] ** 2 + mom["fZ"] ** 2)
    front = (seg["sid"] == 0) & (mom["fZ"] > 0)
    pf = ak.firsts(p[front], axis=-1)
    dn = ak.firsts(mom["fZ"][seg["sid"] == 1], axis=-1) > 0
    sel = ((a["trk.status"] == 1) & (a["trk.pdg"] == 11) & ak.fill_none(dn, False) & ak.fill_none(pf > 80, False)
           & (a["trk.nactive"] >= 15) & (a["trk.chisq"] / a["trk.ndof"] < 5))
    ev = ak.to_numpy(ak.any(sel, axis=1))
    trig = np.zeros(len(ev), bool)
    for q in triggers:
        trig |= ak.to_numpy(a["trig_" + q]).astype(bool)
    n_sel, n_trig = int(ev.sum()), int((ev & trig).sum())
    return dict(n_events=float(t.num_entries), n_selected=float(n_sel), n_triggered=float(n_trig),
                efficiency=n_trig / n_sel if n_sel else float("nan"))


def cmd_point(a):
    """One design point: beam stops of the target (beamfastsim point), CE births, record, trigger, assemble, write;
    metrics.json for the kit."""
    import os
    import subprocess
    import time
    from pathlib import Path
    import numpy as np
    from .. import cesob
    from ..__main__ import main as beam_main, MAPS, OFFLINE_ROOT
    from ..edepana import _mu_minus_stops
    from . import assemble, record
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = {k: os.path.join(a.data, v) for k, v in DATA.items()}
    t0 = time.time()
    stops = str(out / "stops.npz")
    argv = ["point", "--g4", a.g4, "--out", str(out), "--n", str(a.muons), "--seed", str(a.seed), "--repeat", "0",
            "--batch", "131072", "--chunk", "400", "--stops-out", stops]
    argv += ["--geom", a.geom] if a.geom else []
    argv += ["--z-up", str(a.z_up)] if a.z_up is not None else []
    beam_main(argv)
    bm = json.load(open(out / "metrics.json"))
    g4 = np.load(a.g4)
    stops_per_gen = bm["stop_frac"] * int(g4["evt"].size) / float(g4["events"])
    t_beam = time.time() - t0
    births = record.births_frame(_mu_minus_stops(stops), a.electrons, a.seed, run_number=a.run)
    maps = a.maps or [f"{MAPS}/DSMap.header", f"{MAPS}/TSdMap.header"]
    _, cnt = record.run(births, out, a.seed, batch=a.batch, offline_root=a.offline_root or OFFLINE_ROOT, maps=maps,
                        gdml=d["gdml"], library=d["library"], cal1=d["cal1"], digi_cond=d["digi_cond"], geom=a.geom,
                        dtype=a.dtype, edge_library=d["edge"])
    py = [sys.executable, "-m", "beamfastsim.centuple"]
    subprocess.run(py + ["trigger", "--digis", str(out / record.DIGIS), "--out", str(out), "--menu", d["menu"],
                         "--conditions", d["conditions"], "--tc-mva", d["tc_mva"], "--response", d["trig_response"],
                         "--precision", a.dtype], check=True)
    plan, pmeta = assemble.build(out / record.RECORD, d["donor"], out, trigger=out / "centuple_trigger.npz",
                                 seed=a.seed)
    files = [os.path.join(a.data, "donor", f) for f in pmeta["donor_files"]]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    here = Path(__file__).resolve().parent
    subprocess.run(["bash", str(here / "mu2e_env.sh"), "python3", "-m", "beamfastsim.centuple", "write", "--plan",
                    str(plan), "--donor-files", *files, "--out", str(out / FILE), "--n-generated",
                    str(a.electrons)], check=True, env=env, cwd=out)
    m = summary(out / FILE)
    m.update(n_generated=float(a.electrons), births=float(cnt["events"]), wall_s=time.time() - t0,
             electrons_per_s=a.electrons / max(time.time() - t0 - t_beam, 1e-9), beam_s=t_beam,
             stops_per_gen_event=stops_per_gen, stops_per_pot=stops_per_gen * cesob.UPSTREAM_EFF,
             stop_frac=bm["stop_frac"], **{f"class_{k}": float(v) for k, v in pmeta["classes"].items()})
    (out / "metrics.json").write_text(json.dumps({k: float(v) for k, v in m.items()}) + "\n")
    if not a.keep:
        for f in (stops, out / record.DIGIS):
            if os.path.exists(f):
                os.remove(f)
    print(json.dumps(m))


def main(argv=None):
    p = argparse.ArgumentParser(prog="beamfastsim.centuple", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--states", nargs="*", help="cetracker states.root files: their gen trees are the births")
    r.add_argument("--stops", help="mu- stop table (npz x, y, z, t): CeEndpoint births drawn from it")
    r.add_argument("--electrons", type=int, default=0, help="births to draw (--stops) or keep (--states; 0 all)")
    r.add_argument("--run", type=int, default=1, help="run number of births drawn from --stops")
    r.add_argument("--seed", type=int, default=20261111)
    r.add_argument("--batch", type=int, default=20000)
    r.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    r.add_argument("--offline-root", required=True)
    r.add_argument("--maps", nargs=2, required=True)
    r.add_argument("--gdml", required=True)
    r.add_argument("--library", required=True, help="calorimeter shower library (.npz)")
    r.add_argument("--cal1", required=True, help="directory of the calorimeter response inputs (record.CALO_FILES)")
    r.add_argument("--digi-cond", required=True, help="tracker digitiser conditions (.npz), on-spill")
    r.add_argument("--edge-library", help="Geant4 inner-edge calorimeter table (edepana EdgeEnergies); omitted: no edge clusters")
    r.add_argument("--response", default="default")
    r.add_argument("--geom")
    r.set_defaults(fn=cmd_record)
    t = sp.add_parser("trigger")
    t.add_argument("--digis", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--menu", required=True)
    t.add_argument("--conditions", required=True, help="a trigger dump (trgdump.root) for the hit conditions")
    t.add_argument("--tc-mva", required=True)
    t.add_argument("--response", help="KalSeedFilter response layer (trigger/device/run.py response)")
    t.add_argument("--precision", default="float64", choices=("float32", "float64"))
    t.add_argument("--block", type=int, default=1024)
    t.add_argument("--spill", type=int, default=1, help="EventWindowMarker spill type (1 on-spill)")
    t.set_defaults(fn=cmd_trigger)
    d = sp.add_parser("donor")
    d.add_argument("--files", nargs="+", required=True)
    d.add_argument("--covered", required=True, help="a centuple_trigger.npz (its paths) or a comma list of paths")
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_donor)
    s = sp.add_parser("assemble")
    s.add_argument("--record", required=True)
    s.add_argument("--trigger")
    s.add_argument("--donor", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=20261111)
    s.set_defaults(fn=cmd_assemble)
    w = sp.add_parser("write")
    w.add_argument("--plan", required=True)
    w.add_argument("--donor-files", nargs="+", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--n-generated", type=int)
    w.add_argument("--no-check", action="store_true", help="skip the donor files' sha256 check")
    w.set_defaults(fn=cmd_write)
    q = sp.add_parser("point", help="one design point: beam stops -> CE births -> record, trigger, assemble, write")
    q.add_argument("--g4", required=True, help="beam start table with its generated events (g4table --events)")
    q.add_argument("--data", required=True, help="directory of the point inputs (DATA names)")
    q.add_argument("--out", required=True)
    q.add_argument("--geom")
    q.add_argument("--z-up", dest="z_up", type=float)
    q.add_argument("--muons", type=int, default=0)
    q.add_argument("--electrons", type=int, default=100000)
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--run", type=int, default=1)
    q.add_argument("--batch", type=int, default=20000)
    q.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    q.add_argument("--offline-root")
    q.add_argument("--maps", nargs=2)
    q.add_argument("--keep", action="store_true")
    q.set_defaults(fn=cmd_point)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
