"""W7 fast arm with the default shower source (:mod:`calofastsim.entrant_chain`): every front and inner-edge
calorimeter entrant of each stored event of CaloLibraryDumper exports -> per-entrant library draw -> crystal placement
-> device response (parametric default) -> hits -> clusters, written as one ``clusters.npz`` for the W7 scoring.

    python3 -m calofastsim.w7_arm OUT --tables EXPORT.root ... --stock STOCK.root ... --library LIB.npz \\
        --cal1-inputs DIR [--dtype float64] [--seed 20261107] [--batch 2048]

Per event the reco bunch time is the stock one (``evtinfo.pbtime`` of the stock EventNtuple, one ``--stock`` file per
export, same events) rounded to the 5 ns grid, as the front-only arm W7 was scored with. Events whose Geant4 transport
failed are dropped (:func:`entrants.read` ``skip_failed``) and counted in the census.

``clusters.npz``: ``ids`` (run, subrun, event) and ``energy``, ``time``, ``x``, ``y``, ``disk`` per cluster;
``all_ids`` every completed event; ``front_ids`` the frozen W7 population, the events with a primary disk-front entry
(:func:`showers.export.read`, primary mode), or, for an export with failed events (which that reader refuses), the
events with a front-face e- entrant (``summary.json`` says which). ``summary.json``: library path and content hash,
settings, overflow and draw counts, census.

The library must be W7's (:data:`entrant_chain.W7_LIBRARY_HASH`) unless ``--any-library`` is given. ``--d-edge`` > 0
leaves front entrants within D_edge of the disk inner or outer radius undrawn and passes their own Geant4
CaloShowerSteps from the export instead (the Mu2eG4 host's edge rule, for validation).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def stock_bunch_times(path):
    """(ids (n, 3), pbtime) of a stock EventNtuple's events."""
    import awkward as ak
    import uproot
    with uproot.open(path) as f:
        ev = f["prd/ntuple"].arrays(["evtinfo"])["evtinfo"]
    ids = np.column_stack([ak.to_numpy(ev[k]).astype(np.int64) for k in ("run", "subrun", "event")])
    return ids, ak.to_numpy(ev["pbtime"]).astype(np.float64)


def response(cal1_inputs, dtype):
    """The device response (:func:`device_chain.pack_response`) from the calorimeter conditions directory."""
    from . import device_chain as dc, stockio
    from .cluster import Geometry
    ci = Path(cal1_inputs)
    return dc.pack_response(stockio.load_sim_crystal(ci / "Sim_crystal.txt"),
                            stockio.load_pulse(ci / "CsI-waveform-2020-02-12.root"),
                            stockio.load_noise(ci / "ReadoutNoise.root"),
                            stockio.load_propagation(ci / "CsI-propag-2020-07-29.root"),
                            Geometry.load(ci / "calo_geometry_run1430.npz"), adc2mev=stockio.adc_to_mev,
                            dtype=np.dtype(dtype))


def run(a):
    import jax
    jax.config.update("jax_enable_x64", True)
    import uproot
    from mu2efastsim.library.hashing import content_hash

    from . import device_chain as dc, entrant_chain as ec
    from .showers import entrants as E, export

    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    library, _ = E.load(a.library)
    lib_hash = content_hash({k: v for k, v in library.items()})
    if lib_hash != ec.W7_LIBRARY_HASH and not a.any_library:
        raise SystemExit(f"{a.library}: content hash {lib_hash[:12]}... is not W7's library "
                         f"({ec.W7_LIBRARY_HASH[:12]}...); pass --any-library to run it anyway")
    settings = dict(ec.W7_DEFAULT, d_edge=a.d_edge)
    lib = ec.pack_entrants(library, dtype=np.dtype(a.dtype), **settings)
    resp = response(a.cal1_inputs, a.dtype)
    B = a.batch
    caps = dc.Caps(events=B, steps=B * 80, photoelectrons=B * 9000, rows=B * 48, digis=B * 48, digi_length=96,
                   hits=B * 32, event_hits=64, rounds=8)
    n_entrants, n_g4 = 2 * B, B * 64
    chain = ec.make_entrant_chain(lib, resp, caps, digi_response=a.digi_response)
    inner, outer = library["geometry_disk_size_x"], library["geometry_disk_size_y"]
    cpos_z = library["geometry_crystal_position_z"]
    key = jax.random.PRNGKey(a.seed)
    out_cl = {k: [] for k in ("energy", "time", "x", "y", "disk")}
    cl_ids, front_ids, all_ids, overflow, stats, census, front_def = [], [], [], {}, {}, [], []
    if len(a.tables) != len(a.stock):
        raise SystemExit("--tables and --stock need one file each per input file")
    for tables, stock in zip(a.tables, a.stock):
        ent, css, _, cen = E.read(tables, skip_failed=True)
        census.append(dict(cen, file=str(tables)))
        with uproot.open(tables) as f:
            evt = f["dump/events"].arrays(["run", "subrun", "event", "g4_status"], library="np")
        evt = {k: v[evt["g4_status"] == 0] for k, v in evt.items()}
        ids = np.column_stack([evt[k].astype(np.int64) for k in ("run", "subrun", "event")])
        if cen["failed_events_skipped"]:
            front_def.append("front-face e- entrant")
            front_ids.append(ids[np.unique(ent["event_row"][(ent["face"] == E.FRONT) & (ent["pdg"] == 11)])])
        else:
            front_def.append("primary disk-front entry")
            prim = export.read(tables, input_mode="primary")[0]
            front_ids.append(ids[np.unique(prim["event_row"])])
        all_ids.append(ids)
        sids, spbt = stock_bunch_times(stock)
        row_of = {tuple(r): i for i, r in enumerate(sids)}
        srow = np.array([row_of[tuple(r)] for r in ids])
        pbt_mc = (np.round(spbt / 5.0) * 5.0).astype(np.float32)
        slot = np.searchsorted(np.asarray(library["geometry_disk_disk"]), ent["disk"])
        edge = ec.near_edge(ent["face"], ent["local_position"], inner[slot], outer[slot], a.d_edge)
        g4_rows = np.flatnonzero(edge[css["entrant"]])
        n_ev = len(ids)
        order = np.argsort(ent["event_row"], kind="stable")
        bounds = np.searchsorted(ent["event_row"][order], np.arange(0, n_ev + B, B))
        for bi, b0 in enumerate(range(0, n_ev, B)):
            m = min(n_ev, b0 + B) - b0
            sel = order[bounds[bi]:bounds[bi + 1]]
            if len(sel) > n_entrants:
                raise SystemExit(f"{len(sel)} entrants in a batch of {B} events (capacity {n_entrants})")
            e = dict(event=ent["event_row"][sel] - b0, pdg=ent["pdg"][sel], face=ent["face"][sel],
                     disk=ent["disk"][sel], time=ent["time"][sel], local_position=ent["local_position"][sel],
                     local_momentum=ent["local_momentum"][sel])
            rows = srow[b0:b0 + m]
            timing = {"pbtime": pbt_mc[rows], "mbtime": np.full(m, 1695.0, np.float32),
                      "event_length": np.full(m, 1700.0), "onspill": np.ones(m, bool),
                      "marker": np.full(m, 200.0, np.float32), "crystal_length": np.full(m, 200.0, np.float32)}
            q, t = ec.entrants_batch(e, timing, n_entrants, B)
            g4 = None
            if a.d_edge > 0:
                ev_of = ent["event_row"][css["entrant"][g4_rows]]
                r = g4_rows[(ev_of >= b0) & (ev_of < b0 + m)]
                g4 = dc.steps_batch(dict(event=ent["event_row"][css["entrant"][r]] - b0, crystal=css["crystal"][r],
                                         energy_g4=css["energy_g4"][r], energy_birks=css["energy_birks"][r],
                                         time=css["time"][r],
                                         z_local=css["position"][r, 2] - cpos_z[css["crystal"][r]]), n_g4)
            key, sub = jax.random.split(key)
            res = ec.run(chain, q, t, sub, g4=g4, tolerate_overflow=True)
            for k, v in res["overflow"].items():
                overflow[k] = overflow.get(k, 0) + int(v)
            for k in ec.STAT_NAMES:
                stats[k] = stats.get(k, 0) + int(res["stats"][k])
            fl = dc.flat_clusters(res["clusters"])
            keep = fl["event"] < m
            cl_ids.append(ids[b0 + fl["event"][keep]])
            for k in out_cl:
                out_cl[k].append(np.asarray(fl[k])[keep])
        print(f"{Path(tables).name}: {n_ev} events, {len(ent['face'])} entrants; {time.time() - t0:.0f}s", flush=True)
    summary = dict(library=str(a.library), content_hash=lib_hash, w7_library=lib_hash == ec.W7_LIBRARY_HASH,
                   settings={k: v for k, v in settings.items() if k != "psi_bins"}, dtype=a.dtype, seed=a.seed,
                   digi_response=a.digi_response, batch=B, front_definition=front_def, overflow=overflow, stats=stats,
                   census=census, seconds=time.time() - t0)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    np.savez_compressed(a.out / "clusters.npz", ids=np.concatenate(cl_ids), front_ids=np.concatenate(front_ids),
                        all_ids=np.concatenate(all_ids), dtype=a.dtype, seed=a.seed,
                        **{k: np.concatenate(v) for k, v in out_cl.items()})
    if any(overflow.values()):
        print("OVERFLOW", overflow, flush=True)
    print(f"wrote {a.out}: {sum(len(i) for i in all_ids)} events, {sum(len(i) for i in cl_ids)} clusters, "
          f"{time.time() - t0:.0f}s", flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--tables", type=Path, nargs="+", required=True, help="CaloLibraryDumper exports")
    p.add_argument("--stock", type=Path, nargs="+", required=True, help="stock EventNtuple per export (same events)")
    p.add_argument("--library", type=Path, required=True, help="per-entrant library npz (W7's unless --any-library)")
    p.add_argument("--cal1-inputs", type=Path, required=True, help="calorimeter conditions directory")
    p.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    p.add_argument("--digi-response", default="parametric")
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--d-edge", type=float, default=0.0)
    p.add_argument("--any-library", action="store_true", help="allow a library other than W7's")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    main()
