"""B1 (CALOG4) device-host arm: the fastsim calorimeter device chain on library entries -> hists.json + features.npz.

The arm W7-CAL validated: at the first forward direct-CE disk-front entry of each stored event (calofastsim
showers.export, primary mode), a whole-shower library draw and crystal placement (``device_chain.sample_donors``,
``gather_steps``, ``place_steps``), then the device chain's response, hits and clusters on the placed steps
(``device_chain.make_steps_chain(..., shape=True)``): the stages of ``make_chain``, with its own random streams.
The draw and placement run op by op (``--draw eager``, the default): jitted, the K-neighbour radius draw of
``sample_donors`` does not compile within an hour on CPU, while op by op it takes well under a second per batch;
``--draw jit`` uses ``make_chain`` itself (GPU). The reco bunch time of each event is the stock one rounded to the
5 ns grid (MC bunch time), as in W7-CAL. In float64 the step depth enters the response as float32
(``make_steps_chain`` input), as for stock steps. The population is every exported event with a front
entry, in export order; ``events.npy`` (run, subrun, event) beside the output fixes it, and the stock arm is made on
the same events with ``python -m benchmarks.fastsim_score.calo ntuple REF/CALOG4/stock STOCK.root ... --events
events.npy``.

    python -m benchmarks.fastsim_score.calo_device OUT --tables EXPORT.root ... --stock STOCK.root ...
        --library library.npz --cal1-inputs DIR [--dtype float64] [--response parametric] [--seed 20261107]

Needs calofastsim (calo/ on PYTHONPATH), JAX and uproot; the scorer itself does not.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.fastsim_score import calo


def stock_ids(path):
    import awkward as ak
    import uproot
    with uproot.open(path) as f:
        ev = f["prd/ntuple"].arrays(["evtinfo"])["evtinfo"]
    ids = np.column_stack([ak.to_numpy(ev[k]).astype(np.int64) for k in ("run", "subrun", "event")])
    return ids, ak.to_numpy(ev["pbtime"]).astype(np.float64)


def fast(a):
    import jax
    jax.config.update("jax_enable_x64", True)
    import uproot
    from calofastsim import device_chain as dc, stockio
    from calofastsim.cluster import Geometry
    from calofastsim.showers import export
    from calofastsim.showers.library import load

    t0 = time.time()
    dt = np.dtype(a.dtype)
    library, _ = load(a.library)
    lib = dc.pack_library(library, radius_neighbors=a.neighbors, placement=a.placement, dtype=dt)
    ci = a.cal1_inputs
    resp = dc.pack_response(stockio.load_sim_crystal(ci / "Sim_crystal.txt"),
                            stockio.load_pulse(ci / "CsI-waveform-2020-02-12.root"),
                            stockio.load_noise(ci / "ReadoutNoise.root"),
                            stockio.load_propagation(ci / "CsI-propag-2020-07-29.root"),
                            Geometry.load(ci / "calo_geometry_run1430.npz"), adc2mev=stockio.adc_to_mev, dtype=dt)
    B = a.batch
    caps = dc.Caps(events=B, steps=B * 80, photoelectrons=B * 9000, rows=B * 48, digis=B * 48, digi_length=96,
                   hits=B * 32, event_hits=64, rounds=8)
    import jax.numpy as jnp
    if a.draw == "jit":
        chain = dc.make_chain(lib, resp, caps, digi_response=a.response, shape=True)
    else:
        chain = dc.make_steps_chain(resp, caps, digi_response=a.response, shape=True)
    pdt = lib.arrays["centers"].dtype

    def placed_steps(q, key):
        """Library draw and placement op by op: the steps make_chain would pass to its response."""
        with jax.disable_jit():
            qd = {k: jnp.asarray(v) for k, v in q.items()}
            qd = dict(qd, position=qd["position"].astype(pdt), direction=qd["direction"].astype(pdt))
            u = jax.random.uniform(key, qd["energy"].shape, dtype=jnp.float64)
            donor, _, slot = dc.sample_donors(lib, qd, u)
            st, over = dc.gather_steps(lib, qd, donor, slot, caps.steps)
            crystal, depth, mapped = dc.place_steps(lib, qd, st)
            steps = dict(event=qd["event"][st["row"]], crystal=jnp.where(mapped, crystal, -1),
                         energy_g4=st["energy_g4"], energy_birks=st["energy_birks"], time=st["time"],
                         z_local=depth.astype(jnp.float32))
            return {k: np.asarray(v) for k, v in steps.items()}, int(over)

    key = jax.random.PRNGKey(a.seed)
    ids_all, cls, hts, overflow, off = [], [], [], {}, 0
    if len(a.tables) != len(a.stock):
        raise SystemExit("--tables and --stock need one file each per input file")
    for tables, stock in zip(a.tables, a.stock):
        entries, _, _, _, _ = export.read(tables, input_mode="primary")
        with uproot.open(tables) as f:
            evt = f["dump/events"].arrays(["run", "subrun", "event"], library="np")
        ids = np.column_stack([evt[k].astype(np.int64) for k in ("run", "subrun", "event")])
        sids, spbt = stock_ids(stock)
        key_s = {tuple(r): i for i, r in enumerate(sids)}
        srow = np.array([key_s[tuple(r)] for r in ids])
        pbt_mc = (np.round(spbt / 5.0) * 5.0).astype(np.float32)
        fe = np.asarray(entries["event_row"])
        if len(np.unique(fe)) != len(fe):
            raise RuntimeError("more than one front entry per event")
        n = len(fe)
        ids_all.append(ids[fe])
        for b0 in range(0, n, B):
            sl = slice(b0, min(n, b0 + B))
            m = sl.stop - sl.start
            ent = {k: np.asarray(v)[sl] for k, v in entries.items() if k != "event_row"}
            rows = srow[fe[sl]]
            timing = {"pbtime": pbt_mc[rows], "mbtime": np.full(m, 1695.0, np.float32),
                      "event_length": np.full(m, 1700.0), "onspill": np.ones(m, bool),
                      "marker": np.full(m, 200.0, np.float32), "crystal_length": np.full(m, 200.0, np.float32)}
            q, t = dc.entries_batch(ent, timing, np.arange(m), B)
            key, sub = jax.random.split(key)
            if a.draw == "jit":
                out = dc.run(chain, q, t, sub, tolerate_overflow=True)
            else:
                k_draw, k_resp = jax.random.split(sub)
                steps, over_steps = placed_steps(q, k_draw)
                out = dc.run(chain, steps, t, k_resp, tolerate_overflow=True)
                out["overflow"]["steps"] += over_steps
            for k, v in out["overflow"].items():
                overflow[k] = overflow.get(k, 0) + int(v)
            fl = dc.flat_clusters(out["clusters"])
            keep = fl["event"] < m
            c = {k: np.asarray(v)[keep] for k, v in fl.items()}
            c["event"] = c["event"] + off + b0
            cls.append(c)
            h = out["hits"]
            hk = np.asarray(h["ok"]) & (np.asarray(h["event"]) < m)
            hts.append({"event": np.asarray(h["event"])[hk] + off + b0, "energy": np.asarray(h["e"])[hk]})
        off += n
        print(f"{tables.name}: {n} front-entry events; {time.time() - t0:.0f}s", flush=True)
    ids = np.concatenate(ids_all)
    cl = {k: np.concatenate([c[k] for c in cls]) for k in calo.CLUSTER_KEYS}
    ht = {k: np.concatenate([h[k] for h in hts]) for k in ("event", "energy")}
    hists, feats = calo.arm(cl, ht, len(ids), a.block)
    out = Path(a.out)
    calo.write_arm(out, hists, feats)
    np.save(out / "events.npy", ids)
    (out / "run.json").write_text(json.dumps({"dtype": a.dtype, "response": a.response, "seed": a.seed,
                                              "neighbors": a.neighbors, "placement": a.placement, "draw": a.draw,
                                              "batch": B, "events": int(len(ids)), "clusters": int(len(cl["event"])),
                                              "overflow": overflow, "seconds": time.time() - t0}, indent=1) + "\n")
    if any(overflow.values()):
        print("OVERFLOW", overflow, flush=True)
    print(f"wrote {out}: {len(ids)} events, {len(cl['event'])} clusters, {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--tables", type=Path, nargs="+", required=True, help="CaloLibraryDumper exports (primary mode)")
    p.add_argument("--stock", type=Path, nargs="+", required=True,
                   help="stock calo ntuples of the same files (event ids and reco bunch times)")
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--cal1-inputs", type=Path, required=True,
                   help="Sim_crystal.txt, CsI waveform, ReadoutNoise, CsI propagation, calo_geometry_run1430.npz")
    p.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    p.add_argument("--response", default="parametric", help="device_chain digi_response (parametric, waveform, ...)")
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--placement", default="crystal")
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--block", type=int, default=100)
    p.add_argument("--draw", default="eager", choices=("eager", "jit"),
                   help="library draw + placement op by op (CPU) or inside make_chain's jit (GPU)")
    fast(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
