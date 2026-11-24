"""python -m crvfastsim.design {muons, entrycmp, geometry, evaluate, closure, stockcmp}: CRV module design points
(docs/design.md)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import numpy as np

from .approximations import WORKFLOW

DTYPES = {"float32": np.float32, "float64": np.float64}


def _design(text):
    from .module import Design
    if text is None:
        return Design()
    if text.lstrip().startswith("{"):
        return Design.from_dict(json.loads(text))
    with open(text) as f:
        return Design.from_dict(json.load(f))


def cmd_muons(a):
    from mu2efastsim import tables
    from . import fom
    proxy = dict(fom.PROXY, r=a.proxy_r, z0=a.proxy_z[0], z1=a.proxy_z[1])
    cols = ["pdg", "x", "y", "z", "px", "py", "pz"]
    if a.device_cry:                                 # CRY and the overburden on the device, no gen tables
        from . import device_sample
        if a.gen or not a.world or not a.events:
            raise SystemExit("--device-cry takes no gen tables and needs --world and --events")
        m = device_sample.build(a.device_cry, a.world, a.counters, a.events, a.seed, chunk=a.chunk, proxy=proxy,
                                require_gpu=not a.allow_cpu, selected_capacity=a.selected_capacity,
                                transport_batch=a.transport_batch, backend=a.transport)
        device_sample.save(a.out, m, proxy)
        extra = {k: v for k, v in m.items() if k.startswith("n_") or k in ("y_plane", "live_seconds", "provenance")}
    elif a.world:                                    # CRV-entry states: world material + scattering to the MSC plane
        from . import device_sample, sample
        if not a.gen:
            raise SystemExit("muons needs gen tables (or --device-cry)")
        gen = tables.read_many(a.gen, "gen", tables.KEY + cols)
        m = sample.crv_entry_states(gen, a.world, a.counters, np.random.default_rng(a.seed), proxy,
                                    backend=a.transport)
        m["provenance"] = dict(gen=a.gen, seed=a.seed, transport=a.transport)
        device_sample.save(a.out, m, proxy)
        extra = {k: v for k, v in m.items() if k.startswith("n_") or k == "y_plane"}
    else:
        if not a.gen:
            raise SystemExit("muons needs gen tables")
        gen = tables.read_many(a.gen, "gen", cols)
        m = fom.muons_from_gen(gen, proxy)
        extra = {}
        np.savez(a.out, origin=m["origin"], direction=m["direction"], p=m["p"],
                 n_generated_muons=m["n_generated_muons"], proxy=json.dumps(proxy), states="generated")
    print(json.dumps(dict(selected=len(m["p"]), generated_muons=m["n_generated_muons"], out=a.out, **extra),
                     default=str))


def cmd_entrycmp(a):
    from . import device_sample
    _write(a.out, device_sample.entry_compare(a.samples, a.events or None))


def cmd_geometry(a):
    from . import module as M
    dg = M.build(_design(a.design), a.counters)
    rep = dict(design=dg.design.as_dict(), n_bars=int(dg.nbars), n_absorbers=int(len(dg.abs_sector)),
               module_pitch=dict(zip(dg.sector_names, dg.module_pitch.tolist())),
               modules=dict(zip(dg.sector_names, [int(dg.module[dg.sector == s].max()) + 1
                                                  for s in range(len(dg.sector_names))])))
    _write(a.out, rep)


def cmd_evaluate(a):
    from . import deadtime as DT
    from . import fom
    from . import optics as O
    cfg = fom.FomConfig(optics=O.OpticsConfig(pe_scale=a.pe_scale, alpha_g=a.alpha_g),
                        deadtime=DT.DeadtimeConfig(kappa=a.kappa, f_acc=a.f_acc, beta=a.beta),
                        aging_o1=a.aging_o1, aging_c1=a.aging_c1, aging_ref=a.aging_ref)
    z = np.load(a.muons)
    n = len(z["p"]) if a.max_muons <= 0 else min(a.max_muons, len(z["p"]))
    muons = dict(origin=z["origin"][:n], direction=z["direction"][:n], p=z["p"][:n])
    cal = DT.load_calibration(a.calibration) if a.calibration else None
    ctx = fom.Context(counters=a.counters, lookup_dir=a.lookup_dir)
    out = fom.evaluate(_design(a.design), muons, ctx, cfg, cal, DTYPES[a.dtype])
    out["config"] = dict(fom=dataclasses.asdict(cfg), muons=a.muons, counters=a.counters)
    out["approximations"] = WORKFLOW.select("default").record()
    _write(a.out, out)


def cmd_closure(a):
    from . import closure
    from . import fom
    from . import optics as O
    z = np.load(a.muons)
    n = len(z["p"]) if a.max_muons <= 0 else min(a.max_muons, len(z["p"]))
    muons = dict(origin=z["origin"][:n], direction=z["direction"][:n], p=z["p"][:n])
    cfg = fom.FomConfig(optics=O.OpticsConfig(pe_scale=a.pe_scale))
    ctx = fom.Context(counters=a.counters, lookup_dir=a.lookup_dir)
    out = closure.run(_design(a.design), muons, ctx, cfg, a.photon_map, a.mode, a.seed, DTYPES[a.dtype])
    _write(a.out, out)


def cmd_stockcmp(a):
    from . import fom
    from . import optics as O
    from . import stockcmp
    cfg = fom.FomConfig(optics=O.OpticsConfig(pe_scale=a.pe_scale))
    ctx = fom.Context(counters=a.counters, lookup_dir=a.lookup_dir)
    design = _design(a.design)
    stock = stockcmp.read_stock(a.stock)
    out = stockcmp.compare(stock, design, ctx, cfg, DTYPES[a.dtype], a.sectors)
    if a.geometry_check:
        out["geometry_check"] = stockcmp.geometry_check(a.stock, design, ctx)
    if a.gen:
        out["inefficiency"] = stockcmp.inefficiency(stock, a.gen, design, ctx, cfg, DTYPES[a.dtype])
    out["config"] = dict(stock=a.stock, counters=a.counters)
    out["approximations"] = WORKFLOW.select("default").record()
    _write(a.out, out)


def _write(path, obj):
    text = json.dumps(obj, indent=1, default=float)
    if path:
        with open(path, "w") as f:
            f.write(text + "\n")
    print(text)


def main(argv=None):
    from .module import RUN2_COUNTERS
    p = argparse.ArgumentParser(prog="python -m crvfastsim.design", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("muons", help="CE-like proxy muons from CRY gen tables")
    m.add_argument("gen", nargs="*")
    m.add_argument("--out", required=True)
    m.add_argument("--proxy-r", type=float, default=800.0)
    m.add_argument("--proxy-z", type=float, nargs=2, default=(5400.0, 11800.0))
    m.add_argument("--world", help="GDML world: carry the muons through the material above the CRV (stage C) and "
                                   "start them on the MSC plane (design.sample)")
    m.add_argument("--counters", default=RUN2_COUNTERS)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--transport", choices=("numpy", "jax"), default="numpy",
                   help="overburden transport: host reference or the device stage-C kernels (design.entry_jax)")
    m.add_argument("--device-cry", choices=("extracted", "run2"),
                   help="generate CRY on the device with this CosmicCRY preset instead of reading gen tables")
    m.add_argument("--events", type=int, help="--device-cry: accepted CRY events to generate")
    m.add_argument("--chunk", type=int, default=200000, help="--device-cry: accepted events per CRY chunk")
    m.add_argument("--selected-capacity", type=int, default=8192, help="--device-cry: selected rows per raw batch")
    m.add_argument("--transport-batch", type=int, default=4096)
    m.add_argument("--allow-cpu", action="store_true", help="--device-cry on the JAX CPU backend (tests only)")
    m.set_defaults(func=cmd_muons)
    ec = sub.add_parser("entrycmp", help="rate and spectra of CRV-entry samples against the first, per CRY event")
    ec.add_argument("samples", nargs="+", help="muons npz files; the first is the reference")
    ec.add_argument("--events", type=int, nargs="*", help="generated CRY events per sample, where the npz has none")
    ec.add_argument("--out")
    ec.set_defaults(func=cmd_entrycmp)
    g = sub.add_parser("geometry", help="build a design and summarise it")
    g.add_argument("--design")
    g.add_argument("--counters", default=RUN2_COUNTERS)
    g.add_argument("--out")
    g.set_defaults(func=cmd_geometry)
    e = sub.add_parser("evaluate", help="figures of merit of one design point")
    e.add_argument("--design", help="JSON object or file (module.Design keys); absent = nominal")
    e.add_argument("--muons", required=True, help="npz from the muons subcommand")
    e.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    e.add_argument("--counters", default=RUN2_COUNTERS)
    e.add_argument("--lookup-dir")
    e.add_argument("--calibration", help="deadtime calibration JSON (deadtime.load_calibration)")
    e.add_argument("--kappa", type=float, default=1.0)
    e.add_argument("--f-acc", type=float, default=0.5)
    e.add_argument("--beta", type=float, default=1.0)
    e.add_argument("--alpha-g", type=float, default=1.0)
    e.add_argument("--pe-scale", type=float, default=1.0)
    e.add_argument("--aging-o1", type=float, default=1.0)
    e.add_argument("--aging-c1", type=float, default=0.94 ** 5)
    e.add_argument("--aging-ref", type=float, default=0.86)
    e.add_argument("--max-muons", type=int, default=0)
    e.add_argument("--out")
    e.set_defaults(func=cmd_evaluate)
    c = sub.add_parser("closure", help="optics model vs the full photon + SiPM chain on the same chords")
    c.add_argument("--design")
    c.add_argument("--muons", required=True)
    c.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    c.add_argument("--counters", default=RUN2_COUNTERS)
    c.add_argument("--lookup-dir")
    c.add_argument("--photon-map")
    c.add_argument("--mode", choices=("mean", "sampled"), default="mean")
    c.add_argument("--pe-scale", type=float, default=1.0)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--max-muons", type=int, default=0)
    c.add_argument("--out")
    c.set_defaults(func=cmd_closure)
    s = sub.add_parser("stockcmp", help="paired comparison with a stock G4 + CRV reco sample of the same muons")
    s.add_argument("stock", nargs="+", help="TFileService files with dump/ and crvdump/ trees")
    s.add_argument("--design", help="the design the stock geometry implements; absent = nominal")
    s.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    s.add_argument("--counters", default=RUN2_COUNTERS)
    s.add_argument("--lookup-dir")
    s.add_argument("--pe-scale", type=float, default=1.0)
    s.add_argument("--sectors", nargs="*")
    s.add_argument("--gen", nargs="*", help="gen dumps of every generated event: adds the CE-proxy inefficiency")
    s.add_argument("--geometry-check", action="store_true", help="stock CrvSteps inside the design's bars")
    s.add_argument("--out")
    s.set_defaults(func=cmd_stockcmp)
    a = p.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
