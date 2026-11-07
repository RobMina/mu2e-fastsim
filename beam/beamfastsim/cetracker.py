"""Conversion electrons from their birth to tracker StrawGasSteps with the field on: cechain -> fieldsteps (opt-in).

beamfastsim.cechain carries each electron to the tracker entry plane (VD TT_FrontHollow, z = 8539.9 mm); its state
there (position, direction, kinetic energy, global time; Mu2e frame) becomes one start row of the field-on tracker
transport trkfastsim.sim.fieldsteps (position, momentum, time, PDG 11), with the event identity of the start it came
from. Electrons with any other cechain fate get no tracker start and stay in the event denominator. The entry plane
lies inside the fieldsteps envelope, so the tracker transport starts from the hand-off state unchanged. (In the Run 1A
MDC2025ba geometry the Geant4 VD13 is at z = 8539.48 mm, 0.42 mm upstream of the cechain plane, both in vacuum.)

The tracker transport writes StrawGasSteps rows; `write_injector` puts them in the table layout of the art module
StrawGasStepsFromTable (one primary per event, SimParticle key 1), for stock digitisation and reconstruction.
First passage only: no secondaries, no return after the tracker envelope.

  python3 -m beamfastsim.cetracker TABLE.root [...] --offline-root OFF --maps DSMap.header TSdMap.header \
      --gdml mu2e_run1a_v01.gdml --outdir DIR [--backend numpy|device] [--seed S] [--max-events N]

TABLE: StrawGasStepTableDumper tables of Geant4 CeEndpoint events (trees dump/gen, dump/particles); the generated
electron of each event is the start. Instead of tables, `--starts FILE` takes the births from a generator file in the
cechain reference format, e.g. DIO (`beamfastsim.diospec starts`) or RPC (`beamfastsim.rpcspec`) electrons, one event
per row (`--shards FIRST LAST` keeps a range of diospec shards). Writes DIR/fasttrk_field.root (injector table), DIR/states.root (starts,
hand-off states, final tracker states), DIR/cechain.npz and DIR/handoff.json (counts and timings).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import eioni, tables

from . import cechain

KEY = tables.KEY
STATE = ["x", "y", "z", "px", "py", "pz", "t"]
PDG_E = 11
ORIGIN = (-3904.0, 0.0, 10171.0)          # tracker (detector) origin in the Mu2e frame, Run 1A [mm]
B_DS_MIN = 0.5                            # [T] the first map must give at least this field at ORIGIN (DS map)
STEP_COLS = KEY + ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len",
                   "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]


def starts(paths):
    """The generated electron of each event (Mu2e frame), one row per event, from the dumper tables: the electron in the
    gen tree, or, for an event whose gen tree holds no electron (CeEndpoint dts keep the stopped muon there), the
    birth state of its primary electron SimParticle."""
    g = tables.read_many(paths, "gen")
    for k in KEY:
        g[k] = g[k].astype(np.uint64)
    events = g[KEY].drop_duplicates()
    g = g[g.pdg == PDG_E]
    missing = events.merge(g[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    missing = missing[missing._merge == "left_only"][KEY]
    if len(missing):
        p = tables.read_many(paths, "particles")
        for k in KEY:
            p[k] = p[k].astype(np.uint64)
        p = p[(p.pdg == PDG_E) & ((p.isprim == 1) | (p.ismcprim == 1))].rename(
            columns={"sx": "x", "sy": "y", "sz": "z", "spx": "px", "spy": "py", "spz": "pz", "st": "t"})
        g = pd.concat([g[KEY + ["pdg"] + STATE], p.merge(missing, on=KEY)[KEY + ["pdg"] + STATE]], ignore_index=True)
    if g.empty:
        raise ValueError("no generated electrons in the tables")
    g = g[KEY + ["pdg"] + STATE].copy()
    if g.duplicated(KEY).any():
        raise ValueError("more than one generated electron in an event")
    return g.sort_values(KEY).reset_index(drop=True)


def births_from_stops(rng, stops, keys):
    """One conversion electron per row of `keys` (KEY columns of carrier events), drawn as the CeEndpoint generator
    does: a stop resampled with replacement from `stops` (npz with x, y, z [mm, Mu2e frame] and t, the stop time [ns]),
    isotropic direction, |p| = cechain.P_CE_AL, time = stop time + Exp(cechain.TAU_MU_AL). Not filtered by any
    detector selection, so every row is a generated electron."""
    xs = np.stack([np.asarray(stops[c], float) for c in ("x", "y", "z")], axis=1)
    x, u, p, t, _ = cechain.from_stops(rng, xs, np.asarray(stops["t"], float), len(keys))
    g = keys[KEY].reset_index(drop=True).copy()
    g["pdg"] = PDG_E
    g[["x", "y", "z"]] = x
    g[["px", "py", "pz"]] = u * p[:, None]
    g["t"] = t
    return g


def births_from_starts(src, shards=None):
    """Births from a generator file in the cechain reference format (x, y, z [mm, Mu2e frame], px, py, pz [MeV/c],
    time [ns]): the starts `beamfastsim.diospec starts` writes (DIO), or the output of `beamfastsim.rpcspec` (RPC, the
    electron of each pair; the positron is not transported). One event per row, so every row is a generated electron.

    Event identities: run 0, subrun = the row's shard (`shard` column, e.g. diospec or rpcstudy) or 0, event = 1, 2,
    ... within the subrun in row order. Rows keep their order.
    shards: optional (first, last) inclusive range of diospec shards to keep."""
    for k in ("x", "y", "z", "px", "py", "pz", "time"):
        if k not in src:
            raise ValueError(f"generator file has no column {k!r} (cechain reference format expected)")
    n = np.asarray(src["x"]).size
    if "pdg" in src and np.any(np.asarray(src["pdg"]) != PDG_E):
        raise ValueError("generator file holds particles other than electrons")
    sub = np.asarray(src["shard"], np.int64) if "shard" in src else np.zeros(n, np.int64)
    keep = np.ones(n, bool)
    if shards is not None:
        if "shard" not in src:
            raise ValueError("--shards needs a generator file with a shard column (diospec starts)")
        keep = (sub >= shards[0]) & (sub <= shards[1])
    order = np.argsort(sub, kind="stable")
    event = np.empty(n, np.int64)
    event[order] = np.arange(n) - np.searchsorted(sub[order], sub[order], side="left") + 1
    g = pd.DataFrame({"run": np.zeros(n, np.uint64), "subrun": sub.astype(np.uint64),
                      "event": event.astype(np.uint64), "pdg": PDG_E})
    for k in ("x", "y", "z", "px", "py", "pz"):
        g[k] = np.asarray(src[k], float)
    g["t"] = np.asarray(src["time"], float)
    return g[keep].reset_index(drop=True)


def handoff(keys, result, plane_fate=cechain.PLANE):
    """fieldsteps start rows (KEY, sp, pdg, x, y, z, px, py, pz, t) of the electrons that reached the entry plane.

    keys: DataFrame with the KEY columns, row i = start i of the cechain result (rows are matched by position)."""
    if len(keys) != len(result.fate):
        raise ValueError("cechain result and start rows differ in length")
    ok = np.asarray(result.fate) == plane_fate
    out = keys.loc[ok, KEY].reset_index(drop=True)
    T = np.asarray(result.T)[ok]
    p = np.sqrt(T * (T + 2.0 * eioni.ME))
    u = np.asarray(result.u)[ok]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    out[["x", "y", "z"]] = np.asarray(result.x)[ok]
    out[["px", "py", "pz"]] = u * p[:, None]
    out["t"] = np.asarray(result.t)[ok]
    out["sp"], out["pdg"] = 1, PDG_E
    return out[KEY + ["sp", "pdg"] + STATE]


def run_cechain(g, geom, maps, seed=1, backend="numpy", step=10.0, device_float="float64", lanes=None):
    """cechain on the starts g (KEY + STATE columns): numpy (host loop) or device (cechain_jax). Returns the Result."""
    x = g[["x", "y", "z"]].to_numpy(float)
    pv = g[["px", "py", "pz"]].to_numpy(float)
    p = np.linalg.norm(pv, axis=1)
    u = pv / p[:, None]
    t0 = g["t"].to_numpy(float)
    if backend == "device":
        from types import SimpleNamespace
        a = SimpleNamespace(maps=maps, seed=seed, chunk=0, step=step, no_msc=False, no_layer_field=False,
                            z_record=None, flat_shells=False, device_float=device_float, batch=0, lanes=lanes,
                            chunk_iters=256, min_bucket=64, shrink=2, cand_frac=0.125, slab_frac=0.5, unroll=False)
        res, _ = cechain._device(a, geom, x, u, p, t0)
        return res
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    return cechain.transport(rng, geom, cechain.field_maps(maps), x, u, p, t0, h=step)


def tracker_steps(start, gdml, maps, seed=1, origin=ORIGIN, chord_mm=10.0):
    """Field-on tracker StrawGasSteps of the hand-off starts (trkfastsim.sim.fieldsteps, electron loss model, the
    tracker support from the GDML). Uses the tracker default StrawGas material (gas state). The tracker field is the
    first map of `maps` alone, which must be the DS map (checked: |B| at the origin >= B_DS_MIN). Returns
    (steps, final)."""
    from mu2efastsim import ehelix
    from trkfastsim import geometry
    from trkfastsim.sim import fieldsteps as fs
    geom = geometry.from_gdml(gdml)
    org = np.asarray(origin, float)
    env = fs.Envelope(geom, axis_xy=org[:2])
    support = fs.interior(gdml, geom, env, origin=org)
    field = ehelix.grid_field(*ehelix.read_g4bl(maps[0]))
    bnom = ehelix.field_at(field, org[None])[0]
    if np.linalg.norm(bnom) < B_DS_MIN:
        raise ValueError(f"first map {maps[0]} gives |B| = {np.linalg.norm(bnom):.3f} T at the tracker origin: "
                         "the DS map must come first")
    return fs.transport(np.random.default_rng(np.random.SeedSequence([seed, 1])), start, geom, field, support,
                        chord_mm=chord_mm, sagitta_mm=0.02, envelope=env, origin=org, bnom=bnom,
                        loss=fs.electron_loss, tail=False)


def write_injector(path, steps, entries):
    """Injector table (dump/sgsteps, dump/particles as TTrees): one primary electron per event, SimParticle key 1."""
    s = steps.copy()
    s["sp"] = 1
    p = entries[KEY].assign(key=1, pdg=PDG_E, parent=-1, isprim=1, ismcprim=1)
    tables.write(str(path), {"sgsteps": s[STEP_COLS], "particles": p}, as_ttree=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.cetracker", description=__doc__.split("\n")[0])
    ap.add_argument("tables", nargs="*", help="StrawGasStepTableDumper tables of the Geant4 events (starts); "
                                              "none with --starts")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gdml", required=True, help="tracker GDML (e.g. mu2e_run1a_v01.gdml)")
    ap.add_argument("--maps", nargs="+", required=True, help="G4BL .header files, first match wins; the DS map first (the tracker uses it alone)")
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_common.txt")
    ap.add_argument("--backend", choices=("numpy", "device"), default="numpy")
    ap.add_argument("--device-float", choices=("float64", "float32"), default="float64")
    ap.add_argument("--lanes", type=int, help="device chain: lanes per launch (default all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-events", type=int)
    ap.add_argument("--chord-mm", type=float, default=10.0)
    ap.add_argument("--origin", nargs=3, type=float, default=ORIGIN)
    ap.add_argument("--stops", help="stop table npz (x, y, z, t): fresh unfiltered CE births, one per event of the "
                                    "tables, which then only carry the event identities")
    ap.add_argument("--starts", help="generator file in the cechain reference format instead of tables: DIO "
                                     "(diospec starts) or RPC (rpcspec) electrons, one event per row")
    ap.add_argument("--shards", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    help="with --starts: keep these diospec shards (inclusive)")
    a = ap.parse_args(argv)
    if a.starts and (a.tables or a.stops):
        ap.error("--starts replaces the tables and --stops")
    if not a.starts and not a.tables:
        ap.error("give tables or --starts")
    if a.shards and not a.starts:
        ap.error("--shards needs --starts")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if a.starts:
        paths = [a.starts]
        with np.load(a.starts, allow_pickle=False) as src:
            g = births_from_starts(src, a.shards)
    else:
        paths = tables.expand(a.tables)
        g = starts(paths)
    if a.stops:
        g = births_from_stops(np.random.default_rng(np.random.SeedSequence([a.seed, 2])), np.load(a.stops), g)
    if a.max_events is not None:
        g = g.iloc[:a.max_events].reset_index(drop=True)
    cg = cechain.from_offline(a.geom, a.offline_root)
    tic = time.perf_counter()
    r = run_cechain(g, cg, a.maps, seed=a.seed, backend=a.backend, device_float=a.device_float, lanes=a.lanes)
    t_ce = time.perf_counter() - tic
    ent = handoff(g, r)
    np.savez_compressed(out / "cechain.npz", **{k: v for k, v in r._asdict().items() if v is not None},
                        **{k: g[k].to_numpy() for k in KEY})
    tic = time.perf_counter()
    steps, final = tracker_steps(ent, a.gdml, a.maps, seed=a.seed, origin=a.origin, chord_mm=a.chord_mm)
    t_trk = time.perf_counter() - tic
    write_injector(out / "fasttrk_field.root", steps, ent)
    fin = final.reset_index(drop=True)
    if not set(KEY) <= set(fin.columns):
        fin = pd.concat([ent[KEY].reset_index(drop=True), fin], axis=1)
    tables.write(str(out / "states.root"), {"gen": g, "entries": ent, "final": fin}, as_ttree=True)
    report = dict(inputs=[Path(p).name for p in paths], events=len(g), seed=a.seed, backend=a.backend,
                  births=("stops " + Path(a.stops).name if a.stops else
                          "starts " + Path(a.starts).name if a.starts else "tables"),
                  geom=a.geom, maps=[Path(m).name for m in a.maps], gdml=Path(a.gdml).name,
                  ce_fates={f: int((r.fate == k).sum()) for k, f in enumerate(cechain.FATES)},
                  entries=len(ent), steps=len(steps), events_with_steps=int(len(steps[KEY].drop_duplicates())),
                  ce_transport_s=round(t_ce, 3), tracker_transport_s=round(t_trk, 3))
    (out / "handoff.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
