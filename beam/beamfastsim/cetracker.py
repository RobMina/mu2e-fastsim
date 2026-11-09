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
per row (`--shards FIRST LAST` keeps a range of diospec shards). `--tracker device` (opt-in) runs the tracker transport
on the device as well (trkfastsim.sim.fieldsteps_jax, from the device cechain states, no host round trip in between).
Writes DIR/fasttrk_field.root (injector table), DIR/states.root (starts,
hand-off states, final tracker states), DIR/cechain.npz and DIR/handoff.json (counts and timings).

DeviceChain / run (opt-in, Python API): stops or births -> tracker entry -> field-on tracker steps -> digis -> track
observables, every stage on one device. The track observables come from the reco response layer
(trkfastsim.recoresp: efficiency and residual tables measured from stock LoopHelix reconstruction), applied per
event to the device entry state and the number of step rows. Digitisation (optional) runs the device digitiser
(trkfastsim.digi.digi_jax), whose host packing reads the steps once.
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


def tracker_steps(start, gdml, maps, seed=1, origin=ORIGIN, chord_mm=10.0, complete_outer_wall=False):
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
                        loss=fs.electron_loss, tail=False, complete_outer_wall=complete_outer_wall)


def write_injector(path, steps, entries):
    """Injector table (dump/sgsteps, dump/particles as TTrees): one primary electron per event, SimParticle key 1."""
    s = steps.copy()
    s["sp"] = 1
    p = entries[KEY].assign(key=1, pdg=PDG_E, parent=-1, isprim=1, ismcprim=1)
    tables.write(str(path), {"sgsteps": s[STEP_COLS], "particles": p}, as_ttree=True)


DIGI_COLS = ("shape", "ion", "eion", "len", "width", "t", "x0", "y0", "z0", "x1", "y1", "z1")


def _timing(event_timing, e0, ne):
    return {} if event_timing is None else {"event_timing": {k: np.asarray(v)[e0:e0 + ne]
                                                             for k, v in event_timing.items()}}


def _concat(outs, cond):
    from trkfastsim.digi import digi_jax
    if not outs:
        return digi_jax._empty(cond)
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


_RESIDENT = []


def _resident_steps():
    """Jitted (flat, chunk) for DeviceChain._digitize_resident. flat(s, size): the valid steps of the [event, slot]
    arrays in (event, slot) order as flat columns (DIGI_COLS, sid, ev) of length size, zero beyond them. chunk(st, a,
    n, e0, size): rows a..a+n of those with ev - e0, zero beyond n (digi_resident.pad_steps' layout)."""
    if not _RESIDENT:
        from functools import partial
        import jax
        import jax.numpy as jnp

        @partial(jax.jit, static_argnames=("size",))
        def flat(s, size):
            valid = s["valid"]
            keep = jnp.arange(size) < jnp.sum(valid)
            tr, sl = jnp.nonzero(valid, size=size, fill_value=0)
            out = {k: s[k][tr, sl] for k in DIGI_COLS}
            out["sid"] = ((s["plane"][tr, sl].astype(jnp.int64) << 10)
                          | (s["panel"][tr, sl].astype(jnp.int64) << 7) | s["straw"][tr, sl])
            out["ev"] = tr.astype(jnp.int64)
            return {k: jnp.where(keep, v, jnp.zeros_like(v)) for k, v in out.items()}

        @partial(jax.jit, static_argnames=("size",))
        def chunk(st, a, n, e0, size):
            i = jnp.arange(size)
            keep = i < n
            idx = jnp.where(keep, a + i, 0)
            out = {k: v[idx] for k, v in st.items()}
            out["ev"] = out["ev"] - e0
            return {k: jnp.where(keep, v, jnp.zeros_like(v)) for k, v in out.items()}

        _RESIDENT.append((flat, chunk))
    return _RESIDENT[0]


class DeviceChain:
    """Conversion electrons on one device: births (or stops) -> cechain_jax to the tracker entry -> fieldsteps_jax
    StrawGasSteps -> (optional) device digis -> reco response track observables. Built once (geometry, field maps,
    tracker and response tables); each call runs one batch. Per-event outputs stay device arrays.

    offline_root, maps (G4BL headers, DS map first), gdml: as the CLI. response: reco response tables (.npz,
    trkfastsim.recoresp.table), "default" (the tables shipped with trkfastsim.recoresp) or None (no track
    observables). digi_cond: device digitiser conditions (.npz,
    trkfastsim.digi.digicond) or None (no digis). ce_unroll, ce_shrink: cechain_jax's unroll and bucket shrink
    factor (a large shrink compacts less often and compiles fewer bucket shapes). digi_chunk: events per digitiser
    call (default 8,000; the random stream depends on the chunking, the distributions do not). resident_digis: use trkfastsim.digi.digi_resident (the same digis as digi_jax for the same keys and chunks,
    with the packing on the device; both tracker device floats). albedo: the downstream albedo of the tracked electrons
    (trkfastsim.sim.albedo): "default" (the shipped Geant4 Run 1A library, on by default), a library .npz or
    albedo.Library, or None (off: the transport before the albedo); albedo_species "all" (the electron and its shower
    electrons) or "ce" (the electron's own returns); albedo_max_chords: chord budget of the return transport (None:
    no budget)."""

    def __init__(self, offline_root, maps, gdml, geom="Offline/Mu2eG4/geom/geom_2021_PhaseI_v03.txt",
                 response=None, digi_cond=None, dtype="float64", origin=ORIGIN, chord_mm=10.0, lanes=None,
                 ce_unroll=True, ce_shrink=2, digi_chunk=8000, resident_digis=True, complete_outer_wall=False,
                 albedo="default", albedo_species="all", albedo_max_chords=8192):
        import jax  # noqa: F401  (device backend)
        from mu2efastsim import ehelix
        from trkfastsim.sim import fieldsteps_jax as FJ
        from . import cechain_jax
        self.dt = np.dtype(dtype)
        self.lanes = lanes
        self.ce_unroll = ce_unroll
        self.ce_shrink = ce_shrink                 # cechain_jax compaction: bucket shrink factor (2 = its default)
        self.digi_chunk = digi_chunk
        self.resident_digis = resident_digis       # trkfastsim.digi.digi_resident: same digis, packing on the device
        self.cg = cechain.from_offline(geom, offline_root)
        self.maps = cechain_jax.device_maps(maps, self.dt)
        org = np.asarray(origin, float)
        field = ehelix.grid_field(*ehelix.read_g4bl(maps[0]))
        if np.linalg.norm(ehelix.field_at(field, org[None])[0]) < B_DS_MIN:
            raise ValueError(f"first map {maps[0]}: the DS map must come first")
        self.trk = FJ.Tracker.from_gdml(gdml, maps[0], org, chord_mm=chord_mm, dtype=self.dt,
                                        complete_outer_wall=complete_outer_wall)
        self.resp = None
        if response is not None:
            from trkfastsim.recoresp import device as RD, table as RT
            self.resp = RD.to_device(RT.load() if response == "default" else RT.load(response), self.dt)
        self.cond = None
        if digi_cond is not None:
            from trkfastsim.digi import digicond
            self.cond = digicond.Conditions.load(digi_cond)
        self.alb = None                            # trkfastsim.sim.albedo: downstream returns (None: off)
        if albedo is not None:
            from trkfastsim.sim import albedo as AL
            lib = (albedo if isinstance(albedo, AL.Library)
                   else AL.Library.load() if isinstance(albedo, str) and albedo == "default" else AL.Library.load(albedo))
            self.alb = AL.DeviceLibrary(lib.select(albedo_species), self.dt)
            self.alb_max_chords = albedo_max_chords
            self.alb_bz = float(ehelix.field_at(field, np.array([[AL.AXIS_XY[0], AL.AXIS_XY[1], AL.VD_Z]]))[0][2])

    @staticmethod
    def births(stops, n, seed):
        """n CeEndpoint births from a stop table (dict or npz with x, y, z, t), host arrays (x, u, p, t)."""
        xs = np.stack([np.asarray(stops[c], float) for c in ("x", "y", "z")], axis=1)
        x, u, p, t, _ = cechain.from_stops(np.random.default_rng(np.random.SeedSequence([seed, 2])), xs,
                                           np.asarray(stops["t"], float), n)
        return x, u, p, t

    def transport(self, births, seed):
        """cechain + tracker transport of births ((x, u, p, t) host or device arrays, or a DataFrame with KEY +
        STATE columns). Returns a dict: cechain (Result, device), has_entry, entry x, p, t, tracker
        (fieldsteps_jax.Result), nrow (step rows per event), timing (enqueue wall seconds; the device may still run)."""
        import jax
        import jax.numpy as jnp
        from trkfastsim.sim import fieldsteps_jax as FJ
        from . import cechain_jax
        if isinstance(births, pd.DataFrame):
            x = births[["x", "y", "z"]].to_numpy(float)
            pv = births[["px", "py", "pz"]].to_numpy(float)
            p = np.linalg.norm(pv, axis=1)
            u = pv / p[:, None]
            t0 = births["t"].to_numpy(float)
        else:
            x, u, p, t0 = (np.asarray(v, float) for v in births)
        key = jax.random.PRNGKey(seed)
        tic = time.perf_counter()
        r = cechain_jax.transport(jax.random.fold_in(key, 0), self.cg, self.maps, x, u, p, t0, h=10.0,
                                  dtype=self.dt, lanes=self.lanes, unroll=self.ce_unroll, shrink=self.ce_shrink,
                                  device_out=True)
        t_ce = time.perf_counter() - tic
        has = r.fate == cechain.PLANE
        T = r.T
        pm = jnp.sqrt(T * (T + 2.0 * eioni.ME))
        un = r.u / jnp.linalg.norm(r.u, axis=1, keepdims=True)
        pe = un * pm[:, None]
        tic = time.perf_counter()
        trk = FJ.transport(self.trk, jax.random.fold_in(key, 1), r.x, pe, r.t, active=has)
        nrow = trk.steps["valid"].sum(axis=1)       # the conversion electron's rows (the response layer's input)
        out = {}
        if self.alb is not None:
            # the response tables count every row of the Geant4 conversion electron, its own albedo included
            trk, out["albedo"], ce_rows = self._albedo(trk, jax.random.fold_in(key, 4))
            nrow = nrow + ce_rows
        t_trk = time.perf_counter() - tic
        return dict(cechain=r, has_entry=has, x=r.x, p=pe, t=r.t, tracker=trk, nrow=nrow,
                    timing=dict(cechain_s=round(t_ce, 3), tracker_s=round(t_trk, 3)), **out)

    def _albedo(self, trk, key):
        """Downstream albedo (trkfastsim.sim.albedo): tracks that left the tracker downstream pick a Geant4 exit of
        their bin; its returns are transported from the tracker's back plane and their records appended to the
        track's rows (beyond R_CAP dropped and counted). Returns (tracker result, stats, the rows of the electron's
        own returns per track, before the R_CAP cut). One host read (the return count)."""
        import jax
        import jax.numpy as jnp
        from trkfastsim.sim import albedo as AL, fieldsteps_jax as FJ
        f = trk.final
        x = jnp.stack([f["x"], f["y"], f["z"]], 1).astype(jnp.float64)
        pv = jnp.stack([f["px"], f["py"], f["pz"]], 1).astype(jnp.float64)
        p = jnp.linalg.norm(pv, axis=1)
        u = pv / jnp.maximum(p, 1e-12)[:, None]
        ex = (f["status"] == FJ.LEFT) & (f["z"] > AL.VD_Z - 1.0) & (f["pz"] > 0)
        g, phi, te = self.alb.pick(jax.random.fold_in(key, 0), x, u, jnp.maximum(p, 1e-6),
                                   f["t"].astype(jnp.float64), ex, self.alb_bz)
        cnt = jnp.where(g >= 0, self.alb.g_count[jnp.maximum(g, 0)], 0)
        h = jax.device_get(dict(total=jnp.sum(cnt), exits=jnp.sum(ex), picked=jnp.sum(g >= 0)))
        stats = {k: int(v) for k, v in h.items()}
        if not stats["total"]:
            return trk, dict(stats, dropped=0), jnp.zeros_like(trk.steps["valid"].sum(axis=1))
        size = max(64, 1 << int(np.ceil(np.log2(stats["total"]))))
        rr = self.alb.returns(g, phi, te, jnp.arange(g.shape[0]), size)
        ret = FJ.transport(self.trk, jax.random.fold_in(key, 1), rr["x"], rr["p"], rr["t"], active=rr["valid"],
                           max_chords=self.alb_max_chords)
        stats["capped"] = int(jnp.sum((ret.final["status"] == FJ.CAPPED) & rr["valid"]))
        steps, dropped = AL.merge_records(trk.steps, ret.steps, rr["track"], rr["valid"], FJ.R_CAP)
        stats["dropped"] = int(dropped)
        rows = jnp.sum(ret.steps["valid"], axis=1) * (rr["valid"] & rr["isce"])
        ce_rows = jnp.zeros(g.shape[0], rows.dtype).at[rr["track"]].add(rows, mode="drop")   # integer: exact
        stats["return_rows"] = int(jnp.sum(ret.steps["valid"] & rr["valid"][:, None]))
        return trk._replace(steps=steps), stats, ce_rows.astype(trk.steps["valid"].sum(axis=1).dtype)

    def digitize(self, res, seed, event_timing=None, chunk=None):
        """Device digitiser on the chain's steps (trkfastsim.digi.digi_jax; its host packing reads the steps once).
        event_timing: pbtime and event_length per event (default 0 and the conditions' event window). Returns the
        digis (host arrays, `ev` = event index of the batch). chunk: events per digitiser call (fewer, larger calls
        are faster); a chunk beyond the digitiser's cluster capacity is split in two and retried. The random stream
        depends on the chunking (keys are folded with each chunk's first event)."""
        import jax
        from trkfastsim.digi import digi_jax
        chunk = self.digi_chunk if chunk is None else chunk
        s = res["tracker"].steps
        nev = s["valid"].shape[0]
        if event_timing is None and self.cond.onspill:
            event_timing = dict(pbtime=np.zeros(nev), event_length=np.full(nev, float(self.cond.eventWindowLength)))
        key = jax.random.fold_in(jax.random.PRNGKey(seed), 3)
        if self.resident_digis:
            return self._digitize_resident(s, nev, key, chunk, event_timing)
        valid = np.asarray(s["valid"])
        tr, sl = np.nonzero(valid)
        st = {k: np.asarray(s[k])[tr, sl] for k in DIGI_COLS}
        st["sid"] = ((np.asarray(s["plane"])[tr, sl].astype(np.int64) << 10)
                     | (np.asarray(s["panel"])[tr, sl].astype(np.int64) << 7) | np.asarray(s["straw"])[tr, sl])
        st["ev"] = tr.astype(np.int64)
        outs = []
        todo = [(e0, min(chunk, nev - e0)) for e0 in range(0, nev, chunk)]
        while todo:                                   # event chunks, as fastdigi.digitize_steps (cluster capacity)
            e0, ne = todo.pop(0)
            m = (st["ev"] >= e0) & (st["ev"] < e0 + ne)
            if not m.any():
                continue
            sub = {k: v[m] for k, v in st.items()}
            sub["ev"] = sub["ev"] - e0
            try:
                d = digi_jax.digitize(sub, ne, self.cond, jax.random.fold_in(key, e0), **_timing(event_timing, e0, ne))
            except OverflowError:                     # beyond the digitiser's cluster capacity: halve the chunk
                if ne == 1:
                    raise
                h = ne // 2
                todo[:0] = [(e0, h), (e0 + h, ne - h)]
                continue
            d = {k: np.asarray(v) for k, v in d.items()}
            d["ev"] = d["ev"] + e0
            rows = np.flatnonzero(m)
            d["step"] = rows[d["step"]] if len(d["step"]) else d["step"]   # row of the flat steps
            outs.append(d)
        return _concat(outs, self.cond)

    def _digitize_resident(self, s, nev, key, chunk, event_timing):
        """digitize with digi_resident: the flat steps and every event chunk are made on the device at power-of-two
        sizes (jitted; zero beyond the steps, as digi_resident pads them), and the digis come back at a power-of-two
        length and are cut on the host, so new data recompiles nothing unless a size crosses a power of two. Same
        steps, keys and chunks as the per-chunk slicing it replaces: the same digis bit for bit."""
        import jax
        import jax.numpy as jnp
        from trkfastsim.digi import digi_resident as DR
        flat, chunk_steps = _resident_steps()
        start = np.concatenate([[0], np.cumsum(np.asarray(jnp.sum(s["valid"], axis=1)))])   # event -> first row
        nv = int(start[-1])
        st = flat({k: s[k] for k in DIGI_COLS + ("plane", "panel", "straw", "valid")}, DR._b(nv))
        outs = []
        todo = [(e0, min(chunk, nev - e0)) for e0 in range(0, nev, chunk)]
        while todo:                                   # event chunks (steps are in event order: contiguous rows)
            e0, ne = todo.pop(0)
            a, b = int(start[e0]), int(start[e0 + ne])
            if b == a:
                continue
            sub = chunk_steps(st, jnp.int64(a), jnp.int64(b - a), jnp.int64(e0), DR._b(b - a))
            try:
                d, cnt = DR.digitize(sub, ne, self.cond, jax.random.fold_in(key, e0), nstep=b - a, padded=True,
                                     **_timing(event_timing, e0, ne))
            except OverflowError:                     # beyond the digitiser's cluster capacity: halve the chunk
                if ne == 1:
                    raise
                h = ne // 2
                todo[:0] = [(e0, h), (e0 + h, ne - h)]
                continue
            d = {k: np.asarray(v)[:cnt] for k, v in d.items()}
            d["ev"] = d["ev"] + e0
            rows = np.arange(a, b)
            d["step"] = rows[d["step"]] if len(d["step"]) else d["step"]   # row of the flat steps
            outs.append(d)
        return _concat(outs, self.cond)

    def observables(self, res, seed):
        """Reco response track observables per event (device arrays; trkfastsim.recoresp.device.apply)."""
        import jax
        from trkfastsim.recoresp import device as RD
        return RD.apply(self.resp, jax.random.fold_in(jax.random.PRNGKey(seed), 2), res["has_entry"], res["nrow"],
                        res["x"], res["p"], res["t"])

    def __call__(self, births, seed, digis=True, event_timing=None):
        res = self.transport(births, seed)
        if digis and self.cond is not None:
            res["digis"] = self.digitize(res, seed, event_timing)
        if self.resp is not None:
            res["track"] = self.observables(res, seed)
        return res


_CHAIN = {}


def run(stops_or_births, seed, *, n=None, digis=True, event_timing=None, **chain_kw):
    """One batch through DeviceChain (built once per process and configuration): stops_or_births is a stop table
    (dict/npz with x, y, z, t; then n births are drawn) or births (x, u, p, t). Returns DeviceChain's dict."""
    key = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in chain_kw.items()))
    if key not in _CHAIN:
        _CHAIN[key] = DeviceChain(**chain_kw)
    ch = _CHAIN[key]
    if hasattr(stops_or_births, "keys") or hasattr(stops_or_births, "files"):
        births = ch.births(stops_or_births, int(n), seed)
    else:
        births = stops_or_births
    return ch(births, seed, digis=digis, event_timing=event_timing)


def bench_prepare(n, seed=None):
    """mu2efastsim.bench --prepare: the stop table named by $CETRACKER_STOPS, in host memory. With seed (the bench's
    --fresh-inputs passes seed=S+i before every pass), the n births of that seed are drawn here, outside the clock,
    and bench_run uses the same seed for the chain."""
    import os
    with np.load(os.environ["CETRACKER_STOPS"]) as f:
        stops = {k: np.asarray(f[k]) for k in ("x", "y", "z", "t")}
    if seed is None:
        return stops, int(n)
    return DeviceChain.births(stops, int(n), int(seed)), int(n), int(seed)


def bench_run(prepared, seed=1, digis=True, **chain_kw):
    """mu2efastsim.bench entry point: n births drawn from the stops (or the births bench_prepare drew for its seed,
    which then replaces `seed`), then the whole device chain. Returns the device outputs the timer waits on."""
    if len(prepared) == 3:
        births, n, seed = prepared
        res = run(births, seed, digis=digis, **chain_kw)
    else:
        stops, n = prepared
        res = run(stops, seed, n=n, digis=digis, **chain_kw)
    return dict(track=res.get("track"), nrow=res["nrow"], steps_len=res["tracker"].steps["len"],
                digis=None if "digis" not in res else res["digis"]["adc"])


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
    ap.add_argument("--complete-outer-wall", action="store_true",
                    help="tracker transport: the complete outer-minus-gas straw wall, wall-only chords included "
                         "(opt-in; default the gas-gated wall)")
    ap.add_argument("--albedo", default="default",
                    help="device tracker: downstream albedo library ('default': the shipped one; 'none': off)")
    ap.add_argument("--tracker", choices=("numpy", "device"), default="numpy",
                    help="tracker transport: NumPy fieldsteps (default) or the device twin (needs --backend device)")
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
    if a.tracker == "device":
        if a.backend != "device":
            raise SystemExit("--tracker device needs --backend device")
        ch = DeviceChain(a.offline_root, a.maps, a.gdml, geom=a.geom, dtype=a.device_float, origin=a.origin,
                         chord_mm=a.chord_mm, lanes=a.lanes, complete_outer_wall=a.complete_outer_wall,
                         albedo=None if a.albedo == "none" else a.albedo)
        tic = time.perf_counter()
        res = ch.transport(g, a.seed)
        t_ce, t_trk = res["timing"]["cechain_s"], res["timing"]["tracker_s"]
        r = cechain.Result(**{k: (None if v is None else np.asarray(v)) for k, v in res["cechain"]._asdict().items()})
        ent = handoff(g, r)
        np.savez_compressed(out / "cechain.npz", **{k: v for k, v in r._asdict().items() if v is not None},
                            **{k: g[k].to_numpy() for k in KEY})
        from trkfastsim.sim import fieldsteps_jax as FJ
        keys = g[KEY].assign(sp=1)
        sel = np.asarray(res["has_entry"])
        steps = FJ.to_frame(res["tracker"], keys)                    # tracks without an entry have no rows
        final = FJ.final_frame(res["tracker"], keys)[sel].reset_index(drop=True)
    else:
        tic = time.perf_counter()
        r = run_cechain(g, cg, a.maps, seed=a.seed, backend=a.backend, device_float=a.device_float, lanes=a.lanes)
        t_ce = time.perf_counter() - tic
        ent = handoff(g, r)
        np.savez_compressed(out / "cechain.npz", **{k: v for k, v in r._asdict().items() if v is not None},
                            **{k: g[k].to_numpy() for k in KEY})
        tic = time.perf_counter()
        steps, final = tracker_steps(ent, a.gdml, a.maps, seed=a.seed, origin=a.origin, chord_mm=a.chord_mm,
                                     complete_outer_wall=a.complete_outer_wall)
        t_trk = time.perf_counter() - tic
    write_injector(out / "fasttrk_field.root", steps, ent)
    fin = final.reset_index(drop=True)
    if not set(KEY) <= set(fin.columns):
        fin = pd.concat([ent[KEY].reset_index(drop=True), fin], axis=1)
    tables.write(str(out / "states.root"), {"gen": g, "entries": ent, "final": fin}, as_ttree=True)
    report = dict(inputs=[Path(p).name for p in paths], events=len(g), seed=a.seed, backend=a.backend,
                  tracker=a.tracker,
                  births=("stops " + Path(a.stops).name if a.stops else
                          "starts " + Path(a.starts).name if a.starts else "tables"),
                  geom=a.geom, maps=[Path(m).name for m in a.maps], gdml=Path(a.gdml).name,
                  ce_fates={f: int((r.fate == k).sum()) for k, f in enumerate(cechain.FATES)},
                  entries=len(ent), steps=len(steps), events_with_steps=int(len(steps[KEY].drop_duplicates())),
                  ce_transport_s=round(t_ce, 3), tracker_transport_s=round(t_trk, 3))
    if a.complete_outer_wall:
        report["complete_outer_wall"] = True
    (out / "handoff.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
