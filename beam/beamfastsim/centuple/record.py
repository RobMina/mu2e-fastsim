"""Stage 1 (GPU): conversion-electron births -> one fast record per event (npz) and the device digis.

Per batch of births (run, subrun, event, position, momentum, time; Mu2e frame):

* ``cetracker.DeviceChain``: cechain_jax to the tracker entry plane, fieldsteps_jax StrawGasSteps (albedo default),
  device digis (on-spill conditions), and the reco response layer (trkfastsim.recoresp: fit, quality and the drawn
  p_reco, tan(dip), d0, r_max, fit consistency, chi2/ndf, active hits, t0 at the front);
* truth at the tracker surfaces: TT_Front is the entry state; TT_Mid and TT_Back come from the electron's own
  forward straw steps (the last one before the plane, carried to the plane on a helix in the uniform tracker field;
  TT_Back from the tracker exit state, carried back);
* calorimeter: the first VD crossing after the tracker exit in vacuum (``edepana.first_contact``). Front-face
  contacts go through the calorimeter device chain (calofastsim.device_chain) (shower library, crystal response, digis, template fit, clusters);
  inner-edge contacts (optional ``edge_library``, the Geant4 inner-edge table of ``edepana``) give one cluster at the contact
  with the table's visible energy. Contacts whose helix maximum radius stays below 400 mm deposit nothing (as ``edepana``).

Everything else in the EventNtuple (fit bookkeeping, MVA outputs, extrapolated surfaces) comes from the donor library
in the write stage. Detector frame for every stored position: Mu2e frame minus ORIGIN.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ORIGIN = np.array([-3904.0, 0.0, 10171.0])   # tracker (detector) origin in the Mu2e frame, Run 1A [mm]
KB = 0.299792458                           # MeV/c per (T mm)
ME = 0.51099895
C_MM_NS = 299.792458
# KinKalGeom tracker surfaces, detector frame z [mm] (Offline KinKalGeomMaker: TT_Front / TT_Mid / TT_Back disks)
Z_FRONT, Z_MID, Z_BACK = -1631.1, 10.1, 1639.1   # stock trksegs positions (AnalysisMDC2025 v02_02_02)
BZ_TRACKER = 1.0                           # [T] uniform-field approximation inside the tracker (helix to the planes)
MAX_CLUSTERS = 8
RECORD = "centuple_record.npz"
DIGIS = "centuple_digis.npz"
CALO_FILES = ("Sim_crystal.txt", "CsI-waveform-2020-02-12.root", "ReadoutNoise.root", "CsI-propag-2020-07-29.root",
              "calo_geometry_run1430.npz")
# stock OnSpill MC proton bunch time (ProtonBunchTimeMC, MDC2025ax digitization): -212.296 ns + 5 ns x {-2..2}, and
# the reco estimate of it (evtinfo.pbtime) within ~0.5 ns; the digitization window and the TDC zero follow it
PBTIME_BASE, PBTIME_STEP, PBTIME_K, PBT_RECO_SIGMA = -212.296, 5.0, (-2, 2), 0.5
CALO_CAPS = "steps=56,photoelectrons=5000,rows=26,digis=16,digi_length=96,hits=8,event_hits=32,rounds=8"


def helix_to_z(x, p, t, z, bz=BZ_TRACKER, q=-1.0):
    """Carry states (x [n, 3] mm, p [n, 3] MeV/c, t [n] ns; detector frame) along a helix in a uniform field bz along
    z to the plane z (forward or backward). Returns (x, p, t) at the plane; NaN rows where pz == 0."""
    x, p, t = np.asarray(x, float), np.asarray(p, float), np.asarray(t, float)
    pm = np.linalg.norm(p, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = z - x[:, 2]
        s = dz * pm / p[:, 2]                       # path length (signed)
        omega = q * bz * KB / pm                    # turning rate per unit path [1/mm], sign by charge
        phi = omega * s
        c, sn = np.cos(phi), np.sin(phi)
        px = p[:, 0] * c + p[:, 1] * sn
        py = -p[:, 0] * sn + p[:, 1] * c
        # integral of the transverse direction along s
        xo = x[:, 0] + (p[:, 0] * sn + p[:, 1] * (1 - c)) / (omega * pm)
        yo = x[:, 1] + (p[:, 1] * sn - p[:, 0] * (1 - c)) / (omega * pm)
        beta = pm / np.sqrt(pm * pm + ME * ME)
        to = t + s / (beta * C_MM_NS)
    xo = np.stack([xo, yo, np.full_like(xo, z)], axis=1)
    po = np.stack([px, py, p[:, 2]], axis=1)
    bad = ~np.isfinite(s)
    xo[bad], po[bad], to = np.nan, np.nan, np.where(bad, np.nan, to)
    return xo, po, to


def plane_truth(steps, z, forward=True):
    """The electron's state at the plane z (detector frame) from its straw steps [n, R]: the last forward step
    (pz > 0) ending before the plane, carried on a helix to it. Returns found [n], x, p, t."""
    valid = np.asarray(steps["valid"]) & (np.asarray(steps["pz"]) > 0) & (np.asarray(steps["z1"]) < z)
    zz = np.where(valid, np.asarray(steps["z1"]), -np.inf)
    j = np.argmax(zz, axis=1)
    found = valid.any(axis=1)
    r = np.arange(j.size)
    x = np.stack([np.asarray(steps[k])[r, j] for k in ("x1", "y1", "z1")], axis=1)
    p = np.stack([np.asarray(steps[k])[r, j] for k in ("px", "py", "pz")], axis=1)
    t = np.asarray(steps["t"])[r, j]
    xo, po, to = helix_to_z(x, p, t, z)
    xo[~found], po[~found] = np.nan, np.nan
    return found, xo, po, np.where(found, to, np.nan)


def births_frame(stops, n, seed, run_number=1, subrun=0):
    """n CeEndpoint births from a stop table as the record's input DataFrame (run, subrun, event = 1..n)."""
    import pandas as pd
    from ..cetracker import DeviceChain
    x, u, p, t = DeviceChain.births(stops, n, seed)
    pv = u * p[:, None]
    return pd.DataFrame(dict(run=np.full(n, run_number, np.int64), subrun=np.full(n, subrun, np.int64),
                             event=np.arange(1, n + 1, dtype=np.int64), x=x[:, 0], y=x[:, 1], z=x[:, 2],
                             px=pv[:, 0], py=pv[:, 1], pz=pv[:, 2], t=t))


class FastRecord:
    """The record stage, built once (DeviceChain, shower library, calorimeter response, inner-edge table)."""

    def __init__(self, offline_root, maps, gdml, *, library, calo_response, digi_cond, geom=None, dtype="float64",
                 response="onspill", edge_library=None, albedo="default", deltas=None, calo_batch=4096,
                 calo_caps=CALO_CAPS, pbtime_seed=20261107, pbtime="stock"):
        import jax
        jax.config.update("jax_enable_x64", True)
        from calofastsim import device_chain as dc
        from calofastsim import stockio
        from calofastsim.cluster import Geometry
        from calofastsim.showers import handoff_device as HD
        from calofastsim.showers.library import load
        from .. import cechain
        from ..cetracker import DeviceChain
        from ..edepana import EdgeEnergies
        self.dtype = np.dtype(dtype)
        kw = dict(response=response, digi_cond=digi_cond, dtype=dtype, albedo=albedo, deltas=deltas,
                  track_observables="fit")        # the fast track of class-F events too (fits failing the quality)
        if geom:
            kw["geom"] = geom
        self.chain = DeviceChain(offline_root, maps, gdml, **kw)
        lib, _ = load(Path(library))
        off = np.asarray(lib["offsets"])          # per library shower: summed Geant4 and Birks (visible) energy
        cs = np.r_[0.0, np.cumsum(lib["step_energy_g4"], dtype=np.float64)]
        self.lib_eg4 = cs[off[1:]] - cs[off[:-1]]
        cs = np.r_[0.0, np.cumsum(lib["step_energy_birks"], dtype=np.float64)]
        self.lib_evis = cs[off[1:]] - cs[off[:-1]]
        self.geo = HD.disk_geometry(lib, self.dtype)
        self.front_z = np.asarray(self.geo["front"], float)
        self.calo_origin = np.asarray(self.geo["origin"], float)
        ci = Path(calo_response)
        resp = dc.pack_response(stockio.load_sim_crystal(ci / CALO_FILES[0]), stockio.load_pulse(ci / CALO_FILES[1]),
                                stockio.load_noise(ci / CALO_FILES[2]), stockio.load_propagation(ci / CALO_FILES[3]),
                                Geometry.load(ci / CALO_FILES[4]), adc2mev=stockio.adc_to_mev, dtype=self.dtype)
        caps = {k: int(v) for k, v in (kv.split("=") for kv in calo_caps.split(","))}
        per = {k: v * calo_batch if k in ("steps", "photoelectrons", "rows", "digis", "hits") else v
               for k, v in caps.items()}
        self.calo_batch = calo_batch
        self.calo = dc.make_chain(dc.pack_library(lib, radius_neighbors=8, placement="crystal", dtype=self.dtype),
                                  resp, dc.Caps(events=calo_batch, **per), truth=True)
        self.field = cechain.field_maps(maps)
        self.edge = EdgeEnergies(edge_library) if edge_library is not None else None
        self.pbtime_seed = pbtime_seed
        if pbtime not in ("stock", "zero"):
            raise ValueError("pbtime must be 'stock' (the stock MC proton bunch time per event) or 'zero'")
        self.pbtime = pbtime
        self.config = dict(dtype=str(self.dtype), response=str(response), digi_cond=str(digi_cond), albedo=str(albedo),
                           deltas=str(deltas), library=str(library), edge_library=str(edge_library),
                           calo_caps=calo_caps, calo_batch=calo_batch, geom=geom, bz_tracker=BZ_TRACKER, pbtime=pbtime,
                           z_planes=[Z_FRONT, Z_MID, Z_BACK])

    # -- device part -------------------------------------------------------------------------------------------------
    def batch(self, births, seed):
        """One batch: births DataFrame -> (columns, digis, counters). Columns are host arrays, one row per birth."""
        import jax
        from trkfastsim.sim import fieldsteps_jax as FJ
        tic = time.perf_counter()
        n = len(births)
        rng = np.random.default_rng(np.random.SeedSequence([seed, self.pbtime_seed, 1]))
        if self.pbtime == "stock":
            pb_mc = PBTIME_BASE + PBTIME_STEP * rng.integers(PBTIME_K[0], PBTIME_K[1] + 1, n)
        else:
            pb_mc = np.zeros(n)
        pb_reco = pb_mc + PBT_RECO_SIGMA * rng.standard_normal(n) if self.pbtime == "stock" else pb_mc.copy()
        timing = None
        if self.chain.cond is not None and self.chain.cond.onspill:
            timing = dict(pbtime=pb_mc, event_length=np.full(n, float(self.chain.cond.eventWindowLength)))
        res = self.chain(births, seed, digis=self.chain.cond is not None, event_timing=timing)
        trk = res["track"]
        fin = res["tracker"].final
        steps = res["tracker"].steps
        h = jax.device_get(dict(has=res["has_entry"], x=res["x"], p=res["p"], t=res["t"], nrow=res["nrow"],
                                fin={k: fin[k] for k in ("x", "y", "z", "px", "py", "pz", "t", "status")},
                                steps={k: steps[k] for k in ("valid", "x1", "y1", "z1", "px", "py", "pz", "t")},
                                trk=trk))
        t_dev = time.perf_counter() - tic
        cols = {k: births[k].to_numpy() for k in ("run", "subrun", "event")}
        cols["pbtime_mc"], cols["pbtime_reco"] = pb_mc, pb_reco
        b = births
        cols["birth_x"] = np.stack([b.x, b.y, b.z], 1) - ORIGIN
        cols["birth_p"] = np.stack([b.px, b.py, b.pz], 1).astype(float)
        cols["birth_t"] = b.t.to_numpy(float)
        has = np.asarray(h["has"], bool)
        cols["has_entry"] = has
        cols["front_x"] = np.where(has[:, None], np.asarray(h["x"], float) - ORIGIN, np.nan)
        cols["front_p"] = np.where(has[:, None], np.asarray(h["p"], float), np.nan)
        cols["front_t"] = np.where(has, np.asarray(h["t"], float), np.nan)
        cols["nrow"] = np.asarray(h["nrow"], np.int64)
        for k, v in h["trk"].items():
            cols["trk_" + k] = np.asarray(v, bool if v.dtype == bool else float)
        found, x, p, t = plane_truth(h["steps"], Z_MID)
        cols.update(mid_found=found & has, mid_x=x, mid_p=p, mid_t=t)
        f = h["fin"]
        left = has & (np.asarray(f["status"]) == FJ.LEFT)
        xe = np.stack([f["x"], f["y"], f["z"]], 1).astype(float) - ORIGIN
        pe = np.stack([f["px"], f["py"], f["pz"]], 1).astype(float)
        ok = left & (xe[:, 2] > Z_BACK) & (pe[:, 2] > 0)
        xb, pb, tb = helix_to_z(xe, pe, np.asarray(f["t"], float), Z_BACK)
        cols.update(back_found=ok, back_x=np.where(ok[:, None], xb, np.nan), back_p=np.where(ok[:, None], pb, np.nan),
                    back_t=np.where(ok, tb, np.nan))
        for name in ("front", "mid", "back"):      # local field along z at the truth crossings (LoopHelix radius)
            xm = cols[name + "_x"] + ORIGIN
            ok_ = np.all(np.isfinite(xm), axis=1)
            bz = np.full(n, np.nan)
            if ok_.any():
                from mu2efastsim import ehelix
                bz[ok_] = ehelix.field_at(self.field, xm[ok_])[:, 2]
            cols[name + "_bz"] = bz
        cols.update(self._calo(left, xe + ORIGIN, pe, np.asarray(f["t"], float), seed, n, pb_mc))
        digis = res.get("digis")
        cols["nsd"] = (np.bincount(np.asarray(digis["ev"]), minlength=n)[:n].astype(np.int64) if digis is not None
                       else np.full(n, -1, np.int64))
        cnt = dict(births=n, entry=int(has.sum()), fit=int(cols["trk_fit"].sum()), qual=int(cols["trk_qual"].sum()),
                   mid=int(cols["mid_found"].sum()), back=int(ok.sum()), device_s=round(t_dev, 3),
                   calo_s=round(time.perf_counter() - tic - t_dev, 3),
                   digis=0 if digis is None else int(len(digis["ev"])))
        return cols, digis, cnt

    # -- calorimeter -------------------------------------------------------------------------------------------------
    def _calo(self, left, xe, pe, te, seed, n, pb_mc):
        """First calorimeter contact after the tracker exit, clusters per event (sorted by energy)."""
        import jax
        from calofastsim import device_chain as dc
        from ..edepana import EDGE, FRONT, RMAX_MIN, first_contact
        out = dict(calo_kind=np.zeros(n, np.int8), calo_disk=np.full(n, -1, np.int64), calo_etrue=np.zeros(n),
                   calo_etrue_g4=np.zeros(n),
                   calo_contact_x=np.full((n, 3), np.nan), calo_contact_t=np.full(n, np.nan))
        cl = {k: np.zeros((n, MAX_CLUSTERS), dt) for k, dt in
              (("valid", bool), ("energy", float), ("time", float), ("x", float), ("y", float), ("z", float),
               ("size", np.int64), ("disk", np.int64), ("split", bool), ("edge", bool), ("etrue", float),
               ("ttrue", float))}
        k = np.flatnonzero(left)
        pm = np.linalg.norm(pe[k], axis=1)
        T = np.sqrt(pm * pm + ME * ME) - ME
        c = first_contact(self.field, xe[k], pe[k] / pm[:, None], T, self.front_z, float(self.calo_origin[0, 0]))
        low = (c["kind"] > 0) & (c["rmax"] < RMAX_MIN)
        kind = np.where(low, 0, c["kind"]).astype(np.int8)
        cosz = np.abs(c["u"][:, 2])
        tcon = te[k] + np.abs(c["x"][:, 2] - xe[k][:, 2]) / np.where(cosz > 0, cosz, np.nan) / (
            pm / np.sqrt(pm * pm + ME * ME) * C_MM_NS)
        out["calo_kind"][k], out["calo_disk"][k] = kind, np.where(kind > 0, c["disk"], -1)
        out["calo_contact_x"][k] = np.where(kind[:, None] > 0, c["x"] - ORIGIN, np.nan)
        out["calo_contact_t"][k] = np.where(kind > 0, tcon, np.nan)
        rng = np.random.default_rng(np.random.SeedSequence([seed, self.pbtime_seed]))
        fr = np.flatnonzero(kind == FRONT)
        rows = k[fr]
        B = self.calo_batch
        key = jax.random.PRNGKey(seed)
        for i, b0 in enumerate(range(0, rows.size, B)):
            sel = fr[b0:b0 + B]
            m = sel.size
            disk = c["disk"][sel]
            pos = c["x"][sel]
            u = c["u"][sel]
            rad = np.hypot(pos[:, 0] - self.calo_origin[disk, 0], pos[:, 1] - self.calo_origin[disk, 1])
            ent = dict(energy=T[sel] + ME, cosine=u[:, 2], radius=rad, disk=disk, position=pos, direction=u,
                       time=tcon[sel])
            timing = {"pbtime": pb_mc[rows[b0:b0 + B]].astype(np.float32),
                      "mbtime": np.full(m, 1695.0, np.float32), "event_length": np.full(m, 1700.0),
                      "onspill": np.ones(m, bool), "marker": np.full(m, 200.0, np.float32),
                      "crystal_length": np.full(m, 200.0, np.float32)}
            q, t = dc.entries_batch(ent, timing, np.arange(m), B)
            r = dc.run(self.calo, q, t, jax.random.fold_in(key, i))
            ev = rows[b0:b0 + B]
            dn = np.asarray(r["donor"])[:m]
            okd = dn >= 0
            out["calo_etrue"][ev[okd]] = self.lib_evis[dn[okd]]
            out["calo_etrue_g4"][ev[okd]] = self.lib_eg4[dn[okd]]
            cc = r["clusters"]
            for name in ("valid", "energy", "time", "x", "y", "z", "size", "disk", "split", "etrue", "ttrue"):
                cl[name][ev] = np.asarray(cc[name])[:m, :MAX_CLUSTERS]
        if self.edge is not None:          # inner-edge contacts: one cluster with the table's visible energy
            ed = np.flatnonzero(kind == EDGE)
            if ed.size:
                g4, vis = self.edge.draw(rng, c["disk"][ed], c["dz"][ed], c["rmax"][ed])
                ev = k[ed]
                keep = vis > 0
                out["calo_etrue"][ev], out["calo_etrue_g4"][ev] = vis, g4
                ev, ed2 = ev[keep], ed[keep]
                cl["valid"][ev, 0] = True
                cl["edge"][ev, 0] = True
                cl["energy"][ev, 0] = vis[keep]
                cl["etrue"][ev, 0] = vis[keep]
                cl["time"][ev, 0] = tcon[ed2]
                cl["ttrue"][ev, 0] = tcon[ed2]
                d = c["disk"][ed2]
                cl["x"][ev, 0] = c["x"][ed2, 0] - self.calo_origin[d, 0]
                cl["y"][ev, 0] = c["x"][ed2, 1] - self.calo_origin[d, 1]
                cl["z"][ev, 0] = c["x"][ed2, 2] - self.front_z[d]
                cl["size"][ev, 0] = 2
                cl["disk"][ev, 0] = d
        out.update({"cl_" + name: v for name, v in cl.items()})
        return out


def _meta(**kw):
    return json.dumps(kw, default=str)


def run(births, out, seed, *, batch=20000, **kw):
    """Births (DataFrame) -> out/centuple_record.npz (+ out/centuple_digis.npz when digitising). One batch = up to
    `batch` births, seed per batch = seed * 1000 + batch index. Returns (paths, counters)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fr = FastRecord(**kw)
    parts, digis, counters = [], [], []
    tic = time.perf_counter()
    n0 = 0
    for i, b0 in enumerate(range(0, len(births), batch)):
        b = births.iloc[b0:b0 + batch].reset_index(drop=True)
        cols, dg, cnt = fr.batch(b, seed * 1000 + i)
        parts.append(cols)
        if dg is not None:
            d = {k: np.asarray(v) for k, v in dg.items()}
            d["ev"] = d["ev"] + n0
            digis.append(d)
        n0 += len(b)
        cnt["batch"] = i
        counters.append(cnt)
        print(json.dumps(cnt), flush=True)
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    wall = time.perf_counter() - tic
    meta = dict(stage="record", seed=int(seed), batch=int(batch), events=int(n0), wall_s=round(wall, 2),
                config=fr.config, counters=counters)
    np.savez_compressed(out / RECORD, meta=_meta(**meta), **cols)
    paths = dict(record=out / RECORD)
    if digis:
        d = {k: np.concatenate([x[k] for x in digis]) for k in digis[0] if np.ndim(digis[0][k]) > 0}
        np.savez_compressed(out / DIGIS, run=cols["run"], subrun=cols["subrun"], event=cols["event"],
                            pbtime_mc=cols["pbtime_mc"], pbtime_reco=cols["pbtime_reco"], **d)
        paths["digis"] = out / DIGIS
    return paths, dict(events=int(n0), wall_s=round(wall, 2), events_per_s=round(n0 / wall, 2) if wall else 0.0)
