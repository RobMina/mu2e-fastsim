"""Delta rays of tracker electrons as straw steps: an empirical Geant4 library attached to the device transport's gas
crossings.

The field-on transport samples the delta electrons an electron makes (fieldsteps.electron_loss: deltas above E_CUT
leave the track) but does not transport them, so their ionisation in the straws is missing. In Geant4 (CeEndpoint
dts, development files) the secondaries of the conversion electron are 16 % of the tracker StrawGasSteps and 35 % of
the gas eIon; 71 % of that eIon is in the same straw and time as a crossing of the parent and 14 % in other straws of
the same panel (the rest in other planes, not modelled here).

The library, built from Geant4 dumper tables:
- every gas crossing of the conversion electron, binned by its path length in the gas (LEN_EDGES);
- for each crossing, the secondary steps attributed to it: descendants of the electron (not born downstream of the
  tracker) in the same straw within DT_NS of the crossing, or in another straw of the same panel within DT_NS of the
  panel's nearest crossing. Per step: straw offset dk in the panel, eIon, length, width, shape, ionisation type, delay,
  |p|, and its midpoint and chord in the frame of its own straw (u along the wire, w = z, v = w x u): the along-wire
  offset from the crossing's along-wire point, the transverse offset from its wire, the chord components.

On the device, every gas record of a track picks a Geant4 crossing of its length bin uniformly, with replacement; most
crossings have no attributed steps (8.9 % of CE crossings have some in the Run 1A development library). The picked crossing's steps are placed in straw js + dk
of the record's panel (dropped outside it), at the record's along-wire point plus the offsets, and appended to the
track's rows (albedo.merge_records). The steps are electrons of the event; the response layer's nrow is not changed.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LEN_EDGES = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1e9])     # gas path of the parent crossing [mm]
DT_NS = 2.0
Z_DS = 11750.0                                                 # Mu2e z: born beyond it = downstream (albedo, not here)
NSTRAW = 96
PDG_E = 11
DEFAULT = Path(__file__).with_name("data") / "deltas_run1a_v1.npz"
STEP = ("dk", "eion", "len", "width", "shape", "ion", "dt", "pmag", "du", "dv", "dw", "su", "sv", "sw")


def frame(d):
    """Unit (u, v, w) of straws with axis d [n, 3]: u along the wire, w = z, v = w x u."""
    xp = _xp(d)
    w = xp.zeros_like(d) + xp.asarray([0.0, 0.0, 1.0], dtype=d.dtype)
    v = xp.cross(w, d)
    v = v / xp.linalg.norm(v, axis=1)[:, None]
    return d, v, w


def _xp(a):
    if type(a).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp
        return jnp
    return np


@dataclass
class Library:
    bin_cross: np.ndarray      # [nbin] Geant4 crossings per length bin
    g_bin: np.ndarray          # crossings with attributed steps: bin, start, count into the step arrays
    g_start: np.ndarray
    g_count: np.ndarray
    steps: dict                # STEP -> [M]
    source: str = ""

    def save(self, path):
        np.savez_compressed(path, bin_cross=self.bin_cross, g_bin=self.g_bin, g_start=self.g_start,
                            g_count=self.g_count, len_edges=LEN_EDGES, source=np.array(self.source),
                            **{"s_" + k: v for k, v in self.steps.items()})

    @classmethod
    def load(cls, path=DEFAULT):
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z["len_edges"], LEN_EDGES):
                raise ValueError(f"{path}: library built with other bins")
            return cls(z["bin_cross"], z["g_bin"], z["g_start"], z["g_count"], {k: z["s_" + k] for k in STEP},
                       str(z["source"]))


def build(sgsteps, particles, smid, sdir, primary=None):
    """Library from dumper-table frames (KEY columns run, subrun, event): sgsteps (plane, panel, straw, sp, eion, len,
    width, shape, ion, t, x0..z1, px, py, pz; detector frame) and particles (key, parent, pdg, isprim, ismcprim, sz in
    the Mu2e frame). smid, sdir [N, 3]: straw middles (detector frame) and axes by unique straw index
    (plane * 6 + panel) * 96 + straw. primary: per event the electron's key (default: pdg 11 flagged isprim or ismcprim)."""
    import pandas as pd
    key = ["run", "subrun", "event"]
    pa = particles
    if primary is None:
        primary = pa[(pa.pdg == PDG_E) & ((pa.isprim == 1) | (pa.ismcprim == 1))][key + ["key"]]
    if primary.duplicated(key).any():
        raise ValueError("more than one primary electron in an event")
    st = sgsteps.merge(primary.rename(columns={"key": "ce"}), on=key)
    st = st.assign(js=((st.plane.to_numpy(np.int64) * 6 + st.panel) * NSTRAW + st.straw).astype(np.int64),
                   up=(st.plane.to_numpy(np.int64) * 6 + st.panel).astype(np.int64))
    par = dict(zip(zip(pa.run, pa.subrun, pa.event, pa.key), zip(pa.parent, pa.sz)))
    keep = {}
    for r, s, e, sp, ce in st[st.sp != st.ce][key + ["sp", "ce"]].drop_duplicates().itertuples(index=False):
        cur, ok, down = int(sp), False, False
        for _ in range(500):
            if cur == ce:
                ok = True
                break
            pr = par.get((r, s, e, cur))
            if pr is None:
                break
            down |= bool(pr[1] > Z_DS)
            if pr[0] < 0:
                break
            cur = int(pr[0])
        keep[(r, s, e, int(sp))] = ok and not down
    cs = st[st.sp == st.ce].reset_index(drop=True)
    cs["cid"] = np.arange(len(cs))
    sec = st[st.sp != st.ce]
    sec = sec[[keep[(r, s, e, int(sp))] for r, s, e, sp in zip(sec.run, sec.subrun, sec.event, sec.sp)]].copy()
    # attribution: same straw within DT_NS, else the same panel's crossing nearest in time within DT_NS
    cand = sec.reset_index().merge(cs[key + ["up", "js", "t", "cid", "x0", "y0", "z0", "x1", "y1", "z1"]], on=key + ["up"],
                                   suffixes=("", "_c"))
    cand["adt"] = (cand.t - cand.t_c).abs()
    cand = cand[cand.adt < DT_NS]
    cand["same"] = cand.js == cand.js_c
    cand = cand.sort_values(["index", "same", "adt"], ascending=[True, False, True]).drop_duplicates("index")
    m0 = np.c_[cand.x0_c + cand.x1_c, cand.y0_c + cand.y1_c, cand.z0_c + cand.z1_c] * 0.5
    m = np.c_[cand.x0 + cand.x1, cand.y0 + cand.y1, cand.z0 + cand.z1] * 0.5
    seg = np.c_[cand.x1 - cand.x0, cand.y1 - cand.y0, cand.z1 - cand.z0]
    jc, jd = cand.js_c.to_numpy(), cand.js.to_numpy()
    uc = np.sum((m0 - smid[jc]) * sdir[jc], 1)                    # crossing's along-wire point
    u, v, w = frame(sdir[jd])
    rel = m - smid[jd]
    out = dict(dk=(jd - jc).astype(np.int32), eion=cand.eion.to_numpy(float), len=cand.len.to_numpy(float),
               width=cand.width.to_numpy(float), shape=cand["shape"].to_numpy(np.int32), ion=cand.ion.to_numpy(np.int32),
               dt=(cand.t - cand.t_c).to_numpy(float),
               pmag=np.sqrt(cand.px.to_numpy(float) ** 2 + cand.py ** 2 + cand.pz ** 2),
               du=np.sum(rel * u, 1) - uc, dv=np.sum(rel * v, 1), dw=np.sum(rel * w, 1),
               su=np.sum(seg * u, 1), sv=np.sum(seg * v, 1), sw=np.sum(seg * w, 1))
    cid = cand.cid.to_numpy()
    order = np.argsort(cid, kind="stable")
    out = {k: np.asarray(v)[order] for k, v in out.items()}
    cid = cid[order]
    cb = np.clip(np.searchsorted(LEN_EDGES, cs.len.to_numpy(float), side="right") - 1, 0, len(LEN_EDGES) - 2)
    g, start, cnt = np.unique(cid, return_index=True, return_counts=True)
    return Library(np.bincount(cb, minlength=len(LEN_EDGES) - 1).astype(np.int64), cb[g].astype(np.int64),
                   start.astype(np.int64), cnt.astype(np.int64), out)


class DeviceLibrary:
    def __init__(self, lib, dtype=np.float64):
        import jax.numpy as jnp
        self.lib = lib
        order = np.argsort(lib.g_bin, kind="stable")
        self.g_start, self.g_count = jnp.asarray(lib.g_start[order]), jnp.asarray(lib.g_count[order])
        gb = lib.g_bin[order]
        nb = len(LEN_EDGES) - 1
        self.b_g0 = jnp.asarray(np.searchsorted(gb, np.arange(nb)).astype(np.int64))
        self.b_ng = jnp.asarray(np.bincount(gb, minlength=nb).astype(np.int64))
        self.b_n = jnp.asarray(lib.bin_cross.astype(np.int64))
        dt = np.dtype(dtype)
        self.s = {k: jnp.asarray(np.asarray(v, np.int32 if k in ("dk", "shape", "ion") else dt)) for k, v in lib.steps.items()}
        self.dt = dt

    def pick(self, key, steps):
        """Per record of the [n, R] steps: the picked group (-1: none)."""
        import jax
        import jax.numpy as jnp
        valid = steps["valid"]
        b = jnp.clip(jnp.searchsorted(jnp.asarray(LEN_EDGES), steps["len"], side="right") - 1, 0, len(LEN_EDGES) - 2)
        nb = self.b_n[b]
        j = jnp.floor(jax.random.uniform(key, valid.shape, dtype=jnp.float64) * nb).astype(jnp.int64)
        hit = valid & (nb > 0) & (j < self.b_ng[b])
        return jnp.where(hit, self.b_g0[b] + j, -1)

    def make(self, g, steps, T, size):
        """The picked groups' steps as [size, 1] rows of the steps layout (detector frame), with the row each belongs
        to and validity (straw outside the panel: dropped). T: the tracker's device tables (smid, sdir, sid, plane,
        panel, straw, upanel)."""
        import jax.numpy as jnp
        n, R = g.shape
        gf = g.ravel()
        has = gf >= 0
        cnt = jnp.where(has, self.g_count[jnp.maximum(gf, 0)], 0)
        owner = jnp.repeat(jnp.arange(gf.shape[0]), cnt, total_repeat_length=size)
        k = jnp.arange(size) - (jnp.cumsum(cnt) - cnt)[owner]
        valid = jnp.arange(size) < jnp.sum(cnt)
        si = jnp.where(valid, self.g_start[jnp.maximum(gf[owner], 0)] + k, 0)
        S = {kk: v[si] for kk, v in self.s.items()}
        row, slot = owner // R, owner % R
        js = steps["straw_index"][row, slot].astype(jnp.int64)
        straw = js % NSTRAW
        ok = valid & (straw + S["dk"] >= 0) & (straw + S["dk"] < NSTRAW)
        jd = jnp.where(ok, js + S["dk"], 0)
        dt = self.dt
        x0 = jnp.stack([steps[c][row, slot] for c in ("x0", "y0", "z0")], 1)
        x1 = jnp.stack([steps[c][row, slot] for c in ("x1", "y1", "z1")], 1)
        smid, sdir = T["smid"].astype(dt), T["sdir"].astype(dt)
        uc = jnp.sum((0.5 * (x0 + x1) - smid[js]) * sdir[js], 1)
        u, v, w = frame(sdir[jd])
        mid = smid[jd] + (uc + S["du"])[:, None] * u + S["dv"][:, None] * v + S["dw"][:, None] * w
        seg = S["su"][:, None] * u + S["sv"][:, None] * v + S["sw"][:, None] * w
        a, b = mid - 0.5 * seg, mid + 0.5 * seg
        pdir = seg / jnp.maximum(jnp.linalg.norm(seg, axis=1), 1e-9)[:, None]
        p = S["pmag"][:, None] * pdir
        col = lambda x, typ=None: (x if typ is None else x.astype(typ))[:, None]
        out = dict(valid=col(ok), straw_index=col(jd, jnp.int32), sid=col(T["sid"][jd]), plane=col(T["plane"][jd]),
                   panel=col(T["panel"][jd]), straw=col(T["straw"][jd]), upanel=col(T["upanel"][jd]),
                   shape=col(S["shape"]), ion=col(S["ion"]), eion=col(S["eion"]), len=col(S["len"]),
                   width=col(S["width"]), t=col(steps["t"][row, slot] + S["dt"]),
                   x0=col(a[:, 0]), y0=col(a[:, 1]), z0=col(a[:, 2]), x1=col(b[:, 0]), y1=col(b[:, 1]), z1=col(b[:, 2]),
                   px=col(p[:, 0]), py=col(p[:, 1]), pz=col(p[:, 2]))
        return out, row, ok
