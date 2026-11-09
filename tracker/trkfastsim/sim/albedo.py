"""Downstream albedo of tracker electrons: electrons that leave the tracker downstream and come back upstream.

The field-on tracker transport (fieldsteps, fieldsteps_jax) stops a track where it leaves the tracker envelope.
Geant4 follows it into the calorimeter and the material around it, from which a small fraction of the conversion
electrons (about 0.7 % of those leaving downstream), and some shower electrons, come back upstream through the
tracker as slow loopers (0.5 to 30 MeV/c) with many straw hits.

This module is an empirical library of that return, built from Geant4 (the dts dumper tables) at the virtual detector
at the tracker's downstream end (VD_BACK, Mu2e z = VD_Z):

- an exit is the first downstream crossing (pz > 0) of VD_BACK by the tracked electron (the conversion electron);
- its returns are the later upstream crossings (pz < 0) of VD_BACK by electrons descended from it: the electron
  itself (`isce`) and, with species "all", electrons of its downstream showers. Positrons and photons are not kept
  (the tracker transport is for electrons; Geant4 puts about 0.1 % of the tracker steps on photon daughters).

Each fast exit (a track that left the envelope downstream, moved back to the VD_BACK plane along its helix in the
nominal field) picks a random Geant4 exit of the same bin (exit radius, pz/p, sign of the radial momentum) uniformly,
with replacement, and takes that exit's returns: rotated about the tracker axis by the azimuth difference of the two
exits and delayed by the fast exit's time. The returns are then started on the VD_BACK plane through the same tracker
transport. Exits without returns are the large majority of every bin, so the return probability, the multiplicity and
the joint return spectrum (momentum, direction, radius, delay) all come from Geant4 per bin. A return that leaves
downstream again is not returned a second time (Geant4's later returns of the same exit are in its list).
"""
from dataclasses import dataclass

import numpy as np

AXIS_XY = (-3904.0, 0.0)        # tracker axis (Mu2e frame) [mm]
VD_BACK = 15                    # virtual detector at the tracker's downstream end (TT_Back)
VD_Z = 11810.5                  # its Mu2e z [mm]
R_EDGES = np.array([0.0, 335.0, 400.0, 500.0, 600.0, 719.0, 1e9])      # calorimeter disk 0: 335 to 719 mm
COS_EDGES = np.array([-1.0, 0.55, 0.65, 0.75, 1.01])
KB = 0.299792458                # MeV/c per (T mm)
PDG_E = 11
NBIN = (len(R_EDGES) - 1) * (len(COS_EDGES) - 1) * 2


def bin_of(r, cos, ur):
    """Bin index of an exit: radius, pz/p, sign of the radial momentum (numpy or jax arrays)."""
    xp = _xp(r)
    ir = xp.clip(xp.searchsorted(xp.asarray(R_EDGES), r, side="right") - 1, 0, len(R_EDGES) - 2)
    ic = xp.clip(xp.searchsorted(xp.asarray(COS_EDGES), cos, side="right") - 1, 0, len(COS_EDGES) - 2)
    return (ir * (len(COS_EDGES) - 1) + ic) * 2 + (ur > 0).astype(ir.dtype)


def _xp(a):
    if type(a).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp
        return jnp
    return np


@dataclass
class Library:
    """bin_exits[NBIN]: Geant4 exits per bin; groups (exits with returns): bin, start, count into the returns; returns
    in the exit's frame (exit azimuth 0, exit time 0): position azimuth dphi and radius r, momentum (pr, pphi, pz)
    radial/azimuthal/axial at the return point, delay dt, isce."""
    bin_exits: np.ndarray
    g_bin: np.ndarray
    g_start: np.ndarray
    g_count: np.ndarray
    dphi: np.ndarray
    r: np.ndarray
    pr: np.ndarray
    pphi: np.ndarray
    pz: np.ndarray
    dt: np.ndarray
    isce: np.ndarray
    species: str = "all"
    source: str = ""

    FIELDS = ("bin_exits", "g_bin", "g_start", "g_count", "dphi", "r", "pr", "pphi", "pz", "dt", "isce")

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.FIELDS}, species=np.array(self.species),
                            source=np.array(self.source), r_edges=R_EDGES, cos_edges=COS_EDGES)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if not (np.array_equal(z["r_edges"], R_EDGES) and np.array_equal(z["cos_edges"], COS_EDGES)):
                raise ValueError(f"{path}: library built with other bins")
            return cls(**{k: z[k] for k in cls.FIELDS}, species=str(z["species"]), source=str(z["source"]))

    def select(self, species):
        """The library restricted to the conversion electron's own returns (species 'ce') or unchanged ('all')."""
        if species == "all":
            return self
        if species != "ce":
            raise ValueError(f"species must be 'all' or 'ce', not {species!r}")
        keep = self.isce.astype(bool)
        gid = np.repeat(np.arange(len(self.g_bin)), self.g_count)[keep]
        cnt = np.bincount(gid, minlength=len(self.g_bin))
        ok = cnt > 0
        start = np.concatenate([[0], np.cumsum(cnt[ok])[:-1]]).astype(np.int64)
        f = {k: getattr(self, k)[keep] for k in ("dphi", "r", "pr", "pphi", "pz", "dt", "isce")}
        return Library(self.bin_exits, self.g_bin[ok], start, cnt[ok].astype(np.int64), **f, species="ce",
                       source=self.source)

    def return_fraction(self):
        """Per bin: exits with at least one return / exits."""
        n = np.bincount(self.g_bin, minlength=NBIN)
        return np.where(self.bin_exits > 0, n / np.maximum(self.bin_exits, 1), 0.0)


# --------------------------------------------------------------------------- building (host, from dumper tables)

def _polar(x, y, px, py):
    dx, dy = x - AXIS_XY[0], y - AXIS_XY[1]
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    c, s = np.cos(phi), np.sin(phi)
    return r, phi, px * c + py * s, -px * s + py * c


def build(vdsteps, particles, primary=None):
    """Library from dumper-table frames (KEY columns run, subrun, event): vdsteps (vdid, sp, x, y, z, px, py, pz, t)
    and particles (key, parent, pdg, isprim, ismcprim). primary: per event the tracked electron's key (default: the
    pdg 11 particle flagged isprim or ismcprim, as cearms)."""
    import pandas as pd
    key = ["run", "subrun", "event"]
    pa = particles
    if primary is None:
        primary = pa[(pa.pdg == PDG_E) & ((pa.isprim == 1) | (pa.ismcprim == 1))][key + ["key"]]
    if primary.duplicated(key).any():
        raise ValueError("more than one primary electron in an event")
    prim = primary.rename(columns={"key": "ce"})
    v = vdsteps[vdsteps.vdid == VD_BACK].merge(prim, on=key)
    ex = v[(v.sp == v.ce) & (v.pz > 0)].sort_values("t", kind="stable").drop_duplicates(key)
    ex = ex[key + ["x", "y", "px", "py", "pz", "t"]].rename(columns={c: "e_" + c for c in ("x", "y", "px", "py",
                                                                                           "pz", "t")})
    # descendants of the primary: walk the parents once per (event, particle)
    par = dict(zip(zip(pa.run, pa.subrun, pa.event, pa.key), pa.parent))
    pdg = dict(zip(zip(pa.run, pa.subrun, pa.event, pa.key), pa.pdg))
    ce_of = dict(zip(zip(prim.run, prim.subrun, prim.event), prim.ce))
    up = v[(v.pz < 0)].merge(ex[key + ["e_t"]], on=key)
    up = up[up.t > up.e_t]
    keep, isce = [], []
    for r_, sr, e, sp in zip(up.run, up.subrun, up.event, up.sp):
        ev = (r_, sr, e)
        ce = ce_of[ev]
        ok = pdg.get(ev + (sp,)) == PDG_E
        cur, desc = sp, False
        for _ in range(500):
            if cur == ce:
                desc = True
                break
            cur = par.get(ev + (cur,), -1)
            if cur is None or cur < 0:
                break
        keep.append(ok and desc)
        isce.append(sp == ce)
    up = up[np.array(keep, bool)].assign(isce=np.array(isce, bool)[np.array(keep, bool)])
    up = up.merge(ex, on=key + ["e_t"]).sort_values(key + ["t"], kind="stable")
    er, ephi, eur, _ = _polar(ex.e_x.to_numpy(), ex.e_y.to_numpy(), ex.e_px.to_numpy(), ex.e_py.to_numpy())
    ep = np.sqrt(ex.e_px ** 2 + ex.e_py ** 2 + ex.e_pz ** 2).to_numpy()
    ebin = bin_of(er, ex.e_pz.to_numpy() / ep, eur)
    bin_exits = np.bincount(ebin, minlength=NBIN).astype(np.int64)
    ex = ex.assign(bin=ebin, phi=ephi).reset_index(drop=True)
    ex["gid"] = np.arange(len(ex))
    up = up.merge(ex[key + ["gid", "bin", "phi"]], on=key)
    r, phi, pr, pphi = _polar(up.x.to_numpy(), up.y.to_numpy(), up.px.to_numpy(), up.py.to_numpy())
    dphi = np.angle(np.exp(1j * (phi - up["phi"].to_numpy())))
    gid = up.gid.to_numpy()
    u, start, cnt = np.unique(gid, return_index=True, return_counts=True)
    gb = ex.set_index("gid").loc[u, "bin"].to_numpy()
    return Library(bin_exits, gb.astype(np.int64), start.astype(np.int64), cnt.astype(np.int64), dphi, r, pr, pphi,
                   up.pz.to_numpy(), (up.t - up.e_t).to_numpy(), up.isce.to_numpy(bool))


# --------------------------------------------------------------------------- sampling (device)

def back_to_plane(x, u, p, bz, z_plane=VD_Z, q=-1.0):
    """Helix (uniform field bz [T] along z, charge q) from x (Mu2e frame) with direction u and |p| [MeV/c] to the plane
    z = z_plane: position and direction there. jax arrays."""
    import jax.numpy as jnp
    s = (z_plane - x[:, 2]) / jnp.where(jnp.abs(u[:, 2]) > 1e-6, u[:, 2], 1e-6)
    w = q * KB * bz / p                                         # d(u_t)/ds = w (u_y, -u_x)
    a = -w * s
    c, sn = jnp.cos(a), jnp.sin(a)
    ux = c * u[:, 0] - sn * u[:, 1]
    uy = sn * u[:, 0] + c * u[:, 1]
    small = jnp.abs(w) < 1e-12
    wsafe = jnp.where(small, 1.0, w)
    # x_t(s) = x_t0 + (1/w) J (R(a) - I) u_t0, J = [[0, -1], [1, 0]]
    dxr, dyr = ux - u[:, 0], uy - u[:, 1]
    dx = jnp.where(small, u[:, 0] * s, -dyr / wsafe)
    dy = jnp.where(small, u[:, 1] * s, dxr / wsafe)
    xn = jnp.stack([x[:, 0] + dx, x[:, 1] + dy, jnp.zeros_like(x[:, 2]) + z_plane], 1)
    return xn, jnp.stack([ux, uy, u[:, 2]], 1)


class DeviceLibrary:
    """The library on the device, with a sampler."""

    def __init__(self, lib, dtype=np.float64):
        import jax.numpy as jnp
        self.lib = lib
        dt = np.dtype(dtype)
        # per bin: its groups occupy [b_g0, b_g0 + b_ng) of the groups sorted by bin
        order = np.argsort(lib.g_bin, kind="stable")
        self.g_start = jnp.asarray(lib.g_start[order]); self.g_count = jnp.asarray(lib.g_count[order])
        gb = lib.g_bin[order]
        self.b_g0 = jnp.asarray(np.searchsorted(gb, np.arange(NBIN)).astype(np.int64))
        self.b_ng = jnp.asarray(np.bincount(gb, minlength=NBIN).astype(np.int64))
        self.b_ex = jnp.asarray(lib.bin_exits.astype(np.int64))
        self.ret = {k: jnp.asarray(np.asarray(getattr(lib, k), dt)) for k in ("dphi", "r", "pr", "pphi", "pz", "dt")}
        self.isce = jnp.asarray(lib.isce.astype(bool))
        self.dt = dt

    def pick(self, key, x, u, p, t, exit_mask, bz):
        """Per track (jax arrays; x Mu2e frame): the picked group (-1: none), the exit azimuth and the time at the
        plane."""
        import jax
        import jax.numpy as jnp
        xe, ue = back_to_plane(x, u, p, bz)
        dx, dy = xe[:, 0] - AXIS_XY[0], xe[:, 1] - AXIS_XY[1]
        r = jnp.hypot(dx, dy)
        phi = jnp.arctan2(dy, dx)
        ur = (ue[:, 0] * dx + ue[:, 1] * dy) / jnp.maximum(r, 1e-9)
        b = bin_of(r, ue[:, 2], ur)
        nb = self.b_ex[b]
        j = jnp.floor(jax.random.uniform(key, p.shape, dtype=jnp.float64) * nb).astype(jnp.int64)
        hit = exit_mask & (nb > 0) & (j < self.b_ng[b])           # the picked Geant4 exit has returns
        g = jnp.where(hit, self.b_g0[b] + j, -1)
        s_back = (VD_Z - x[:, 2]) / jnp.where(jnp.abs(u[:, 2]) > 1e-6, u[:, 2], 1e-6)
        te = t + s_back / (299.792458 * p / jnp.sqrt(p * p + 0.51099895 ** 2))
        return g, phi, te

    def returns(self, g, phi, te, track_index, size):
        """Expand the picked groups into return starts at the plane, padded to `size` (>= their count): x [size, 3]
        (Mu2e), momentum [size, 3], t, the track each belongs to, isce, valid."""
        import jax.numpy as jnp
        has = g >= 0
        cnt = jnp.where(has, self.g_count[jnp.maximum(g, 0)], 0)
        owner = jnp.repeat(jnp.arange(g.shape[0]), cnt, total_repeat_length=size)
        first = jnp.cumsum(cnt) - cnt
        k = jnp.arange(size) - first[owner]
        valid = jnp.arange(size) < jnp.sum(cnt)
        ri = jnp.where(valid, self.g_start[jnp.maximum(g[owner], 0)] + k, 0)
        R = self.ret
        ph = phi[owner] + R["dphi"][ri]
        c, s = jnp.cos(ph), jnp.sin(ph)
        xr = jnp.stack([AXIS_XY[0] + R["r"][ri] * c, AXIS_XY[1] + R["r"][ri] * s, jnp.full(size, VD_Z, self.dt)], 1)
        pr = jnp.stack([R["pr"][ri] * c - R["pphi"][ri] * s, R["pr"][ri] * s + R["pphi"][ri] * c, R["pz"][ri]], 1)
        return dict(x=xr, p=pr.astype(self.dt), t=(te[owner] + R["dt"][ri]).astype(self.dt),
                    track=track_index[owner], isce=self.isce[ri] & valid, valid=valid)


def merge_records(steps, extra, track, extra_valid, r_cap):
    """Append the records of extra tracks (rows of `extra`, each belonging to row track[i] of `steps`) after the valid
    records of their row, in extra-row order; records beyond r_cap are dropped and counted. jax arrays.
    Returns (merged steps, dropped record count)."""
    import jax
    import jax.numpy as jnp
    n, R0 = steps["valid"].shape
    m, R = extra["valid"].shape
    ev = extra["valid"] & extra_valid[:, None]
    rank = jnp.cumsum(ev, axis=1) - 1                                   # rank of each valid record in its extra row
    per = jnp.sum(ev, axis=1)
    # records of earlier extra rows of the same target row: rows sorted by target (stable) and an exclusive scan
    order = jnp.argsort(jnp.where(extra_valid, track, n), stable=True)
    tr_s, per_s = track[order], per[order]
    csum = jnp.cumsum(per_s) - per_s
    seg0 = jnp.concatenate([jnp.ones(1, bool), tr_s[1:] != tr_s[:-1]])
    seg_start = jax.lax.cummax(jnp.where(seg0, jnp.arange(m), 0))
    before_s = csum - csum[seg_start]
    before = jnp.zeros(m, before_s.dtype).at[order].set(before_s)
    used = jnp.max(jnp.where(steps["valid"], jnp.arange(R0) + 1, 0), axis=1)      # after the row's last record
    base = used[jnp.clip(track, 0, n - 1)] + before
    slot = base[:, None] + rank
    row = jnp.where(ev & (slot < r_cap), track[:, None], n)             # n: dropped
    col = jnp.clip(slot, 0, r_cap - 1)
    out = {k: v.at[row, col].set(extra[k].astype(v.dtype), mode="drop") for k, v in steps.items()}
    return out, jnp.sum(ev & (slot >= r_cap))
