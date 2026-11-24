"""Calorimeter-seeded tracker trigger (cpr) on the device: port of the stock CalTimePeakFinder (Offline v13_41_01,
CalPatRec), with its arithmetic.

For each calo cluster (collection order) with energy >= MinClusterEnergy and size >= MinClusterSize, the time peak
takes the panel hits (collection order) that carry a HitSelBits flag and none of BkgSelBits, whose time residual
dt = (cluster time + DtOffset) - (hit time + (z_cluster - z_hit) / sin(PitchAngle) / (c * Beta)) lies in
[DtMin, DtMax), and whose polyAtan2 azimuth is within pi/2 of the cluster's. It becomes a TimeCluster when the hits'
straw-hit count reaches MinNHits.

The cluster position is the cog in the disk frame (cog z = 0) moved to Mu2e (disk origin + cog; the disks are not
rotated) and then to the tracker frame (minus the tracker centre: -solenoidOffset, 0, detectorSystemZ0), in that order. Types as stock: float
hit and cluster fields, double positions, times and residuals, the float polyAtan2 azimuths subtracted in double.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .exact import strict
from .hits import STRAW_FLAGS
from .mpr import _d, _f, poly_atan2

C_LIGHT = 299.792458                 # CLHEP::c_light [mm/ns]
DETECTOR_SYSTEM_Z0 = 10171.0         # mu2e.detectorSystemZ0 (geom_run1_a.txt) [mm]


def _bits(names):
    m = 0
    for n in names:
        if n not in STRAW_FLAGS:
            raise ValueError(f"CalTimePeakFinder: flag {n} not on the device")
        m |= 1 << STRAW_FLAGS[n]
    return m


def settings(cfg, geo):
    """Static settings of one CalTimePeakFinder module (menu table) and the calorimeter geometry (`calogeom.build`)."""
    if float(cfg.get("Beta", 1)) <= 0:
        raise ValueError("CalTimePeakFinder: Beta must be positive")
    centre = np.array([-3904.0, 0.0, DETECTOR_SYSTEM_Z0])
    if not np.isclose(-geo["origin"][0, 0], 3904.0):
        raise ValueError("CalTimePeakFinder: tracker centre assumes solenoidOffset 3904 mm")
    return {"hsel": _bits(cfg["HitSelBits"]), "bkg": _bits(cfg["BkgSelBits"]),
            "dtmin": float(cfg["DtMin"]), "dtmax": float(cfg["DtMax"]), "dtoff": float(cfg["DtOffset"]),
            "minnhits": int(cfg["MinNHits"]), "mine": float(cfg["MinClusterEnergy"]),
            "minsize": int(cfg["MinClusterSize"]), "sinpitch": math.sin(float(cfg["PitchAngle"])),
            "beta": float(cfg["Beta"]), "disk_origin": np.asarray(geo["origin"], np.float64), "centre": centre}


def time_peaks(ph, cl, s):
    """Panel hits [E, H] (valid, flag, time, x, y, z, ncombo: the straw-hit count) and calo clusters [E, C] (valid, disk, time, energy, size,
    cog_x, cog_y) -> time peaks [E, C]: `tc` (a TimeCluster), `nsh`, `t0`, `pos` [E, C, 3] (tracker frame) and the
    hit mask `hits` [E, C, H]. Peaks come in cluster order, as stock's TimeClusterCollection."""
    with strict(ph["flag"]):
        flag = jnp.asarray(ph["flag"]).astype(jnp.int64)
        sel = ph["valid"] & ((flag & s["hsel"]) != 0) & ((flag & s["bkg"]) == 0)
        t_hit = _d(_f(ph["time"]))[:, None, :]
        z_hit = _d(_f(ph["z"]))[:, None, :]
        phi_hit = _d(poly_atan2(ph["y"], ph["x"]))[:, None, :]
        org = jnp.asarray(s["disk_origin"])[jnp.asarray(cl["disk"]).astype(jnp.int32)]          # [E, C, 3]
        c = s["centre"]                          # diskToMu2e (origin + cog, cog z 0), then mu2eToTracker (- centre)
        xcl = (org[..., 0] + _d(cl["cog_x"])) - c[0]
        ycl = (org[..., 1] + _d(cl["cog_y"])) - c[1]
        zcl = (org[..., 2] + 0.0) - c[2]
        cl_time = _d(_f(cl["time"])) + s["dtoff"]
        mphi = _d(poly_atan2(_f(ycl), _f(xcl)))[..., None]
        tof = (zcl[..., None] - z_hit) / s["sinpitch"] / (C_LIGHT * s["beta"])
        dt = cl_time[..., None] - (t_hit + tof)
        dphi = phi_hit - mphi
        dphi = jnp.where(dphi > math.pi, dphi - 2 * math.pi, dphi)
        dphi = jnp.where(dphi < -math.pi, dphi + 2 * math.pi, dphi)
        ok_cl = cl["valid"] & (_d(_f(cl["energy"])) >= s["mine"]) & (jnp.asarray(cl["size"]) >= s["minsize"])
        take = (ok_cl[..., None] & sel[:, None, :] & (dt < s["dtmax"]) & (dt >= s["dtmin"])
                & (jnp.abs(dphi) <= math.pi / 2.0))
        nsh = jnp.sum(jnp.where(take, jnp.asarray(ph["ncombo"])[:, None, :], 0), axis=-1)
        tc = ok_cl & (nsh >= s["minnhits"])
        return {"tc": tc, "nsh": nsh, "t0": cl_time, "pos": jnp.stack([xcl, ycl, zcl], -1), "hits": take & tc[..., None]}


def tc_filter_pass(peaks, cfg):
    """TimeClusterFilter on the time peaks: a peak (all have a calo cluster) with nsh >= minNStrawHits; the plane
    requirements are not supported."""
    if int(cfg.get("minNPlanes", 0)) > 0 or int(cfg.get("minPlaneSpan", 0)) > 0:
        raise ValueError("TimeClusterFilter plane requirements not on the device")
    return jnp.any(peaks["tc"] & (peaks["nsh"] >= int(cfg["minNStrawHits"])), axis=-1)


def calo_batch(digis, rows, n_events, slots=None):
    """The batch's CaloDigis for the tracker chain: flat stock digis (`caloart.decode_digis`) of the events `rows`
    (input indices, in the block's order) packed per event (`calo.pack_digis`) as "cd_*" arrays [n_events, D], padded
    to n_events rows and to a power-of-two digi axis (>= 16), so a few shapes compile. `slots` fixes the digi axis
    (for example the largest digi count of a file, `digi_slots`), so that only the tracker hit axis changes between
    blocks: every (hit axis, digi axis) pair compiles the whole chain once."""
    from . import calo
    rows = np.asarray(rows, np.int64)
    start = np.r_[0, np.cumsum(digis["n"])]
    take = np.concatenate([np.arange(start[r], start[r + 1]) for r in rows]) if rows.size else np.zeros(0, np.int64)
    sub = {k: digis[k][take] for k in ("sipm", "t0", "peakpos", "wf_len", "wf_start")}
    sub["entry"] = np.repeat(np.arange(rows.size), digis["n"][rows])
    sub["samples"] = digis["samples"]
    sub["n"] = np.r_[digis["n"][rows], np.zeros(n_events - rows.size, np.int64)]
    pk = calo.pack_digis(sub, n_events, slots)
    D = pk["sipm"].shape[1]
    Dp = max(16, 1 << (D - 1).bit_length())
    out = {}
    for k, v in pk.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and Dp != D:
            v = np.pad(v, ((0, 0), (0, Dp - D)))
        out["cd_" + k] = v
    return out


def digi_slots(digis):
    """A fixed digi axis for `calo_batch`: the largest digi count of any event, as a power of two (>= 16)."""
    n = int(np.max(digis["n"], initial=1))
    return max(16, 1 << (n - 1).bit_length())


def jit_time_peaks(s):
    return jax.jit(lambda ph, cl: time_peaks(ph, cl, s))
