"""Flash route B: target photons at their emitting-foil exit folded through the flash response libraries.

Input is one design's target source (``flash_flux`` output: ``births.npz`` with the e+- terminal states and
``foil_exit.npz`` with the bremsstrahlung photons at the exit of the foil they were born in, with their uncollided
weight in that foil). The fold adds the at-rest annihilation photons of positrons that stopped inside a foil and
returns the expected tracker ionising deposit per source event:

- photons at or above the hard library's lowest energy edge: hard library at the exit point (the hard library carries
  its own target configuration inside the source volume, recorded in its manifest);
- photons below it: soft library (vacuum target) at the exit point times the uncollided transmission through the
  design's other foils (straight line, full slab ``2 h / |cos theta|`` per crossed annulus, own foil skipped);
- photons below the soft library's lowest edge: 0 (the soft library's declared fallback), counted.

In-flight annihilation photons are counted by the transport but have no recorded positions: they are reported as an
omitted term (count and energy per event), not folded. Deposits not owned by target photons (the residual) are not
part of this fold. Lookups run on the host in float64 and on the device in float32 and float64; all three are
reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cechain import STOPPED
from .flash_attenuation import uncollided_weight
from .flash_containment import containing_foil
from .flash_exit import foil_exit
from . import flash_response_library as F

ME = 0.51099895


def sphere_directions(n):
    """n unit vectors of a Fibonacci sphere (deterministic, near-uniform)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def _rotations(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def annihilation_photons(births, beam, *, n_dirs=128, seed=0):
    """At-rest annihilation photons of positrons stopped inside a foil annulus, as weighted rays.

    Each stopped positron gives ``n_dirs`` back-to-back pairs on a randomly rotated Fibonacci sphere (rotation
    drawn from ``seed``), each photon of weight ``1 / n_dirs`` times its uncollided transmission to the foil exit, so
    the two photons of one annihilation carry total weight 2 before attenuation. Returns photon rows and a census
    of stopped positrons and of the in-flight annihilations the transport counted without positions.
    """
    pdg = np.asarray(births["source_pdg"])
    fate = np.asarray(births["state_fate"])
    x = np.asarray(births["state_x"], float)
    stopped = (pdg == -11) & (fate == STOPPED)
    k = containing_foil(beam, x)
    rest = stopped & (k >= 0)
    nan = np.asarray(births["state_nan"], np.int64)
    ean = np.asarray(births["state_ean"], float)
    inflight_n = nan - 2 * stopped
    inflight_e = ean - 2 * ME * stopped
    census = dict(positrons=int((pdg == -11).sum()), stopped=int(stopped.sum()), stopped_in_foil=int(rest.sum()),
                  stopped_outside_foils=int((stopped & (k < 0)).sum()),
                  inflight_photons=int(np.maximum(inflight_n, 0).sum()),
                  inflight_energy_mev=float(np.maximum(inflight_e, 0).sum()),
                  counter_mismatch=int((inflight_n < 0).sum()))
    parents = np.flatnonzero(rest)
    if not parents.size:
        empty = np.zeros((0, 3))
        return dict(parent=np.zeros(0, np.int64), energy=np.zeros(0), exit_position=empty, direction=empty,
                    weight=np.zeros(0), foil_index=np.zeros(0, np.int64)), census
    base = sphere_directions(n_dirs)
    rot = _rotations(np.random.default_rng(seed), parents.size)
    d = np.einsum("pij,nj->pni", rot, base)
    d = np.concatenate([d, -d], 1)                                    # back-to-back partners
    m = d.shape[1]
    par = np.repeat(parents, m)
    pos = np.repeat(x[parents], m, axis=0)
    dd = d.reshape(-1, 3)
    kk = np.repeat(k[parents], m)
    b = beam
    ex = foil_exit(pos, dd, b.foil_z[kk], b.foil_ht[kk], b.foil_rin[kk], b.foil_rout[kk], b.axis_x)
    if not np.all(ex["valid"]):
        raise ValueError("annihilation rays without a valid foil exit")
    w = uncollided_weight(np.full(len(par), ME), ex["distance"])
    return dict(parent=par, energy=np.full(len(par), ME), exit_position=ex["position"], direction=ex["direction"],
                weight=w["weight"] / n_dirs, foil_index=kk), census


def other_foil_path_mm(position, direction, own, beam, dtype=np.float64):
    """Straight-line path through the foils other than ``own``: full slab 2 h / |cos theta| per crossed annulus."""
    p = np.asarray(position, dtype)
    d = np.asarray(direction, dtype)
    L = np.zeros(len(p), dtype)
    for j in range(len(beam.foil_z)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (dtype(beam.foil_z[j]) - p[:, 2]) / d[:, 2]
            r = np.hypot(p[:, 0] + t * d[:, 0] - dtype(beam.axis_x), p[:, 1] + t * d[:, 1])
            hit = (own != j) & np.isfinite(t) & (t > 0) & (r >= beam.foil_rin[j]) & (r <= beam.foil_rout[j])
            L = L + np.where(hit, dtype(2 * beam.foil_ht[j]) / np.abs(d[:, 2]), dtype(0))
    return L


def _keys(p, d, dtype):
    x, y, z = (np.asarray(p[:, i], dtype) for i in range(3))
    dx, dy, dz = (np.asarray(d[:, i], dtype) for i in range(3))
    ddx = x - dtype(F.DS_X)
    r = np.hypot(ddx, y)
    nrm = np.sqrt(dx * dx + dy * dy + dz * dz)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_r = np.where(r > 0, (ddx * dx + y * dy) / (r * nrm), dtype(0))
    return dict(z=z, r=r, cos_theta=dz / nrm, u_r=u_r.astype(dtype))


def response(hard, soft, photons, beam, dtype):
    """Expected deposit per photon (MeV, float64 array) and masks, for one lookup dtype (device lookup)."""
    dt = np.dtype(dtype).type
    e = np.asarray(photons["energy"], float)
    k = _keys(photons["exit_position"], photons["direction"], dt)
    h_lo = hard.axis("energy").edges[0]
    s_lo = soft.axis("energy").edges[0]
    is_soft = e < h_lo
    below = e < s_lo
    hv = np.asarray(F.lookup(F.to_device(hard, dt), e.astype(dt), k["z"], k["r"], k["cos_theta"], k["u_r"]), float)
    sv = np.asarray(F.lookup(F.to_device(soft, dt), e.astype(dt), k["z"], k["r"], k["cos_theta"], k["u_r"]), float)
    L = other_foil_path_mm(photons["exit_position"], photons["direction"], photons["foil_index"], beam, dt)
    tw = uncollided_weight(e, L.astype(float))
    trans = np.where(is_soft & ~below, tw["weight"], 1.0)
    val = np.where(is_soft, np.where(below, 0.0, sv * trans), hv)
    refused = ~np.isfinite(val)
    return dict(value=np.where(refused, 0.0, val), refused=refused, soft=is_soft & ~below, below=below,
                transmission=trans)


def _host_check(hard, soft, photons, dev64):
    """Host float64 lookups on the rows the device answered; True when they equal the device float64 values."""
    e = np.asarray(photons["energy"], float)
    k = _keys(photons["exit_position"], photons["direction"], np.float64)
    ok = ~dev64["refused"]
    hard_rows = ok & (e >= hard.axis("energy").edges[0])
    soft_rows = ok & dev64["soft"]
    same = True
    for lib, m in ((hard, hard_rows), (soft, soft_rows)):
        if m.any():
            sub = {n: v[m] for n, v in k.items()}
            hv = F.lookup_numpy(lib, e[m], sub["z"], sub["r"], sub["cos_theta"], sub["u_r"])
            dv = np.asarray(F.lookup(F.to_device(lib, np.float64), e[m], sub["z"], sub["r"], sub["cos_theta"],
                                     sub["u_r"]), float)
            same &= bool(np.array_equal(hv, dv))
    return same


def event_index(births, parent):
    """Source-event row of each photon's parent, from the births' event ledger."""
    ledger = np.asarray(births["source_event_ledger"]).astype(np.int64)
    ids = np.stack([births["source_run"], births["source_subrun"], births["source_event"]], 1).astype(np.int64)
    u, inv = np.unique(np.concatenate([ledger, ids]), axis=0, return_inverse=True)
    inv = inv.ravel()
    if len(u) != len(ledger):
        raise ValueError("source parents outside the event ledger, or duplicate ledger rows")
    row = np.empty(len(u), np.int64)
    row[inv[:len(ledger)]] = np.arange(len(ledger))
    return row[inv[len(ledger):]][parent], len(ledger)


def fold(births, exits, beam, hard, soft, *, n_dirs=128, seed=0, keep_per_event=False):
    """Fold one design: per-event expected deposit by component, for host and device dtypes.

    ``keep_per_event`` keeps the per-source-event arrays (``out[component][dtype]["per_event"]``) for resampling
    over source events (e.g. a bootstrap shared by several designs on the same source)."""
    ok = np.asarray(exits["foil_exit"], bool) & np.asarray(exits["attenuation_supported"], bool)
    brems = dict(parent=np.asarray(exits["parent"])[ok], energy=np.asarray(exits["energy"], float)[ok],
                 exit_position=np.asarray(exits["exit_position"], float)[ok],
                 direction=np.asarray(exits["direction"], float)[ok],
                 weight=np.asarray(exits["weight"], float)[ok], foil_index=np.asarray(exits["foil_index"])[ok])
    ann, census = annihilation_photons(births, beam, n_dirs=n_dirs, seed=seed)
    out = dict(census=dict(annihilation=census, brems_exits=int(ok.sum()),
                           brems_dropped=int((~ok).sum())))
    ev = {}
    n_events = None
    for name, ph in (("brems", brems), ("annihilation", ann)):
        evi, n_events = event_index(births, ph["parent"])
        ev[name] = evi
        comp = {}
        for dt in (np.float32, np.float64):
            r = response(hard, soft, ph, beam, dt)
            contrib = ph["weight"] * r["value"]
            per_event = np.bincount(evi, weights=contrib, minlength=n_events)
            parts = {lab: float((ph["weight"] * r["value"] * m).sum() / n_events)
                     for lab, m in (("hard", ~r["soft"] & ~r["below"]), ("soft", r["soft"]))}
            comp[np.dtype(dt).name] = dict(per_event=per_event, mean=float(per_event.mean()),
                                           se=float(per_event.std(ddof=1) / np.sqrt(n_events)), **parts,
                                           refused=int(r["refused"].sum()),
                                           refused_weight=float(ph["weight"][r["refused"]].sum() / n_events),
                                           below_soft=int(r["below"].sum()),
                                           below_soft_energy=float((ph["weight"] * ph["energy"])[r["below"]].sum()
                                                                   / n_events),
                                           soft_transmission_mean=float(np.average(r["transmission"][r["soft"]],
                                                                                   weights=ph["weight"][r["soft"]])
                                                                        if r["soft"].any() else 1.0))
            if dt is np.float64:
                comp["host_equals_device_float64"] = _host_check(hard, soft, ph, r)
        out[name] = comp
    out["events"] = int(n_events)
    tot = {}
    for dt in ("float32", "float64"):
        pe = out["brems"][dt]["per_event"] + out["annihilation"][dt]["per_event"]
        tot[dt] = dict(mean=float(pe.mean()), se=float(pe.std(ddof=1) / np.sqrt(n_events)))
    out["total"] = tot
    out["omitted"] = dict(inflight_annihilation_photons_per_event=census["inflight_photons"] / n_events,
                          inflight_annihilation_energy_per_event_mev=census["inflight_energy_mev"] / n_events,
                          note="in-flight annihilation photons have no recorded positions; not folded")
    if not keep_per_event:
        for name in ("brems", "annihilation"):
            for dt in ("float32", "float64"):
                out[name][dt].pop("per_event")
    return out


def run(args):
    from . import flash_geometry as FG
    hard = F.load(args.hard, expect={"content_hash": args.hard_hash} if args.hard_hash else None)
    soft = F.load(args.soft, expect={"content_hash": args.soft_hash} if args.soft_hash else None)
    geom = FG.from_offline(args.geom, args.offline_root)
    s1 = Path(args.s1)
    with np.load(s1 / "births.npz") as b, np.load(s1 / "foil_exit.npz") as e:
        births = {k: b[k] for k in b.files}
        exits = {k: e[k] for k in e.files}
    res = fold(births, exits, geom.beam, hard, soft, n_dirs=args.n_dirs, seed=args.seed,
               keep_per_event=bool(args.per_event_out))
    if args.per_event_out:
        np.savez(args.per_event_out, event=np.asarray(births["source_event_ledger"]),
                 **{f"{c}_{dt}": res[c][dt].pop("per_event") for c in ("brems", "annihilation")
                    for dt in ("float32", "float64")})
    from mu2efastsim.library import provenance as LP
    res.update(design=args.name or s1.name, geometry=str(args.geom), libraries=LP.summary(),
               n_dirs=args.n_dirs, seed=args.seed)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    t = res["total"]
    print(f"{res['design']}: {t['float64']['mean']:.4e} +- {t['float64']['se']:.2e} MeV/event "
          f"(float32 {t['float32']['mean']:.4e}); brems {res['brems']['float64']['mean']:.3e}, "
          f"annihilation {res['annihilation']['float64']['mean']:.3e}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s1", required=True, help="flash_flux output directory (births.npz, foil_exit.npz)")
    ap.add_argument("--geom", required=True)
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--hard", required=True)
    ap.add_argument("--soft", required=True)
    ap.add_argument("--hard-hash")
    ap.add_argument("--soft-hash")
    ap.add_argument("--n-dirs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-event-out", help="also write the per-source-event deposits (npz) for resampling")
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
