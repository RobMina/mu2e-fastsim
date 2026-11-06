"""Opt-in Run 1B TS transport: field RK4, thin material loss/scattering and pion decay.

This development model follows one charged mu/pi lineage per recorded entrance.
Nuclear interactions, wall return, muon-decay electrons and stopped-pion daughters
are absent. Material columns are straight chords of each RK4 arc; halving the step
is a development check, not an exact navigation claim. No other chain imports it.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from mu2efastsim import ehelix, eloss

from . import tsfield, tsphysics as phys

FATES = ("live", "ds", "rangeout", "mu_decay", "wall", "upstream", "path_limit", "field_missing")
LIVE, DS, RANGEOUT, MU_DECAY, WALL, UPSTREAM, PATH_LIMIT, FIELD_MISSING = range(len(FATES))


def charge(pdg):
    """PDG signs differ between pions and muons."""
    phys.mass_mev(pdg)
    return np.where(np.abs(pdg) == 211, np.sign(pdg), -np.sign(pdg))


def radiation_length(mat):
    return mat.x0_mm if hasattr(mat, "x0_mm") else eloss.radiation_length_mm(mat)


def transport(source, field, geometry, seed=20261105, step_mm=5.0, max_path_mm=100000.0,
              material_step_mm=0.25, loss_fraction=0.05, method="rk4", progress=False,
              boundary_safety=True, trace=None):
    """Transport aligned x/p/t/pdg arrays, returning every input row including losses.

    geometry supplies inside(x), columns(start,end)->{name:length_mm}, materials,
    z_start and z_end. Missing field coverage, upstream returns and path-capped
    tracks have explicit fates. The terminal downstream plane is not called a VD.
    Decay is drawn once in proper time, reset for the muon daughter; loss and time
    are split at step resolution. Per-material Highland history is cumulative.
    Optional trace observes copied selected-lane states; it must not alter RNG or physics.
    """
    settings = (step_mm, max_path_mm, material_step_mm, loss_fraction)
    if not all(np.isfinite(v) and v > 0 for v in settings) or loss_fraction > 0.2:
        raise ValueError("positive finite steps/path and 0<loss_fraction<=0.2 required")
    if method not in ("rk4", "helix"):
        raise ValueError("method must be rk4 or helix")
    x, p = (np.array(source[k], float, copy=True) for k in ("x", "p"))
    pdg = np.array(source["pdg"], dtype=np.int64, copy=True)
    t = np.array(source["t"], dtype=float, copy=True)
    n = len(pdg)
    if trace is not None:
        trace_rows = np.asarray(trace.rows)
        if (trace_rows.ndim != 1 or trace_rows.dtype.kind not in 'iu'
                or np.any((trace_rows < 0) | (trace_rows >= n))
                or len(np.unique(trace_rows)) != len(trace_rows)):
            raise ValueError('trace rows must be distinct valid source indices')
        trace_mask = np.zeros(n, dtype=bool)
        trace_mask[trace_rows] = True
    if x.shape != (n, 3) or p.shape != (n, 3) or t.shape != (n,):
        raise ValueError("x,p must be N by 3; pdg,t must be N")
    if not all(np.all(np.isfinite(a)) for a in (x, p, t)):
        raise ValueError("nonfinite initial state")
    masses = phys.mass_mev(pdg)
    norm = np.linalg.norm(p, axis=1)
    if np.any(norm <= 0) or np.any(~geometry.inside(x)):
        raise ValueError("entrance states must move inside the TS aperture")
    rng = np.random.default_rng(seed)
    remaining = phys.sample_proper_lifetime(rng, pdg)
    fate, path, decayed = np.zeros(n, np.int8), np.zeros(n), np.zeros(n, bool)
    steps = np.zeros(n, np.int32)
    history = {name: np.zeros(n) for name in geometry.materials}
    material_path = np.zeros(n)
    start = time.perf_counter()
    iteration = 0
    while np.any(fate == LIVE):
        live = np.flatnonzero(fate == LIVE)
        mass = phys.mass_mev(pdg[live])
        pmag = np.linalg.norm(p[live], axis=1)
        kinetic = pmag ** 2 / (np.hypot(pmag, mass) + mass)
        cold = kinetic <= phys.TK_STOP_MEV
        fate[live[cold]] = RANGEOUT
        live = live[~cold]
        if not len(live):
            continue
        mass, pmag, kinetic = mass[~cold], pmag[~cold], kinetic[~cold]
        u, q, oldx = p[live] / pmag[:, None], charge(pdg[live]), x[live].copy()
        decay_path = remaining[live] * pmag * phys.C_MM_NS / mass
        h = np.minimum(np.minimum(step_mm, max_path_mm - path[live]), decay_path)
        initial_field, valid = field.evaluate(oldx)
        fate[live[~valid]] = FIELD_MISSING
        h[~valid] = 0
        h = np.minimum(h, 0.1 * pmag / np.maximum(ehelix.K * np.linalg.norm(initial_field, axis=1), 1e-30))
        stage_valid = valid.copy()

        def advance(hh, lanes=slice(None)):
            nonlocal stage_valid
            stage_valid = valid[lanes].copy()
            first = True
            def evaluate(xx):
                nonlocal stage_valid, first
                if first:
                    first = False
                    return initial_field[lanes]
                b, ok = field.evaluate(xx)
                stage_valid &= ok
                return b
            if method == "rk4":
                return ehelix.rk4(oldx[lanes], u[lanes], pmag[lanes], q[lanes], evaluate, hh)
            mid = ehelix.helix(oldx[lanes], u[lanes], pmag[lanes], q[lanes], evaluate(oldx[lanes]), 0.5 * hh)[0]
            return ehelix.helix(oldx[lanes], u[lanes], pmag[lanes], q[lanes], evaluate(mid), hh)

        def material_scale(columns):
            mean, length = np.zeros(len(live)), np.zeros(len(live))
            for name, mm in columns.items():
                hit = mm > 0
                if hit.any():
                    mean[hit] += eloss.dedx_restricted(geometry.materials[name], pmag[hit], 1e9,
                                                     mass=mass[hit]) * mm[hit]
                    length += mm
            scale = np.minimum(1, material_step_mm / np.maximum(length, 1e-30))
            return np.minimum(scale, loss_fraction * kinetic / np.maximum(mean, 1e-30))

        for boundary_attempt in range(129):
            xn, un = advance(h)
            columns = geometry.columns(oldx, xn)
            for attempt in range(129):
                scale = material_scale(columns)
                changed = scale < 1
                if not changed.any():
                    break
                if attempt == 128:
                    raise RuntimeError("material step bounds did not converge")
                full_valid = stage_valid.copy()
                shrink = scale[changed] * (1 - 1e-6)
                h[changed] *= np.minimum(0.5, shrink) if attempt >= 5 else shrink
                if np.any(h[changed] <= 0):
                    raise RuntimeError("material step shortening underflow")
                xn[changed], un[changed] = advance(h[changed], changed)
                full_valid[changed] = stage_valid
                stage_valid = full_valid
                shortened = geometry.columns(oldx[changed], xn[changed])
                for name in columns:
                    columns[name][changed] = shortened[name]

            proposed_h = h.copy()
            cross = (oldx[:, 2] < geometry.z_end) & (xn[:, 2] >= geometry.z_end) & (xn[:, 0] < 0)
            accepted_valid = stage_valid.copy()
            if cross.any():
                lo, hi = np.zeros(np.count_nonzero(cross)), h[cross].copy()
                for _ in range(25):
                    mid = (lo + hi) / 2
                    xm, _ = advance(mid, cross)
                    before = xm[:, 2] < geometry.z_end
                    lo = np.where(before, mid, lo)
                    hi = np.where(before, hi, mid)
                h[cross] = (lo + hi) / 2
                xn[cross], un[cross] = advance(h[cross], cross)
                accepted_valid[cross] = stage_valid
                shortened = geometry.columns(oldx[cross], xn[cross])
                for name in columns:
                    columns[name][cross] = shortened[name]
                xn[cross, 2] = geometry.z_end
            previous = np.zeros(len(live))
            first_bad = np.full(len(live), np.nan)
            probe = np.ones(len(live), bool)
            bound = getattr(field, 'max_field_t', np.inf)
            if boundary_safety and np.isfinite(bound) and hasattr(field, 'safety') and hasattr(geometry, 'safety'):
                a = h * ehelix.K * bound / pmag
                radius = h * np.maximum(1+a/2+a*a/4, 1+a/2+a*a/6+a**3/24) if method == 'rk4' else h.copy()
                radius += 1e-8 + 1e-12 * (np.max(np.abs(oldx), axis=1) + radius)
                probe = ~(valid & (geometry.safety(oldx) > radius) & (field.safety(oldx) > radius))
            for fraction in (0.25, 0.5, 0.75, 1.0):
                bad = np.zeros(len(live), bool)
                if fraction == 1.0:
                    bad = ~geometry.inside(xn) & np.isnan(first_bad)
                elif probe.any():
                    xm, _ = advance(h[probe] * fraction, probe)
                    accepted_valid[probe] &= stage_valid
                    bad[probe] = ~geometry.inside(xm) & np.isnan(first_bad[probe])
                first_bad[bad] = h[bad] * fraction
                previous = np.where(np.isnan(first_bad), h * fraction, previous)
            crossed_wall = np.isfinite(first_bad) & valid
            upstream_exit = np.zeros(len(live), bool)
            if crossed_wall.any():
                lo, hi = previous[crossed_wall].copy(), first_bad[crossed_wall].copy()
                for _ in range(20):
                    mid = (lo + hi) / 2
                    xm, _ = advance(mid, crossed_wall)
                    inside = geometry.inside(xm)
                    lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
                outside_x, _ = advance(hi, crossed_wall)
                candidates = (outside_x[:, 2] < geometry.z_start) & (outside_x[:, 0] > 0)
                if candidates.any():
                    lanes = np.flatnonzero(crossed_wall)[candidates]
                    pl, ph = lo[candidates].copy(), hi[candidates].copy()
                    for _ in range(20):
                        mid = (pl + ph) / 2
                        xm, _ = advance(mid, lanes)
                        before = xm[:, 2] >= geometry.z_start
                        pl, ph = np.where(before, mid, pl), np.where(before, ph, mid)
                    plane_x, _ = advance((pl + ph) / 2, lanes)
                    plane_x[:, 2] = geometry.z_start
                    upstream_exit[lanes] = geometry.inside(plane_x)
                h[crossed_wall] = lo
                xn[crossed_wall], un[crossed_wall] = advance(h[crossed_wall], crossed_wall)
                accepted_valid[crossed_wall] = stage_valid
                for fraction in (0.25, 0.5, 0.75):
                    advance(h[crossed_wall] * fraction, crossed_wall)
                    accepted_valid[crossed_wall] &= stage_valid
                updated = crossed_wall | cross
                shortened = geometry.columns(oldx[updated], xn[updated])
                for name in columns:
                    columns[name][updated] = shortened[name]
            scale = material_scale(columns)
            retry = scale < 1
            if not retry.any():
                break
            if boundary_attempt == 128:
                raise RuntimeError("boundary material step bounds did not converge")
            h = np.where(retry, h * np.minimum(0.5, scale), proposed_h)
        fate[live[valid & ~accepted_valid]] = FIELD_MISSING
        valid &= accepted_valid
        if trace is not None:
            selected = np.flatnonzero(trace_mask[live])
            trace.begin(row=live[selected].copy(), step=steps[live[selected]].copy(),
                        x0=oldx[selected].copy(), proposed_x=xn[selected].copy(),
                        p0=p[live[selected]].copy(), t0=t[live[selected]].copy(),
                        path0=path[live[selected]].copy(), h=h[selected].copy(),
                        pdg0=pdg[live[selected]].copy(), valid=valid[selected].copy())
        newp = un * pmag[:, None]
        travelled = np.ones(len(live))
        stopped = np.zeros(len(live), bool)
        for name, mm in columns.items():
            active = (mm > 0) & ~stopped & valid
            if active.any():
                if trace is not None:
                    traced_active = np.flatnonzero(trace_mask[live[active]])
                    before_material = newp[active][traced_active].copy()
                out, fraction, stop = phys.material_step(rng, geometry.materials[name], newp[active],
                                                        pdg[live[active]], mm[active], history[name][live[active]])
                if trace is not None and len(traced_active):
                    trace.material(name=name, row=live[active][traced_active].copy(),
                                   requested_mm=mm[active][traced_active].copy(),
                                   fraction=fraction[traced_active].copy(), stopped=stop[traced_active].copy(),
                                   p_before=before_material, p_after=out[traced_active].copy())
                newp[active] = out
                travelled[active] = np.minimum(travelled[active], fraction)
                stopped[active] |= stop
                history[name][live[active]] += mm[active] * fraction / radiation_length(geometry.materials[name])
                material_path[live[active]] += mm[active] * fraction
        actual = h * travelled
        x[live] = oldx + (xn - oldx) * travelled[:, None]
        p[live] = newp
        path[live] += actual
        steps[live] += 1
        t[live] += actual * np.hypot(pmag, mass) / (pmag * phys.C_MM_NS)
        remaining[live] -= actual * mass / (pmag * phys.C_MM_NS)
        fate[live[stopped & valid]] = RANGEOUT
        _, end_valid = field.evaluate(x[live])
        fate[live[valid & ~end_valid & ~stopped]] = FIELD_MISSING
        running = fate[live] == LIVE
        at_decay = running & (remaining[live] <= 1e-9)
        mu = at_decay & (np.abs(pdg[live]) == 13)
        fate[live[mu]] = MU_DECAY
        pion = at_decay & ~mu
        if pion.any():
            ids = live[pion]
            p[ids], _ = phys.decay_pions(rng, p[ids])
            pdg[ids] = -13 * np.sign(pdg[ids])
            remaining[ids] = phys.sample_proper_lifetime(rng, pdg[ids])
            decayed[ids] = True
        running = fate[live] == LIVE
        downstream = running & cross & ~crossed_wall & (np.abs(x[live, 2] - geometry.z_end) < 1e-7) & geometry.inside(x[live])
        fate[live[downstream]] = DS
        upstream = running & upstream_exit
        fate[live[upstream]] = UPSTREAM
        outside = running & (crossed_wall | ~geometry.inside(x[live])) & ~upstream & ~downstream
        fate[live[outside]] = WALL
        capped = (fate[live] == LIVE) & (path[live] >= max_path_mm - 1e-8)
        fate[live[capped]] = PATH_LIMIT
        if trace is not None:
            trace.end(x1=x[live[selected]].copy(), p1=p[live[selected]].copy(),
                      t1=t[live[selected]].copy(), path1=path[live[selected]].copy(),
                      pdg1=pdg[live[selected]].copy(), fate=fate[live[selected]].copy())
        iteration += 1
        if progress and iteration % 1000 == 0:
            print(f"steps={iteration} live={np.count_nonzero(fate == LIVE)} ds={np.count_nonzero(fate == DS)}", flush=True)
    return dict(x=x, p=p, t=t, pdg=pdg, fate=fate, path=path, pion_decay=decayed, steps=steps,
                material_path=material_path, seconds=np.asarray(time.perf_counter() - start))


def main(argv=None):
    from .tsgeometry import Geometry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--field-config", required=True)
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--maps", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--max-path", type=float, default=100000.0)
    parser.add_argument("--method", choices=("rk4", "helix"), default="rk4")
    parser.add_argument("--shards", type=int, nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    with np.load(args.reference) as data:
        source = {k: data[k] for k in data.files}
    mask = np.isin(source["shard"], args.shards) if args.shards else np.ones(len(source["pdg"]), bool)
    selected = np.flatnonzero(mask)[:args.limit]
    batch = {k: source[k][selected] for k in ("x", "p", "t", "pdg")}
    field = tsfield.from_config(args.field_config, args.offline_root, args.maps)
    geometry = Geometry(args.gdml)
    result = transport(batch, field, geometry, args.seed, args.step, args.max_path, method=args.method, progress=True)
    result.update(id=source["id"][selected], shard=source["shard"][selected])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    print(json.dumps({"n": len(selected), "fates": {name: int(np.sum(result["fate"] == i))
                      for i, name in enumerate(FATES)}, "seconds": float(result["seconds"]),
                      "step_mm": args.step, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
