"""Diagnostic replay of a saved arm, frozen first-foil entries and same-law mean-loss range.

The temporary JAX observation hooks leave production sources/defaults untouched. Replay must reproduce all saved
output arrays exactly before its histories are analyzed. Mean-only frozen-entry results remove straggling, MSC
and trajectory changes; the integrated range is a numerical reference, not independent low-energy muon physics.
"""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from scipy.integrate import cumulative_trapezoid

from mu2efastsim import eloss
from . import g4ref, geometry
from .reference_audit import _identity
from .slicestudy import CORE_FILES, PHYSICS_FILES, verify_hashes


SLICES = (4, 8, 16, 32, 64, 128)


def momentum(kinetic):
    return np.sqrt(kinetic * (kinetic + 2 * eloss.MMU))


def kinetic(momentum_mev):
    return np.sqrt(momentum_mev ** 2 + eloss.MMU ** 2) - eloss.MMU


def mean_history(energy, length, slices, mat=geometry.STOPPING_TARGET_AL, tcut=1.0):
    """Vectorized explicit mean-loss slices, with the production layer's zero-energy clamp."""
    if isinstance(slices, bool) or not isinstance(slices, (int, np.integer)) or slices <= 0:
        raise ValueError("slices must be a positive integer")
    energy, length = np.broadcast_arrays(np.asarray(energy, float), np.asarray(length, float))
    if energy.ndim != 1 or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(length)):
        raise ValueError("finite one-dimensional inputs required")
    if np.any(energy < 0) or np.any(length < 0) or not np.isfinite(tcut) or tcut <= 0:
        raise ValueError("nonnegative energies/paths and positive cut required")
    history = np.empty((slices + 1, len(energy)))
    history[0] = energy
    requested = np.empty((slices, len(energy)))
    for i in range(slices):
        requested[i] = eloss.dedx_restricted(mat, momentum(history[i]), tcut) * length / slices
        history[i + 1] = np.maximum(history[i] - requested[i], 0.0)
    return history, requested


def range_table(max_energy, points=32769, threshold=0.1, mat=geometry.STOPPING_TARGET_AL, tcut=1.0):
    """Integral of 1/S from threshold; no extrapolation to zero or independent physics claim."""
    if not np.isfinite(max_energy) or max_energy <= threshold or threshold <= 0 or points < 3:
        raise ValueError("invalid range-table domain")
    energy = np.geomspace(threshold, max_energy, points)
    stopping = eloss.dedx_restricted(mat, momentum(energy), tcut)
    if not np.all(np.isfinite(stopping)) or np.any(stopping <= 0):
        raise ValueError("nonpositive stopping power in range-table domain")
    return energy, cumulative_trapezoid(1 / stopping, energy, initial=0)


def range_out(energy, length, table):
    """Remaining energy and threshold-stop flag for the same-law integrated range."""
    energy, length = np.broadcast_arrays(np.asarray(energy, float), np.asarray(length, float))
    if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(length)) or np.any(energy < 0) or np.any(length < 0):
        raise ValueError("finite nonnegative energy/path required")
    grid, ranges = table
    if np.any(energy > grid[-1]):
        raise ValueError("entry exceeds range-table domain")
    residual = np.interp(energy, grid, ranges) - length
    stop = (energy < grid[0]) | (residual < 0)
    return np.where(stop, 0.0, np.interp(residual, ranges, grid)), stop


def quantiles(values):
    values = np.asarray(values)
    return np.quantile(values, [0, .1, .5, .9, 1]).tolist() if len(values) else []


class LayerCapture:
    """Collect active material encounters with original rows through the production compaction layout."""

    def __init__(self, geom, n, min_bucket=256):
        self.geom, self.rows, self.min_bucket = geom, np.arange(n), min_bucket
        self.chunks, self.step = 0, -1
        self.parts = []
        self.position = None
        self.samples = []

    def sample(self, p, loss):
        self.samples.append((np.asarray(p), np.asarray(loss)))

    def flight(self, x, xn):
        self.position = np.asarray(x), np.asarray(xn)
        self.step += 1

    def layer(self, key, material_i, p, u, length, active, pout):
        x, xn = self.position
        active = np.asarray(active) & (self.rows >= 0)
        lanes = np.flatnonzero(active)
        samples, self.samples = self.samples, []
        if not samples:
            raise ValueError("layer has no sampled loss histories")
        if not len(lanes):
            return
        z0, z1 = x[lanes, 2], xn[lanes, 2]
        fz = self.geom.foil_z
        crossed = ((fz[None, :] > z0[:, None]) & (fz[None, :] <= z1[:, None])) | \
                  ((fz[None, :] < z0[:, None]) & (fz[None, :] >= z1[:, None]))
        foil = np.argmax(crossed, axis=1)
        al = np.isclose(np.broadcast_to(np.asarray(material_i), np.shape(p))[lanes],
                        geometry.STOPPING_TARGET_AL.I_eV * 1e-6, rtol=1e-12, atol=0)
        if np.any(al & ~crossed.any(axis=1)):
            raise ValueError("Al layer without a foil-plane crossing")
        fraction = (fz[foil] - z0) / np.where(z1 != z0, z1 - z0, 1.0)
        xc = x[lanes] + fraction[:, None] * (xn[lanes] - x[lanes])
        count = len(lanes)
        self.parts.append(dict(row=self.rows[lanes].copy(), lane=lanes, lanes=np.full(count, len(p)),
                               chunk=np.full(count, self.chunks), step=np.full(count, self.step),
                               key=np.broadcast_to(np.asarray(key), (count, 2)).copy(),
                               foil=np.where(al, foil, -1), material=np.where(al, 0, 1),
                               p_in=np.asarray(p)[lanes], p_out=np.asarray(pout)[lanes],
                               u_in=np.asarray(u)[lanes], length_mm=np.asarray(length)[lanes],
                               slice_p_in=np.stack([s[0][lanes] for s in samples], axis=1),
                               slice_loss_MeV=np.stack([s[1][lanes] for s in samples], axis=1),
                               x_before=x[lanes], x_after=xn[lanes], foil_crossing_mm=xc))

    def compact(self, state):
        fate = np.asarray(state.fate)
        live = (self.rows >= 0) & (fate == 0)
        self.rows = self.rows[live]
        if len(self.rows):
            m = max(self.min_bucket, 1 << int(np.ceil(np.log2(len(self.rows)))))
            self.rows = np.concatenate([self.rows, np.full(m - len(self.rows), -1)])
        self.chunks += 1
        self.step = -1

    def arrays(self):
        if not self.parts:
            raise ValueError("no material encounters recorded")
        return {k: np.concatenate([part[k] for part in self.parts]) for k in self.parts[0]}


def capture_transport(fm, geom, prm, key, state, chunk=200, min_bucket=256):
    """Standalone-process observation hooks; restore functions and isolate the JIT cache on every exit."""
    import jax
    from . import transport
    capture = LayerCapture(geom, len(state.p), min_bucket)
    rk4, layer, stepper = transport._rk4, transport._layer, transport.make_stepper
    sample = transport.eloss_jax._sample

    def observed_sample(key, mat, p, length, tcut, mass, **kw):
        loss = sample(key, mat, p, length, tcut, mass, **kw)
        jax.debug.callback(capture.sample, p, loss, ordered=True)
        return loss

    def observed_rk4(*args):
        xn, un = rk4(*args)
        jax.debug.callback(capture.flight, args[1], xn, ordered=True)
        return xn, un

    def observed_layer(key, mat, x0, p, u, length, active, prm, *tables):
        result = layer(key, mat, x0, p, u, length, active, prm, *tables)
        jax.debug.callback(capture.layer, key, mat.I, p, u, length, active, result[0], ordered=True)
        return result

    def observed_stepper(*args, **kwargs):
        run = stepper(*args, **kwargs)

        def observed_run(key, state):
            result, steps = run(key, state)
            jax.block_until_ready(result)
            jax.effects_barrier()
            capture.compact(result)
            return result, steps

        return observed_run

    with patch.object(transport, "_CACHE", {}), patch.object(transport, "_rk4", observed_rk4), \
            patch.object(transport.eloss_jax, "_sample", observed_sample), \
            patch.object(transport, "_layer", observed_layer), patch.object(transport, "make_stepper", observed_stepper):
        result, steps = transport.transport_all(fm, geom, prm, key, state, chunk=chunk, min_bucket=min_bucket)
    return result, steps, capture.arrays()


def verify_replay(result, saved):
    matches = {k: bool(np.array_equal(np.asarray(getattr(result, k)), saved[k]))
               for k in ("x", "p", "t", "fate", "foil", "nfoil")}
    if not all(matches.values()):
        raise ValueError(f"Observed replay differs from saved arm: {matches}")
    return matches


def frozen_study(history, g, geom):
    """First encounter with foil zero per muon; all ancestry/outcome-conditioned uses are diagnostics."""
    eligible = np.flatnonzero((history["material"] == 0) & (history["foil"] == 0))
    rows, first = np.unique(history["row"][eligible], return_index=True)
    selected = eligible[first]
    if not len(selected):
        raise ValueError("no first-foil entrants")
    entry = kinetic(history["p_in"][selected])
    length = history["length_mm"][selected]
    limit = max(float(entry.max()) * 1.01, 10.0)
    table = range_table(limit)
    fine = range_table(limit, points=65537)
    integrated, stops = range_out(entry, length, table)
    integrated_fine, stops_fine = range_out(entry, length, fine)
    prior = np.minimum.reduceat(np.arange(len(history["row"]))[np.argsort(history["row"], kind="stable")],
                                np.r_[0, np.flatnonzero(np.diff(np.sort(history["row"]))) + 1])
    allrows = np.unique(history["row"])
    first_any = prior[np.searchsorted(allrows, rows)]
    store = dict(row=rows, history_index=selected, entry_MeV=entry, length_mm=length,
                 range_out_MeV=integrated, range_stop=stops, range_energy=table[0], range_mm=table[1])
    report = dict(entrants=len(rows), selected_rows=rows.tolist(), prior_material=int(np.sum(selected != first_any)),
                  backward=int(np.sum(history["u_in"][selected, 2] < 0)),
                  angle_floor=int(np.sum(np.abs(history["u_in"][selected, 2]) < .02)),
                  entry_MeV_quantiles=quantiles(entry), length_mm_quantiles=quantiles(length),
                  entry_below_0p2_MeV=int(np.sum(entry < .2)),
                  observed_layer_stops=int(np.sum(kinetic(history["p_out"][selected]) < .1)),
                  g4_first_foil_for_entrants=int(np.sum(g["foil"][rows] == 0)),
                  g4_first_foil_total=int(np.sum(g["foil"] == 0)),
                  same_law_range_stops=int(stops.sum()),
                  range_grid_max_abs_mm=float(np.max(np.abs(table[1] - np.interp(table[0], *fine)))),
                  range_grid_stop_disagreements=int(np.sum(stops != stops_fine)),
                  range_grid_max_abs_exit_MeV=float(np.max(np.abs(integrated - integrated_fine))),
                  mean_slices={})
    for slices in SLICES:
        energies, requested = mean_history(entry, length, slices)
        stop = energies[-1] < .1
        alive = energies[:-1] >= .1
        fractions = requested / np.maximum(energies[:-1], 1e-300)
        store[f"energy_s{slices}"] = energies
        store[f"requested_loss_s{slices}"] = requested
        report["mean_slices"][str(slices)] = dict(
            stops=int(stop.sum()), disagree_range=int(np.sum(stop != stops)),
            understop=int(np.sum(~stop & stops)), overstop=int(np.sum(stop & ~stops)),
            entry_rows_understop=rows[~stop & stops].tolist(),
            samples_below_0p2_MeV=int(np.sum((energies[:-1] >= .1) & (energies[:-1] < .2))),
            large_fraction_slices=int(np.sum(alive & (fractions > .1))),
            remaining_MeV_quantiles=quantiles(energies[-1][~stops]),
            mean_excess_MeV_on_range_survivors=float(np.mean((energies[-1] - integrated)[~stops]))
            if np.any(~stops) else None)
    paths = 2 * geom.foil_ht[0] / np.array([1.0, .5, .2])
    if np.any(paths >= table[1][-1]):
        raise ValueError("critical-energy paths exceed the range-table bracket")
    critical = dict(path_mm=paths.tolist(), same_law_range_MeV=np.interp(paths, table[1], table[0]).tolist(), slices={})
    for slices in SLICES:
        lo, hi = np.full(3, .1), np.full(3, limit)
        upper, _ = mean_history(hi, paths, slices)
        if np.any(upper[-1] < .1):
            raise ValueError("critical-energy upper bracket does not survive")
        for _ in range(48):
            mid = (lo + hi) / 2
            energy, _ = mean_history(mid, paths, slices)
            stop = energy[-1] < .1
            lo, hi = np.where(stop, mid, lo), np.where(stop, hi, mid)
        critical["slices"][str(slices)] = ((lo + hi) / 2).tolist()
    report["critical_entry_energy"] = critical
    return report, store


def main(argv=None):
    from . import __main__ as cli
    from . import fieldmap, transport
    import jax
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geom")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    base = Path(a.baseline)
    manifest = json.loads((base / "manifest.json").read_text())
    saved_summary = json.loads((base / "summary.json").read_text())
    source = Path(__file__).resolve().parent
    core_dir = Path(__import__("mu2efastsim").__file__).resolve().parent
    code = {Path(p["path"]).name: p for p in manifest["code"]}
    physics = [_identity(source / p) for p in PHYSICS_FILES]
    core = [_identity(core_dir / p) for p in CORE_FILES]
    verify_hashes([code[p] for p in PHYSICS_FILES], physics, "beam physics")
    verify_hashes(manifest["core_code"], core, "core physics")
    input_path = base / "input_block.npz"
    verify_hashes([manifest["selected_input"]], [_identity(input_path)], "input")
    geom = cli._geom(a.geom, target_contract=True)
    sources = []
    geometry.from_offline(a.geom or cli.GEOM_FILE, cli.OFFLINE_ROOT, source_files=sources)
    geo = [_identity(p) for p in sources]
    field = [_identity(Path(cli.MAPS) / f"{cli.MAP}.{ext}") for ext in ("header", "bin")]
    verify_hashes(manifest["geometry"], geo, "geometry")
    verify_hashes(manifest["field"], field, "field")
    tag = f"nominal_seed{a.seed}"
    saved_path = base / f"{tag}.npz"
    g, saved = g4ref.load(input_path), g4ref.load(saved_path)
    meta = json.loads(str(saved["meta"]))
    prm = transport.Params()
    n = len(g["evt"])
    if a.seed not in manifest["seeds"] or meta != saved_summary["runs"][tag]["meta"]:
        raise ValueError("seed/metadata mismatch")
    if meta["params"] != prm._asdict() or meta["n"] != n or len(saved["fate"]) != n:
        raise ValueError("requires nominal 4-slice, one-batch whole-event pilot")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance = dict(baseline_manifest=_identity(base / "manifest.json"), input=_identity(input_path),
                      reused_arm=_identity(saved_path), physics=physics, core=core, geometry=geo, field=field,
                      diagnostic=_identity(__file__), seed=a.seed, chunk=200, min_bucket=256,
                      numpy=np.__version__, jax=jax.__version__, device=str(jax.devices()[0]),
                      scope="Fixed-entry mean-loss diagnostic; same-law range is not independent physics")
    (out / "manifest.json").write_text(json.dumps(provenance, indent=2))
    fm = fieldmap.to_device(fieldmap.load(f"{cli.MAPS}/{cli.MAP}.header"))
    key = jax.random.fold_in(jax.random.PRNGKey(a.seed), 0)
    state = transport.initial_state(jax.random.fold_in(key, 1), np.column_stack([g[k] for k in ("x", "y", "z")]),
                                    np.column_stack([g[k] for k in ("px", "py", "pz")]), g["time"], g["q"], prm)
    result, steps, history = capture_transport(fm, geom, prm, jax.random.fold_in(key, 2), state)
    matches = verify_replay(result, saved)
    np.savez_compressed(out / "layer_history.npz", **history)
    report, frozen = frozen_study(history, g, geom)
    np.savez_compressed(out / "frozen_first_foil.npz", **frozen)
    report.update(complete=True, replay_exact=matches, muons=n, layer_records=len(history["row"]),
                  lane_steps=steps, capped=int(np.sum(result.fate == 9)), wall_s=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != "selected_rows"}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
