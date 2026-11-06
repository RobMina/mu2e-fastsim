"""Primary-brems uncollided flux at the emitting foil exit; not a shower model.

The optional device source carries e-/e+ primaries in the CE navigator geometry.
Postprocessing uses exact straight photon rays in finite annuli and tabulated Al
attenuation. Absorbed/scattered photons and their descendants are not transported.
All source times are retained; the production timing cut belongs to detector steps.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .flash_containment import containing_foil
from .flash_exit import foil_exit, stock_time_selection

ENERGY_EDGES = np.geomspace(0.001, 100.0, 41)
SOURCE_FIELDS = ("run", "subrun", "event", "particle_key", "pdg", "x", "p", "time_ns")
C_MM_NS = 299.792458


def read_states(path):
    """Read the existing flat DS states, keeping times at their saved precision."""
    with np.load(path, allow_pickle=False) as archive:
        source = {k: archive[k] for k in SOURCE_FIELDS}
    n = len(source["pdg"])
    for k, value in source.items():
        shape = (n, 3) if k in ("x", "p") else (n,)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid source field {k}")
    if not np.isin(source["pdg"], (11, -11)).all():
        raise ValueError("Source must contain only e-/e+; split other species explicitly")
    for k in ("run", "subrun", "event", "particle_key", "pdg"):
        if source[k].dtype.kind not in "iu":
            raise ValueError(f"Source identity {k} must be integer")
    return source


def exit_records(payload, beam):
    """Flat brems births, first finite-foil exit, and uncollided weights.

    Non-foil births remain represented in the count summary, not moved to a foil.
    A planar charged trajectory can leave its annulus before emitting; such births
    cannot supply a finite-foil flux and are explicitly counted as outside foils.
    """
    from .flash_attenuation import uncollided_weight

    valid = np.asarray(payload["photon_valid"], bool)
    if valid.ndim != 2:
        raise ValueError("photon_valid must have shape (parents, capacity)")
    mask = valid.copy()
    if "photon_mechanism" in payload:
        mask &= payload["photon_mechanism"] == 1
    parent, slot = np.nonzero(mask)
    xyz = np.asarray(payload["photon_position"])[parent, slot]
    direction = np.asarray(payload["photon_direction"])[parent, slot]
    energy = np.asarray(payload["photon_energy"])[parent, slot]
    relative_time = np.asarray(payload["photon_time"])[parent, slot]
    if not (np.isfinite(xyz).all() and np.isfinite(direction).all() and
            np.isfinite(energy).all() and (energy > 0).all() and np.isfinite(relative_time).all()):
        raise ValueError("Nonfinite or nonpositive valid photon payload")
    z, ht = np.asarray(beam.foil_z), np.asarray(beam.foil_ht)
    containing = containing_foil(beam, xyz)
    k = np.maximum(containing, 0)
    result = foil_exit(xyz, direction, z[k], ht[k], beam.foil_rin[k], beam.foil_rout[k], beam.axis_x)
    foil = (containing >= 0) & result["valid"]
    distance = result["distance"]
    attenuation = uncollided_weight(energy, distance)
    weight, supported = attenuation["weight"], attenuation["supported"]
    use = foil & supported
    exit_dt = relative_time + np.where(foil, distance, 0.0) / C_MM_NS
    source_time = np.asarray(payload["source_time_ns"], dtype=np.float64)[parent]
    absolute_time = source_time + exit_dt
    return dict(parent=parent, energy=energy, birth_position=xyz, direction=direction,
                foil_index=np.where(foil, k, -1), foil_exit=foil, exit_position=result["position"],
                path_in_foil_mm=distance, exit_relative_time_ns=exit_dt, source_time_ns=source_time,
                exit_time_ns=absolute_time, face=result["face"], attenuation_supported=use,
                weight=np.where(use, weight, 0.0),
                time_predicate_diagnostic=foil & stock_time_selection(absolute_time))


def _identity_rows(columns):
    """Exact nonnegative integer rows, never NumPy's signed/uint float promotion."""
    values = [np.asarray(v) for v in columns]
    if any(v.ndim != 1 or v.dtype.kind not in "iu" or np.any(v < 0) for v in values):
        raise ValueError("Source identities must be nonnegative integer columns")
    if any(v.shape != values[0].shape for v in values):
        raise ValueError("Source identity columns must have equal lengths")
    return np.column_stack([v.astype(np.uint64, copy=False) for v in values])


def _groups(payload, fields):
    rows = _identity_rows([payload["source_" + k] for k in fields])
    return np.unique(rows, axis=0, return_inverse=True)


def _validated_ledger(ledger):
    ledger = np.asarray(ledger)
    if ledger.ndim != 2 or ledger.shape[1] != 3:
        raise ValueError("Event ledger must have shape (events, 3)")
    ledger = _identity_rows(ledger.T)
    unique = np.unique(ledger, axis=0)
    if len(unique) != len(ledger):
        raise ValueError("Duplicate source event identity in event ledger")
    return unique


def _event_groups(payload, source_events, event_ledger):
    event, event_id = _groups(payload, ("run", "subrun", "event"))
    represented = len(event)
    if (isinstance(source_events, (bool, np.bool_)) or not isinstance(source_events, (int, np.integer))
            or source_events < max(1, represented)):
        raise ValueError("source_events must be a positive integer including all represented source events")
    saved = payload.get("source_event_ledger")
    if event_ledger is not None:
        event_ledger = _validated_ledger(event_ledger)
    if saved is not None:
        saved = _validated_ledger(saved)
        if event_ledger is not None and not np.array_equal(saved, event_ledger):
            raise ValueError("Explicit event ledger disagrees with saved source event ledger")
        event_ledger = saved
    if event_ledger is not None:
        if len(event_ledger) != source_events:
            raise ValueError("Event ledger size must equal source_events; do not change the denominator")
        # The exact-integer union maps source rows to the sorted complete ledger.
        complete, inverse = np.unique(np.concatenate((event_ledger, event)), axis=0, return_inverse=True)
        if len(complete) != len(event_ledger):
            raise ValueError("Source event absent from event ledger")
        event_id = inverse[len(event_ledger):][event_id]
        event = complete
    return event, event_id, represented, event_ledger is not None


def summarize(payload, photons, *, source_events, event_ledger=None, energy_edges=ENERGY_EDGES):
    """Keep the scalar denominator; zero-fill only identities in a verified ledger.

    ``event_ledger`` is the extractor's complete integer (run, subrun, event)
    array, including events without e-/e+ source rows. Saved births may embed it
    as ``source_event_ledger``. Without a ledger, only represented events are
    emitted, so matched comparisons still reject any unverified missing inputs.
    """
    n = len(payload["source_pdg"])
    event, event_id, represented, complete_ledger = _event_groups(payload, source_events, event_ledger)
    if event_id.shape != (n,):
        raise ValueError("Source identity columns must align with source parents")
    shard, event_shard = np.unique(event[:, :2], axis=0, return_inverse=True)
    shard_id = event_shard[event_id]
    p = photons["parent"]
    foil, supported = photons["foil_exit"], photons["attenuation_supported"]
    energy, weight = photons["energy"], photons["weight"]
    count_parent = np.bincount(p[foil], minlength=n).astype(float)
    flux_parent = np.bincount(p, weights=weight, minlength=n)
    energy_parent = np.bincount(p, weights=weight * energy, minlength=n)
    raw_energy_parent = np.bincount(p[foil], weights=energy[foil], minlength=n)
    totals = dict(transparent_photons=count_parent, uncollided_photons=flux_parent,
                  transparent_energy_mev=raw_energy_parent, uncollided_energy_mev=energy_parent)
    by_shard = {k: np.bincount(shard_id, weights=v, minlength=len(shard)).tolist() for k, v in totals.items()}
    hist = {}
    for name, w in (("transparent", foil.astype(float)), ("uncollided", weight)):
        hist[name] = np.histogram(energy, bins=energy_edges, weights=w)[0].tolist()
    report = dict(parents=n, source_events=int(source_events), represented_source_events=represented,
                  complete_event_ledger=complete_ledger,
                  zero_source_events=len(event) - represented if complete_ledger else None,
                  source_species={str(k): int(np.count_nonzero(payload["source_pdg"] == k)) for k in (11, -11)},
                  all_recorded_photons=int(np.count_nonzero(payload["photon_valid"])),
                  recorded_brems=len(p), finite_foil_births=int(foil.sum()),
                  nonfoil_brems=int((~foil).sum()), nonfoil_brems_energy_mev=float(energy[~foil].sum()),
                  unsupported_attenuation_photons=int((foil & ~supported).sum()),
                  unsupported_attenuation_energy_mev=float(energy[foil & ~supported].sum()),
                  dropped_births=int(np.sum(payload["overflow"])),
                  capped_parents=int(np.count_nonzero(payload["capped"])),
                  loopers=int(np.count_nonzero(payload["loopers"])),
                  transport_complete=not any(np.any(payload[k]) for k in ("overflow", "capped", "loopers")),
                  source_observable="Path-truncated diagnostic if any loopers; partial emission remains in all-event denominator",
                  totals={k: float(v.sum()) for k, v in totals.items()},
                  per_source_event={k: float(v.sum() / source_events) for k, v in totals.items()},
                  source_shards=shard.tolist(), by_shard=by_shard,
                  energy_edges_mev=np.asarray(energy_edges).tolist(), histograms=hist,
                  source_time_max_ns=float(np.max(payload["source_time_ns"])) if n else None,
                  source_time_rows_above_period=int(np.count_nonzero(payload["source_time_ns"] > 1695.)),
                  timing="No source cut. fmod predicate at foil exit is diagnostic, not detector acceptance.",
                  exit_time_predicate_diagnostic_photons=int(photons["time_predicate_diagnostic"].sum()),
                  scope="Primary brems, planar charged layers, uncollided original photons at emitting foil exit; no shower/secondary/annihilation flux or downstream response")
    for key in ("nan", "ean", "nde", "ede"):
        if "state_" + key in payload:
            report["omitted_" + key] = float(np.sum(payload["state_" + key]))
    if "state_T" in payload:
        unfinished = np.asarray(payload["capped"]) | np.asarray(payload["loopers"])
        report["unfinished_kinetic_energy_mev"] = float(np.asarray(payload["state_T"])[unfinished].sum())
        report["unfinished_energy_scope"] = "Upper bound on future primary brems energy only; not missing cascades or annihilation energy"
    event_totals = {k: np.bincount(event_id, weights=v, minlength=len(event)) for k, v in totals.items()}
    return report, dict(event=event, source_parents=np.bincount(event_id, minlength=len(event)), **event_totals)


def run(args):
    from .flash_geometry import from_offline
    geom = from_offline(str(args.geom), str(args.offline_root))
    ledger_path = args.event_ledger
    if ledger_path is None and args.states is not None:
        sidecar = args.states.with_name(args.states.stem + "_events.npz")
        if sidecar.exists():
            ledger_path = sidecar
    event_ledger = None
    if ledger_path is not None:
        with np.load(ledger_path, allow_pickle=False) as archive:
            event_ledger = _validated_ledger(archive["event"])
    args.out.mkdir(parents=True, exist_ok=False)
    elapsed = None
    if args.births is not None:
        with np.load(args.births, allow_pickle=False) as archive:
            payload = {k: archive[k] for k in archive.files}
    else:
        import jax
        from . import cechain_jax as CJ
        from .flash_target_jax import target_source
        source = read_states(args.states)
        payload = {"source_" + k: v for k, v in source.items()}
        _event_groups(payload, args.source_events, event_ledger)  # fail before charged transport
        if event_ledger is not None:
            payload["source_event_ledger"] = event_ledger
        maps = CJ.device_maps([str(p) for p in args.maps], dtype=np.float64)
        p = np.linalg.norm(source["p"], axis=1)
        if np.any(p <= 0):
            raise ValueError("Zero source momentum")
        start = time.perf_counter()
        result = target_source(jax.random.PRNGKey(args.seed), geom, maps, source["x"], source["p"] / p[:, None],
                               p, source["time_ns"], source["pdg"], capacity=args.capacity,
                               max_steps=args.max_steps, h=args.h, s_max=args.s_max, dtype=np.float64,
                               unroll=args.unroll, cpu_vacuum=args.cpu_vacuum, cpu_adaptive=args.cpu_adaptive)
        result = jax.tree_util.tree_map(np.asarray, result)
        elapsed = time.perf_counter() - start
        payload.update({"state_" + k: v for k, v in result["state"].items()})
        payload.update({"photon_" + k: v for k, v in result["records"].items()})
        payload.update({k: result[k] for k in ("overflow", "capped", "loopers")})
        np.savez_compressed(args.out / "births.npz", **payload)
    photons = exit_records(payload, geom.beam)
    report, event_totals = summarize(payload, photons, source_events=args.source_events, event_ledger=event_ledger)
    np.savez_compressed(args.out / "foil_exit.npz", **photons)
    np.savez_compressed(args.out / "event_totals.npz", **event_totals)
    report.update(source=str(args.states if args.births is None else args.births), geometry=str(args.geom),
                  offline_root=str(args.offline_root), cold_transport_seconds=elapsed,
                  event_ledger=str(ledger_path) if ledger_path is not None else
                  "embedded in births" if "source_event_ledger" in payload else None)
    if args.births is None:
        report.update(maps=list(map(str, args.maps)), seed=args.seed, h_mm=args.h,
                      max_steps=args.max_steps, capacity=args.capacity, s_max_mm=args.s_max,
                      unroll=args.unroll, cpu_vacuum=args.cpu_vacuum, cpu_adaptive=args.cpu_adaptive,
                      iterations=result["diagnostics"]["iterations"].tolist(),
                      backend=jax.default_backend(), devices=list(map(str, jax.devices())))
    else:
        report["transport_parameters"] = "Saved source; CLI transport settings were not applied"
    (args.out / "flux.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 2 if report["dropped_births"] or report["capped_parents"] else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--states", type=Path)
    source.add_argument("--births", type=Path, help="Postprocess saved births without rerunning charged transport")
    p.add_argument("--geom", type=Path, required=True)
    p.add_argument("--offline-root", type=Path, required=True)
    p.add_argument("--maps", type=Path, nargs="+")
    p.add_argument("--source-events", type=int, required=True)
    p.add_argument("--event-ledger", type=Path,
                   help="Complete extractor *_events.npz; defaults to --states sidecar or ledger embedded in --births")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=500000)
    p.add_argument("--h", type=float, default=1.)
    p.add_argument("--s-max", type=float, default=30000.)
    p.add_argument("--unroll", action="store_true")
    p.add_argument("--cpu-vacuum", type=int, default=0, help="Opt-in CPU bucket for sparse vacuum tails")
    p.add_argument("--cpu-adaptive", action="store_true", help="Shrink CPU vacuum buckets as parents finish")
    args = p.parse_args(argv)
    if args.states is not None and not args.maps:
        p.error("--states requires --maps")
    if args.source_events < 1:
        p.error("--source-events must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
