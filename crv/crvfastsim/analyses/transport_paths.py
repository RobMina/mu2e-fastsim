"""Read-only, truth-conditioned path decomposition of persisted primary-muon crossings.

Reference-direction strata are diagnostics, never correction models. No transport is run.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from crvfastsim import geometry, runconfig, tables, transport
from crvfastsim.analyses import transport_validation as V
from mu2efastsim import eloss

KEY = tables.KEY + ["genidx", "bar"]
XYZ0, XYZ1, PXYZ = ["x0", "y0", "z0"], ["x1", "y1", "z1"], ["px0", "py0", "pz0"]
TOL_MM = .01
INCIDENCE = [0., .2, .5, .8, 1.]
SCHEMA = "crv-transport-paths-v1"


def ray_chord(entry, direction, center, half, tolerance=TOL_MM):
    """Remaining entry-ray distance to box exit; NaN for invalid/outside/zero rays.

    Only coordinates outside by <= tolerance are snapped to the boundary. Parallel
    axes impose no exit distance. Directions need not be normalized.
    """
    q, d = np.asarray(entry, float) - center, np.asarray(direction, float)
    norm = np.linalg.norm(d, axis=1)
    u = np.divide(d, norm[:, None], out=np.zeros_like(d), where=norm[:, None] > 0)
    valid = np.isfinite(q).all(1) & np.isfinite(d).all(1) & (norm > 0)
    valid &= (np.abs(q) <= half + tolerance).all(1)
    q = np.clip(q, -half, half)
    distance = np.divide(np.where(u >= 0, half, -half) - q, u,
                         out=np.full_like(q, np.inf), where=np.abs(u) > 1e-14).min(1)
    return np.where(valid & (distance > tolerance), distance, np.nan)


def classify_faces(entry, exit, center, half, thickness_axis, tolerance=TOL_MM):
    """Unique opposite thickness faces -> through; other unique faces -> edge/clipped.

    Interior, outside and corner/edge-tied endpoints are ambiguous, including stops
    ending inside. Stop flags remain independent: fast stopped endpoints can be through.
    """
    q0, q1 = np.asarray(entry) - center, np.asarray(exit) - center
    f0, f1 = np.abs(np.abs(q0) - half) <= tolerance, np.abs(np.abs(q1) - half) <= tolerance
    unique = (f0.sum(1) == 1) & (f1.sum(1) == 1)
    unique &= (np.abs(q0) <= half + tolerance).all(1) & (np.abs(q1) <= half + tolerance).all(1)
    axis = np.asarray(thickness_axis, int)[:, None]
    through = unique & (f0.argmax(1) == axis[:, 0]) & (f1.argmax(1) == axis[:, 0])
    through &= (np.take_along_axis(q0, axis, 1) * np.take_along_axis(q1, axis, 1))[:, 0] < 0
    return np.where(through, "thickness_through", np.where(unique, "edge_clipped", "ambiguous"))


def path_crossings(path, gen, geom, fast_format=False):
    """V identity-safe totals plus earliest-entry/latest-exit geometry from raw records.

    nrecords==1 is a conservative single-entry proxy, not a claim that a CrvStep
    contains one Geant4 integration step. len - endpoint chord is summed per record.
    """
    x = V.primary_crossings(path, gen, geom, fast_format=fast_format)
    parts, steps = tables.read(path, "particles"), tables.read(path, "steps")
    prim = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == 13)]
    if "genidx" in prim:
        identity = prim[tables.KEY + ["key", "genidx"]]
    else:  # V already rejects this fallback for reference-format inputs.
        gm = gen.loc[gen.pdg.abs() == 13].sort_values(tables.KEY + ["idx"], kind="stable").copy()
        gm["key"] = gm.groupby(tables.KEY, sort=False).cumcount() + 1
        identity = prim[tables.KEY + ["key"]].merge(
            gm[tables.KEY + ["key", "idx"]].rename(columns={"idx": "genidx"}),
            on=tables.KEY + ["key"], validate="one_to_one")
    s = steps.merge(identity.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], validate="many_to_one")
    finite = np.isfinite(s[XYZ0 + XYZ1 + PXYZ + ["p1", "len", "t0"]])
    if not finite.all().all():
        raise ValueError(f"nonfinite raw primary path data: {(~finite).sum().to_dict()}")
    if (s[["len", "p1"]] < 0).any().any():
        raise ValueError("negative path/momentum")
    s["invalid_time"] = ~np.isfinite(s.t1) | (s.t1 < s.t0)
    s["record_chord"] = np.linalg.norm(s[XYZ1].to_numpy() - s[XYZ0].to_numpy(), axis=1)
    s["record_excess"] = s.len - s.record_chord
    first = s.sort_values(KEY + ["t0"], kind="stable").drop_duplicates(KEY)[KEY + XYZ0 + PXYZ]
    last = s.sort_values(KEY + ["t0"], kind="stable").drop_duplicates(KEY, keep="last")[KEY + XYZ1 + ["p1"]]
    totals = s.groupby(KEY, sort=False).agg(nrecords=("len", "size"), record_chord=("record_chord", "sum"),
                                           record_excess=("record_excess", "sum"), invalid_time=("invalid_time", "any")).reset_index()
    x = x.merge(first, on=KEY, validate="one_to_one").merge(last, on=KEY, validate="one_to_one")
    x = x.merge(totals, on=KEY, validate="one_to_one")
    bars = x.bar.to_numpy(dtype=int)
    if ((bars < 0) | (bars >= len(geom.sector))).any():
        raise ValueError("bar index outside geometry")
    center, half, axis = geom.center[bars], geom.halflen[bars], geom.local_to_world[bars, 0]
    entry, exit, direction = x[XYZ0].to_numpy(), x[XYZ1].to_numpy(), x[PXYZ].to_numpy()
    x["ray_chord"] = ray_chord(entry, direction, center, half)
    x["endpoint_chord"] = np.linalg.norm(exit - entry, axis=1)
    x["face_class"] = classify_faces(entry, exit, center, half, axis)
    x["incidence"] = np.divide(np.abs(np.take_along_axis(direction, axis[:, None], 1)[:, 0]),
                               x.p, out=np.full(len(x), np.nan), where=x.p.to_numpy() > 0)
    x["stopped"] = x.p1 <= 1.
    x["shortened"] = x.length < x.endpoint_chord - TOL_MM
    x["clean"] = (~x.stopped & ~x.shortened & ~x.invalid_time & (x.nrecords == 1) & (x.face_class != "ambiguous")
                  & np.isfinite(x.ray_chord))
    return x


def _compare(a, b, universe, a_length="length", b_length="length", length_only=False):
    aa, bb = a.copy(), b.copy()
    aa["length"], bb["length"] = a[a_length], b[b_length]
    out = V.compare_measures(aa, bb, universe)
    if length_only:
        out["paired_measure_moments"] = {"length_mean": out["paired_measure_moments"]["length_mean"]}
        for key in list(out):
            if key.startswith(("edep_mean_", "energy_per_path_")):
                del out[key]
    return out


def decomposition(a, b, universe):
    """A=candidate, B=G4, matched identities only. Ratio fields are ratio-minus-one."""
    matched = a[KEY].merge(b[KEY], on=KEY, validate="one_to_one")
    a = matched.merge(a, on=KEY, validate="one_to_one")
    b = matched.merge(b, on=KEY, validate="one_to_one")
    valid = np.isfinite(a.ray_chord) & np.isfinite(b.ray_chord)
    aa, bb = a.loc[valid], b.loc[valid]
    flags = {arm: {**{label: int((f.face_class == label).sum()) for label in
                     ("thickness_through", "edge_clipped", "ambiguous")},
                   **{label: int(f[label].sum()) for label in ("stopped", "shortened", "invalid_time", "clean")},
                   "multiple_records": int((f.nrecords > 1).sum()), "records": int(f.nrecords.sum()),
                   "record_excess_mm": float(f.record_excess.sum())}
             for arm, f in (("candidate", a), ("reference", b))}
    return {"counts": flags, "transport": _compare(a, b, universe),
            "valid_ray_pairs": int(valid.sum()),
            "entry_ray_candidate_vs_reference": _compare(aa, bb, universe, "ray_chord", "ray_chord", True),
            "candidate_true_vs_entry_ray": _compare(aa, aa, universe, "length", "ray_chord", True),
            "reference_true_vs_entry_ray": _compare(bb, bb, universe, "length", "ray_chord", True),
            "reference_true_vs_record_chords": _compare(b, b, universe, "length", "record_chord", True)}


def populations(a, b, universe):
    joined = a[KEY].merge(b[KEY], on=KEY, validate="one_to_one")
    a, b = joined.merge(a, on=KEY), joined.merge(b, on=KEY)
    clean = a.clean & b.clean
    masks = {"all_matched": np.ones(len(a), bool), "both_nonstopping_clean_single_record": clean,
             "clean_both_thickness_through": clean & (a.face_class == "thickness_through") &
                                             (b.face_class == "thickness_through"),
             "clean_edge_in_either": clean & ((a.face_class == "edge_clipped") | (b.face_class == "edge_clipped"))}
    for i, (lo, hi) in enumerate(zip(INCIDENCE[:-1], INCIDENCE[1:])):
        masks[f"clean_reference_incidence_{i}"] = clean & (b.incidence >= lo) & ((b.incidence <= hi) if hi == 1 else (b.incidence < hi))
    for i, (lo, hi) in enumerate(zip(V.P_EDGES[:-1], V.P_EDGES[1:])):
        masks[f"clean_reference_p_{i}"] = clean & (b.p >= lo) & (b.p < hi)
    return {name: decomposition(a.loc[mask], b.loc[mask], universe) for name, mask in masks.items()}


def _write(out, output):
    text = json.dumps(out, indent=2, allow_nan=False) + "\n"
    with Path(output).open("x") as stream:
        stream.write(text)
    return out


def run(gen_path, reference, candidate, config, expected_events, output, gdml=None):
    if Path(output).exists():
        raise FileExistsError(output)
    rc = runconfig.get(config)
    geom, gen = rc.geometry(), tables.read(gen_path, "gen")
    universe = gen[tables.KEY].drop_duplicates()
    if len(universe) != expected_events or gen.duplicated(tables.KEY + ["idx"]).any():
        raise ValueError("generated event count or generator identity mismatch")
    paths = {"gen": gen_path, "reference": reference, "candidate": candidate}
    for arm, path in (("reference", reference), ("candidate", candidate)):
        saved = tables.read(path, "gen")
        m = saved.merge(gen, on=tables.KEY + ["idx"], how="left", indicator=True,
                        suffixes=("_s", "_g"), validate="one_to_one")
        if (m._merge != "both").any() or (arm == "candidate" and len(saved) != len(gen)):
            raise ValueError(f"{arm}: incompatible source identities/coverage")
        for col in ("pdg", "px", "py", "pz", "x", "y", "z", "t"):
            if not np.allclose(m[col + "_s"], m[col + "_g"], rtol=1e-10, atol=1e-8):
                raise ValueError(f"{arm}: source mismatch in {col}")
    a, b = path_crossings(candidate, gen, geom, True), path_crossings(reference, gen, geom)
    groups = {name: [i] for i, name in enumerate(geom.sector_names)}
    groups["all_sectors"] = list(range(len(geom.sector_names)))
    cfg = rc.counters_cfg()
    for typ, name in runconfig.sector_type_names(geom, cfg).items():
        groups["group:" + name] = [i for i, s in enumerate(geom.sector_names) if int(cfg[f"crs.sectorType{s}"]) == typ]
    source_modules = [V, geometry, runconfig, tables, transport, transport.crossings,
                      eloss, transport.core_tr, transport.world_mod, transport.materials, tables._core]
    source_hashes = {m.__name__: V.fingerprint(m.__file__)["sha256"] for m in source_modules}
    source_hashes["crvfastsim.analyses.transport_paths"] = V.fingerprint(__file__)["sha256"]
    out = {"schema": SCHEMA, "run_config": config, "persisted_generated_events": len(universe),
           "event_ranges": universe.groupby(["run", "subrun"]).event.agg(["min", "max", "size"]).reset_index().to_dict("records"),
           "inputs": {k: V.audit(p) for k, p in paths.items()},
           "source_ids": [V.fingerprint(gen_path)["sha256"]],
           "source_sha256": source_hashes,
           "source_hash_scope": "Analysis-time sources, not proof of the revisions that produced saved tables.",
           "geometry": [V.fingerprint(geometry.find_counters(rc.counters))] + ([V.fingerprint(gdml)] if gdml else []),
           "settings": {"face_tolerance_mm": TOL_MM, "stop_p1_MeV": 1., "incidence_edges": INCIDENCE,
                        "reference_p_edges_MeV": [0, 300, 600, 1000, 2000, 5000, None]},
           "scope": "Saved-table truth diagnostic, no correction model or physics gate. Reference completeness external. "
                    "All-matched includes stops; geometrical attribution uses clean populations (both nonstopping, "
                    "single record, unique boundary faces, valid rays/times, no shortened endpoints). Last exit uses "
                    "the last start-time-ordered record; invalid end times are counted and excluded from clean. Strata use G4 entry "
                    "direction/p within that clean population. Ratio fields are relative shifts; errors cluster by generated event. "
                    "CrvStep records aggregate intra-record paths: record excess is not the whole scattering contribution.",
           "crossing_key": KEY,
           "input_path_census": {arm: {"crossings": len(frame), "records": int(frame.nrecords.sum()),
                                       **{flag: int(frame[flag].sum()) for flag in
                                          ("stopped", "shortened", "invalid_time", "clean")}}
                                 for arm, frame in (("candidate", a), ("reference", b))},
           "sectors": {name: populations(a[a.sector.isin(sectors)], b[b.sector.isin(sectors)], universe)
                       for name, sectors in groups.items()}}
    return _write(out, output)


def merge(paths, output):
    paths = tables.expand(paths)
    rows = [json.loads(Path(p).read_text()) for p in paths]
    if not rows or any(r.get("schema") != SCHEMA for r in rows):
        raise ValueError("merge requires transport-paths schema")
    first = rows[0]
    for r in rows:
        if any(r[k] != first[k] for k in ("run_config", "source_sha256", "settings", "crossing_key")) or [
                g["sha256"] for g in r["geometry"]] != [g["sha256"] for g in first["geometry"]]:
            raise ValueError("incompatible source, geometry or settings")
        if r["sectors"].keys() != first["sectors"].keys() or any(
                r["sectors"][s].keys() != first["sectors"][s].keys() for s in first["sectors"]):
            raise ValueError("incompatible sector/population coverage")
    sources = [s for r in rows for s in r["source_ids"]]
    ranges = [e for r in rows for e in r["event_ranges"]]
    if len(set(sources)) != len(sources) or len({(e["run"], e["subrun"]) for e in ranges}) != len(ranges):
        raise ValueError("merge requires disjoint generator sources and run/subrun IDs")
    out = {k: first[k] for k in ("schema", "run_config", "source_sha256", "settings", "geometry", "scope", "crossing_key", "source_hash_scope")}
    out.update(source_ids=sources, event_ranges=ranges, source_audits=[V.fingerprint(p) for p in paths],
               persisted_generated_events=sum(r["persisted_generated_events"] for r in rows), sectors={},
               input_path_census={arm: {k: sum(r["input_path_census"][arm][k] for r in rows) for k in census}
                                  for arm, census in first["input_path_census"].items()})
    for sector, pops in first["sectors"].items():
        out["sectors"][sector] = {}
        for pop, template in pops.items():
            items = [r["sectors"][sector][pop] for r in rows]
            merged = {"valid_ray_pairs": sum(r["valid_ray_pairs"] for r in items),
                      "counts": {arm: {k: sum(r["counts"][arm][k] for r in items) for k in counts}
                                 for arm, counts in template["counts"].items()}}
            merged.update({k: V.merge_counts([r[k] for r in items]) for k in template if k not in merged})
            out["sectors"][sector][pop] = merged
    return _write(out, output)


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    if "--merge" in argv:
        p.add_argument("--merge", nargs="+", required=True)
        args = p.parse_args(argv)
        merge(args.merge, args.output)
    else:
        for flag in ("gen", "reference", "candidate", "config"):
            p.add_argument("--" + flag, required=True)
        p.add_argument("--expected-events", type=int, required=True)
        p.add_argument("--gdml")
        args = p.parse_args(argv)
        run(args.gen, args.reference, args.candidate, args.config, args.expected_events, args.output, args.gdml)


if __name__ == "__main__":
    main()
