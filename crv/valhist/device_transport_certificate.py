"""Read-only, fail-closed certificate for persisted device-transport shards.

Run on the data host, not a login node. This never transports particles, starts
JAX, submits work, or changes the existing V1 comparator. For example::

    python -m valhist.device_transport_certificate --summaries RUN/summary.json \
        --inputs GEN --reference FASTC --stock G4 --counters COUNTERS --out NEW.json

Physics-port evidence is distinct from stock agreement: the former requires the
unchanged V1 rules against staged NumPy plus the existing stephist deck flags;
the latter requires unchanged V1 against stock for BOTH arms. Histograms retain
all extracted sector groups, flow, clustered independent and same-event errors.
Timing ranges describe observed repeats, not confidence intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables
from crvfastsim.analyses import v1compare as v1
from . import stephist as sh, steplayers as sl


STEP_COLUMNS = tables.KEY + ["nstep", "bar", "sp", "t0", "t1", "x0", "x1", "px0",
                             "y0", "y1", "py0", "z0", "z1", "pz0", "p1", "len", "edep"]
PART_COLUMNS = tables.KEY + ["key", "parent", "pdg", "isprim"]
GEN_COLUMNS = tables.KEY + ["idx", "pdg", "x", "y", "z", "px", "py", "pz", "t"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def numeric_frame(frame, columns, integer=()):
    require(set(columns).issubset(frame.columns), f"missing columns: {set(columns) - set(frame.columns)}")
    require(frame.columns.is_unique, "duplicate columns")
    for col in frame:
        require(pd.api.types.is_numeric_dtype(frame[col]), f"nonnumeric column {col}")
        require(bool(np.isfinite(frame[col].to_numpy()).all()), f"nonfinite column {col}")
    for col in integer:
        x = frame[col].to_numpy()
        require(bool(np.equal(x, np.floor(x)).all()), f"nonintegral identity {col}")
        require(pd.api.types.is_integer_dtype(frame[col]), f"noninteger schema {col}")


def exact_frame(a, b, keys):
    require(set(a.columns) == set(b.columns), "payload column mismatch")
    pd.testing.assert_frame_equal(a.sort_values(keys).reset_index(drop=True),
                                  b[a.columns].sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, check_exact=True)


def subset_ids(a, b, keys):
    return bool(pd.MultiIndex.from_frame(a[keys]).isin(pd.MultiIndex.from_frame(b[keys])).all())


def validate_tables(steps, parts, events, nbars, expected=None):
    numeric_frame(steps, STEP_COLUMNS, tables.KEY + ["bar", "sp", "nstep"])
    numeric_frame(parts, PART_COLUMNS, tables.KEY + ["key", "parent", "pdg", "isprim"])
    require(not parts.duplicated(tables.KEY + ["key"]).any(), "duplicate particle identity")
    require(subset_ids(parts, events, tables.KEY), "particles outside generator population")
    require(subset_ids(steps, parts.rename(columns={"key": "sp"}), tables.KEY + ["sp"]),
            "step has no saved particle")
    require(bool(steps.bar.between(0, nbars - 1).all()), "bar outside geometry")
    require(bool((steps[["len", "edep", "p1"]] >= 0).all().all()), "negative length/energy/momentum")
    require(bool((steps.nstep >= 1).all()), "invalid nstep")
    require(bool((steps.t1 >= steps.t0).all()), "reversed step times")
    if expected is not None:
        exact_frame(parts[PART_COLUMNS], expected, tables.KEY + ["key"])
        require(not steps.duplicated(tables.KEY + ["sp", "bar"]).any(), "duplicate transport crossing")
        counts = steps.groupby(tables.KEY + ["sp"], sort=False).bar.transform("size")
        require(bool(np.array_equal(steps.nstep.to_numpy(), counts.to_numpy())), "nstep multiplicity mismatch")
    return {"valid": True, "steps": len(steps), "particles": len(parts),
            "events_with_steps": len(steps[tables.KEY].drop_duplicates()),
            "stopped_steps_p1_zero": int(steps.p1.eq(0).sum()),
            "stopped_particles_p1_zero": len(steps.loc[steps.p1.eq(0), tables.KEY + ["sp"]].drop_duplicates())}


def fingerprint_frames(*frames):
    """Same persisted-table binding as device_transport, without importing JAX."""
    h = hashlib.sha256()
    for frame in frames:
        h.update(str(list(frame.columns)).encode())
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return h.hexdigest()


def receipts(row, staged):
    errors = []
    r = row.get("baseline_receipt", {})
    if not isinstance(r, dict):
        r = {}
    delta = r.get("max_abs", {})
    if (r.get("same_schema") is not True or r.get("rows") != len(staged)
            or not isinstance(delta, dict) or set(delta) != set(staged.columns)
            or not all(finite_number(x) and x == 0 for x in delta.values())):
        errors.append("incomplete_or_nonexact_baseline_receipt")
    d = row.get("deterministic", {})
    if not isinstance(d, dict):
        d = {}
    counts = [d.get("count_numpy"), d.get("count_jax")]
    if (d.get("exact_track_bar") is not True
            or not finite_number(d.get("max_distance_delta_mm")) or d.get("max_distance_delta_mm") != 0
            or not all(finite_number(x) and x >= 0 and int(x) == x for x in counts)
            or counts[0] != counts[1]):
        errors.append("incomplete_or_nonexact_deterministic_crossings")
    repeat = "certified" if row.get("repeat_exact") is True else (
        "uncertified" if "repeat_exact" not in row else "failed")
    if repeat != "certified":
        errors.append("repeat_exact_" + repeat)
    fingerprints = row.get("repeat_fingerprints")
    if (not isinstance(fingerprints, list) or len(fingerprints) != len(row.get("trials", []))
            or not fingerprints or not all(isinstance(x, str) and len(x) == 64 for x in fingerprints)
            or len(set(fingerprints)) != 1):
        errors.append("incomplete_or_nonidentical_repeat_fingerprints")
    return {"pass": not errors, "repeat_status": repeat, "errors": errors,
            "baseline_receipt": r, "deterministic": d}


def describe(values):
    return {"n": len(values), "min": min(values), "max": max(values), "median": statistics.median(values)}


def timing(row, summary):
    result = {"pass": False, "errors": [], "warm_policy": "exclude first call for BOTH arms; >=3 warm repeats per arm"}
    try:
        require("synchron" in str(summary.get("timing_note", "")).lower(), "synchronized timing receipt absent")
        nt = row["numpy_times_s"]
        jt = [trial["total_s"] for trial in row["trials"]]
        require(isinstance(nt, list), "numpy_times_s is not a list")
        require(all(finite_number(x) and x > 0 for x in nt + jt), "invalid wall time")
        result.update(numpy_all_s=nt, jax_all_s=jt, numpy_warm_s=nt[1:], jax_warm_s=jt[1:])
        require(len(nt[1:]) >= 3 and len(jt[1:]) >= 3, "fewer than three warm repeats")
        n, j = describe(nt[1:]), describe(jt[1:])
        speedup = n["median"] / j["median"]
        result.update(numpy_warm=n, jax_warm=j, median_speedup=speedup, **{"pass": speedup > 1})
        if speedup <= 1:
            result["errors"].append("median_speedup_not_greater_than_one")
    except (KeyError, TypeError, ValueError) as exc:
        result["errors"].append(str(exc))
    return result


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args):
    out = Path(args.out).expanduser().resolve()
    # Never clobber an input or an earlier certificate.
    if out.exists():
        raise FileExistsError(out)
    result = {"schema_version": 1, "physics_port_pass": False, "stock_V1_pass": False,
              "throughput_pass": False, "ready_for_handoff": False, "complete": False,
              "errors": [], "shards": {}, "provenance": {"files": {}, "code_sha256": {}},
              "policy": {"physics_port": "complete exact receipts, valid saved tables, combined unchanged V1 vs staged NumPy, and no stephist deck flags",
                         "stock_V1": "combined unchanged V1 vs stock must pass for NumPy AND candidate",
                         "throughput": ">=3 warm repeats per arm per shard, excluding first for both, median speedup >1 in EVERY shard",
                         "uncertainty": "stephist event-clustered independent and same-event paired errors; timing ranges are observations only",
                         "scope": "persisted extracted-geometry transport only, not named-world or response/reco certification",
                         "provenance": "audit-time source hashes do not prove historical producer revision",
                         "stock_completeness": "accepted-event production completeness remains external evidence"}}
    files = result["provenance"]["files"]
    seen_paths, seen_objects = set(), set()

    def track(path):
        path = Path(path).expanduser().resolve()
        require(path.is_file(), f"missing input {path}")
        if str(path) not in files:
            stat = path.stat()
            files[str(path)] = {"sha256": digest(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return path

    try:
        import crvfastsim
        import mu2efastsim
        code_paths = [Path(__file__), Path(sh.__file__), Path(sl.__file__), Path(v1.__file__),
                      Path(__file__).with_name("device_transport.py")]
        for package, names in ((crvfastsim, ["geometry.py", "tables.py", "transport.py", "transport_jax.py",
                                           "crossings.py", "crossings_jax.py", "eloss_transport_jax.py", "analyses/census.py"]),
                               (mu2efastsim, ["tables.py", "transport.py", "crossings.py", "eloss.py", "materials.py", "world.py"])):
            code_paths.extend(Path(package.__file__).parent / name for name in names)
        for path in code_paths:
            if path.is_file():
                result["provenance"]["code_sha256"][str(path.resolve())] = digest(path)
        counters = track(args.counters)
        geom = geometry.from_offline_config(str(counters))
        # stephist/classify's groups and event packing are the extracted contract.
        require(len(geom.sector_names) == 11 and set(np.unique(geom.sector)) == set(range(11)),
                "certificate requires the extracted 11-sector geometry")
        v1.set_groups()
        sh._GEOM.update(geom=geom, grp=sl.bar_table(geom).grp.to_numpy())
        edges = {"y": sl.Y_EDGES, "z": sl.Z_EDGES}
        result["booking"] = {"stephist": sh.BOOK, "position_edges": edges,
                             "sector_groups": sl.POS_GROUPS, "deck_flag_threshold": sh.FLAG,
                             "V1_groups": v1.GROUPS, "V1_thresholds": v1.THRESH}
        discovered, seen_shards = [], set()
        for name in args.summaries:
            summary_path = track(name)
            require(summary_path not in seen_paths, f"duplicate summary {summary_path}")
            seen_paths.add(summary_path)
            with summary_path.open() as stream:
                summary = json.load(stream, object_pairs_hook=unique_object)
            require(isinstance(summary.get("shards"), dict) and bool(summary["shards"]), "missing shard inventory")
            inventory = {}
            for key, row in summary["shards"].items():
                shard = int(key)
                require(shard >= 0 and shard not in seen_shards, f"duplicate/invalid shard {key}")
                require(isinstance(row, dict), "invalid shard receipt")
                seen_shards.add(shard)
                suffix = f"001400_{shard:08d}.root"
                candidate = summary_path.parent / f"fastC_jax.{suffix}"
                inventory[candidate.name] = shard
                discovered.append((shard, row, summary, summary_path, candidate, suffix))
            actual = {p.name for p in summary_path.parent.glob("fastC_jax.*.root")}
            require(actual == set(inventory), f"candidate files and summary shard inventory differ in {summary_path.parent}")
        require(bool(discovered), "no shards discovered")
        ah, bh, ph = {}, {}, {}
        np_stock, jax_stock, port = v1._new_state(), v1._new_state(), v1._new_state()
        total_stops = {arm: {"stopped_steps_p1_zero": 0, "stopped_particles_p1_zero": 0}
                       for arm in ("numpy", "candidate", "stock")}
        for shard, row, summary, summary_path, candidate, suffix in sorted(discovered, key=lambda x: x[0]):
            entry = {"summary": str(summary_path), "errors": [], "valid": False}
            result["shards"][str(shard)] = entry
            try:
                paths = {"gen": track(Path(args.inputs) / f"gen.{suffix}"),
                         "numpy": track(Path(args.reference) / f"fastC.{suffix}"),
                         "candidate": track(candidate), "stock": track(Path(args.stock) / f"g4all.{suffix}")}
                entry["files"] = {k: str(p) for k, p in paths.items()}
                for path in paths.values():
                    stat = path.stat()
                    identity = (stat.st_dev, stat.st_ino)
                    require(identity not in seen_objects, f"input file reused {path}")
                    seen_objects.add(identity)
                gen = tables.read(str(paths["gen"]), "gen")
                numeric_frame(gen, GEN_COLUMNS, tables.KEY + ["idx", "pdg"])
                require(not gen.duplicated(tables.KEY + ["idx"]).any(), "duplicate generator particle")
                events = gen[tables.KEY].drop_duplicates()
                require(len(events) > 0 and events.run.eq(1400).all() and events.subrun.eq(shard).all(),
                        "empty or mislabelled generator population")
                sl.evkey(events)  # enforce the same event-packing bounds as stephist
                mu = gen.loc[gen.pdg.abs().eq(13)].sort_values(tables.KEY + ["idx"], kind="stable")
                expected = mu[tables.KEY].copy()
                expected["key"] = mu.groupby(tables.KEY, sort=False).cumcount().to_numpy() + 1
                expected["parent"], expected["pdg"], expected["isprim"] = -1, mu.pdg.to_numpy(), 1
                entry.update(generated_events=len(events), primary_muons=len(expected), table_validity={})
                loaded = {}
                for arm in ("numpy", "candidate", "stock"):
                    steps = tables.read(str(paths[arm]), "steps")
                    parts = tables.read(str(paths[arm]), "particles")
                    if arm == "stock":
                        required = tables.KEY + ["bar", "sp", "t0", "len", "edep"]
                        entry["stock_nonfinite_columns"] = {c:int((~np.isfinite(steps[c].to_numpy())).sum()) for c in steps}
                        numeric_frame(steps[required], required, tables.KEY + ["bar", "sp"])
                        numeric_frame(parts, PART_COLUMNS, tables.KEY + ["key", "parent", "pdg", "isprim"])
                        require(subset_ids(steps, parts.rename(columns={"key":"sp"}), tables.KEY+["sp"]), "stock step identity absent")
                        require(subset_ids(parts, events, tables.KEY), "stock particle outside generator population")
                        entry["table_validity"][arm] = dict(valid=True, steps=len(steps), particles=len(parts),
                            stopped_steps_p1_zero=int(steps.p1.eq(0).sum()),
                            stopped_particles_p1_zero=len(steps.loc[steps.p1.eq(0), tables.KEY+["sp"]].drop_duplicates()),
                            scope="required V1 columns; unused stock sentinel fields reported separately")
                    else:
                        entry["table_validity"][arm] = validate_tables(steps, parts, events, geom.nbars, expected)
                    loaded[arm] = (steps, parts)
                    if arm != "stock":
                        exact_frame(tables.read(str(paths[arm]), "gen"), gen, tables.KEY + ["idx"])
                        exact_frame(tables.read(str(paths[arm]), "subruns"),
                                    tables.read(str(paths["gen"]), "subruns"), ["run", "subrun"])
                    for key in total_stops[arm]:
                        total_stops[arm][key] += entry["table_validity"][arm][key]
                sn, pn = loaded["numpy"]
                sj, pj = loaded["candidate"]
                require(set(sn.columns) == set(sj.columns) and set(pn.columns) == set(pj.columns), "arm schema mismatch")
                entry["receipts"] = receipts(row, sn)
                saved_fingerprint = fingerprint_frames(sj, pj)
                entry["candidate_fingerprint"] = saved_fingerprint
                fingerprints = row.get("repeat_fingerprints", [])
                if not fingerprints or saved_fingerprint != fingerprints[0]:
                    entry["receipts"]["errors"].append("saved_candidate_not_bound_to_repeat_fingerprint")
                    entry["receipts"]["pass"] = False
                for key, value in (("generated_events", len(events)), ("primary_muons", len(expected)),
                                   ("numpy_steps", len(sn)), ("jax_steps", len(sj))):
                    if row.get(key) != value:
                        entry["receipts"]["errors"].append(f"summary_{key}_mismatch")
                        entry["receipts"]["pass"] = False
                entry["timing"] = timing(row, summary)
                ha, ia = sh.fill(sn, pn, events, edges)
                hb, ib = sh.fill(sj, pj, events, edges)
                pair = sh.paired(ia, ib, edges)
                sh.add(ah, ha)
                sh.add(bh, hb)
                sh.add_paired(ph, pair)
                v1._accumulate(np_stock, [str(paths["numpy"])], [str(paths["stock"])], geom.sector)
                v1._accumulate(jax_stock, [str(paths["candidate"])], [str(paths["stock"])], geom.sector)
                v1._accumulate(port, [str(paths["candidate"])], [str(paths["numpy"])], geom.sector)
                entry["valid"] = True
            except Exception as exc:
                entry["errors"].append(f"{type(exc).__name__}: {exc}")
        result["stop_counts"] = total_stops
        result["complete"] = len(result["shards"]) == len(discovered) and all(e["valid"] for e in result["shards"].values())
        if ah and bh:
            comparison = sh.compare(ah, bh, ph)
            result["aggregate_histograms"] = comparison
            result["histogram_accumulators"] = {"numpy": sh.to_json(ah), "candidate": sh.to_json(bh), "paired_d2": ph}
            shapes = {k: r for k, r in comparison.items() if not k.startswith("_total/")}
            # Empty required groups are unsupported evidence, never a free pass.
            shape_checks = {k: bool(r["ndf"] > 0 and finite_number(r["deck"]) and not r["flag"])
                            for k, r in shapes.items()}
            result["histogram_shape_checks"] = shape_checks
            result["port_V1"] = v1.summarise(port)
            result["numpy_V1_vs_stock"] = v1.summarise(np_stock)
            result["candidate_V1_vs_stock"] = v1.summarise(jax_stock)
            result["V1_states"] = {"port": port, "numpy_stock": np_stock, "candidate_stock": jax_stock}
            complete = result["complete"]
            result["physics_port_pass"] = bool(complete and all(e["receipts"]["pass"] for e in result["shards"].values())
                                                and shape_checks and all(shape_checks.values()) and result["port_V1"]["pass"])
            result["stock_V1_pass"] = bool(complete and result["numpy_V1_vs_stock"]["pass"]
                                          and result["candidate_V1_vs_stock"]["pass"])
        result["throughput_pass"] = bool(result["complete"] and all(e["timing"]["pass"] for e in result["shards"].values()))
        speedups = [e["timing"]["median_speedup"] for e in result["shards"].values()
                    if "median_speedup" in e.get("timing", {})]
        if speedups:
            result["per_shard_median_speedup_observed_range"] = describe(speedups)
        for name, record in files.items():
            stat = Path(name).stat()
            require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), f"input changed during audit: {name}")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    if result["errors"]:
        result["complete"] = False
        result["physics_port_pass"] = False
        result["stock_V1_pass"] = False
        result["throughput_pass"] = False
    result["ready_for_handoff"] = bool(result["physics_port_pass"] and result["stock_V1_pass"]
                                      and result["throughput_pass"])
    with out.open("x") as stream:
        json.dump(clean_json(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key {key}")
        out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summaries", required=True, nargs="+")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--stock", required=True)
    parser.add_argument("--counters", required=True)
    parser.add_argument("--out", required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("physics_port_pass", "stock_V1_pass", "throughput_pass", "ready_for_handoff")}))
    return 0 if result["ready_for_handoff"] else 1


if __name__ == "__main__":
    sys.exit(main())
