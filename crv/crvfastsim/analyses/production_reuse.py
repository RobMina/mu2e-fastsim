"""Read-only production-reuse evidence and independent-reference precision, never a V1 acceptance gate."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from crvfastsim import runconfig, tables
from crvfastsim.analyses import scatter_convergence as SC, transport_validation as TV, v1compare as VC, v1strata as VS


def event_moments(crossings, universe):
    """Raw moments of [crossing count, summed length, summed energy] with zero-crossing events retained."""
    if universe.duplicated(tables.KEY).any():
        raise ValueError("duplicate event universe")
    if not np.isfinite(crossings[["len", "edep"]]).all().all() or (crossings[["len", "edep"]] < 0).any().any():
        raise ValueError("nonfinite or negative crossing value")
    u = universe[tables.KEY].copy()
    u["ev"] = np.arange(len(u))
    x = crossings.merge(u, on=tables.KEY, how="left", validate="many_to_one")
    if x.ev.isna().any():
        raise ValueError("crossing outside event universe")
    ev = x.ev.to_numpy(dtype=int)
    v = np.stack([np.bincount(ev, weights=w, minlength=len(u)) for w in
                  (None, x.len.to_numpy(), x.edep.to_numpy())])
    return {"events": len(u), "totals": v.sum(axis=1).tolist(), "cross": (v @ v.T).tolist(),
            "contributing_events": int(np.count_nonzero(v[0]))}


def merge_moments(a, b):
    """Pool disjoint files, not their ratios or standard errors."""
    if not a:
        return b
    return {"events": a["events"] + b["events"],
            "contributing_events": a["contributing_events"] + b["contributing_events"],
            "totals": (np.array(a["totals"]) + b["totals"]).tolist(),
            "cross": (np.array(a["cross"]) + b["cross"]).tolist()}


def precision(m):
    """Marginal event-cluster SEs; hypothetical independent-arm scaling does not address selection bias."""
    n = m["events"]
    t, raw = np.array(m["totals"]), np.array(m["cross"])
    if n < 2 or t[0] <= 0:
        return {"supported": False, "reason": "insufficient_events_or_crossings"}
    covariance = n / (n - 1.) * (raw - np.outer(t, t) / n)
    out = {"supported": True, "events": n, "crossings": int(t[0]),
           "contributing_events": m["contributing_events"], "crossings_per_event": float(t[0] / n)}
    for name, gradient in (("rate", np.array([1 / t[0], 0., 0.])),
                           ("length_mean", np.array([-1 / t[0], 1 / t[1], 0.]) if t[1] > 0 else None),
                           ("energy_mean", np.array([-1 / t[0], 0., 1 / t[2]]) if t[2] > 0 else None)):
        if gradient is None:
            out[name] = None
            continue
        var = float(gradient @ covariance @ gradient)
        bound = 1e-12 * max(1., float(np.abs(gradient) @ np.abs(covariance) @ np.abs(gradient)))
        if var < -bound:
            raise ValueError("negative event-cluster variance")
        se = float(np.sqrt(max(var, 0.)))
        out[name] = {"single_arm_relative_se": se,
                     "equal_distribution_independent_ratio_se": {str(k): se * np.sqrt(1 + 1 / k)
                                                                  for k in (1, 10, 20)},
                     "infinite_reference_ratio_se": se}
    return out


def normalization(log, metadata):
    """Bind original production log to its exact art shard and distinguish retained from generated events."""
    name = metadata["file_name"]
    if name not in log:
        raise ValueError("production log does not identify the metadata file")
    rows = re.findall(r"TrigReport Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)", log)
    if len(rows) != 1:
        raise ValueError("ambiguous production event summary")
    generated, retained, failed = map(int, rows[0])
    if generated != retained + failed or retained != metadata["event_count"] or generated <= 0:
        raise ValueError("production log/catalog event count mismatch")
    live = re.findall(r"Total live time simulated:\s*([0-9.eE+-]+)", log)
    if len(live) != 1 or not np.isfinite(float(live[0])) or float(live[0]) <= 0:
        raise ValueError("missing or ambiguous generated livetime")
    return {"generated_log": generated, "retained_log": retained, "failed_log": failed,
            "catalog_gencount": metadata.get("dh.gencount"), "generated_livetime_log_s": float(live[0]),
            "retained_fraction": retained / generated,
            "catalog_gencount_is_generated": metadata.get("dh.gencount") == generated}


def geometry_log_comparison(production, reference):
    """Record actual CRY projection boxes; equal top-level geometry does not imply equal world extents."""
    out = {}
    for label in ("Ref. point", "Target box", "World"):
        values = []
        for text in (production, reference):
            rows = re.findall(re.escape(label) + r":\s*([^\n]+)", text)
            if len(rows) != 1:
                raise ValueError(f"missing or ambiguous {label} log evidence")
            values.append([float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", rows[0])])
        out[label] = {"production": values[0], "reference": values[1], "equal": values[0] == values[1]}
    return out


def production_evidence(path, metadata, log):
    """Inspect a complete selected-event dump, without claiming it is an all-generated-event reference."""
    norm = normalization(log, metadata)
    with uproot.open(path) as f:
        cycles = {k: f[k].num_entries for k in f.keys(recursive=True, cycle=True)
                  if k.startswith("dump/events;")}
        columns = tables.KEY + ["n_gen", "n_particles", "n_steps", "associations_checked", "full_generator_stream"]
        census = f["dump/events"].arrays(columns, library="pd")
    if len(census) != norm["retained_log"] or census.duplicated(tables.KEY).any():
        raise ValueError("persisted census/catalog retained count mismatch")
    if not (census.associations_checked == 1).all() or (census.full_generator_stream != 0).any():
        raise ValueError("expected validated selected-event dump, not a full generator stream")
    source_ids = {tuple(row[:2]) for row in metadata.get("runs", [])}
    if source_ids and set(map(tuple, census[["run", "subrun"]].drop_duplicates().to_numpy())) != source_ids:
        raise ValueError("dump run/subrun differs from catalog")
    if "dh.first_event" in metadata and int(census.event.min()) != metadata["dh.first_event"]:
        raise ValueError("dump first event differs from catalog")
    if "dh.last_event" in metadata and int(census.event.max()) != metadata["dh.last_event"]:
        raise ValueError("dump last event differs from catalog")
    parts, steps, gen = (tables.read(path, name) for name in ("particles", "steps", "gen"))
    if parts.duplicated(tables.KEY + ["key"]).any() or gen.duplicated(tables.KEY + ["idx"]).any():
        raise ValueError("duplicate truth identity")
    for name, frame, column in (("gen", gen, "n_gen"), ("particles", parts, "n_particles"),
                                 ("steps", steps, "n_steps")):
        counts = census[tables.KEY + [column]].merge(frame.groupby(tables.KEY).size().rename("actual").reset_index(),
                                                     on=tables.KEY, how="left").fillna({"actual": 0})
        if not (counts[column] == counts.actual).all() or int(counts.actual.sum()) != len(frame):
            raise ValueError(f"persisted {name} rows do not match census")
    sr = tables.read(path, "subruns")
    if sr.duplicated(["run", "subrun"]).any() or len(sr) != 1 or not np.isfinite(sr.livetime).all():
        raise ValueError("expected one unambiguous production subrun")
    live = float(sr.livetime.sum())
    if not np.isclose(live, norm["generated_livetime_log_s"], rtol=1e-5, atol=5e-5):
        raise ValueError("dump/log livetime mismatch")
    rc = runconfig.get("run2_v10")
    geom = rc.geometry()
    if ((steps.bar < 0) | (steps.bar >= len(geom.sector))).any():
        raise ValueError("bar outside declared geometry")
    x = tables.primary_muon_crossings(steps, parts, geom.sector)
    groups = VS.groups_of(rc, geom)
    identity = steps[tables.KEY + ["sp"]].merge(parts[tables.KEY + ["key"]].rename(columns={"key": "sp"}),
                                               on=tables.KEY + ["sp"], how="left", indicator=True, validate="many_to_one")
    if (identity._merge != "both").any():
        raise ValueError("step truth identity absent from particles")
    return {"normalization": norm, "persisted_event_cycles": cycles, "persisted_livetime_s": live,
            "events": len(census), "gen_rows": len(gen), "particle_rows": len(parts), "step_rows": len(steps),
            "groups": {name: precision(event_moments(x[x.sector.isin(sectors)], census[tables.KEY]))
                       for name, sectors in groups.items()},
            "scope": "DS-entry-conditioned, pre-DS S1; not an all-generated V1 gate"}


def generator_identity_map(retained, full):
    """Map compressed generator indices to the source by exact stored values, never nearest states or time."""
    fields = ["pdg", "genid", "x", "y", "z", "px", "py", "pz", "e", "t"]
    keys = tables.KEY + fields
    if retained.duplicated(tables.KEY + ["idx"]).any() or full.duplicated(tables.KEY + ["idx"]).any():
        raise ValueError("duplicate generator index")
    if retained.duplicated(keys).any() or full.duplicated(keys).any():
        raise ValueError("ambiguous exact generator-value identity")
    identity = retained.merge(full[keys + ["idx"]].rename(columns={"idx": "source_genidx"}),
                              on=keys, how="left", validate="one_to_one")
    if identity.source_genidx.isna().any():
        raise ValueError("retained generator is not an exact subset of full source")
    return identity[tables.KEY + ["idx", "source_genidx", "pdg"]].rename(columns={"idx": "genidx"})


def projection_diagnostic(path, gen, geom, world):
    """Truth-identity muon crossings whose generated straight ray would enter a smaller world through a side."""
    lo, hi = np.array(world)[::2], np.array(world)[1::2]
    if len(lo) != 3 or np.any(hi <= lo):
        raise ValueError("invalid world bounds")
    mu = gen.loc[gen.pdg.abs() == 13].copy()
    pos, mom = mu[["x", "y", "z"]].to_numpy(), mu[["px", "py", "pz"]].to_numpy()
    direction = np.divide(mom, np.linalg.norm(mom, axis=1)[:, None],
                          out=np.zeros_like(mom), where=np.linalg.norm(mom, axis=1)[:, None] > 0)
    parallel = np.abs(direction) < 1e-14
    near = np.divide(np.where(direction >= 0, lo, hi) - pos, direction,
                     out=np.full_like(pos, -np.inf), where=~parallel)
    far = np.divide(np.where(direction >= 0, hi, lo) - pos, direction,
                    out=np.full_like(pos, np.inf), where=~parallel)
    valid = (~(parallel & ((pos < lo) | (pos > hi))).any(axis=1)
             & (far.min(axis=1) >= np.maximum(near.max(axis=1), 0)) & np.isfinite(near.max(axis=1)))
    axis = np.argmax(near, axis=1)
    mu["ray_misses_smaller_world"] = ~valid
    mu["ray_side_entry"] = valid & (axis != 1)
    retained = tables.read(path, "gen")
    identity = generator_identity_map(retained, gen)
    x = TV.primary_crossings(path, retained, geom)
    x = x.merge(identity, on=tables.KEY + ["genidx"], how="left", validate="many_to_one")
    bad = x.source_genidx.isna() | (x.pdg.abs() != 13)
    if bad.any():
        return {"supported": False, "reason": "primary_muon_retained_generator_identity_mismatch",
                "truth_muon_bar_crossings": len(x), "unresolved_crossings": int(bad.sum()),
                "examples": x.loc[bad, tables.KEY + ["genidx", "source_genidx", "pdg"]].head(8).to_dict("records"),
                "generated_muons": len(mu), "generated_ray_side_entry": int(mu.ray_side_entry.sum()),
                "generated_ray_misses_smaller_world": int(mu.ray_misses_smaller_world.sum())}
    x = x.merge(mu[tables.KEY + ["idx", "ray_side_entry", "ray_misses_smaller_world"]].rename(columns={"idx": "source_genidx"}),
                on=tables.KEY + ["source_genidx"], how="left", validate="many_to_one")
    if x.ray_side_entry.isna().any():
        raise ValueError("crossing missing signed-muon generator identity")
    out = {"supported": True, "generated_muons": len(mu), "generated_ray_side_entry": int(mu.ray_side_entry.sum()),
           "generated_ray_misses_smaller_world": int(mu.ray_misses_smaller_world.sum()), "groups": {}}
    for name, sectors in VS.groups_of(runconfig.get("run2_v10"), geom).items():
        s = x[x.sector.isin(sectors)]
        out["groups"][name] = {"truth_muon_bar_crossings": len(s), "ray_side_entry": int(s.ray_side_entry.sum()),
                               "ray_misses_smaller_world": int(s.ray_misses_smaller_world.sum())}
    return out


def reference_precision(root, subruns, smaller_world=None):
    """Audit the retained complete reference without transport or candidate regeneration."""
    if len(subruns) != len(set(subruns)):
        raise ValueError("duplicate subruns")
    manifest, rows = SC.reference_shards(root, subruns)
    rc = runconfig.get("run2_v10")
    geom = rc.geometry()
    groups = VS.groups_of(rc, geom)
    moments, records = {}, []
    for subrun, paths in rows:
        gen = tables.read(paths["gen"], "gen")
        u = gen[tables.KEY].drop_duplicates()
        if len(u) != 50000:
            raise ValueError("expected 50000 generated events per reference shard")
        x, accepted = VC._arm_crossings([paths["crvsteps"]], geom.sector)
        if len(accepted) != paths["expected_accepted"]:
            raise ValueError("reference accepted count mismatch")
        for name, sectors in groups.items():
            m = event_moments(x[x.sector.isin(sectors)], u)
            moments[name] = merge_moments(moments.get(name), m)
        record = {"subrun": subrun, "generated": len(u), "accepted": len(accepted), "paths": paths}
        if smaller_world is not None:
            record["projection_diagnostic"] = projection_diagnostic(paths["crvsteps"], gen, geom, smaller_world)
        records.append(record)
    return {"reference_manifest": TV.fingerprint(manifest), "records": records,
            "groups": {name: precision(m) for name, m in moments.items()}, "moments": moments,
            "scope": "single-arm statistical design, not evidence of production selection equivalence"}


def write_table(result, path):
    """Render statistical design separately from unverified production coverage."""
    ref = result.get("reference_precision")
    if not ref:
        return
    lines = ["# Independent-reference precision design", "",
             "Event-cluster relative SEs (%), same-distribution design only; not production equivalence or V1 acceptance.", "",
             "| Group | Crossings | Events contributing | Rate single arm | Length single arm | Length independent equal arms | Length independent 20x reference |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in ref["groups"].items():
        if not name.startswith("group:") or not row["supported"]:
            continue
        rate, length = row["rate"], row["length_mean"]
        lines.append(f"| {name[6:]} | {row['crossings']} | {row['contributing_events']} | "
                     f"{100 * rate['single_arm_relative_se']:.3f} | {100 * length['single_arm_relative_se']:.3f} | "
                     f"{100 * length['equal_distribution_independent_ratio_se']['1']:.3f} | "
                     f"{100 * length['equal_distribution_independent_ratio_se']['20']:.3f} |")
    lines += ["", "An infinite independent reference still leaves the fast arm's marginal variance. These are",
              "fixed-generated-event design errors, not livetime-normalized production errors or a coverage correction."]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-root")
    ap.add_argument("--subruns", type=int, nargs="+", default=list(range(101, 121)))
    ap.add_argument("--production-table")
    ap.add_argument("--metadata")
    ap.add_argument("--log")
    ap.add_argument("--reference-worker-log")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    result = {"schema": "crv-production-reuse-v1", "analysis_source": TV.fingerprint(__file__)}
    if a.metadata or a.log:
        if not (a.metadata and a.log):
            ap.error("--metadata and --log must be supplied together")
        metadata = json.loads(Path(a.metadata).read_text())
        log = Path(a.log).read_text()
        result["production_normalization"] = normalization(log, metadata)
        result["artifacts"] = {p: TV.fingerprint(p) for p in (a.metadata, a.log)}
        if a.reference_worker_log:
            result["generator_projection"] = geometry_log_comparison(log, Path(a.reference_worker_log).read_text())
            result["artifacts"][a.reference_worker_log] = TV.fingerprint(a.reference_worker_log)
        if a.production_table:
            result["production"] = production_evidence(a.production_table, metadata, log)
            result["artifacts"][a.production_table] = TV.fingerprint(a.production_table)
    elif a.production_table:
        ap.error("production table requires metadata and original log")
    if a.reference_root:
        bounds = result.get("generator_projection", {}).get("World", {}).get("production")
        result["reference_precision"] = reference_precision(a.reference_root, a.subruns, bounds)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    write_table(result, out.with_suffix(".md"))
    print(json.dumps(result.get("production_normalization", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
