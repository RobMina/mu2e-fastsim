"""Reproducible opt-in Run 2 scatter-in scan and unchanged V1/clustered-statistic proposal.

One invocation processes an explicit list of disjoint reference shards. Reuse candidates with
--candidate-root DIR (DIR/SUBRUN/tables/fastC.001400_SUBRUN.root); otherwise transport is opt-in.
Outputs are exclusive, and all configured sectors are retained. No acceptance rule is changed.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import runconfig, tables
from crvfastsim.analyses import transport, transport_validation as TV, v1compare as VC, v1strata as VS


PREFIX = "/exp/mu2e/data/users/ram2aq/crv_fastsim/"


def reference_shards(root, subruns):
    """Resolve the certified campaign and verify the original generator/reference byte hashes."""
    root = Path(root).resolve()
    manifest = root / "thread_a_named1M_20261026/manifest.json"
    campaign = json.loads(manifest.read_text())
    rows = []
    for subrun in subruns:
        matches = [s for s in campaign["shards"] if s["config"] == "run2" and s["subrun"] == subrun]
        if len(matches) != 1:
            raise ValueError("expected exactly one campaign shard")
        paths = {}
        for key in ("gen", "crvsteps"):
            evidence = matches[0]["artifacts"][key]
            old = evidence["path"]
            if not old.startswith(PREFIX):
                raise ValueError("reference path outside the declared campaign root")
            path = root / old[len(PREFIX):]
            got = TV.fingerprint(path)
            if got["sha256"] != evidence["sha256"]:
                raise ValueError(f"reference fingerprint mismatch: {path}")
            paths[key] = str(path)
        paths["expected_accepted"] = matches[0]["counts"]["accepted_events"]
        rows.append((subrun, paths))
    return manifest, rows


def proposal(group, shapes, threshold):
    """Replace only the two shape checks; never upgrade unsupported shape evidence."""
    checks = dict(group["checks"])
    for col in ("len", "edep"):
        s = shapes[col]
        value = s.get("chi2_ndf_clustered")
        checks[col + "_ratio"] = bool(group[col + "_ratio"]["valid"] and s.get("valid") and value is not None
                                            and np.isfinite(value) and value <= threshold)
    return {"pass": all(checks.values()), "checks": checks}


def exact_moments(candidate, reference, gen, geom, edges):
    """V1 event/bar crossings and generated-event clusters, including zero-crossing events."""
    universe = gen[tables.KEY].drop_duplicates().reset_index(drop=True)
    universe["ev"] = np.arange(len(universe))
    a, _ = VC._arm_crossings([str(candidate)], geom.sector)
    b, _ = VC._arm_crossings([str(reference)], geom.sector)
    a = a.rename(columns={"len": "length"}).merge(universe, on=tables.KEY, validate="many_to_one")
    b = b.rename(columns={"len": "length"}).merge(universe, on=tables.KEY, validate="many_to_one")
    groups = {"group:" + k: v for k, v in VC.GROUPS.items()}
    groups.update({s: [i] for i, s in enumerate(geom.sector_names)})
    out = {}
    for name, sectors in groups.items():
        xa, xb = a[a.sector.isin(sectors)], b[b.sector.isin(sectors)]
        ea, eb = xa.ev.to_numpy(), xb.ev.to_numpy()
        row = {"means": VS._moments(VS._event_vectors(xa, xb, ea, eb, len(universe))), "shapes": {}}
        if name.startswith("group:"):
            for col, var in (("len", "length"), ("edep", "edep")):
                bins, full = edges[name[6:]][col]
                row["shapes"][col] = VS._moments(VS._shape_vectors(
                    xa[var].to_numpy(), xb[var].to_numpy(), ea, eb, len(universe), bins, full))
        out[name] = row
    return len(universe), out


def format_number(value):
    """Keep unsupported numeric evidence explicit in the human-readable table."""
    return f"{value:.3f}" if value is not None and np.isfinite(value) else "unsupported"


def run(args):
    if len(args.subruns) != len(set(args.subruns)):
        raise ValueError("duplicate subruns")
    manifest, refs = reference_shards(args.reference_root, args.subruns)
    out = Path(args.outdir).resolve()
    rc = runconfig.get("run2_v10")
    geom = rc.geometry()
    campaign = json.loads(manifest.read_text())
    expected_geometry = campaign["snapshot"]["configs"]["run2"]["geometry"]
    geometry_evidence = [TV.fingerprint(rc.counters_path()), TV.fingerprint(args.gdml)]
    if [g["sha256"] for g in geometry_evidence] != [g["sha256"] for g in expected_geometry]:
        raise ValueError("geometry does not match the certified reference")
    out.mkdir(parents=True, exist_ok=False)
    VC.set_groups(rc.name)
    state = VC._new_state()
    records = []
    for subrun, paths in refs:
        dest = out / str(subrun) / "tables"
        dest.mkdir(parents=True)
        name = f"fastC.001400_{subrun:08d}.root"
        if args.candidate_root:
            candidate = Path(args.candidate_root).resolve() / str(subrun) / "tables" / name
        else:
            transport.run([paths["gen"]], str(dest), seed=subrun - 100, stage="C", run_config=rc.name,
                          gdml=args.gdml, below_msc=True, below_scatter_in=True,
                          scatter_in_step_mm=args.step_mm, scatter_in_envelope_mm=args.envelope_mm,
                          scatter_in_max_steps=args.max_steps, below_margin=args.material_margin_mm)
            candidate = dest / name
        shard = VC.run([str(candidate)], str(out / str(subrun)), compare=[paths["crvsteps"]],
                       emit=str(out / str(subrun) / "v1_emit.json"), run_config=rc.name)
        VC._merge_into(state, shard)
        record = {"subrun": subrun, "paths": paths, "candidate": TV.fingerprint(candidate)}
        diag = Path(str(candidate) + ".scatter_in.json")
        if diag.exists():
            record["diagnostics"] = json.loads(diag.read_text())["diagnostics"]
        records.append(record)
    v1 = VC.summarise(state)
    edges = {g: {c: VS.shape_edges(row, c) for c in ("len", "edep")} for g, row in v1["groups"].items()}
    totals, events = {}, 0
    for record in records:
        gen = tables.read(record["paths"]["gen"], "gen")
        reference = record["paths"]["crvsteps"]
        accepted = tables.read(reference, "particles")[tables.KEY].drop_duplicates().shape[0]
        if accepted != record["paths"]["expected_accepted"]:
            raise ValueError("persisted reference accepted-event count mismatch")
        candidate_gen = tables.read(record["candidate"]["path"], "gen")
        pd.testing.assert_frame_equal(gen, candidate_gen, check_dtype=False)
        record["accepted_events"] = accepted
        record["candidate_generator_equal"] = True
        n, moments = exact_moments(record["candidate"]["path"], reference, gen, geom, edges)
        if n != 50000:
            raise ValueError("reference must retain 50000 generated events per shard")
        events += n
        VS._add(totals, moments)
    metrics, proposed = {}, {}
    for name, m in totals.items():
        metrics[name] = VS.summarise(m["means"], events)
        metrics[name]["shapes"] = {c: VS.summarise_shape(v, events) for c, v in m["shapes"].items()}
        if name.startswith("group:"):
            group = name[6:]
            for col, shape in metrics[name]["shapes"].items():
                if shape["valid"]:
                    np.testing.assert_allclose(shape["chi2_ndf_poisson"], v1["groups"][group][col + "_ratio"]["chi2_ndf"], rtol=1e-9)
            proposed[group] = proposal(v1["groups"][group], metrics[name]["shapes"], VC.THRESH["ratio_chi2_ndf"])
    result = {"schema": "crv-scatter-convergence-v1", "events": events, "records": records,
              "reference_manifest": TV.fingerprint(manifest), "geometry": geometry_evidence, "settings": vars(args),
              "statistical_unit": "generated event; V1 event/bar crossing aggregation",
              "v1": v1, "clustered_proposal": proposed, "metrics": metrics,
              "source": {m.__name__: TV.fingerprint(m.__file__) for m in (transport, VC, VS)},
              "analysis_source": TV.fingerprint(__file__),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_table(result, out)


def write_table(result, out):
    """Render saved diagnostic evidence without regenerating transport."""
    events, v1 = result["events"], result["v1"]
    metrics, proposed = result["metrics"], result["clustered_proposal"]
    lines = ["# Run 2 V1 and clustered-variance proposal", "", f"Generated events: {events:,}; unchanged V1: {'PASS' if v1['pass'] else 'FAIL'}.",
             "Only the shape variances are replaced in the proposal; no rule is promoted.", "",
             "| Group | Crossing % +/- SE | Length % +/- SE | Energy % +/- SE | Length chi2 Poisson / clustered | Energy chi2 Poisson / clustered | V1 / proposal |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for g, row in v1["groups"].items():
        m = metrics["group:" + g]
        cells = [f"{100 * row[key]:+.3f} +/- {100 * m[err]:.3f}" for key, err in
                 (("crossings_per_event_rel", "crossings_se"), ("len_mean_rel", "len_mean_se"), ("edep_mean_rel", "edep_mean_se"))]
        shapes = [f"{format_number(row[c + '_ratio']['chi2_ndf'])} / {format_number(m['shapes'][c].get('chi2_ndf_clustered'))}" for c in ("len", "edep")]
        verdict = f"{'PASS' if row['pass'] else 'FAIL'} / {'PASS' if proposed[g]['pass'] else 'FAIL'}"
        lines.append("| " + " | ".join([g] + cells + shapes + [verdict]) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", help="render an existing summary without rerunning transport")
    parser.add_argument("--reference-root")
    parser.add_argument("--outdir")
    parser.add_argument("--gdml")
    parser.add_argument("--subruns", nargs="+", type=int, default=[101])
    parser.add_argument("--candidate-root")
    parser.add_argument("--step-mm", type=float, default=12.5)
    parser.add_argument("--envelope-mm", type=float, default=1500.)
    parser.add_argument("--material-margin-mm", type=float, default=20000.)
    parser.add_argument("--max-steps", type=int, default=100000)
    args = parser.parse_args(argv)
    if args.render:
        path = Path(args.render)
        write_table(json.loads(path.read_text()), path.parent)
    else:
        if not all((args.reference_root, args.outdir, args.gdml)):
            parser.error("--reference-root, --outdir and --gdml are required for a study")
        run(args)


if __name__ == "__main__":
    main()
