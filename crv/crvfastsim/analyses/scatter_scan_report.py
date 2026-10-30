"""Summarize completed opt-in scatter-in probes, preserving unsupported and large-angle warnings."""
import argparse
import json
from pathlib import Path

from crvfastsim import runconfig, tables
from crvfastsim.analyses import transport_validation as TV, v1compare as VC, v1strata as VS


def diagnostic_strata(summary, outdir):
    """Keep truth-identity momentum/incidence diagnostics separate from exact V1 statistics."""
    settings = summary["settings"]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    v1path = outdir / "v1.json"
    v1path.write_text(json.dumps(summary["v1"]))
    rows = []
    for record in summary["records"]:
        rows.append(VS.shard(record["paths"]["gen"], record["paths"]["crvsteps"], record["candidate"]["path"],
                             "run2_v10", v1_stats=str(v1path)))
    moments = {}
    for row in rows:
        VS._add(moments, row["strata"])
    events = sum(r["events"] for r in rows)
    result = {"scope": "truth-identity strata, not cuts or V1 verdicts", "events": events,
              "strata": {g: {k: VS.summarise(m, events) for k, m in v.items()} for g, v in moments.items()},
              "source": TV.fingerprint(VS.__file__), "requested_settings": settings}
    (outdir / "strata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def paired_probes(a, b):
    """Event-paired control differences; common seeds are not common scattering draws."""
    geom = runconfig.get("run2_v10").geometry()
    VC.set_groups("run2_v10")
    if [r["subrun"] for r in a["records"]] != [r["subrun"] for r in b["records"]]:
        raise ValueError("paired probes require identical subruns")
    measures = {g: [] for g in VC.GROUPS}
    for ra, rb in zip(a["records"], b["records"]):
        if ra["paths"]["gen"] != rb["paths"]["gen"]:
            raise ValueError("paired probes require identical generators")
        universe = tables.read(ra["paths"]["gen"], "gen")[tables.KEY].drop_duplicates()
        arms = []
        for record in (ra, rb):
            x, _ = VC._arm_crossings([record["candidate"]["path"]], geom.sector)
            arms.append(x.rename(columns={"len": "length"}))
        xa, xb = arms
        for g, sectors in VC.GROUPS.items():
            measures[g].append(TV.compare_measures(xa[xa.sector.isin(sectors)], xb[xb.sector.isin(sectors)], universe))
    return {g: TV.merge_counts(rows) for g, rows in measures.items()}


def run(paths, outdir, with_strata=False):
    rows = [json.loads(Path(path).read_text()) for path in paths]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    lines = ["# Scatter-in scan", "", "Each probe uses the same generated sample; controls change RNG draw order.",
             "No V1 threshold or transport default is changed.", "",
             "| Probe | events | step / envelope mm | recovered misses | stops | max sigma rad | U crossing % +/- SE | D crossing % +/- SE | V1 / clustered proposal |",
             "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    evidence = []
    for path, row in zip(paths, rows):
        diagnostics = [r.get("diagnostics", {}) for r in row["records"]]
        completed = bool(diagnostics and all(d.get("complete") and not d.get("cap_hit") for d in diagnostics))
        if not completed:
            raise ValueError("scan report requires completed, uncapped scatter diagnostics")
        config = {(d["step_mm"], d["envelope_mm"]) for d in diagnostics}
        if len(config) != 1:
            raise ValueError("a probe cannot mix controls")
        step, envelope = config.pop()
        sigma = max(d["max_scatter_sigma_rad"] for d in diagnostics)
        cells = []
        for group in ("U", "D1-D4"):
            m = row["metrics"]["group:" + group]
            cells.append(f"{100 * m['crossings_rel']:+.3f} +/- {100 * m['crossings_se']:.3f}")
        name = Path(path).parent.name
        verdict = f"{'PASS' if row['v1']['pass'] else 'FAIL'} / {'PASS' if all(g['pass'] for g in row['clustered_proposal'].values()) else 'FAIL'}"
        lines.append(f"| {name} | {row['events']} | {step:g} / {envelope:g} | "
                     f"{sum(d['recovered_initial_misses'] for d in diagnostics)} | {sum(d['stopped'] for d in diagnostics)} | "
                     f"{sigma:.3f} | {' | '.join(cells)} | {verdict} |")
        item = {"input": TV.fingerprint(path), "complete": completed, "max_sigma_rad": sigma,
                "small_angle_warning": sigma > .1, "settings": row["settings"]}
        if with_strata:
            item["truth_strata"] = diagnostic_strata(row, outdir / name)
        evidence.append(item)
    pairs = [{"a": str(pa), "b": str(pb), "groups": paired_probes(a, b)}
             for pa, a, pb, b in zip(paths[1:], rows[1:], paths[:-1], rows[:-1])]
    (outdir / "scan.json").write_text(json.dumps({"probes": evidence, "adjacent_probe_differences": pairs}, indent=2) + "\n")
    (outdir / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--strata", action="store_true")
    args = parser.parse_args(argv)
    run(args.summaries, args.outdir, args.strata)


if __name__ == "__main__":
    main()
