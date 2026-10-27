"""Summarise saved named-response and scatter-in probes with exact source fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path

import uproot


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(root, code, core):
    result = {"response": {}, "scatter_in": {}, "source_sha256": {}}
    for path in [code / "crvfastsim/transport.py", code / "crvfastsim/analyses/transport.py",
                 code / "crvfastsim/analyses/v2compare.py", code / "valhist/fasttables.py",
                 code / "crvfastsim/named_conditions.py", core / "mu2efastsim/transport.py"]:
        result["source_sha256"][str(path)] = sha(path)
    lines = ["# Named-run checkpoint", "", "## Response diagnostic (1000 stock-accepted events/config)", "",
             "A3 = fast reco on stock digis; A4 = fast response/reco on G4 steps. Not transport V2 or DQM.", "",
             "| Config/arm/group | PE median shift ± nominal SE (%) | Clusters shift ± nominal SE (%) | Spectrum band after 2SE (%) | State |",
             "|---|---:|---:|---:|---|"]
    for cfg in ("run1a", "run2"):
        for arm in ("a3", "a4"):
            path = root / f"response_1k_v1/{cfg}/{arm}_v2_guarded/v2compare_stats.json"
            value = json.loads(path.read_text())
            with uproot.open(root / f"response_1k_v1/{cfg}/{arm}.root") as f:
                actual = f["crvdump/events"].num_entries
            if actual != 1000 or value["events_fast"] != 1000 or value["events_ref"] != 1000:
                raise ValueError("incomplete response event universe")
            result["response"][f"{cfg}/{arm}"] = {"stats": value, "sha256": sha(path), "events": actual}
            for name, g in value["groups"].items():
                def cell(a, b):
                    return "unsupported" if a is None or b is None else f"{100*a:+.3f} ± {100*b:.3f}"
                band = g["pe_ratio"]["max_abs_dev_minus_2sigma"]
                lines.append(f"| {cfg}/{arm}/{name} | {cell(g['pe_median_rel'], g['pe_median_rel_err'])} | "
                             f"{cell(g['clusters_per_event_rel'], g['clusters_per_event_rel_err'])} | "
                             f"{100*band if band is not None else 'unsupported'} | {g['status']} |")
    lines += ["", "The retained V2 errors use independent pulse/Poisson formulas, not event-paired covariance;",
              "these are diagnostic checks, not a new precision closure certification. All overall named V2 results",
              "remain unsupported until both input arms are bound to configuration/time/conditions evidence.", "", "## Scatter-in probes", "",
              "| Step (mm) | Recovered initial misses | Segments | Max x/X0 | Max sigma (rad) | V1 |", "|---|---:|---:|---:|---:|---|"]
    for step in (50, 25):
        d = root / f"scatter_step{step}_v1"
        diag = json.loads((d / "tables/fastC.001400_00000101.root.scatter_in.json").read_text())["diagnostics"]
        verdict = json.loads((d / "v1/v1compare_stats.json").read_text())
        result["scatter_in"][str(step)] = {"diagnostics": diag, "v1": verdict}
        lines.append(f"| {step} | {diag['recovered_initial_misses']} | {diag['segments']} | "
                     f"{diag['max_segment_xx0']:.3f} | {diag['max_scatter_sigma_rad']:.3f} | {verdict['pass']} |")
    lines += ["", "## Run1A DQM replay (1000 events)", "",
              "No independent stock-seed floor; low-statistics comparison only. Run2 layout is unsupported.", "",
              "| Arm/client | Compared | Nonempty OK | Flagged | Both empty |", "|---|---:|---:|---:|---:|"]
    result["dqm"] = {}
    for arm in ("stock", "a3", "a4"):
        with uproot.open(root / f"dqm_1k_v1/{arm}.root") as f:
            if f["crvdump/events"].num_entries != 1000:
                raise ValueError("DQM event universe is incomplete")
    for arm in ("a3", "a4"):
        path = root / f"dqm_1k_v1/compare_{arm}/summary.json"
        value = json.loads(path.read_text())
        result["dqm"][arm] = {"summary": value, "sha256": sha(path)}
        for name, counts in value["sets"].items():
            lines.append(f"| {arm}/{name} | {counts['compared']} | {counts['ok']} | "
                         f"{counts['flagged']} | {counts['both_empty']} |")
    return result, "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--core", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    a = p.parse_args()
    result, text = report(a.root, Path(__file__).resolve().parents[1], a.core)
    a.outdir.mkdir(exist_ok=False)
    (a.outdir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.outdir / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
