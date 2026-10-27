"""Runtime negative/full-generator checks for the isolated schema-v2 dumper."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import pandas as pd
import uproot

from crvfastsim import genassoc, tables
from valhist.genassoc_campaign import _new_json
from valhist.library_variants import sha256


def loaded_plugin(trace, expected):
    """Require the loader to initialize exactly the expected dumper library."""
    paths = re.findall(r"calling init:\s*(\S*libcrvfastsim_CrvStepTableDumper_module\.so)", trace)
    if paths != [str(expected)]:
        raise ValueError(f"unexpected loaded dumper: {paths}")
    return paths[0]


def smoke(plan, gen_input, legacy_gen, outdir, events=100):
    """Run two small art jobs, retain failures, and compare generator payload without editing it."""
    plan = Path(plan).resolve(strict=True)
    meta = json.loads(plan.read_text())
    area = Path(meta["area"])
    if Path(os.environ.get("MUSE_WORK_DIR", "")) != area:
        raise ValueError("wrong MUSE_WORK_DIR")
    donor = meta["donors"][0]
    done = json.loads((plan.parent / "logs" / f"dump.{donor['sequence']}.done.json").read_text())
    plugin = next(Path(p) for p in done["bound"] if p.endswith("_module.so"))
    for path, digest in done["bound"].items():
        if sha256(path) != digest:
            raise ValueError("first-donor source/binary/plan drift: " + path)
    out = Path(outdir).resolve()
    out.mkdir()
    wrong = out / "wrong_generator.fcl"
    wrong.write_text('#include "CrvFastSim/fcl/dump_crvsteps_genassoc.fcl"\n'
                     'physics.analyzers.dump.genParticles: "cosmicTimeOffset"\n')
    gen_config = area / "CrvFastSim/fcl/dump_gen.fcl"
    frozen = {str(p): sha256(p) for p in (wrong, gen_config, Path(legacy_gen))}
    results = {}
    for name, config, source, count in (("wrong_generator", wrong, donor["art"], 10),
                                        ("full_generator", gen_config, gen_input, events)):
        target = out / (name + ".root")
        cmd = ["mu2e", "-c", str(config), "-s", source, "-n", str(count), "-T", str(target)]
        env = dict(os.environ, LD_DEBUG="libs", LD_DEBUG_OUTPUT=str(out / (name + ".loader")))
        with (out / (name + ".log")).open("x") as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            code = proc.wait()
        log = (out / (name + ".log")).read_text()
        trace = out / f"{name}.loader.{proc.pid}"
        loaded_plugin(trace.read_text(), plugin)
        results[name] = {"command": cmd, "exit": code, "loaded_plugin": str(plugin),
                         "plugin_sha256": sha256(plugin), "trace_sha256": sha256(trace)}
        if name == "wrong_generator":
            if code == 0 or not re.search(r"references GenParticle .*selected cosmicTimeOffset ProductID", log):
                raise ValueError("wrong-generator job did not fail at the pointer ProductID guard")
            if target.exists():
                with uproot.open(target) as root:
                    rows = {t: int(root["dump/" + t].num_entries) for t in ("events", "gen", "particles", "steps")}
                if any(rows.values()):
                    raise ValueError("unexpected payload before first-event wrong-generator rejection")
                results[name]["payload_rows"] = rows
        else:
            if code != 0 or "Art has completed and will exit with status 0" not in log:
                raise ValueError("full-generator smoke failed")
            report = genassoc.validate_generator_stream(str(target))
            with uproot.open(target) as root:
                census = pd.DataFrame(root["dump/events"].arrays(library="np"))
            if len(census) != events or not census.gen_tag.eq("cosmicTimeOffset").all():
                raise ValueError("full-generator event count or configured tag mismatch")
            actual = tables.read(str(target), "gen")
            old = tables.read(legacy_gen, "gen")
            expected = old.merge(census[tables.KEY], on=tables.KEY, how="inner", validate="many_to_one")
            pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual[old.columns].reset_index(drop=True),
                                          check_exact=True, check_dtype=False)
            results[name].update(validation=report, legacy_payload_identical=True, compared_rows=len(expected),
                                 output_sha256=sha256(target), limitation="Configuration smoke only; denominator bank unchanged.")
    for path, digest in {**done["bound"], **frozen}.items():
        if sha256(path) != digest:
            raise ValueError("input/source drift: " + path)
    results["bound"] = {**done["bound"], **frozen, str(Path(__file__)): sha256(__file__)}
    results["environment"] = {k: os.environ.get(k) for k in ("MUSE_WORK_DIR", "FHICL_FILE_PATH", "CET_PLUGIN_PATH", "LD_LIBRARY_PATH")}
    _new_json(out / "smoke.json", results)
    print(json.dumps({k: v for k, v in results.items() if k in ("wrong_generator", "full_generator")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--gen-input", required=True)
    parser.add_argument("--legacy-gen", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--events", type=int, default=100)
    args = parser.parse_args()
    smoke(args.plan, args.gen_input, args.legacy_gen, args.outdir, args.events)


if __name__ == "__main__":
    main()
