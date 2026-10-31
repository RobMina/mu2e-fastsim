"""Audit legacy full-generator exposure; outputs NEVER replace the denominator bank.

Run in the production plan's isolated MUSE area using its Python environment::

    python3 -m valhist.genassoc_fullgen PRODUCTION_PLAN \\
        --historical-source DATA/.../takeover/lib_shower_g4train.json --outdir NEWDIR

No build, pnfs discovery/stat/hash, retry, overwrite, or denominator publication is
performed. Only art opens the twenty exact historical gen input paths. A failed
attempt keeps its output directory and evidence; retry with another NEWDIR.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import uproot

from crvfastsim import genassoc, tables
from valhist import genassoc_campaign as campaign, genassoc_smoke as smoke
from valhist import library_variants as variants
from valhist import sources

GENERATED_PER_FILE = 50_000
def art_template():
    """Historical g4ref_v1 gen-stream art path (CRVFS_PNFS, CRVFS_OWNER), field {sequence}."""
    return sources.g4ref_art("gen")
POLICY = "Audit only: new fullgen outputs are NOT replacements for the denominator bank."


def _hashes(paths):
    return {str(p): variants.sha256(p) for p in paths}


def _unchanged(before):
    after = _hashes(before)
    for path, digest in before.items():
        if after[path] != digest:
            raise ValueError("input/code hash drift: " + path)
    return after


def parse_dump_log(text, expected_art):
    """Require exact input and exclusively successful completion, not a substring hit."""
    inputs = set(re.findall(r'Opened input file "([^\"]+)"', text))
    exits = re.findall(r"Art has completed and will exit with status (-?\d+)\b", text)
    if inputs != {expected_art} or not exits or any(code != "0" for code in exits):
        raise ValueError("cannot certify original input/completion: " + expected_art)
    return expected_art


def parse_driver_log(text):
    lines = re.findall(r"(?m)^shard 1 steps_rc=(-?\d+) gen_rc=(-?\d+)(?:\s+\d{2}:\d{2}:\d{2})?[ \t]*$", text)
    if not lines or any(pair != ("0", "0") for pair in lines):
        raise ValueError("missing successful shard 1 steps/gen driver evidence")


def legacy_paths(meta, historical):
    """Derive 11--20 by replacing only the exact shard token in filename 1."""
    training = meta.get("gen", [])
    if len(training) != 10 or len(set(training)) != 10:
        raise ValueError("require ten original production-plan fullgen paths")
    first = Path(training[0])
    token = "001400_00000001"
    if not first.is_absolute() or first.name.count(token) != 1 or first.suffix != ".root":
        raise ValueError("unexpected original fullgen filename")
    paths = [str(first.with_name(first.name.replace(token, f"001400_{i:08d}"))) for i in range(1, 21)]
    if training != paths[:10] or set(meta.get("gen_sha256", {})) != set(training):
        raise ValueError("training fullgen sequence/hash coverage mismatch")
    evaluation = historical.get("evaluation_gen")
    if (not isinstance(evaluation, list) or len(evaluation) != 10
            or any(not isinstance(r, dict) for r in evaluation)
            or [r.get("path") for r in evaluation] != paths[10:]):
        raise ValueError("historical evaluation_gen must match exact derived paths 11--20")
    if "training_gen" in historical and [r.get("path") for r in historical["training_gen"]] != training:
        raise ValueError("historical training_gen differs from production plan")
    return paths


def original_inputs(paths):
    """Read local dumpgen logs only; missing shard 1 has an explicitly weaker derivation."""
    rows, bound = [], {}
    for index, old in enumerate(paths, 1):
        sequence = f"001400_{index:08d}"
        log = Path(old).parent / "logs" / f"dumpgen.{sequence}.log"
        art = art_template().format(sequence=sequence)
        row = {"sequence": sequence, "oldpath": old, "art": art}
        if index == 1 and not os.path.lexists(log):
            row["source_evidence"] = None
        else:
            bound.update(_hashes([log]))
            parse_dump_log(log.read_text(), art)
            row["source_evidence"] = {"status": "certified historical dump log", "logs": [str(log)]}
        rows.append(row)
    if rows[0]["source_evidence"] is None:
        driver = Path(paths[0]).parent / "logs/driver.log"
        bound.update(_hashes([driver]))
        parse_driver_log(driver.read_text())
        second = rows[1]
        rows[0]["art"] = second["art"].replace("001400_00000002", "001400_00000001")
        rows[0]["source_evidence"] = {
            "status": "derived path; NOT certified historical dump log",
            "derived_from_sequence": second["sequence"], "derived_from_art": second["art"],
            "substitution": ["001400_00000002", "001400_00000001"],
            "missing_log": str(Path(paths[0]).parent / "logs/dumpgen.001400_00000001.log"),
            "logs": [str(driver), *second["source_evidence"]["logs"]],
            "required_gate": "Full strict re-dump and exact original gen/subruns payload comparison."}
    _unchanged(bound)
    return rows, bound


def compare_payload(old, new, expected_events=None):
    """Compare every original column/row, preserving order and exact values (no tolerance)."""
    expected_events = GENERATED_PER_FILE if expected_events is None else variants._positive_int(expected_events, "expected events")
    validation = genassoc.validate_generator_stream(str(new))
    if validation["schema_version"] != 2:
        raise ValueError("fullgen audit requires schema-v2 event census")
    # NumPy reads string branches without the optional awkward-pandas adapter.
    with uproot.open(new) as root:
        census = pd.DataFrame(root["dump/events"].arrays(library="np"))
    if len(census) != expected_events:
        raise ValueError(f"fullgen event count must be {expected_events}, got {len(census)}")
    if "gen_tag" not in census or not census.gen_tag.eq("cosmicTimeOffset").all():
        raise ValueError("fullgen configured generator tag mismatch")
    comparisons = {}
    for tree in ("gen", "subruns"):
        a, b = tables.read(str(old), tree), tables.read(str(new), tree)
        missing = set(a.columns) - set(b.columns)
        if missing:
            raise ValueError(f"{tree}: missing legacy columns {sorted(missing)}")
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b[a.columns].reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
        comparisons[tree] = {"rows": len(a), "legacy_columns": list(a.columns), "identical": True}
    keys = census[tables.KEY].to_numpy(dtype=np.int64)
    identities = lambda row: {name: int(value) for name, value in zip(tables.KEY, row)}
    result = {"event_count": len(census), "genrows": comparisons["gen"]["rows"],
              "zero_generator_events": int(census.n_gen.eq(0).sum()),
              "legacy_values_identical": True, "first_event": identities(keys[0]),
              "last_event": identities(keys[-1]), "comparison": comparisons, "validation": validation}
    return result, keys


def _config_files(config, area):
    """Bind the actual include search choices as well as the top-level configuration."""
    search = [area, *(Path(p) if Path(p).is_absolute() else area / p
                      for p in os.environ.get("FHICL_FILE_PATH", "").split(":") if p)]
    found = set()

    def visit(path):
        path = path.resolve(strict=True)
        if path in found:
            return
        found.add(path)
        for name in re.findall(r'^\s*#include\s+"([^\"]+)"', path.read_text(), re.MULTILINE):
            candidates = [base / name for base in search]
            include = next((p for p in candidates if p.is_file()), None)
            if include is None:
                raise FileNotFoundError("cannot bind FHiCL include: " + name)
            if name.startswith("CrvFastSim/") and include.resolve() != (area / name).resolve():
                raise ValueError("FHiCL include outside isolated area: " + name)
            visit(include)
    visit(config)
    return found


def certify(production_plan, historical_source, outdir):
    """Run all twenty files sequentially; publish a certificate only after every gate passes."""
    out = variants._new_output(outdir)
    out.mkdir()  # Exclusive reservation, including for failed preflight attempts.
    try:
        return _certify(production_plan, historical_source, out)
    except BaseException as exc:
        campaign._new_json(out / "failed.json", {"status": "FAILED; no certificate", "error": str(exc),
                                                "exception": type(exc).__name__, "policy": POLICY})
        raise


def _certify(production_plan, historical_source, out):
    plan = Path(production_plan).resolve(strict=True)
    historical_path = Path(historical_source).resolve(strict=True)
    input_pre = _hashes([plan, historical_path])
    _, meta = campaign._load_plan(plan)
    historical = json.loads(historical_path.read_text())
    paths = legacy_paths(meta, historical)
    area = Path(meta["area"])
    if not area.is_absolute() or os.environ.get("MUSE_WORK_DIR") != str(area):
        raise ValueError("MUSE_WORK_DIR must be " + str(area))
    input_pre.update(_hashes([meta["source_metadata"], *paths]))
    if input_pre[meta["source_metadata"]] != meta["source_metadata_sha256"]:
        raise ValueError("source metadata drift during preflight")
    for path, digest in meta["gen_sha256"].items():
        if input_pre[path] != digest:
            raise ValueError("production old hash mismatch: " + path)
    claims = historical.get("input_sha256", {})
    if not isinstance(claims, dict):
        raise ValueError("historical input_sha256 must be a mapping")
    # Older historical metadata has inventories but no checksum claims. Freeze those
    # bytes now, and explicitly distinguish that baseline from historical hash proof.
    hash_evidence = {}
    for path in paths:
        records = historical.get("training_gen", []) + historical["evaluation_gen"]
        expected = [r["sha256"] for r in records if r.get("path") == path and "sha256" in r]
        if path in claims:
            expected.append(claims[path])
        if any(digest != input_pre[path] for digest in expected):
            raise ValueError("historical old hash mismatch: " + path)
        hash_evidence[path] = ("historical checksum checked" if expected else
                               "production checksum checked" if path in meta["gen_sha256"] else
                               "no historical checksum supplied; frozen audit baseline only")
    rows, logs = original_inputs(paths)
    input_pre.update(logs)
    libs = list((area / "build").glob("*/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so"))
    if len(libs) != 1:
        raise ValueError("expected exactly one user-built dumper in isolated area")
    plugin = libs[0]
    source = area / "CrvFastSim/src/CrvStepTableDumper_module.cc"
    if plugin.stat().st_mtime < source.stat().st_mtime:
        raise ValueError("dumper binary predates source; user build required")
    config = area / "CrvFastSim/fcl/dump_gen.fcl"
    executable = shutil.which("mu2e")
    if executable is None:
        raise FileNotFoundError("mu2e is not on PATH; set up the isolated MUSE area")
    code_files = [Path(__file__).resolve(), Path(sys.executable).resolve(), Path(executable).resolve(),
                  source, plugin, *_config_files(config, area)]
    # Include all local Python implementation dependencies used for the certificate.
    for module in (genassoc, tables, tables._core, campaign, smoke, variants):
        code_files.append(Path(module.__file__).resolve())
    code_pre = _hashes(code_files)
    environment = {k: os.environ.get(k) for k in ("MUSE_WORK_DIR", "FHICL_FILE_PATH", "CET_PLUGIN_PATH",
                                                  "LD_LIBRARY_PATH", "MU2E_SEARCH_PATH", "PATH", "PYTHONPATH")}
    # Make the include lookup order identical to the recursively frozen configs.
    environment["FHICL_FILE_PATH"] = str(area) + ":" + (environment["FHICL_FILE_PATH"] or "")
    manifest = {"policy": POLICY, "production_plan": str(plan), "historical_source": str(historical_path),
                "input_sha256_pre": input_pre, "code_sha256_pre": code_pre, "inputs": rows,
                "old_hash_evidence": hash_evidence, "environment": environment,
                "python": sys.executable, "expected_events_per_file": GENERATED_PER_FILE,
                "art_checksum_policy": "Exact historical paths only; no pnfs metadata scans or content hashing."}
    campaign._new_json(out / "inputs.json", manifest)
    _unchanged(input_pre)
    _unchanged(code_pre)
    outputs, all_keys, completed_hashes = {}, [], {}
    for row in rows:
        seq = row["sequence"]
        target, log = out / ("fullgen." + seq + ".root"), out / ("dumpgen." + seq + ".log")
        command = [executable, "-c", str(config), "-s", row["art"], "-T", str(target)]
        prefix = out / ("dumpgen." + seq + ".loader")
        env = dict(os.environ, LD_DEBUG="libs", LD_DEBUG_OUTPUT=str(prefix),
                   FHICL_FILE_PATH=environment["FHICL_FILE_PATH"])
        campaign._new_json(out / (seq + ".attempt.json"), {"command": command, "cwd": str(area),
                           "source": row, "old_sha256": input_pre[row["oldpath"]], "loader_prefix": str(prefix)})
        with log.open("x") as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=area)
            try:
                code = proc.wait()
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
        if code != 0:
            raise RuntimeError(f"{seq}: art exit {code}; retained {log}")
        parse_dump_log(log.read_text(), row["art"])
        trace = Path(str(prefix) + "." + str(proc.pid))
        loaded = smoke.loaded_plugin(trace.read_text(), plugin)
        artifact_hashes = _hashes([target, log, trace])
        _unchanged({row["oldpath"]: input_pre[row["oldpath"]]})
        report, keys = compare_payload(row["oldpath"], target)
        # Reject overlap, including empty-generator events, across the certified bank.
        all_keys.append(keys)
        joined = np.concatenate(all_keys)
        if len(np.unique(joined, axis=0)) != len(joined):
            raise ValueError("cross-file fullgen event overlap")
        input_post, code_post = _unchanged(input_pre), _unchanged(code_pre)
        _unchanged(artifact_hashes)
        done = {"command": command, "exit": code, "source": row, "old_sha256": input_pre[row["oldpath"]], "loaded_plugin": loaded,
                "input_sha256_pre": input_pre, "input_sha256_post": input_post,
                "code_sha256_pre": code_pre, "code_sha256_post": code_post,
                "artifact_sha256": artifact_hashes, "report": report, "policy": POLICY}
        done_path = out / (seq + ".done.json")
        campaign._new_json(done_path, done)
        completed_hashes.update(artifact_hashes)
        completed_hashes.update(_hashes([done_path]))
        outputs[row["oldpath"]] = {"v2path": str(target), "digest": artifact_hashes[str(target)],
            **{key: report[key] for key in ("event_count", "genrows", "zero_generator_events",
                                           "legacy_values_identical", "first_event", "last_event")}}
        print(f"{seq}: certified {report['event_count']} events; audit only", flush=True)
    _unchanged(completed_hashes)
    result = {"schema": "valhist.genassoc_fullgen.v1", "policy": POLICY, "files": outputs,
              "input_sha256_pre": input_pre, "input_sha256_post": _unchanged(input_pre),
              "code_sha256_pre": code_pre, "code_sha256_post": _unchanged(code_pre),
              "evidence_sha256": {**_hashes([out / "inputs.json"]), **completed_hashes},
              "event_count": sum(r["event_count"] for r in outputs.values()),
              "cross_file_event_overlap": 0}
    campaign._new_json(out / "certificate.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production_plan")
    parser.add_argument("--historical-source", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    certify(args.production_plan, args.historical_source, args.outdir)


if __name__ == "__main__":
    main()
