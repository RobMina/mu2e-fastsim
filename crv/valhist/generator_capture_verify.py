"""Recheck existing captures against current strict gates without rerunning art or changing evidence."""
import argparse
import json
from pathlib import Path

from . import generator_bank_capture as B, genassoc_campaign as C, genassoc_fullgen as F, genassoc_smoke as S


def verify(certificates, out):
    output = Path(out)
    if output.exists():
        raise FileExistsError(output)
    if not certificates:
        raise ValueError("at least one completed capture is required")
    code = F._hashes([__file__, B.__file__, F.__file__, S.__file__, C.__file__, B.genassoc.__file__])
    rows, seen, bindings = [], set(), {}
    for name in certificates:
        path = Path(name).resolve(strict=True)
        bindings.update(F._hashes([path]))
        cert = json.loads(path.read_text())
        if cert["downstream_authorized"] is not False or cert["art_run"]["exit"] != 0:
            raise ValueError("capture must be successful and nonauthorizing")
        if cert["input_sha256"] != cert["input_sha256_post"] or cert["code_sha256"] != cert["code_sha256_post"]:
            raise ValueError("capture recorded input/code drift")
        F._unchanged(cert["input_sha256"])
        F._unchanged(cert["artifact_sha256"])
        bindings.update(cert["input_sha256"])
        bindings.update(cert["artifact_sha256"])
        configs = {p: h for p, h in cert["code_sha256"].items() if p.endswith(".fcl")}
        B.approved_configs(configs)
        F._unchanged(configs)
        row = cert["source"]
        part = row["run"], row["subrun"]
        if part in seen:
            raise ValueError("duplicate capture partition")
        seen.add(part)
        candidate = json.loads(Path(cert["candidate"]).read_text())
        if B.select_record(candidate, row["subrun"]) != row:
            raise ValueError("candidate/capture source mismatch")
        area = Path(cert["area"])
        source = area / "CrvFastSim/src/CrvStepTableDumper_module.cc"
        plugin = area / "build/al9-prof-e29-p107/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so"
        expected = {str(source): B.SOURCE_SHA256, str(plugin): B.PLUGIN_SHA256}
        if any(cert["code_sha256"].get(p) != h for p, h in expected.items()):
            raise ValueError("capture used an unapproved dumper")
        F._unchanged(expected)
        F.parse_dump_log((path.parent / "dump.log").read_text(), row["art"])
        traces = [p for p in cert["artifact_sha256"] if Path(p).name.startswith("loader.")]
        if len(traces) != 1:
            raise ValueError("ambiguous loader evidence")
        S.loaded_plugin(Path(traces[0]).read_text(), plugin)
        report = B.compare_file(row["path"], path.parent / "fullgen.root", row["run"], row["subrun"], row["events_inferred"])
        if report != cert["report"]:
            raise ValueError("current payload/census checks differ from original certificate")
        rows.append({"certificate": str(path), "source": row, "report": report,
                     "approved_configuration_closure": True})
    result = {"schema": "valhist.generator_capture_verify.v1", "captures": rows,
              "event_count": sum(r["report"]["event_count"] for r in rows),
              "input_sha256": bindings, "input_sha256_post": F._unchanged(bindings),
              "code_sha256": code, "code_sha256_post": F._unchanged(code),
              "downstream_authorized": False, "whole_bank_certified": False,
              "scope": "Current read-only recheck of historical captured files; original execution code hashes remain historical."}
    C._new_json(output, result)
    print(json.dumps({"files": len(rows), "events": result["event_count"], "downstream_authorized": False}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--certificates", nargs="+", required=True)
    p.add_argument("--out", required=True)
    verify(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
