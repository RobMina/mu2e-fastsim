"""Summarize immutable zero-event seed probes; distinguish observed and inferred collisions."""
import argparse
import json
from pathlib import Path
import re

from . import genassoc_campaign as C, genassoc_fullgen as F
from . import generator_seed_probe as P, generator_bank_capture as B


def assignment(text):
    P.verify_zero(text)
    headings = list(re.finditer(r"(?m)^\s*Seed Value\s+ModuleLabel.InstanceName\s*$", text))
    if len(headings) != 1:
        raise ValueError("require exactly one populated seed summary")
    block = text[headings[0].end():].split("=============================================", 1)[0]
    rows = re.findall(r"(?m)^\s*(\d+)\s+(\w+)\s*$", block)
    result = {name: int(seed) for seed, name in rows}
    if len(rows) != 3 or set(result) != {"generate", "cosmicTimeOffset", "g4run"}:
        raise ValueError("unexpected seed summary modules")
    bases = re.findall(r"(?m)^\s*Base Seed:\s*(\d+)\s*$", text)
    if not bases or len(set(bases)) != 1:
        raise ValueError("ambiguous base seed")
    base = int(bases[0])
    return base, result


def read_probe(path):
    path = Path(path).resolve(strict=True)
    initial = F._hashes([path])
    record = json.loads(path.read_text())
    if (record["generated_events"] != 0 or record["downstream_authorized"] is not False
            or record["historical_rng_state_certified"] is not False or record["execution"]["exit"] != 0):
        raise ValueError("probe is not a successful nonauthorizing zero-event initialization")
    if record["input_sha256"] != record["input_sha256_post"]:
        raise ValueError("probe input drift")
    F._unchanged(record["input_sha256"])
    F._unchanged(record["log_sha256"])
    if len(record["log_sha256"]) != 1:
        raise ValueError("ambiguous probe log")
    log = next(iter(record["log_sha256"]))
    base, seeds = assignment(Path(log).read_text())
    if base != record["base_seed"]:
        raise ValueError("probe seed differs from manifest")
    F._unchanged(initial)
    return {"path": str(path), "sha256": initial[str(path)], "mode": record["mode"],
            "subrun": record["subrun"], "base_seed": base, "seeds": seeds,
            "offsets": {name: seed - base for name, seed in seeds.items()}, "generated_events": 0}


def overlaps(rows):
    result = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            for aname in ("generate", "cosmicTimeOffset"):
                for bname in ("generate", "cosmicTimeOffset"):
                    if a["seeds"][aname] == b["seeds"][bname]:
                        result.append({"first": [a["mode"], a["subrun"], aname],
                                       "second": [b["mode"], b["subrun"], bname],
                                       "seed": a["seeds"][aname]})
    return result


def run(probes, candidate, out):
    output = Path(out)
    if output.exists():
        raise FileExistsError(output)
    bound = F._hashes([*probes, candidate, __file__, P.__file__, B.__file__, F.__file__, C.__file__])
    rows = [read_probe(p) for p in probes]
    if len({(r["mode"], r["subrun"]) for r in rows}) != len(rows):
        raise ValueError("duplicate probe partition")
    cry = [r for r in rows if r["mode"] == "crygen"]
    if len(cry) < 2 or any(r["offsets"] != cry[0]["offsets"] for r in rows):
        raise ValueError("require matching offsets from two candidate files and any reference probes")
    inventory = json.loads(Path(candidate).read_text())
    B.select_record(inventory, cry[0]["subrun"])
    projected = [{"mode": "crygen", "subrun": r["subrun"],
                  "seeds": {name: 20263000 + r["subrun"] - 2001 + offset
                            for name, offset in cry[0]["offsets"].items()}}
                 for r in inventory["files"]]
    result = {"schema": "valhist.generator_seed_report.v1", "probes": rows,
              "observed_initialization_collisions": overlaps(rows),
              "inferred_candidate_collisions_if_offsets_constant": overlaps(projected),
              "input_sha256": bound, "input_sha256_post": F._unchanged(bound),
              "downstream_authorized": False, "historical_rng_state_certified": False,
              "limitations": [
                  "Observed mappings are zero-event initialization probes, not original historical RNG states.",
                  "Candidate-wide collision list assumes constant observed offsets; not all 94 configs were read.",
                  "CRY generators have distinct seeds across candidate files under that assumption.",
                  "Same seed in CRY generation and another file's time-offset MixMaxRng couples full streams; it is not a duplicated shower claim.",
                  "Denominator species/KE counts do not use time; seed reuse alone does not establish biased counts or invalidate their marginal uncertainty.",
                  "G4Engine uses a different engine kind and is excluded from same-MixMaxRng collision counting.",
                  "Six omitted jobs and full-bank census/producer gates remain unresolved."]}
    C._new_json(output, result)
    print(json.dumps({"observed_collisions": len(result["observed_initialization_collisions"]),
                      "inferred_candidate_collisions": len(result["inferred_candidate_collisions_if_offsets_constant"]),
                      "downstream_authorized": False}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", nargs="+", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
