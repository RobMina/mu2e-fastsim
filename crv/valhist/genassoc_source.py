"""Plan strict reference-training re-dumps; never modify historical inputs or repair kinematics."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from crvfastsim import genassoc as G
from valhist import genassoc_campaign as C
from valhist import library_variants as V
from valhist import sources

GENERATED_PER_FILE = 50_000
def art_template():
    """Historical g4ref_v1 dts art path (CRVFS_PNFS, CRVFS_OWNER), field {sequence}."""
    return sources.g4ref_art("dts")


def _paths(meta, field):
    records = meta.get(field)
    if not isinstance(records, list) or not records:
        raise ValueError("missing historical source records: " + field)
    paths = []
    for record in records:
        path = record.get("path") if isinstance(record, dict) else None
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("absolute historical source path required: " + field)
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate historical source paths: " + field)
    return paths


def _disjoint(a, b, label):
    if len(np.unique(np.concatenate([a, b]), axis=0)) != len(a) + len(b):
        raise ValueError(label + " event overlap")


def _bank(paths, bound, generator=False, claims=None, per_file=None):
    """Use exact integer triples and the highest cycle; v2 exposure includes empty events."""
    populations, records = [], []
    for index, path in enumerate(paths):
        digest = V.sha256(path)
        if path in bound and bound[path] != digest:
            raise ValueError("input hash drift: " + path)
        bound[path] = digest
        role = G.validate_generator_stream(path) if generator else None
        tree = "dump/events" if role and role["schema_version"] == 2 else (
            "dump/gen" if generator else "dump/particles")
        keys, inventory = V._event_inventory(path, tree)
        if per_file is not None and len(keys) != per_file:
            raise ValueError("full-generator per-file exposure mismatch: " + path)
        if claims is not None:
            claim = claims[index]
            for key in ("sha256", "unique_events", "event_keys_sha256"):
                actual = digest if key == "sha256" else inventory[key]
                if key in claim and claim[key] != actual:
                    raise ValueError("historical source " + key + " mismatch: " + path)
        records.append({"path": path, "sha256": digest, "events": inventory,
                        "generator_validation": role})
        populations.append(keys)
    all_keys = np.concatenate(populations)
    unique = np.unique(all_keys, axis=0)
    if len(all_keys) != len(unique):
        raise ValueError("cross-file event overlap: " + tree)
    return unique, records, populations


def _logs(donors, out, bound):
    rows = []
    for index, donor in enumerate(donors, 1):
        p = Path(donor)
        seq = f"001400_{index:08d}"
        if p.name != f"crvsteps.{seq}.root" or p.parent != Path(donors[0]).parent:
            raise ValueError("unexpected reference donor sequence/path: " + donor)
        log = p.parent / "logs" / f"dump.{seq}.log"
        row = {"sequence": seq, "old_table": donor, "old_sha256": bound[donor],
               "new_table": str(out / "donors" / p.name)}
        if log.exists():
            record = V._record(log)
            bound[str(log)] = record["sha256"]
            text = log.read_text()
            inputs = set(re.findall(r'Opened input file "([^\"]+)"', text))
            exits = re.findall(r"Art has completed and will exit with status (-?\d+)\b", text)
            if len(inputs) != 1 or not exits or any(exit != "0" for exit in exits):
                raise ValueError("cannot certify original input/completion from " + str(log))
            art = inputs.pop()
            if art != art_template().format(sequence=seq):
                raise ValueError("unexpected original reference art path: " + art)
            row.update(art=art, old_log=str(log), old_log_sha256=record["sha256"],
                       source_evidence={"status": "certified historical dump log", "logs": [record]})
        elif index == 1 and not log.is_symlink():
            # Resolve only after shard 2 has passed its own exact path/completion checks.
            row.update(art=None, old_log=None, old_log_sha256=None)
        else:
            raise FileNotFoundError("missing historical donor log: " + str(log))
        rows.append(row)
    if rows[0]["art"] is None:
        driver = Path(donors[0]).parent / "logs/driver.log"
        record = V._record(driver)
        bound[str(driver)] = record["sha256"]
        lines = re.findall(r"(?m)^shard 1 steps_rc=(-?\d+) gen_rc=(-?\d+)(?:\s+\d{2}:\d{2}:\d{2})?\s*$", driver.read_text())
        if not lines or any(pair != ("0", "0") for pair in lines):
            raise ValueError("missing successful shard 1 steps/gen driver evidence")
        second = rows[1]
        rows[0].update(
            art=second["art"].replace("001400_00000002", "001400_00000001"),
            source_evidence={"status": "derived path; NOT certified historical dump log",
                             "derived_from_sequence": second["sequence"],
                             "derived_from_art": second["art"],
                             "substitution": ["001400_00000002", "001400_00000001"],
                             "missing_log": str(Path(donors[0]).parent / "logs/dump.001400_00000001.log"),
                             "logs": [record, *second["source_evidence"]["logs"]],
                             "required_gate": "Strict redump and exact original physics-payload comparison."})
    return rows


def plan(production_plan, historical_source, outdir):
    """Freeze ten reference donors and the production plan's unchanged external full-gen bank."""
    out = V._new_output(outdir)
    production = V._record(production_plan)
    historical = V._record(historical_source)
    _, prod = C._load_plan(production["path"])
    source = json.loads(Path(historical["path"]).read_text())
    donors = _paths(source, "training")
    gen = _paths(source, "training_gen")
    evaluation = _paths(source, "evaluation_gen")
    if len(donors) != 10 or len(gen) != 10 or len(prod["donors"]) != 20:
        raise ValueError("require ten reference donors/full-gen files and twenty production donors")
    if prod["generated_per_file"] != 500_000 or prod["generated"] != 10_000_000:
        raise ValueError("unexpected production generated exposure")
    generated = V._positive_int(source.get("training_generated_events"), "training generated exposure")
    if generated != GENERATED_PER_FILE * 10:
        raise ValueError("historical training generated exposure must be 500000 (50000/file)")
    if (len(prod["gen"]) != 10 or set(gen) != set(prod["gen"])
            or set(prod["gen_sha256"]) != set(gen)):
        raise ValueError("training full-gen bank must match all ten production denominator paths/hashes")
    if not Path(prod["area"]).is_absolute():
        raise ValueError("production area must be absolute")
    if set(donors) & set(gen + evaluation):
        raise ValueError("donor/full-generator source role collision")
    bound = {production["path"]: production["sha256"], historical["path"]: historical["sha256"],
             prod["source_metadata"]: prod["source_metadata_sha256"], **prod["gen_sha256"]}
    # Older sidecars have paths/inventories only. Check all historical hashes when supplied.
    if "input_sha256" in source:
        hashes = source["input_sha256"]
        if not isinstance(hashes, dict) or not set(donors + gen + evaluation) <= set(hashes):
            raise ValueError("incomplete historical input_sha256 binding")
        for path in donors + gen + evaluation:
            if path in bound and bound[path] != hashes[path]:
                raise ValueError("historical/production hash mismatch: " + path)
            bound[path] = hashes[path]
    train_keys, gen_records, gen_parts = _bank(gen, bound, True, source["training_gen"], GENERATED_PER_FILE)
    eval_keys, eval_records, _ = _bank(evaluation, bound, True, source["evaluation_gen"])
    _disjoint(train_keys, eval_keys, "training/evaluation full-generator")
    if "evaluation_generated_events" in source:
        count = V._positive_int(source["evaluation_generated_events"], "evaluation generated exposure")
        if count != len(eval_keys):
            raise ValueError("evaluation generated exposure mismatch")
    donor_keys, donor_records, donor_parts = _bank(donors, bound, claims=source["training"])
    for accepted, full in zip(donor_parts, gen_parts):
        if len(np.unique(np.concatenate([accepted, full]), axis=0)) != len(full):
            raise ValueError("training donor particles outside corresponding full-generator universe")
    prod_paths = [row["old_table"] for row in prod["donors"]]
    if len(set(prod_paths)) != 20 or any(not Path(p).is_absolute() for p in prod_paths):
        raise ValueError("invalid production donor paths")
    for row in prod["donors"]:
        if row["old_table"] in bound and bound[row["old_table"]] != row["old_sha256"]:
            raise ValueError("production donor hash conflict")
        bound[row["old_table"]] = row["old_sha256"]
    prod_keys, prod_records, _ = _bank(prod_paths, bound)
    _disjoint(train_keys, prod_keys, "training/production accepted donor")
    _disjoint(eval_keys, prod_keys, "evaluation/production accepted donor")
    rows = _logs(donors, out, bound)
    provenance = {
        "historical_source": historical, "production_plan": production,
        "input_sha256": bound, "source_evidence": [r["source_evidence"] for r in rows],
        "training": donor_records, "training_gen": gen_records, "evaluation_gen": eval_records,
        "production_accepted_donors": prod_records,
        "certification": {"training_generated_events": len(train_keys),
                          "evaluation_generated_events": len(eval_keys),
                          "training_persisted_particle_events": len(donor_keys),
                          "train_eval_overlap": 0, "training_production_accepted_overlap": 0,
                          "evaluation_production_accepted_overlap": 0},
        "limitations": ["Legacy full-gen stream role remains explicitly unverified when no v2 census exists.",
                        "Production independence covers persisted particle event keys, not unpersisted accepted events.",
                        "Accepted donor counts never establish generated exposure.",
                        "Derived shard-1 path requires strict redump and exact original physics-payload comparison.",
                        "Association repair is not physics closure; no kinematic repair or rate tuning."]}
    metadata = {"inputs": donors, "gen": gen, "generated_scaled_to": generated, "provenance": provenance}
    result = {"schema": "valhist.genassoc_source.v1", "source_metadata": str(out / "source_metadata.json"),
              "area": prod["area"], "generated_per_file": GENERATED_PER_FILE, "generated": generated,
              "gen": gen, "gen_sha256": {p: prod["gen_sha256"][p] for p in gen}, "donors": rows,
              "provenance": provenance,
              "policy": "Full-generator denominator unchanged; no overwrite; strict redump plus exact legacy comparison."}
    for path, digest in bound.items():
        if V.sha256(path) != digest:
            raise ValueError("input changed during planning: " + path)
    out.mkdir()
    for name in ("donors", "logs", "comparisons", "library", "validation"):
        (out / name).mkdir()
    C._new_json(out / "source_metadata.json", metadata)
    result["source_metadata_sha256"] = V.sha256(out / "source_metadata.json")
    C._new_json(out / "plan.json", result)
    with (out / "donors.tsv").open("x") as stream:
        for row in rows:
            stream.write("\t".join(row[key] for key in ("sequence", "art", "old_table", "new_table")) + "\n")
    print(f"planned {len(rows)} reference donors, {generated} generated exposure, output {out}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("plan")
    command.add_argument("--production-plan", required=True)
    command.add_argument("--historical-source", required=True)
    command.add_argument("--outdir", required=True)
    args = parser.parse_args()
    plan(args.production_plan, args.historical_source, args.outdir)


if __name__ == "__main__":
    main()
