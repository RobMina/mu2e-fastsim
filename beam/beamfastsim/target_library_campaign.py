"""Offline completed-library inventory, never a physics or independence certificate.

Manifest schema: {"arms": [{"arm_id": "A", "arm_complete": "A.done",
"pot": STAGE, "stops": STAGE}]}. STAGE has integer cap (1..20000), run,
subrun, base_seed, max_unique_engines and explicit paths generation_log,
read_log, admission, selected_art, generation_complete, read_complete,
selected_read_complete, selected_tree and stage_checksums (the saved SHA256
file from that extraction). Stops additionally has parent_npz.
Paths are relative to the manifest, or absolute; admission must be named
admission.json. All completion paths must be existing regular files. The
receipts are those of target_library_audit, including selected-art verification
and, for stops, selected-parent extraction and physical-order verification.

Seed intervals are declared reservation slots [base_seed, base_seed +
max_unique_engines), NOT verified SeedService seeds or expanded configuration.
No release-specific seed arithmetic, final overrides, upstream ancestry or
sourceEventIDSequence is inferred. No generation, polling, refill or exclusions.
Run with --manifest manifest.json --out inventory.json; output is exclusive.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import target_library_audit as A
from .target_parent_audit import checksum, physical_event_order
from .target_source_events import _integers, parent_layout
from .target_beam_parent_audit import COLUMNS as BEAM_COLUMNS, selected_beam


def decode_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(token):
        raise ValueError("nonfinite JSON: " + token)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    return decode_json(Path(path).read_text())


def equal(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("receipt/metadata mismatch: " + label)


def path_at(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty artifact path required")
    path = base / value
    if not path.is_file():
        raise ValueError("missing artifact or completion marker: " + str(path))
    return path


def verify_stage_hashes(path, artifacts):
    """Require the closed extraction manifest to bind selected ART and tree together."""
    saved = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if (len(fields) != 2 or len(fields[0]) != 64
                or any(c not in '0123456789abcdef' for c in fields[0])):
            raise ValueError('malformed stage checksum manifest')
        name = fields[1].lstrip('*')
        if name in saved:
            raise ValueError('duplicate stage checksum artifact')
        saved[name] = fields[0]
    for artifact in artifacts:
        candidate = Path(artifact)
        if candidate.parent.resolve() != path.parent.resolve():
            raise ValueError('stage artifacts must share their checksum manifest directory')
        if saved.get(candidate.name) != checksum(candidate):
            raise ValueError('stage extraction hash mismatch: ' + candidate.name)


def histogram(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(n) for k, n in zip(keys, counts)}


def disjoint(rows, seen, label):
    rows = np.asarray(rows, dtype='<i8')
    unique = np.unique(rows, axis=0)
    dtype = np.dtype([(str(i), '<i8') for i in range(rows.shape[1])])
    keys = np.ascontiguousarray(unique).view(dtype).ravel()
    previous = np.concatenate(seen) if seen else np.empty(0, dtype=dtype)
    if len(unique) != len(rows) or np.isin(keys, previous).any():
        raise ValueError("duplicate " + label)
    seen.append(keys)


def verify_tree_rows(parent, tree):
    """Bind every eligible raw stop row to the immutable selected extraction."""
    import uproot
    with uproot.open(tree) as root:
        columns = root[A.TREE].arrays(list(A.COLUMNS), library='np')
    keep = (columns['pdg'] == 13) & (columns['codestop'] == 32)
    if not keep.any():
        raise ValueError('selected tree contains no eligible negative muons')
    expected = {k: np.asarray(v)[keep] for k, v in columns.items()}
    def order(data):
        identity = np.column_stack([_integers(data[k], k) for k in ('run', 'subrun', 'evt', 'trk')])
        if np.unique(identity, axis=0).shape[0] != len(identity):
            raise ValueError('duplicate selected tree stop identity')
        return np.lexsort(identity.T[::-1])
    left, right = order(expected), order(parent)
    for key in A.COLUMNS:
        if not np.array_equal(expected[key][left], parent[key][right]):
            raise ValueError('parent NPZ differs from selected tree: ' + key)


def parent_support(path, receipt, accepted, art, tree):
    with np.load(path, allow_pickle=False) as parent:
        if "metadata" not in parent:
            raise ValueError("missing parent metadata")
        embedded = decode_json(str(parent["metadata"].item()))
        equal(embedded, receipt, "embedded parent receipt")
        counts, groups, keys = parent_layout(parent)
        verify_tree_rows(parent, tree)
        equal(A.sorted_keys(keys).tolist(), accepted, "parent accepted keys")
        identity = np.column_stack([_integers(parent[k], k) for k in ("run", "subrun", "evt", "trk")])
        n = int(groups.size)
        report = receipt["parent"]
        for key, value in dict(parent_events=int(counts.size), eligible_muons=n,
                               electrons_per_parent_event=float(counts.mean()), multiplicity=histogram(counts),
                               event_order_preserved=True, expected_event_count_verified=True).items():
            equal(report.get(key), value, "parent." + key)
        momentum = np.asarray(parent["pstop"])
        if momentum.shape != (n,) or not np.isfinite(momentum).all() or np.any(momentum < 0):
            raise ValueError("invalid stop momentum")
        equal(report.get("max_end_momentum"), float(momentum.max()), "max_end_momentum")
        equal(receipt.get("library_muons"), n, "library_muons")
        equal(receipt.get("selected_parent_verified"), True, "selected_parent_verified")
        equal(receipt.get("parent_tree"), A.TREE, "parent_tree")
        order = receipt["physical_event_order"]
        equal(order.get("method"), "exact event-key join to FileIndex, sorted by physical Events entry", "physical order method")
        equal(order.get("art_events"), int(counts.size), "physical art events")
        swapped = order.get("explicit_uproot_metadata_byteswap")
        reordered = order.get("reordered_events")
        if (type(swapped) is not bool or type(reordered) is not int or not 0 <= reordered <= counts.size):
            raise ValueError("invalid physical ordering receipt")
        ordered, _ = physical_event_order(parent, art, swapped)
        for key in ("event_keys", "event_index", "event_multiplicity"):
            if not np.array_equal(ordered[key], parent[key]):
                raise ValueError("parent physical ordering mismatch: " + key)
        times = np.asarray(parent["tstop"], dtype=np.float64)
        return dict(events=int(counts.size), stops=n, multiplicity=histogram(counts),
                    foil=histogram(parent["foil"]), time=dict(count=n, minimum=float(times.min()),
                    maximum=float(times.max()), mean=float(times.mean()), sd=float(times.std()))), identity


def inventory(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    arms = manifest.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("explicit nonempty arms list required")
    base = manifest_path.parent
    ids, event_ids, stop_ids, intervals, hashes = set(), [], [], [], {}
    output = []
    totals = dict(POT_events=0, beam_draw_events=0, POT_accepted_events=0, stop_accepted_events=0, stops=0)
    multiplicity, foil = {}, {}
    time_n, time_mean, time_m2, time_min, time_max = 0, 0., 0., None, None
    for arm in arms:
        name = arm.get("arm_id")
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("unique nonempty arm_id required")
        ids.add(name)
        path_at(base, arm["arm_complete"])
        result = dict(arm_id=name, stages={})
        receipts = {}
        for stage, process in (("pot", "R1LibraryPOT"), ("stops", "R1LibraryStops")):
            spec = arm[stage]
            for field in ("cap", "run", "subrun", "base_seed", "max_unique_engines"):
                value = spec[field]
                if type(value) is not int or value < (1 if field in ("cap", "max_unique_engines") else 0):
                    raise ValueError("invalid declared " + field)
            A.positive(spec["cap"], "cap", 20000)
            lo, hi = spec["base_seed"], spec["base_seed"] + spec["max_unique_engines"]
            if any(lo < end and start < hi for start, end in intervals):
                raise ValueError("overlapping declared seed intervals")
            intervals.append((lo, hi))
            names = ("generation_log", "read_log", "admission", "selected_art",
                     "generation_complete", "read_complete", "selected_read_complete")
            paths = {key: path_at(base, spec[key]) for key in names}
            selected_tree = path_at(base, spec['selected_tree'])
            stage_checksums = path_at(base, spec['stage_checksums'])
            verify_stage_hashes(stage_checksums, [paths['selected_art'], selected_tree,
                                paths['generation_log'], paths['read_log'], paths['admission']])
            if paths["admission"].name != "admission.json":
                raise ValueError("per-stage admission.json required")
            ledger, core = A.reconcile(paths["generation_log"], paths["read_log"], process, spec["cap"])
            expected = np.column_stack([np.full(spec["cap"], spec["run"], dtype=np.int64),
                                        np.full(spec["cap"], spec["subrun"], dtype=np.int64),
                                        np.arange(1, spec["cap"] + 1, dtype=np.int64)])
            if not np.array_equal(A.sorted_keys(ledger["identities"]), expected):
                raise ValueError("ledger identities must equal declared run/subrun/event 1..cap")
            disjoint(ledger["identities"], event_ids, "event identities across stages/arms")
            receipt = read_json(paths["admission"])
            for key, value in core.items():
                equal(receipt.get(key), value, stage + "." + key)
            accepted = core["accepted_events"]
            equal(receipt.get("selected_art_events"), accepted, "selected_art_events")
            equal(A.art_entries(paths["selected_art"]), accepted, "actual selected art count")
            for artifact, field in (("selected_art", "parent_art_sha256"),):
                digest = checksum(paths[artifact])
                equal(receipt.get(field), digest, field)
                if digest in hashes and hashes[digest] != name:
                    raise ValueError("duplicate selected artifact hash across arms")
                hashes[digest] = name
            stage_result = dict(declared_cap=spec["cap"], accepted_events=accepted,
                                declared_seed_interval=[lo, hi], admission_sha256=checksum(paths["admission"]),
                                extraction_manifest_sha256=checksum(stage_checksums),
                                selected_art_sha256=receipt["parent_art_sha256"])
            receipts[stage] = receipt
            if stage == "pot":
                import uproot
                tree = path_at(base, spec['selected_tree'])
                with uproot.open(tree) as root:
                    columns = root[A.TREE].arrays(list(BEAM_COLUMNS), library='np')
                stage_result['selected_beam_content'] = selected_beam(columns, core['accepted_event_keys'])
                digest = checksum(tree)
                if digest in hashes and hashes[digest] != name:
                    raise ValueError('duplicate selected tree hash across arms')
                hashes[digest] = name
                stage_result['selected_tree_sha256'] = digest
                totals["POT_events"] += spec["cap"]
                totals["POT_accepted_events"] += accepted
            else:
                tree = path_at(base, spec["selected_tree"])
                digest = checksum(tree)
                equal(receipt.get("parent_tree_sha256"), digest, "parent_tree_sha256")
                if digest in hashes and hashes[digest] != name:
                    raise ValueError("duplicate selected tree hash across arms")
                hashes[digest] = name
                parent_path = path_at(base, spec["parent_npz"])
                support, identity = parent_support(parent_path, receipt, core["accepted_event_keys"], paths["selected_art"], tree)
                disjoint(identity, stop_ids, "stop identities across arms")
                stage_result.update(support=support, parent_npz_sha256=checksum(parent_path), selected_tree_sha256=digest)
                if receipt.get("beam_parent_events") is not None:
                    equal(receipt["beam_parent_events"], receipts["pot"]["accepted_events"], "declared beam parent events")
                totals["beam_draw_events"] += spec["cap"]
                totals["stop_accepted_events"] += accepted
                totals["stops"] += support["stops"]
                for target, source in ((multiplicity, support["multiplicity"]), (foil, support["foil"])):
                    for key, value in source.items():
                        target[key] = target.get(key, 0) + value
                t = support["time"]
                delta, combined = t["mean"] - time_mean, time_n + t["count"]
                time_m2 += t["count"] * t["sd"] ** 2 + delta ** 2 * time_n * t["count"] / combined
                time_mean += delta * t["count"] / combined
                time_n = combined
                time_min = t["minimum"] if time_min is None else min(time_min, t["minimum"])
                time_max = t["maximum"] if time_max is None else max(time_max, t["maximum"])
            result["stages"][stage] = stage_result
        output.append(result)
    return dict(schema="completed-target-library-campaign-v1", manifest_sha256=checksum(manifest_path),
                arms=output, totals=totals, support=dict(multiplicity=multiplicity, foil=foil,
                time=dict(count=time_n, minimum=time_min, maximum=time_max, mean=time_mean, sd=float(np.sqrt(time_m2 / time_n)))),
                physics_validated=False, generated_yield_validated=False, upstream_sourceEventIDSequence_audited=False,
                historical_equivalence_established=False, independent_parent_provenance_established=False,
                actual_configuration_verified=False, actual_seed_schedule_verified=False,
                automatic_refill=False, exclusions_applied=False, certificate_established=False,
                scope="offline completed artifact inventory; descriptive support only; no merged source correlations, POT normalization or independent-parent inference",
                seed_scope="manifest-declared reservation slots only; no SeedService arithmetic, final-override guards or actual configuration verification")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("refusing existing output: " + str(args.out))
    encoded = json.dumps(inventory(args.manifest), indent=2, allow_nan=False) + "\n"
    with args.out.open("x") as stream:
        stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
