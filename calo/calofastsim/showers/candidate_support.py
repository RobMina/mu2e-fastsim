"""Deterministic fixed-K8 audit of a separate candidate on authenticated exposed file0."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import uproot

from . import coverage, export, library, production_compare, source_audit, source_events, source_overlap_report, source_split


def pinned_json(inputs, path, digest):
    result = inputs.json(path)
    if inputs.sha256[str(Path(path).absolute())] != source_audit._digest(digest):
        raise ValueError("JSON input hash mismatch")
    return result


def pinned_npz(inputs, path, digest):
    result = inputs.npz(path)
    if inputs.sha256[str(Path(path).absolute())] != source_audit._digest(digest):
        raise ValueError("NPZ input hash mismatch")
    return result


def parent_bundle(inputs, directory, parent_summary):
    """Require all three bundle hashes in the externally pinned parent report."""
    directory = Path(directory)
    for name in ("summary.json", "expected.npz", "ledger.npz"):
        suffix = "/" + directory.parent.name + "/bound/" + name
        matches = [value for key, value in parent_summary["input_sha256"].items() if key.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("missing unique parent-bundle digest")
        if source_audit._sha(inputs.read(directory / name)) != matches[0]:
            raise ValueError("parent-bundle hash mismatch")
    return source_overlap_report._bound(directory, inputs, role="development", item=directory.parent.name,
                                       fid="107658739", parent_fid=source_audit.PARENT_FID, count=2813)


def validate_query(path, proof, bound):
    """Check every stored identity and source tuple, not just front-entry events."""
    digest = source_events.file_digest(path)
    if digest != proof["input_sha256"]["tables"]:
        raise ValueError("query tables differ from completed runtime validation")
    if (proof.get("events") != len(bound["expected"]["event"]) or
            proof.get("printer_source_ids_exact") is not True or
            proof.get("ordered_event_census_exact") is not True or
            proof.get("comparison", {}).get("all_old_columns_exact") is not True):
        raise ValueError("query proof lacks complete old-column/source certification")
    numeric, checked = source_events.read(path, bound["expected"], source_tag="TargetStopResampler::Primary",
                                         source_branch="art::EventIDs_TargetStopResampler__Primary.")
    for field in export.EVENT + ("source_run", "source_subrun", "source_event", "source_count"):
        if not np.array_equal(numeric[field], bound["ledger"][field]):
            raise ValueError("query source ledger mismatch: " + field)
    if source_events.file_digest(path) != digest:
        raise ValueError("query tables changed during validation")
    return checked


def geometry_parity(path, geometry, *libraries):
    """Check every saved physical coordinate field; never invoke prepare or sampling."""
    checked = []
    with uproot.open(path) as source:
        for packed in libraries:
            complete = dict(geometry)
            for prefix, table in (("crystal", "crystals"), ("disk", "disks"), ("vd", "virtualdetectors")):
                extra = sorted({k.removeprefix("geometry_" + prefix + "_") for k in packed
                                if k.startswith("geometry_" + prefix + "_") and k.removeprefix("geometry_") not in complete})
                if extra:
                    complete.update({prefix + "_" + k: v for k, v in export.columns(source, table, extra).items()})
            checked.append(production_compare.check_geometry(packed, complete))
    return checked


def legacy_pool_parity(current, saved):
    """Require the historical K8 pools and probabilities to match without a new draw."""
    names = ("eligible", "query_cell", "chosen_cell", "occupied", "library_cell", "counts", "size", "full_size",
             "exact_size", "supported", "fallback", "outside", "substituted", "event_row",
             "prob_empty", "prob_weak", "prob_energy", "pool_radius_delta", "pool_angle_deg")
    coordinate_dtypes = {}
    for name in names:
        if name not in current or name not in saved:
            raise ValueError("missing legacy pool field: " + name)
        a, b = np.asarray(current[name]), np.asarray(saved[name])
        if name in ("query_cell", "occupied") and a.dtype == np.dtype("int64") and b.dtype == np.dtype("float64"):
            bounds = np.array([1, 3, 2, 3])
            if (b.ndim != 2 or b.shape[1] != 4 or not np.all(np.isfinite(b)) or
                    np.any(b < 0) or np.any(b > bounds) or not np.array_equal(b, np.floor(b))):
                raise ValueError("invalid saved floating query-cell coordinates")
            coordinate_dtypes[name] = "saved float64 integer-valued bounded bin labels compared exactly as int64"
            b = b.astype(np.int64)
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            raise ValueError(f"historical deterministic pool mismatch: {name}; current {a.dtype}/{a.shape}, saved {b.dtype}/{b.shape}")
    return dict(fields=list(names), exact=True, coordinate_dtype_normalization=coordinate_dtypes, new_draw_performed=False)


def run(args):
    from . import support_audit

    source_audit._new(args.out)
    started = time.monotonic()
    inputs = source_overlap_report.Inputs()
    candidate = pinned_json(inputs, args.candidate / "summary.json", args.candidate_sha256)
    parents = pinned_json(inputs, args.parent_summary, args.parent_sha256)
    query_proof = pinned_json(inputs, args.query_proof, args.query_proof_sha256)
    legacy_report = pinned_json(inputs, args.legacy_report, args.legacy_report_sha256)
    saved_pools = pinned_npz(inputs, args.legacy_pools, args.legacy_pools_sha256)
    bound = parent_bundle(inputs, args.query_bound, parents)
    query_check = validate_query(args.tables, query_proof, bound)
    old_library_sha = legacy_report["inputs"]["library"]["sha256"]
    if (source_events.file_digest(args.legacy_library) != old_library_sha or
            query_proof["input_sha256"]["baseline"] != legacy_report["inputs"]["tables"]["sha256"]):
        raise ValueError("legacy/query baseline identity mismatch")
    if (candidate.get("candidate_library_constructed") is not True or candidate.get("sampled") is not False or
            candidate.get("legacy_donors_included") is not False or candidate.get("training_only") is not True or
            candidate.get("parent_summary_sha256") != args.parent_sha256):
        raise ValueError("candidate is not the declared unsampled training-only population")
    for name, digest in candidate["output_sha256"].items():
        if Path(name).name != name or source_events.file_digest(args.candidate / name) != source_audit._digest(digest):
            raise ValueError("candidate artifact hash mismatch")
        inputs.sha256[str((args.candidate / name).absolute())] = digest
    candidate_path = args.candidate / "library.npz"
    packed, metadata = library.load(candidate_path)
    saved_metadata = inputs.json(args.candidate / "metadata.json")
    if metadata != saved_metadata or any(candidate.get(key) != value for key, value in metadata.items()):
        raise ValueError("candidate embedded metadata differs from frozen receipt")
    old, _ = library.load(args.legacy_library)
    candidate_cells, candidate_coverage = coverage.frozen(packed)
    declared_cells = pinned_npz(inputs, args.candidate / "training_cells.npz",
                                candidate["output_sha256"]["training_cells.npz"])
    if set(candidate_cells) != set(declared_cells) or any(
            not np.array_equal(value, declared_cells[key]) for key, value in candidate_cells.items()):
        raise ValueError("candidate cells differ from frozen coverage")
    for key in ("front", "in_bins", "out_of_bins", "entry_disk", "highEinnerdisk1", "thresholds"):
        if candidate_coverage[key] != candidate["training"][key]:
            raise ValueError("candidate whole-event coverage differs from declared training")
    train_expected = pinned_npz(inputs, args.candidate / "training_expected.npz", candidate["output_sha256"]["training_expected.npz"])
    train_ledger = pinned_npz(inputs, args.candidate / "training_ledger.npz", candidate["output_sha256"]["training_ledger.npz"])
    donor = pinned_npz(inputs, args.candidate / "donor_provenance.npz", candidate["output_sha256"]["donor_provenance.npz"])
    original_expected = pinned_npz(inputs, args.parent_summary.parent / "training/expected.npz",
                                   parents["output_sha256"]["training/expected.npz"])
    original_ledger = pinned_npz(inputs, args.parent_summary.parent / "training/ledger.npz",
                                 parents["output_sha256"]["training/ledger.npz"])
    for actual, original in ((train_expected, original_expected), (train_ledger, original_ledger)):
        if set(actual) != set(original) or any(not np.array_equal(actual[k], original[k]) for k in original):
            raise ValueError("candidate training identities differ from completed parent audit")
    row = np.asarray(packed["entry_event_row"])
    n = len(row)
    if (row.dtype.kind not in "iu" or row.shape != (n,) or np.any(row < 0) or
            np.any(row >= len(train_ledger["event"])) or np.any(row[1:] <= row[:-1])):
        raise ValueError("invalid candidate global stored-event rows")
    for name in ("donor_row", "global_event_row", "file_ordinal", "file_event_row"):
        value = np.asarray(donor[name])
        if value.shape != (n,) or value.dtype.kind not in "iu" or np.any(value < 0):
            raise ValueError("invalid integer donor coordinates: " + name)
    fid = original_ledger["output_fid"]
    starts = np.r_[0, np.flatnonzero(fid[1:] != fid[:-1]) + 1]
    if len(np.unique(fid[starts])) != len(starts):
        raise ValueError("authenticated training files are not contiguous")
    sizes = np.diff(np.r_[starts, len(fid)])
    ordinal = np.repeat(np.arange(1, len(starts) + 1), sizes)
    local_row = np.arange(len(fid)) - np.repeat(starts, sizes)
    if (not np.array_equal(donor["file_ordinal"], ordinal[row]) or
            not np.array_equal(donor["file_event_row"], local_row[row])):
        raise ValueError("donor file coordinates differ from authenticated training boundaries")
    if not np.array_equal(donor["global_event_row"], packed["entry_event_row"]) or not np.array_equal(donor["donor_row"], np.arange(n)):
        raise ValueError("candidate donor provenance/order mismatch")
    for field in source_split.OUTPUT_KEY + source_split.PARENT_KEY:
        if not np.array_equal(donor[field], train_ledger[field][packed["entry_event_row"]]):
            raise ValueError("candidate donor/full-ledger identity mismatch")
    overlap = source_split.compare_ledgers(train_ledger, bound["ledger"], train_expected, bound["expected"])
    if overlap["shared_parent_events"] or overlap["shared_output_events"]:
        raise ValueError("candidate/query immediate-parent or output overlap")
    e, css, geometry, census, total = export.read(args.tables, input_mode="primary")
    _, query_coverage = coverage.census(e, css, census, total)
    geometry_fields = geometry_parity(args.tables, geometry, packed, old)
    new_report, new_arrays = support_audit.audit(packed, e, css)
    old_report, old_arrays = support_audit.audit(old, e, css)
    parity = legacy_pool_parity(old_arrays, saved_pools)
    for path, sha in ((args.tables, query_proof["input_sha256"]["tables"]), (args.legacy_library, old_library_sha),
                      (candidate_path, candidate["output_sha256"]["library.npz"])):
        if source_events.file_digest(path) != sha:
            raise ValueError("input changed during support audit")
    inputs.unchanged()
    result = dict(candidate=new_report, legacy=old_report, query_census=census, query_coverage=query_coverage,
                  query_source_check=query_check, legacy_pool_parity=parity, geometry_fields=geometry_fields,
                  candidate_query_overlap=overlap, sampling_performed=False, reconstruction_performed=False,
                  source_independence_certified=False,
                  limitations=["Conditional fixed-library eligible-pool statistics, not reconstruction or inclusive-acceptance closure.",
                               "Selected production and legacy unfiltered pilot differ in selection/runtime; not a sample-size-only effect.",
                               "Development/pilot has 165 shared parents, 40 among frozen donors; file0/full-pilot overlap was zero.",
                               "No original-proton independence; no confirmation inputs read; all donor zeros retained."],
                  input_sha256=dict(inputs.sha256, query_tables=query_proof["input_sha256"]["tables"],
                                    candidate_library=candidate["output_sha256"]["library.npz"], legacy_library=old_library_sha),
                  source_sha256={m.__name__: source_events.file_digest(m.__file__) for m in
                                 (support_audit, library, export, coverage, source_events, source_split)},
                  elapsed_seconds=time.monotonic() - started)
    result["source_sha256"][__name__] = source_events.file_digest(__file__)
    source_audit._text(result)
    args.out.mkdir(parents=True)
    for name, arrays in (("candidate_pools.npz", new_arrays), ("legacy_pools.npz", old_arrays)):
        with (args.out / name).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    result["output_sha256"] = {name: source_events.file_digest(args.out / name)
                               for name in ("candidate_pools.npz", "legacy_pools.npz")}
    source_audit._save_json(args.out / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "parent-summary", "query-proof", "legacy-report", "legacy-pools", "query-bound",
                 "tables", "legacy-library", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("candidate", "parent", "query-proof", "legacy-report", "legacy-pools"):
        parser.add_argument("--" + name + "-sha256", required=True)
    result = run(parser.parse_args(argv))
    print(source_audit._text({name: result[name] for name in ("query_census", "legacy_pool_parity", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
