"""Replay saved raw crystal steps against ordered stock showers without transport or proximity matching."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import uproot

from calofastsim.compression import compress, lookup
from calofastsim.identities import EVENT_FIELDS, keys
from calofastsim.ancestry import SIM_FIELDS


QUANTITIES = ("n_compress", "energy_g4", "energy_birks", "time", "momentum_in", "position_x", "position_y", "position_z")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration(path):
    text = path.read_text()
    blocks = re.findall(r"\bCaloShowerStepMaker\s*:\s*\{([^{}]*)\}", text)
    if len(blocks) != 1:
        raise ValueError("one fully expanded flat CaloShowerStepMaker block required")
    block = blocks[0]
    if not re.search(r"\bcompressData\s*:\s*true\b", block):
        raise ValueError("only compressData=true is supported")
    tags = re.findall(r'\bcaloStepPointCollection\s*:\s*\[\s*"([^"\n]+)"\s*\]', block)
    if tags != ["g4run:calorimeter"]:
        raise ValueError("only the explicit single g4run:calorimeter input is supported")
    result = {}
    for source, target in (("numZSlices", "num_z_slices"), ("deltaTime", "delta_time"), ("eDepThreshold", "energy_threshold")):
        found = re.findall(r"\b" + source + r"\s*:\s*([^\s]+)", block)
        if len(found) != 1:
            raise ValueError(f"one resolved {source} required")
        result[target] = int(found[0]) if source == "numZSlices" else float(found[0])
    return result


def compare(predicted, saved):
    layout = (len(predicted["event"]) == len(saved["event"]) and all(
        np.array_equal(predicted[name], saved[name]) for name in SIM_FIELDS + ("crystal",)))
    report = {"ordered_persistent_identity_and_crystal_equal": layout, "quantities": {}}
    if layout:
        for name in QUANTITIES:
            left, right = predicted[name], saved[name]
            diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
            report["quantities"][name] = {"exact_equal": bool(np.array_equal(left, right)),
                                         "mismatches": int(np.count_nonzero(left != right)),
                                         "max_abs_difference": float(np.max(diff, initial=0))}
        mismatch = predicted["n_compress"] != saved["n_compress"]
        report["count_mismatch_rows"] = {
            "predicted": {name: predicted[name][mismatch].tolist() for name in SIM_FIELDS + ("crystal",) + QUANTITIES + ("bucket", "generation", "flush_is_final")},
            "predicted_flush_trigger_time": np.where(np.isfinite(predicted["flush_trigger_time"][mismatch]), predicted["flush_trigger_time"][mismatch], -1).tolist(),
            "saved": {name: saved[name][mismatch].tolist() for name in SIM_FIELDS + ("crystal",) + QUANTITIES},
        }
        canonical_fields = SIM_FIELDS + ("crystal", "n_compress", "time", "position_z", "position_x", "position_y")
        canonical = {}
        for name in canonical_fields:
            dtype = np.int64 if name in SIM_FIELDS + ("crystal", "n_compress") else np.float64
            canonical[name] = np.asarray(predicted[name], dtype=dtype)
        left_order = np.lexsort(tuple(canonical[name] for name in reversed(canonical_fields)))
        right_order = np.lexsort(tuple(saved[name] for name in reversed(canonical_fields)))
        report["canonical_without_row_order"] = {}
        for name in QUANTITIES:
            diff = np.abs(predicted[name][left_order].astype(np.float64) - saved[name][right_order].astype(np.float64))
            report["canonical_without_row_order"][name] = {"mismatches": int(np.count_nonzero(diff)), "max_abs_difference": float(np.max(diff, initial=0))}
        canonical_identity_equal = all(np.array_equal(predicted[name][left_order], saved[name][right_order]) for name in SIM_FIELDS + ("crystal",))
        report["canonical_persistent_identity_and_crystal_equal"] = canonical_identity_equal
        report["canonical_stored_scalar_fields_exact"] = bool(canonical_identity_equal and all(
            report["canonical_without_row_order"][name]["mismatches"] == 0 for name in ("n_compress", "energy_g4", "energy_birks", "time", "momentum_in")))
        report["position_tolerance_mm"] = 1e-10
        report["canonical_positions_within_tolerance"] = bool(canonical_identity_equal and all(
            report["canonical_without_row_order"][name]["max_abs_difference"] <= 1e-10 for name in ("position_x", "position_y", "position_z")))
        trigger = predicted["flush_trigger_time"]
        tied_trigger_fields = SIM_FIELDS + ("crystal",)
        trigger_keys = np.rec.fromarrays([predicted[name].astype(np.int64) for name in tied_trigger_fields] + [trigger], names=tied_trigger_fields + ("flush_trigger_time",))
        _, inverse_trigger, trigger_counts = np.unique(trigger_keys, return_inverse=True, return_counts=True)
        report["count_mismatch_rows_in_equal_time_nonfinal_flush_blocks"] = int(np.count_nonzero(
            mismatch & np.isfinite(trigger) & (trigger_counts[inverse_trigger] > 1)))
        report["count_mismatch_flush_triggers_all_finite"] = bool(np.all(np.isfinite(trigger[mismatch])))
        report["canonical_comparison_is_not_an_identity_join"] = True
        for name in ("position_x", "position_y", "position_z"):
            left, right = predicted[name], saved[name]
            diff = np.abs(left - right)
            matched = ~mismatch
            report["quantities"][name]["count_matched_max_abs_difference"] = float(np.max(diff[matched], initial=0))
    report["equal_time_order_unspecified_rows"] = int(np.count_nonzero(predicted["equal_time_order_unspecified"]))
    report["ordered_layout_counts_exact"] = bool(layout and report["quantities"]["n_compress"]["exact_equal"])
    report["all_numeric_fields_exact"] = bool(layout and all(row["exact_equal"] for row in report["quantities"].values()))
    report["bitwise_row_replay_certified"] = bool(report["all_numeric_fields_exact"] and not report["equal_time_order_unspecified_rows"])
    report["row_multiset_replay_within_tolerance"] = bool(report.get("canonical_stored_scalar_fields_exact", False)
                                                         and report.get("canonical_positions_within_tolerance", False))
    report["constituent_identity_certified"] = False
    report["complete_incident_census"] = False
    report["zero_response_certified"] = False
    report["scope"] = "source-algorithm replay; chosen equal-time convention is not C++ sort order; no independent stock constituent links exist"
    return report


def check(path, expanded_fcl, *, expected_events, crystal_length, membership_out=None, equal_time_order="stable"):
    started = time.perf_counter()
    with uproot.open(path) as source:
        states = tuple(f"{prefix}_{axis}" for prefix in ("start", "end") for axis in "xyz")
        fields = {
            "events": EVENT_FIELDS + ("g4_status",),
            "particles": SIM_FIELDS + ("parent_valid", "parent_product_id", "parent_key", "pdg", "creation", "end_time") + states,
            "steps": SIM_FIELDS + ("product_id", "index", "volume", "kind", "time", "energy_g4", "energy_visible")
                     + tuple(f"{prefix}_{axis}" for prefix in ("position", "momentum") for axis in "xyz"),
            "showers": SIM_FIELDS + ("index", "crystal") + QUANTITIES,
            "crystals": ("run", "crystal", "disk") + tuple(f"{prefix}_{axis}" for prefix in ("local_position", "size") for axis in "xyz"),
            "disks": ("run", "disk") + tuple(f"{prefix}_{axis}" for prefix in ("origin", "front_face_origin", "size", "axis_x", "axis_y", "axis_z") for axis in "xyz"),
        }
        tables = {name: source[f"dump/{name}"].arrays(names, library="np") for name, names in fields.items()}
    events = tables["events"]
    if len(events["event"]) != expected_events or np.any(events["g4_status"] != 0):
        raise ValueError("expected completed status-zero event population required")
    raw = {name: value[tables["steps"]["kind"] == 1] for name, value in tables["steps"].items()}
    config = configuration(expanded_fcl)
    predicted = compress(tables["particles"], raw, tables["crystals"], tables["disks"], crystal_length=crystal_length,
                         equal_time_order=equal_time_order, **config)
    report = compare(predicted, tables["showers"])
    raw_slot = lookup(keys(tables["particles"], SIM_FIELDS), keys(raw, SIM_FIELDS))
    late_order = np.argsort(raw["time"])[-10:][::-1]
    report["largest_raw_time_rows"] = {name: raw[name][late_order].tolist() for name in EVENT_FIELDS + ("product_id", "index", "sim_product_id", "sim_key", "volume", "time", "energy_g4")}
    report["largest_raw_time_particle_pdg"] = tables["particles"]["pdg"][raw_slot[late_order]].tolist()
    report["largest_raw_time_particle_creation"] = tables["particles"]["creation"][raw_slot[late_order]].tolist()
    report["largest_raw_time_particle_end_time"] = tables["particles"]["end_time"][raw_slot[late_order]].tolist()
    report["raw_times_above_float32_exact_integer_ns"] = int(np.count_nonzero(raw["time"] > 2**24))
    report["raw_time_threshold_is_diagnostic_not_selection"] = True
    report.update(events=expected_events, raw_crystal_rows=len(raw["event"]), saved_showers=len(tables["showers"]["event"]),
                  predicted_showers=len(predicted["event"]), excluded_raw_rows=len(predicted["excluded_raw_index"]),
                  configuration=config, equal_time_order=equal_time_order, crystal_length_mm=crystal_length,
                  expanded_fcl_sha256=sha256(expanded_fcl), input_sha256=sha256(path))
    if membership_out is not None:
        member = predicted["member_index"]
        arrays = {f"raw_{name}": raw[name][member] for name in EVENT_FIELDS + ("product_id", "index")}
        arrays.update({f"predicted_{name}": predicted[name] for name in SIM_FIELDS + ("crystal",) + QUANTITIES})
        arrays["member_offsets"] = predicted["member_offsets"]
        arrays["excluded_raw_index"] = predicted["excluded_raw_index"]
        for name in ("equal_time_order_unspecified", "bucket", "generation", "flush_is_final", "flush_trigger_time"):
            arrays[name] = predicted[name]
        arrays["provenance_json"] = np.asarray(json.dumps(report, sort_keys=True, allow_nan=False))
        with membership_out.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report["diagnostic_membership_sha256"] = sha256(membership_out)
    report["measured_check_wall_seconds"] = time.perf_counter() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expanded-fcl", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--crystal-length", type=float, required=True)
    parser.add_argument("--membership-out", type=Path)
    parser.add_argument("--equal-time-order", choices=("stable", "reverse"), default="stable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or (args.membership_out is not None and args.membership_out.exists()):
        raise SystemExit("refuse to overwrite existing replay evidence")
    result = check(args.input, args.expanded_fcl, expected_events=args.expected_events,
                   crystal_length=args.crystal_length, membership_out=args.membership_out, equal_time_order=args.equal_time_order)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["ordered_layout_counts_exact"]:
        raise SystemExit("ordered shower identities/counts did not pass")


if __name__ == "__main__":
    main()
