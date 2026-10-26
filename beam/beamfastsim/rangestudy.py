"""Diagnostic bounded mean-loss integration on saved entries, with release-produced G4 range references.

No transport or stochastic baseline is rerun. All entrance/outcome selections are diagnostics. Restricted G4
continuous loss alone excludes discrete delta rays; it is not a complete replacement for total muon loss.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import g4ref, layerstudy as ls, physics_tables as pt
from .adaptive_loss import adaptive_mean_loss
from .reference_audit import _identity
from .slicestudy import CORE_FILES, verify_hashes

FRACTIONS = (.1, .05, .02, .01, .005, .002, .001)


def compare_integrators(frozen, tables, info, fractions=FRACTIONS):
    energy, length = frozen["entry_MeV"], frozen["length_mm"]
    if len(energy) == 0:
        raise ValueError("nonempty frozen-entry population required")
    maximum = max(float(energy.max()) * 1.01, 10.)
    same = ls.range_table(maximum, points=65537)
    matched = ls.range_table(maximum, points=65537, tcut=info["energy_cuts_MeV"]["e-"])
    g4_full = pt.integrated_range(tables, "DEDXnr", maximum)
    g4_restricted = pt.integrated_range(tables, "DEDX", maximum)
    model_tables = dict(fast=same, fast_cutmatched=matched, g4_unrestricted=g4_full, g4_restricted=g4_restricted)
    store = dict(row=frozen["row"], entry_MeV=energy, length_mm=length)
    report = dict(entrants=len(energy), models={}, adaptive={}, critical_entry_MeV={}, fraction_list=list(fractions))
    for name, table in model_tables.items():
        outgoing, stopped = ls.range_out(energy, length, table)
        store[f"{name}_energy"] = table[0]
        store[f"{name}_range_mm"] = table[1]
        store[f"{name}_out_MeV"] = outgoing
        store[f"{name}_stop"] = stopped
        report["models"][name] = dict(stops=int(stopped.sum()),
            stop_rows=frozen["row"][stopped].tolist(), outgoing_MeV_quantiles=ls.quantiles(outgoing[~stopped]))
    paths = np.array([.1056, .2, .4, 1.])
    report["critical_paths_mm"] = paths.tolist()
    for name, table in model_tables.items():
        if np.any(paths >= table[1][-1]):
            raise ValueError("critical paths outside numerical range domain")
        report["critical_entry_MeV"][name] = np.interp(paths, table[1], table[0]).tolist()
    report["g4_native_range_check"] = {}
    for name, vector in (("g4_unrestricted", "CSDARange"), ("g4_restricted", "Range")):
        native = pt.interpolate(tables[vector], np.maximum(energy, .1)) - float(pt.interpolate(tables[vector], .1))
        native_stop = (energy < .1) | (native < length)
        report["g4_native_range_check"][name] = dict(stops=int(native_stop.sum()),
            disagree_integral=int(np.sum(native_stop != store[f"{name}_stop"])),
            max_abs_range_mm=float(np.max(np.abs(native - np.interp(energy, *model_tables[name])))))
    for name in ("fast", "g4_unrestricted"):
        result = {}
        power = None if name == "fast" else pt.interpolator(tables["DEDXnr"])
        reference, stopped = store[f"{name}_out_MeV"], store[f"{name}_stop"]
        for fraction in fractions:
            started = time.monotonic()
            output = adaptive_mean_loss(energy, length, fraction, stopping=power)
            if np.any(output.capped):
                raise ValueError(f"adaptive diagnostic capped: {name}, fraction={fraction}")
            tag = f"{name}_frac{fraction:g}"
            for field, value in output._asdict().items():
                store[f"{tag}_{field}"] = value
            both_survive = ~stopped & ~output.stopped
            common_stop = stopped & output.stopped
            expected_range = np.interp(energy, *model_tables[name])
            result[f"{fraction:g}"] = dict(stops=int(output.stopped.sum()),
                disagree_integral=int(np.sum(output.stopped != stopped)),
                understop=int(np.sum(stopped & ~output.stopped)), overstop=int(np.sum(~stopped & output.stopped)),
                steps_quantiles=ls.quantiles(output.steps), total_steps=int(output.steps.sum()),
                maximum_fraction=float(np.max(output.max_requested_fraction)),
                survivor_error_MeV_quantiles=ls.quantiles((output.energy - reference)[both_survive]),
                stop_path_error_mm_quantiles=ls.quantiles((output.consumed_length - expected_range)[common_stop]),
                understop_rows=frozen["row"][stopped & ~output.stopped].tolist(), wall_s=time.monotonic() - started)
        report["adaptive"][name] = result
    fast_stop, g4_stop = store["fast_stop"], store["g4_unrestricted_stop"]
    report["fast_vs_g4_integral"] = dict(fast_only_rows=frozen["row"][fast_stop & ~g4_stop].tolist(),
                                         g4_only_rows=frozen["row"][~fast_stop & g4_stop].tolist())
    return report, store


def negative_entries(frozen, g):
    """Select mu- explicitly; the cached source accepts both charges and must not share a mu- table."""
    pdg = g["pdg"][frozen["row"]]
    if np.any(~np.isin(pdg, [13, -13])):
        raise ValueError("non-muon frozen entry")
    selected = pdg == 13
    result = {k: frozen[k][selected] for k in ("row", "entry_MeV", "length_mm")}
    info = dict(all_entrants=len(pdg), negative=int(selected.sum()), positive_excluded=int((~selected).sum()),
                positive_rows=frozen["row"][~selected].tolist(),
                positive_g4_first=int(np.sum(g["foil"][frozen["row"][~selected]] == 0)))
    return result, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer-report", required=True, help="final saved-history report with input hashes")
    ap.add_argument("--table-report", required=True, help="completed physics_tables report")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    layer = Path(a.layer_report)
    tables_root = Path(a.table_report)
    lm = json.loads((layer / "manifest.json").read_text())
    tm = json.loads((tables_root / "manifest.json").read_text())
    table_summary = json.loads((tables_root / "summary.json").read_text())
    for key in ("study", "summary", "history", "frozen"):
        verify_hashes([lm[key]], [_identity(lm[key]["path"])], f"saved layer {key}")
    study = json.loads(Path(lm["study"]["path"]).read_text())
    core_dir = Path(__import__("mu2efastsim").__file__).resolve().parent
    verify_hashes(study["core"], [_identity(core_dir / p) for p in CORE_FILES], "core physics")
    verify_hashes([study["input"]], [_identity(study["input"]["path"])], "selected input")
    for artifact in tm["artifacts"]:
        verify_hashes([artifact], [_identity(artifact["path"])], "export artifact")
    export_dir = Path(next(p["path"] for p in tm["artifacts"] if Path(p["path"]).name == "couple.dat")).parent
    tables, info, _ = pt.load_target(export_dir)
    if info != table_summary["couple"]:
        raise ValueError("table-report couple mismatch")
    frozen = g4ref.load(lm["frozen"]["path"])
    g = g4ref.load(study["input"]["path"])
    frozen, charges = negative_entries(frozen, g)
    report, store = compare_integrators(frozen, tables, info)
    report["charge_selection"] = charges
    report["export_grid_stop_check"] = {}
    compare_couples = [p for p in tm["comparisons"] if Path(p["path"]).name == "couple.dat"]
    if compare_couples:
        for artifact in tm["comparisons"]:
            verify_hashes([artifact], [_identity(artifact["path"])], "comparison export artifact")
        coarse, coarse_info, _ = pt.load_target(Path(compare_couples[0]["path"]).parent)
        if info != coarse_info:
            raise ValueError("comparison material/cuts mismatch")
        for name, column in (("g4_unrestricted", "DEDXnr"), ("g4_restricted", "DEDX")):
            alternate = pt.integrated_range(coarse, column, maximum=store[f"{name}_energy"][-1])
            other_energy, other_stop = ls.range_out(frozen["entry_MeV"], frozen["length_mm"], alternate)
            report["export_grid_stop_check"][name] = dict(
                stops=int(other_stop.sum()), disagree=int(np.sum(other_stop != store[f"{name}_stop"])),
                max_abs_exit_MeV=float(np.max(np.abs(other_energy - store[f"{name}_out_MeV"]))))
    truth = g["foil"][frozen["row"]] == 0
    report["g4_event_identity_comparison"] = {}
    for name in report["models"]:
        predicted = store[f"{name}_stop"]
        report["g4_event_identity_comparison"][name] = dict(
            both=int(np.sum(predicted & truth)), predicted_only=int(np.sum(predicted & ~truth)),
            g4_event_only=int(np.sum(~predicted & truth)), neither=int(np.sum(~predicted & ~truth)))
    report["entrant_pdg"] = 13
    report["g4_event_first_foil_stops"] = int(np.sum((g["foil"] == 0) & (g["pdg"] == 13)))
    report["g4_event_first_foil_stops_among_entries"] = int(np.sum(g["foil"][frozen["row"]] == 0))
    report["scope"] = "Deterministic frozen-entry model/integration diagnostics, not stochastic stop-rate closure"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "range_comparisons.npz", **store)
    (out / "manifest.json").write_text(json.dumps(dict(
        layer_report=_identity(layer / "manifest.json"), table_report=_identity(tables_root / "manifest.json"),
        table_summary=_identity(tables_root / "summary.json"), frozen=lm["frozen"], input=study["input"],
        core=study["core"], code=[_identity(p) for p in (Path(__file__), Path(ls.__file__), Path(pt.__file__),
                                                      Path(__file__).with_name("adaptive_loss.py"))]), indent=2))
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    brief = {k: v for k, v in report.items() if k not in ("models", "adaptive")}
    brief["models"] = {k: r["stops"] for k, r in report["models"].items()}
    brief["adaptive"] = {name: {fraction: {k: result[k] for k in
        ("stops", "disagree_integral", "total_steps", "steps_quantiles", "survivor_error_MeV_quantiles")}
        for fraction, result in series.items()} for name, series in report["adaptive"].items()}
    print(json.dumps(brief, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
