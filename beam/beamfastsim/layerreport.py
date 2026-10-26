"""Inspect persisted first-foil histories and saved slice arms without rerunning transport."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import g4ref, layerstudy as ls
from .reference_audit import _identity
from .slicestudy import verify_hashes


def inspect(history, frozen, g, arms):
    """Vectorized cohort selection; small output rows are outcome-conditioned diagnostics only."""
    rows = frozen["row"]
    index = frozen["history_index"]
    missing = (frozen["energy_s4"][-1] >= .1) & frozen["range_stop"]
    same_entrants = rows[missing]
    records = index[missing]
    entry = frozen["entry_MeV"]
    length = frozen["length_mm"]
    table = frozen["range_energy"], frozen["range_mm"]
    own_range = np.interp(entry, *table)
    sub_point2 = float(np.interp(.2, *table))
    out_coarse, stop_coarse = ls.range_out(entry, length, table)
    out_fine, stop_fine = ls.range_out(entry, length, ls.range_table(table[0][-1], points=65537))
    if not np.array_equal(stop_coarse, frozen["range_stop"]):
        raise ValueError("saved frozen range flags differ from current calculation")
    slice_energy = ls.kinetic(history["slice_p_in"][index])
    sampled_loss = history["slice_loss_MeV"][index]
    mean_loss = ls.eloss.dedx_restricted(ls.geometry.STOPPING_TARGET_AL,
                                        history["slice_p_in"][index], 1.0) * length[:, None] / 4
    actual_fraction = sampled_loss / np.maximum(slice_energy, 1e-300)
    active = slice_energy >= .1
    row_report = dict(row=same_entrants.tolist(),
                      ids=np.column_stack([g[k][same_entrants] for k in ("run", "subrun", "evt", "trk")]).tolist(),
                      g4_foil=g["foil"][same_entrants].tolist(),
                      entry_MeV=entry[missing].tolist(), path_mm=length[missing].tolist(),
                      same_law_range_mm=own_range[missing].tolist(),
                      abs_uz=np.abs(history["u_in"][records, 2]).tolist(),
                      sampled_slice_entry_MeV=slice_energy[missing].tolist(),
                      sampled_slice_loss_MeV=sampled_loss[missing].tolist(),
                      mean_loss_at_sampled_energy_MeV=mean_loss[missing].tolist(),
                      observed_exit_MeV=ls.kinetic(history["p_out"][records]).tolist(),
                      deterministic_energy_MeV={str(s): frozen[f"energy_s{s}"][:, missing].T.tolist()
                                                for s in (4, 32, 128)},
                      saved_arm_foil={str(s): arms[s]["foil"][same_entrants].tolist() for s in arms},
                      saved_arm_fate={str(s): arms[s]["fate"][same_entrants].tolist() for s in arms})
    all_g4_first = g["foil"] == 0
    entrant_mask = np.zeros(len(g["evt"]), bool)
    entrant_mask[rows] = True
    report = dict(entrants=len(rows), n_mean4_understop=int(missing.sum()),
                  range_grid_stop_disagreements=int(np.sum(stop_coarse != stop_fine)),
                  range_grid_max_abs_exit_MeV=float(np.max(np.abs(out_coarse - out_fine))),
                  understop_g4_first=int(np.sum(g["foil"][same_entrants] == 0)),
                  understop_sampled_exit_MeV_quantiles=ls.quantiles(ls.kinetic(history["p_out"][records])),
                  understop_last_slice_fraction_quantiles=ls.quantiles(actual_fraction[missing, -1]),
                  understop_rows=row_report, g4_first_not_entering_rows=np.flatnonzero(all_g4_first & ~entrant_mask).tolist(),
                  understop_entry_MeV_quantiles=ls.quantiles(entry[missing]),
                  understop_path_minus_range_mm_quantiles=ls.quantiles((length - own_range)[missing]),
                  sampled_slices_below_point2=int(np.sum((slice_energy >= .1) & (slice_energy < .2))),
                  sampled_slices_above_10pct_loss=int(np.sum(active & (actual_fraction > .1))),
                  sampled_slices_above_50pct_loss=int(np.sum(active & (actual_fraction > .5))),
                  mean4_understops_saved_arms={str(s): int(np.sum(arms[s]["foil"][same_entrants] == 0)) for s in arms},
                  total_first_foil_saved_arms={str(s): int(np.sum(arms[s]["foil"] == 0)) for s in arms},
                  same_law_residual_range_0p1_to_0p2_mm=sub_point2,
                  threshold_sensitivity={})
    for threshold in (.1, .2):
        offset = np.interp(threshold, *table)
        paths = np.array([float(length.min()), float(np.median(length)), float(length.max())])
        report["threshold_sensitivity"][str(threshold)] = dict(
            paths_mm=paths.tolist(),
            critical_entry_MeV=np.interp(paths + offset, table[1], table[0]).tolist(),
            frozen_range_stops=int(np.sum((entry < threshold) | (own_range - offset < length))))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True)
    ap.add_argument("--slices", required=True, help="saved 16/32-slice extension directory")
    ap.add_argument("--out", required=True, help="new report directory")
    a = ap.parse_args(argv)
    study, slices = Path(a.study), Path(a.slices)
    manifest = json.loads((study / "manifest.json").read_text())
    summary = json.loads((study / "summary.json").read_text())
    if not summary["complete"] or not all(summary["replay_exact"].values()):
        raise ValueError("completed exact replay required")
    input_path = Path(manifest["input"]["path"])
    verify_hashes([manifest["input"]], [_identity(input_path)], "input")
    baseline = Path(manifest["baseline_manifest"]["path"]).parent
    extension = json.loads((slices / "manifest.json").read_text())
    extension_summary = json.loads((slices / "summary.json").read_text())
    verify_hashes([manifest["input"]], [extension["input"]], "extension input")
    verify_hashes(manifest["physics"], extension["physics"], "extension beam physics")
    for key in ("core", "geometry", "field"):
        verify_hashes(manifest[key], extension[key], f"extension {key}")
    if not extension_summary["complete"]:
        raise ValueError("incomplete slice extension")
    arm_paths = {4: baseline / f"nominal_seed{manifest['seed']}.npz",
                 8: baseline / f"slices8_seed{manifest['seed']}.npz"}
    arm_paths.update({s: slices / f"slices{s}_seed{manifest['seed']}.npz" for s in (16, 32)})
    verify_hashes([manifest["reused_arm"]], [_identity(arm_paths[4])], "observed arm")
    g = g4ref.load(input_path)
    arms = {s: g4ref.load(p) for s, p in arm_paths.items()}
    reused = {Path(p["path"]).name: p for p in extension["reused_arms"]}
    for s, arm in arms.items():
        meta = json.loads(str(arm["meta"]))
        expected = dict(ds=5.0, n_slices=s, tcut=1.0, max_steps=3000,
                        z_up=4100.0, r_wall=950.0, msc=True, decay=True)
        if len(arm["fate"]) != len(g["evt"]) or meta["params"] != expected:
            raise ValueError("arm length or parameter mismatch")
        if s in (4, 8):
            verify_hashes([reused[arm_paths[s].name]], [_identity(arm_paths[s])], "extension reused arm")
        elif meta != extension_summary["runs"][arm_paths[s].stem]["meta"]:
            raise ValueError("extension output metadata mismatch")
    history_path, frozen_path = study / "layer_history.npz", study / "frozen_first_foil.npz"
    report = inspect(g4ref.load(history_path), g4ref.load(frozen_path), g, arms)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "manifest.json").write_text(json.dumps(dict(
        study=_identity(study / "manifest.json"), summary=_identity(study / "summary.json"),
        extension=_identity(slices / "manifest.json"), history=_identity(history_path), frozen=_identity(frozen_path),
        arms={str(s): _identity(p) for s, p in arm_paths.items()}, code=_identity(__file__),
        layerstudy_code=_identity(ls.__file__),
        note="Outcome-conditioned histories; no new transport, no independent G4 entry histories"), indent=2))
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != "understop_rows"}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
