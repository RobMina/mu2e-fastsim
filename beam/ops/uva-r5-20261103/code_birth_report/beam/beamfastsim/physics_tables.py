"""Read release-produced Geant4 ASCII loss/range tables with explicit material/cuts-couple mapping.

The formats/units follow Geant4 11.3.2 G4Physics{Table,Vector} and G4ProductionCutsTable. Exported native range
includes a below-grid tail; threshold-relative comparisons subtract R(threshold), never use the full range.
Spline reconstruction uses SciPy not-a-knot cubic splines in energy, not a bitwise port of G4's spline routine.
Dense-export refinement and native-range closure bound interpolation effects on the checked domain; source-grid
values are exact reads, while between-node queries are reconstructed diagnostic values.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from mu2efastsim import eloss
from . import geometry
from .reference_audit import _identity

TABLES = ("Ionisation", "DEDX", "DEDXnr", "Range", "CSDARange")


def read_vectors(path):
    """Read all ASCII physics vectors without assuming a fixed number of grid points."""
    tokens = Path(path).read_text().split()
    try:
        count, at, vectors = int(tokens[0]), 1, []
        if count <= 0:
            raise ValueError("empty physics table")
        for _ in range(count):
            kind = int(tokens[at])
            low, high, nodes = float(tokens[at + 1]), float(tokens[at + 2]), int(tokens[at + 3])
            size = int(tokens[at + 4])
            at += 5
            if nodes != size or size < 0 or kind not in (0, 1, 2, 3, 4, 5, 6):
                raise ValueError("invalid physics-vector header")
            data = np.asarray(tokens[at:at + 2 * size], float).reshape(size, 2)
            at += 2 * size
            if size and (not np.all(np.isfinite(data)) or np.any(np.diff(data[:, 0]) <= 0)):
                raise ValueError("nonfinite or unordered physics vector")
            if size and not np.allclose(data[[0, -1], 0], [low, high], rtol=1e-9, atol=0):
                raise ValueError("physics-vector endpoints disagree")
            vectors.append(dict(kind=kind, energy=data[:, 0], values=data[:, 1]))
        if at != len(tokens):
            raise ValueError("trailing physics-table data")
    except (IndexError, TypeError) as exc:
        raise ValueError("malformed ASCII physics table") from exc
    return vectors


def read_couples(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 2 or lines[0] != "COUPLE-V3.0":
        raise ValueError("unsupported couple-table format")
    n = int(lines[1])
    if len(lines) != 2 + 7 * n or n <= 0:
        raise ValueError("couple-table size mismatch")
    result = []
    for i in range(n):
        block = lines[2 + 7 * i:9 + 7 * i]
        if int(block[0]) != i:
            raise ValueError("nonsequential material-cuts indices")
        result.append(dict(index=i, material=block[1], region=block[2], range_cut_mm=np.asarray(block[3:], float)))
    return result


def read_cuts(path, count):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2 or lines[0].strip() != "CUT-V3.0" or int(lines[1]) != count:
        raise ValueError("cut-table header/count mismatch")
    values = np.asarray(" ".join(lines[2:]).split(), dtype=float)
    if len(values) != 4 * count * 2 or not np.all(np.isfinite(values)):
        raise ValueError("cut-table payload mismatch")
    values = values.reshape(4, count, 2)
    return values[:, :, 0].T, values[:, :, 1].T * .001


def read_materials(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2 or lines[0].strip() != "MATERIAL-V3.0" or len(lines) != int(lines[1]) + 2:
        raise ValueError("unsupported material-table format")
    result = {}
    for line in lines[2:]:
        name, density = line.split()
        if name in result:
            raise ValueError("duplicate material name")
        result[name] = float(density)
        if not np.isfinite(result[name]) or result[name] <= 0:
            raise ValueError("material density must be finite and positive")
    return result


def load_target(directory, material="StoppingTarget_Al", region="DefaultRegionForTheWorld", particle="mu-"):
    directory = Path(directory)
    couples = read_couples(directory / "couple.dat")
    cuts, energies = read_cuts(directory / "cut.dat", len(couples))
    densities = read_materials(directory / "material.dat")
    candidates = [c for c in couples if c["material"] == material and c["region"] == region]
    if len(candidates) != 1:
        raise ValueError("unique named material/region couple required")
    couple = candidates[0]
    index = couple["index"]
    if not np.array_equal(couple["range_cut_mm"], cuts[index]):
        raise ValueError("couple/cut range mismatch")
    tables = {}
    paths = [directory / name for name in ("couple.dat", "cut.dat", "material.dat")]
    for name in TABLES:
        path = directory / f"{name}.muIoni.{particle}.asc"
        paths.append(path)
        vectors = read_vectors(path)
        if len(vectors) != len(couples):
            raise ValueError("physics/couple table count mismatch")
        vector = vectors[index]
        if len(vector["energy"]) < 5 or np.any(vector["values"] <= 0):
            raise ValueError("selected material has empty/nonpositive table; base-material mapping must be resolved")
        tables[name] = vector
    info = dict(index=index, material=material, region=region, density_g_cm3=densities[material], particle=particle,
                range_cuts_mm=dict(zip(("gamma", "e-", "e+", "proton"), cuts[index].tolist())),
                energy_cuts_MeV=dict(zip(("gamma", "e-", "e+", "proton"), energies[index].tolist())))
    return tables, info, paths


def interpolator(vector):
    """Construct a reusable checked SciPy interpolation, not a G4 implementation-parity claim."""
    grid, values = vector["energy"], vector["values"]
    spline = CubicSpline(grid, values, extrapolate=False)

    def evaluate(energy):
        energy = np.asarray(energy)
        if np.any(~np.isfinite(energy)) or np.any(energy < grid[0]) or np.any(energy > grid[-1]):
            raise ValueError("query outside physics-vector domain")
        result = spline(energy)
        if np.any(~np.isfinite(result)) or np.any(result <= 0):
            raise ValueError("nonpositive or nonfinite reconstructed physics table")
        return result

    return evaluate


def interpolate(vector, energy):
    return interpolator(vector)(energy)


def integrated_range(tables, name="DEDXnr", maximum=50., threshold=.1, points=65537):
    grid = np.geomspace(threshold, maximum, points)
    power = interpolate(tables[name], grid)
    return grid, cumulative_trapezoid(1 / power, grid, initial=0)


def summarize(tables, info):
    energy = np.array([.1, .15, .199, .2, .201, .3, .5, 1., 1.5, 2., 3., 5., 10.])
    p = np.sqrt(energy * (energy + 2 * eloss.MMU))
    sampled = {name: interpolate(vector, energy) for name, vector in tables.items()}
    fast = eloss.dedx_restricted(geometry.STOPPING_TARGET_AL, p, 1.0)
    cutmatched = eloss.dedx_restricted(geometry.STOPPING_TARGET_AL, p, info["energy_cuts_MeV"]["e-"])
    report = dict(couple=info, energy_MeV=energy.tolist(),
                  values={name: value.tolist() for name, value in sampled.items()},
                  units="DEDX/Ionisation/DEDXnr MeV/mm; Range/CSDARange mm; queried energy MeV",
                  fast_dedx_MeV_mm=fast.tolist(), fast_with_g4_cut_MeV_mm=cutmatched.tolist(),
                  fast_over_g4_unrestricted=(fast / sampled["DEDXnr"]).tolist(),
                  fast_cutmatched_over_g4_ionisation=(cutmatched / sampled["Ionisation"]).tolist(),
                  table_nodes={name: len(v["energy"]) for name, v in tables.items()},
                  range_consistency={})
    dense = np.geomspace(.1, 40., 1001)
    for dedx, range_name in (("DEDX", "Range"), ("DEDXnr", "CSDARange")):
        grid, ranges = integrated_range(tables, dedx)
        native = interpolate(tables[range_name], dense) - float(interpolate(tables[range_name], .1))
        numeric = np.interp(dense, grid, ranges)
        delta = native - numeric
        report["range_consistency"][range_name] = dict(max_abs_mm=float(np.max(np.abs(delta))),
            max_rel_above_0p2=float(np.max(np.abs(delta[dense >= .2] / numeric[dense >= .2]))))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, help="completed export directory with tables/ and job.log")
    ap.add_argument("--compare-export", help="a second export for grid-refinement comparison")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.export)
    log = (root / "job.log").read_text()
    if "exit=0" not in log or "Art has completed and will exit with status 0" not in log:
        raise ValueError("successful table-export job required")
    release = re.search(r"Geant4 version Name:\s+([^\n]+)", log)
    if not release or "geant4-11-03-patch-02" not in release.group(1):
        raise ValueError("this reader has only been checked against Geant4 11.3.2")
    expanded = (root / "expanded.fcl").read_text()
    if 'physicsListName: "ShieldingM"' not in expanded or 'physicsListDecider invoked with ShieldingM' not in log:
        raise ValueError("ShieldingM release diagnostic required")
    if '/Musings/Offline/v10_47_00/' not in log:
        raise ValueError("expected Offline v10_47_00 provenance absent")
    tables, info, paths = load_target(root / "tables")
    report = summarize(tables, info)
    report["release"] = release.group(1).strip()
    artifacts = paths + [root / name for name in ("job.log", "expanded.fcl", "export.fcl", "provenance.txt")]
    if (root / "job.mac").exists():
        artifacts.append(root / "job.mac")
    comparisons = []
    if a.compare_export:
        other_root = Path(a.compare_export)
        other_log = (other_root / "job.log").read_text()
        if "exit=0" not in other_log or release.group(1) not in other_log:
            raise ValueError("comparison export release/outcome mismatch")
        other, other_info, other_paths = load_target(other_root / "tables")
        if info != other_info:
            raise ValueError("comparison material/cuts mismatch")
        energy = np.geomspace(.1, 40., 10001)
        report["grid_comparison"] = {name: dict(
            max_relative=float(np.max(np.abs(interpolate(other[name], energy) / interpolate(tables[name], energy) - 1))),
            max_relative_above_0p3=float(np.max(np.abs(interpolate(other[name], energy[energy >= .3]) /
                                                         interpolate(tables[name], energy[energy >= .3]) - 1))))
            for name in TABLES}
        report["threshold_relative_grid_comparison"] = {}
        for name in ("Range", "CSDARange"):
            primary = interpolate(tables[name], energy) - float(interpolate(tables[name], .1))
            secondary = interpolate(other[name], energy) - float(interpolate(other[name], .1))
            delta = secondary - primary
            report["threshold_relative_grid_comparison"][name] = dict(
                max_abs_mm=float(np.max(np.abs(delta))),
                max_relative_above_0p2=float(np.max(np.abs(delta[energy >= .2] / primary[energy >= .2]))))
        comparisons = [_identity(p) for p in other_paths + [other_root / "job.log", other_root / "expanded.fcl"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "target_tables.npz", **{f"{name}_{key}": v[key] for name, v in tables.items()
                                                    for key in ("energy", "values")})
    (out / "manifest.json").write_text(json.dumps(dict(
        artifacts=[_identity(p) for p in artifacts], comparisons=comparisons, code=_identity(__file__),
        fast_model_code=[_identity(p) for p in (Path(eloss.__file__), Path(eloss.__file__).with_name("materials.py"),
                                               Path(geometry.__file__))],
        interpolation="SciPy not-a-knot reconstruction; not G4 spline implementation parity",
        scope="Release-produced mu- tables; CSDA enabled for export, not a historical binary identity certificate",
        units_source="G4ProductionCutsTable::StoreCutsInfo (range mm, energy keV); physics vectors use MeV/mm/mm"), indent=2))
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
