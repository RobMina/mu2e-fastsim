"""Raw-G4-range exhaustion sensitivity on monoenergetic slabs; not a production endpoint recommendation.

The fluctuation driver still absorbs sampled residual energy below 0.1 MeV. Only the deterministic bypass threshold
is changed: raw Range(T), not Range(T)-Range(0.1). Below 0.1 the inverse range is a linear endpoint bridge, which is
an explicitly unvalidated diagnostic, not a low-energy physics extrapolation. Nothing in rangefluct is modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import physics_tables as pt, rangefluct as rf
from .foilcompare import contrast
from .g4ref import load
from .reference_audit import _identity
from .slicestudy import verify_hashes


class RawRangeLaw:
    def __init__(self, table, cut, maximum=40.):
        energy = np.geomspace(.1, maximum, 65537)
        self.energy = np.r_[0., energy]
        self.ranges = np.r_[0., pt.interpolate(table, energy)]
        self.cut = cut
        if np.any(np.diff(self.ranges) <= 0):
            raise ValueError("raw range must be increasing")

    def mean(self, energy, length, mode):
        if mode != "range":
            raise ValueError("raw endpoint diagnostic supports range mode only")
        energy, length = np.broadcast_arrays(np.asarray(energy, float), np.asarray(length, float))
        if np.any(energy < .1) or np.any(energy > self.energy[-1]) or np.any(length < 0):
            raise ValueError("raw mean query outside domain")
        residual = np.interp(energy, self.energy, self.ranges) - length
        forced = residual <= 0
        outgoing = np.interp(np.maximum(residual, 0), self.ranges, self.energy)
        return np.where(length == 0, 0., np.where(forced, energy, energy - outgoing)), forced


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--table-export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=50000)
    a = ap.parse_args(argv)
    root, out = Path(a.comparison), Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    original_manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("complete comparison required")
    for item in original_manifest["outputs"]:
        verify_hashes([item], [_identity(item["path"])], "saved comparisons")
    tables, info, paths = pt.load_target(Path(a.table_export) / "tables", material="G4_Al")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(scope=__doc__, comparisons={})
    for index, label in enumerate(("nom_edge", "moved_edge", "oblique_edge")):
        g4 = load(root / f"g4_{label}_loss_fine.npz")
        case = summary["g4"][label + "_loss_fine"]["case"]
        for arm, vector, cut, deltas in (("unrestricted", "CSDARange", 1e9, False),
                                        ("split", "Range", info["energy_cuts_MeV"]["e-"], True)):
            law = RawRangeLaw(tables[vector], cut)
            for slices in (4, 16, 32):
                result = rf.run_layer(np.array([case["energy_MeV"]]), np.array([case["thickness_mm"] / case["cosine"]]),
                    law, mode="range", deltas=deltas, slices=slices, replicas=a.samples, seed=2026102600 + index)
                tag = f"{label}_{arm}_s{slices}"
                np.savez_compressed(out / f"{tag}.npz", **result)
                comparison = contrast(result["exit_MeV"].ravel(), g4["exit_MeV"], case["energy_MeV"])
                comparison["stop_fraction"] = float(result["stop"].mean())
                comparison["forced_fraction"] = float(result["forced_range_stop"].mean())
                report["comparisons"][tag] = comparison
                print(json.dumps(dict(tag=tag, stop=comparison["stop_fraction"], forced=comparison["forced_fraction"],
                    delta=comparison["stop"]["delta"], stop_verdict=comparison["stop"]["state"])), flush=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / "manifest.json").write_text(json.dumps(dict(inputs=[_identity(root / name) for name in ("manifest.json", "summary.json")],
        tables=[_identity(p) for p in paths], sources=[_identity(Path(__file__)), _identity(Path(rf.__file__))],
        outputs=[_identity(p) for p in sorted(out.glob("*.npz"))]), indent=2))


if __name__ == "__main__":
    main()
