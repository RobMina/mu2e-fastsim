"""Final identity-checked foil report: reuse all fast arrays and G4 jobs, add endpoint/refinement comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

from mu2efastsim import eloss, eloss_jax, materials
from . import foilcompare as fc, foilreport as fr, geometry, layerstudy, rangefluct
from .g4ref import load
from .reference_audit import _identity
from .slicestudy import verify_hashes


def verify_study(root, outputs_key="outputs"):
    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest.get("inputs", []) + manifest.get("tables", []) + manifest[outputs_key]:
        verify_hashes([item], [_identity(item["path"])], "saved study")
    files = {str(p.resolve()) for p in root.glob("*.npz")}
    if files != {str(Path(p["path"]).resolve()) for p in manifest[outputs_key]}:
        raise ValueError("saved output coverage mismatch")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--refinement", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--table-export", required=True)
    ap.add_argument("--table-reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    comp, refine, endpoint, out = map(Path, (a.comparison, a.refinement, a.endpoint, a.out))
    if out.exists():
        raise FileExistsError(out)
    source_paths = [Path(__file__), Path(fc.__file__), Path(fr.__file__), Path(eloss.__file__), Path(eloss_jax.__file__),
                    Path(materials.__file__), Path(geometry.__file__), Path(layerstudy.__file__), Path(rangefluct.__file__),
                    Path(__file__).with_name("foilclosure.py"), Path(__file__).with_name("foilcampaign.py"),
                    Path(__file__).with_name("fluctstudy.py"), Path(__file__).with_name("physics_tables.py"),
                    Path(__file__).with_name("g4ref.py"), Path(__file__).with_name("slicestudy.py"),
                    Path(__file__).with_name("reference_audit.py")]
    sources = [_identity(p) for p in source_paths]
    cm, em = verify_study(comp), verify_study(endpoint)
    cs = json.loads((comp / "summary.json").read_text())
    rm = json.loads((refine / "manifest.json").read_text())
    if not cs["complete"] or not rm["complete"]:
        raise ValueError("completed inputs required")
    material = fr.table_audit(a.table_export, a.table_reference)
    if not material["mu-"]["compatible"]:
        raise ValueError("material check failed")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, material=material, tolerances=fc.TOLERANCES, g4={}, g4_contrasts={},
                  fast_contrasts={}, endpoint_contrasts={}, receipts=[], source_scope="Final reporting code and unchanged production dependencies; initial candidate source hashes retained in original manifests")
    cases, all_g4 = {}, {}
    for tag in cs["g4"]:
        data = load(comp / f"g4_{tag}.npz")
        receipt = next(p for p in cm["inputs"] if Path(p["path"]).parent.name == tag)
        census, fresh, _ = fr.audit(Path(receipt["path"]).parent)
        if any(not np.array_equal(data[k], fresh[k], equal_nan=True) for k in data):
            raise ValueError("saved and re-audited primary census differs")
        report["g4"][tag] = census
        report["receipts"].append(receipt)
        cases[tag] = census["case"]
        all_g4[tag] = data
    for tag, item in rm["runs"].items():
        verify_hashes([item["receipt"]], [_identity(item["receipt"]["path"])], "refinement receipt")
        census, data, _ = fr.audit(Path(item["receipt"]["path"]).parent)
        if np.any(data["other"]):
            raise ValueError("unresolved G4 outcomes")
        report["g4"][tag] = census
        report["receipts"].append(item["receipt"])
        cases[tag] = census["case"]
        all_g4[tag] = data
        np.savez_compressed(out / f"g4_{tag}.npz", **data)
    labels = ["nom_edge", "moved_edge", "nom_5", "moved_5", "moved_20", "oblique_edge"]
    for label in labels:
        modes = {tag[len(label) + 1:]: data for tag, data in all_g4.items() if tag.startswith(label + "_")}
        energy = cases[label + "_full"]["energy_MeV"]
        for cand, base in (("full", "loss"), ("loss_fine", "loss"), ("loss_gridfine", "loss_fine"),
                           ("loss_finer", "loss_gridfine"), ("loss_finest", "loss_finer")):
            if cand in modes and base in modes:
                report["g4_contrasts"][f"{label}_{cand}_minus_{base}"] = fc.contrast(modes[cand]["exit_MeV"], modes[base]["exit_MeV"], energy)
        for arm in fc.ARMS:
            for slices in (4, 16, 32):
                tag = f"{label}_{arm}_s{slices}"
                fast = load(comp / f"fast_{tag}.npz")
                ledger = sum(fast[k] for k in ("exit_MeV", "continuous_applied", "hard_applied", "threshold_residual"))
                if not np.allclose(ledger, energy, rtol=1e-12, atol=1e-12):
                    raise ValueError("fast energy ledger mismatch")
                for mode, data in modes.items():
                    result = fc.contrast(fast["exit_MeV"].ravel(), data["exit_MeV"], energy)
                    result["hard_mean_difference_MeV"] = float(fast["hard_applied"].mean() - data["hard_MeV"].mean())
                    report["fast_contrasts"][f"{tag}_vs_{mode}"] = result
        if "loss_finest" in modes:
            for arm in ("unrestricted", "split"):
                for slices in (4, 16, 32):
                    tag = f"{label}_{arm}_s{slices}"
                    fast = load(endpoint / f"{tag}.npz")
                    report["endpoint_contrasts"][tag] = fc.contrast(fast["exit_MeV"].ravel(), modes["loss_finest"]["exit_MeV"], energy)
        refmode = "loss_finest" if "loss_finest" in modes else "loss_fine"
        r = report["fast_contrasts"][f"{label}_g4_range_cut_split_s32_vs_{refmode}"]
        print(json.dumps(dict(case=label, reference=refmode, g4_threshold=float(np.mean(modes[refmode]["exit_MeV"] < .1)),
            split_stop_delta=r["stop"], mean_exit_delta=r["mean_exit"], variance=r["variance"], cdf=r["cdf"])), flush=True)
    report["complete"] = True
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for item in sources:
        verify_hashes([item], [_identity(item["path"])], "source during final report")
    inputs = [_identity(p / name) for p in (comp, endpoint) for name in ("manifest.json", "summary.json")]
    inputs += [_identity(refine / "manifest.json")]
    (out / "manifest.json").write_text(json.dumps(dict(inputs=inputs, sources=sources,
        report=_identity(out / "summary.json"), outputs=[_identity(p) for p in out.glob("*.npz")],
        caveat="Core/geometry/layerstudy hashes match prior frozen study separately; original run wrappers are not retroactively replaced"), indent=2))


if __name__ == "__main__":
    main()
