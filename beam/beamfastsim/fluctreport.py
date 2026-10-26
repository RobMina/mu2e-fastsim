"""Hash-checked distribution summaries of saved frozen-entry fluctuation arms; never rerun transport."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import g4ref
from .fluctstudy import ARMS, PAIRS, mc_summary, paired
from .reference_audit import _identity
from .slicestudy import verify_hashes


CUTS = (0., .1, .2, .5, 1., 2., 5., 10., 20., 40.)
QUANTILES = (.01, .1, .5, .9, .99)


def distribution(exit_energy, entry, mask):
    """Unconditional exit/loss including stop atom; between-entry variation is not straggling width."""
    outgoing = np.asarray(exit_energy)[:, mask]
    loss = np.asarray(entry)[None, mask] - outgoing
    if outgoing.shape[1] == 0:
        return dict(entries=0)
    return dict(entries=int(mask.sum()), exit_quantiles=np.quantile(outgoing, QUANTILES).tolist(),
                loss_quantiles=np.quantile(loss, QUANTILES).tolist(),
                mean_exit_MeV=mc_summary(outgoing.mean(axis=1)),
                mean_within_entry_exit_variance_MeV2=float(outgoing.var(axis=0, ddof=1).mean()),
                cdf_le={str(c): mc_summary((outgoing <= c).mean(axis=1)) for c in CUTS})


def contrast(candidate, baseline, mask):
    """Conditional replica errors on fixed CDF thresholds and matched per-entry energy differences."""
    cand, base = candidate[:, mask], baseline[:, mask]
    if cand.shape[1] == 0:
        return dict(entries=0)
    delta = cand - base
    return dict(entries=int(mask.sum()), mean_exit_delta_MeV=mc_summary(delta.mean(axis=1)),
                per_entry_mean_delta_MeV_quantiles=np.quantile(delta.mean(axis=0), [0., *QUANTILES, 1.]).tolist(),
                cdf_le_deltas={str(c): mc_summary(((cand <= c).astype(float) - (base <= c)).mean(axis=1))
                               for c in CUTS})


def report_saved(study):
    study = Path(study)
    manifest = json.loads((study / "manifest.json").read_text())
    summary = json.loads((study / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("complete study required")
    for item in manifest["inputs"] + manifest["outputs"]:
        verify_hashes([item], [_identity(item["path"])], "saved fluctuation data")
    actual = {str(p.resolve()) for p in study.glob("*.npz")}
    recorded = {str(Path(p["path"]).resolve()) for p in manifest["outputs"]}
    expected = {str((study / "entries.npz").resolve())}
    expected.update(str((study / f"{name}_s{s}.npz").resolve()) for name in ARMS for s in summary["slices"])
    if actual != recorded or actual != expected:
        raise ValueError("saved output coverage mismatch")
    tables = [p for p in manifest["inputs"] if Path(p["path"]).name == "couple.dat"]
    if len(tables) != 1:
        raise ValueError("unique table couple required")
    from .physics_tables import load_target
    _, _, consumed = load_target(Path(tables[0]["path"]).parent)
    if not {str(p.resolve()) for p in consumed} <= {str(Path(p["path"]).resolve()) for p in manifest["inputs"]}:
        raise ValueError("manifest omits consumed tables")
    entries = g4ref.load(study / "entries.npz")
    entry = entries["entry_MeV"]
    masks = {"all": np.ones(len(entry), bool), "entry_below3MeV": entry < 3., "entry_ge3MeV": entry >= 3.}
    output = dict(scope="Saved frozen fast-entry distributions, not G4 validation; CDF thresholds predeclared",
                  quantile_probabilities=QUANTILES, cdf_thresholds_MeV=CUTS, runs={}, pairs={}, slices={},
                  completed_arrays=0, source_status="Original run source hashes retained; current reporter has separate hashes")
    old = {}
    for s in summary["slices"]:
        current = {}
        for name in ARMS:
            tag = f"{name}_s{s}"
            result = g4ref.load(study / f"{tag}.npz")
            shape = (summary["replicas"], len(entry))
            if any(v.shape != shape or np.any(~np.isfinite(v)) for v in result.values()):
                raise ValueError("malformed result shape/values")
            if not np.array_equal(result["stop"], result["exit_MeV"] < .1):
                raise ValueError("stop/exit mismatch")
            if np.any(result["exit_MeV"] < 0) or np.any(result["exit_MeV"] > entry[None, :]):
                raise ValueError("exit energy outside physical bounds")
            ledger = sum(result[k] for k in ("exit_MeV", "continuous_applied", "hard_applied", "threshold_residual"))
            if not np.allclose(ledger, entry[None, :], atol=1e-12, rtol=1e-12):
                raise ValueError("saved energy ledger does not close")
            if float(result["stop"].sum(axis=1).mean()) != summary["runs"][tag]["expected_stops"]["mean"]:
                raise ValueError("summary/saved stops mismatch")
            output["completed_arrays"] += 1
            output["runs"][tag] = {group: distribution(result["exit_MeV"], entry, mask) for group, mask in masks.items()}
            current[name] = {k: result[k] for k in ("stop", "exit_MeV")}
        for cand, base in PAIRS:
            output["pairs"][f"{cand}_minus_{base}_s{s}"] = {
                group: contrast(current[cand]["exit_MeV"], current[base]["exit_MeV"], mask)
                for group, mask in masks.items()}
        for name in sorted(old.keys() & current.keys()):
            output["slices"][f"{name}_s{s}_minus_first"] = dict(
                paired=paired(current[name], old[name]),
                cohorts={group: contrast(current[name]["exit_MeV"], old[name]["exit_MeV"], mask)
                         for group, mask in masks.items()})
        if not old:
            old = current
    output["global_caps"] = {key: sum(r["energy_budget"][key]["nonzero_entry_replica_pairs"]
                                     for r in summary["runs"].values())
                             for key in ("continuous_cap_excess", "hard_cap_excess")}
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    result = report_saved(a.study)
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    study = Path(a.study)
    (out / "manifest.json").write_text(json.dumps(dict(
        inputs=[_identity(study / name) for name in ("manifest.json", "summary.json")],
        code=[_identity(Path(__file__)), _identity(Path(__file__).with_name("fluctstudy.py")),
              _identity(Path(__file__).with_name("g4ref.py")), _identity(Path(__file__).with_name("slicestudy.py")),
              _identity(Path(__file__).with_name("reference_audit.py"))]), indent=2))
    print(json.dumps(dict(completed_arrays=result["completed_arrays"], global_caps=result["global_caps"], out=str(out))))


if __name__ == "__main__":
    main()
