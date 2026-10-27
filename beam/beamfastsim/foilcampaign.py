"""Serial bounded single-foil pilot with declared diagnostic tolerances and verified persisted counts."""
import argparse
import json
from pathlib import Path
import time

from .foilclosure import run_case
from .foilreport import audit
from .reference_audit import _identity


CASES = (("nom_edge", 1.4, .1056, 1.), ("moved_edge", 2.02, .2, 1.),
         ("nom_5", 5., .1056, 1.), ("moved_5", 5., .2, 1.),
         ("moved_20", 20., .2, 1.), ("oblique_edge", 2.3, .2, .8))
MODES = {"full": (False, None), "loss": (True, None), "loss_fine": (True, .05)}
TOLERANCES = dict(stop_abs=.02, mean_loss_relative=.01, mean_loss_floor_MeV=.002,
                  variance_relative=.05, cdf_abs=.02,
                  scope="Provisional layer bands, not full-chain acceptance; pointwise 3-SE moments, exact-binomial 99% stop and 99% DKW CDF brackets")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", type=int, default=5000)
    ap.add_argument("--budget-seconds", type=float, default=900.)
    ap.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES], default=[c[0] for c in CASES])
    ap.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    ap.add_argument("--refine", action="store_true", help="loss 0.01/0.0001mm and 0.005/0.00005mm; no default-reference rerun")
    ap.add_argument("--bins", type=int)
    a = ap.parse_args(argv)
    if a.events <= 0 or not 0 < a.budget_seconds < float("inf"):
        raise ValueError("positive finite controls required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    modes = {k: (*v, .001) for k, v in MODES.items() if k in a.modes}
    if a.refine:
        modes = dict(loss_gridfine=(True, .05, .001), loss_finer=(True, .01, .0001), loss_finest=(True, .005, .00005))
    manifest = dict(complete=False, cases=CASES, modes=modes, events=a.events, tolerances=TOLERANCES,
                    sources=[_identity(Path(__file__)), _identity(Path(__file__).with_name("foilclosure.py")),
                             _identity(Path(__file__).with_name("foilreport.py"))], runs={})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for index, (label, energy, thickness, cosine) in enumerate(CASES):
        if label not in a.cases:
            continue
        for j, mode in enumerate(modes):
            tag = f"{label}_{mode}"
            if time.monotonic() - started > a.budget_seconds:
                manifest["unstarted_from"] = tag
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
                raise SystemExit(2)
            no_msc, step_fraction, final_range = modes[mode]
            run_case(out / tag, energy=energy, thickness=thickness, cosine=cosine, pdg=13,
                     events=a.events, seed=202610260 + 10 * index + j + (1000 if a.refine else 0),
                     no_msc=no_msc, step_fraction=step_fraction, final_range=final_range, bins=a.bins)
            report, _, _ = audit(out / tag)
            manifest["runs"][tag] = dict(receipt=_identity(out / tag / "receipt.json"), census=report)
            manifest["wall_s"] = time.monotonic() - started
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
            print(json.dumps(dict(tag=tag, census=report)), flush=True)
    manifest["complete"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(dict(complete=True, runs=len(manifest["runs"]), wall_s=manifest["wall_s"])), flush=True)


if __name__ == "__main__":
    main()
