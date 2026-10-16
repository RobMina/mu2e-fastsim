"""Single CLI entrypoint:  .venv/bin/python3 -m crvfastsim <subcommand> <inputs> [...]

Validation subcommands (docs/plan.md section 5) and, later, the fast-sim drivers. Run from the repository root so the
package imports (Mu2e machines: $CRVFS_CODE; Polaris: the Eagle clone).
Common flags: --outdir DIR (default plots/<cmd>), --compare REF [...], --emit X.json / --merge for batchable ones.
Fast CRV reco: reco-vr2, reco-vr2stock, reco-vr2jax, reco-bench, reco-vr1, reco-direct, reco-cost, reco-inspect.
"""
import argparse
import sys

from .analyses import census, g4cost, transport, v0check, v1compare
from .reco import cli as reco_cli

ANALYSES = {
    "g4cost":    (g4cost.run,    "baseline cost of the stock G4 chain from TimeTracker SQLite (sim + digi), per accepted event"),
    "census":    (census.run,    "what reaches the CRV: visible energy and steps by origin (primary muon / descendants / non-muon)"),
    "transport": (transport.run, "fast muon transport on CRY primaries (gen tables) -> CrvStep tables; --stage A = air only"),
    "v0check":   (v0check.run,   "V0: CRV bar geometry (CosmicRayShieldMaker re-implementation) vs the Geant4 GDML, per bar"),
    "v1compare": (v1compare.run, "V1: primary-muon bar crossings (count, path length, visible energy, time) fast sim vs G4"),
}

# Subcommands that scale via the --emit (worker) / --merge (combiner) batch JSON pattern.
BATCHABLE = {"v1compare", "census"}


def build_parser():
    p = argparse.ArgumentParser(prog="crvfastsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, (fn, help_) in ANALYSES.items():
        sp = sub.add_parser(name, help=help_, description=sys.modules[fn.__module__].__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        sp.add_argument("inputs", nargs="+", help="input files (globs allowed, quote them)")
        sp.add_argument("--outdir", default=None, help="output dir (default plots/<cmd>)")
        sp.add_argument("--counters", default=None,
                        help="Offline CRV counters file (default Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt)")
        if name in BATCHABLE:
            sp.add_argument("--emit", default=None, metavar="JSON", help="write this batch's counts, no plots")
            sp.add_argument("--merge", action="store_true", help="inputs are --emit JSONs: sum and report")
        if name == "transport":
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, i)")
            sp.add_argument("--stage", default="A", choices=["A", "B", "C"],
                            help="world model (A: air, no MSC; B: air + MSC; C: + GDML world material)")
            sp.add_argument("--gdml", default=None, help="GDML world export for --stage C (geometry/README.md)")
        if name == "g4cost":
            sp.add_argument("--digi", required=True, help="the digi job's TimeTracker SQLite (timing_digi.db)")
        if name == "v0check":
            sp.add_argument("--g4log", required=True, help="mu2e log with the crs.verbosityLevel 5 bar printout")
            sp.add_argument("--table", default=None, metavar="CSV", help="also write the candidate bar table")
        if name == "v1compare":
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT",
                            help="G4 reference CrvStep tables (crvsteps.*.root)")
    for name in reco_cli.RECO_CMDS:
        sp = sub.add_parser(f"reco-{name}", help=f"fast CRV reco: {name} (reco/cli.py)")
        reco_cli.add_reco_args(sp, name)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd.startswith("reco-"):
        return reco_cli.run_reco(args.cmd[len("reco-"):], args)
    kw = vars(args)
    cmd = kw.pop("cmd")
    inputs = kw.pop("inputs")
    outdir = kw.pop("outdir") or f"plots/{cmd}"
    ANALYSES[cmd][0](inputs, outdir, **kw)


if __name__ == "__main__":
    main()
